use serde::{Deserialize, Serialize};

#[cfg(feature = "sqlite")]
use crate::error::{KartaError, Result};
#[cfg(feature = "sqlite")]
use chrono::Utc;
#[cfg(feature = "sqlite")]
use rusqlite::Connection;

/// Current schema version. Increment this when adding new migrations.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Tracks the current schema state of a Karta data directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMeta {
    pub schema_version: u32,
    pub applied_migrations: Vec<String>,
    pub pending_migrations: Vec<String>,
    pub warnings: Vec<String>,
}

impl SchemaMeta {
    pub fn new(
        version: u32,
        applied: Vec<String>,
        pending: Vec<String>,
        warnings: Vec<String>,
    ) -> Self {
        Self {
            schema_version: version,
            applied_migrations: applied,
            pending_migrations: pending,
            warnings,
        }
    }
}

/// A single migration step.
///
/// Migrations are applied forward with `up_sql`. `down_sql` is recorded so future
/// tooling has an explicit rollback hook, but runtime rollback of a committed
/// migration is not currently automated.
#[derive(Debug, Clone)]
pub struct Migration {
    pub id: &'static str,
    pub description: &'static str,
    pub up_sql: &'static str,
    pub down_sql: Option<&'static str>,
}

/// Returns all migrations in order. Add new migrations here.
pub fn all_migrations() -> Vec<Migration> {
    vec![]
}

/// Load the current schema meta from the database.
#[cfg(feature = "sqlite")]
pub fn load_schema_meta(conn: &Connection) -> Result<SchemaMeta> {
    let migrations = all_migrations();
    load_schema_meta_with_migrations(conn, &migrations)
}

#[cfg(feature = "sqlite")]
fn load_schema_meta_with_migrations(
    conn: &Connection,
    migrations: &[Migration],
) -> Result<SchemaMeta> {
    let result = conn.query_row(
        "SELECT schema_version, applied_migrations_json FROM schema_meta WHERE id = 1",
        [],
        |row| {
            let version: u32 = row.get(0)?;
            let applied_json: String = row.get(1)?;
            Ok((version, applied_json))
        },
    );

    match result {
        Ok((version, applied_json)) => {
            let applied: Vec<String> = serde_json::from_str(&applied_json).map_err(|e| {
                KartaError::Serialization(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid applied_migrations_json in schema_meta: {e}"),
                )))
            })?;
            let all_ids: Vec<&str> = migrations.iter().map(|m| m.id).collect();
            let pending: Vec<String> = all_ids
                .iter()
                .filter(|id| !applied.iter().any(|a| a == *id))
                .map(|s| s.to_string())
                .collect();
            Ok(SchemaMeta::new(version, applied, pending, vec![]))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            let pending = migrations.iter().map(|m| m.id.to_string()).collect();
            Ok(SchemaMeta::new(0, vec![], pending, vec![]))
        }
        Err(e) => Err(KartaError::GraphStore(e.to_string())),
    }
}

/// Apply pending migrations. Returns the updated SchemaMeta.
#[cfg(feature = "sqlite")]
pub fn apply_migrations(conn: &Connection) -> Result<SchemaMeta> {
    let migrations = all_migrations();
    apply_migrations_with(conn, &migrations)
}

#[cfg(feature = "sqlite")]
fn persist_schema_meta(conn: &Connection, schema_version: u32, applied: &[String]) -> Result<()> {
    let applied_json = serde_json::to_string(applied).map_err(KartaError::Serialization)?;
    let now = Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO schema_meta (id, schema_version, applied_migrations_json, last_migration_at)
         VALUES (1, ?1, ?2, ?3)
         ON CONFLICT(id) DO UPDATE SET
             schema_version = ?1,
             applied_migrations_json = ?2,
             last_migration_at = ?3",
        rusqlite::params![schema_version, applied_json, now],
    )
    .map(|_| ())
    .map_err(|e| KartaError::GraphStore(e.to_string()))
}

#[cfg(feature = "sqlite")]
fn apply_migrations_with(conn: &Connection, migrations: &[Migration]) -> Result<SchemaMeta> {
    let meta = load_schema_meta_with_migrations(conn, migrations)?;

    if meta.pending_migrations.is_empty() {
        if meta.schema_version < CURRENT_SCHEMA_VERSION {
            persist_schema_meta(conn, CURRENT_SCHEMA_VERSION, &meta.applied_migrations)?;
            return load_schema_meta_with_migrations(conn, migrations);
        }
        return Ok(meta);
    }

    let mut applied = meta.applied_migrations.clone();

    for pending_id in &meta.pending_migrations {
        let migration = migrations.iter().find(|m| m.id == pending_id.as_str());
        let migration = match migration {
            Some(m) => m,
            None => {
                return Err(KartaError::Config(format!(
                    "Migration {} not found in migration list",
                    pending_id
                )));
            }
        };

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        if let Err(e) = tx.execute_batch(migration.up_sql) {
            let _ = tx.rollback();
            return Err(KartaError::GraphStore(format!(
                "Migration {} failed: {}",
                migration.id, e
            )));
        }

        applied.push(migration.id.to_string());
        let applied_json = serde_json::to_string(&applied).map_err(KartaError::Serialization)?;
        let now = Utc::now().to_rfc3339();

        if let Err(e) = tx.execute(
            "INSERT INTO schema_meta (id, schema_version, applied_migrations_json, last_migration_at)
             VALUES (1, ?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET
                 schema_version = ?1,
                 applied_migrations_json = ?2,
                 last_migration_at = ?3",
            rusqlite::params![CURRENT_SCHEMA_VERSION, applied_json, now],
        ) {
            let _ = tx.rollback();
            return Err(KartaError::GraphStore(format!(
                "Failed to update schema_meta after migration {}: {}",
                migration.id, e
            )));
        }

        if let Err(e) = tx.commit() {
            return Err(KartaError::GraphStore(format!(
                "Failed to commit migration {}: {}",
                migration.id, e
            )));
        }
    }

    load_schema_meta_with_migrations(conn, migrations)
}

/// Initialize the schema_meta table if it doesn't exist, then apply pending migrations.
#[cfg(feature = "sqlite")]
pub fn init_and_migrate(conn: &Connection) -> Result<SchemaMeta> {
    // Create schema_meta table first (bootstrap)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            schema_version INTEGER NOT NULL DEFAULT 0,
            applied_migrations_json TEXT NOT NULL DEFAULT '[]',
            last_migration_at TEXT
        )",
        [],
    )
    .map_err(|e| KartaError::GraphStore(e.to_string()))?;

    // Ensure the singleton metadata row exists even when there are no pending
    // migrations. Without this bootstrap row, a freshly initialized database
    // reports schema_version=0 indefinitely.
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta (
            id,
            schema_version,
            applied_migrations_json,
            last_migration_at
        ) VALUES (1, ?1, '[]', ?2)",
        rusqlite::params![CURRENT_SCHEMA_VERSION, Utc::now().to_rfc3339()],
    )
    .map_err(|e| KartaError::GraphStore(e.to_string()))?;

    apply_migrations(conn)
}

#[cfg(all(test, feature = "sqlite"))]
mod tests {
    use super::*;

    fn setup_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE schema_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                schema_version INTEGER NOT NULL DEFAULT 0,
                applied_migrations_json TEXT NOT NULL DEFAULT '[]',
                last_migration_at TEXT
            )",
            [],
        )
        .unwrap();
        conn
    }

    #[test]
    fn apply_migrations_persists_all_ids_and_is_idempotent() {
        let conn = setup_conn();
        let migrations = vec![
            Migration {
                id: "001_test",
                description: "first synthetic migration",
                up_sql: "CREATE TABLE synthetic_one (id INTEGER PRIMARY KEY);",
                down_sql: Some("DROP TABLE synthetic_one;"),
            },
            Migration {
                id: "002_test",
                description: "second synthetic migration",
                up_sql: "CREATE TABLE synthetic_two (id INTEGER PRIMARY KEY);",
                down_sql: Some("DROP TABLE synthetic_two;"),
            },
        ];

        // Seed schema_meta empty; apply_migrations_with computes pending IDs from the supplied synthetic list.
        conn.execute(
            "INSERT INTO schema_meta (id, schema_version, applied_migrations_json) VALUES (1, 0, '[]')",
            [],
        )
        .unwrap();

        let meta = apply_migrations_with(&conn, &migrations).unwrap();
        assert_eq!(meta.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(
            meta.applied_migrations,
            vec!["001_test".to_string(), "002_test".to_string()]
        );
        assert!(meta.pending_migrations.is_empty());

        let rerun = apply_migrations_with(&conn, &migrations).unwrap();
        assert_eq!(rerun.applied_migrations, meta.applied_migrations);
        assert!(rerun.pending_migrations.is_empty());
    }

    #[test]
    fn init_bootstraps_empty_migration_list_to_current_schema_version() {
        let conn = Connection::open_in_memory().unwrap();

        let meta = init_and_migrate(&conn).unwrap();

        assert_eq!(meta.schema_version, CURRENT_SCHEMA_VERSION);
        assert!(meta.applied_migrations.is_empty());
        assert!(meta.pending_migrations.is_empty());
    }

    #[test]
    fn malformed_applied_migrations_json_returns_error() {
        let conn = setup_conn();
        conn.execute(
            "INSERT INTO schema_meta (id, schema_version, applied_migrations_json) VALUES (1, 1, 'not-json')",
            [],
        )
        .unwrap();

        let err = load_schema_meta(&conn).expect_err("malformed JSON should fail");
        match err {
            KartaError::Serialization(_) => {}
            other => panic!("expected serialization error, got {other:?}"),
        }
    }
}
