use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::Connection;
use std::sync::Mutex;

use crate::dream::DreamRun;
use crate::error::{KartaError, Result};
use crate::migrate;
use crate::note::EvolutionRecord;

pub struct SqliteGraphStore {
    conn: Mutex<Connection>,
}

impl SqliteGraphStore {
    pub fn new(data_dir: &str) -> Result<Self> {
        let path = format!("{}/karta.db", data_dir);
        std::fs::create_dir_all(data_dir).map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let conn = Connection::open(&path).map_err(|e| KartaError::GraphStore(e.to_string()))?;

        // Use DELETE journal mode for compatibility with network filesystems (e.g. GCS FUSE).
        // WAL requires shared memory / file locking that FUSE mounts don't support.
        conn.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA foreign_keys=ON;")
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let store = Self {
            conn: Mutex::new(conn),
        };
        // We'll call init() from Karta::new()
        Ok(store)
    }
}

impl SqliteGraphStore {
    /// Idempotent migration for the `links` table: adds `weight` + `link_type`
    /// columns on pre-ACTIVATE databases and rebuilds the PK to
    /// `(from_id, to_id, link_type)`. Safe to call on a fresh database.
    fn migrate_links_table(conn: &Connection) -> Result<()> {
        let table_exists: bool = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='links'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !table_exists {
            return Ok(());
        }

        let mut has_link_type = false;
        let mut has_weight = false;
        {
            let mut stmt = conn
                .prepare("PRAGMA table_info(links)")
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            let cols = stmt
                .query_map([], |row| row.get::<_, String>(1))
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            for col in cols {
                let name = col.map_err(|e| KartaError::GraphStore(e.to_string()))?;
                if name == "link_type" {
                    has_link_type = true;
                }
                if name == "weight" {
                    has_weight = true;
                }
            }
        }

        if has_link_type && has_weight {
            return Ok(());
        }

        // Full rebuild: the original PK is (from_id, to_id); we need it to be
        // (from_id, to_id, link_type) so semantic + follows can coexist.
        // Use an explicit transaction that auto-rolls-back on drop if any
        // statement fails partway through — safer than inline BEGIN/COMMIT
        // inside execute_batch, which leaves the txn open on mid-batch error.
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS links_new (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                link_type TEXT NOT NULL DEFAULT 'semantic',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (from_id, to_id, link_type)
            );
            INSERT OR IGNORE INTO links_new (from_id, to_id, reason, weight, link_type, created_at)
              SELECT from_id, to_id, reason, 1.0, 'semantic', created_at FROM links;
            DROP TABLE links;
            ALTER TABLE links_new RENAME TO links;
            CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_id);
            CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_id);
            CREATE INDEX IF NOT EXISTS idx_links_type ON links(link_type);
            ",
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        tx.commit()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl crate::store::GraphStore for SqliteGraphStore {
    async fn init(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        // Pre-ACTIVATE databases still have the old (from_id, to_id) PK and no
        // weight/link_type columns. Run the migration FIRST — otherwise the
        // CREATE INDEX on link_type below fails with "no such column" before
        // we ever get a chance to add the column. (Production bug 2026-04-22.)
        Self::migrate_links_table(&conn)?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS links (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                link_type TEXT NOT NULL DEFAULT 'semantic',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (from_id, to_id, link_type)
            );

            CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_id);
            CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_id);
            CREATE INDEX IF NOT EXISTS idx_links_type ON links(link_type);

            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id TEXT NOT NULL,
                triggered_by TEXT NOT NULL,
                previous_context TEXT NOT NULL,
                evolved_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_evolution_note ON evolution_history(note_id);

            CREATE TABLE IF NOT EXISTS dream_runs (
                id TEXT PRIMARY KEY,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                notes_inspected INTEGER NOT NULL DEFAULT 0,
                dreams_attempted INTEGER NOT NULL DEFAULT 0,
                dreams_written INTEGER NOT NULL DEFAULT 0,
                dreams_json TEXT,
                total_tokens_used INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS dream_cursor (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_processed_at TEXT NOT NULL
            );

            -- Foresight signals (Phase 2B.2)
            CREATE TABLE IF NOT EXISTS foresight_signals (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_until TEXT,
                source_note_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'Active',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_foresight_status ON foresight_signals(status);
            CREATE INDEX IF NOT EXISTS idx_foresight_source ON foresight_signals(source_note_id);

            -- Profiles (Phase 2B.3)
            CREATE TABLE IF NOT EXISTS profiles (
                entity_id TEXT PRIMARY KEY,
                note_id TEXT NOT NULL,
                last_updated TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Episodes (Phase 2B.1)
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                narrative_note_id TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                session_id TEXT NOT NULL,
                topic_tags_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);

            CREATE TABLE IF NOT EXISTS note_episodes (
                note_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                PRIMARY KEY (note_id, episode_id)
            );
            CREATE INDEX IF NOT EXISTS idx_note_episodes_episode ON note_episodes(episode_id);

            -- Atomic facts metadata (Phase Next)
            CREATE TABLE IF NOT EXISTS atomic_facts (
                id TEXT PRIMARY KEY,
                source_note_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL DEFAULT 0,
                subject TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_facts_source ON atomic_facts(source_note_id);
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON atomic_facts(subject);

            -- Episode digests (Phase Next)
            CREATE TABLE IF NOT EXISTS episode_digests (
                id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL UNIQUE,
                entities_json TEXT NOT NULL DEFAULT '[]',
                date_range_json TEXT,
                aggregations_json TEXT NOT NULL DEFAULT '[]',
                topic_sequence_json TEXT NOT NULL DEFAULT '[]',
                events_json TEXT NOT NULL DEFAULT '[]',
                digest_text TEXT NOT NULL DEFAULT '',
                digest_note_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_digests_episode ON episode_digests(episode_id);

            -- Cross-episode digests (structural cross-level metadata)
            CREATE TABLE IF NOT EXISTS cross_episode_digests (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                entity_timeline_json TEXT NOT NULL DEFAULT '[]',
                cross_aggregations_json TEXT NOT NULL DEFAULT '[]',
                events_json TEXT NOT NULL DEFAULT '[]',
                topic_progression_json TEXT NOT NULL DEFAULT '[]',
                digest_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_cross_digests_scope ON cross_episode_digests(scope_id);

            -- Episode links (Phase Next)
            CREATE TABLE IF NOT EXISTS episode_links (
                from_episode_id TEXT NOT NULL,
                to_episode_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                entity TEXT,
                reason TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (from_episode_id, to_episode_id, link_type, entity)
            );
            CREATE INDEX IF NOT EXISTS idx_ep_links_from ON episode_links(from_episode_id);
            CREATE INDEX IF NOT EXISTS idx_ep_links_to ON episode_links(to_episode_id);
            CREATE INDEX IF NOT EXISTS idx_ep_links_entity ON episode_links(entity);

            -- Procedural rules (Issue #6)
            CREATE TABLE IF NOT EXISTS procedural_rules (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                condition_json TEXT NOT NULL,
                actions_json TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                protected INTEGER NOT NULL DEFAULT 0,
                source_note_id TEXT,
                fire_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_rules_enabled ON procedural_rules(enabled);
            ",
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        // Initialize schema_meta table and apply pending migrations
        migrate::init_and_migrate(&conn)?;

        Ok(())
    }

    async fn add_link(&self, from_id: &str, to_id: &str, reason: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let now = Utc::now().to_rfc3339();

        // Bidirectional semantic link; weight starts at 1.0 and is bumped by Hebbian.
        conn.execute(
            "INSERT OR IGNORE INTO links (from_id, to_id, reason, weight, link_type, created_at) VALUES (?1, ?2, ?3, 1.0, 'semantic', ?4)",
            rusqlite::params![from_id, to_id, reason, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        conn.execute(
            "INSERT OR IGNORE INTO links (from_id, to_id, reason, weight, link_type, created_at) VALUES (?1, ?2, ?3, 1.0, 'semantic', ?4)",
            rusqlite::params![to_id, from_id, reason, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(())
    }

    async fn add_link_typed(
        &self,
        from_id: &str,
        to_id: &str,
        link_type: &str,
        reason: &str,
        weight: f32,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let now = Utc::now().to_rfc3339();

        // "follows" is single-direction (prev -> next); reverse is derived via turn_delta.
        // "semantic" is bidirectional.
        conn.execute(
            "INSERT OR IGNORE INTO links (from_id, to_id, reason, weight, link_type, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![from_id, to_id, reason, weight, link_type, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        if link_type == "semantic" {
            conn.execute(
                "INSERT OR IGNORE INTO links (from_id, to_id, reason, weight, link_type, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![to_id, from_id, reason, weight, link_type, now],
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        }

        Ok(())
    }

    async fn get_links_with_weights(
        &self,
        note_id: &str,
        link_type: Option<&str>,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let rows: Vec<(String, f32)> = if let Some(lt) = link_type {
            let mut stmt = conn
                .prepare("SELECT to_id, weight FROM links WHERE from_id = ?1 AND link_type = ?2 ORDER BY weight DESC")
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            stmt.query_map(rusqlite::params![note_id, lt], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
        } else {
            let mut stmt = conn
                .prepare("SELECT to_id, weight FROM links WHERE from_id = ?1 ORDER BY weight DESC")
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            stmt.query_map(rusqlite::params![note_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
        };
        Ok(rows)
    }

    async fn get_sequential_neighbors(
        &self,
        note_id: &str,
        radius: usize,
    ) -> Result<Vec<(String, i32)>> {
        if radius == 0 {
            return Ok(Vec::new());
        }
        let mut neighbors: Vec<(String, i32)> = Vec::new();
        let mut current = note_id.to_string();
        // Walk forward via "follows" links (prev -> next).
        {
            let conn = self
                .conn
                .lock()
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            for step in 1..=radius as i32 {
                let mut stmt = conn
                    .prepare("SELECT to_id FROM links WHERE from_id = ?1 AND link_type = 'follows' LIMIT 1")
                    .map_err(|e| KartaError::GraphStore(e.to_string()))?;
                let next: Option<String> = stmt
                    .query_row(rusqlite::params![current], |row| row.get(0))
                    .ok();
                match next {
                    Some(id) => {
                        neighbors.push((id.clone(), step));
                        current = id;
                    }
                    None => break,
                }
            }
        }
        // Walk backward: find predecessor (where to_id = current and link_type = 'follows').
        current = note_id.to_string();
        {
            let conn = self
                .conn
                .lock()
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            for step in 1..=radius as i32 {
                let mut stmt = conn
                    .prepare("SELECT from_id FROM links WHERE to_id = ?1 AND link_type = 'follows' LIMIT 1")
                    .map_err(|e| KartaError::GraphStore(e.to_string()))?;
                let prev: Option<String> = stmt
                    .query_row(rusqlite::params![current], |row| row.get(0))
                    .ok();
                match prev {
                    Some(id) => {
                        neighbors.push((id.clone(), -step));
                        current = id;
                    }
                    None => break,
                }
            }
        }
        Ok(neighbors)
    }

    async fn bump_link_weight(
        &self,
        from_id: &str,
        to_id: &str,
        delta: f32,
        max: f32,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        // Hebbian strengthening on the semantic edge only; clamp to max.
        conn.execute(
            "UPDATE links
             SET weight = MIN(CAST(?3 AS REAL), weight + CAST(?4 AS REAL))
             WHERE from_id = ?1 AND to_id = ?2 AND link_type = 'semantic'",
            rusqlite::params![from_id, to_id, max as f64, delta as f64],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        // Mirror in the reverse direction to preserve bidirectional symmetry.
        conn.execute(
            "UPDATE links
             SET weight = MIN(CAST(?3 AS REAL), weight + CAST(?4 AS REAL))
             WHERE from_id = ?2 AND to_id = ?1 AND link_type = 'semantic'",
            rusqlite::params![from_id, to_id, max as f64, delta as f64],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn bump_link_weights_batch(
        &self,
        pairs: &[(&str, &str)],
        delta: f32,
        max: f32,
    ) -> Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        {
            let mut stmt = tx
                .prepare(
                    "UPDATE links
                     SET weight = MIN(CAST(?3 AS REAL), weight + CAST(?4 AS REAL))
                     WHERE from_id = ?1 AND to_id = ?2 AND link_type = 'semantic'",
                )
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            let max_f = max as f64;
            let delta_f = delta as f64;
            for (a, b) in pairs {
                // Forward direction
                stmt.execute(rusqlite::params![a, b, max_f, delta_f])
                    .map_err(|e| KartaError::GraphStore(e.to_string()))?;
                // Reverse direction to preserve bidirectional symmetry
                stmt.execute(rusqlite::params![b, a, max_f, delta_f])
                    .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            }
        }
        tx.commit()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn decay_link_weights(&self, factor: f32) -> Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        // Multiplicative decay floored at 1.0 so nothing ever decays below the
        // initial semantic-link weight.
        let n = conn
            .execute(
                "UPDATE links SET weight = MAX(1.0, weight * CAST(?1 AS REAL)) WHERE link_type = 'semantic'",
                rusqlite::params![factor as f64],
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(n)
    }

    async fn get_links(&self, note_id: &str) -> Result<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT to_id FROM links WHERE from_id = ?1")
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let links = stmt
            .query_map(rusqlite::params![note_id], |row| row.get(0))
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<Vec<String>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(links)
    }

    async fn get_links_with_reasons(&self, note_id: &str) -> Result<Vec<(String, String)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT to_id, reason FROM links WHERE from_id = ?1")
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let links = stmt
            .query_map(rusqlite::params![note_id], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<Vec<(String, String)>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(links)
    }

    async fn record_evolution(
        &self,
        note_id: &str,
        triggered_by: &str,
        previous_context: &str,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO evolution_history (note_id, triggered_by, previous_context, evolved_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![note_id, triggered_by, previous_context, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(())
    }

    async fn get_evolution_history(&self, note_id: &str) -> Result<Vec<EvolutionRecord>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT triggered_by, previous_context, evolved_at FROM evolution_history WHERE note_id = ?1 ORDER BY id ASC",
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let records = stmt
            .query_map(rusqlite::params![note_id], |row| {
                let evolved_at_str: String = row.get(2)?;
                let evolved_at = DateTime::parse_from_rfc3339(&evolved_at_str)
                    .unwrap_or_default()
                    .with_timezone(&Utc);
                Ok(EvolutionRecord {
                    triggered_by: row.get(0)?,
                    previous_context: row.get(1)?,
                    evolved_at,
                })
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(records)
    }

    async fn record_dream_run(&self, run: &DreamRun) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let dreams_json = serde_json::to_string(&run.dreams)?;

        conn.execute(
            "INSERT INTO dream_runs (id, scope_type, scope_id, started_at, completed_at, notes_inspected, dreams_attempted, dreams_written, dreams_json, total_tokens_used) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                run.id,
                run.scope_type,
                run.scope_id,
                run.started_at.to_rfc3339(),
                run.completed_at.map(|t| t.to_rfc3339()),
                run.notes_inspected,
                run.dreams_attempted,
                run.dreams_written,
                dreams_json,
                run.total_tokens_used,
            ],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(())
    }

    async fn get_dream_cursor(&self) -> Result<Option<DateTime<Utc>>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let result = conn.query_row(
            "SELECT last_processed_at FROM dream_cursor WHERE id = 1",
            [],
            |row| {
                let s: String = row.get(0)?;
                Ok(s)
            },
        );

        match result {
            Ok(s) => {
                let dt = DateTime::parse_from_rfc3339(&s)
                    .unwrap_or_default()
                    .with_timezone(&Utc);
                Ok(Some(dt))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(KartaError::GraphStore(e.to_string())),
        }
    }

    async fn set_dream_cursor(&self, cursor: DateTime<Utc>) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        conn.execute(
            "INSERT INTO dream_cursor (id, last_processed_at) VALUES (1, ?1)
             ON CONFLICT(id) DO UPDATE SET last_processed_at = ?1",
            rusqlite::params![cursor.to_rfc3339()],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(())
    }

    // --- Foresight signals ---

    async fn upsert_foresight(&self, signal: &crate::note::ForesightSignal) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let status_str = serde_json::to_string(&signal.status)?;
        conn.execute(
            "INSERT INTO foresight_signals (id, content, valid_from, valid_until, source_note_id, confidence, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(id) DO UPDATE SET content=?2, valid_until=?4, confidence=?6, status=?7",
            rusqlite::params![
                signal.id,
                signal.content,
                signal.valid_from.to_rfc3339(),
                signal.valid_until.map(|t| t.to_rfc3339()),
                signal.source_note_id,
                signal.confidence,
                status_str.trim_matches('"'),
            ],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_active_foresights(&self) -> Result<Vec<crate::note::ForesightSignal>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id, content, valid_from, valid_until, source_note_id, confidence FROM foresight_signals WHERE status = 'Active'")
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let signals = stmt
            .query_map([], |row| {
                let valid_until_str: Option<String> = row.get(3)?;
                Ok(crate::note::ForesightSignal {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    valid_from: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                        .unwrap_or_default()
                        .with_timezone(&Utc),
                    valid_until: valid_until_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .ok()
                            .map(|dt| dt.with_timezone(&Utc))
                    }),
                    source_note_id: row.get(4)?,
                    confidence: row.get(5)?,
                    status: crate::note::ForesightStatus::Active,
                })
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(signals)
    }

    async fn expire_foresights(&self, before: DateTime<Utc>) -> Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let count = conn
            .execute(
                "UPDATE foresight_signals SET status = 'Expired' WHERE status = 'Active' AND valid_until IS NOT NULL AND valid_until < ?1",
                rusqlite::params![before.to_rfc3339()],
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(count)
    }

    async fn get_foresights_for_note(
        &self,
        note_id: &str,
    ) -> Result<Vec<crate::note::ForesightSignal>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id, content, valid_from, valid_until, source_note_id, confidence, status FROM foresight_signals WHERE source_note_id = ?1")
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let signals = stmt
            .query_map(rusqlite::params![note_id], |row| {
                let valid_until_str: Option<String> = row.get(3)?;
                let status_str: String = row.get(6)?;
                let status = match status_str.as_str() {
                    "Expired" => crate::note::ForesightStatus::Expired,
                    "Fulfilled" => crate::note::ForesightStatus::Fulfilled,
                    _ => crate::note::ForesightStatus::Active,
                };
                Ok(crate::note::ForesightSignal {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    valid_from: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                        .unwrap_or_default()
                        .with_timezone(&Utc),
                    valid_until: valid_until_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .ok()
                            .map(|dt| dt.with_timezone(&Utc))
                    }),
                    source_note_id: row.get(4)?,
                    confidence: row.get(5)?,
                    status,
                })
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(signals)
    }

    // --- Profiles ---

    async fn upsert_profile(&self, entity_id: &str, note_id: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO profiles (entity_id, note_id, last_updated) VALUES (?1, ?2, ?3)
             ON CONFLICT(entity_id) DO UPDATE SET note_id = ?2, last_updated = ?3",
            rusqlite::params![entity_id, note_id, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_profile_note_id(&self, entity_id: &str) -> Result<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let result = conn.query_row(
            "SELECT note_id FROM profiles WHERE entity_id = ?1",
            rusqlite::params![entity_id],
            |row| row.get(0),
        );
        match result {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(KartaError::GraphStore(e.to_string())),
        }
    }

    async fn get_all_profiles(&self) -> Result<Vec<(String, String)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT entity_id, note_id FROM profiles")
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let profiles = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(profiles)
    }

    // --- Episodes ---

    async fn upsert_episode(&self, episode: &crate::note::Episode) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let tags_json = serde_json::to_string(&episode.topic_tags)?;
        conn.execute(
            "INSERT INTO episodes (id, narrative_note_id, start_time, end_time, session_id, topic_tags_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET narrative_note_id=?2, end_time=?4, topic_tags_json=?6",
            rusqlite::params![
                episode.id,
                episode.narrative_note_id,
                episode.start_time.to_rfc3339(),
                episode.end_time.to_rfc3339(),
                episode.session_id,
                tags_json,
            ],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_episode(&self, id: &str) -> Result<Option<crate::note::Episode>> {
        let episode_result = {
            let conn = self
                .conn
                .lock()
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            conn.query_row(
                "SELECT id, narrative_note_id, start_time, end_time, session_id, topic_tags_json FROM episodes WHERE id = ?1",
                rusqlite::params![id],
                |row| {
                    let tags_json: String = row.get(5)?;
                    Ok(crate::note::Episode {
                        id: row.get(0)?,
                        narrative: String::new(),
                        narrative_note_id: row.get(1)?,
                        start_time: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                            .unwrap_or_default()
                            .with_timezone(&Utc),
                        end_time: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                            .unwrap_or_default()
                            .with_timezone(&Utc),
                        session_id: row.get(4)?,
                        topic_tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                        note_ids: Vec::new(),
                    })
                },
            )
        }; // MutexGuard dropped here

        match episode_result {
            Ok(mut ep) => {
                ep.note_ids = self.get_notes_for_episode(&ep.id).await?;
                Ok(Some(ep))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(KartaError::GraphStore(e.to_string())),
        }
    }

    async fn get_episodes_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::note::Episode>> {
        let ids: Vec<String> = {
            let conn = self
                .conn
                .lock()
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            let mut stmt = conn
                .prepare("SELECT id FROM episodes WHERE session_id = ?1 ORDER BY start_time ASC")
                .map_err(|e| KartaError::GraphStore(e.to_string()))?;
            stmt.query_map(rusqlite::params![session_id], |row| row.get(0))
                .map_err(|e| KartaError::GraphStore(e.to_string()))?
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| KartaError::GraphStore(e.to_string()))?
        }; // MutexGuard dropped here before any .await

        let mut episodes = Vec::new();
        for id in ids {
            if let Some(ep) = self.get_episode(&id).await? {
                episodes.push(ep);
            }
        }
        Ok(episodes)
    }

    async fn add_note_to_episode(&self, note_id: &str, episode_id: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        conn.execute(
            "INSERT OR IGNORE INTO note_episodes (note_id, episode_id) VALUES (?1, ?2)",
            rusqlite::params![note_id, episode_id],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_episode_for_note(&self, note_id: &str) -> Result<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let result = conn.query_row(
            "SELECT episode_id FROM note_episodes WHERE note_id = ?1 LIMIT 1",
            rusqlite::params![note_id],
            |row| row.get(0),
        );
        match result {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(KartaError::GraphStore(e.to_string())),
        }
    }

    async fn get_notes_for_episode(&self, episode_id: &str) -> Result<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT note_id FROM note_episodes WHERE episode_id = ?1")
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let ids = stmt
            .query_map(rusqlite::params![episode_id], |row| row.get(0))
            .map_err(|e| KartaError::GraphStore(e.to_string()))?
            .collect::<std::result::Result<Vec<String>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(ids)
    }

    // --- Efficient link count ---

    async fn get_link_count(&self, note_id: &str) -> Result<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM links WHERE from_id = ?1",
                rusqlite::params![note_id],
                |row| row.get(0),
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(count as usize)
    }

    // --- Episode Digests (Phase Next) ---

    async fn upsert_episode_digest(&self, digest: &crate::note::EpisodeDigest) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let entities_json = serde_json::to_string(&digest.entities)?;
        let date_range_json = digest
            .date_range
            .as_ref()
            .map(|d| serde_json::to_string(d).unwrap_or_default());
        let aggregations_json = serde_json::to_string(&digest.aggregations)?;
        let topic_sequence_json = serde_json::to_string(&digest.topic_sequence)?;
        let events_json = serde_json::to_string(&digest.events)?;
        conn.execute(
            "INSERT OR REPLACE INTO episode_digests (id, episode_id, entities_json, date_range_json, aggregations_json, topic_sequence_json, events_json, digest_text, digest_note_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![digest.id, digest.episode_id, entities_json, date_range_json, aggregations_json, topic_sequence_json, events_json, digest.digest_text, digest.digest_note_id, digest.created_at.to_rfc3339()],
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_episode_digest(
        &self,
        episode_id: &str,
    ) -> Result<Option<crate::note::EpisodeDigest>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, episode_id, entities_json, date_range_json, aggregations_json, topic_sequence_json, events_json, digest_text, digest_note_id, created_at FROM episode_digests WHERE episode_id = ?1"
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let result = stmt.query_row(rusqlite::params![episode_id], |row| {
            let entities_str: String = row.get(2)?;
            let date_range_str: Option<String> = row.get(3)?;
            let agg_str: String = row.get(4)?;
            let topic_str: String = row.get(5)?;
            let events_str: String = row.get(6)?;
            let created_str: String = row.get(9)?;
            Ok(crate::note::EpisodeDigest {
                id: row.get(0)?,
                episode_id: row.get(1)?,
                entities: serde_json::from_str(&entities_str).unwrap_or_default(),
                date_range: date_range_str.and_then(|s| serde_json::from_str(&s).ok()),
                aggregations: serde_json::from_str(&agg_str).unwrap_or_default(),
                topic_sequence: serde_json::from_str(&topic_str).unwrap_or_default(),
                events: serde_json::from_str(&events_str).unwrap_or_default(),
                digest_text: row.get(7)?,
                digest_note_id: row.get(8)?,
                created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                    .unwrap_or_default()
                    .with_timezone(&chrono::Utc),
            })
        });

        match result {
            Ok(digest) => Ok(Some(digest)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(KartaError::GraphStore(e.to_string())),
        }
    }

    async fn get_all_episode_digests(&self) -> Result<Vec<crate::note::EpisodeDigest>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, episode_id, entities_json, date_range_json, aggregations_json, topic_sequence_json, events_json, digest_text, digest_note_id, created_at FROM episode_digests ORDER BY created_at"
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let digests = stmt
            .query_map([], |row| {
                let entities_str: String = row.get(2)?;
                let date_range_str: Option<String> = row.get(3)?;
                let agg_str: String = row.get(4)?;
                let topic_str: String = row.get(5)?;
                let events_str: String = row.get(6)?;
                let created_str: String = row.get(9)?;
                Ok(crate::note::EpisodeDigest {
                    id: row.get(0)?,
                    episode_id: row.get(1)?,
                    entities: serde_json::from_str(&entities_str).unwrap_or_default(),
                    date_range: date_range_str.and_then(|s| serde_json::from_str(&s).ok()),
                    aggregations: serde_json::from_str(&agg_str).unwrap_or_default(),
                    topic_sequence: serde_json::from_str(&topic_str).unwrap_or_default(),
                    events: serde_json::from_str(&events_str).unwrap_or_default(),
                    digest_text: row.get(7)?,
                    digest_note_id: row.get(8)?,
                    created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                        .unwrap_or_default()
                        .with_timezone(&chrono::Utc),
                })
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(digests.filter_map(|r| r.ok()).collect())
    }

    async fn get_undigested_episode_ids(&self) -> Result<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT e.id FROM episodes e LEFT JOIN episode_digests d ON e.id = d.episode_id WHERE d.id IS NULL"
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let ids = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(ids.filter_map(|r| r.ok()).collect())
    }

    async fn upsert_cross_episode_digest(
        &self,
        digest: &crate::note::CrossEpisodeDigest,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let entity_timeline_json = serde_json::to_string(&digest.entity_timeline)?;
        let cross_aggregations_json = serde_json::to_string(&digest.cross_aggregations)?;
        let events_json = serde_json::to_string(&digest.events)?;
        let topic_progression_json = serde_json::to_string(&digest.topic_progression)?;
        conn.execute(
            "INSERT OR REPLACE INTO cross_episode_digests (id, scope_id, entity_timeline_json, cross_aggregations_json, events_json, topic_progression_json, digest_text, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![digest.id, digest.scope_id, entity_timeline_json, cross_aggregations_json, events_json, topic_progression_json, digest.digest_text, digest.created_at.to_rfc3339()],
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_all_cross_episode_digests(&self) -> Result<Vec<crate::note::CrossEpisodeDigest>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, scope_id, entity_timeline_json, cross_aggregations_json, events_json, topic_progression_json, digest_text, created_at FROM cross_episode_digests ORDER BY created_at"
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let digests = stmt
            .query_map([], |row| {
                let timeline_str: String = row.get(2)?;
                let agg_str: String = row.get(3)?;
                let events_str: String = row.get(4)?;
                let topic_str: String = row.get(5)?;
                let created_str: String = row.get(7)?;
                Ok(crate::note::CrossEpisodeDigest {
                    id: row.get(0)?,
                    scope_id: row.get(1)?,
                    entity_timeline: serde_json::from_str(&timeline_str).unwrap_or_default(),
                    cross_aggregations: serde_json::from_str(&agg_str).unwrap_or_default(),
                    events: serde_json::from_str(&events_str).unwrap_or_default(),
                    topic_progression: serde_json::from_str(&topic_str).unwrap_or_default(),
                    digest_text: row.get(6)?,
                    created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                        .unwrap_or_default()
                        .with_timezone(&chrono::Utc),
                })
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(digests.filter_map(|r| r.ok()).collect())
    }

    // --- Atomic Fact Metadata (Phase Next) ---

    async fn record_fact(
        &self,
        fact_id: &str,
        source_note_id: &str,
        ordinal: u32,
        subject: Option<&str>,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO atomic_facts (id, source_note_id, ordinal, subject) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![fact_id, source_note_id, ordinal, subject],
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_facts_by_subject(&self, subject: &str) -> Result<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id FROM atomic_facts WHERE subject = ?1 ORDER BY created_at")
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let ids = stmt
            .query_map(rusqlite::params![subject], |row| row.get(0))
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(ids.filter_map(|r| r.ok()).collect())
    }

    // --- Episode Links (Phase Next) ---

    async fn add_episode_link(
        &self,
        from_id: &str,
        to_id: &str,
        link_type: &str,
        entity: Option<&str>,
        reason: &str,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        conn.execute(
            "INSERT OR IGNORE INTO episode_links (from_episode_id, to_episode_id, link_type, entity, reason) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![from_id, to_id, link_type, entity, reason],
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_episode_links(
        &self,
        episode_id: &str,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT to_episode_id, link_type, entity FROM episode_links WHERE from_episode_id = ?1 UNION SELECT from_episode_id, link_type, entity FROM episode_links WHERE to_episode_id = ?1"
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let links = stmt
            .query_map(rusqlite::params![episode_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(links.filter_map(|r| r.ok()).collect())
    }

    async fn get_episodes_for_entity(&self, entity: &str) -> Result<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT from_episode_id FROM episode_links WHERE entity = ?1 AND link_type = 'entity_continuity' UNION SELECT DISTINCT to_episode_id FROM episode_links WHERE entity = ?1 AND link_type = 'entity_continuity'"
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let ids = stmt
            .query_map(rusqlite::params![entity], |row| row.get(0))
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(ids.filter_map(|r| r.ok()).collect())
    }

    async fn get_schema_meta(&self) -> Result<crate::migrate::SchemaMeta> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        migrate::load_schema_meta(&conn)
    }

    // --- Contradictions ---

    async fn upsert_contradiction(
        &self,
        contradiction: &crate::contradiction::Contradiction,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let source_ids_json = serde_json::to_string(&contradiction.source_note_ids)
            .map_err(KartaError::Serialization)?;
        let resolution_json = contradiction
            .resolution
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(KartaError::Serialization)?;
        let status_str = match contradiction.status {
            crate::contradiction::ContradictionStatus::Open => "open",
            crate::contradiction::ContradictionStatus::Resolved => "resolved",
            crate::contradiction::ContradictionStatus::Ignored => "ignored",
        };
        conn.execute(
            "INSERT INTO contradictions (id, entity, scope_id, source_note_ids_json, description, dream_run_id, status, resolution_json, resolved_at, resolved_by, ignore_reason, ignored_at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
             ON CONFLICT(id) DO UPDATE SET
                 entity=?2, scope_id=?3, source_note_ids_json=?4, description=?5,
                 dream_run_id=?6, status=?7, resolution_json=?8, resolved_at=?9,
                 resolved_by=?10, ignore_reason=?11, ignored_at=?12",
            rusqlite::params![
                contradiction.id,
                contradiction.entity,
                contradiction.scope_id,
                source_ids_json,
                contradiction.description,
                contradiction.dream_run_id,
                status_str,
                resolution_json,
                contradiction.resolved_at.map(|t| t.to_rfc3339()),
                contradiction.resolved_by,
                contradiction.ignore_reason,
                contradiction.ignored_at.map(|t| t.to_rfc3339()),
                contradiction.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_contradiction(
        &self,
        id: &str,
    ) -> Result<Option<crate::contradiction::Contradiction>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, entity, scope_id, source_note_ids_json, description, dream_run_id, status, resolution_json, resolved_at, resolved_by, ignore_reason, ignored_at, created_at FROM contradictions WHERE id = ?1",
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        match stmt.query_row(rusqlite::params![id], parse_contradiction_row) {
            Ok(c) => Ok(Some(c)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(KartaError::GraphStore(e.to_string())),
        }
    }

    async fn list_contradictions(
        &self,
        scope_id: Option<&str>,
        status: Option<crate::contradiction::ContradictionStatus>,
    ) -> Result<Vec<crate::contradiction::Contradiction>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut sql = "SELECT id, entity, scope_id, source_note_ids_json, description, dream_run_id, status, resolution_json, resolved_at, resolved_by, ignore_reason, ignored_at, created_at FROM contradictions WHERE 1=1".to_string();
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(scope) = scope_id {
            sql.push_str(" AND scope_id = ?");
            params.push(Box::new(scope.to_string()));
        }
        if let Some(s) = status {
            let status_str = match s {
                crate::contradiction::ContradictionStatus::Open => "open",
                crate::contradiction::ContradictionStatus::Resolved => "resolved",
                crate::contradiction::ContradictionStatus::Ignored => "ignored",
            };
            sql.push_str(" AND status = ?");
            params.push(Box::new(status_str.to_string()));
        }

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let rows = stmt
            .query_map(
                rusqlite::params_from_iter(params.iter()),
                parse_contradiction_row,
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))
    }

    async fn list_contradictions_for_entity(
        &self,
        entity: &str,
    ) -> Result<Vec<crate::contradiction::Contradiction>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, entity, scope_id, source_note_ids_json, description, dream_run_id, status, resolution_json, resolved_at, resolved_by, ignore_reason, ignored_at, created_at FROM contradictions WHERE entity = ?1 ORDER BY created_at DESC",
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let rows = stmt
            .query_map(rusqlite::params![entity], parse_contradiction_row)
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| KartaError::GraphStore(e.to_string()))
    }

    async fn resolve_contradiction(
        &self,
        id: &str,
        resolution: crate::contradiction::ContradictionResolution,
        resolved_by: Option<&str>,
    ) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let resolution_json =
            serde_json::to_string(&resolution).map_err(KartaError::Serialization)?;
        let now = Utc::now().to_rfc3339();
        let rows = conn
            .execute(
                "UPDATE contradictions SET status = 'resolved', resolution_json = ?2, resolved_at = ?3, resolved_by = ?4, ignored_at = NULL WHERE id = ?1 AND status = 'open'",
                rusqlite::params![id, resolution_json, now, resolved_by],
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        if rows == 0 {
            return Err(KartaError::GraphStore(format!(
                "Contradiction {id} not found or already resolved/ignored"
            )));
        }
        Ok(())
    }

    async fn ignore_contradiction(&self, id: &str, reason: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let rows = conn
            .execute(
                "UPDATE contradictions SET status = 'ignored', ignore_reason = ?2, ignored_at = ?3, resolved_at = NULL, resolved_by = NULL WHERE id = ?1 AND status = 'open'",
                rusqlite::params![id, reason, now],
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        if rows == 0 {
            return Err(KartaError::GraphStore(format!(
                "Contradiction {id} not found or already resolved/ignored"
            )));
        }
        Ok(())
    }

    // --- Procedural Rules ---

    async fn upsert_procedural_rule(&self, rule: &crate::rules::ProceduralRule) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let condition_json =
            serde_json::to_string(&rule.condition).map_err(KartaError::Serialization)?;
        let actions_json =
            serde_json::to_string(&rule.actions).map_err(KartaError::Serialization)?;
        conn.execute(
            "INSERT INTO procedural_rules (id, name, description, condition_json, actions_json, enabled, protected, source_note_id, fire_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(id) DO UPDATE SET
                 name=?2, description=?3, condition_json=?4, actions_json=?5,
                 enabled=?6, protected=?7, source_note_id=?8, fire_count=?9, updated_at=?11",
            rusqlite::params![
                rule.id,
                rule.name,
                rule.description,
                condition_json,
                actions_json,
                rule.enabled as i32,
                rule.protected as i32,
                rule.source_note_id,
                rule.fire_count as i64,
                rule.created_at.to_rfc3339(),
                rule.updated_at.to_rfc3339(),
            ],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn list_procedural_rules(&self) -> Result<Vec<crate::rules::ProceduralRule>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, name, description, condition_json, actions_json, enabled, protected, source_note_id, fire_count, created_at, updated_at FROM procedural_rules ORDER BY name"
        ).map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                let condition: crate::rules::RuleCondition =
                    serde_json::from_str(&row.get::<_, String>(3)?)
                        .unwrap_or(crate::rules::RuleCondition::Always);
                let actions: Vec<crate::rules::RuleAction> =
                    serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default();
                let created_at = DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                    .unwrap_or_default()
                    .with_timezone(&Utc);
                let updated_at = DateTime::parse_from_rfc3339(&row.get::<_, String>(10)?)
                    .unwrap_or_default()
                    .with_timezone(&Utc);
                Ok(crate::rules::ProceduralRule {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                    condition,
                    actions,
                    enabled: row.get::<_, i32>(5)? != 0,
                    protected: row.get::<_, i32>(6)? != 0,
                    source_note_id: row.get(7)?,
                    fire_count: row.get::<_, i64>(8)? as u64,
                    created_at,
                    updated_at,
                })
            })
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    async fn disable_procedural_rule(&self, rule_id: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE procedural_rules SET enabled = 0, updated_at = ?2 WHERE id = ?1",
            rusqlite::params![rule_id, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn increment_rule_fire_count(&self, rule_id: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE procedural_rules SET fire_count = fire_count + 1, updated_at = ?2 WHERE id = ?1",
            rusqlite::params![rule_id, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }
}

fn parse_contradiction_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<crate::contradiction::Contradiction> {
    let status_str: String = row.get(6)?;
    let status = match status_str.as_str() {
        "resolved" => crate::contradiction::ContradictionStatus::Resolved,
        "ignored" => crate::contradiction::ContradictionStatus::Ignored,
        _ => crate::contradiction::ContradictionStatus::Open,
    };

    let source_note_ids: Vec<String> = {
        let json: String = row.get(3)?;
        serde_json::from_str(&json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
        })?
    };

    let resolution: Option<crate::contradiction::ContradictionResolution> = {
        let json: Option<String> = row.get(7)?;
        match json {
            Some(j) => Some(serde_json::from_str(&j).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    7,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?),
            None => None,
        }
    };

    let resolved_at: Option<DateTime<Utc>> = {
        let s: Option<String> = row.get(8)?;
        s.and_then(|t| {
            DateTime::parse_from_rfc3339(&t)
                .ok()
                .map(|d| d.with_timezone(&Utc))
        })
    };

    let ignored_at: Option<DateTime<Utc>> = {
        let s: Option<String> = row.get(11)?;
        s.and_then(|t| {
            DateTime::parse_from_rfc3339(&t)
                .ok()
                .map(|d| d.with_timezone(&Utc))
        })
    };

    let created_at_str: String = row.get(12)?;
    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .unwrap_or_default()
        .with_timezone(&Utc);

    Ok(crate::contradiction::Contradiction {
        id: row.get(0)?,
        entity: row.get(1)?,
        scope_id: row.get(2)?,
        source_note_ids,
        description: row.get(4)?,
        dream_run_id: row.get(5)?,
        status,
        resolution,
        resolved_at,
        resolved_by: row.get(9)?,
        ignore_reason: row.get(10)?,
        ignored_at,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols_of(conn: &Connection, table: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({})", table))
            .unwrap();
        stmt.query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    }

    /// Running the migration twice must be a no-op on the second call and
    /// preserve any rows already present.
    #[test]
    fn migrate_links_table_is_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        // New-schema table (what init() creates) — migration should be a no-op.
        conn.execute_batch(
            "CREATE TABLE links (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                link_type TEXT NOT NULL DEFAULT 'semantic',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (from_id, to_id, link_type)
            );
            INSERT INTO links (from_id, to_id, reason, weight, link_type, created_at)
                VALUES ('a', 'b', 'r', 1.0, 'semantic', '2024-01-01T00:00:00Z');",
        )
        .unwrap();

        SqliteGraphStore::migrate_links_table(&conn).expect("1st migrate");
        SqliteGraphStore::migrate_links_table(&conn).expect("2nd migrate");

        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM links", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1, "row preserved across idempotent migrations");
    }

    /// Upgrading from the pre-ACTIVATE schema must add weight + link_type,
    /// default existing rows to weight=1.0 / link_type='semantic', and
    /// rebuild the PK to `(from_id, to_id, link_type)`.
    #[test]
    fn migrate_links_table_upgrades_legacy_schema() {
        let conn = Connection::open_in_memory().unwrap();
        // Legacy schema: no weight / link_type, PK = (from_id, to_id).
        conn.execute_batch(
            "CREATE TABLE links (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (from_id, to_id)
            );
            INSERT INTO links (from_id, to_id, reason, created_at)
                VALUES ('a', 'b', 'legacy', '2024-01-01T00:00:00Z');",
        )
        .unwrap();

        SqliteGraphStore::migrate_links_table(&conn).expect("migrate");

        let cols = cols_of(&conn, "links");
        assert!(cols.iter().any(|c| c == "weight"), "weight column added");
        assert!(
            cols.iter().any(|c| c == "link_type"),
            "link_type column added"
        );

        let (weight, link_type): (f64, String) = conn
            .query_row(
                "SELECT weight, link_type FROM links WHERE from_id = 'a' AND to_id = 'b'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert!((weight - 1.0).abs() < 1e-9, "default weight applied");
        assert_eq!(link_type, "semantic", "default link_type applied");

        // PK is (from_id, to_id, link_type): we should be able to insert a
        // second row with a different link_type.
        conn.execute(
            "INSERT INTO links (from_id, to_id, reason, weight, link_type, created_at)
                VALUES ('a', 'b', 'chain', 1.0, 'follows', '2024-01-02T00:00:00Z')",
            [],
        )
        .expect("second row with different link_type should fit the new PK");

        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM links WHERE from_id = 'a' AND to_id = 'b'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2);
    }

    /// Regression test for the production bug on Cloud Run revision -00016
    /// (2026-04-22): a database with the pre-ACTIVATE `links` schema (no
    /// `link_type` column) caused `init()` to fail with
    /// `no such column: link_type` because `CREATE INDEX idx_links_type`
    /// ran before `migrate_links_table`. Init must succeed and migrate.
    #[tokio::test]
    async fn init_succeeds_on_pre_activate_legacy_links_schema() {
        // No tempfile dev-dep here; mirror the on-disk fixture pattern used
        // in tests/activate_retrieval_invariants.rs.
        let suffix = uuid::Uuid::new_v4().to_string();
        let data_dir = format!("/tmp/karta-pre-activate-{}", &suffix[..8]);
        let _ = std::fs::remove_dir_all(&data_dir);
        std::fs::create_dir_all(&data_dir).unwrap();
        let db_path = format!("{}/karta.db", &data_dir);

        // Simulate a pre-ACTIVATE database: create the old links schema directly.
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE links (
                     from_id TEXT NOT NULL,
                     to_id TEXT NOT NULL,
                     reason TEXT NOT NULL,
                     created_at TEXT NOT NULL DEFAULT (datetime('now')),
                     PRIMARY KEY (from_id, to_id)
                 );
                 INSERT INTO links (from_id, to_id, reason) VALUES ('a', 'b', 'test-edge');",
            )
            .unwrap();
        }

        // Now run the real init path — this would fail before the fix because
        // CREATE INDEX on link_type runs before the migration that adds the column.
        let store = SqliteGraphStore::new(&data_dir).unwrap();
        crate::store::GraphStore::init(&store).await.expect(
            "init must succeed on a legacy pre-ACTIVATE schema (production bug 2026-04-22)",
        );

        // Verify the migration actually ran: link_type column exists, weight defaults to 1.0,
        // and the existing edge survived with link_type='semantic'.
        let conn = Connection::open(&db_path).unwrap();
        let (to_id, link_type, weight): (String, String, f64) = conn
            .query_row(
                "SELECT to_id, link_type, weight FROM links WHERE from_id = 'a' LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("legacy edge should survive migration");
        assert_eq!(to_id, "b");
        assert_eq!(link_type, "semantic");
        assert!((weight - 1.0).abs() < 1e-9, "default weight should be 1.0");

        let _ = std::fs::remove_dir_all(&data_dir);
    }
}
