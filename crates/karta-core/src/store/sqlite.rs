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

#[async_trait]
impl crate::store::GraphStore for SqliteGraphStore {
    async fn init(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS links (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (from_id, to_id)
            );

            CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_id);
            CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_id);

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

        // Bidirectional: insert both directions
        conn.execute(
            "INSERT OR IGNORE INTO links (from_id, to_id, reason, created_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![from_id, to_id, reason, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        conn.execute(
            "INSERT OR IGNORE INTO links (from_id, to_id, reason, created_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![to_id, from_id, reason, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(())
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
