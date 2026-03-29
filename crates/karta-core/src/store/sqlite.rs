use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::Connection;
use std::sync::Mutex;

use crate::dream::DreamRun;
use crate::error::{KartaError, Result};
use crate::note::EvolutionRecord;

pub struct SqliteGraphStore {
    conn: Mutex<Connection>,
}

impl SqliteGraphStore {
    pub fn new(data_dir: &str) -> Result<Self> {
        let path = format!("{}/karta.db", data_dir);
        std::fs::create_dir_all(data_dir)
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        let conn = Connection::open(&path)
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        // Enable WAL mode for concurrent reads during dream writes
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
            ",
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn add_link(&self, from_id: &str, to_id: &str, reason: &str) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO evolution_history (note_id, triggered_by, previous_context, evolved_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![note_id, triggered_by, previous_context, now],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;

        Ok(())
    }

    async fn get_evolution_history(&self, note_id: &str) -> Result<Vec<EvolutionRecord>> {
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
                        DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let count = conn
            .execute(
                "UPDATE foresight_signals SET status = 'Expired' WHERE status = 'Active' AND valid_until IS NOT NULL AND valid_until < ?1",
                rusqlite::params![before.to_rfc3339()],
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(count)
    }

    async fn get_foresights_for_note(&self, note_id: &str) -> Result<Vec<crate::note::ForesightSignal>> {
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
                        DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
            let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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

    async fn get_episodes_for_session(&self, session_id: &str) -> Result<Vec<crate::note::Episode>> {
        let ids: Vec<String> = {
            let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
        conn.execute(
            "INSERT OR IGNORE INTO note_episodes (note_id, episode_id) VALUES (?1, ?2)",
            rusqlite::params![note_id, episode_id],
        )
        .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(())
    }

    async fn get_episode_for_note(&self, note_id: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
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
        let conn = self.conn.lock().map_err(|e| KartaError::GraphStore(e.to_string()))?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM links WHERE from_id = ?1",
                rusqlite::params![note_id],
                |row| row.get(0),
            )
            .map_err(|e| KartaError::GraphStore(e.to_string()))?;
        Ok(count as usize)
    }
}
