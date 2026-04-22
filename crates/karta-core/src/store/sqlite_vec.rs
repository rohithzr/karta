// SQLite + sqlite-vec vector store implementation.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension};
use serde_json;

use crate::error::{KartaError, Result};
use crate::note::{AtomicFact, MemoryNote, NoteStatus, Provenance};

pub struct SqliteVectorStore {
    conn: Arc<Mutex<Connection>>,
    embedding_dim: usize,
}

impl SqliteVectorStore {
    /// Open (or create) a SQLite vector store at `data_dir/karta.db`.
    ///
    /// Registers the sqlite-vec FFI extension, enables WAL mode, and
    /// initialises the schema before returning.
    pub async fn new(data_dir: &str, embedding_dim: usize) -> Result<Self> {
        // Register the sqlite-vec extension process-globally before any
        // connection is opened — sqlite3_auto_extension is the only safe
        // way to load a virtual-table extension into a bundled SQLite.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(
                std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ()),
            ));
        }

        std::fs::create_dir_all(data_dir)
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let path = format!("{}/karta.db", data_dir);
        let conn = Connection::open(&path)
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            embedding_dim,
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Wrap an existing shared connection (used to share one DB file with
    /// `SqliteGraphStore` — avoids two WAL writers on the same path).
    pub fn with_connection(conn: Arc<Mutex<Connection>>, embedding_dim: usize) -> Result<Self> {
        let store = Self { conn, embedding_dim };
        store.init_schema()?;
        Ok(store)
    }

    /// Expose the underlying connection so callers can pass it to
    /// `SqliteGraphStore::with_connection`.
    pub fn connection(&self) -> Arc<Mutex<Connection>> {
        Arc::clone(&self.conn)
    }

    // ── Schema ────────────────────────────────────────────────────────────────

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let dim = self.embedding_dim;

        // Schema is recreated from scratch on first run of the new code.
        // No migration support — existing data dirs (`/tmp/karta-*`, etc.)
        // must be deleted and re-ingested. We're experimental; this is fine.
        let ddl = format!(
            "
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY NOT NULL,
                content TEXT NOT NULL,
                context TEXT NOT NULL,
                keywords_json TEXT NOT NULL DEFAULT '[]',
                tags_json TEXT NOT NULL DEFAULT '[]',
                provenance_json TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status_json TEXT NOT NULL DEFAULT '\"Active\"',
                last_accessed_at TEXT NOT NULL,
                turn_index INTEGER,
                source_timestamp TEXT NOT NULL,
                session_id TEXT,
                embedding BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS atomic_facts (
                id TEXT PRIMARY KEY NOT NULL,
                content TEXT NOT NULL,
                source_note_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL DEFAULT 0,
                subject TEXT,
                created_at TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_source ON atomic_facts(source_note_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(
                id text primary key,
                embedding float[{dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(
                id text primary key,
                embedding float[{dim}]
            );
            "
        );

        conn.execute_batch(&ddl)?;
        Ok(())
    }

    // ── Embedding helpers ─────────────────────────────────────────────────────

    pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
        embedding.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
        blob.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    // ── Row deserialiser ──────────────────────────────────────────────────────

    fn row_to_note(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryNote> {
        let id: String = row.get("id")?;
        let content: String = row.get("content")?;
        let context: String = row.get("context")?;
        let keywords_json: String = row.get("keywords_json")?;
        let tags_json: String = row.get("tags_json")?;
        let provenance_json: String = row.get("provenance_json")?;
        let confidence: f32 = row.get("confidence")?;
        let created_at: String = row.get("created_at")?;
        let updated_at: String = row.get("updated_at")?;
        let status_json: String = row.get("status_json")?;
        let last_accessed_at: String = row.get("last_accessed_at")?;
        let turn_index: Option<u32> = row.get("turn_index")?;
        let source_timestamp: String = row.get("source_timestamp")?;
        let session_id: Option<String> = row.get("session_id")?;
        let embedding_blob: Vec<u8> = row.get("embedding")?;

        let keywords: Vec<String> =
            serde_json::from_str(&keywords_json).unwrap_or_default();
        let tags: Vec<String> =
            serde_json::from_str(&tags_json).unwrap_or_default();
        let provenance: Provenance =
            serde_json::from_str(&provenance_json).unwrap_or(Provenance::Observed);
        let status: NoteStatus =
            serde_json::from_str(&status_json).unwrap_or_default();

        let created_at = created_at
            .parse()
            .unwrap_or_else(|_| chrono::Utc::now());
        let updated_at = updated_at
            .parse()
            .unwrap_or_else(|_| chrono::Utc::now());
        let last_accessed_at = last_accessed_at
            .parse()
            .unwrap_or_else(|_| chrono::Utc::now());
        let source_timestamp = source_timestamp
            .parse()
            .unwrap_or_else(|_| chrono::Utc::now());

        Ok(MemoryNote {
            id,
            content,
            context,
            keywords,
            tags,
            links: Vec::new(), // Links live in GraphStore
            embedding: Self::blob_to_embedding(&embedding_blob),
            created_at,
            updated_at,
            evolution_history: Vec::new(), // History lives in GraphStore
            provenance,
            confidence,
            status,
            last_accessed_at,
            turn_index,
            source_timestamp,
            session_id,
        })
    }
}

// ── VectorStore trait ─────────────────────────────────────────────────────────

#[async_trait]
impl crate::store::VectorStore for SqliteVectorStore {
    async fn count(&self) -> Result<usize> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    async fn upsert(&self, note: &MemoryNote) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;

        let keywords_json = serde_json::to_string(&note.keywords)?;
        let tags_json = serde_json::to_string(&note.tags)?;
        let provenance_json = serde_json::to_string(&note.provenance)?;
        let status_json = serde_json::to_string(&note.status)?;
        let created_at = note.created_at.to_rfc3339();
        let updated_at = note.updated_at.to_rfc3339();
        let last_accessed_at = note.last_accessed_at.to_rfc3339();
        let source_timestamp = note.source_timestamp.to_rfc3339();
        let embedding_blob = Self::embedding_to_blob(&note.embedding);

        let tx = conn.unchecked_transaction()?;

        tx.execute(
            "INSERT OR REPLACE INTO notes
             (id, content, context, keywords_json, tags_json, provenance_json,
              confidence, created_at, updated_at, status_json, last_accessed_at,
              turn_index, source_timestamp, session_id, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            rusqlite::params![
                note.id, note.content, note.context,
                keywords_json, tags_json, provenance_json,
                note.confidence, created_at, updated_at,
                status_json, last_accessed_at,
                note.turn_index, source_timestamp, note.session_id,
                embedding_blob,
            ],
        )?;

        tx.execute("DELETE FROM notes_vec WHERE id = ?1", [&note.id])?;
        tx.execute(
            "INSERT INTO notes_vec (id, embedding) VALUES (?1, ?2)",
            rusqlite::params![note.id, embedding_blob],
        )?;

        tx.commit()?;
        Ok(())
    }

    async fn find_similar(
        &self,
        embedding: &[f32],
        top_k: usize,
        exclude_ids: &[&str],
    ) -> Result<Vec<(MemoryNote, f32)>> {
        let query_blob = Self::embedding_to_blob(embedding);
        let limit = (top_k + exclude_ids.len()) as i64;

        // Scope the lock so conn and stmt are dropped before any await point.
        let filtered: Vec<(String, f64)> = {
            let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
            let mut stmt = conn.prepare(
                "SELECT id, distance FROM notes_vec \
                 WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2",
            )?;
            let rows: Vec<(String, f64)> = stmt
                .query_map(rusqlite::params![query_blob, limit], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                })?
                .filter_map(|r| r.ok())
                .collect();
            rows.into_iter()
                .filter(|(id, _)| !exclude_ids.contains(&id.as_str()))
                .take(top_k)
                .collect()
        }; // conn and stmt dropped here

        if filtered.is_empty() {
            return Ok(Vec::new());
        }

        let distance_map: std::collections::HashMap<String, f64> =
            filtered.iter().map(|(id, d)| (id.clone(), *d)).collect();
        let ids: Vec<&str> = filtered.iter().map(|(id, _)| id.as_str()).collect();

        let notes = self.get_many(&ids).await?;

        let mut scored: Vec<(MemoryNote, f32)> = notes
            .into_iter()
            .map(|note| {
                let distance = distance_map.get(&note.id).copied().unwrap_or(0.0);
                (note, 1.0 / (1.0 + distance as f32))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryNote>> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let result = conn
            .prepare_cached(
                "SELECT id, content, context, keywords_json, tags_json, provenance_json,
                        confidence, created_at, updated_at, status_json, last_accessed_at,
                        turn_index, source_timestamp, session_id, embedding
                 FROM notes WHERE id = ?1",
            )?
            .query_row([id], Self::row_to_note)
            .optional()?;
        Ok(result)
    }

    async fn get_many(&self, ids: &[&str]) -> Result<Vec<MemoryNote>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let placeholders: String = (1..=ids.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, content, context, keywords_json, tags_json, provenance_json,
                    confidence, created_at, updated_at, status_json, last_accessed_at,
                    turn_index, source_timestamp, session_id, embedding
             FROM notes WHERE id IN ({})",
            placeholders
        );
        let params: Vec<&dyn rusqlite::types::ToSql> =
            ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params.as_slice(), Self::row_to_note)?;
        let mut notes = Vec::new();
        for row in rows {
            notes.push(row?);
        }
        Ok(notes)
    }

    async fn get_all(&self) -> Result<Vec<MemoryNote>> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, content, context, keywords_json, tags_json, provenance_json,
                    confidence, created_at, updated_at, status_json, last_accessed_at,
                    turn_index, source_timestamp, session_id, embedding
             FROM notes",
        )?;
        let rows = stmt.query_map([], Self::row_to_note)?;
        let mut notes = Vec::new();
        for row in rows {
            notes.push(row?);
        }
        Ok(notes)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        conn.execute("DELETE FROM notes WHERE id = ?1", [id])?;
        conn.execute("DELETE FROM notes_vec WHERE id = ?1", [id])?;
        Ok(())
    }

    async fn upsert_fact(&self, fact: &AtomicFact) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let created_at = fact.created_at.to_rfc3339();
        let embedding_blob = Self::embedding_to_blob(&fact.embedding);
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO atomic_facts
             (id, content, source_note_id, ordinal, subject, created_at, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                fact.id, fact.content, fact.source_note_id,
                fact.ordinal, fact.subject, created_at, embedding_blob,
            ],
        )?;
        tx.execute("DELETE FROM facts_vec WHERE id = ?1", [&fact.id])?;
        tx.execute(
            "INSERT INTO facts_vec (id, embedding) VALUES (?1, ?2)",
            rusqlite::params![fact.id, embedding_blob],
        )?;
        tx.commit()?;
        Ok(())
    }

    async fn find_similar_facts(
        &self,
        embedding: &[f32],
        top_k: usize,
        exclude_source_note_ids: &[&str],
    ) -> Result<Vec<(AtomicFact, f32)>> {
        let query_blob = Self::embedding_to_blob(embedding);
        let limit = (top_k + exclude_source_note_ids.len() * 5) as i64;

        // All DB work in one block — lock dropped before any await
        let result: Vec<(AtomicFact, f32)> = {
            let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;

            // Step 1: kNN query
            let mut stmt = conn.prepare(
                "SELECT id, distance FROM facts_vec WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2",
            )?;
            let knn_rows: Vec<(String, f64)> = stmt
                .query_map(rusqlite::params![query_blob, limit], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                })?
                .filter_map(|r| r.ok())
                .collect();
            drop(stmt); // explicit drop to release borrow on conn

            if knn_rows.is_empty() {
                Vec::new()
            } else {
                // Step 2: Batch fetch all facts in one query
                let distance_map: std::collections::HashMap<String, f64> = knn_rows
                    .iter()
                    .map(|(id, d)| (id.clone(), *d))
                    .collect();
                let placeholders: String = (1..=knn_rows.len())
                    .map(|i| format!("?{}", i))
                    .collect::<Vec<_>>()
                    .join(",");
                let sql = format!(
                    "SELECT id, content, source_note_id, ordinal, subject, created_at, embedding
                     FROM atomic_facts WHERE id IN ({})",
                    placeholders
                );
                let mut stmt2 = conn.prepare(&sql)?;
                let ids: Vec<&str> = knn_rows.iter().map(|(id, _)| id.as_str()).collect();
                let params: Vec<&dyn rusqlite::types::ToSql> =
                    ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();

                let facts: Vec<AtomicFact> = stmt2
                    .query_map(params.as_slice(), |row| {
                        let embedding_blob: Vec<u8> = row.get("embedding")?;
                        let created_str: String = row.get("created_at")?;
                        let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                            .unwrap_or_default()
                            .with_timezone(&chrono::Utc);
                        Ok(AtomicFact {
                            id: row.get("id")?,
                            content: row.get("content")?,
                            source_note_id: row.get("source_note_id")?,
                            ordinal: row.get("ordinal")?,
                            subject: row.get("subject")?,
                            embedding: Self::blob_to_embedding(&embedding_blob),
                            created_at,
                            // STEP1.5 Task 2 will populate these from new columns.
                            source_timestamp: created_at,
                            occurred_start: None,
                            occurred_end: None,
                            occurred_confidence: crate::read::temporal::ConfidenceBand::None,
                        })
                    })?
                    .filter_map(|r| r.ok())
                    .collect();

                // Step 3: Filter by exclude + pair with scores
                let mut scored: Vec<(AtomicFact, f32)> = facts
                    .into_iter()
                    .filter(|f| !exclude_source_note_ids.contains(&f.source_note_id.as_str()))
                    .filter_map(|fact| {
                        let distance = distance_map.get(&fact.id).copied().unwrap_or(0.0);
                        Some((fact, 1.0 / (1.0 + distance as f32)))
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(top_k);
                scored
            }
        }; // lock dropped here

        Ok(result)
    }

    async fn get_facts_for_note(&self, note_id: &str) -> Result<Vec<AtomicFact>> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, content, source_note_id, ordinal, subject, created_at, embedding
             FROM atomic_facts WHERE source_note_id = ?1 ORDER BY ordinal",
        )?;
        let facts = stmt
            .query_map([note_id], |row| {
                let embedding_blob: Vec<u8> = row.get("embedding")?;
                let created_str: String = row.get("created_at")?;
                let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                    .unwrap_or_default()
                    .with_timezone(&chrono::Utc);
                Ok(AtomicFact {
                    id: row.get("id")?,
                    content: row.get("content")?,
                    source_note_id: row.get("source_note_id")?,
                    ordinal: row.get("ordinal")?,
                    subject: row.get("subject")?,
                    embedding: Self::blob_to_embedding(&embedding_blob),
                    created_at,
                    // STEP1.5 Task 2 will populate these from new columns.
                    source_timestamp: created_at,
                    occurred_start: None,
                    occurred_end: None,
                    occurred_confidence: crate::read::temporal::ConfidenceBand::None,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(facts)
    }
}
