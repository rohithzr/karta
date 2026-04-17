// SQLite + sqlite-vec vector store implementation.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension};
use serde_json;

use crate::error::{KartaError, Result};
use crate::note::{MemoryNote, NoteStatus, Provenance};

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
                source_timestamp TEXT,
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
        let source_timestamp: Option<String> = row.get("source_timestamp")?;
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
            .as_deref()
            .and_then(|s| s.parse().ok());

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
        let source_timestamp = note.source_timestamp.map(|t| t.to_rfc3339());
        let embedding_blob = Self::embedding_to_blob(&note.embedding);

        let tx = conn.unchecked_transaction()?;

        tx.execute(
            "INSERT OR REPLACE INTO notes
             (id, content, context, keywords_json, tags_json, provenance_json,
              confidence, created_at, updated_at, status_json, last_accessed_at,
              turn_index, source_timestamp, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            rusqlite::params![
                note.id, note.content, note.context,
                keywords_json, tags_json, provenance_json,
                note.confidence, created_at, updated_at,
                status_json, last_accessed_at,
                note.turn_index, source_timestamp,
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
        _embedding: &[f32],
        _top_k: usize,
        _exclude_ids: &[&str],
    ) -> Result<Vec<(MemoryNote, f32)>> {
        Ok(Vec::new())
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryNote>> {
        let conn = self.conn.lock().map_err(|e| KartaError::VectorStore(e.to_string()))?;
        let result = conn
            .prepare_cached(
                "SELECT id, content, context, keywords_json, tags_json, provenance_json,
                        confidence, created_at, updated_at, status_json, last_accessed_at,
                        turn_index, source_timestamp, embedding
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
                    turn_index, source_timestamp, embedding
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
                    turn_index, source_timestamp, embedding
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
}
