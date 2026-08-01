//! Shared handle that bundles a `Karta` instance with the underlying stores.
//!
//! The wrapper's `session.rs` needs direct store access for rule-based
//! consolidation so that it can create promoted fact notes without invoking
//! the LLM. `Karta` does not expose its stores, so `KartaHandle` keeps the
//! same `Arc` references that were passed into `Karta::new` (or reopened for
//! the non-mock path) and makes them available to the session layer.

use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use karta_core::Karta;
use karta_core::store::VectorStore;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use rusqlite::Connection;

/// Apply wrapper-side SQLite pragmas to a shared connection.
///
/// karta_core enables WAL and foreign keys, but it does not set a busy
/// timeout. Every wrapper-side connection to `karta.db` must set
/// `PRAGMA busy_timeout=5000` so concurrent writes do not immediately return
/// `SQLITE_BUSY`.
pub(crate) fn set_busy_timeout(shared_conn: &Arc<Mutex<Connection>>) -> Result<()> {
    let conn = shared_conn
        .lock()
        .map_err(|e| anyhow!("store connection mutex poisoned: {e}"))?;
    conn.execute_batch("PRAGMA busy_timeout = 5000;")
        .context("failed to set busy_timeout on store connection")?;
    Ok(())
}

/// Bundles a `Karta` with the underlying vector/graph stores and the shared
/// SQLite connection so session logic can perform direct, LLM-free upserts.
#[derive(Clone)]
pub struct KartaHandle {
    pub karta: Arc<Karta>,
    pub vector_store: Arc<dyn VectorStore>,
    pub graph_store: Arc<SqliteGraphStore>,
    pub connection: Arc<Mutex<Connection>>,
}

impl KartaHandle {
    /// Build a handle from an existing `Karta`, the stores that were used to
    /// construct it, and the shared connection those stores use.
    pub fn new(
        karta: Arc<Karta>,
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<SqliteGraphStore>,
        connection: Arc<Mutex<Connection>>,
    ) -> Self {
        Self {
            karta,
            vector_store,
            graph_store,
            connection,
        }
    }

    /// Construct a handle using the mock LLM provider and fresh SQLite
    /// stores at the given data directory. This is the canonical test
    /// construction pattern.
    pub async fn open_mock(data_dir: &str) -> Result<Self> {
        Self::open_mock_with_dim(data_dir, 1536).await
    }

    /// Construct a handle using the mock LLM provider and an explicit
    /// embedding dimension. Use this when the store is known to have been
    /// created with a dimension other than 1536.
    pub async fn open_mock_with_dim(data_dir: &str, dim: usize) -> Result<Self> {
        let vector_store = SqliteVectorStore::new(data_dir, dim)
            .await
            .context("failed to open mock vector store")?;
        let shared_conn = vector_store.connection();
        set_busy_timeout(&shared_conn)?;
        let vector_store: Arc<dyn VectorStore> = Arc::new(vector_store);
        let graph_store = Arc::new(SqliteGraphStore::with_connection(shared_conn.clone()));
        let llm: Arc<dyn karta_core::llm::LlmProvider> =
            Arc::new(karta_core::llm::MockLlmProvider::new());
        let config = karta_core::config::KartaConfig::default();
        let karta = Arc::new(
            Karta::new(vector_store.clone(), graph_store.clone(), llm, config)
                .await
                .context("failed to build mock Karta")?,
        );
        Ok(Self::new(karta, vector_store, graph_store, shared_conn))
    }

    /// Open a mock-backed handle for an existing data directory, reading the
    /// embedding dimension from the on-disk `notes_vec` schema. Falls back to
    /// 1536 if the database has not been created yet.
    ///
    /// This is the right entry point for read-only diagnostics like `status`
    /// that must work against real stores created with non-1536-dim models.
    pub async fn open_mock_for_data_dir(data_dir: &str) -> Result<Self> {
        let path = Path::new(data_dir).join("karta.db");
        let dim = if path.exists() {
            read_embedding_dim_from_db(data_dir)
                .context("failed to read embedding dimension from existing store")?
        } else {
            1536
        };
        Self::open_mock_with_dim(data_dir, dim).await
    }

    /// Re-open a `SqliteVectorStore` for a `Karta` that was created via
    /// `Karta::with_defaults`. The embedding dimension is read from the
    /// existing `notes_vec` virtual-table schema so the re-opened store is
    /// compatible with the live store.
    ///
    /// Returns the vector store, graph store, and the shared SQLite connection
    /// so callers can bundle them into a `KartaHandle`.
    pub async fn open_stores_for_data_dir(
        data_dir: &str,
    ) -> Result<(
        Arc<dyn VectorStore>,
        Arc<SqliteGraphStore>,
        Arc<Mutex<Connection>>,
    )> {
        let dim = read_embedding_dim_from_db(data_dir)
            .context("failed to determine embedding dimension from existing store")?;
        let vector_store = SqliteVectorStore::new(data_dir, dim)
            .await
            .context("failed to re-open vector store for session consolidation")?;
        let shared_conn = vector_store.connection();
        set_busy_timeout(&shared_conn)?;
        let vector_store: Arc<dyn VectorStore> = Arc::new(vector_store);
        let graph_store = Arc::new(SqliteGraphStore::with_connection(shared_conn.clone()));
        Ok((vector_store, graph_store, shared_conn))
    }
}

/// Parse the embedding dimension out of the `notes_vec` virtual table schema.
///
/// The schema SQL is expected to contain `embedding float[<dim>]`. If the
/// table does not exist yet or the SQL is unrecognised, this returns an error.
fn read_embedding_dim_from_db(data_dir: &str) -> Result<usize> {
    let path = std::path::Path::new(data_dir).join("karta.db");
    let conn = Connection::open(&path).with_context(|| {
        format!(
            "failed to open {} to read embedding dimension",
            path.display()
        )
    })?;
    conn.execute_batch("PRAGMA busy_timeout = 5000;")
        .context("failed to set busy_timeout on dimension probe connection")?;
    let sql: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notes_vec'",
            [],
            |row| row.get(0),
        )
        .with_context(|| format!("notes_vec table not found in {}", path.display()))?;
    let prefix = "embedding float[";
    let start = sql
        .find(prefix)
        .with_context(|| format!("notes_vec schema missing embedding dimension: {sql}"))?
        + prefix.len();
    let end = sql[start..]
        .find(']')
        .with_context(|| format!("notes_vec schema missing closing bracket: {sql}"))?;
    let dim: usize = sql[start..start + end]
        .parse()
        .with_context(|| format!("invalid embedding dimension in notes_vec schema: {sql}"))?;
    Ok(dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn open_mock_returns_usable_handle() {
        let dir = TempDir::new().unwrap();
        let handle = KartaHandle::open_mock(dir.path().to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(handle.karta.note_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn open_mock_sets_busy_timeout_to_5000() {
        let dir = TempDir::new().unwrap();
        let handle = KartaHandle::open_mock(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let conn = handle.connection.lock().unwrap();
        let timeout: i32 = conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
    }

    #[tokio::test]
    async fn read_embedding_dim_from_existing_store() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        drop(store);

        let dim = read_embedding_dim_from_db(data_dir).unwrap();
        assert_eq!(dim, 1536);
    }

    #[tokio::test]
    async fn set_busy_timeout_helper_sets_5000() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        let conn = store.connection();
        set_busy_timeout(&conn).unwrap();
        let timeout: i32 = conn
            .lock()
            .unwrap()
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
    }

    #[tokio::test]
    async fn open_mock_for_data_dir_reads_non_default_embedding_dim() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        // Create a store with a non-default dimension.
        let store = SqliteVectorStore::new(data_dir, 768).await.unwrap();
        drop(store);

        let handle = KartaHandle::open_mock_for_data_dir(data_dir).await.unwrap();
        assert_eq!(handle.karta.note_count().await.unwrap(), 0);
        // Re-opening with the wrong dimension would corrupt the sqlite-vec
        // virtual table; reaching this point means the dimension was read
        // correctly from the on-disk schema.
    }
}
