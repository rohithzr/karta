//! Shared handle that bundles a `Karta` instance with the underlying stores.
//!
//! The wrapper's `session.rs` needs direct store access for rule-based
//! consolidation so that it can create promoted fact notes without invoking
//! the LLM. `Karta` does not expose its stores, so `KartaHandle` keeps the
//! same `Arc` references that were passed into `Karta::new` (or reopened for
//! the non-mock path) and makes them available to the session layer.

use std::sync::Arc;

use anyhow::{Context, Result};
use karta_core::Karta;
use karta_core::store::VectorStore;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use rusqlite::Connection;

/// Bundles a `Karta` with the underlying vector store so session logic can
/// perform direct, LLM-free upserts.
#[derive(Clone)]
pub struct KartaHandle {
    pub karta: Arc<Karta>,
    pub vector_store: Arc<dyn VectorStore>,
    pub graph_store: Arc<SqliteGraphStore>,
}

impl KartaHandle {
    /// Build a handle from an existing `Karta` and the stores that were used
    /// to construct it.
    pub fn new(
        karta: Arc<Karta>,
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<SqliteGraphStore>,
    ) -> Self {
        Self {
            karta,
            vector_store,
            graph_store,
        }
    }

    /// Construct a handle using the mock LLM provider and fresh SQLite
    /// stores at the given data directory. This is the canonical test
    /// construction pattern.
    pub async fn open_mock(data_dir: &str) -> Result<Self> {
        const EMBEDDING_DIM: usize = 1536;
        let vector_store = SqliteVectorStore::new(data_dir, EMBEDDING_DIM)
            .await
            .context("failed to open mock vector store")?;
        let shared_conn = vector_store.connection();
        let vector_store: Arc<dyn VectorStore> = Arc::new(vector_store);
        let graph_store = Arc::new(SqliteGraphStore::with_connection(shared_conn));
        let llm: Arc<dyn karta_core::llm::LlmProvider> =
            Arc::new(karta_core::llm::MockLlmProvider::new());
        let config = karta_core::config::KartaConfig::default();
        let karta = Arc::new(
            Karta::new(vector_store.clone(), graph_store.clone(), llm, config)
                .await
                .context("failed to build mock Karta")?,
        );
        Ok(Self::new(karta, vector_store, graph_store))
    }

    /// Re-open a `SqliteVectorStore` for a `Karta` that was created via
    /// `Karta::with_defaults`. The embedding dimension is read from the
    /// existing `notes_vec` virtual-table schema so the re-opened store is
    /// compatible with the live store.
    pub async fn open_stores_for_data_dir(
        data_dir: &str,
    ) -> Result<(Arc<dyn VectorStore>, Arc<SqliteGraphStore>)> {
        let dim = read_embedding_dim_from_db(data_dir)
            .context("failed to determine embedding dimension from existing store")?;
        let vector_store = SqliteVectorStore::new(data_dir, dim)
            .await
            .context("failed to re-open vector store for session consolidation")?;
        let shared_conn = vector_store.connection();
        let vector_store: Arc<dyn VectorStore> = Arc::new(vector_store);
        let graph_store = Arc::new(SqliteGraphStore::with_connection(shared_conn));
        Ok((vector_store, graph_store))
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
    async fn read_embedding_dim_from_existing_store() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        drop(store);

        let dim = read_embedding_dim_from_db(data_dir).unwrap();
        assert_eq!(dim, 1536);
    }
}
