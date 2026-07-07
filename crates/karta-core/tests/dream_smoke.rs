//! REG-2 — Live dream cycle still works end-to-end.

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;

    use karta_core::config::KartaConfig;
    use karta_core::llm::MockLlmProvider;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::Karta;
    use tempfile::TempDir;

    #[tokio::test]
    async fn live_dream_cycle_runs_without_panic() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let vec_store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        let shared_conn = vec_store.connection();
        let graph_store = SqliteGraphStore::with_connection(shared_conn);
        let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
        let graph_store = Arc::new(graph_store) as Arc<dyn GraphStore>;
        let llm = Arc::new(MockLlmProvider::new());
        let karta = Karta::new(vector_store, graph_store, llm, KartaConfig::default())
            .await
            .unwrap();

        karta.add_note("Sarah prefers Slack notifications").await.unwrap();
        karta.add_note("Sarah's team uses Slack channels for project work").await.unwrap();
        karta
            .add_note("Quarterly budget review scheduled next Friday")
            .await
            .unwrap();

        // Live default: no ctx provided.
        let run = karta.run_dreaming("test", "test-scope").await.unwrap();
        assert_eq!(run.notes_inspected, 3, "all 3 notes are eligible");
    }
}
