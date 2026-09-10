//! T13 — Live E2E. Every entry point works without an explicit ClockContext.

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
    async fn live_write_dream_search_ask_fetch_no_ctx() {
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

        // Write — no ctx.
        karta.add_note("Sarah's preferred Slack workflow notification").await.unwrap();
        karta.add_note("Sarah uses Slack channels for project workflow").await.unwrap();
        karta.add_note("Quarterly review scheduled for Friday").await.unwrap();

        // Dream — no ctx.
        let _ = karta.run_dreaming("test", "scope").await.unwrap();

        // Search — no ctx.
        let s = karta.search("Slack workflow", 3).await.unwrap();
        assert!(!s.is_empty());

        // Ask — no ctx.
        let _ = karta.ask("What does Sarah prefer", 3).await.unwrap();

        // fetch_memories — no ctx.
        let f = karta.fetch_memories("Slack workflow", 3).await.unwrap();
        assert!(!f.notes.is_empty());
    }
}
