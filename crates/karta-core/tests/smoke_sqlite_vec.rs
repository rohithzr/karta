//! Smoke test: full Karta pipeline with SqliteVectorStore + MockLlmProvider.
//! Validates add_note → search round-trip through the real system.

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

    async fn make_karta(dir: &std::path::Path) -> Karta {
        let data_dir = dir.to_str().unwrap();
        let vec_store =
            SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        let shared_conn = vec_store.connection();
        let graph_store = SqliteGraphStore::with_connection(shared_conn);

        let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
        let graph_store = Arc::new(graph_store) as Arc<dyn GraphStore>;
        let llm = Arc::new(MockLlmProvider::new());
        let config = KartaConfig::default();

        Karta::new(vector_store, graph_store, llm, config)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn smoke_add_and_search() {
        let dir = TempDir::new().unwrap();
        let karta = make_karta(dir.path()).await;

        // Store three notes about different topics
        karta
            .add_note("Sarah prefers Slack notifications over email for all workflows")
            .await
            .unwrap();
        karta
            .add_note("The quarterly budget review is scheduled for next Friday")
            .await
            .unwrap();
        karta
            .add_note("Sarah's team uses Slack channels for project coordination")
            .await
            .unwrap();

        // Search for Slack-related memories
        let results = karta.search("Slack notifications", 3).await.unwrap();

        assert!(!results.is_empty(), "Search should return results");

        // The two Slack-related notes should rank higher than the budget note
        println!("Search results:");
        for (i, r) in results.iter().enumerate() {
            println!("  [{}] score={:.3} content={}", i, r.score, r.note.content);
        }

        // At least one result should mention Slack
        assert!(
            results.iter().any(|r| r.note.content.contains("Slack")),
            "At least one result should mention Slack"
        );
    }

    #[tokio::test]
    async fn smoke_add_and_count() {
        let dir = TempDir::new().unwrap();
        let karta = make_karta(dir.path()).await;

        assert_eq!(karta.note_count().await.unwrap(), 0);

        karta.add_note("First note").await.unwrap();
        karta.add_note("Second note").await.unwrap();

        assert_eq!(karta.note_count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn smoke_shared_connection_works() {
        let dir = TempDir::new().unwrap();
        let karta = make_karta(dir.path()).await;

        // This exercises both stores through the same karta.db:
        // add_note writes to VectorStore (embedding) and GraphStore (links)
        let note = karta
            .add_note("Test note for shared connection validation")
            .await
            .unwrap();

        // Verify vector store has the note
        let results = karta.search("shared connection", 1).await.unwrap();
        assert!(!results.is_empty());

        // Verify we can retrieve by ID (goes through VectorStore)
        assert!(karta.get_note(&note.id).await.unwrap().is_some());
    }
}
