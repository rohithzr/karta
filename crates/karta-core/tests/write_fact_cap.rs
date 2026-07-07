#![cfg(feature = "sqlite-vec")]
//! The fact-store loop must honor config.write.max_facts_per_note, not a
//! hardcoded 5. Uses MockLlmProvider so it is offline + deterministic.

use std::sync::Arc;
use karta_core::config::KartaConfig;
use karta_core::llm::MockLlmProvider;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_core::Karta;
use tempfile::TempDir;

#[tokio::test]
async fn respects_max_facts_per_note_above_five() {
    let dir = TempDir::new().unwrap();
    let data_dir = dir.path().to_str().unwrap();
    let vec_store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
    let conn = vec_store.connection();
    let graph = SqliteGraphStore::with_connection(conn);
    let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
    let graph_store = Arc::new(graph) as Arc<dyn GraphStore>;

    let mut config = KartaConfig::default();
    config.write.max_facts_per_note = 8; // > 5

    // MockLlmProvider is configured to emit 8 durable atomic facts (see Step 3).
    let llm = Arc::new(MockLlmProvider::with_fact_count(8));
    let karta = Karta::new(vector_store.clone(), graph_store, llm, config)
        .await
        .unwrap();

    let note = karta.add_note("eight distinct durable facts here").await.unwrap();
    let facts = vector_store.get_facts_for_note(&note.id).await.unwrap();
    assert_eq!(facts.len(), 8, "expected all 8 facts stored, got {}", facts.len());
}
