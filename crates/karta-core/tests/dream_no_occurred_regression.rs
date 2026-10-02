//! STEP1.5 Task 14: dream prompts unchanged.
//!
//! STEP1.5 explicitly does NOT touch dream-time LLM prompts (per the
//! plan's "What this step does NOT do"). Dream notes are inferences,
//! not facts, and don't carry occurred_* bounds. Any AtomicFact derived
//! from a dream output must still satisfy the validate_occurred()
//! invariants — this regression catches a future drift where a dream
//! prompt sneaks in occurred_* fields without going through the F7
//! schema discipline.

#![cfg(feature = "sqlite-vec")]

use std::sync::Arc;

use karta_core::config::KartaConfig;
use karta_core::llm::MockLlmProvider;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_core::Karta;
use tempfile::TempDir;

#[tokio::test]
async fn dream_notes_facts_still_validate() {
    let dir = TempDir::new().unwrap();
    let vs = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();
    let conn = vs.connection();
    let graph: Arc<dyn GraphStore> = Arc::new(SqliteGraphStore::with_connection(conn));
    let vector_store: Arc<dyn VectorStore> = Arc::new(vs);
    let llm = Arc::new(MockLlmProvider::new());
    let karta = Karta::new(vector_store, graph, llm, KartaConfig::default())
        .await
        .unwrap();

    for i in 0..3 {
        karta
            .add_note(&format!("Flask is used in project {}", i))
            .await
            .unwrap();
    }
    let _ = karta.run_dreaming("workspace", "test").await;

    let all_notes = karta.get_all_notes().await.unwrap();
    for note in &all_notes {
        let facts = karta
            .get_facts_for_note(&note.id)
            .await
            .unwrap_or_default();
        for fact in facts {
            assert!(
                fact.validate_occurred().is_ok(),
                "fact from note {} broke invariants: {:?}",
                note.id,
                fact.validate_occurred()
            );
        }
    }
}
