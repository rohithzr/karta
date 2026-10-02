//! End-to-end check that an attached Decider drives query mode through
//! `Karta`, and that low confidence or a failing decider falls back to the
//! built-in classifier. MockLlmProvider + real sqlite-vec, no network.
#![cfg(feature = "sqlite-vec")]

use std::sync::Arc;

use karta_core::config::KartaConfig;
use karta_core::decide::questions::{QUERY_MODE, QUERY_PREDICATE};
use karta_core::decide::{Decision, Decisions, MockDecider};
use karta_core::llm::{LlmProvider, MockLlmProvider};
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_core::Karta;

const QUERY: &str = "What tools does Alice use for the dashboard?";

async fn make_karta(tag: &str) -> Karta {
    let data_dir = format!("/tmp/karta-test-decider-routing-{}", tag);
    let _ = std::fs::remove_dir_all(&data_dir);
    let vec_store = SqliteVectorStore::new(&data_dir, 1536).await.unwrap();
    let shared_conn = vec_store.connection();
    let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
    let graph_store =
        Arc::new(SqliteGraphStore::with_connection(shared_conn)) as Arc<dyn GraphStore>;
    let llm = Arc::new(MockLlmProvider::new()) as Arc<dyn LlmProvider>;
    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;
    let karta = Karta::new(vector_store, graph_store, llm, config).await.unwrap();
    karta
        .add_note("Alice uses Grafana and Postgres for the dashboard.")
        .await
        .unwrap();
    karta
}

fn answers(mode: &str, confidence: f32) -> Decisions {
    let mut d = Decisions::new();
    d.insert(
        QUERY_MODE.into(),
        Decision::Choice { label: mode.into(), probs: vec![], confidence },
    );
    d.insert(
        QUERY_PREDICATE.into(),
        Decision::Choice { label: "none".into(), probs: vec![], confidence },
    );
    d
}

async fn mode_of(karta: &Karta) -> String {
    karta.fetch_memories(QUERY, 5).await.unwrap().query_mode
}

#[tokio::test]
async fn confident_decider_sets_query_mode() {
    let builtin = mode_of(&make_karta("builtin").await).await;
    assert_ne!(builtin, "Temporal", "pick a decider answer that differs from the built-in mode");

    let mut karta = make_karta("confident").await;
    let decider = Arc::new(MockDecider::new(answers("temporal", 0.95)));
    karta.attach_decider(decider.clone());

    assert_eq!(mode_of(&karta).await, "Temporal");
    assert_eq!(decider.calls().len(), 1);
    assert_eq!(decider.calls()[0].0, QUERY);
}

#[tokio::test]
async fn low_confidence_or_failing_decider_falls_back() {
    let builtin = mode_of(&make_karta("builtin2").await).await;

    let mut low = make_karta("low").await;
    low.attach_decider(Arc::new(MockDecider::new(answers("temporal", 0.5))));
    assert_eq!(mode_of(&low).await, builtin);

    // A decider that cannot answer (validation error) must not fail the query.
    let mut failing = make_karta("failing").await;
    failing.attach_decider(Arc::new(MockDecider::new(Decisions::new())));
    assert_eq!(mode_of(&failing).await, builtin);
}
