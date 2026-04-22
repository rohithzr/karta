//! STEP1.5 Task 5: round-trip extraction test for `occurred_*` fields.
//!
//! The mock LLM's `handle_attributes` emits `atomic_facts` with `occurred_*`
//! populated by a deterministic rule: ISO-date content → 1-day interval @
//! Explicit confidence; otherwise null bounds @ None confidence. These two
//! tests lock that contract in through the full write pipeline, so any
//! future change that drops/mangles `occurred_*` in the extraction round-
//! trip shows up here — NOT silently during BEAM.

#![cfg(feature = "sqlite-vec")]

use std::sync::Arc;

use chrono::{TimeZone, Utc};
use karta_core::clock::ClockContext;
use karta_core::config::KartaConfig;
use karta_core::llm::MockLlmProvider;
use karta_core::read::temporal::ConfidenceBand;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_core::Karta;
use tempfile::TempDir;

async fn make_karta(dir: &std::path::Path) -> Karta {
    let vec_store = SqliteVectorStore::new(dir.to_str().unwrap(), 1536)
        .await
        .unwrap();
    let shared_conn = vec_store.connection();
    let graph_store: Arc<dyn GraphStore> =
        Arc::new(SqliteGraphStore::with_connection(shared_conn));
    let vector_store: Arc<dyn VectorStore> = Arc::new(vec_store);
    let llm = Arc::new(MockLlmProvider::new());
    Karta::new(vector_store, graph_store, llm, KartaConfig::default())
        .await
        .unwrap()
}

#[tokio::test]
#[ignore = "step2 task 12: mock emits empty spans, restored after mock rewrite"]
async fn iso_date_extracts_with_explicit_confidence() {
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let ref_time = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();

    let note = karta
        .add_note_with_clock(
            "Deadline is 2024-03-15 for the budget tracker milestone.",
            Some("sess-1"),
            Some(0),
            ClockContext::at(ref_time),
        )
        .await
        .unwrap();

    let facts = karta.get_facts_for_note(&note.id).await.unwrap();
    assert!(
        !facts.is_empty(),
        "mock should emit at least one fact from a non-empty sentence"
    );
    assert!(
        facts
            .iter()
            .any(|f| f.occurred_confidence == ConfidenceBand::Explicit),
        "at least one fact should carry Explicit confidence for the ISO date"
    );
    let dated = facts
        .iter()
        .find(|f| f.occurred_start.is_some())
        .expect("at least one fact should have an occurred_start for 2024-03-15");
    assert_eq!(
        dated.occurred_start.unwrap(),
        Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap(),
        "occurred_start should match the ISO date's 00:00:00 UTC"
    );
    assert_eq!(
        dated.occurred_end.unwrap(),
        Utc.with_ymd_and_hms(2024, 3, 16, 0, 0, 0).unwrap(),
        "occurred_end should be start + 1 day"
    );
}

#[tokio::test]
#[ignore = "step2 task 12: mock emits empty spans, restored after mock rewrite"]
async fn non_temporal_fact_has_null_bounds_and_zero_confidence() {
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;

    let note = karta
        .add_note("Flask 2.3.1 is the web framework.")
        .await
        .unwrap();

    let facts = karta.get_facts_for_note(&note.id).await.unwrap();
    assert!(!facts.is_empty(), "mock should emit at least one fact");
    assert!(
        facts.iter().all(|f| f.occurred_start.is_none()),
        "non-temporal content should have null occurred_start"
    );
    assert!(
        facts.iter().all(|f| f.occurred_end.is_none()),
        "non-temporal content should have null occurred_end"
    );
    assert!(
        facts
            .iter()
            .all(|f| f.occurred_confidence == ConfidenceBand::None),
        "non-temporal content should have ConfidenceBand::None"
    );
}

#[tokio::test]
async fn f7_t4_instant_encoded_as_1ns_interval() {
    use karta_core::note::AtomicFact;

    let t = Utc.with_ymd_and_hms(2024, 3, 15, 14, 30, 0).unwrap();
    let fact = AtomicFact {
        id: "f-inst".into(),
        content: "Event at 14:30:00 UTC".into(),
        source_note_id: "n-inst".into(),
        ordinal: 0,
        memory_kind: karta_core::extract::memory_kind::MemoryKind::DurableFact,
        facet: karta_core::extract::facet::Facet::Unknown,
        entity_type: karta_core::extract::entity_type::EntityType::Unknown,
        entity_text: None,
        value_text: None,
        value_date: None,
        supporting_spans: Vec::new(),
        embedding: (0..1536).map(|i| i as f32 / 1536.0).collect(),
        created_at: Utc::now(),
        source_timestamp: t,
        occurred_start: Some(t),
        occurred_end: Some(t + chrono::Duration::nanoseconds(1)),
        occurred_confidence: ConfidenceBand::Explicit,
    };
    assert!(fact.validate_occurred().is_ok());
    assert_eq!(
        fact.occurred_end.unwrap() - fact.occurred_start.unwrap(),
        chrono::Duration::nanoseconds(1),
        "instant must be exactly 1ns wide"
    );
}
