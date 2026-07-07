//! Stress tests for the failure modes called out in the codex review
//! (`docs/reviews/2026-04-22-codex-fact-extraction-review.md`).
//!
//! FM1: pure requests / greetings → zero facts
//! FM2: temporal-role distinction (deadline → value_date, not occurred_*)
//! FM7: jargon leakage (e.g. "time anchor")
//! FM8: over-aggregation (multi-clause → ≥2 facts)
//! FM9: under-typing for ANN (facets must be typed)
//!
//! Plus the A1 ordering regression test using ScriptedMockLlmProvider —
//! this is the load-bearing one. If anyone reverts the pre-admission
//! filter from Task 9.6, this test fails fast with a precise diagnostic.

#![cfg(feature = "sqlite-vec")]

use std::sync::Arc;

use chrono::{TimeZone, Utc};
use karta_core::clock::ClockContext;
use karta_core::config::KartaConfig;
use karta_core::llm::MockLlmProvider;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_core::Karta;
use tempfile::TempDir;

// chrono builders aren't const fn, so use a helper instead of `const REF`.
fn ref_time() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap()
}

async fn make_karta(dir: &std::path::Path) -> Karta {
    let vec_store = SqliteVectorStore::new(dir.to_str().unwrap(), 1536)
        .await
        .unwrap();
    let conn = vec_store.connection();
    let graph: Arc<dyn GraphStore> = Arc::new(SqliteGraphStore::with_connection(conn));
    let vs: Arc<dyn VectorStore> = Arc::new(vec_store);
    let llm = Arc::new(MockLlmProvider::new());
    Karta::new(vs, graph, llm, KartaConfig::default())
        .await
        .unwrap()
}

async fn ingest_and_count(content: &str) -> usize {
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let n = karta
        .add_note_with_clock(content, Some("sess"), Some(0), ClockContext::at(ref_time()))
        .await
        .unwrap();
    karta.get_facts_for_note(&n.id).await.unwrap().len()
}

#[tokio::test]
async fn fm1_no_zero_fact_pollution_greeting() {
    assert_eq!(ingest_and_count("thanks!").await, 0);
}

#[tokio::test]
async fn fm1_no_zero_fact_pollution_question() {
    assert_eq!(ingest_and_count("Can you help me build a schedule?").await, 0);
}

#[tokio::test]
async fn fm2_temporal_role_deadline_lands_in_value_date() {
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let n = karta
        .add_note_with_clock(
            "I have an April 15 deadline for the budget tracker.",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time()),
        )
        .await
        .unwrap();
    let facts = karta.get_facts_for_note(&n.id).await.unwrap();
    assert!(!facts.is_empty(), "expected a deadline fact");
    let f = facts
        .iter()
        .find(|f| matches!(f.facet, karta_core::extract::facet::Facet::Deadline))
        .unwrap();
    assert!(
        f.value_date.is_some(),
        "deadline should populate value_date, not occurred_start"
    );
    assert!(
        f.occurred_start.is_none(),
        "deadline is not an event — occurred_* should be null"
    );
}

#[tokio::test]
async fn fm7_jargon_leakage_time_anchor() {
    // "Time Anchor" is benchmark jargon; it should not appear in any stored fact.
    // The heuristic mock won't synthesize it, so this test passes by virtue
    // of the mock — but it documents the behavior we expect from the real
    // LLM under the new prompt.
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let n = karta
        .add_note_with_clock(
            "I'm working on a project with a Time Anchor of March 15, 2024.",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time()),
        )
        .await
        .unwrap();
    let facts = karta.get_facts_for_note(&n.id).await.unwrap();
    for f in &facts {
        let lower = f.content.to_lowercase();
        assert!(
            !lower.contains("time anchor"),
            "fact leaks jargon: {}",
            f.content
        );
    }
}

#[tokio::test]
async fn fm8_over_aggregation_multi_clause_yields_multi_facts() {
    // Two clauses → expect ≥2 facts (one per clause's clean claim).
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let n = karta
        .add_note_with_clock(
            "Project uses Flask 2.3.1 and Python 3.11.",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time()),
        )
        .await
        .unwrap();
    let facts = karta.get_facts_for_note(&n.id).await.unwrap();
    assert!(
        facts.len() >= 2,
        "expected ≥2 facts, got {}",
        facts.len()
    );
}

#[tokio::test]
async fn fm9_under_typing_for_ann_facts_are_typed() {
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let n = karta
        .add_note_with_clock(
            "Project uses Flask 2.3.1.",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time()),
        )
        .await
        .unwrap();
    let facts = karta.get_facts_for_note(&n.id).await.unwrap();
    assert!(!facts.is_empty());
    for f in &facts {
        assert!(
            !f.entity_type.is_generic(),
            "entity_type should not be Unknown"
        );
        assert!(!f.facet.is_generic(), "facet should not be Unknown");
    }
}

// ============================================================
// A1 ordering regression test — uses ScriptedMockLlmProvider
// to construct an adversarial LLM response.
// ============================================================

use karta_core::error::Result as KartaResult;
use karta_core::llm::{
    ChatMessage, ChatResponse, GenConfig, LlmProvider, ScriptedMockLlmProvider,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct EmbedCountingLlm {
    inner: Arc<ScriptedMockLlmProvider>,
    embed_calls: Arc<AtomicUsize>,
    embed_batch_total_items: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl LlmProvider for EmbedCountingLlm {
    async fn chat(&self, m: &[ChatMessage], c: &GenConfig) -> KartaResult<ChatResponse> {
        self.inner.chat(m, c).await
    }
    async fn embed(&self, t: &[&str]) -> KartaResult<Vec<Vec<f32>>> {
        self.embed_calls.fetch_add(1, Ordering::SeqCst);
        self.embed_batch_total_items
            .fetch_add(t.len(), Ordering::SeqCst);
        self.inner.embed(t).await
    }
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn embedding_model_id(&self) -> &str {
        self.inner.embedding_model_id()
    }
}

const ADVERSARIAL_RESPONSE: &str = r#"{
    "reasoning": "test",
    "context": "test",
    "keywords": [],
    "tags": [],
    "foresight_signals": [],
    "atomic_facts": [
        {
            "content": "User wants help with the April 15 deadline.",
            "memory_kind": "ephemeral_request",
            "supporting_spans": ["help with the April 15 deadline"],
            "facet": "deadline",
            "entity_type": "project",
            "entity_text": "project",
            "value_text": null,
            "value_date": "2024-04-15T00:00:00Z",
            "occurred_start": null,
            "occurred_end": null,
            "occurred_confidence": 0.0
        },
        {
            "content": "Project has an April 15 deadline.",
            "memory_kind": "future_commitment",
            "supporting_spans": ["April 15 deadline"],
            "facet": "deadline",
            "entity_type": "project",
            "entity_text": "project",
            "value_text": null,
            "value_date": "2024-04-15T00:00:00Z",
            "occurred_start": null,
            "occurred_end": null,
            "occurred_confidence": 0.0
        }
    ]
}"#;

#[tokio::test]
async fn ephemeral_collision_does_not_steal_durable_slot() {
    let dir = TempDir::new().unwrap();
    let scripted = Arc::new(ScriptedMockLlmProvider::new(vec![(
        "ADVERSARIAL_TURN",
        ADVERSARIAL_RESPONSE,
    )]));
    let embed_calls = Arc::new(AtomicUsize::new(0));
    let embed_total = Arc::new(AtomicUsize::new(0));
    let counting = Arc::new(EmbedCountingLlm {
        inner: scripted,
        embed_calls: embed_calls.clone(),
        embed_batch_total_items: embed_total.clone(),
    });

    let vec_store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();
    let conn = vec_store.connection();
    let graph: Arc<dyn GraphStore> = Arc::new(SqliteGraphStore::with_connection(conn));
    let vs: Arc<dyn VectorStore> = Arc::new(vec_store);
    let karta = Karta::new(vs, graph, counting, KartaConfig::default())
        .await
        .unwrap();

    let n = karta
        .add_note_with_clock(
            "ADVERSARIAL_TURN: April 15 deadline coming up.",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time()),
        )
        .await
        .unwrap();

    // ASSERTION 1: durable fact survived the adversarial collision.
    // If dedup ran first, the ephemeral_request (emitted first) would
    // have claimed the slot and the durable would have been dropped.
    let facts = karta.get_facts_for_note(&n.id).await.unwrap();
    assert_eq!(
        facts.len(),
        1,
        "expected exactly one stored fact (durable survived collision); got {}",
        facts.len()
    );
    assert_eq!(
        facts[0].memory_kind,
        karta_core::extract::memory_kind::MemoryKind::FutureCommitment,
        "wrong fact survived — ordering bug regressed",
    );

    // ASSERTION 2: total embed items stayed at the no-bug baseline.
    // Working baseline = 3: embed_raw (1) + embed_enriched (1) + embed_facts
    // with just the durable (1). Episodes are disabled by default so no
    // narrative embed fires. Anything ≥4 means the dedup+admission pipeline
    // pushed an extra fact through embed_facts (e.g. pre-filter regressed
    // and the ephemeral got admitted as a separate row, or dedup stopped
    // collapsing siblings). Note: in this specific scenario dedup collapses
    // ephemeral+durable to a single row even with the pre-filter removed,
    // so ASSERTION 1 above is the load-bearing check; this is a coarse
    // perimeter on extra fact-embed traffic.
    let total_items = embed_total.load(Ordering::SeqCst);
    assert!(
        total_items <= 3,
        "embed batch contained {total_items} items; expected ≤3 (raw + enriched + 1 fact). \
         If 4+, the fact-level embed batch grew unexpectedly — dedup or pre-admission filter regressed.",
    );
}

// ============================================================
// FM1 admission-gate scripted test — proves the gate fires
// even when the LLM emits a fully-grounded ephemeral fact
// (the heuristic mock can't construct this scenario alone).
// ============================================================

const SCRIPTED_PURE_REQUEST: &str = r#"{
    "reasoning": "test",
    "context": "test",
    "keywords": [],
    "tags": [],
    "foresight_signals": [],
    "atomic_facts": [{
        "content": "User wants help.",
        "memory_kind": "ephemeral_request",
        "supporting_spans": ["help me"],
        "facet": "unknown",
        "entity_type": "user",
        "entity_text": "user",
        "value_text": null,
        "value_date": null,
        "occurred_start": null,
        "occurred_end": null,
        "occurred_confidence": 0.0
    }]
}"#;

#[tokio::test]
async fn fm1_admission_drops_scripted_ephemeral() {
    let dir = TempDir::new().unwrap();
    let scripted: Arc<dyn LlmProvider> = Arc::new(ScriptedMockLlmProvider::new(vec![(
        "FM1_SCRIPTED",
        SCRIPTED_PURE_REQUEST,
    )]));
    let vec_store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();
    let conn = vec_store.connection();
    let graph: Arc<dyn GraphStore> = Arc::new(SqliteGraphStore::with_connection(conn));
    let vs: Arc<dyn VectorStore> = Arc::new(vec_store);
    let karta = Karta::new(vs, graph, scripted, KartaConfig::default())
        .await
        .unwrap();

    let n = karta
        .add_note_with_clock(
            "FM1_SCRIPTED please help me",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time()),
        )
        .await
        .unwrap();
    let facts = karta.get_facts_for_note(&n.id).await.unwrap();
    assert_eq!(
        facts.len(),
        0,
        "admission gate should have dropped the scripted ephemeral fact",
    );
}
