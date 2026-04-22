//! STEP1.5 Task 9 — F7-T11/T12/T13: interval-overlap SQL filter wired
//! into the read path under the tier 1 resolver.
//!
//! The mock LLM extracts `occurred_*` bounds deterministically from
//! ISO-date content, so these tests exercise the end-to-end read path
//! without needing a live model: ingest a dated fact, issue a temporal
//! query, and assert that only in-window facts survive retrieval.

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

/// F7-T11: A fact dated 2024-04-19 falls inside the "last week" window
/// [2024-04-15, 2024-04-22) at ref_time 2024-04-22 and MUST appear in the
/// retrieved context. We assert via `fetch_memories_with_clock` rather than
/// `ask_with_clock` because the mock synthesis LLM doesn't compose grounded
/// answers — what matters here is that the interval-overlap SQL lets the
/// in-window fact through to the retrieval pool.
#[tokio::test]
async fn f7_t11_last_week_matches_overlapping_interval() {
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let ref_time = Utc.with_ymd_and_hms(2024, 4, 22, 10, 0, 0).unwrap();

    karta
        .add_note_with_clock(
            "Closed the billing bug on 2024-04-19",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time - chrono::Duration::days(3)),
        )
        .await
        .unwrap();

    let results = karta
        .fetch_memories_with_clock("what did I close last week", 5, ClockContext::at(ref_time))
        .await
        .unwrap();

    assert!(
        results.context.contains("billing") || results.context.contains("2024-04-19"),
        "in-window fact should be present in retrieval context; got: {}",
        results.context
    );
}

/// F7-T12: A fact dated 2024-04-01 is OUTSIDE the "last week" window
/// [2024-04-15, 2024-04-22) at ref_time 2024-04-22 and MUST NOT appear.
/// This is the core invariant — the SQL filter must actually reject
/// out-of-window facts, not merely re-rank them.
#[tokio::test]
async fn f7_t12_last_week_excludes_out_of_window() {
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let ref_time = Utc.with_ymd_and_hms(2024, 4, 22, 10, 0, 0).unwrap();

    karta
        .add_note_with_clock(
            "Closed the auth refactor on 2024-04-01",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time - chrono::Duration::days(21)),
        )
        .await
        .unwrap();

    let results = karta
        .fetch_memories_with_clock("closed last week", 5, ClockContext::at(ref_time))
        .await
        .unwrap();

    // The fact itself, injected via fact-to-note expansion, would surface as
    // a "[Matched fact: ...]" prefix on the parent note. With the interval
    // filter, no fact overlaps the window, so the parent note must not be
    // surfaced by the fact-expansion path. The flat-ANN path still retrieves
    // the parent note by embedding similarity — so we specifically assert
    // the *fact-matched marker* is absent, which is the load-bearing bit.
    assert!(
        !results.context.contains("[Matched fact:"),
        "no fact should be interval-matched when the only fact is out of window; got: {}",
        results.context
    );
}

/// F7-T13: A fact with null `occurred_*` bounds MUST be excluded from
/// temporal queries. The partial index predicate `occurred_start IS NOT NULL`
/// is exactly the gate that enforces this — a null-bound fact has no proven
/// when-it-happened, so it can't be attested to fall inside any window.
#[tokio::test]
async fn f7_t13_null_bound_facts_excluded_from_temporal_queries() {
    let dir = TempDir::new().unwrap();
    let karta = make_karta(dir.path()).await;
    let ref_time = Utc.with_ymd_and_hms(2024, 4, 22, 10, 0, 0).unwrap();

    karta
        .add_note_with_clock(
            "Flask 2.3.1 is the web framework",
            Some("sess"),
            Some(0),
            ClockContext::at(ref_time),
        )
        .await
        .unwrap();

    let results = karta
        .fetch_memories_with_clock(
            "what happened last week",
            5,
            ClockContext::at(ref_time),
        )
        .await
        .unwrap();

    // Same invariant as T12: the null-bound fact must not drive a fact-match
    // into retrieval. The parent note may still surface via flat ANN, so we
    // assert on the fact-expansion marker — which is the path specifically
    // gated by `find_similar_facts_in_interval`.
    assert!(
        !results.context.contains("[Matched fact:"),
        "null-bound fact should not match any interval; got: {}",
        results.context
    );
}
