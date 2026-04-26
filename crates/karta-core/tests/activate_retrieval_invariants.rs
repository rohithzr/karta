//! Retrieval-invariant tests for the ACTIVATE pipeline, Existence mode in
//! particular.
//!
//! Targets the Hebbian-suppression regression diagnosed in BEAM Conv 1 Q4
//! (see commits `229818c` and `f5f4e81`): in the unfixed state, the
//! Existence-mode channel weights let Hebbian co-activation pull a cluster
//! of "consistent" notes to the top and shove a single contradicting
//! outlier out of the returned pool. The fix is to drop `hebbian` to 0.0
//! and raise `facts`/`integration` for Existence mode.
//!
//! These tests are deterministic (no OpenAI calls, no LLM judging, no
//! randomness) via `MockLlmProvider`, whose embedding function is a
//! word-hash projection so texts that share words have higher cosine
//! similarity.
//!
//! ## Discrimination caveat
//!
//! Test 1 asserts two things that *do* discriminate under MockLlmProvider:
//! (1) an Existence-style query classifies into `QueryMode::Existence` (so
//! the Existence channel-weight row is selected at all — catches prototype
//! and keyword-fallback regressions), and (2) end-to-end retrieval still
//! surfaces the contradicting outlier in top-K at a modest flood-cluster
//! size.
//!
//! It does *not* reliably discriminate the `hebbian` weight value on its
//! own — MockLlmProvider's linking produces a sparse Hebbian graph and
//! extracts zero atomic facts, so the `hebbian`/`facts`/`integration`
//! channels contribute little RRF mass under this fixture. A
//! full-scale discrimination test requires real embeddings + real LLM
//! linking, which is what BEAM Conv 1 Q4 provides. See the report that
//! accompanied this commit for the empirical numbers.
//!
//! Run: `cargo test -p karta-core --test activate_retrieval_invariants`

use std::sync::Arc;

use karta_core::Karta;
use karta_core::config::KartaConfig;
use karta_core::llm::{LlmProvider, MockLlmProvider};
use karta_core::store::lance::LanceVectorStore;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::{GraphStore, VectorStore};

/// Build a Karta instance backed by real LanceDB + SQLite, wired to
/// `MockLlmProvider`. Each invocation gets its own scratch directory under
/// `/tmp` so tests are isolated and can run in parallel.
///
/// Mirrors the fixture patterns in `tests/eval.rs` and `tests/bench_beam.rs`.
async fn make_karta(enable_activate: bool, label: &str) -> Karta {
    // Unique suffix per call so parallel tests (and repeated runs) don't
    // collide on the same LanceDB / SQLite files.
    let suffix = uuid::Uuid::new_v4().to_string();
    let data_dir = format!("/tmp/karta-invariant-{}-{}", label, &suffix[..8]);
    let _ = std::fs::remove_dir_all(&data_dir);

    let vector_store =
        Arc::new(LanceVectorStore::new(&data_dir).await.unwrap()) as Arc<dyn VectorStore>;
    let graph_store = Arc::new(SqliteGraphStore::new(&data_dir).unwrap()) as Arc<dyn GraphStore>;
    let llm = Arc::new(MockLlmProvider::new()) as Arc<dyn LlmProvider>;

    let mut config = KartaConfig::default();
    config.read.activate.enabled = enable_activate;

    Karta::new(vector_store, graph_store, llm, config)
        .await
        .unwrap()
}

/// **Core invariant.** An Existence-classified query that asks whether the
/// user has *ever* said something must surface a contradicting outlier note
/// even when the ANN pool is dominated by a large consistent cluster.
///
/// If the Existence channel weights regress (e.g. `hebbian` goes back above
/// 0.0), the flood cluster's Hebbian co-activation boosts push the
/// contradicting note out of the top-K.
#[tokio::test]
async fn existence_mode_surfaces_contradicting_note() {
    let karta = make_karta(true, "existence-contradicting").await;

    // 1 contradicting outlier note. Phrased to share multiple meaningful
    // words with the query ("have", "never", "integrated", "flask-login") so
    // the ANN channel can place it within the anchor pool.
    karta
        .add_note(
            "I have never integrated Flask-Login in this project — \
             rolled my own session system instead.",
        )
        .await
        .unwrap();

    // 10 consistent notes that claim the opposite. We vary the wording per
    // note so ANN sees distinct embeddings while the shared "Flask-Login
    // integration" token still forms a Hebbian cluster (MockLlmProvider's
    // linking threshold is 2+ shared words >4 chars, which this wording
    // satisfies). Empirically under MockLlmProvider this size keeps the
    // contradicting outlier reliably in top-10 when the pipeline is
    // healthy; much larger pools saturate the RRF fusion and the outlier
    // drops out regardless of channel weights (see module-level
    // discrimination caveat).
    let variations = [
        "configured login_manager and user loader",
        "wired up session cookies with remember_me",
        "added UserMixin to the User model",
        "debugging login_required decorator behavior",
        "extended UserMixin with role flags",
        "serialized user ids through flask_login",
    ];
    for i in 0..10 {
        let variation = variations[i % variations.len()];
        let note = format!(
            "Day {}: working on Flask-Login integration today — {}.",
            i, variation
        );
        karta.add_note(&note).await.unwrap();
    }

    // Existence-style question. Wording chosen to overlap with the
    // Existence prototype set ("Have I been inconsistent about my
    // preferences?", "Is it true that I contradicted myself about the
    // tool?") and the keyword-fallback trigger ("contradict"), so the
    // classifier routes it into Existence mode whether it is running on
    // the embedding centroid path or the keyword fallback.
    let query = "Have I contradicted myself about Flask-Login integration? \
                 Did I ever say I have never integrated it?";

    // --- Sub-assertion A: the query classifies as Existence mode. ---
    // ask() exposes the classified mode in its result, which is the only
    // externally-observable channel selector. Catches regressions in the
    // prototype set (like 229818c → f5f4e81) or the keyword-fallback list.
    let ask_result = karta.ask(query, 10).await.unwrap();
    assert_eq!(
        ask_result.query_mode, "Existence",
        "An Existence-style query must classify into `QueryMode::Existence` so \
         the Existence channel-weight row is applied. If this fails, the \
         prototype set or keyword-fallback list regressed. Got: {:?}",
        ask_result.query_mode
    );

    // --- Sub-assertion B: end-to-end retrieval surfaces the contradicting note. ---
    let results = karta.search(query, 10).await.unwrap();

    assert!(
        !results.is_empty(),
        "Existence-mode retrieval returned zero results — fixture broken."
    );

    let contents: Vec<&str> = results.iter().map(|r| r.note.content.as_str()).collect();

    assert!(
        contents.iter().any(|c| c.contains("never integrated")),
        "Existence-mode retrieval must surface the contradicting outlier \
         note when the ANN pool is dominated by a consistent cluster. \
         If this fails, the retrieval pipeline (or the Existence channel \
         weights — hebbian=0.0, facts=1.3, integration=1.0 per \
         `default_activate_channel_weights` in `config.rs`) may have \
         regressed. Top-{} contents: {:#?}",
        contents.len(),
        contents
    );
}

/// **Baseline control.** With ACTIVATE disabled, the legacy scalar scorer
/// still returns *some* results for the same data. Guards against a broken
/// test fixture: if this fails, the failure in Test 1 would not be
/// attributable to the Existence channel-weights invariant.
#[tokio::test]
async fn baseline_standard_mode_without_activate_returns_results() {
    let karta = make_karta(false, "baseline").await;

    karta
        .add_note("I have never integrated Flask-Login in this project.")
        .await
        .unwrap();
    for i in 0..5 {
        let note = format!("Day {}: working on Flask-Login integration today.", i);
        karta.add_note(&note).await.unwrap();
    }

    let results = karta.search("Flask-Login integration", 5).await.unwrap();

    assert!(
        !results.is_empty(),
        "Baseline (ACTIVATE off) retrieval returned zero results — \
         fixture infrastructure is broken, not the ACTIVATE invariant."
    );
}

/// **Embedding-discrimination control.** MockLlmProvider's word-hash
/// embedding must produce enough shared-word similarity for the Flask-Login
/// note to rank above unrelated content. If this test fails, Test 1 cannot
/// be trusted — its premise depends on embedding similarity tracking
/// lexical overlap.
#[tokio::test]
async fn ann_retrieval_finds_keyword_matches() {
    let karta = make_karta(false, "ann-control").await;

    karta
        .add_note("Recipe for homemade pasta carbonara with pancetta and eggs.")
        .await
        .unwrap();
    karta
        .add_note("Observations of Jupiter's moons through my backyard telescope.")
        .await
        .unwrap();
    karta
        .add_note("Gardening tips for tomato plants in clay-heavy soil.")
        .await
        .unwrap();
    karta
        .add_note("Python snakes require precise humidity and temperature control.")
        .await
        .unwrap();
    karta
        .add_note("Bicycle repair: adjusting derailleur and cable tension.")
        .await
        .unwrap();
    karta
        .add_note("Wired up Flask-Login session handling for the auth module.")
        .await
        .unwrap();

    let results = karta.search("Flask-Login session", 3).await.unwrap();

    assert!(
        !results.is_empty(),
        "ANN control returned zero results — fixture broken."
    );

    let top_three: Vec<&str> = results.iter().map(|r| r.note.content.as_str()).collect();

    assert!(
        top_three.iter().any(|c| c.contains("Flask-Login")),
        "MockLlmProvider's word-hash embedding must rank a Flask-Login \
         note above unrelated topics. Top-3 contents: {:#?}",
        top_three
    );
}
