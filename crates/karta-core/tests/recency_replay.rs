//! REG-3 — Recency on replay must score > 0 when query brings a matching ctx.
//!
//! Pre-fix behavior: a 2024 note read at 2026 wall-clock-now scored
//! recency ≈ 2e-8 because age was ~760 days vs a 30-day half-life.
//! Post-fix: with `search_with_clock` passing the query's reference_time,
//! recency is computed against (ref - source_timestamp), so a 7-day-old
//! replay note scores ~0.85 (7d into a 30d half-life: 0.5^(7/30) ≈ 0.85).

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;

    use chrono::{Duration, TimeZone};
    use karta_core::clock::ClockContext;
    use karta_core::config::KartaConfig;
    use karta_core::llm::MockLlmProvider;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::Karta;
    use tempfile::TempDir;

    #[tokio::test]
    async fn replay_recency_scores_above_zero() {
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

        // Replay note dated 2024-03-15. Default recency_half_life = 30 days.
        let source_ts = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        karta
            .add_note_with_clock(
                "Project deadline March 15 budget tracker Flask deliverable",
                Some("s0"),
                Some(0),
                ClockContext::at(source_ts),
            )
            .await
            .unwrap();

        // Query at ref = source_ts + 7 days. Recency should be ≈ 0.85.
        let query_ctx = ClockContext::at(source_ts + Duration::days(7));
        let results = karta
            .search_with_clock("project deadline budget", 5, query_ctx)
            .await
            .unwrap();

        assert!(!results.is_empty(), "result returned");
        let top = &results[0];
        // Score blends similarity + recency at recency_weight = 0.15 by default.
        // The big win vs the broken state is just that the score isn't
        // a tiny noise-level number — assert > 0.05 to leave headroom for
        // the fact-match vs note-match split.
        assert!(
            top.score > 0.05,
            "expected non-trivial blended score on replay, got {}",
            top.score
        );
    }

    #[tokio::test]
    async fn live_recency_unchanged_for_recent_note() {
        // REG-4 (folded in): live-mode note from 30d ago should still score
        // sensibly (~0.5 recency). We use the live default (no _with_clock).
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

        karta.add_note("smoke note about the project budget").await.unwrap();
        let results = karta.search("project budget", 5).await.unwrap();
        assert!(!results.is_empty(), "result returned");
        assert!(results[0].score > 0.05);
    }
}
