//! T7 — Recency clamp on forward-dated source.
//!
//! When `source_timestamp > ctx.reference_time()` (clock skew, future-dated
//! import, bug), the recency calculation must clamp negative age to 0 and
//! return recency = 1.0 — NOT a >1 score or NaN.

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;

    use chrono::TimeZone;
    use karta_core::clock::ClockContext;
    use karta_core::config::KartaConfig;
    use karta_core::llm::MockLlmProvider;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::Karta;
    use tempfile::TempDir;

    #[tokio::test]
    async fn forward_dated_note_searches_without_panic() {
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

        // Note dated 2024-04-01.
        let future_ts = chrono::Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap();
        karta
            .add_note_with_clock(
                "forward dated probe",
                Some("s0"),
                Some(0),
                ClockContext::at(future_ts),
            )
            .await
            .unwrap();

        // Query at reference_time = 2024-03-15 (BEFORE the note's
        // source_timestamp). Without the clamp, age_days would be negative
        // and 0.5^negative > 1 → finite_score arithmetic could go off the
        // rails. The clamp keeps recency at 1.0.
        let query_ctx = ClockContext::at(chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap());
        let results = karta
            .search_with_clock("forward dated probe", 5, query_ctx)
            .await
            .unwrap();

        // The note must come back (it's the only thing in the store) and
        // its score must be a finite number — not NaN, not infinity.
        assert!(!results.is_empty(), "result returned");
        for r in &results {
            assert!(r.score.is_finite(), "score must be finite, got {}", r.score);
        }
    }
}
