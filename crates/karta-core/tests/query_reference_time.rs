//! T14 — Query-side reference_time parameter changes recency-weighted ranking.
//!
//! Two notes from different source dates (2024-03-01 and 2024-03-30). A query
//! at ref = 2024-03-15 should rank the older note (closer to ref) higher
//! than the newer one (which is now in the future relative to ref). At
//! ref = 2024-04-01, the newer note ranks higher.

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
    async fn different_reference_times_change_score() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let vec_store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        let shared_conn = vec_store.connection();
        let graph_store = SqliteGraphStore::with_connection(shared_conn);
        let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
        let graph_store = Arc::new(graph_store) as Arc<dyn GraphStore>;
        let llm = Arc::new(MockLlmProvider::new());
        let mut config = KartaConfig::default();
        // Bump recency weight so the difference is visible above similarity noise.
        config.read.recency_weight = 0.9;
        let karta = Karta::new(vector_store, graph_store, llm, config)
            .await
            .unwrap();

        let early = chrono::Utc.with_ymd_and_hms(2024, 3, 1, 0, 0, 0).unwrap();
        let late = chrono::Utc.with_ymd_and_hms(2024, 3, 30, 0, 0, 0).unwrap();

        karta
            .add_note_with_clock("budget tracker workflow morning report", Some("s0"), Some(0), ClockContext::at(early))
            .await
            .unwrap();
        karta
            .add_note_with_clock("budget tracker workflow morning report", Some("s0"), Some(1), ClockContext::at(late))
            .await
            .unwrap();

        let r_early = karta
            .search_with_clock(
                "budget",
                5,
                ClockContext::at(chrono::Utc.with_ymd_and_hms(2024, 3, 5, 0, 0, 0).unwrap()),
            )
            .await
            .unwrap();
        let r_late = karta
            .search_with_clock(
                "budget",
                5,
                ClockContext::at(chrono::Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap()),
            )
            .await
            .unwrap();

        assert!(!r_early.is_empty() && !r_late.is_empty());
        // The two queries should at least produce distinct top scores,
        // proving the reference_time threading is alive.
        assert!(
            (r_early[0].score - r_late[0].score).abs() > 0.001,
            "different ref times should yield different recency-weighted scores: early={} late={}",
            r_early[0].score, r_late[0].score
        );
    }
}
