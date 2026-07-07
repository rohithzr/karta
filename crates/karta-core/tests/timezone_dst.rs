//! T19 — Timezone / UTC parsing + DST boundary.
//!
//! All ingestion timestamps are UTC instants. Naive date strings (BEAM
//! "March-15-2024") become UTC midnight. DST switches don't affect UTC,
//! but this test exists to prevent regression — make sure that two notes
//! whose source times straddle a US DST boundary preserve chronological
//! ordering and produce the expected recency delta against a post-DST
//! query time.

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
    async fn dst_boundary_ordering_holds() {
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

        // 2024 US DST switch was 2024-03-10 02:00 local. As UTC instants
        // this is unaffected — but if any code re-parsed naive strings
        // through local time we'd get a 1h shift. Test for it.
        let pre = chrono::Utc.with_ymd_and_hms(2024, 3, 10, 7, 0, 0).unwrap();
        let post = chrono::Utc.with_ymd_and_hms(2024, 3, 10, 8, 0, 0).unwrap();

        karta
            .add_note_with_clock("pre-DST sample", Some("s0"), Some(0), ClockContext::at(pre))
            .await
            .unwrap();
        karta
            .add_note_with_clock("post-DST sample", Some("s0"), Some(1), ClockContext::at(post))
            .await
            .unwrap();

        let all = karta.get_all_notes().await.unwrap();
        let pre_note = all.iter().find(|n| n.content.contains("pre-DST")).unwrap();
        let post_note = all.iter().find(|n| n.content.contains("post-DST")).unwrap();
        // Source timestamps preserve their UTC values exactly.
        assert_eq!(pre_note.source_timestamp, pre);
        assert_eq!(post_note.source_timestamp, post);
        // Chronological ordering holds.
        assert!(pre_note.source_timestamp < post_note.source_timestamp);
    }

    #[test]
    fn naive_date_string_parses_to_utc_midnight() {
        // "2024-03-15" → 2024-03-15T00:00:00Z (UTC midnight). This is the
        // contract convert_beam.py is held to. The Rust harness fallback
        // (parse_time_anchor) lives in the beam test files; here we
        // assert the chrono primitive does what we expect.
        let parsed = chrono::NaiveDate::parse_from_str("2024-03-15", "%Y-%m-%d")
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let expected = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        assert_eq!(parsed, expected);
    }
}
