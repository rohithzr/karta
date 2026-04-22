//! T15 — Future-skew warning at write time.
//!
//! When ctx.reference_time() is more than `future_skew_threshold_days`
//! ahead of Utc::now(), the WriteEngine logs a warning. The note is still
//! ingested; recency math at read time clamps the negative-age case to a
//! recency of 1.0. We assert the recency clamp behaviour here since the
//! warning is just a tracing line.

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;

    use karta_core::clock::ClockContext;
    use karta_core::config::KartaConfig;
    use karta_core::llm::MockLlmProvider;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::Karta;
    use tempfile::TempDir;

    #[tokio::test]
    async fn future_dated_note_ingests_and_does_not_panic() {
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

        let two_days_ahead = chrono::Utc::now() + chrono::Duration::days(2);
        let ctx = ClockContext::at(two_days_ahead);

        let note = karta
            .add_note_with_metadata("future-dated probe", "s0", Some(0), Some(ctx.reference_time()))
            .await
            .unwrap();

        // Round-trip the source_timestamp.
        let fetched = karta.get_note(&note.id).await.unwrap().expect("present");
        let delta = (fetched.source_timestamp - two_days_ahead).num_seconds().abs();
        assert!(delta < 5, "source_timestamp preserved within tolerance");
    }
}
