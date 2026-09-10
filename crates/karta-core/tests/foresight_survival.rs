//! T4 — Foresight survives a dream cycle on replay.
//!
//! Replay ingest with valid_until = ref + 30d, then run a dream at the
//! same reference_time. The foresight must still be active.

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;

    use chrono::TimeZone;
    use karta_core::clock::ClockContext;
    use karta_core::config::KartaConfig;
    use karta_core::llm::MockLlmProvider;
    use karta_core::note::ForesightSignal;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::Karta;
    use tempfile::TempDir;

    #[tokio::test]
    async fn foresight_survives_replay_dream() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let vec_store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        let shared_conn = vec_store.connection();
        let graph_store_concrete = SqliteGraphStore::with_connection(shared_conn);
        let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
        let graph_store: Arc<dyn GraphStore> = Arc::new(graph_store_concrete);
        let llm = Arc::new(MockLlmProvider::new());
        let karta = Karta::new(
            Arc::clone(&vector_store),
            Arc::clone(&graph_store),
            Arc::clone(&llm) as Arc<_>,
            KartaConfig::default(),
        )
        .await
        .unwrap();

        let ref_time = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();

        let note = karta
            .add_note_with_clock("replay seed", Some("s0"), Some(0), ClockContext::at(ref_time))
            .await
            .unwrap();

        let valid_until = ref_time + chrono::Duration::days(30);
        let fs = ForesightSignal::new("survives".into(), note.id.clone(), Some(valid_until));
        graph_store.upsert_foresight(&fs).await.unwrap();

        karta
            .run_dreaming_with_clock("test", "scope", ClockContext::at(ref_time))
            .await
            .unwrap();

        let active = graph_store.get_active_foresights().await.unwrap();
        assert!(
            active.iter().any(|f| f.id == fs.id),
            "in-window foresight must survive replay-anchored dream"
        );
    }
}
