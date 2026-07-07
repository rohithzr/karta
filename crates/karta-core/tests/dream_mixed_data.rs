//! T5 — Mixed live + replay data dreamed without partition.
//!
//! Karta works on a mixed-mode dataset (live + replay-imported notes
//! coexisting) because each note carries its own source_timestamp and
//! each dream brings its own ctx.reference_time. Specifically:
//!  - dream_with_clock(now()) at wall-clock-now expires foresights whose
//!    valid_until < now (the 2024 ones go).
//!  - dream_with_clock(at(2024-03-15)) leaves the 2024 foresights alone.

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
    async fn dream_with_replay_clock_preserves_in_window_foresights() {
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

        // Add one live note + one replay note.
        karta.add_note("live note about workflow").await.unwrap();
        let replay_ts = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        let replay_note = karta
            .add_note_with_clock(
                "replay note about workflow",
                Some("s0"),
                Some(0),
                ClockContext::at(replay_ts),
            )
            .await
            .unwrap();

        // Inject a 2024-style foresight directly through the graph store.
        let valid_until = replay_ts + chrono::Duration::days(30);
        let fs = ForesightSignal::new(
            "2024-style foresight".into(),
            replay_note.id.clone(),
            Some(valid_until),
        );
        graph_store.upsert_foresight(&fs).await.unwrap();

        // Dream at the replay reference_time. The 2024 foresight (valid_until
        // 2024-04-14) is in the future relative to ref=2024-03-15 — should
        // survive expire_foresights.
        karta
            .run_dreaming_with_clock("test", "scope", ClockContext::at(replay_ts))
            .await
            .unwrap();

        let active = graph_store.get_active_foresights().await.unwrap();
        assert!(
            active.iter().any(|f| f.id == fs.id),
            "in-window foresight must survive a replay-anchored dream"
        );

        // Dream at wall-clock-now — the 2024 foresight is now past
        // valid_until (we're in 2026) and should expire.
        karta
            .run_dreaming("test", "scope")
            .await
            .unwrap();

        let active_after = graph_store.get_active_foresights().await.unwrap();
        assert!(
            !active_after.iter().any(|f| f.id == fs.id),
            "out-of-window foresight must expire when dreamt with the live clock"
        );
    }
}
