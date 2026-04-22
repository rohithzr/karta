//! T20 — Episode narrative-note source_timestamp == max(input.source_timestamp).
//!
//! Mirrors T16 for the write-side narrative inference. An episode with three
//! notes spanning 2024-03-01..2024-03-15 must produce a narrative note
//! stamped with 2024-03-15 (the bounded claim), not Utc::now().

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;

    use chrono::TimeZone;
    use karta_core::clock::ClockContext;
    use karta_core::config::{EpisodeConfig, KartaConfig};
    use karta_core::llm::MockLlmProvider;
    use karta_core::note::Provenance;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::Karta;
    use tempfile::TempDir;

    #[tokio::test]
    async fn narrative_note_uses_max_evidence_source_timestamp() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let vec_store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        let shared_conn = vec_store.connection();
        let graph_store = SqliteGraphStore::with_connection(shared_conn);
        let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
        let graph_store = Arc::new(graph_store) as Arc<dyn GraphStore>;
        let llm = Arc::new(MockLlmProvider::new());

        let mut config = KartaConfig::default();
        config.episode = EpisodeConfig {
            enabled: true,
            // Big window so the three turns stay in the same episode.
            time_gap_threshold_secs: 10 * 86_400,
        };

        let karta = Karta::new(vector_store.clone(), graph_store.clone(), llm, config)
            .await
            .unwrap();

        let dates = ["2024-03-01", "2024-03-08", "2024-03-15"];
        let max_expected = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        for (i, date) in dates.iter().enumerate() {
            let ts: chrono::DateTime<chrono::Utc> = format!("{}T00:00:00Z", date).parse().unwrap();
            karta
                .add_note_with_metadata(
                    &format!("project budget update entry {}", i),
                    "narr-session",
                    Some(i as u32),
                    Some(ts),
                )
                .await
                .unwrap();
        }

        // The narrative-note is an Episode-provenance note. Pull all notes
        // and find ones with that provenance.
        let all = vector_store.get_all().await.unwrap();
        let narrative_notes: Vec<_> = all
            .iter()
            .filter(|n| matches!(n.provenance, Provenance::Episode { .. }))
            .collect();

        assert!(
            !narrative_notes.is_empty(),
            "expected at least one episode narrative note"
        );

        // After 3 notes added, the latest narrative-note's source_timestamp
        // must equal the max input source_timestamp (2024-03-15).
        let latest = narrative_notes
            .iter()
            .max_by_key(|n| n.updated_at)
            .unwrap();
        assert_eq!(
            latest.source_timestamp, max_expected,
            "narrative-note source_timestamp must be max(input), got {}",
            latest.source_timestamp
        );

        // ctx import is exercised here so the compiler keeps the use line.
        let _ = ClockContext::now();
    }
}
