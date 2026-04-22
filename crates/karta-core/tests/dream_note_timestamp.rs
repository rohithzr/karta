//! T16 — Dream-output `source_timestamp == max(evidence.source_timestamp)`.
//!
//! Run a dream over an input batch whose source_timestamps span 2024-03-01
//! ... 2024-03-15 at ctx.reference_time = 2024-04-01. Persisted dream-notes
//! must carry source_timestamp == 2024-03-15 — NOT 2024-04-01 (which would
//! create the time-travel-confidence bug for back-dated queries).

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;

    use chrono::TimeZone;
    use karta_core::clock::ClockContext;
    use karta_core::config::KartaConfig;
    use karta_core::dream::DreamEngine;
    use karta_core::llm::MockLlmProvider;
    use karta_core::note::Provenance;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::Karta;
    use tempfile::TempDir;

    #[tokio::test]
    async fn dream_note_source_timestamp_is_max_evidence() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let vec_store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        let shared_conn = vec_store.connection();
        let graph_store = SqliteGraphStore::with_connection(shared_conn);
        let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
        let graph_store_arc = Arc::new(graph_store) as Arc<dyn GraphStore>;
        let llm = Arc::new(MockLlmProvider::new());
        let karta = Karta::new(
            Arc::clone(&vector_store),
            Arc::clone(&graph_store_arc),
            Arc::clone(&llm) as Arc<_>,
            KartaConfig::default(),
        )
        .await
        .unwrap();

        // 4 notes with source_timestamps spanning 2024-03-01..2024-03-15.
        // Need ≥4 for Induction to fire (cross-cluster, no linking required).
        let dates = ["2024-03-01", "2024-03-05", "2024-03-10", "2024-03-15"];
        let max_expected = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        let bodies = [
            "Enterprise audit trail policy requires structured notification workflow",
            "Compliance verification mandates quarterly audit reports policy",
            "Audit policy must include workflow timestamps for compliance",
            "Audit notification workflow uses structured policy documents",
        ];
        for (date, body) in dates.iter().zip(bodies.iter()) {
            let ts: chrono::DateTime<chrono::Utc> = format!("{}T00:00:00Z", date).parse().unwrap();
            karta
                .add_note_with_clock(body, Some("s0"), Some(0), ClockContext::at(ts))
                .await
                .unwrap();
        }

        // Run dreaming at reference_time = 2024-04-01 — well past all evidence.
        let dream_ref = chrono::Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap();
        let dream_engine = DreamEngine::new(
            Arc::clone(&vector_store),
            Arc::clone(&graph_store_arc),
            Arc::clone(&llm) as Arc<_>,
            KartaConfig::default().dream,
        );
        let run = dream_engine
            .run_with_clock("test", "scope", ClockContext::at(dream_ref))
            .await
            .unwrap();

        // At least one Induction dream should have written. Pull the persisted
        // dream notes and assert their source_timestamp.
        let written = run.dreams.iter().filter(|d| d.would_write).count();
        assert!(written > 0, "expected at least one written dream");

        let all = vector_store.get_all().await.unwrap();
        let dream_notes: Vec<_> = all.iter().filter(|n| n.is_dream()).collect();
        assert!(!dream_notes.is_empty(), "expected persisted dream notes");

        for note in &dream_notes {
            // source_timestamp must equal max evidence (2024-03-15) — NOT
            // ctx.reference_time (2024-04-01) and NOT Utc::now().
            assert_eq!(
                note.source_timestamp, max_expected,
                "dream-note source_timestamp must be max(evidence), not ctx.reference_time"
            );
            // sanity: provenance is Dream
            assert!(matches!(note.provenance, Provenance::Dream { .. }));
        }
    }
}
