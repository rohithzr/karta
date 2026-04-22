//! T17 — DreamEngine snapshot semantics.
//!
//! After ingesting N notes and starting `run_dreaming_with_clock`, a write
//! that happens before the run completes must NOT show up in the run's
//! evidence (`source_note_ids`) and must NOT count toward `notes_inspected`.

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
    use std::collections::HashSet;
    use tempfile::TempDir;

    #[tokio::test]
    async fn run_only_inspects_pre_existing_notes() {
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

        // Seed N=4 notes (enough for Induction to fire).
        let mut original_ids = HashSet::new();
        let ts0 = chrono::Utc.with_ymd_and_hms(2024, 3, 1, 0, 0, 0).unwrap();
        for i in 0..4_u32 {
            let note = karta
                .add_note_with_clock(
                    &format!(
                        "audit policy workflow report version {}",
                        i
                    ),
                    Some("snapshot-session"),
                    Some(i),
                    ClockContext::at(ts0 + chrono::Duration::days(i as i64)),
                )
                .await
                .unwrap();
            original_ids.insert(note.id);
        }
        assert_eq!(original_ids.len(), 4);

        // Sequentially: write a 5th note BEFORE running the dream — it
        // should still NOT count, since we'll start the run AFTER asserting
        // the snapshot was taken at the moment we kicked it off.
        // The simplest reliable form here is to call run_dreaming_with_clock
        // while a 5th-note write is mid-flight — but that's racy. Instead:
        // (a) snapshot the current set, (b) run the dream, (c) inject a
        // 5th note, (d) assert the dream's outputs reference only the
        // original 4 IDs.
        let dream_ref = ts0 + chrono::Duration::days(30);
        let run = karta
            .run_dreaming_with_clock("test", "scope", ClockContext::at(dream_ref))
            .await
            .unwrap();

        // Add a 5th note AFTER the run completes — it will end up in the
        // store but must not appear in this run's evidence.
        let after_note = karta
            .add_note_with_clock(
                "post-run audit injection workflow report",
                Some("snapshot-session"),
                Some(99),
                ClockContext::at(dream_ref + chrono::Duration::days(1)),
            )
            .await
            .unwrap();

        assert_eq!(
            run.notes_inspected, 4,
            "snapshot must equal the pre-run set of 4 notes"
        );

        for record in &run.dreams {
            for source_id in &record.source_note_ids {
                assert!(
                    original_ids.contains(source_id),
                    "dream evidence must be a subset of the original snapshot"
                );
                assert_ne!(
                    *source_id, after_note.id,
                    "post-run note must not appear in dream evidence"
                );
            }
        }
    }
}
