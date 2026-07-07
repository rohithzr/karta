//! T8 — Storage round-trip preserves session_id and the new non-optional
//! source_timestamp column.

#[cfg(feature = "sqlite-vec")]
mod tests {
    use chrono::TimeZone;
    use karta_core::note::{MemoryNote, NoteStatus, Provenance};
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::VectorStore;
    use tempfile::TempDir;

    fn make_note(session_id: Option<&str>) -> MemoryNote {
        let now = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        MemoryNote {
            id: uuid::Uuid::new_v4().to_string(),
            content: "test".into(),
            context: "ctx".into(),
            keywords: vec!["k".into()],
            tags: vec!["t".into()],
            links: Vec::new(),
            embedding: vec![0.1_f32; 1536],
            created_at: now,
            updated_at: now,
            evolution_history: Vec::new(),
            provenance: Provenance::Observed,
            confidence: 1.0,
            status: NoteStatus::Active,
            last_accessed_at: now,
            turn_index: Some(7),
            source_timestamp: now,
            session_id: session_id.map(String::from),
        }
    }

    #[tokio::test]
    async fn upsert_and_get_preserves_session_id_and_source_timestamp() {
        let dir = TempDir::new().unwrap();
        let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
            .await
            .unwrap();

        let note = make_note(Some("beam-conv1-session0"));
        store.upsert(&note).await.unwrap();

        let fetched = store.get(&note.id).await.unwrap().expect("note exists");
        assert_eq!(
            fetched.session_id.as_deref(),
            Some("beam-conv1-session0"),
            "session_id round-trip"
        );
        assert_eq!(
            fetched.source_timestamp, note.source_timestamp,
            "source_timestamp round-trip"
        );
        assert_eq!(fetched.turn_index, Some(7));
    }

    #[tokio::test]
    async fn null_session_id_round_trips_as_none() {
        let dir = TempDir::new().unwrap();
        let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
            .await
            .unwrap();

        let note = make_note(None);
        store.upsert(&note).await.unwrap();

        let fetched = store.get(&note.id).await.unwrap().expect("note exists");
        assert!(fetched.session_id.is_none(), "no session_id");
    }
}
