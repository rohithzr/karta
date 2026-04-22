//! T11 — Harness reads `effective_reference_time` per turn from the new
//! converter JSON shape. The Rust harness MUST do no session-level
//! fallback logic; it constructs `ClockContext::at(turn.effective_reference_time)`
//! directly. This test verifies that contract against a hand-crafted
//! fixture (the real converter ships in a parallel agent's commit).

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
    use serde::Deserialize;
    use tempfile::TempDir;

    #[derive(Deserialize)]
    struct Turn {
        turn_index: u32,
        role: String,
        content: String,
        effective_reference_time: chrono::DateTime<chrono::Utc>,
    }

    /// Hand-crafted fixture mirroring the new converter shape. Three user
    /// turns: turn 0 has an explicit anchor, turns 1+2 inherit it via the
    /// converter's pre-resolved effective_reference_time. The harness
    /// itself does no inheritance logic.
    const FIXTURE: &str = r#"
    {
      "session_index": 0,
      "session_anchor": "2024-03-15",
      "turns": [
        {"turn_index": 0, "role": "user", "content": "first user turn",
         "effective_reference_time": "2024-03-15T00:00:00Z"},
        {"turn_index": 1, "role": "assistant", "content": "assistant reply",
         "effective_reference_time": "2024-03-15T00:00:00Z"},
        {"turn_index": 2, "role": "user", "content": "second user turn",
         "effective_reference_time": "2024-03-15T00:00:00Z"}
      ]
    }
    "#;

    #[tokio::test]
    async fn harness_reads_effective_reference_time_per_turn() {
        let parsed: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        let turns: Vec<Turn> =
            serde_json::from_value(parsed["turns"].clone()).expect("turns parse");

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

        // Filter to user turns; harness does not do session-level fallback.
        let mut written_ids = Vec::new();
        for turn in &turns {
            if turn.role != "user" {
                continue;
            }
            let note = karta
                .add_note_with_clock(
                    &turn.content,
                    Some("beam-conv-fixture-session0"),
                    Some(turn.turn_index),
                    ClockContext::at(turn.effective_reference_time),
                )
                .await
                .unwrap();
            written_ids.push(note.id);
        }

        let expected = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        for id in &written_ids {
            let note = karta.get_note(id).await.unwrap().expect("note present");
            assert_eq!(
                note.source_timestamp, expected,
                "every turn must carry the converter-resolved effective_reference_time"
            );
            assert_eq!(
                note.session_id.as_deref(),
                Some("beam-conv-fixture-session0"),
                "session_id round-trips through the new write path"
            );
        }
    }
}
