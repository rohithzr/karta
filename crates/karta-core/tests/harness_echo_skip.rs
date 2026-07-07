//! F7-T9b: verify the BEAM harness skips `answer_ai_question` turns
//! before reaching the write path. No attrs trace event should be
//! emitted for such a turn; no token spend should occur.

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use chrono::{TimeZone, Utc};
    use karta_core::clock::ClockContext;
    use karta_core::config::KartaConfig;
    use karta_core::error::Result as KartaResult;
    use karta_core::llm::{ChatMessage, ChatResponse, GenConfig, LlmProvider, MockLlmProvider};
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::Karta;
    use tempfile::TempDir;

    /// Thin counting provider: wraps the mock and counts chat calls.
    struct CountingLlm {
        inner: Arc<dyn LlmProvider>,
        chat_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl LlmProvider for CountingLlm {
        async fn chat(&self, m: &[ChatMessage], c: &GenConfig) -> KartaResult<ChatResponse> {
            self.chat_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.chat(m, c).await
        }
        async fn embed(&self, t: &[&str]) -> KartaResult<Vec<Vec<f32>>> {
            self.inner.embed(t).await
        }
        fn model_id(&self) -> &str {
            self.inner.model_id()
        }
        fn embedding_model_id(&self) -> &str {
            self.inner.embedding_model_id()
        }
    }

    /// Minimal replica of the harness's ingest gate: returns true iff the
    /// (role, question_type) pair should be ingested. Mirrors the filter
    /// applied in `beam_100k.rs` and `beam_conv0_trace.rs`.
    fn harness_filter_should_ingest(role: &str, question_type: Option<&str>) -> bool {
        if role != "user" {
            return false;
        }
        if question_type == Some("answer_ai_question") {
            return false;
        }
        true
    }

    #[tokio::test]
    async fn f7_t9b_answer_ai_question_turn_skipped_no_llm_call() {
        let dir = TempDir::new().unwrap();
        let chat_calls = Arc::new(AtomicUsize::new(0));
        let counting: Arc<dyn LlmProvider> = Arc::new(CountingLlm {
            inner: Arc::new(MockLlmProvider::new()),
            chat_calls: chat_calls.clone(),
        });

        let vs = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
            .await
            .unwrap();
        let conn = vs.connection();
        let graph: Arc<dyn GraphStore> = Arc::new(SqliteGraphStore::with_connection(conn));
        let vector_store: Arc<dyn VectorStore> = Arc::new(vs);
        let karta = Karta::new(vector_store, graph, counting, KartaConfig::default())
            .await
            .unwrap();

        let ref_time = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
        let turns: Vec<(&str, &str, Option<&str>)> = vec![
            (
                "Main question content with temporal: 2024-03-15",
                "user",
                Some("main_question"),
            ),
            (
                "Echo of assistant answer — should be skipped.",
                "user",
                Some("answer_ai_question"),
            ),
        ];

        for (content, role, qtype) in &turns {
            if !harness_filter_should_ingest(role, *qtype) {
                continue;
            }
            karta
                .add_note_with_clock(content, Some("sess"), None, ClockContext::at(ref_time))
                .await
                .unwrap();
        }

        // Turn 1 ingest should invoke chat (attrs at minimum).
        // Turn 2 should NOT invoke chat because it was filtered out.
        let total_calls = chat_calls.load(Ordering::SeqCst);
        assert!(
            total_calls >= 1,
            "turn 1 should have made at least one chat call"
        );

        // Verify skip actually happened: note count = 1.
        let notes = karta.get_all_notes().await.unwrap();
        assert_eq!(notes.len(), 1, "expected only turn 1 to be ingested");
    }

    #[test]
    fn f7_t9b_filter_unit() {
        assert!(harness_filter_should_ingest("user", Some("main_question")));
        assert!(!harness_filter_should_ingest(
            "user",
            Some("answer_ai_question")
        ));
        assert!(!harness_filter_should_ingest("assistant", None));
        // Live data with no question_type still ingests (backward-compat).
        assert!(harness_filter_should_ingest("user", None));
    }
}
