//! T3 — Foresight DOA filter at write.
//!
//! Verifies WriteEngine drops foresights whose valid_until falls inside
//! the 24h DOA horizon of ctx.reference_time(), and keeps ones that fall
//! outside.

#[cfg(feature = "sqlite-vec")]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use chrono::TimeZone;
    use karta_core::clock::ClockContext;
    use karta_core::config::{EpisodeConfig, WriteConfig};
    use karta_core::error::Result;
    use karta_core::llm::{ChatMessage, ChatResponse, GenConfig, LlmProvider, Role};
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use karta_core::write::WriteEngine;
    use tempfile::TempDir;

    struct FixedForesightLlm {
        valid_until: String,
    }

    #[async_trait]
    impl LlmProvider for FixedForesightLlm {
        async fn chat(&self, messages: &[ChatMessage], _config: &GenConfig) -> Result<ChatResponse> {
            let system_msg = messages
                .iter()
                .find(|m| matches!(m.role, Role::System))
                .map(|m| m.content.as_str())
                .unwrap_or("");

            let content = if system_msg.contains("memory indexing system") {
                serde_json::json!({
                    "context": "test note",
                    "keywords": ["test"],
                    "tags": ["pattern"],
                    "foresight_signals": [
                        { "content": "probe foresight", "valid_until": self.valid_until }
                    ],
                    "atomic_facts": []
                })
                .to_string()
            } else if system_msg.contains("should be linked") {
                serde_json::json!({ "links": [] }).to_string()
            } else {
                serde_json::json!({}).to_string()
            };

            Ok(ChatResponse {
                content,
                tokens_used: 10,
                input_tokens: 5,
                output_tokens: 5,
            })
        }

        async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1_f32; 1536]).collect())
        }

        fn model_id(&self) -> &str {
            "fixed-foresight-shim"
        }

        fn embedding_model_id(&self) -> &str {
            "fixed-foresight-shim"
        }
    }

    async fn write_one(dir: &std::path::Path, valid_until: &str, ref_time_iso: &str) -> Arc<dyn GraphStore> {
        let data_dir = dir.to_str().unwrap();
        let vec_store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        let shared_conn = vec_store.connection();
        let graph_store = SqliteGraphStore::with_connection(shared_conn);
        graph_store.init().await.unwrap();

        let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
        let graph_store_arc = Arc::new(graph_store) as Arc<dyn GraphStore>;
        let llm: Arc<dyn LlmProvider> = Arc::new(FixedForesightLlm {
            valid_until: valid_until.to_string(),
        });

        let engine = WriteEngine::new(
            Arc::clone(&vector_store),
            Arc::clone(&graph_store_arc),
            llm,
            WriteConfig::default(),
            EpisodeConfig::default(),
        );

        let ref_time: chrono::DateTime<chrono::Utc> = ref_time_iso.parse().unwrap();
        engine
            .add_note_with_clock("hello", Some("s0"), Some(0), ClockContext::at(ref_time))
            .await
            .unwrap();

        graph_store_arc
    }

    #[tokio::test]
    async fn doa_drops_same_day_foresight() {
        let dir = TempDir::new().unwrap();
        let graph = write_one(dir.path(), "2024-03-15", "2024-03-15T00:00:00Z").await;
        let active = graph.get_active_foresights().await.unwrap();
        assert!(
            active.is_empty(),
            "expected same-day foresight to be dropped, got {} active",
            active.len()
        );
    }

    #[tokio::test]
    async fn doa_keeps_future_dated_foresight() {
        let dir = TempDir::new().unwrap();
        let graph = write_one(dir.path(), "2024-05-15", "2024-03-15T00:00:00Z").await;
        let active = graph.get_active_foresights().await.unwrap();
        assert_eq!(
            active.len(),
            1,
            "expected 2-month-out foresight to survive the DOA filter"
        );
    }

    #[test]
    fn default_thresholds_are_one_day() {
        assert_eq!(WriteConfig::default().foresight_doa_threshold_days, 1.0);
        assert_eq!(WriteConfig::default().future_skew_threshold_days, 1.0);
        let _ = chrono::Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap(); // keep imports used
    }
}
