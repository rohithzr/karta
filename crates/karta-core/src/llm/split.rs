use async_trait::async_trait;
use std::sync::Arc;

use super::traits::{ChatMessage, ChatResponse, GenConfig, LlmProvider};
use crate::error::Result;

/// Composite provider that routes `chat()` to one backend and `embed()` to
/// another. Built for the "Ollama chat + Azure embeddings" setup where a
/// local GPU serves generation while a hosted provider handles embeddings
/// (better quality, larger context, free concurrency).
pub struct SplitProvider {
    chat_backend: Arc<dyn LlmProvider>,
    embed_backend: Arc<dyn LlmProvider>,
}

impl SplitProvider {
    pub fn new(
        chat_backend: Arc<dyn LlmProvider>,
        embed_backend: Arc<dyn LlmProvider>,
    ) -> Self {
        Self { chat_backend, embed_backend }
    }
}

#[async_trait]
impl LlmProvider for SplitProvider {
    async fn chat(&self, messages: &[ChatMessage], config: &GenConfig) -> Result<ChatResponse> {
        self.chat_backend.chat(messages, config).await
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_backend.embed(texts).await
    }

    fn model_id(&self) -> &str {
        self.chat_backend.model_id()
    }

    fn embedding_model_id(&self) -> &str {
        self.embed_backend.embedding_model_id()
    }
}
