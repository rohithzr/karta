//! `LlmProvider` wrapper that emits trace events for every chat/embed call.
//!
//! Events are emitted only when a `TraceWriter` is bound on the calling task.
//! When no trace context is active, this is a transparent pass-through.

use async_trait::async_trait;
use chrono::Utc;
use std::sync::Arc;
use std::time::Instant;

use super::traits::{ChatMessage, ChatResponse, GenConfig, LlmProvider};
use crate::error::Result;
use crate::trace::{self, TraceEvent};

pub struct TracingLlmProvider {
    inner: Arc<dyn LlmProvider>,
}

impl TracingLlmProvider {
    pub fn new(inner: Arc<dyn LlmProvider>) -> Self {
        Self { inner }
    }
}

fn render_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| {
            let role = match m.role {
                super::traits::Role::System => "system",
                super::traits::Role::User => "user",
                super::traits::Role::Assistant => "assistant",
            };
            format!("[{role}]\n{}", m.content)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[async_trait]
impl LlmProvider for TracingLlmProvider {
    async fn chat(&self, messages: &[ChatMessage], config: &GenConfig) -> Result<ChatResponse> {
        let start = Instant::now();
        let result = self.inner.chat(messages, config).await;
        let wall_ms = start.elapsed().as_millis() as u64;

        if let Ok(ref response) = result {
            let heavy = trace::heavy();
            trace::try_emit(TraceEvent::LlmChat {
                ts: Utc::now(),
                turn_idx: trace::current_turn(),
                stage: trace::current_stage(),
                model: self.inner.model_id().to_string(),
                wall_ms,
                input_tokens: response.input_tokens,
                output_tokens: response.output_tokens,
                prompt: if heavy { Some(render_prompt(messages)) } else { None },
                completion: if heavy { Some(response.content.clone()) } else { None },
            });
        }

        result
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let start = Instant::now();
        let result = self.inner.embed(texts).await;
        let wall_ms = start.elapsed().as_millis() as u64;

        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let heavy = trace::heavy();
        trace::try_emit(TraceEvent::LlmEmbed {
            ts: Utc::now(),
            turn_idx: trace::current_turn(),
            stage: trace::current_stage(),
            model: self.inner.embedding_model_id().to_string(),
            wall_ms,
            input_count: texts.len(),
            total_chars,
            inputs: if heavy {
                Some(texts.iter().map(|s| s.to_string()).collect())
            } else {
                None
            },
        });

        result
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn embedding_model_id(&self) -> &str {
        self.inner.embedding_model_id()
    }
}
