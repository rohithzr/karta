//! Scriptable LLM mock for adversarial-output validator tests.
//!
//! Hand it a list of (input_substring, response_json) pairs. On each
//! `chat` call, finds the first scripted entry whose input substring
//! appears in the user message, returns the paired JSON verbatim.
//!
//! This bypasses the heuristic mock's "smart" extraction so tests can
//! assert "validator drops X when LLM emits Y" — Y is exactly what you
//! scripted, not what a heuristic happens to produce.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::Result;
use crate::llm::{ChatMessage, ChatResponse, GenConfig, LlmProvider, MockLlmProvider, Role};

pub struct ScriptedMockLlmProvider {
    entries: Vec<(String, String)>,
    fallback: MockLlmProvider,
    fired: Mutex<Vec<usize>>,
}

impl ScriptedMockLlmProvider {
    pub fn new<K: Into<String>, V: Into<String>>(entries: Vec<(K, V)>) -> Self {
        Self {
            entries: entries
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            fallback: MockLlmProvider::new(),
            fired: Mutex::new(Vec::new()),
        }
    }

    pub fn fired_indices(&self) -> Vec<usize> {
        self.fired.lock().unwrap().clone()
    }
}

#[async_trait]
impl LlmProvider for ScriptedMockLlmProvider {
    async fn chat(&self, messages: &[ChatMessage], config: &GenConfig) -> Result<ChatResponse> {
        let user_text: String = messages
            .iter()
            .filter(|m| matches!(m.role, Role::User))
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        for (i, (needle, response)) in self.entries.iter().enumerate() {
            if user_text.contains(needle) {
                self.fired.lock().unwrap().push(i);
                return Ok(ChatResponse {
                    content: response.clone(),
                    tokens_used: 100,
                    input_tokens: 50,
                    output_tokens: 50,
                });
            }
        }
        // No entry matched — fall back to heuristic mock so the test author
        // isn't forced to script every chat call (e.g. dream prompts).
        self.fallback.chat(messages, config).await
    }

    async fn embed(&self, t: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.fallback.embed(t).await
    }

    fn model_id(&self) -> &str {
        self.fallback.model_id()
    }
    fn embedding_model_id(&self) -> &str {
        self.fallback.embedding_model_id()
    }
}
