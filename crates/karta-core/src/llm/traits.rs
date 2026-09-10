use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GenConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub json_mode: bool,
    /// Optional JSON schema for structured output. When set, the model is
    /// constrained to produce valid JSON matching this schema.
    /// Takes precedence over `json_mode`.
    pub json_schema: Option<JsonSchema>,
}

/// A JSON schema definition for structured output.
#[derive(Debug, Clone)]
pub struct JsonSchema {
    pub name: String,
    pub schema: serde_json::Value,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.0,
            json_mode: true,
            json_schema: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub tokens_used: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Pluggable LLM provider. Implement for OpenAI, Anthropic, Ollama, etc.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate a chat completion.
    async fn chat(&self, messages: &[ChatMessage], config: &GenConfig) -> Result<ChatResponse>;

    /// Generate embeddings for one or more texts.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// The model identifier.
    fn model_id(&self) -> &str;

    /// The embedding model identifier.
    fn embedding_model_id(&self) -> &str;
}
