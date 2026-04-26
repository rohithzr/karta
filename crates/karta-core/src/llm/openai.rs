use async_openai::{
    Client,
    config::{AzureConfig, OpenAIConfig},
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs,
        CreateEmbeddingRequestArgs,
    },
};
use async_trait::async_trait;
use std::time::Duration;
use tracing::warn;

use super::traits::*;
use crate::error::{KartaError, Result};

const MAX_RETRIES: u32 = 5;
const INITIAL_BACKOFF_MS: u64 = 1000;

/// Internal enum to hold either OpenAI or Azure client.
enum AnyClient {
    OpenAI(Client<OpenAIConfig>),
    Azure {
        /// One client per deployment (chat model)
        chat_client: Client<AzureConfig>,
        /// Separate client for embedding deployment
        embed_client: Client<AzureConfig>,
    },
}

/// OpenAI-compatible LLM provider with retry and exponential backoff.
/// Works with OpenAI, Azure OpenAI, Ollama, vLLM, Groq, Together, etc.
pub struct OpenAiProvider {
    client: AnyClient,
    chat_model: String,
    embedding_model: String,
}

impl OpenAiProvider {
    /// Standard OpenAI (reads OPENAI_API_KEY from env).
    pub fn new(chat_model: &str, embedding_model: &str) -> Self {
        Self {
            client: AnyClient::OpenAI(Client::with_config(OpenAIConfig::default())),
            chat_model: chat_model.to_string(),
            embedding_model: embedding_model.to_string(),
        }
    }

    /// Custom base URL (for Ollama, vLLM, Groq, Together, etc.)
    pub fn with_base_url(chat_model: &str, embedding_model: &str, base_url: &str) -> Self {
        let config = OpenAIConfig::new().with_api_base(base_url);
        Self {
            client: AnyClient::OpenAI(Client::with_config(config)),
            chat_model: chat_model.to_string(),
            embedding_model: embedding_model.to_string(),
        }
    }

    /// Custom API key + optional base URL (standard OpenAI-compatible).
    pub fn with_api_key(
        chat_model: &str,
        embedding_model: &str,
        api_key: &str,
        base_url: Option<&str>,
    ) -> Self {
        let mut config = OpenAIConfig::new().with_api_key(api_key);
        if let Some(url) = base_url {
            config = config.with_api_base(url);
        }
        Self {
            client: AnyClient::OpenAI(Client::with_config(config)),
            chat_model: chat_model.to_string(),
            embedding_model: embedding_model.to_string(),
        }
    }

    /// Azure OpenAI with separate deployment IDs for chat and embedding.
    pub fn azure(
        endpoint: &str,
        api_key: &str,
        api_version: &str,
        chat_deployment: &str,
        embedding_deployment: &str,
    ) -> Self {
        let chat_config = AzureConfig::new()
            .with_api_base(endpoint)
            .with_api_key(api_key)
            .with_api_version(api_version)
            .with_deployment_id(chat_deployment);

        let embed_config = AzureConfig::new()
            .with_api_base(endpoint)
            .with_api_key(api_key)
            .with_api_version(api_version)
            .with_deployment_id(embedding_deployment);

        Self {
            client: AnyClient::Azure {
                chat_client: Client::with_config(chat_config),
                embed_client: Client::with_config(embed_config),
            },
            chat_model: chat_deployment.to_string(),
            embedding_model: embedding_deployment.to_string(),
        }
    }
}

fn build_messages(messages: &[ChatMessage]) -> Vec<ChatCompletionRequestMessage> {
    messages
        .iter()
        .map(|m| match m.role {
            Role::System => {
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: m.content.clone().into(),
                    name: None,
                })
            }
            Role::User => ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: m.content.clone().into(),
                name: None,
            }),
            Role::Assistant => ChatCompletionRequestMessage::Assistant(
                async_openai::types::ChatCompletionRequestAssistantMessage {
                    content: Some(m.content.clone().into()),
                    ..Default::default()
                },
            ),
        })
        .collect()
}

fn build_chat_request(
    model: &str,
    messages: Vec<ChatCompletionRequestMessage>,
    config: &GenConfig,
) -> std::result::Result<async_openai::types::CreateChatCompletionRequest, KartaError> {
    use async_openai::types::{ResponseFormat, ResponseFormatJsonSchema};

    let mut builder = CreateChatCompletionRequestArgs::default();
    builder
        .model(model)
        .messages(messages)
        .max_completion_tokens(config.max_tokens);

    if config.temperature > 0.0 {
        builder.temperature(config.temperature);
    }

    // Structured output with JSON schema takes precedence over json_mode
    if let Some(ref schema) = config.json_schema {
        builder.response_format(ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                name: schema.name.clone(),
                description: None,
                schema: Some(schema.schema.clone()),
                strict: Some(true),
            },
        });
    } else if config.json_mode {
        builder.response_format(ResponseFormat::JsonObject);
    }

    builder.build().map_err(|e| KartaError::Llm(e.to_string()))
}

fn build_embed_request(
    model: &str,
    texts: &[&str],
) -> std::result::Result<async_openai::types::CreateEmbeddingRequest, KartaError> {
    let input: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
    CreateEmbeddingRequestArgs::default()
        .model(model)
        .input(input)
        .build()
        .map_err(|e| KartaError::Llm(e.to_string()))
}

/// Check if an error is retryable (transient network/rate-limit issues).
fn is_retryable(err_msg: &str) -> bool {
    let lower = err_msg.to_lowercase();
    lower.contains("timeout")
        || lower.contains("connection")
        || lower.contains("rate limit")
        || lower.contains("429")
        || lower.contains("500")
        || lower.contains("502")
        || lower.contains("503")
        || lower.contains("504")
        || lower.contains("error sending request")
        || lower.contains("timed out")
        || lower.contains("reset by peer")
}

/// Retry an async operation with exponential backoff.
async fn with_retry<F, Fut, T>(operation: &str, f: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last_err = KartaError::Llm("no attempts made".into());

    for attempt in 0..=MAX_RETRIES {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                let err_msg = e.to_string();
                if attempt < MAX_RETRIES && is_retryable(&err_msg) {
                    let backoff = INITIAL_BACKOFF_MS * 2u64.pow(attempt);
                    warn!(
                        attempt = attempt + 1,
                        max = MAX_RETRIES,
                        backoff_ms = backoff,
                        error = %err_msg,
                        "{} failed, retrying",
                        operation
                    );
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                    last_err = e;
                } else {
                    return Err(e);
                }
            }
        }
    }

    Err(last_err)
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn chat(&self, messages: &[ChatMessage], config: &GenConfig) -> Result<ChatResponse> {
        let oai_messages = build_messages(messages);

        with_retry("chat", || async {
            let request = build_chat_request(&self.chat_model, oai_messages.clone(), config)?;

            let response = match &self.client {
                AnyClient::OpenAI(c) => c
                    .chat()
                    .create(request)
                    .await
                    .map_err(|e| KartaError::Llm(e.to_string()))?,
                AnyClient::Azure { chat_client, .. } => chat_client
                    .chat()
                    .create(request)
                    .await
                    .map_err(|e| KartaError::Llm(e.to_string()))?,
            };

            let content = response
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();

            let tokens_used = response.usage.map(|u| u.total_tokens as u64).unwrap_or(0);

            Ok(ChatResponse {
                content,
                tokens_used,
            })
        })
        .await
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let texts_owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();

        with_retry("embed", || async {
            let text_refs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
            let request = build_embed_request(&self.embedding_model, &text_refs)?;

            let response = match &self.client {
                AnyClient::OpenAI(c) => c
                    .embeddings()
                    .create(request)
                    .await
                    .map_err(|e| KartaError::Llm(e.to_string()))?,
                AnyClient::Azure { embed_client, .. } => embed_client
                    .embeddings()
                    .create(request)
                    .await
                    .map_err(|e| KartaError::Llm(e.to_string()))?,
            };

            Ok(response.data.into_iter().map(|d| d.embedding).collect())
        })
        .await
    }

    fn model_id(&self) -> &str {
        &self.chat_model
    }

    fn embedding_model_id(&self) -> &str {
        &self.embedding_model
    }
}
