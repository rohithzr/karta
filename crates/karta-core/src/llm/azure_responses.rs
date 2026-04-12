use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;
use tracing::warn;

use super::traits::*;
use crate::error::{KartaError, Result};

const MAX_RETRIES: u32 = 5;
const INITIAL_BACKOFF_MS: u64 = 1000;

/// Azure OpenAI provider using the Responses API (`/openai/responses`).
///
/// GPT-5.x models on Azure don't support the Chat Completions endpoint —
/// they require the newer Responses API with a different request/response shape.
/// Embeddings still use the standard Azure deployments endpoint.
pub struct AzureResponsesProvider {
    client: Client,
    endpoint: String,
    api_key: String,
    api_version: String,
    chat_model: String,
    embedding_deployment: String,
}

impl AzureResponsesProvider {
    pub fn new(
        endpoint: &str,
        api_key: &str,
        api_version: &str,
        chat_model: &str,
        embedding_deployment: &str,
    ) -> Self {
        Self {
            client: Client::new(),
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            api_version: api_version.to_string(),
            chat_model: chat_model.to_string(),
            embedding_deployment: embedding_deployment.to_string(),
        }
    }
}

fn is_retryable(status: Option<u16>, err_msg: &str) -> bool {
    if let Some(code) = status {
        return matches!(code, 429 | 500 | 502 | 503 | 504);
    }
    let lower = err_msg.to_lowercase();
    lower.contains("timeout")
        || lower.contains("timed out")
        || lower.contains("connection")
        || lower.contains("error sending request")
        || lower.contains("reset by peer")
}

/// Extract text content from Responses API output array.
fn extract_response_text(data: &Value) -> String {
    let empty = vec![];
    let output = data["output"].as_array().unwrap_or(&empty);
    let mut text = String::new();
    for item in output {
        if item["type"] == "message" {
            if let Some(content) = item["content"].as_array() {
                for block in content {
                    if block["type"] == "output_text" {
                        if let Some(s) = block["text"].as_str() {
                            text.push_str(s);
                        }
                    }
                }
            }
        }
    }
    text
}

#[async_trait]
impl LlmProvider for AzureResponsesProvider {
    async fn chat(&self, messages: &[ChatMessage], config: &GenConfig) -> Result<ChatResponse> {
        let url = format!(
            "{}/openai/responses?api-version={}",
            self.endpoint, self.api_version
        );

        // Map roles: System -> developer per Responses API convention
        let input: Vec<Value> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "developer",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };
                json!({"role": role, "content": m.content})
            })
            .collect();

        let mut body = json!({
            "model": self.chat_model,
            "input": input,
            "max_output_tokens": config.max_tokens,
        });

        if config.temperature != 0.0 {
            body["temperature"] = json!(config.temperature);
        }

        // Structured output via text.format
        if let Some(ref schema) = config.json_schema {
            body["text"] = json!({
                "format": {
                    "type": "json_schema",
                    "name": schema.name,
                    "schema": schema.schema,
                    "strict": true
                }
            });
        } else if config.json_mode {
            body["text"] = json!({
                "format": {"type": "json_object"}
            });
        }

        let mut last_err = KartaError::Llm("no attempts made".into());
        for attempt in 0..=MAX_RETRIES {
            let resp = self
                .client
                .post(&url)
                .header("api-key", &self.api_key)
                .json(&body)
                .send()
                .await;

            match resp {
                Ok(r) => {
                    let status = r.status().as_u16();
                    let text = r.text().await.map_err(|e| KartaError::Llm(e.to_string()))?;

                    if status == 200 {
                        let data: Value = serde_json::from_str(&text)
                            .map_err(|e| KartaError::Llm(format!("JSON parse error: {e}")))?;

                        let content = extract_response_text(&data);
                        let tokens_used =
                            data["usage"]["total_tokens"].as_u64().unwrap_or(0);

                        return Ok(ChatResponse {
                            content,
                            tokens_used,
                        });
                    }

                    if attempt < MAX_RETRIES && is_retryable(Some(status), &text) {
                        let backoff = INITIAL_BACKOFF_MS * 2u64.pow(attempt);
                        warn!(
                            attempt = attempt + 1,
                            status,
                            backoff_ms = backoff,
                            "Responses API failed, retrying"
                        );
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        last_err = KartaError::Llm(format!("HTTP {status}: {text}"));
                    } else {
                        return Err(KartaError::Llm(format!("HTTP {status}: {text}")));
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    if attempt < MAX_RETRIES && is_retryable(None, &msg) {
                        let backoff = INITIAL_BACKOFF_MS * 2u64.pow(attempt);
                        warn!(
                            attempt = attempt + 1,
                            backoff_ms = backoff,
                            error = %msg,
                            "Responses API request failed, retrying"
                        );
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        last_err = KartaError::Llm(msg);
                    } else {
                        return Err(KartaError::Llm(msg));
                    }
                }
            }
        }
        Err(last_err)
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.endpoint, self.embedding_deployment, self.api_version
        );

        let input: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let body = json!({"input": input});

        let mut last_err = KartaError::Llm("no attempts made".into());
        for attempt in 0..=MAX_RETRIES {
            let resp = self
                .client
                .post(&url)
                .header("api-key", &self.api_key)
                .json(&body)
                .send()
                .await;

            match resp {
                Ok(r) => {
                    let status = r.status().as_u16();
                    let text = r.text().await.map_err(|e| KartaError::Llm(e.to_string()))?;

                    if status == 200 {
                        let data: Value = serde_json::from_str(&text)
                            .map_err(|e| KartaError::Llm(format!("JSON parse error: {e}")))?;

                        let empty = vec![];
                        let embeddings = data["data"]
                            .as_array()
                            .ok_or_else(|| {
                                KartaError::Llm("No embeddings in response".into())
                            })?
                            .iter()
                            .map(|d| {
                                d["embedding"]
                                    .as_array()
                                    .unwrap_or(&empty)
                                    .iter()
                                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                                    .collect()
                            })
                            .collect();

                        return Ok(embeddings);
                    }

                    if attempt < MAX_RETRIES && is_retryable(Some(status), &text) {
                        let backoff = INITIAL_BACKOFF_MS * 2u64.pow(attempt);
                        warn!(
                            attempt = attempt + 1,
                            status,
                            backoff_ms = backoff,
                            "Embedding request failed, retrying"
                        );
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        last_err = KartaError::Llm(format!("HTTP {status}: {text}"));
                    } else {
                        return Err(KartaError::Llm(format!("HTTP {status}: {text}")));
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    if attempt < MAX_RETRIES && is_retryable(None, &msg) {
                        let backoff = INITIAL_BACKOFF_MS * 2u64.pow(attempt);
                        warn!(
                            attempt = attempt + 1,
                            backoff_ms = backoff,
                            error = %msg,
                            "Embedding request failed, retrying"
                        );
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        last_err = KartaError::Llm(msg);
                    } else {
                        return Err(KartaError::Llm(msg));
                    }
                }
            }
        }
        Err(last_err)
    }

    fn model_id(&self) -> &str {
        &self.chat_model
    }

    fn embedding_model_id(&self) -> &str {
        &self.embedding_deployment
    }
}
