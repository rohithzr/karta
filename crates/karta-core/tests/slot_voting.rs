//! Scripted-mock tests for the adaptive 2×-with-tiebreak slot voting
//! wrapper (`karta_core::extract::slots::vote_slots`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::Utc;

use karta_core::error::Result;
use karta_core::extract::slots::{vote_slots, Predicate};
use karta_core::llm::{ChatMessage, ChatResponse, GenConfig, LlmProvider};

/// A scripted mock provider: each `chat` call pops the next response body
/// off a queue and bumps a call counter. Panics (via unwrap) if `chat` is
/// called more times than scripted responses were provided — that would
/// indicate `vote_slots` made an unexpected extra call.
struct QueuedMockLlmProvider {
    responses: Mutex<Vec<String>>,
    calls: AtomicUsize,
}

impl QueuedMockLlmProvider {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().map(|s| s.to_string()).rev().collect()),
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for QueuedMockLlmProvider {
    async fn chat(&self, _messages: &[ChatMessage], _config: &GenConfig) -> Result<ChatResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let content = self
            .responses
            .lock()
            .unwrap()
            .pop()
            .expect("vote_slots made more chat calls than scripted");
        Ok(ChatResponse {
            content,
            tokens_used: 10,
            input_tokens: 5,
            output_tokens: 5,
        })
    }

    async fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(vec![])
    }

    fn model_id(&self) -> &str {
        "mock"
    }

    fn embedding_model_id(&self) -> &str {
        "mock"
    }
}

fn deadline_payload(entity: &str, value: &str) -> String {
    format!(
        r#"{{"slots": [{{"entity": "{entity}", "predicate": "deadline", "value": "{value}", "event_time": null, "source_span": "the deadline is {value}"}}]}}"#,
    )
}

#[tokio::test]
async fn two_agreeing_runs_yield_two_calls_and_keep_the_tuple() {
    let payload = deadline_payload("first sprint", "April 5");
    let llm = QueuedMockLlmProvider::new(vec![payload.as_str(), payload.as_str()]);

    let out = vote_slots(&llm, "the deadline is April 5", Utc::now(), true).await;

    assert_eq!(llm.call_count(), 2);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].predicate, Predicate::Deadline);
    assert_eq!(out[0].value, "April 5");
    assert_eq!(out[0].entity_text, "first sprint");
}

#[tokio::test]
async fn disagreeing_runs_trigger_third_call_and_keep_the_majority() {
    let payload_a = deadline_payload("first sprint", "April 5");
    let payload_b = deadline_payload("second sprint", "May 1");
    let llm = QueuedMockLlmProvider::new(vec![
        payload_a.as_str(),
        payload_b.as_str(),
        payload_a.as_str(),
    ]);

    let out = vote_slots(&llm, "sprint deadlines", Utc::now(), true).await;

    assert_eq!(llm.call_count(), 3);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].entity_text, "first sprint");
    assert_eq!(out[0].value, "April 5");
    assert!(!out.iter().any(|t| t.entity_text == "second sprint"));
}
