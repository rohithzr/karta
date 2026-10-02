//! Drives the deterministic mock to produce the expected shape per
//! input pattern. These rules are documented in mock.rs::handle_attributes.

use karta_core::llm::schemas::note_attributes_schema;
use karta_core::llm::{
    ChatMessage, GenConfig, LlmProvider, MockLlmProvider, Prompts, Role, ScriptedMockLlmProvider,
};
use std::sync::Arc;

async fn ask(mock: Arc<dyn LlmProvider>, content: &str) -> serde_json::Value {
    let msgs = vec![
        ChatMessage {
            role: Role::System,
            content: Prompts::note_attributes_system().to_string(),
        },
        ChatMessage {
            role: Role::User,
            content: format!(
                "reference_time: 2024-03-15T00:00:00Z\n\nMessage:\n{}",
                content
            ),
        },
    ];
    let cfg = GenConfig {
        max_tokens: 1024,
        temperature: 0.0,
        json_mode: false,
        json_schema: Some(note_attributes_schema()),
    };
    let resp = mock.chat(&msgs, &cfg).await.unwrap();
    serde_json::from_str(&resp.content).unwrap()
}

#[tokio::test]
async fn pure_request_returns_zero_facts() {
    let mock: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new());
    let r = ask(mock, "Can you help me build a schedule?").await;
    let facts = r["atomic_facts"].as_array().unwrap();
    assert_eq!(facts.len(), 0, "pure request should produce zero facts");
}

#[tokio::test]
async fn deadline_message_emits_future_commitment() {
    let mock: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new());
    let r = ask(mock, "I have an April 15 deadline for the budget tracker.").await;
    let facts = r["atomic_facts"].as_array().unwrap();
    assert!(!facts.is_empty(), "deadline message should produce ≥1 fact");
    let f = &facts[0];
    assert_eq!(f["memory_kind"].as_str(), Some("future_commitment"));
    assert_eq!(f["facet"].as_str(), Some("deadline"));
    let spans = f["supporting_spans"].as_array().unwrap();
    assert!(!spans.is_empty());
    assert!(spans.iter().any(|s| s.as_str() == Some("April 15")));
}

#[tokio::test]
async fn tech_stack_message_emits_typed_facts() {
    let mock: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new());
    let r = ask(mock, "Project uses Flask 2.3.1 with Python 3.11.").await;
    let facts = r["atomic_facts"].as_array().unwrap();
    assert!(!facts.is_empty());
    let f = &facts[0];
    assert_eq!(f["facet"].as_str(), Some("tech_stack"));
    assert_eq!(f["entity_type"].as_str(), Some("project"));
}

#[tokio::test]
async fn scripted_mock_returns_canned_response() {
    let mock: Arc<dyn LlmProvider> = Arc::new(ScriptedMockLlmProvider::new(vec![(
        "DEADLINE_TEST",
        r#"{
            "reasoning": "test",
            "context": "test",
            "keywords": [],
            "tags": [],
            "foresight_signals": [],
            "atomic_facts": [{
                "content": "test fact",
                "memory_kind": "future_commitment",
                "supporting_spans": ["DEADLINE_TEST"],
                "facet": "deadline",
                "entity_type": "project",
                "entity_text": "test",
                "value_text": null,
                "value_date": "2024-04-15T00:00:00Z",
                "occurred_start": null,
                "occurred_end": null,
                "occurred_confidence": 0.0
            }]
        }"#,
    )]));
    let r = ask(mock, "DEADLINE_TEST scenario").await;
    let facts = r["atomic_facts"].as_array().unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0]["memory_kind"].as_str(), Some("future_commitment"));
}
