//! Contract replay harness: golden stdin JSON per event per client.
//!
//! Each fixture under `tests/fixtures/hooks/{droid,claude}/` is replayed through
//! the in-process `/capture` endpoint. After the queue drains we assert that the
//! server-side event mapping and the stored note content match the fixture.

mod common;

use std::path::{Path, PathBuf};

use serde_json::Value;

/// Map a fixture payload to the server-side event type it should produce.
fn expected_event_type(payload: &Value) -> String {
    if let Some(event) = payload.get("event").and_then(|v| v.as_str()) {
        match event {
            "user_prompt" | "observation" | "turn_summary" | "subagent_result" | "session_end"
            | "pre_compact" => return event.to_string(),
            _ => {}
        }
    }
    match payload.get("hook_event_name").and_then(|v| v.as_str()) {
        Some("UserPromptSubmit") => "user_prompt",
        Some("PostToolUse") => "observation",
        Some("Stop") => "turn_summary",
        Some("SubagentStop") => "subagent_result",
        Some("SessionEnd") => "session_end",
        Some("PreCompact") => "pre_compact",
        _ => "unknown",
    }
    .to_string()
}

/// Return a substring that must appear in the stored note for a fixture.
fn expected_note_substring(event_type: &str, payload: &Value) -> String {
    match event_type {
        "user_prompt" => payload["prompt"].as_str().unwrap_or("").to_string(),
        "observation" => payload["tool_output"]
            .as_str()
            .or_else(|| payload["tool_response"].as_str())
            .unwrap_or("")
            .to_string(),
        "turn_summary" => payload["last_assistant_message"]
            .as_str()
            .unwrap_or("")
            .to_string(),
        "subagent_result" => payload["task_result"].as_str().unwrap_or("").to_string(),
        "session_end" => payload["summary"].as_str().unwrap_or("").to_string(),
        "pre_compact" => "karta:pre_compact".to_string(),
        _ => "".to_string(),
    }
}

#[tokio::test]
async fn contract_replay_maps_all_client_events_and_stores_content() {
    let rt = common::TestRuntime::with_precompact(true).await;
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hooks");
    let clients = ["droid", "claude"];

    let mut all_fixtures: Vec<(String, PathBuf, Value)> = Vec::new();
    for client in &clients {
        let dir = fixtures_dir.join(client);
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap())
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let payload: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            all_fixtures.push((client.to_string(), path, payload));
        }
    }

    for (client, path, payload) in &all_fixtures {
        let (status, body) = rt.post_capture(payload.clone()).await;
        assert_eq!(
            status,
            reqwest::StatusCode::ACCEPTED,
            "fixture {}:{} should be accepted: {:?}",
            client,
            path.display(),
            body
        );
    }

    rt.drain().await;

    let rows = common::all_queue_rows(rt.data_dir());
    assert_eq!(
        rows.len(),
        all_fixtures.len(),
        "every fixture should produce a queue row"
    );

    for ((client, path, payload), (row_id, row_event, _, _)) in all_fixtures.iter().zip(rows.iter())
    {
        let expected_event = expected_event_type(payload);
        assert_eq!(
            row_event,
            &expected_event,
            "fixture {}:{} should map to event type {} (row id {})",
            client,
            path.display(),
            expected_event,
            row_id
        );
    }

    let notes = rt.handle.karta.get_all_notes().await.unwrap();
    for (client, path, payload) in &all_fixtures {
        let event = expected_event_type(payload);
        let expected = expected_note_substring(&event, payload);
        assert!(
            !expected.is_empty(),
            "fixture {}:{} should have a non-empty expected substring",
            client,
            path.display()
        );
        assert!(
            notes.iter().any(|n| n.content.contains(&expected)),
            "expected note content containing {:?} for fixture {}:{}",
            expected,
            client,
            path.display()
        );
    }

    rt.cleanup().await;
}
