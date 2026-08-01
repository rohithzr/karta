//! Transcript parsing for Droid and Claude Code session JSONL files.
//!
//! When a `session_end` capture includes a `transcript_path`, the session
//! layer calls [`sweep_transcript`] to read the JSONL transcript and add a
//! note for each recoverable event. This acts as a fallback sweep for
//! captures that were missed while the session was running.
//!
//! The top-level API is client-agnostic: callers pass a path and the parser
//! auto-detects whether the file is a Droid or Claude Code transcript by
//! inspecting the first JSON line.

use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, anyhow};
use serde_json::Value;

use crate::karta_handle::KartaHandle;

/// A single event recovered from a transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct SweptEvent {
    /// Capture-style event type, e.g. `user_prompt`, `observation`,
    /// `turn_summary`, `subagent_result`.
    pub event_type: String,
    /// Human-readable content to store as a note.
    pub content: String,
    /// Optional turn index within the session.
    pub turn_index: Option<u32>,
}

/// Supported transcript clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptClient {
    Droid,
    ClaudeCode,
}

/// Parse a transcript file and add one note per recovered event.
///
/// The client is auto-detected from the first line. Returns the number of
/// events that were written as notes.
pub async fn sweep_transcript(path: &str, session_id: &str, handle: &KartaHandle) -> Result<usize> {
    let client = detect_client(path)?;
    let events = match client {
        TranscriptClient::Droid => parse_droid_transcript(path),
        TranscriptClient::ClaudeCode => parse_claude_transcript(path),
    }?;

    let mut count = 0;
    for event in events {
        let content = format!("[transcript:{}] {}", event.event_type, event.content);
        handle
            .karta
            .add_note_with_clock(
                &content,
                Some(session_id),
                event.turn_index,
                karta_core::ClockContext::now(),
            )
            .await?;
        count += 1;
    }

    Ok(count)
}

/// Inspect the first non-empty line of a JSONL transcript to decide which
/// parser should handle it.
///
/// Droid transcripts start with `{"type":"session_start",...}`.
/// Claude Code transcripts start with `{"hook_event_name":"SessionStart",...}`.
pub fn detect_client(path: &str) -> Result<TranscriptClient> {
    let file = File::open(path).with_context(|| format!("failed to open transcript: {path}"))?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .with_context(|| format!("failed to parse first transcript line as JSON: {path}"))?;

        if value.get("hook_event_name").is_some() {
            return Ok(TranscriptClient::ClaudeCode);
        }
        if value.get("type").is_some() {
            return Ok(TranscriptClient::Droid);
        }
        return Err(anyhow!(
            "cannot detect transcript client for {path}: first line has neither 'type' nor 'hook_event_name'"
        ));
    }

    Err(anyhow!("transcript file is empty: {path}"))
}

/// Parse a Droid session transcript JSONL file.
///
/// Recognised events:
/// - `message` with `message.role = "user"` and text content blocks (and no
///   `tool_result` blocks) -> `user_prompt`. Real Droid transcripts do not put a
///   `hookEventName` on content-bearing user prompt messages, so the parser
///   must not require one.
/// - `message` with `message.role = "assistant"` containing `tool_use` blocks
///   -> `observation` (tool input)
/// - `message` with `message.role = "user"` containing `tool_result` blocks
///   -> `observation` (tool output)
/// - `message` with `hookEventName = "SubagentStop"` -> `subagent_result`
/// - `message` with `message.role = "assistant"` final text -> `turn_summary`
///
/// Malformed JSONL lines are logged and skipped rather than aborting the whole
/// sweep.
pub fn parse_droid_transcript(path: &str) -> Result<Vec<SweptEvent>> {
    let file =
        File::open(path).with_context(|| format!("failed to open Droid transcript: {path}"))?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();
    let mut turn_index = 0u32;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    line = line_no + 1,
                    error = %e,
                    "skipping malformed Droid transcript line"
                );
                continue;
            }
        };

        if let Some(message) = value.get("message") {
            let role = message.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let hook_event_name = message.get("hookEventName").and_then(|v| v.as_str());

            match role {
                "user" => {
                    if hook_event_name == Some("SubagentStop") {
                        let content = build_subagent_result_content(message);
                        events.push(SweptEvent {
                            event_type: "subagent_result".to_string(),
                            content,
                            turn_index: Some(turn_index),
                        });
                    } else if let Some(text) = extract_text_from_content_blocks(message) {
                        if !has_tool_result_blocks(message) {
                            events.push(SweptEvent {
                                event_type: "user_prompt".to_string(),
                                content: text,
                                turn_index: Some(turn_index),
                            });
                            turn_index += 1;
                        }
                    } else if let Some(tool_result) = extract_tool_result(message) {
                        events.push(SweptEvent {
                            event_type: "observation".to_string(),
                            content: tool_result,
                            turn_index: Some(turn_index),
                        });
                    }
                }
                "assistant" => {
                    if let Some(tool_input) = extract_tool_use_input(message) {
                        events.push(SweptEvent {
                            event_type: "observation".to_string(),
                            content: tool_input,
                            turn_index: Some(turn_index),
                        });
                    }
                    if let Some(text) = extract_final_assistant_text(message) {
                        events.push(SweptEvent {
                            event_type: "turn_summary".to_string(),
                            content: text,
                            turn_index: Some(turn_index),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    Ok(events)
}

/// Parse a Claude Code session transcript JSONL file.
///
/// Claude Code's internal transcript format is versioned and not publicly
/// stable. This parser recognises the hook-input shapes documented in the
/// fixture README:
///
/// - `hook_event_name = "UserPromptSubmit"` -> `user_prompt`
/// - `hook_event_name = "PostToolUse"` -> `observation`
/// - `hook_event_name = "Stop"` -> `turn_summary`
/// - `hook_event_name = "SubagentStop"` -> `subagent_result`
///
/// Unknown event types and missing optional fields are skipped.
pub fn parse_claude_transcript(path: &str) -> Result<Vec<SweptEvent>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open Claude Code transcript: {path}"))?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();
    let mut turn_index = 0u32;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    line = line_no + 1,
                    error = %e,
                    "skipping malformed Claude Code transcript line"
                );
                continue;
            }
        };

        let hook_event_name = value.get("hook_event_name").and_then(|v| v.as_str());

        match hook_event_name {
            Some("UserPromptSubmit") => {
                if let Some(prompt) = value.get("prompt").and_then(|v| v.as_str()) {
                    events.push(SweptEvent {
                        event_type: "user_prompt".to_string(),
                        content: prompt.to_string(),
                        turn_index: Some(turn_index),
                    });
                    turn_index += 1;
                }
            }
            Some("PostToolUse") => {
                let content = build_claude_observation_content(&value);
                events.push(SweptEvent {
                    event_type: "observation".to_string(),
                    content,
                    turn_index: Some(turn_index),
                });
            }
            Some("Stop") => {
                if let Some(msg) = value.get("last_assistant_message").and_then(|v| v.as_str()) {
                    events.push(SweptEvent {
                        event_type: "turn_summary".to_string(),
                        content: msg.to_string(),
                        turn_index: Some(turn_index),
                    });
                }
            }
            Some("SubagentStop") => {
                let content = build_claude_subagent_content(&value);
                events.push(SweptEvent {
                    event_type: "subagent_result".to_string(),
                    content,
                    turn_index: Some(turn_index),
                });
            }
            _ => {}
        }
    }

    Ok(events)
}

fn extract_text_from_content_blocks(message: &Value) -> Option<String> {
    let content = message.get("content")?;
    let blocks = content.as_array()?;
    let mut texts = Vec::new();
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) == Some("text")
            && let Some(text) = block.get("text").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            texts.push(text.to_string());
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(texts.join(" "))
    }
}

fn extract_final_assistant_text(message: &Value) -> Option<String> {
    let content = message.get("content")?;
    let blocks = content.as_array()?;
    let mut texts = Vec::new();
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) == Some("text")
            && let Some(text) = block.get("text").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            texts.push(text.to_string());
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(texts.join(" "))
    }
}

fn extract_tool_use_input(message: &Value) -> Option<String> {
    let content = message.get("content")?;
    let blocks = content.as_array()?;
    let mut tools = Vec::new();
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
            let name = block
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let input = block.get("input").cloned().unwrap_or(Value::Null);
            tools.push(format!("tool: {name} input: {input}"));
        }
    }
    if tools.is_empty() {
        None
    } else {
        Some(tools.join(" | "))
    }
}

fn extract_tool_result(message: &Value) -> Option<String> {
    let content = message.get("content")?;
    let blocks = content.as_array()?;
    let mut results = Vec::new();
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) == Some("tool_result") {
            let tool_use_id = block
                .get("tool_use_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let is_error = block
                .get("is_error")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let result_content = block.get("content").and_then(|v| v.as_str()).unwrap_or("");
            results.push(format!(
                "tool_use_id: {tool_use_id} is_error: {is_error} content: {result_content}"
            ));
        }
    }
    if results.is_empty() {
        None
    } else {
        Some(results.join(" | "))
    }
}

fn has_tool_result_blocks(message: &Value) -> bool {
    let Some(content) = message.get("content") else {
        return false;
    };
    let Some(blocks) = content.as_array() else {
        return false;
    };
    blocks
        .iter()
        .any(|block| block.get("type").and_then(|v| v.as_str()) == Some("tool_result"))
}

fn build_subagent_result_content(message: &Value) -> String {
    let task_name = message
        .get("taskName")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let task_result = message
        .get("taskResult")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    format!("task: {task_name} result: {task_result}")
}

fn build_claude_observation_content(value: &Value) -> String {
    let tool_name = value
        .get("tool_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let tool_input = value.get("tool_input").cloned().unwrap_or(Value::Null);
    let tool_response = value
        .get("tool_response")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    format!("tool: {tool_name} input: {tool_input} response: {tool_response}")
}

fn build_claude_subagent_content(value: &Value) -> String {
    let task_name = value
        .get("task_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let task_result = value
        .get("task_result")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let task_error = value
        .get("task_error")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if task_error.is_empty() {
        format!("task: {task_name} result: {task_result}")
    } else {
        format!("task: {task_name} error: {task_error}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;
    use tempfile::TempDir;

    fn fixture_path(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/transcripts")
            .join(name);
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn detect_client_recognises_droid() {
        let path = fixture_path("droid-session.jsonl");
        assert_eq!(detect_client(&path).unwrap(), TranscriptClient::Droid);
    }

    #[test]
    fn detect_client_recognises_claude_code() {
        let path = fixture_path("claude-code-session.jsonl");
        assert_eq!(detect_client(&path).unwrap(), TranscriptClient::ClaudeCode);
    }

    #[test]
    fn detect_client_fails_on_empty_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();
        let err = detect_client(path.to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn parse_droid_transcript_extracts_events() {
        let path = fixture_path("droid-session.jsonl");
        let events = parse_droid_transcript(&path).unwrap();

        assert!(!events.is_empty());

        let user_prompts: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "user_prompt")
            .collect();
        assert_eq!(user_prompts.len(), 1);
        assert!(user_prompts[0].content.contains("Add a transcript parser"));

        let observations: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "observation")
            .collect();
        assert!(!observations.is_empty());
        let observation_texts: Vec<String> =
            observations.iter().map(|e| e.content.clone()).collect();
        let joined = observation_texts.join("\n");
        assert!(joined.contains("Create fixture directory"));
        assert!(joined.contains("tool-use-001"));

        let turn_summaries: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "turn_summary")
            .collect();
        assert!(!turn_summaries.is_empty());
        assert!(turn_summaries[0].content.contains("Directory created"));

        let subagent_results: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "subagent_result")
            .collect();
        assert_eq!(subagent_results.len(), 1);
        assert!(
            subagent_results[0]
                .content
                .contains("transcript-design-review")
        );
        assert!(
            subagent_results[0]
                .content
                .contains("parser design approved")
        );
    }

    #[test]
    fn parse_claude_transcript_extracts_events() {
        let path = fixture_path("claude-code-session.jsonl");
        let events = parse_claude_transcript(&path).unwrap();

        assert!(!events.is_empty());

        let user_prompts: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "user_prompt")
            .collect();
        assert_eq!(user_prompts.len(), 1);
        assert!(user_prompts[0].content.contains("Add a transcript parser"));

        let observations: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "observation")
            .collect();
        assert!(!observations.is_empty());
        assert!(observations[0].content.contains("Create fixture directory"));
        assert!(observations[0].content.contains("Execute"));

        let turn_summaries: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "turn_summary")
            .collect();
        assert!(!turn_summaries.is_empty());
        assert!(turn_summaries[0].content.contains("Directory created"));

        let subagent_results: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "subagent_result")
            .collect();
        assert_eq!(subagent_results.len(), 1);
        assert!(
            subagent_results[0]
                .content
                .contains("transcript-design-review")
        );
        assert!(
            subagent_results[0]
                .content
                .contains("parser design approved")
        );
    }

    #[tokio::test]
    async fn sweep_transcript_writes_notes_for_droid() {
        let dir = TempDir::new().unwrap();
        let handle = KartaHandle::open_mock(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let path = fixture_path("droid-session.jsonl");
        let session_id = "droid-sweep-test";

        let count = sweep_transcript(&path, session_id, &handle).await.unwrap();
        assert!(count > 0);

        let notes = handle.karta.get_all_notes().await.unwrap();
        let transcript_notes: Vec<_> = notes
            .iter()
            .filter(|n| n.content.starts_with("[transcript:"))
            .collect();
        assert_eq!(transcript_notes.len(), count);
        assert!(
            transcript_notes
                .iter()
                .any(|n| n.content.contains("Add a transcript parser"))
        );
        assert!(
            transcript_notes
                .iter()
                .any(|n| n.content.contains("transcript-design-review"))
        );
    }

    #[tokio::test]
    async fn sweep_transcript_writes_notes_for_claude_code() {
        let dir = TempDir::new().unwrap();
        let handle = KartaHandle::open_mock(dir.path().to_str().unwrap())
            .await
            .unwrap();
        let path = fixture_path("claude-code-session.jsonl");
        let session_id = "claude-sweep-test";

        let count = sweep_transcript(&path, session_id, &handle).await.unwrap();
        assert!(count > 0);

        let notes = handle.karta.get_all_notes().await.unwrap();
        let transcript_notes: Vec<_> = notes
            .iter()
            .filter(|n| n.content.starts_with("[transcript:"))
            .collect();
        assert_eq!(transcript_notes.len(), count);
        assert!(
            transcript_notes
                .iter()
                .any(|n| n.content.contains("Add a transcript parser"))
        );
        assert!(
            transcript_notes
                .iter()
                .any(|n| n.content.contains("transcript-design-review"))
        );
    }

    #[test]
    fn parse_droid_transcript_extracts_user_prompt_without_hook_event_name() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("real-shaped.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, "{{\"type\":\"session_start\",\"sessionId\":\"s1\"}}").unwrap();
        writeln!(
            file,
            "{{\"type\":\"message\",\"message\":{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"Add a transcript parser\"}}]}}}}"
        )
        .unwrap();
        drop(file);

        let events = parse_droid_transcript(path.to_str().unwrap()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "user_prompt");
        assert!(events[0].content.contains("Add a transcript parser"));
    }

    #[test]
    fn parse_droid_transcript_skips_malformed_lines() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mixed.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, "{{\"type\":\"session_start\",\"sessionId\":\"s1\"}}").unwrap();
        writeln!(
            file,
            "{{\"type\":\"message\",\"message\":{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"first\"}}]}}}}"
        )
        .unwrap();
        writeln!(file, "not valid json").unwrap();
        writeln!(
            file,
            "{{\"type\":\"message\",\"message\":{{\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"second\"}}]}}}}"
        )
        .unwrap();
        writeln!(file, "").unwrap();
        drop(file);

        let events = parse_droid_transcript(path.to_str().unwrap()).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "user_prompt");
        assert_eq!(events[0].content, "first");
        assert_eq!(events[1].event_type, "turn_summary");
        assert_eq!(events[1].content, "second");
    }

    #[test]
    fn parse_claude_transcript_tolerates_unknown_events() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("claude-unknown.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            "{{\"hook_event_name\":\"SessionStart\",\"session_id\":\"s1\"}}"
        )
        .unwrap();
        writeln!(file, "{{\"hook_event_name\":\"CustomEvent\",\"data\":123}}").unwrap();
        writeln!(
            file,
            "{{\"hook_event_name\":\"UserPromptSubmit\",\"prompt\":\"hello\"}}"
        )
        .unwrap();
        drop(file);

        let events = parse_claude_transcript(path.to_str().unwrap()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "user_prompt");
        assert_eq!(events[0].content, "hello");
    }

    #[test]
    fn parse_claude_transcript_skips_malformed_lines() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("claude-mixed.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            "{{\"hook_event_name\":\"SessionStart\",\"session_id\":\"s1\"}}"
        )
        .unwrap();
        writeln!(
            file,
            "{{\"hook_event_name\":\"UserPromptSubmit\",\"prompt\":\"first\"}}"
        )
        .unwrap();
        writeln!(file, "not valid json").unwrap();
        writeln!(
            file,
            "{{\"hook_event_name\":\"Stop\",\"last_assistant_message\":\"second\"}}"
        )
        .unwrap();
        writeln!(file, "").unwrap();
        drop(file);

        let events = parse_claude_transcript(path.to_str().unwrap()).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "user_prompt");
        assert_eq!(events[0].content, "first");
        assert_eq!(events[1].event_type, "turn_summary");
        assert_eq!(events[1].content, "second");
    }
}
