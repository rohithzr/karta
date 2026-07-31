//! HTTP capture endpoints used by Droid and Claude Code lifecycle hooks.
//!
//! The axum router is bound to `127.0.0.1` only. It exposes:
//!
//! - `POST /capture`: parse the incoming event JSON, map `hook_event_name` to
//!   a server-side event type, and insert a durable row into `capture_queue`
//!   before returning `202 Accepted`.
//! - `POST /orient`: synchronous orientation call that runs `fetch_memories`
//!   and returns `{context, note_ids}` immediately with `200 OK`.

use std::sync::Arc;

use axum::{Router, body::Bytes, extract::State, http::StatusCode, response::Json};
use serde_json::{Value, json};
use thiserror::Error;

use crate::queue::CaptureQueue;

/// Shared application state for the capture router.
#[derive(Clone)]
pub struct AppState {
    pub karta: Arc<karta_core::Karta>,
    pub queue: Arc<CaptureQueue>,
}

/// Build the capture router around a shared `Karta` and `CaptureQueue`.
pub fn router(karta: Arc<karta_core::Karta>, queue: Arc<CaptureQueue>) -> Router {
    Router::new()
        .route("/capture", axum::routing::post(capture_handler))
        .route("/orient", axum::routing::post(orient_handler))
        .with_state(AppState { karta, queue })
}

async fn capture_handler(State(state): State<AppState>, body: Bytes) -> (StatusCode, Json<Value>) {
    let payload = match serde_json::from_slice::<Value>(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "capture request body is not valid JSON");
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "invalid JSON"})),
            );
        }
    };

    let (event_type, session_id) = match resolve_event_type(&payload) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, payload = ?payload, "capture event rejected");
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e.to_string()})),
            );
        }
    };

    if let Err(e) = state
        .queue
        .enqueue(&event_type, &payload, session_id.as_deref())
        .await
    {
        tracing::error!(error = %e, "failed to insert capture queue row");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "queue insert failed"})),
        );
    }

    tracing::info!(
        event_type = %event_type,
        session_id = ?session_id,
        "capture event queued"
    );

    (StatusCode::ACCEPTED, Json(json!({"status": "queued"})))
}

/// Map the incoming payload to a server-side event type and session id.
///
/// The primary mapping uses the Droid/Claude Code `hook_event_name` field.
/// An explicit `event` field is accepted as an override when it matches one
/// of the known capture types. `SessionStart` is rejected because it belongs
/// to `/orient`, not `/capture`.
fn resolve_event_type(payload: &Value) -> Result<(String, Option<String>), CaptureError> {
    // Accept an explicit `event` override if it names a known capture type.
    if let Some(event) = payload.get("event").and_then(|v| v.as_str()) {
        match event {
            "user_prompt" | "observation" | "turn_summary" | "subagent_result" | "session_end"
            | "pre_compact" => {
                return Ok((event.to_string(), session_id_from_payload(payload)));
            }
            _ => {}
        }
    }

    match payload.get("hook_event_name").and_then(|v| v.as_str()) {
        Some("UserPromptSubmit") => {
            Ok(("user_prompt".to_string(), session_id_from_payload(payload)))
        }
        Some("PostToolUse") => Ok(("observation".to_string(), session_id_from_payload(payload))),
        Some("Stop") => Ok(("turn_summary".to_string(), session_id_from_payload(payload))),
        Some("SubagentStop") => Ok((
            "subagent_result".to_string(),
            session_id_from_payload(payload),
        )),
        Some("SessionEnd") => Ok(("session_end".to_string(), session_id_from_payload(payload))),
        Some("PreCompact") => Ok(("pre_compact".to_string(), session_id_from_payload(payload))),
        Some("SessionStart") => Err(CaptureError::Orient),
        Some(other) => Err(CaptureError::Unknown(other.to_string())),
        None => Err(CaptureError::Missing),
    }
}

/// Extract a session id from the payload, preserving `None` when it is missing
/// or explicitly `null`.
fn session_id_from_payload(payload: &Value) -> Option<String> {
    payload.get("session_id").and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        _ => None,
    })
}

#[derive(Debug, Error)]
enum CaptureError {
    #[error("unknown hook_event_name: {0}")]
    Unknown(String),
    #[error("SessionStart should be sent to /orient, not /capture")]
    Orient,
    #[error("capture event missing hook_event_name or event")]
    Missing,
}

async fn orient_handler(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<Value>, StatusCode> {
    let payload = match serde_json::from_slice::<Value>(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "orient request body is not valid JSON");
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    let query = derive_orient_query(&payload);

    let result = match state.karta.fetch_memories(&query, 5).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "fetch_memories failed during orient");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    tracing::info!(
        query = %query,
        note_count = result.note_ids.len(),
        "orientation completed"
    );

    Ok(Json(json!({
        "context": result.context,
        "note_ids": result.note_ids,
    })))
}

/// Derive an orientation query from the incoming payload.
///
/// If the payload contains a `query` string it is used directly. Otherwise a
/// query is built from `agent`, `project`, and `cwd` (the latter is common in
/// both Droid and Claude Code `SessionStart` hook bodies).
fn derive_orient_query(payload: &Value) -> String {
    if let Some(query) = payload.get("query").and_then(|v| v.as_str()) {
        return query.to_string();
    }

    let agent = payload
        .get("agent")
        .and_then(|v| v.as_str())
        .unwrap_or("agent");
    let project = payload
        .get("project")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    if let Some(cwd) = payload.get("cwd").and_then(|v| v.as_str()) {
        format!("agent: {agent} project: {project} cwd: {cwd}")
    } else {
        format!("agent: {agent} project: {project}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_maps_user_prompt_submit() {
        let payload = json!({"hook_event_name": "UserPromptSubmit", "prompt": "hello"});
        let (event, sid) = resolve_event_type(&payload).unwrap();
        assert_eq!(event, "user_prompt");
        assert_eq!(sid, None);
    }

    #[test]
    fn resolve_maps_post_tool_use() {
        let payload = json!({"hook_event_name": "PostToolUse", "tool_input": {"file": "x.rs"}});
        let (event, _) = resolve_event_type(&payload).unwrap();
        assert_eq!(event, "observation");
    }

    #[test]
    fn resolve_maps_stop() {
        let payload = json!({"hook_event_name": "Stop", "last_assistant_message": "done"});
        let (event, _) = resolve_event_type(&payload).unwrap();
        assert_eq!(event, "turn_summary");
    }

    #[test]
    fn resolve_maps_subagent_stop() {
        let payload = json!({"hook_event_name": "SubagentStop", "task_name": "t"});
        let (event, _) = resolve_event_type(&payload).unwrap();
        assert_eq!(event, "subagent_result");
    }

    #[test]
    fn resolve_maps_session_end() {
        let payload = json!({"hook_event_name": "SessionEnd", "summary": "end"});
        let (event, _) = resolve_event_type(&payload).unwrap();
        assert_eq!(event, "session_end");
    }

    #[test]
    fn resolve_maps_pre_compact() {
        let payload = json!({"hook_event_name": "PreCompact"});
        let (event, _) = resolve_event_type(&payload).unwrap();
        assert_eq!(event, "pre_compact");
    }

    #[test]
    fn resolve_accepts_explicit_event_override() {
        let payload = json!({"event": "observation"});
        let (event, _) = resolve_event_type(&payload).unwrap();
        assert_eq!(event, "observation");
    }

    #[test]
    fn resolve_rejects_session_start_on_capture() {
        let payload = json!({"hook_event_name": "SessionStart"});
        let err = resolve_event_type(&payload).unwrap_err();
        assert!(err.to_string().contains("SessionStart"));
    }

    #[test]
    fn resolve_rejects_unknown_event() {
        let payload = json!({"hook_event_name": "Notification"});
        let err = resolve_event_type(&payload).unwrap_err();
        assert!(err.to_string().contains("Notification"));
    }

    #[test]
    fn resolve_rejects_missing_event() {
        let payload = json!({"session_id": "s1"});
        let err = resolve_event_type(&payload).unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn derive_orient_query_prefers_query_field() {
        let payload = json!({"query": "fox", "agent": "droid"});
        assert_eq!(derive_orient_query(&payload), "fox");
    }

    #[test]
    fn derive_orient_query_builds_from_agent_project_cwd() {
        let payload = json!({"agent": "droid", "project": "karta", "cwd": "/tmp"});
        assert_eq!(
            derive_orient_query(&payload),
            "agent: droid project: karta cwd: /tmp"
        );
    }

    #[test]
    fn derive_orient_query_uses_defaults() {
        let payload = json!({});
        assert_eq!(
            derive_orient_query(&payload),
            "agent: agent project: default"
        );
    }
}
