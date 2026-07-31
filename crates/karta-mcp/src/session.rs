//! Session lifecycle and rule-based consolidation helpers.
//!
//! This module is intentionally minimal for the `mcp-server-tools` feature.
//! The dedicated `session-consolidate` feature will expand the consolidation
//! logic and add unit tests.

use anyhow::Result;
use chrono::Utc;

const CONFIDENCE_THRESHOLD: f32 = 0.9;

/// Start a new session, returning a unique session id and orientation context.
pub async fn session_start(
    agent: &str,
    project: Option<&str>,
    karta: &karta_core::Karta,
) -> Result<(String, String)> {
    let project = project.unwrap_or("default");
    let session_id = format!(
        "{}-{}-{}",
        sanitize(agent),
        sanitize(project),
        Utc::now().timestamp_millis()
    );
    let query = format!("agent: {} project: {}", agent, project);
    let result = karta.fetch_memories(&query, 5).await?;
    Ok((session_id, result.context))
}

/// End a session by writing a marker/summary note and triggering consolidation.
pub async fn session_end(
    session_id: &str,
    summary: Option<&str>,
    karta: &karta_core::Karta,
) -> Result<String> {
    let summary_text = summary.unwrap_or("no summary provided");
    let content = format!("Session {} ended. Summary: {}", session_id, summary_text);
    let note = karta
        .add_note_with_clock(
            &content,
            Some(session_id),
            None,
            karta_core::ClockContext::now(),
        )
        .await?;
    // Trigger rule-based consolidation (count only, no LLM).
    let _ = consolidate(Some(session_id), karta).await?;
    Ok(note.id)
}

/// Rule-based consolidation: no LLM, no dream. Returns the number of active
/// notes whose confidence is above the threshold, optionally restricted to a
/// session.
pub async fn consolidate(session_id: Option<&str>, karta: &karta_core::Karta) -> Result<usize> {
    let notes = karta.get_all_notes().await?;
    let promoted = notes
        .iter()
        .filter(|n| {
            n.is_active()
                && n.confidence >= CONFIDENCE_THRESHOLD
                && session_id.is_none_or(|sid| n.session_id.as_deref() == Some(sid))
        })
        .count();
    Ok(promoted)
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}
