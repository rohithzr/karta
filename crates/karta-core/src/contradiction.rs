use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Resolution outcome for a contradiction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ContradictionResolution {
    /// One source note is correct; the other is deprecated.
    Prefer { preferred_note_id: String },
    /// Both notes are partially correct; merged into a new note.
    Merge { merged_note_id: String },
    /// Both notes are outdated; superseded by newer information.
    Supersede { superseding_note_id: String },
    /// Marked as resolved without action (e.g., context-dependent).
    Dismissed { reason: String },
}

/// Lifecycle state of a contradiction.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ContradictionStatus {
    #[default]
    Open,
    Resolved,
    Ignored,
}

/// A first-class contradiction between two or more notes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub id: String,
    /// Entity or topic the contradiction is about.
    pub entity: String,
    /// Scope this contradiction belongs to (e.g., session ID or global).
    pub scope_id: String,
    /// IDs of the source notes that contradict each other.
    pub source_note_ids: Vec<String>,
    /// Human-readable description of the contradiction.
    pub description: String,
    /// Which dream run produced this contradiction (if any).
    pub dream_run_id: Option<String>,
    pub status: ContradictionStatus,
    pub resolution: Option<ContradictionResolution>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolved_by: Option<String>,
    pub ignore_reason: Option<String>,
    pub ignored_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

impl Contradiction {
    pub fn new(
        entity: String,
        scope_id: String,
        source_note_ids: Vec<String>,
        description: String,
        dream_run_id: Option<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            entity,
            scope_id,
            source_note_ids,
            description,
            dream_run_id,
            status: ContradictionStatus::Open,
            resolution: None,
            resolved_at: None,
            resolved_by: None,
            ignore_reason: None,
            ignored_at: None,
            created_at: Utc::now(),
        }
    }

    pub fn is_open(&self) -> bool {
        self.status == ContradictionStatus::Open
    }
}
