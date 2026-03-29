use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DreamType {
    Deduction,
    Induction,
    Abduction,
    Consolidation,
    Contradiction,
}

impl DreamType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Deduction => "deduction",
            Self::Induction => "induction",
            Self::Abduction => "abduction",
            Self::Consolidation => "consolidation",
            Self::Contradiction => "contradiction",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "deduction" => Some(Self::Deduction),
            "induction" => Some(Self::Induction),
            "abduction" => Some(Self::Abduction),
            "consolidation" => Some(Self::Consolidation),
            "contradiction" => Some(Self::Contradiction),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamRecord {
    pub id: String,
    pub dream_type: DreamType,
    pub source_note_ids: Vec<String>,
    pub reasoning: String,
    pub dream_content: String,
    pub confidence: f32,
    pub would_write: bool,
    pub written_note_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamRun {
    pub id: String,
    pub scope_type: String,
    pub scope_id: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub notes_inspected: usize,
    pub dreams_attempted: usize,
    pub dreams_written: usize,
    pub dreams: Vec<DreamRecord>,
    pub total_tokens_used: u64,
}
