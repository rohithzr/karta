use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DreamType {
    Deduction,
    Induction,
    Abduction,
    Consolidation,
    Contradiction,
    EpisodeDigest,
    CrossEpisodeDigest,
}

impl DreamType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Deduction => "deduction",
            Self::Induction => "induction",
            Self::Abduction => "abduction",
            Self::Consolidation => "consolidation",
            Self::Contradiction => "contradiction",
            Self::EpisodeDigest => "episode_digest",
            Self::CrossEpisodeDigest => "cross_episode_digest",
        }
    }

    pub fn parse_kind(s: &str) -> Option<Self> {
        match s {
            "deduction" => Some(Self::Deduction),
            "induction" => Some(Self::Induction),
            "abduction" => Some(Self::Abduction),
            "consolidation" => Some(Self::Consolidation),
            "contradiction" => Some(Self::Contradiction),
            "episode_digest" => Some(Self::EpisodeDigest),
            "cross_episode_digest" => Some(Self::CrossEpisodeDigest),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseDreamTypeError;

impl fmt::Display for ParseDreamTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid dream type")
    }
}

impl std::error::Error for ParseDreamTypeError {}

impl FromStr for DreamType {
    type Err = ParseDreamTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_kind(s).ok_or(ParseDreamTypeError)
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
