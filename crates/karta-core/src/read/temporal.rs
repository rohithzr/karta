//! Shared temporal types for `occurred_*` bounds and resolver outputs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Discrete confidence rubric. The LLM and the Rust resolver MUST emit
/// exactly one of these values. Continuous output is a schema violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub enum ConfidenceBand {
    /// No temporal content. Paired with null bounds.
    None,
    /// Vague reference, range chosen ("recently", "around March").
    Vague,
    /// Relative reference, deterministic resolution ("yesterday", "next Friday").
    Relative,
    /// Natural-language absolute date ("March 15, 2024").
    NLAbsolute,
    /// Explicit ISO date in source ("2024-03-15").
    Explicit,
}

impl ConfidenceBand {
    pub fn as_f32(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Vague => 0.5,
            Self::Relative => 0.7,
            Self::NLAbsolute => 0.8,
            Self::Explicit => 1.0,
        }
    }

    /// Try to parse a raw f32 into a band. Returns `Err` on any value
    /// outside the closed set — this is the schema-enforcement gate.
    pub fn try_from_f32(v: f32) -> Result<Self, ConfidenceError> {
        match v {
            x if x == 0.0 => Ok(Self::None),
            x if x == 0.5 => Ok(Self::Vague),
            x if x == 0.7 => Ok(Self::Relative),
            x if x == 0.8 => Ok(Self::NLAbsolute),
            x if x == 1.0 => Ok(Self::Explicit),
            _ => Err(ConfidenceError::NotInClosedSet(v)),
        }
    }
}

impl TryFrom<f32> for ConfidenceBand {
    type Error = ConfidenceError;
    fn try_from(v: f32) -> Result<Self, Self::Error> { Self::try_from_f32(v) }
}

impl From<ConfidenceBand> for f32 {
    fn from(b: ConfidenceBand) -> f32 { b.as_f32() }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfidenceError {
    #[error("confidence {0} not in closed set {{0.0, 0.5, 0.7, 0.8, 1.0}}")]
    NotInClosedSet(f32),
}

/// Half-open interval `[start, end)`. Instants are `[t, t+1ns)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}
