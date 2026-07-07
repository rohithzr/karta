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

/// Raw output from either resolver tier, pre-validation.
#[derive(Debug, Clone, Copy)]
pub struct RawResolverOutput {
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
    pub confidence_f32: f32,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ResolverValidationError {
    #[error("confidence {0} not in closed set {{0.0, 0.5, 0.7, 0.8, 1.0}}")]
    ConfidenceNotInClosedSet(f32),
    #[error("start and end must both be Some or both be None")]
    UnpairedBounds,
    #[error("end must be strictly greater than start")]
    EndNotAfterStart,
    #[error("confidence == 0.0 iff both bounds are None (got conf={conf}, paired={paired})")]
    ConfidenceBoundsMismatch { conf: f32, paired: bool },
}

/// Validate a resolver's raw output against the closed-set confidence
/// rubric AND the three interval invariants. This is SCHEMA ENFORCEMENT,
/// not a routing policy — bad outputs are rejected, callers fall through
/// to vector-only retrieval.
///
/// Invariant #3 (confidence in [0, 1]) is subsumed by the closed-set
/// check: values outside the closed set are rejected regardless of range.
pub fn validate_resolver_output(
    raw: RawResolverOutput,
) -> Result<(Option<Interval>, ConfidenceBand), ResolverValidationError> {
    let conf = ConfidenceBand::try_from_f32(raw.confidence_f32)
        .map_err(|_| ResolverValidationError::ConfidenceNotInClosedSet(raw.confidence_f32))?;

    let both_some = raw.start.is_some() && raw.end.is_some();
    let both_none = raw.start.is_none() && raw.end.is_none();

    if !(both_some || both_none) {
        return Err(ResolverValidationError::UnpairedBounds);
    }

    if let (Some(s), Some(e)) = (raw.start, raw.end) {
        if e <= s {
            return Err(ResolverValidationError::EndNotAfterStart);
        }
    }

    let conf_is_none = conf == ConfidenceBand::None;
    if conf_is_none != both_none {
        return Err(ResolverValidationError::ConfidenceBoundsMismatch {
            conf: raw.confidence_f32,
            paired: both_some,
        });
    }

    let interval = both_some.then(|| Interval { start: raw.start.unwrap(), end: raw.end.unwrap() });
    Ok((interval, conf))
}

/// Constant referenced only for readability; the validator enforces this
/// as part of the closed-set check. DO NOT read this at routing time —
/// it is not a policy threshold.
pub const RESOLVER_SCHEMA_MIN_CONFIDENCE: f32 = 0.5;
