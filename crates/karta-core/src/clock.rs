//! `ClockContext` — separates "the data's now" from "the system's now".
//!
//! Threaded through ingest, dream, and read. Every clock-aware site agrees on
//! what time to reason from. Sites that record system events (row provenance,
//! tracing, last-accessed touches) keep using `Utc::now()` directly — there is
//! intentionally no `wall_time` field on this type.

use chrono::{DateTime, Utc};

/// The data's "now". Threaded through ingest/dream/read so every clock-aware
/// site agrees on what time to reason from.
///
/// `Utc::now()` is the system clock and is used directly at sites that need
/// it (row provenance, tracing, etc.) — there is intentionally no
/// `wall_time` field here. The whole point of the type is to carry the
/// thing that's *not* always `Utc::now()`.
#[derive(Debug, Clone, Copy)]
pub struct ClockContext {
    pub(crate) reference_time: DateTime<Utc>,
}

impl ClockContext {
    /// Default for live ingestion / queries: `reference_time = Utc::now()`.
    pub fn now() -> Self {
        Self {
            reference_time: Utc::now(),
        }
    }

    /// For benchmarks, replays, backfills, imports: the data's "now"
    /// at the moment the message was sent / event happened.
    pub fn at(reference_time: DateTime<Utc>) -> Self {
        Self { reference_time }
    }

    pub fn reference_time(&self) -> DateTime<Utc> {
        self.reference_time
    }
}

impl Default for ClockContext {
    fn default() -> Self {
        Self::now()
    }
}
