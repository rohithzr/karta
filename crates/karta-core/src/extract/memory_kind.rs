//! `MemoryKind` — admission classification.
//!
//! Five durable kinds (admit), three ephemeral kinds (reject in validator).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Generic claim about the user's world that should outlive this turn.
    DurableFact,
    /// Future-tense fact: deadlines, scheduled events, plans.
    FutureCommitment,
    /// User preference (favors X, prefers Y).
    Preference,
    /// Explicit choice the user made.
    Decision,
    /// Hard requirement or limitation.
    Constraint,
    /// Speech act: "I want help with X" — the request, not a fact.
    EphemeralRequest,
    /// Discourse marker: "thanks", "ok", "got it".
    SpeechAct,
    /// User echoing the assistant's prior message.
    Echo,
}

impl MemoryKind {
    /// Should facts of this kind be persisted?
    /// `false` = admission gate drops the fact silently.
    pub fn is_durable(self) -> bool {
        matches!(
            self,
            Self::DurableFact
                | Self::FutureCommitment
                | Self::Preference
                | Self::Decision
                | Self::Constraint
        )
    }
}
