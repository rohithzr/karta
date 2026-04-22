//! `EntityType` — coarse type classification for the entity field.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// The user themselves ("the user", "I").
    User,
    /// A project the user owns or works on.
    Project,
    /// Another named individual.
    Person,
    /// A company, team, or organization.
    Org,
    /// A specific task or work item.
    Task,
    /// LLM did not commit to a typed entity — validator may reject.
    Unknown,
}

impl EntityType {
    /// `true` only for `Unknown`. Used by the specificity gate together
    /// with `Facet::is_generic` to reject "project + unknown" style facts.
    pub fn is_generic(self) -> bool {
        matches!(self, Self::Unknown)
    }
}
