//! `Facet` — what aspect of an entity a fact describes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Facet {
    /// A hard end date the entity must hit ("April 15 deadline").
    Deadline,
    /// A planned date that could slip ("targeting March 15 for v1").
    TargetDate,
    /// User-stated preference about the entity.
    Preference,
    /// Technology stack / dependencies.
    TechStack,
    /// Physical or logical location.
    Location,
    /// Who owns / is responsible for the entity.
    Ownership,
    /// A hard constraint or non-negotiable.
    Constraint,
    /// A discrete event involving the entity.
    Event,
    /// LLM did not commit to a typed facet — validator may reject.
    Unknown,
}

impl Facet {
    /// `true` only for `Unknown`. Used by the specificity gate together with
    /// `EntityType::is_generic` to reject "project + unknown" style facts.
    pub fn is_generic(self) -> bool {
        matches!(self, Self::Unknown)
    }
}
