use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::config::ForgetConfig;
use crate::error::Result;
use crate::note::{MemoryNote, NoteStatus, Provenance};
use crate::store::{GraphStore, VectorStore};

/// Result of a forgetting run.
#[derive(Debug, Clone, Serialize)]
pub struct ForgetRun {
    pub run_id: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub notes_inspected: usize,
    pub notes_archived: usize,
    pub notes_deprecated: usize,
    pub notes_protected: usize,
    pub archived_ids: Vec<String>,
    pub deprecated_ids: Vec<String>,
    pub protected_ids: Vec<String>,
    pub links_decayed: Option<usize>,
    pub warnings: Vec<String>,
}

/// A candidate for forgetting (used in preview and actual runs).
#[derive(Debug, Clone, Serialize)]
pub struct ForgetCandidate {
    pub note_id: String,
    pub decay_score: f64,
    pub last_accessed_at: DateTime<Utc>,
    pub current_status: NoteStatus,
    pub is_protected: bool,
    pub protection_reason: Option<String>,
    pub action: ForgetAction,
}

#[derive(Debug, Clone, Serialize)]
pub enum ForgetAction {
    Archive,
    Deprecate,
    Skip,
}

/// Preview of what a forgetting run would do, without mutations.
#[derive(Debug, Clone, Serialize)]
pub struct ForgetPreview {
    pub generated_at: DateTime<Utc>,
    pub candidates: Vec<ForgetCandidate>,
    pub total_archived: usize,
    pub total_deprecated: usize,
    pub total_protected: usize,
    pub links_decayed: Option<usize>,
    pub warnings: Vec<String>,
}

/// Engine that performs access-based forgetting over notes and link decay.
pub struct ForgetEngine {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    config: ForgetConfig,
}

impl ForgetEngine {
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        config: ForgetConfig,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            config,
        }
    }

    /// Run forgetting: archive stale low-activation notes, decay link weights.
    /// Idempotent — running twice produces the same result.
    pub async fn run_forgetting(&self) -> Result<ForgetRun> {
        if !self.config.enabled {
            let now = Utc::now();
            return Ok(ForgetRun {
                run_id: uuid::Uuid::new_v4().to_string(),
                started_at: now,
                completed_at: now,
                notes_inspected: 0,
                notes_archived: 0,
                notes_deprecated: 0,
                notes_protected: 0,
                archived_ids: vec![],
                deprecated_ids: vec![],
                protected_ids: vec![],
                links_decayed: Some(0),
                warnings: vec!["Forgetting engine is disabled".into()],
            });
        }

        let started_at = Utc::now();
        let notes = self.vector_store.get_all().await?;

        let mut archived_ids = Vec::new();
        let mut deprecated_ids = Vec::new();
        let mut protected_ids = Vec::new();
        let notes_inspected = notes.len();

        for note in &notes {
            if matches!(
                note.status,
                NoteStatus::Archived | NoteStatus::Superseded { .. }
            ) {
                continue;
            }

            if self.is_protected(note) {
                protected_ids.push(note.id.clone());
                continue;
            }

            let decay_score = self.compute_decay_score(note, &started_at);

            if decay_score < self.config.archive_threshold as f64 {
                let mut updated = note.clone();
                updated.status = NoteStatus::Archived;
                updated.updated_at = Utc::now();
                self.vector_store.upsert(&updated).await?;
                archived_ids.push(note.id.clone());
            } else if note.status == NoteStatus::Active
                && decay_score < (self.config.archive_threshold as f64 * 2.0)
            {
                let mut updated = note.clone();
                updated.status = NoteStatus::Deprecated {
                    by: "forgetting_engine".into(),
                };
                updated.updated_at = Utc::now();
                self.vector_store.upsert(&updated).await?;
                deprecated_ids.push(note.id.clone());
            }
        }

        let links_decayed = self.decay_semantic_links(&started_at).await?;
        let mut warnings = Vec::new();
        if links_decayed.is_none() {
            warnings.push(
                "Semantic link decay is not implemented for this graph store/schema yet".into(),
            );
        }

        Ok(ForgetRun {
            run_id: uuid::Uuid::new_v4().to_string(),
            started_at,
            completed_at: Utc::now(),
            notes_inspected,
            notes_archived: archived_ids.len(),
            notes_deprecated: deprecated_ids.len(),
            notes_protected: protected_ids.len(),
            archived_ids,
            deprecated_ids,
            protected_ids,
            links_decayed,
            warnings,
        })
    }

    /// Dry-run preview: returns the same candidates as an actual run without mutations.
    pub async fn preview_forgetting(&self) -> Result<ForgetPreview> {
        let notes = self.vector_store.get_all().await?;
        let now = Utc::now();
        let links_decayed = if self.config.enabled {
            self.decay_semantic_links(&now).await?
        } else {
            Some(0)
        };
        let mut warnings = Vec::new();
        if !self.config.enabled {
            warnings.push("Forgetting engine is disabled".into());
        } else if links_decayed.is_none() {
            warnings.push(
                "Semantic link decay is not implemented for this graph store/schema yet".into(),
            );
        }

        let mut candidates = Vec::new();
        let mut total_archived = 0;
        let mut total_deprecated = 0;
        let mut total_protected = 0;

        for note in &notes {
            if !self.config.enabled
                || matches!(
                    note.status,
                    NoteStatus::Archived | NoteStatus::Superseded { .. }
                )
            {
                continue;
            }

            let is_protected = self.is_protected(note);
            let protection_reason = if is_protected {
                Some(self.protection_reason(note))
            } else {
                None
            };

            let decay_score = self.compute_decay_score(note, &now);

            let action = if is_protected {
                total_protected += 1;
                ForgetAction::Skip
            } else if decay_score < self.config.archive_threshold as f64 {
                total_archived += 1;
                ForgetAction::Archive
            } else if note.status == NoteStatus::Active
                && decay_score < (self.config.archive_threshold as f64 * 2.0)
            {
                total_deprecated += 1;
                ForgetAction::Deprecate
            } else {
                continue;
            };

            candidates.push(ForgetCandidate {
                note_id: note.id.clone(),
                decay_score,
                last_accessed_at: note.last_accessed_at,
                current_status: note.status.clone(),
                is_protected,
                protection_reason,
                action,
            });
        }

        Ok(ForgetPreview {
            generated_at: now,
            candidates,
            total_archived,
            total_deprecated,
            total_protected,
            links_decayed,
            warnings,
        })
    }

    /// Compute an exponential decay score based on time since last access.
    /// Score of 1.0 = just accessed, approaches 0.0 over time.
    fn compute_decay_score(&self, note: &MemoryNote, now: &DateTime<Utc>) -> f64 {
        let elapsed_days = (now
            .signed_duration_since(note.last_accessed_at)
            .num_milliseconds() as f64
            / 86_400_000.0)
            .max(0.0);
        let half_life = self.config.decay_half_life_days;
        if half_life <= 0.0 {
            return 1.0;
        }
        // Exponential decay: score = 0.5^(elapsed / half_life)
        0.5_f64.powf(elapsed_days / half_life).clamp(0.0, 1.0)
    }

    /// Check if a note should be protected from forgetting.
    /// Observed notes (user input), profiles, and episodes are protected by default.
    fn is_protected(&self, note: &MemoryNote) -> bool {
        matches!(note.provenance, Provenance::Observed) || note.is_profile() || note.is_episode()
    }

    /// Human-readable reason why a note is protected.
    fn protection_reason(&self, note: &MemoryNote) -> String {
        debug_assert!(self.is_protected(note));
        if matches!(note.provenance, Provenance::Observed) {
            "observed_note".into()
        } else if note.is_profile() {
            "profile_note".into()
        } else {
            "episode_note".into()
        }
    }

    /// Decay semantic link weights.
    ///
    /// Returns `Some(count)` when link decay is supported and ran, and `None`
    /// when the current graph store/schema cannot represent weighted link decay yet.
    async fn decay_semantic_links(&self, _now: &DateTime<Utc>) -> Result<Option<usize>> {
        // TODO(forgetting-engine): Link decay requires typed/weighted link support from PR #3 (ACTIVATE).
        // When that PR merges, this will call graph_store.decay_link_weights().
        // For now, return None so consumers can distinguish "not implemented"
        // from "implemented and matched zero links".
        let _ = &self.graph_store;
        Ok(None)
    }
}
