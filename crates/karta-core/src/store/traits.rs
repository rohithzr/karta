use async_trait::async_trait;

use crate::error::Result;
use crate::note::{Episode, ForesightSignal, MemoryNote};

/// Stores note embeddings and metadata. Provides ANN similarity search.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert or update a note with its embedding.
    async fn upsert(&self, note: &MemoryNote) -> Result<()>;

    /// Find the top-K most similar notes by embedding.
    async fn find_similar(
        &self,
        embedding: &[f32],
        top_k: usize,
        exclude_ids: &[&str],
    ) -> Result<Vec<(MemoryNote, f32)>>;

    /// Get a single note by ID.
    async fn get(&self, id: &str) -> Result<Option<MemoryNote>>;

    /// Get multiple notes by IDs.
    async fn get_many(&self, ids: &[&str]) -> Result<Vec<MemoryNote>>;

    /// Get all notes (for dreaming). Use sparingly.
    async fn get_all(&self) -> Result<Vec<MemoryNote>>;

    /// Delete a note.
    async fn delete(&self, id: &str) -> Result<()>;

    /// Total note count.
    async fn count(&self) -> Result<usize>;
}

/// Stores graph edges (links), evolution history, dream state,
/// foresight signals, episodes, and profiles.
#[async_trait]
pub trait GraphStore: Send + Sync {
    // --- Links ---

    /// Add a bidirectional link between two notes.
    async fn add_link(&self, from_id: &str, to_id: &str, reason: &str) -> Result<()>;

    /// Get all note IDs linked to a given note.
    async fn get_links(&self, note_id: &str) -> Result<Vec<String>>;

    /// Get links with reasons.
    async fn get_links_with_reasons(&self, note_id: &str) -> Result<Vec<(String, String)>>;

    /// Get the number of links for a note (for graph-aware scoring).
    async fn get_link_count(&self, note_id: &str) -> Result<usize> {
        Ok(self.get_links(note_id).await?.len())
    }

    // --- Evolution history ---

    /// Record an evolution event.
    async fn record_evolution(
        &self,
        note_id: &str,
        triggered_by: &str,
        previous_context: &str,
    ) -> Result<()>;

    /// Get evolution history for a note.
    async fn get_evolution_history(&self, note_id: &str) -> Result<Vec<crate::note::EvolutionRecord>>;

    // --- Dream state ---

    /// Record a dream run.
    async fn record_dream_run(&self, run: &crate::dream::DreamRun) -> Result<()>;

    /// Get the last dream cursor (timestamp of last processed note).
    async fn get_dream_cursor(&self) -> Result<Option<chrono::DateTime<chrono::Utc>>>;

    /// Update the dream cursor.
    async fn set_dream_cursor(&self, cursor: chrono::DateTime<chrono::Utc>) -> Result<()>;

    // --- Foresight signals (Phase 2B.2) ---

    async fn upsert_foresight(&self, _signal: &ForesightSignal) -> Result<()> { Ok(()) }
    async fn get_active_foresights(&self) -> Result<Vec<ForesightSignal>> { Ok(Vec::new()) }
    async fn expire_foresights(&self, _before: chrono::DateTime<chrono::Utc>) -> Result<usize> { Ok(0) }
    async fn get_foresights_for_note(&self, _note_id: &str) -> Result<Vec<ForesightSignal>> { Ok(Vec::new()) }

    // --- Episodes (Phase 2B.1) ---

    async fn upsert_episode(&self, _episode: &Episode) -> Result<()> { Ok(()) }
    async fn get_episode(&self, _id: &str) -> Result<Option<Episode>> { Ok(None) }
    async fn get_episodes_for_session(&self, _session_id: &str) -> Result<Vec<Episode>> { Ok(Vec::new()) }
    async fn add_note_to_episode(&self, _note_id: &str, _episode_id: &str) -> Result<()> { Ok(()) }
    async fn get_episode_for_note(&self, _note_id: &str) -> Result<Option<String>> { Ok(None) }
    async fn get_notes_for_episode(&self, _episode_id: &str) -> Result<Vec<String>> { Ok(Vec::new()) }

    // --- Profiles (Phase 2B.3) ---

    async fn upsert_profile(&self, _entity_id: &str, _note_id: &str) -> Result<()> { Ok(()) }
    async fn get_profile_note_id(&self, _entity_id: &str) -> Result<Option<String>> { Ok(None) }
    async fn get_all_profiles(&self) -> Result<Vec<(String, String)>> { Ok(Vec::new()) }

    // --- Lifecycle ---

    /// Initialize tables/schema if needed.
    async fn init(&self) -> Result<()>;
}
