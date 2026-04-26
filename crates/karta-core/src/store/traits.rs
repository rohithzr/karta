use async_trait::async_trait;

use crate::error::Result;
use crate::note::{
    AtomicFact, CrossEpisodeDigest, Episode, EpisodeDigest, ForesightSignal, MemoryNote,
};

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

    // --- Atomic Facts (Phase Next) ---

    /// Insert or update an atomic fact with its embedding.
    async fn upsert_fact(&self, _fact: &AtomicFact) -> Result<()> {
        Ok(())
    }

    /// Find the top-K most similar atomic facts by embedding.
    async fn find_similar_facts(
        &self,
        _embedding: &[f32],
        _top_k: usize,
        _exclude_source_note_ids: &[&str],
    ) -> Result<Vec<(AtomicFact, f32)>> {
        Ok(Vec::new())
    }

    /// Get all facts for a given source note.
    async fn get_facts_for_note(&self, _note_id: &str) -> Result<Vec<AtomicFact>> {
        Ok(Vec::new())
    }
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
    async fn get_evolution_history(
        &self,
        note_id: &str,
    ) -> Result<Vec<crate::note::EvolutionRecord>>;

    // --- Dream state ---

    /// Record a dream run.
    async fn record_dream_run(&self, run: &crate::dream::DreamRun) -> Result<()>;

    /// Get the last dream cursor (timestamp of last processed note).
    async fn get_dream_cursor(&self) -> Result<Option<chrono::DateTime<chrono::Utc>>>;

    /// Update the dream cursor.
    async fn set_dream_cursor(&self, cursor: chrono::DateTime<chrono::Utc>) -> Result<()>;

    // --- Foresight signals (Phase 2B.2) ---

    async fn upsert_foresight(&self, _signal: &ForesightSignal) -> Result<()> {
        Ok(())
    }
    async fn get_active_foresights(&self) -> Result<Vec<ForesightSignal>> {
        Ok(Vec::new())
    }
    async fn expire_foresights(&self, _before: chrono::DateTime<chrono::Utc>) -> Result<usize> {
        Ok(0)
    }
    async fn get_foresights_for_note(&self, _note_id: &str) -> Result<Vec<ForesightSignal>> {
        Ok(Vec::new())
    }

    // --- Episodes (Phase 2B.1) ---

    async fn upsert_episode(&self, _episode: &Episode) -> Result<()> {
        Ok(())
    }
    async fn get_episode(&self, _id: &str) -> Result<Option<Episode>> {
        Ok(None)
    }
    async fn get_episodes_for_session(&self, _session_id: &str) -> Result<Vec<Episode>> {
        Ok(Vec::new())
    }
    async fn add_note_to_episode(&self, _note_id: &str, _episode_id: &str) -> Result<()> {
        Ok(())
    }
    async fn get_episode_for_note(&self, _note_id: &str) -> Result<Option<String>> {
        Ok(None)
    }
    async fn get_notes_for_episode(&self, _episode_id: &str) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    // --- Profiles (Phase 2B.3) ---

    async fn upsert_profile(&self, _entity_id: &str, _note_id: &str) -> Result<()> {
        Ok(())
    }
    async fn get_profile_note_id(&self, _entity_id: &str) -> Result<Option<String>> {
        Ok(None)
    }
    async fn get_all_profiles(&self) -> Result<Vec<(String, String)>> {
        Ok(Vec::new())
    }

    // --- Episode Digests (Phase Next) ---

    async fn upsert_episode_digest(&self, _digest: &EpisodeDigest) -> Result<()> {
        Ok(())
    }
    async fn get_episode_digest(&self, _episode_id: &str) -> Result<Option<EpisodeDigest>> {
        Ok(None)
    }
    async fn get_all_episode_digests(&self) -> Result<Vec<EpisodeDigest>> {
        Ok(Vec::new())
    }
    /// Get episode IDs that have no digest yet (for incremental dream processing).
    async fn get_undigested_episode_ids(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    // --- Cross-Episode Digests ---

    async fn upsert_cross_episode_digest(&self, _digest: &CrossEpisodeDigest) -> Result<()> {
        Ok(())
    }
    async fn get_all_cross_episode_digests(&self) -> Result<Vec<CrossEpisodeDigest>> {
        Ok(Vec::new())
    }

    // --- Atomic Fact Metadata (Phase Next) ---

    async fn record_fact(
        &self,
        _fact_id: &str,
        _source_note_id: &str,
        _ordinal: u32,
        _subject: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }
    async fn get_facts_by_subject(&self, _subject: &str) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    // --- Episode Links (Phase Next) ---

    async fn add_episode_link(
        &self,
        _from_id: &str,
        _to_id: &str,
        _link_type: &str,
        _entity: Option<&str>,
        _reason: &str,
    ) -> Result<()> {
        Ok(())
    }
    /// Returns (linked_episode_id, link_type, entity).
    async fn get_episode_links(
        &self,
        _episode_id: &str,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        Ok(Vec::new())
    }
    /// Get all episode IDs linked to a given entity via entity_continuity.
    async fn get_episodes_for_entity(&self, _entity: &str) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    // --- Lifecycle ---

    /// Initialize tables/schema if needed.
    async fn init(&self) -> Result<()>;

    /// Get current schema metadata (version, applied/pending migrations).
    async fn get_schema_meta(&self) -> Result<crate::migrate::SchemaMeta> {
        Ok(crate::migrate::SchemaMeta::new(
            0,
            Vec::new(),
            Vec::new(),
            vec!["Migration tracking unavailable for this graph store".to_string()],
        ))
    }

    // --- Procedural Rules (Issue #6) ---

    async fn upsert_procedural_rule(&self, _rule: &crate::rules::ProceduralRule) -> Result<()> {
        Ok(())
    }
    async fn list_procedural_rules(&self) -> Result<Vec<crate::rules::ProceduralRule>> {
        Ok(Vec::new())
    }
    async fn disable_procedural_rule(&self, _rule_id: &str) -> Result<()> {
        Ok(())
    }
    async fn increment_rule_fire_count(&self, _rule_id: &str) -> Result<()> {
        Ok(())
    }
}
