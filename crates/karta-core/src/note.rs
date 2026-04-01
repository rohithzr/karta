use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single memory note in the Karta knowledge graph.
///
/// Each note is a structured unit of knowledge — not just raw text, but
/// enriched with LLM-generated context, keywords, tags, and semantic
/// links to related notes. Notes evolve retroactively when new related
/// information arrives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNote {
    pub id: String,
    pub content: String,
    /// LLM-generated rich description capturing implications and deeper meaning.
    pub context: String,
    pub keywords: Vec<String>,
    pub tags: Vec<String>,
    /// IDs of semantically linked notes (bidirectional).
    pub links: Vec<String>,
    /// Embedding vector (populated by vector store).
    #[serde(skip)]
    pub embedding: Vec<f32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub evolution_history: Vec<EvolutionRecord>,
    /// Provenance: "observed" for direct input, "dream:{type}" for inferred.
    pub provenance: Provenance,
    /// Confidence score. 1.0 for observed facts, <1.0 for dream-derived.
    pub confidence: f32,
    /// Lifecycle status: Active, Deprecated, Superseded, or Archived.
    #[serde(default)]
    pub status: NoteStatus,
    /// Last time this note was retrieved or traversed.
    #[serde(default = "Utc::now")]
    pub last_accessed_at: DateTime<Utc>,
    /// Position of this message within its conversation/session (0-indexed).
    #[serde(default)]
    pub turn_index: Option<u32>,
    /// Original timestamp from source data (e.g., BEAM time_anchor parsed to DateTime).
    /// Distinct from `created_at` which is the ingestion time.
    #[serde(default)]
    pub source_timestamp: Option<DateTime<Utc>>,
}

impl MemoryNote {
    pub fn new(content: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            context: String::new(),
            keywords: Vec::new(),
            tags: Vec::new(),
            links: Vec::new(),
            embedding: Vec::new(),
            created_at: now,
            updated_at: now,
            evolution_history: Vec::new(),
            provenance: Provenance::Observed,
            confidence: 1.0,
            status: NoteStatus::Active,
            last_accessed_at: now,
            turn_index: None,
            source_timestamp: None,
        }
    }

    pub fn is_active(&self) -> bool {
        self.status == NoteStatus::Active
    }

    pub fn is_dream(&self) -> bool {
        matches!(self.provenance, Provenance::Dream { .. })
    }

    pub fn is_profile(&self) -> bool {
        matches!(self.provenance, Provenance::Profile { .. })
    }

    pub fn is_episode(&self) -> bool {
        matches!(self.provenance, Provenance::Episode { .. })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionRecord {
    /// ID of the note that triggered this evolution.
    pub triggered_by: String,
    /// Context before this evolution.
    pub previous_context: String,
    pub evolved_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Provenance {
    /// Directly observed (user input).
    Observed,
    /// Inferred by dream engine.
    Dream {
        dream_type: String,
        source_note_ids: Vec<String>,
        confidence: f32,
    },
    /// Entity profile built from consolidation dreams.
    Profile {
        entity_id: String,
    },
    /// Episode narrative synthesis.
    Episode {
        episode_id: String,
    },
}

/// Result of a similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub note: MemoryNote,
    pub score: f32,
    pub linked_notes: Vec<MemoryNote>,
}

/// Result of an ask() call, including the answer and retrieval metadata for debugging.
#[derive(Debug, Clone, Serialize)]
pub struct AskResult {
    /// The synthesized answer.
    pub answer: String,
    /// Query classification mode used for retrieval.
    pub query_mode: String,
    /// Number of unique notes used in synthesis (after dedup).
    pub notes_used: usize,
    /// IDs of notes used in synthesis.
    pub note_ids: Vec<String>,
    /// Number of contradiction source notes force-injected.
    pub contradiction_injected: usize,
    /// Whether the LLM flagged contradictory information.
    pub has_contradiction: bool,
    /// Best reranker relevance score (None if reranker disabled).
    pub reranker_best_score: Option<f32>,
}

/// A forward-looking statement extracted by the LLM during attribute generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForesightExtraction {
    pub content: String,
    /// Optional ISO 8601 date (YYYY-MM-DD) when this prediction/deadline expires.
    pub valid_until: Option<String>,
}

/// LLM-generated attributes for a note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteAttributes {
    pub context: String,
    pub keywords: Vec<String>,
    pub tags: Vec<String>,
    /// Forward-looking statements extracted from the content.
    #[serde(default)]
    pub foresight_signals: Vec<ForesightExtraction>,
}

/// LLM decision about whether to link two notes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkDecision {
    pub note_id: String,
    pub reason: String,
}

// ─── Foresight Signals (Phase 2B.2) ──────────────────────────────────────────

/// A forward-looking prediction with a validity window.
/// Extracted during ingestion or generated by abduction dreams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForesightSignal {
    pub id: String,
    pub content: String,
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub source_note_id: String,
    pub confidence: f32,
    pub status: ForesightStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ForesightStatus {
    Active,
    Expired,
    Fulfilled,
}

impl ForesightSignal {
    pub fn new(content: String, source_note_id: String, valid_until: Option<DateTime<Utc>>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            valid_from: Utc::now(),
            valid_until,
            source_note_id,
            confidence: 0.7,
            status: ForesightStatus::Active,
        }
    }

    pub fn is_active(&self) -> bool {
        self.status == ForesightStatus::Active
    }

    pub fn is_expired_at(&self, now: DateTime<Utc>) -> bool {
        self.valid_until.is_some_and(|until| now > until)
    }
}

// ─── Episodes (Phase 2B.1) ──────────────────────────────────────────────────

/// A thematically coherent group of notes from a conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub narrative: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub session_id: String,
    pub topic_tags: Vec<String>,
    pub note_ids: Vec<String>,
    /// ID of the narrative synthesis note stored in VectorStore.
    pub narrative_note_id: Option<String>,
}

impl Episode {
    pub fn new(session_id: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            narrative: String::new(),
            start_time: now,
            end_time: now,
            session_id,
            topic_tags: Vec::new(),
            note_ids: Vec::new(),
            narrative_note_id: None,
        }
    }
}

// ─── Note Lifecycle (Phase 3.1) ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NoteStatus {
    Active,
    Deprecated { by: String },
    Superseded { by: String },
    Archived,
}

impl Default for NoteStatus {
    fn default() -> Self {
        NoteStatus::Active
    }
}
