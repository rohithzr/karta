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
    /// The data's "now" at the moment this note was generated. Non-optional —
    /// every note carries one. Live ingest sets it from `Utc::now()`; replay
    /// sets it from the source data's parsed reference time. Distinct from
    /// `created_at`, which is the ingestion (row-write) time.
    #[serde(default = "Utc::now")]
    pub source_timestamp: DateTime<Utc>,
    /// Optional session identifier — the conversation/scope this note belongs
    /// to. `None` for one-shot adds (e.g. quick scripts, smoke tests). Used
    /// by episode segmentation and any caller that wants to scope a query.
    #[serde(default)]
    pub session_id: Option<String>,
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
            source_timestamp: now,
            session_id: None,
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
    /// Atomic fact extracted from a note.
    Fact {
        source_note_id: String,
    },
    /// Episode digest produced by dream engine.
    Digest {
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

/// Retrieved memory context for a query, without calling any LLM.
///
/// This is the "retrieve only" output Karta produces — the caller is
/// responsible for feeding `context` (plus their own prompt, their own
/// model, their own generation settings) into whatever downstream LLM
/// they want. Separates retrieval from answer composition so callers
/// keep full control over the generation step.
#[derive(Debug, Clone, Serialize)]
pub struct FetchedMemories {
    /// Original query string, echoed back for convenience.
    pub query: String,
    /// Assembled, LLM-ready context string. Includes the optional EVENTS
    /// block (for temporal/computation queries) plus the formatted notes
    /// with provenance markers. Callers can drop this directly into a
    /// synthesis prompt or post-process as they wish.
    pub context: String,
    /// Raw notes in the order they appear in `context`, including any
    /// contradiction-injected source notes at the tail.
    pub notes: Vec<MemoryNote>,
    /// IDs of notes used in `context` (parallel to `notes`).
    pub note_ids: Vec<String>,
    /// Query classification mode used for retrieval, as a string (e.g.
    /// "Breadth", "Temporal"). Useful for telemetry / debugging.
    pub query_mode: String,
    /// Number of contradiction source notes force-injected into `notes`.
    pub contradiction_injected: usize,
    /// Best reranker relevance score across the retrieved set. `None` if
    /// the reranker is disabled. Callers can use this as an abstention
    /// signal before spending LLM cost on composition.
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
    /// Atomic facts extracted from the content (1-5 discrete statements).
    #[serde(default)]
    pub atomic_facts: Vec<AtomicFactExtraction>,
}

/// A single atomic fact as extracted by the LLM (before embedding/storage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicFactExtraction {
    pub content: String,
    pub subject: Option<String>,
    /// ISO-8601 UTC or null. Null iff `occurred_end` is also null.
    #[serde(default)]
    pub occurred_start: Option<DateTime<Utc>>,
    /// ISO-8601 UTC or null. Null iff `occurred_start` is also null.
    #[serde(default)]
    pub occurred_end: Option<DateTime<Utc>>,
    /// Discrete confidence band (see STEP1.5 resolved decision #4).
    #[serde(default = "default_confidence_band")]
    pub occurred_confidence: crate::read::temporal::ConfidenceBand,
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

// ─── Atomic Facts (Phase Next) ─────────────────────────────────────────────

/// A single, independently verifiable statement extracted from a MemoryNote.
/// Each fact gets its own embedding in a dedicated LanceDB table for fine-grained retrieval.
/// Intentionally lightweight: the fact IS the embedding unit. No context/keywords/tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicFact {
    pub id: String,
    /// The atomic statement text.
    pub content: String,
    /// ID of the parent MemoryNote this fact was extracted from.
    pub source_note_id: String,
    /// Position within the source note's fact list (0-indexed, preserves micro-ordering).
    pub ordinal: u32,
    /// Primary entity or topic for aggregation grouping (e.g., "Flask", "budget", "Coco").
    pub subject: Option<String>,
    /// Embedding vector (stored in LanceDB atomic_facts table).
    #[serde(skip)]
    pub embedding: Vec<f32>,
    pub created_at: DateTime<Utc>,
    /// Inherited from parent MemoryNote.source_timestamp at write time.
    /// Enables fact-level recency on replay data.
    #[serde(default = "Utc::now")]
    pub source_timestamp: DateTime<Utc>,
    /// Lower bound (inclusive) of when the asserted event occurred.
    /// `None` iff `occurred_end` is also None (invariant #1).
    #[serde(default)]
    pub occurred_start: Option<DateTime<Utc>>,
    /// Upper bound (exclusive). Half-open semantics.
    #[serde(default)]
    pub occurred_end: Option<DateTime<Utc>>,
    /// Discrete confidence band. `None` iff both bounds are None (invariant #4).
    #[serde(default = "default_confidence_band")]
    pub occurred_confidence: crate::read::temporal::ConfidenceBand,
}

fn default_confidence_band() -> crate::read::temporal::ConfidenceBand {
    crate::read::temporal::ConfidenceBand::None
}

#[derive(Debug, thiserror::Error)]
pub enum OccurredValidationError {
    #[error("invariant #1 broken: start and end must both be Some or both be None")]
    UnpairedBounds,
    #[error("invariant #2 broken: end ({end}) must be strictly greater than start ({start})")]
    EndNotAfterStart { start: DateTime<Utc>, end: DateTime<Utc> },
    #[error("invariant #4 broken: confidence == None iff both bounds are None (got conf={conf:?}, bounds paired={paired})")]
    ConfidenceBoundsMismatch { conf: crate::read::temporal::ConfidenceBand, paired: bool },
}

impl AtomicFact {
    pub fn new(content: String, source_note_id: String, ordinal: u32) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            source_note_id,
            ordinal,
            subject: None,
            embedding: Vec::new(),
            created_at: Utc::now(),
            source_timestamp: Utc::now(),
            occurred_start: None,
            occurred_end: None,
            occurred_confidence: crate::read::temporal::ConfidenceBand::None,
        }
    }

    /// Enforce the 4 invariants from STEP1.5 decisions (null pairing,
    /// end-after-start, closed-set confidence via type, confidence-bounds
    /// pairing). Invariant #3 (confidence ∈ [0,1]) is enforced at the
    /// type level by `ConfidenceBand` — impossible to violate here.
    pub fn validate_occurred(&self) -> Result<(), OccurredValidationError> {
        use crate::read::temporal::ConfidenceBand;

        let both_some = self.occurred_start.is_some() && self.occurred_end.is_some();
        let both_none = self.occurred_start.is_none() && self.occurred_end.is_none();

        if !(both_some || both_none) {
            return Err(OccurredValidationError::UnpairedBounds);
        }

        if let (Some(s), Some(e)) = (self.occurred_start, self.occurred_end) {
            if e <= s {
                return Err(OccurredValidationError::EndNotAfterStart { start: s, end: e });
            }
        }

        let conf_is_none = self.occurred_confidence == ConfidenceBand::None;
        if conf_is_none != both_none {
            return Err(OccurredValidationError::ConfidenceBoundsMismatch {
                conf: self.occurred_confidence,
                paired: both_some,
            });
        }

        Ok(())
    }
}

// ─── Episode Digests (Phase Next) ──────────────────────────────────────────

/// Structured metadata produced by dream-time analysis of an episode.
/// Contains pre-computed entities, date ranges, aggregations, and topic ordering.
/// The digest_text is also stored as a MemoryNote for ANN searchability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeDigest {
    pub id: String,
    pub episode_id: String,
    /// Named entities with their types and mention counts.
    pub entities: Vec<EntityMention>,
    /// Date range covered by the episode (extracted from note content, not timestamps).
    pub date_range: Option<DateRange>,
    /// Pre-computed aggregation summaries (e.g., "5 movies discussed: [list]").
    pub aggregations: Vec<AggregationEntry>,
    /// Topics in the order they appeared in the episode.
    pub topic_sequence: Vec<String>,
    /// Retrieval-optimized summary text (also stored as a MemoryNote).
    pub digest_text: String,
    /// ID of the MemoryNote storing the digest_text (for vector search).
    pub digest_note_id: Option<String>,
    /// Dated events extracted from the episode content. Enables direct
    /// lookup for date-arithmetic questions (e.g. "days between A and B").
    /// Events with an unknown date are kept with date = None so they still
    /// contribute to ordering queries via source_turn.
    #[serde(default)]
    pub events: Vec<TimedEvent>,
    pub created_at: DateTime<Utc>,
}

/// Structured metadata produced by dream-time analysis across multiple
/// episodes. Complements the per-episode digest with a deduped, normalized
/// view of entities, aggregations, and events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEpisodeDigest {
    pub id: String,
    /// Opaque scope identifier — typically the karta instance or user scope.
    pub scope_id: String,
    /// Entity values tracked across episodes.
    pub entity_timeline: Vec<EntityTimelineEntry>,
    /// Aggregations that span episodes (e.g. "12 books read across all conversations").
    pub cross_aggregations: Vec<AggregationEntry>,
    /// Merged chronological events across all episodes, deduped.
    #[serde(default)]
    pub events: Vec<TimedEvent>,
    /// Overall topic progression across episodes.
    pub topic_progression: Vec<String>,
    /// Retrieval-optimized summary text.
    pub digest_text: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTimelineEntry {
    pub name: String,
    pub entity_type: String,
    /// Ordered value changes for this entity across episodes.
    pub changes: Vec<EntityTimelineChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTimelineChange {
    pub episode_id: String,
    pub value: String,
}

/// A specific dated event extracted from a conversation. Used to answer
/// date-arithmetic questions directly ("days between A and B") without
/// relying on the synthesis model to infer dates from free-form note text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedEvent {
    /// Human-readable description of the event (e.g. "finished transaction
    /// management features", "met with Wyatt expressing skepticism").
    pub description: String,
    /// ISO YYYY-MM-DD date of the event, or None if undated.
    #[serde(default)]
    pub date: Option<String>,
    /// turn_index of the source note, for provenance / stable ordering.
    #[serde(default)]
    pub source_turn: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMention {
    pub name: String,
    /// Type: "person", "tool", "framework", "project", "date", "number", "other".
    pub entity_type: String,
    pub count: u32,
    /// Most recent value if this entity was updated during the episode.
    pub latest_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub earliest: String, // ISO date string
    pub latest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationEntry {
    /// Human-readable label (e.g., "movies discussed").
    pub label: String,
    pub count: u32,
    /// The individual items (e.g., ["Inception", "Matrix", "Coco"]).
    pub items: Vec<String>,
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
