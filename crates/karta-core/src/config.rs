use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KartaConfig {
    pub storage: StorageConfig,
    pub llm: LlmConfig,
    pub read: ReadConfig,
    pub write: WriteConfig,
    pub dream: DreamConfig,
    pub episode: EpisodeConfig,
    pub forget: ForgetConfig,
    pub reranker: crate::rerank::RerankerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeConfig {
    /// Whether episode segmentation is enabled.
    pub enabled: bool,
    /// Time gap in seconds that forces a new episode boundary.
    pub time_gap_threshold_secs: i64,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            time_gap_threshold_secs: 1800, // 30 minutes
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Directory for embedded storage (SQLite graph store).
    pub data_dir: String,
    /// Optional URI for LanceDB vector store (e.g. "gs://bucket/path").
    /// If not set, defaults to "{data_dir}/lance".
    pub lance_uri: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: ".karta".into(),
            lance_uri: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmModelRef {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Default model for all operations.
    pub default: LlmModelRef,
    /// Per-operation overrides. Keys: "write.attributes", "write.linking",
    /// "write.evolve", "read.synthesize", "dream.deduction", etc.
    #[serde(default)]
    pub overrides: HashMap<String, LlmModelRef>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            default: LlmModelRef {
                provider: "openai".into(),
                model: "gpt-4o-mini".into(),
                base_url: None,
            },
            overrides: HashMap::new(),
        }
    }
}

impl LlmConfig {
    /// Get the model ref for a specific operation, falling back to default.
    pub fn model_for(&self, operation: &str) -> &LlmModelRef {
        self.overrides.get(operation).unwrap_or(&self.default)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadConfig {
    /// Weight for temporal recency in scoring (0.0 = pure similarity, 1.0 = heavy recency bias).
    pub recency_weight: f32,
    /// Half-life for temporal decay in days. A note this many days old gets 50% recency score.
    pub recency_half_life_days: f64,
    /// Boost for notes with active foresight signals.
    pub foresight_boost: f32,
    /// Weight for graph connectivity (PageRank-lite). 0.0 = disabled.
    pub graph_weight: f32,
    /// Max graph traversal depth (1 = current behavior, 2-3 = multi-hop).
    pub max_hop_depth: usize,
    /// Decay factor per hop (0.5 = each hop worth half the previous).
    pub hop_decay_factor: f32,
    /// Minimum similarity score for the best result before the system abstains.
    /// If no note scores above this threshold, the system says "no relevant information."
    pub abstention_threshold: f32,
    /// Top-K multiplier for summarization queries (detected by keywords).
    /// Summarization needs broader coverage than factual queries.
    pub summarization_top_k_multiplier: usize,
    /// Whether to use two-level episode retrieval (ANN on episode narratives → drill into notes).
    pub episode_retrieval_enabled: bool,
    /// Max episodes to drill into per query.
    pub max_episode_drilldowns: usize,
    /// Max notes to include per drilled episode.
    pub max_notes_per_episode: usize,
    /// Min ANN score for an episode narrative to trigger drilldown.
    pub episode_drilldown_min_score: f32,
    /// Whether to search atomic facts alongside notes.
    pub fact_retrieval_enabled: bool,
    /// Score boost for notes found via fact match.
    pub fact_match_boost: f32,
}

impl Default for ReadConfig {
    fn default() -> Self {
        Self {
            recency_weight: 0.15,
            recency_half_life_days: 30.0,
            foresight_boost: 0.1,
            graph_weight: 0.05,
            max_hop_depth: 2,
            hop_decay_factor: 0.5,
            abstention_threshold: 0.20,
            summarization_top_k_multiplier: 3,
            episode_retrieval_enabled: true,
            max_episode_drilldowns: 3,
            max_notes_per_episode: 10,
            episode_drilldown_min_score: 0.25,
            fact_retrieval_enabled: true,
            fact_match_boost: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteConfig {
    /// How many similar notes to consider for linking.
    pub top_k_candidates: usize,
    /// Minimum cosine similarity to be a link candidate.
    pub similarity_threshold: f32,
    /// Whether to retroactively update linked notes' context.
    pub evolve_linked_notes: bool,
    /// Max evolutions before a note is flagged for consolidation instead.
    pub max_evolutions_per_note: usize,
    /// Default TTL in days for foresight signals when no explicit expiry is extracted.
    pub foresight_default_ttl_days: i64,
    /// Whether to extract and store atomic facts during note ingestion.
    pub extract_atomic_facts: bool,
    /// Maximum number of atomic facts to extract per note.
    pub max_facts_per_note: usize,
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            top_k_candidates: 5,
            similarity_threshold: 0.3,
            evolve_linked_notes: true,
            max_evolutions_per_note: 5,
            foresight_default_ttl_days: 90,
            extract_atomic_facts: true,
            max_facts_per_note: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Minimum confidence for a dream to be written back as a note.
    pub write_threshold: f32,
    /// Max notes to feed into one dreaming prompt.
    pub max_notes_per_prompt: usize,
    /// Which dream types to run.
    pub enabled_types: Vec<String>,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            write_threshold: 0.65,
            max_notes_per_prompt: 8,
            enabled_types: vec![
                "deduction".into(),
                "induction".into(),
                "abduction".into(),
                "consolidation".into(),
                "contradiction".into(),
                "episode_digest".into(),
                "cross_episode_digest".into(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetConfig {
    /// Whether forgetting is enabled.
    pub enabled: bool,
    /// Half-life in days for access-based decay. Notes not accessed in this
    /// many days get 50% decay score.
    pub decay_half_life_days: f64,
    /// Notes with decay score below this threshold get archived.
    pub archive_threshold: f32,
    /// Whether to run forgetting sweep at the end of each dream pass.
    pub sweep_on_dream: bool,
}

impl Default for ForgetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            decay_half_life_days: 90.0,
            archive_threshold: 0.1,
            sweep_on_dream: true,
        }
    }
}
