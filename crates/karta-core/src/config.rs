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
    /// ACTIVATE cognitive-retrieval pipeline: ACT-R + Hebbian + PAS + RRF.
    /// When enabled, supersedes the additive scalar scorer in `search_wide()`.
    #[serde(default)]
    pub activate: ActivateConfig,
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
            activate: ActivateConfig::default(),
        }
    }
}

// ─── ACTIVATE pipeline configuration ────────────────────────────────────────

/// Configuration for the ACTIVATE 6-phase cognitive retrieval pipeline.
///
/// Feature-flagged via `enabled`. Also honours the `KARTA_ACTIVATE_ENABLED`
/// environment variable so benchmarks can toggle without editing TOML.
///
/// Custom `Deserialize` impl: after parsing, any per-mode channel-weight
/// map the user supplied is unioned over the default matrix so a partial
/// TOML override (e.g. just tweaking `channel_weights.Standard.ann`)
/// preserves weights for the other modes.
#[derive(Debug, Clone, Serialize)]
pub struct ActivateConfig {
    /// Master switch. Defaults to `false` so the existing `search_wide()`
    /// scalar scorer remains the production path until this is validated.
    pub enabled: bool,
    /// ACT-R base-level learning decay rate. Default 0.5 (Anderson 2004).
    pub act_r_decay_d: f64,
    /// Drop notes below this base-level activation from the ACT-R channel.
    pub act_r_min_activation: f64,
    /// Hebbian strengthening: per-retrieval weight increment on co-activated semantic links.
    pub hebbian_weight_step: f32,
    /// Upper bound on Hebbian link weight to prevent runaway.
    pub hebbian_max_weight: f32,
    /// Co-activation channel: top-K weight-sorted neighbors to pull per anchor.
    pub hebbian_neighbors_per_anchor: usize,
    /// Reciprocal Rank Fusion constant. 60 is the canonical Cormack 2009 value.
    pub rrf_k: f32,
    /// PAS sequential walk radius (turns in each direction) for Temporal queries.
    pub pas_window: usize,
    /// Fraction of queries that run phase_trace writes. 1.0 = every query.
    pub trace_sample_rate: f32,
    /// Anchor cap: how many top ANN hits are used as seeds for co-activation
    /// and integration BFS.
    pub anchor_top_k: usize,
    /// Facts channel: minimum atomic-fact similarity score to expand to the parent note.
    pub facts_min_score: f32,
    /// Integration BFS cap: max neighbors accumulated before early termination.
    pub integration_bfs_cap: usize,
    /// Per-QueryMode channel weight overrides. Key = channel name.
    /// Channels: "ann", "keyword", "hebbian", "actr", "integration", "rerank",
    /// "pas", "facts", "foresight", "profile".
    pub channel_weights: HashMap<String, HashMap<String, f32>>,
}

fn default_activate_channel_weights() -> HashMap<String, HashMap<String, f32>> {
    use crate::read::QueryMode;

    fn mk(pairs: &[(&str, f32)]) -> HashMap<String, f32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
    let mut m = HashMap::new();
    m.insert(
        QueryMode::Standard.as_str().into(),
        mk(&[
            ("ann", 1.0),
            ("keyword", 0.5),
            ("hebbian", 0.7),
            ("actr", 0.3),
            ("integration", 0.5),
            ("rerank", 1.0),
            ("pas", 0.0),
            ("facts", 0.6),
            ("foresight", 0.4),
            ("profile", 1.2),
        ]),
    );
    m.insert(
        QueryMode::Recency.as_str().into(),
        mk(&[
            ("ann", 0.6),
            ("keyword", 0.4),
            ("hebbian", 0.3),
            ("actr", 1.2),
            ("integration", 0.3),
            ("rerank", 0.6),
            ("pas", 0.0),
            ("facts", 0.4),
            ("foresight", 0.8),
            ("profile", 0.8),
        ]),
    );
    m.insert(
        QueryMode::Breadth.as_str().into(),
        mk(&[
            ("ann", 1.0),
            ("keyword", 0.5),
            ("hebbian", 1.0),
            ("actr", 0.3),
            ("integration", 0.8),
            ("rerank", 0.8),
            ("pas", 0.0),
            ("facts", 0.5),
            ("foresight", 0.4),
            ("profile", 1.0),
        ]),
    );
    m.insert(
        QueryMode::Computation.as_str().into(),
        mk(&[
            ("ann", 0.8),
            ("keyword", 0.8),
            ("hebbian", 0.4),
            ("actr", 0.3),
            ("integration", 0.6),
            ("rerank", 1.2),
            ("pas", 0.0),
            ("facts", 1.0),
            ("foresight", 0.5),
            ("profile", 0.8),
        ]),
    );
    m.insert(
        QueryMode::Temporal.as_str().into(),
        mk(&[
            ("ann", 0.3),
            ("keyword", 0.3),
            ("hebbian", 0.0),
            ("actr", 1.0),
            ("integration", 0.2),
            ("rerank", 0.0),
            ("pas", 1.5),
            ("facts", 0.2),
            ("foresight", 0.3),
            ("profile", 0.4),
        ]),
    );
    // Existence ("Have I X?" / "Did I ever X?" / contradiction checks):
    // - hebbian: 0.0 — Hebbian boosts consistent co-activated clusters, which is
    //   exactly what suppresses contradicting outlier notes in Existence queries.
    // - facts: 1.3 — the contradicting evidence is typically captured as an
    //   atomic fact; boost that channel to surface the "I have never X" counter-fact.
    // - integration: 1.0 — structural graph neighbours (rather than co-activated
    //   ones) are more likely to connect to contradicting notes.
    m.insert(
        QueryMode::Existence.as_str().into(),
        mk(&[
            ("ann", 1.0),
            ("keyword", 0.8),
            ("hebbian", 0.0),
            ("actr", 0.5),
            ("integration", 1.0),
            ("rerank", 1.2),
            ("pas", 0.0),
            ("facts", 1.3),
            ("foresight", 0.5),
            ("profile", 1.0),
        ]),
    );
    m
}

fn default_anchor_top_k() -> usize {
    10
}
fn default_facts_min_score() -> f32 {
    0.3
}
fn default_integration_bfs_cap() -> usize {
    64
}

impl Default for ActivateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            act_r_decay_d: 0.5,
            act_r_min_activation: -0.5,
            hebbian_weight_step: 0.05,
            hebbian_max_weight: 3.0,
            hebbian_neighbors_per_anchor: 5,
            rrf_k: 60.0,
            pas_window: 6,
            trace_sample_rate: 1.0,
            anchor_top_k: default_anchor_top_k(),
            facts_min_score: default_facts_min_score(),
            integration_bfs_cap: default_integration_bfs_cap(),
            channel_weights: default_activate_channel_weights(),
        }
    }
}

// Helper shadow struct for field-level defaults in the custom Deserialize
// impl below. Every field has a sensible default so a partial TOML stanza
// (e.g. `[read.activate] enabled = true`) still deserializes cleanly.
#[derive(Deserialize)]
struct ActivateConfigShadow {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_act_r_decay_d")]
    act_r_decay_d: f64,
    #[serde(default = "default_act_r_min_activation")]
    act_r_min_activation: f64,
    #[serde(default = "default_hebbian_weight_step")]
    hebbian_weight_step: f32,
    #[serde(default = "default_hebbian_max_weight")]
    hebbian_max_weight: f32,
    #[serde(default = "default_hebbian_neighbors_per_anchor")]
    hebbian_neighbors_per_anchor: usize,
    #[serde(default = "default_rrf_k")]
    rrf_k: f32,
    #[serde(default = "default_pas_window")]
    pas_window: usize,
    #[serde(default = "default_trace_sample_rate")]
    trace_sample_rate: f32,
    #[serde(default = "default_anchor_top_k")]
    anchor_top_k: usize,
    #[serde(default = "default_facts_min_score")]
    facts_min_score: f32,
    #[serde(default = "default_integration_bfs_cap")]
    integration_bfs_cap: usize,
    #[serde(default)]
    channel_weights: HashMap<String, HashMap<String, f32>>,
}

fn default_act_r_decay_d() -> f64 {
    0.5
}
fn default_act_r_min_activation() -> f64 {
    -0.5
}
fn default_hebbian_weight_step() -> f32 {
    0.05
}
fn default_hebbian_max_weight() -> f32 {
    3.0
}
fn default_hebbian_neighbors_per_anchor() -> usize {
    5
}
fn default_rrf_k() -> f32 {
    60.0
}
fn default_pas_window() -> usize {
    6
}
fn default_trace_sample_rate() -> f32 {
    1.0
}

impl<'de> Deserialize<'de> for ActivateConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let shadow = ActivateConfigShadow::deserialize(deserializer)?;
        // Union user-provided per-mode maps over the default matrix so a
        // partial override preserves defaults for the other modes (and any
        // unspecified channels within a mode).
        let mut merged = default_activate_channel_weights();
        for (mode_key, user_map) in shadow.channel_weights {
            let entry = merged.entry(mode_key).or_default();
            for (ch, w) in user_map {
                entry.insert(ch, w);
            }
        }
        Ok(Self {
            enabled: shadow.enabled,
            act_r_decay_d: shadow.act_r_decay_d,
            act_r_min_activation: shadow.act_r_min_activation,
            hebbian_weight_step: shadow.hebbian_weight_step,
            hebbian_max_weight: shadow.hebbian_max_weight,
            hebbian_neighbors_per_anchor: shadow.hebbian_neighbors_per_anchor,
            rrf_k: shadow.rrf_k,
            pas_window: shadow.pas_window,
            trace_sample_rate: shadow.trace_sample_rate,
            anchor_top_k: shadow.anchor_top_k,
            facts_min_score: shadow.facts_min_score,
            integration_bfs_cap: shadow.integration_bfs_cap,
            channel_weights: merged,
        })
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
    /// ACTIVATE: archive notes whose ACT-R base-level activation drops below
    /// this floor (and whose age exceeds `decay_half_life_days`).
    #[serde(default = "default_actr_floor")]
    pub actr_decay_floor: f32,
    /// ACTIVATE: multiplicative decay applied to semantic-link weights on
    /// every sweep. Floored at 1.0 so links never fall below their initial weight.
    #[serde(default = "default_link_decay")]
    pub link_weight_decay: f32,
}

fn default_actr_floor() -> f32 {
    -1.0
}
fn default_link_decay() -> f32 {
    0.99
}

impl Default for ForgetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            decay_half_life_days: 90.0,
            archive_threshold: 0.1,
            sweep_on_dream: true,
            actr_decay_floor: default_actr_floor(),
            link_weight_decay: default_link_decay(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A partial `channel_weights` override must not wipe defaults for other
    /// modes or untouched channels within the overridden mode.
    #[test]
    fn channel_weights_partial_override_merges_with_defaults() {
        let toml_str = r#"
            enabled = true
            [channel_weights.Standard]
            ann = 2.5
        "#;
        let cfg: ActivateConfig = toml::from_str(toml_str).expect("parse");
        assert!(cfg.enabled);

        // Other modes still populated with defaults
        for mode in ["Recency", "Breadth", "Computation", "Temporal", "Existence"] {
            let m = cfg
                .channel_weights
                .get(mode)
                .unwrap_or_else(|| panic!("missing default map for {}", mode));
            assert!(!m.is_empty(), "{} map was dropped", mode);
        }

        // Standard: override applied, remaining channels retain defaults
        let std_map = cfg.channel_weights.get("Standard").expect("Standard map");
        assert_eq!(std_map.get("ann").copied(), Some(2.5));
        assert!(
            std_map.contains_key("keyword"),
            "default channels preserved"
        );
        assert!(std_map.contains_key("rerank"), "default channels preserved");
    }

    /// Defaults survive when TOML omits channel_weights entirely.
    #[test]
    fn channel_weights_missing_yields_defaults() {
        let cfg: ActivateConfig = toml::from_str("enabled = true").expect("parse");
        assert!(cfg.enabled);
        for mode in [
            "Standard",
            "Recency",
            "Breadth",
            "Computation",
            "Temporal",
            "Existence",
        ] {
            assert!(cfg.channel_weights.contains_key(mode), "missing {}", mode);
        }
    }

    /// Missing ACTIVATE-added scalar fields fall back to the documented defaults.
    #[test]
    fn scalar_fields_default_when_missing() {
        let cfg: ActivateConfig = toml::from_str("enabled = true").expect("parse");
        assert_eq!(cfg.anchor_top_k, 10);
        assert!((cfg.facts_min_score - 0.3).abs() < 1e-6);
        assert_eq!(cfg.integration_bfs_cap, 64);
    }

    /// Existence-mode channel weights were rebalanced to stop Hebbian
    /// co-activation from suppressing contradicting outlier notes (BEAM Conv 1
    /// Q4 regression; see PR #3).
    #[test]
    fn existence_channel_weights_rebalanced_for_contradictions() {
        let cfg = ActivateConfig::default();
        let existence = cfg
            .channel_weights
            .get("Existence")
            .expect("Existence map present");
        assert_eq!(existence.get("hebbian").copied(), Some(0.0));
        assert_eq!(existence.get("facts").copied(), Some(1.3));
        assert_eq!(existence.get("integration").copied(), Some(1.0));
    }
}
