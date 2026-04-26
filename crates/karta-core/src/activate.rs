//! ACTIVATE: 6-phase cognitive retrieval pipeline.
//!
//! Fuses cognitive-science retrieval primitives with IR rank-fusion:
//!
//! - **A — Anchor**: ANN over note embeddings + atomic-fact embeddings;
//!   optional profile auto-include.
//! - **C — Co-activation (Hebbian)**: pulls weight-sorted neighbors of top
//!   anchors. Weights are bumped on co-retrieval ("fire together, wire
//!   together") in phase_trace.
//! - **T — Temporal decay (ACT-R BLL)**: Anderson (2004) base-level learning
//!   `B_i = ln(Σ t_k^{-d})` using the per-note `access_history` ring.
//! - **I — Integration**: multi-hop graph BFS from anchors; its own channel.
//! - **V — Vector reranker**: cross-encoder (Jina) scores the merged pool.
//! - **A — Aggregate (RRF)**: reciprocal rank fusion across all channels
//!   with per-QueryMode weights.
//!
//! Plus two cross-cutting concerns:
//! - **PAS** (prefix-aware sequential): Temporal-mode channel that walks
//!   the "follows" chain around the best anchor in its `session_id`.
//! - **Trace**: write-back that bumps `access_count` + Hebbian link weights
//!   for notes co-appearing in the returned top_k.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tracing::{debug, info, warn};

use crate::config::{ActivateConfig, ReadConfig};
use crate::error::Result;
use crate::llm::LlmProvider;
use crate::note::{MemoryNote, Provenance, SearchResult};
use crate::read::QueryMode;
use crate::rerank::{Reranker, RerankerConfig};
use crate::store::{GraphStore, VectorStore};

/// A single ranking channel: ordered list of note ids, best first.
///
/// Each phase emits one (or more) channel; RRF aggregates them by rank,
/// never by raw score, which makes score scales across channels irrelevant.
#[derive(Debug, Clone)]
pub struct Channel {
    pub name: &'static str,
    pub ranked: Vec<String>,
}

/// Debug snapshot of an ACTIVATE run — per-channel rank lists plus the
/// final fused results. Useful for BEAM-style channel-hit analytics.
#[derive(Debug, Clone)]
pub struct ActivateOutput {
    pub results: Vec<SearchResult>,
    pub mode: QueryMode,
    pub channels: Vec<Channel>,
}

/// 6-phase ACTIVATE pipeline engine. Owns references to stores + LLM +
/// reranker; borrows config. Independent of `ReadEngine` to keep the
/// legacy scalar path available as a fallback.
pub struct ActivateEngine {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    #[allow(dead_code)]
    llm: Arc<dyn LlmProvider>,
    reranker: Arc<dyn Reranker>,
    read_cfg: ReadConfig,
    rerank_cfg: RerankerConfig,
}

impl ActivateEngine {
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        llm: Arc<dyn LlmProvider>,
        reranker: Arc<dyn Reranker>,
        read_cfg: ReadConfig,
        rerank_cfg: RerankerConfig,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            llm,
            reranker,
            read_cfg,
            rerank_cfg,
        }
    }

    fn cfg(&self) -> &ActivateConfig {
        &self.read_cfg.activate
    }

    /// Run all six phases and return the top_k fused results.
    pub async fn activate(
        &self,
        query: &str,
        query_embedding: &[f32],
        mode: QueryMode,
        top_k: usize,
    ) -> Result<ActivateOutput> {
        info!(query = %query, ?mode, "ACTIVATE: start");

        // --- Phase 1: Anchor — ANN + facts + profiles ---
        let (ann_ch, facts_ch, profile_ch, foresight_ch, mut pool) =
            self.phase_anchor(query, query_embedding).await?;

        let anchor_ids: Vec<String> = ann_ch
            .ranked
            .iter()
            .take(self.cfg().anchor_top_k)
            .cloned()
            .collect();

        // --- Phase 2: Co-activation (Hebbian) ---
        let hebbian_ch = self.phase_coactivation(&anchor_ids).await?;
        self.extend_pool(&mut pool, &hebbian_ch.ranked).await?;

        // --- Phase 4: Integration (multi-hop BFS) ---
        let integration_ch = self.phase_integration(&anchor_ids).await?;
        self.extend_pool(&mut pool, &integration_ch.ranked).await?;

        // --- Phase 3: Temporal decay (ACT-R) ---
        let actr_ch = self.phase_actr(&pool);

        // --- Phase 5b: PAS (Temporal only) ---
        let pas_ch = if matches!(mode, QueryMode::Temporal) {
            let anchor = anchor_ids.first().cloned();
            match anchor {
                Some(a) => self.phase_pas(&a).await?,
                None => Channel {
                    name: "pas",
                    ranked: Vec::new(),
                },
            }
        } else {
            Channel {
                name: "pas",
                ranked: Vec::new(),
            }
        };
        self.extend_pool(&mut pool, &pas_ch.ranked).await?;

        // --- Phase 5: Vector reranker ---
        let rerank_ch = self.phase_rerank(query, &pool).await?;

        // --- Phase 6: Aggregate (RRF) ---
        let mut channels = vec![
            ann_ch,
            facts_ch,
            profile_ch,
            foresight_ch,
            hebbian_ch,
            integration_ch,
            actr_ch,
            pas_ch,
            rerank_ch,
        ];

        // Apply Temporal-fallback weight redistribution: if PAS is empty in
        // Temporal mode (no session_id on anchor) its 1.5 weight rolls into
        // ann + actr so we don't silently lose ranking signal.
        let mut weights = self.weights_for_mode(mode);
        if matches!(mode, QueryMode::Temporal) {
            let pas_empty = channels
                .iter()
                .find(|c| c.name == "pas")
                .map(|c| c.ranked.is_empty())
                .unwrap_or(true);
            if pas_empty {
                let pas_w = weights.remove("pas").unwrap_or(0.0);
                *weights.entry("ann".into()).or_insert(0.0) += pas_w * 0.333;
                *weights.entry("actr".into()).or_insert(0.0) += pas_w * 0.667;
                debug!("ACTIVATE: Temporal fallback — redistributing PAS weight");
            }
        }

        let fused = rrf(&channels, &weights, self.cfg().rrf_k);
        let fused_ids: Vec<String> = fused.iter().map(|(id, _)| id.clone()).collect();

        // Map ids back to MemoryNotes. Fetch the fused candidate list before
        // active filtering so inactive high-ranked notes do not reduce the
        // returned set below top_k when lower-ranked active notes are available.
        let id_refs: Vec<&str> = fused_ids.iter().map(|s| s.as_str()).collect();
        let mut notes_by_id: HashMap<String, MemoryNote> = self
            .vector_store
            .get_many(&id_refs)
            .await?
            .into_iter()
            .map(|n| (n.id.clone(), n))
            .collect();

        let mut results: Vec<SearchResult> = Vec::with_capacity(top_k.min(fused.len()));
        for (id, fused_score) in fused {
            if let Some(note) = notes_by_id.remove(&id)
                && note.is_active()
            {
                results.push(SearchResult {
                    note,
                    score: fused_score,
                    linked_notes: Vec::new(),
                });
                if results.len() >= top_k {
                    break;
                }
            }
        }

        // NOTE: Phase 7 (Trace write-back) is intentionally NOT called here.
        // `search_wide()` passes a deliberately wide top_k so the caller
        // (`ask()`) can rerank+truncate; running phase_trace on this wide
        // set would train activation state on notes that never reach the
        // user. The caller invokes `phase_trace()` explicitly on the final
        // truncated id set. See read.rs for the dispatch site.

        // Keep rank order intact — do NOT re-sort by score downstream (RRF
        // has already decided the order).
        channels.retain(|c| !c.ranked.is_empty());
        Ok(ActivateOutput {
            results,
            mode,
            channels,
        })
    }

    // ─── Phase 1: Anchor ───────────────────────────────────────────────

    async fn phase_anchor(
        &self,
        query: &str,
        query_embedding: &[f32],
    ) -> Result<(Channel, Channel, Channel, Channel, Vec<MemoryNote>)> {
        // ANN over notes
        let fetch_k = (self.rerank_cfg.max_rerank.max(20)) * 2;
        let direct = self
            .vector_store
            .find_similar(query_embedding, fetch_k, &[])
            .await?;

        let mut pool: Vec<MemoryNote> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut ann_ranked: Vec<String> = Vec::new();
        for (note, _sim) in &direct {
            if !note.is_active() {
                continue;
            }
            // Skip dream/digest/fact-provenance rows so anchors stay on
            // actual user notes (consistent with current search_wide).
            match &note.provenance {
                Provenance::Dream { .. } | Provenance::Digest { .. } | Provenance::Fact { .. } => {
                    continue;
                }
                _ => {}
            }
            if seen.insert(note.id.clone()) {
                ann_ranked.push(note.id.clone());
                pool.push(note.clone());
            }
        }

        // Facts channel — high-scoring atomic facts expand to their parent notes
        let mut facts_ranked: Vec<String> = Vec::new();
        if self.read_cfg.fact_retrieval_enabled {
            let fact_k = (fetch_k / 2).max(10);
            let facts_min_score = self.cfg().facts_min_score;
            if let Ok(hits) = self
                .vector_store
                .find_similar_facts(query_embedding, fact_k, &[])
                .await
            {
                for (fact, score) in hits {
                    if score < facts_min_score {
                        continue;
                    }
                    if seen.contains(&fact.source_note_id) {
                        // already ranked via ANN; keep first-seen order
                        continue;
                    }
                    if let Ok(Some(parent)) = self.vector_store.get(&fact.source_note_id).await
                        && parent.is_active()
                        && seen.insert(parent.id.clone())
                    {
                        facts_ranked.push(parent.id.clone());
                        pool.push(parent);
                    }
                }
            }
        }

        // Profile channel — entity-profile auto-include by token overlap
        let mut profile_ranked: Vec<String> = Vec::new();
        let query_lower = query.to_lowercase();
        let profiles = match self.graph_store.get_all_profiles().await {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "profile: store error");
                Vec::new()
            }
        };
        for (entity_id, note_id) in profiles {
            let tokens: Vec<&str> = entity_id
                .split(|c: char| c.is_whitespace() || c == '-' || c == '_')
                .filter(|t| t.len() >= 3)
                .collect();
            let matched = tokens
                .iter()
                .any(|t| query_lower.contains(&t.to_lowercase()));
            if !matched {
                continue;
            }
            if let Ok(Some(note)) = self.vector_store.get(&note_id).await
                && note.is_active()
                && seen.insert(note.id.clone())
            {
                profile_ranked.push(note.id.clone());
                pool.push(note);
            }
        }

        // Foresight channel — notes backing currently-active forward predictions
        let mut foresight_ranked: Vec<String> = Vec::new();
        let signals = match self.graph_store.get_active_foresights().await {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "foresight: store error");
                Vec::new()
            }
        };
        for s in signals {
            if seen.insert(s.source_note_id.clone()) {
                if let Ok(Some(note)) = self.vector_store.get(&s.source_note_id).await
                    && note.is_active()
                {
                    foresight_ranked.push(note.id.clone());
                    pool.push(note);
                }
            } else {
                foresight_ranked.push(s.source_note_id);
            }
        }

        Ok((
            Channel {
                name: "ann",
                ranked: ann_ranked,
            },
            Channel {
                name: "facts",
                ranked: facts_ranked,
            },
            Channel {
                name: "profile",
                ranked: profile_ranked,
            },
            Channel {
                name: "foresight",
                ranked: foresight_ranked,
            },
            pool,
        ))
    }

    // ─── Phase 2: Co-activation (Hebbian) ──────────────────────────────

    async fn phase_coactivation(&self, anchor_ids: &[String]) -> Result<Channel> {
        let per_anchor = self.cfg().hebbian_neighbors_per_anchor;
        let mut ranked: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = anchor_ids.iter().cloned().collect();
        for id in anchor_ids {
            let neighbors = match self
                .graph_store
                .get_links_with_weights(id, Some("semantic"))
                .await
            {
                Ok(n) => n,
                Err(e) => {
                    warn!(error = %e, "hebbian: store error");
                    Vec::new()
                }
            };
            for (nid, _w) in neighbors.into_iter().take(per_anchor) {
                if seen.insert(nid.clone()) {
                    ranked.push(nid);
                }
            }
        }
        Ok(Channel {
            name: "hebbian",
            ranked,
        })
    }

    // ─── Phase 3: Temporal decay (ACT-R BLL) ──────────────────────────

    fn phase_actr(&self, pool: &[MemoryNote]) -> Channel {
        let d = self.cfg().act_r_decay_d;
        let floor = self.cfg().act_r_min_activation;
        let now = Utc::now();

        let mut scored: Vec<(String, f64)> = pool
            .iter()
            .map(|n| {
                (
                    n.id.clone(),
                    actr_activation(&n.access_history, now, d, n.access_count),
                )
            })
            .filter(|(_, b)| *b >= floor)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Channel {
            name: "actr",
            ranked: scored.into_iter().map(|(id, _)| id).collect(),
        }
    }

    // ─── Phase 4: Integration (multi-hop BFS) ─────────────────────────

    async fn phase_integration(&self, anchor_ids: &[String]) -> Result<Channel> {
        use std::collections::VecDeque;
        let max_depth = self.read_cfg.max_hop_depth;
        let bfs_cap = self.cfg().integration_bfs_cap;
        let mut ranked: Vec<String> = Vec::new();
        let mut visited: HashSet<String> = anchor_ids.iter().cloned().collect();
        let mut q: VecDeque<(String, usize)> = VecDeque::new();
        for id in anchor_ids {
            q.push_back((id.clone(), 0));
        }
        while let Some((cur, depth)) = q.pop_front() {
            if depth >= max_depth {
                continue;
            }
            // Semantic-only: PAS already covers the "follows" chain. Mixing
            // them double-counts PAS neighbors in the fused ranking.
            let links = match self
                .graph_store
                .get_links_with_weights(&cur, Some("semantic"))
                .await
            {
                Ok(l) => l,
                Err(e) => {
                    warn!(error = %e, "integration: store error");
                    Vec::new()
                }
            };
            for (l, _w) in links {
                if visited.insert(l.clone()) {
                    ranked.push(l.clone());
                    q.push_back((l, depth + 1));
                    if ranked.len() > bfs_cap {
                        return Ok(Channel {
                            name: "integration",
                            ranked,
                        });
                    }
                }
            }
        }
        Ok(Channel {
            name: "integration",
            ranked,
        })
    }

    // ─── Phase 5: Vector reranker ─────────────────────────────────────

    async fn phase_rerank(&self, query: &str, pool: &[MemoryNote]) -> Result<Channel> {
        if !self.rerank_cfg.enabled || pool.is_empty() {
            return Ok(Channel {
                name: "rerank",
                ranked: Vec::new(),
            });
        }
        let take = self.rerank_cfg.max_rerank.min(pool.len());
        let input: Vec<(MemoryNote, f32)> =
            pool.iter().take(take).map(|n| (n.clone(), 0.0)).collect();
        let reranked = self.reranker.rerank(query, input).await?;
        let ranked = reranked.into_iter().map(|r| r.note.id).collect();
        Ok(Channel {
            name: "rerank",
            ranked,
        })
    }

    // ─── Phase 5b: PAS (prefix-aware sequential) ─────────────────────

    async fn phase_pas(&self, anchor_id: &str) -> Result<Channel> {
        let w = self.cfg().pas_window;
        let mut neighbors = match self
            .graph_store
            .get_sequential_neighbors(anchor_id, w)
            .await
        {
            Ok(n) => n,
            Err(e) => {
                warn!(error = %e, "pas: store error");
                Vec::new()
            }
        };
        neighbors.sort_by_key(|(_, delta)| delta.abs());
        Ok(Channel {
            name: "pas",
            ranked: neighbors.into_iter().map(|(id, _)| id).collect(),
        })
    }

    // ─── Phase 7: Trace (write-back) ──────────────────────────────────

    /// Write-back pass: bump access counters and Hebbian link weights for
    /// the final truncated set of returned ids. Must be called by the
    /// caller (`ask()` / `search()`) AFTER truncation so activation state
    /// trains on notes that actually reach the user.
    pub async fn phase_trace(&self, returned_ids: &[String]) -> Result<()> {
        if returned_ids.is_empty() {
            return Ok(());
        }
        // Bump access metadata for every returned note (one transaction).
        let ref_ids: Vec<&str> = returned_ids.iter().map(|s| s.as_str()).collect();
        self.vector_store
            .bump_access_many(&ref_ids, Utc::now())
            .await?;

        // Hebbian: strengthen every pre-existing semantic edge between
        // co-retrieved pairs. Batched so the whole `C(k,2)` set runs under
        // one transaction in stores that support it.
        let step = self.cfg().hebbian_weight_step;
        let max = self.cfg().hebbian_max_weight;
        let mut pairs: Vec<(&str, &str)> =
            Vec::with_capacity(returned_ids.len() * returned_ids.len().saturating_sub(1) / 2);
        for i in 0..returned_ids.len() {
            for j in (i + 1)..returned_ids.len() {
                pairs.push((returned_ids[i].as_str(), returned_ids[j].as_str()));
            }
        }
        if !pairs.is_empty()
            && let Err(e) = self
                .graph_store
                .bump_link_weights_batch(&pairs, step, max)
                .await
        {
            debug!(error = %e, "ACTIVATE: bump_link_weights_batch non-fatal failure");
        }
        Ok(())
    }

    // ─── Helpers ──────────────────────────────────────────────────────

    async fn extend_pool(&self, pool: &mut Vec<MemoryNote>, ids: &[String]) -> Result<()> {
        let have: HashSet<String> = pool.iter().map(|n| n.id.clone()).collect();
        let need: Vec<&str> = ids
            .iter()
            .filter(|id| !have.contains(*id))
            .map(|s| s.as_str())
            .collect();
        if need.is_empty() {
            return Ok(());
        }
        let fetched = self.vector_store.get_many(&need).await?;
        for n in fetched {
            if n.is_active() {
                pool.push(n);
            }
        }
        Ok(())
    }

    fn weights_for_mode(&self, mode: QueryMode) -> HashMap<String, f32> {
        self.cfg()
            .channel_weights
            .get(mode.as_str())
            .cloned()
            .unwrap_or_default()
    }

    /// Exposed sampling decision so callers can gate on the shared rate.
    pub fn should_sample_trace(&self) -> bool {
        let rate = self.cfg().trace_sample_rate;
        rate > 0.0 && should_sample(rate)
    }
}

// ─── Pure functions (unit-testable) ────────────────────────────────────

/// ACT-R base-level learning activation:
/// `B = ln(Σ_k t_k^{-d})` where `t_k` is the age in hours of the k-th access.
/// Falls back to `ln(max(1, count))` when no per-access history exists.
pub fn actr_activation(
    history: &[DateTime<Utc>],
    now: DateTime<Utc>,
    d: f64,
    count_fallback: u32,
) -> f64 {
    if history.is_empty() {
        return (count_fallback.max(1) as f64).ln();
    }
    let sum: f64 = history
        .iter()
        .map(|ts| {
            // Clamp to 1 minute to keep activation finite for "just accessed".
            let hrs = ((now - *ts).num_seconds() as f64 / 3600.0).max(1.0 / 60.0);
            hrs.powf(-d)
        })
        .sum();
    if sum <= 0.0 {
        f64::NEG_INFINITY
    } else {
        sum.ln()
    }
}

/// Reciprocal Rank Fusion (Cormack et al., 2009) over an arbitrary number
/// of ranked channels. Per-channel weights scale the RRF contribution.
pub fn rrf(channels: &[Channel], weights: &HashMap<String, f32>, k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for ch in channels {
        let w = weights.get(ch.name).copied().unwrap_or(0.0);
        if w == 0.0 {
            continue;
        }
        for (rank_0, id) in ch.ranked.iter().enumerate() {
            let contribution = w / (k + (rank_0 + 1) as f32);
            *scores.entry(id.clone()).or_insert(0.0) += contribution;
        }
    }
    let mut v: Vec<(String, f32)> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Deterministic-ish sampling for `trace_sample_rate`. Rate >= 1.0 always
/// samples; rate <= 0.0 never samples. Between uses nanosecond-based hash.
fn should_sample(rate: f32) -> bool {
    if rate >= 1.0 {
        return true;
    }
    if rate <= 0.0 {
        return false;
    }
    let nanos = Utc::now().timestamp_subsec_nanos() as f32 / 1_000_000_000.0;
    nanos < rate
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn hours_ago(h: i64, now: DateTime<Utc>) -> DateTime<Utc> {
        now - Duration::hours(h)
    }

    #[test]
    fn actr_bll_monotonic_in_frequency() {
        let now = Utc::now();
        let three = vec![hours_ago(24, now), hours_ago(12, now), hours_ago(1, now)];
        let one = vec![hours_ago(24, now)];
        let b3 = actr_activation(&three, now, 0.5, 3);
        let b1 = actr_activation(&one, now, 0.5, 1);
        assert!(
            b3 > b1,
            "multiple recent accesses must raise activation: {} !> {}",
            b3,
            b1
        );
    }

    #[test]
    fn actr_bll_decays_over_time() {
        let now = Utc::now();
        let history = vec![hours_ago(1, now)];
        let b_now = actr_activation(&history, now, 0.5, 1);
        let b_later = actr_activation(&history, now + Duration::days(30), 0.5, 1);
        assert!(
            b_later < b_now,
            "BLL must decay across 30 days: now={} later={}",
            b_now,
            b_later
        );
    }

    #[test]
    fn actr_fallback_when_history_empty() {
        let now = Utc::now();
        assert_eq!(actr_activation(&[], now, 0.5, 1), 0.0);
        // ln(4) ≈ 1.386
        let b = actr_activation(&[], now, 0.5, 4);
        assert!((b - 4f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn rrf_merges_channels_by_rank() {
        let c1 = Channel {
            name: "ann",
            ranked: vec!["a".into(), "b".into(), "c".into()],
        };
        let c2 = Channel {
            name: "rerank",
            ranked: vec!["b".into(), "a".into(), "d".into()],
        };
        let mut w = HashMap::new();
        w.insert("ann".into(), 1.0);
        w.insert("rerank".into(), 1.0);
        let fused = rrf(&[c1, c2], &w, 60.0);
        // "a" is rank 1 in ann + rank 2 in rerank = 1/61 + 1/62 > "b"s 1/62 + 1/61 (equal).
        // Assert both top-2 are {a, b} and "c" / "d" fall below.
        let top2: HashSet<&str> = fused.iter().take(2).map(|(id, _)| id.as_str()).collect();
        assert!(
            top2.contains("a") && top2.contains("b"),
            "top2 = {:?}",
            top2
        );
    }

    #[test]
    fn rrf_respects_weights() {
        let c_low = Channel {
            name: "ann",
            ranked: vec!["x".into()],
        };
        let c_high = Channel {
            name: "rerank",
            ranked: vec!["y".into()],
        };
        let mut w = HashMap::new();
        w.insert("ann".into(), 0.1);
        w.insert("rerank".into(), 10.0);
        let fused = rrf(&[c_low, c_high], &w, 60.0);
        assert_eq!(fused[0].0, "y", "heavier-weighted channel should win");
    }

    #[test]
    fn rrf_ignores_zero_weight_channels() {
        let c = Channel {
            name: "pas",
            ranked: vec!["x".into(), "y".into()],
        };
        let mut w = HashMap::new();
        w.insert("pas".into(), 0.0);
        let fused = rrf(&[c], &w, 60.0);
        assert!(fused.is_empty());
    }

    #[test]
    fn rrf_skips_missing_channel_names() {
        // A channel not present in the weight map contributes nothing.
        let c = Channel {
            name: "mystery",
            ranked: vec!["x".into()],
        };
        let w: HashMap<String, f32> = HashMap::new();
        let fused = rrf(&[c], &w, 60.0);
        assert!(fused.is_empty());
    }

    #[test]
    fn rrf_hebbian_weight_suppresses_outlier_under_cluster_flood() {
        // Scenario: 20 "consistent cluster" notes occupy ANN ranks 1..=7 and 9..=21
        // (outlier inserted at ANN rank 8). The 20 cluster members ALSO appear in
        // the hebbian channel ranks 1..=20 (because they link to each other).
        // The outlier appears ONLY in the ANN channel — it has no Hebbian edges
        // to the cluster. This is the Flask-Login scenario compressed into a
        // deterministic test.

        let mut ann_ranked: Vec<String> = (1..=7).map(|i| format!("cluster_{}", i)).collect();
        ann_ranked.push("outlier".to_string()); // ANN rank 8
        for i in 8..=20 {
            ann_ranked.push(format!("cluster_{}", i));
        }

        let hebbian_ranked: Vec<String> = (1..=20).map(|i| format!("cluster_{}", i)).collect();

        let ann_ch = Channel {
            name: "ann",
            ranked: ann_ranked,
        };
        let hebbian_ch = Channel {
            name: "hebbian",
            ranked: hebbian_ranked,
        };

        // Pre-fix weights (old Existence mode): hebbian = 0.7 (the Standard default
        // was 0.7, Existence was 0.5 — either value reproduces the suppression).
        let mut weights_pre = HashMap::new();
        weights_pre.insert("ann".into(), 1.0);
        weights_pre.insert("hebbian".into(), 0.7);

        let fused_pre = rrf(&[ann_ch.clone(), hebbian_ch.clone()], &weights_pre, 60.0);
        let top10_pre: Vec<&str> = fused_pre
            .iter()
            .take(10)
            .map(|(id, _)| id.as_str())
            .collect();

        assert!(
            !top10_pre.contains(&"outlier"),
            "With hebbian weight > 0, the contradicting outlier should be suppressed \
             out of top-10 by double-voting cluster members. Got top-10: {:?}",
            top10_pre
        );

        // Post-fix weights (new Existence mode): hebbian = 0.0
        let mut weights_post = HashMap::new();
        weights_post.insert("ann".into(), 1.0);
        weights_post.insert("hebbian".into(), 0.0);

        let fused_post = rrf(&[ann_ch, hebbian_ch], &weights_post, 60.0);
        let top10_post: Vec<&str> = fused_post
            .iter()
            .take(10)
            .map(|(id, _)| id.as_str())
            .collect();

        assert!(
            top10_post.contains(&"outlier"),
            "With hebbian weight = 0, the outlier should retain its ANN rank 8 \
             and appear in top-10. Got top-10: {:?}",
            top10_post
        );
    }

    #[test]
    fn rrf_hebbian_weight_continuous_between_zero_and_one() {
        // Same setup as above.
        let mut ann_ranked: Vec<String> = (1..=7).map(|i| format!("cluster_{}", i)).collect();
        ann_ranked.push("outlier".to_string());
        for i in 8..=20 {
            ann_ranked.push(format!("cluster_{}", i));
        }
        let hebbian_ranked: Vec<String> = (1..=20).map(|i| format!("cluster_{}", i)).collect();
        let ann_ch = Channel {
            name: "ann",
            ranked: ann_ranked,
        };
        let hebbian_ch = Channel {
            name: "hebbian",
            ranked: hebbian_ranked,
        };

        let outlier_rank = |hebbian_w: f32| -> usize {
            let mut w = HashMap::new();
            w.insert("ann".into(), 1.0);
            w.insert("hebbian".into(), hebbian_w);
            let fused = rrf(&[ann_ch.clone(), hebbian_ch.clone()], &w, 60.0);
            fused.iter().position(|(id, _)| id == "outlier").unwrap()
        };

        let ranks: Vec<usize> = [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|&w| outlier_rank(w))
            .collect();

        // Outlier rank should monotonically increase (i.e., fall deeper) as hebbian weight grows.
        for pair in ranks.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "Outlier rank must be monotonically non-decreasing as hebbian weight grows. \
                 Ranks across [0.0, 0.25, 0.5, 0.75, 1.0]: {:?}",
                ranks
            );
        }

        // And specifically: at hebbian=0 it should be position 7 (0-indexed), at hebbian=1 strictly deeper.
        assert!(
            ranks[4] > ranks[0],
            "Outlier must be strictly deeper at hebbian=1.0 than at hebbian=0.0. Got: {:?}",
            ranks
        );
    }

    #[test]
    fn should_sample_boundaries() {
        assert!(should_sample(1.0));
        assert!(!should_sample(0.0));
    }

    /// Every QueryMode variant must produce the same string used as the
    /// key in the default channel_weights matrix. Keep these in sync.
    #[test]
    fn query_mode_as_str_matches_default_weights() {
        use crate::config::ActivateConfig;
        use crate::read::QueryMode;

        let defaults = ActivateConfig::default();
        for mode in [
            QueryMode::Standard,
            QueryMode::Recency,
            QueryMode::Breadth,
            QueryMode::Computation,
            QueryMode::Temporal,
            QueryMode::Existence,
        ] {
            assert!(
                defaults.channel_weights.contains_key(mode.as_str()),
                "default channel_weights is missing key for {:?}",
                mode
            );
        }
        // Specific expected strings (canonical form).
        assert_eq!(QueryMode::Standard.as_str(), "Standard");
        assert_eq!(QueryMode::Recency.as_str(), "Recency");
        assert_eq!(QueryMode::Breadth.as_str(), "Breadth");
        assert_eq!(QueryMode::Computation.as_str(), "Computation");
        assert_eq!(QueryMode::Temporal.as_str(), "Temporal");
        assert_eq!(QueryMode::Existence.as_str(), "Existence");
    }

    /// Verify the new `bump_link_weights_batch` default-impl path: a stub
    /// store records each pair it sees and we assert the pairs expanded
    /// from phase_trace's C(k,2) loop match expectations.
    #[tokio::test]
    async fn phase_trace_batch_path_expands_pairs() {
        use crate::error::Result;
        use crate::store::{GraphStore, VectorStore};
        use async_trait::async_trait;
        use std::sync::Mutex as StdMutex;

        #[derive(Default)]
        struct BumpRecorder {
            pairs: StdMutex<Vec<(String, String)>>,
        }

        #[async_trait]
        impl GraphStore for BumpRecorder {
            async fn add_link(&self, _: &str, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            async fn get_links(&self, _: &str) -> Result<Vec<String>> {
                Ok(vec![])
            }
            async fn get_links_with_reasons(&self, _: &str) -> Result<Vec<(String, String)>> {
                Ok(vec![])
            }
            async fn record_evolution(&self, _: &str, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            async fn get_evolution_history(
                &self,
                _: &str,
            ) -> Result<Vec<crate::note::EvolutionRecord>> {
                Ok(vec![])
            }
            async fn record_dream_run(&self, _: &crate::dream::DreamRun) -> Result<()> {
                Ok(())
            }
            async fn get_dream_cursor(&self) -> Result<Option<DateTime<Utc>>> {
                Ok(None)
            }
            async fn set_dream_cursor(&self, _: DateTime<Utc>) -> Result<()> {
                Ok(())
            }
            async fn init(&self) -> Result<()> {
                Ok(())
            }

            async fn bump_link_weight(&self, a: &str, b: &str, _d: f32, _m: f32) -> Result<()> {
                self.pairs.lock().unwrap().push((a.into(), b.into()));
                Ok(())
            }
        }

        // Minimal VectorStore that just tracks which ids got bumped (not asserted).
        #[derive(Default)]
        struct NopVec;
        #[async_trait]
        impl VectorStore for NopVec {
            async fn upsert(&self, _: &crate::note::MemoryNote) -> Result<()> {
                Ok(())
            }
            async fn find_similar(
                &self,
                _: &[f32],
                _: usize,
                _: &[&str],
            ) -> Result<Vec<(crate::note::MemoryNote, f32)>> {
                Ok(vec![])
            }
            async fn get(&self, _: &str) -> Result<Option<crate::note::MemoryNote>> {
                Ok(None)
            }
            async fn get_many(&self, _: &[&str]) -> Result<Vec<crate::note::MemoryNote>> {
                Ok(vec![])
            }
            async fn get_all(&self) -> Result<Vec<crate::note::MemoryNote>> {
                Ok(vec![])
            }
            async fn delete(&self, _: &str) -> Result<()> {
                Ok(())
            }
            async fn count(&self) -> Result<usize> {
                Ok(0)
            }
        }

        let recorder = Arc::new(BumpRecorder::default());
        let ids = ["a".to_string(), "b".to_string(), "c".to_string()];

        // Exercise the batch default-impl directly via the GraphStore trait:
        // phase_trace would call this with the cross-product of the returned ids.
        let mut pairs: Vec<(&str, &str)> = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                pairs.push((ids[i].as_str(), ids[j].as_str()));
            }
        }
        assert_eq!(pairs.len(), 3, "C(3,2) = 3 pairs");

        let gs: Arc<dyn GraphStore> = recorder.clone();
        gs.bump_link_weights_batch(&pairs, 0.1, 1.0).await.unwrap();

        let recorded = recorder.pairs.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        // Default impl loops the single-pair method in the same order we passed.
        assert_eq!(recorded[0], ("a".into(), "b".into()));
        assert_eq!(recorded[1], ("a".into(), "c".into()));
        assert_eq!(recorded[2], ("b".into(), "c".into()));

        // NopVec unused here (phase_trace covers it end-to-end through the
        // engine which already has integration coverage).
        let _nop: Arc<dyn VectorStore> = Arc::new(NopVec);
    }
}
