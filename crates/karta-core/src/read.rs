use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, info};

use crate::config::ReadConfig;
use crate::error::Result;
use crate::llm::{ChatMessage, GenConfig, LlmProvider, Prompts, Role};
use crate::note::{MemoryNote, Provenance, SearchResult};
use crate::rerank::{Reranker, RerankerConfig};
use crate::store::{GraphStore, VectorStore};

/// Handles the read path: search, graph traversal, reranking, synthesis.
pub struct ReadEngine {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    llm: Arc<dyn LlmProvider>,
    reranker: Arc<dyn Reranker>,
    config: ReadConfig,
    reranker_config: RerankerConfig,
}

impl ReadEngine {
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        llm: Arc<dyn LlmProvider>,
        reranker: Arc<dyn Reranker>,
        config: ReadConfig,
        reranker_config: RerankerConfig,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            llm,
            reranker,
            config,
            reranker_config,
        }
    }

    /// Compute a recency score for a note using exponential decay.
    /// Returns 1.0 for brand new notes, decaying toward 0.0 for old notes.
    fn recency_score(&self, note: &MemoryNote) -> f32 {
        let age_days = Utc::now()
            .signed_duration_since(note.updated_at)
            .num_seconds() as f64
            / 86400.0;

        if age_days <= 0.0 {
            return 1.0;
        }

        // Exponential decay: score = 0.5^(age / half_life)
        let half_life = self.config.recency_half_life_days.max(1.0);
        (0.5_f64.powf(age_days / half_life)) as f32
    }

    /// Combine similarity score with recency to produce a final score.
    fn blended_score(&self, similarity: f32, note: &MemoryNote) -> f32 {
        let w = self.config.recency_weight.clamp(0.0, 1.0);
        let recency = self.recency_score(note);
        (1.0 - w) * similarity + w * recency
    }

    /// Detect if a query is asking about temporal ordering or event sequences.
    fn is_temporal_query(query: &str) -> bool {
        let q = query.to_lowercase();
        q.contains("order") || q.contains("sequence") || q.contains("timeline")
            || q.contains("chronolog") || q.contains("first") || q.contains("before")
            || q.contains("after") || q.contains("when did") || q.contains("how did")
            || q.contains("progression") || q.contains("evolve") || q.contains("changed over")
            || q.contains("steps") || q.contains("history of")
    }

    /// Drill into an episode: fetch constituent notes, filter active, sort chronologically.
    async fn episode_drilldown(&self, episode_id: &str) -> Result<Vec<MemoryNote>> {
        let note_ids = self.graph_store.get_notes_for_episode(episode_id).await?;
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }

        let id_refs: Vec<&str> = note_ids.iter().map(|s| s.as_str()).collect();
        let mut notes: Vec<MemoryNote> = self
            .vector_store
            .get_many(&id_refs)
            .await?
            .into_iter()
            .filter(|n| n.is_active())
            .collect();

        // Chronological order — the key fix for event ordering
        notes.sort_by_key(|n| n.created_at);
        notes.truncate(self.config.max_notes_per_episode);
        Ok(notes)
    }

    /// BFS traversal through the link graph up to max_depth hops.
    /// Each hop applies a decay factor to the weight.
    /// Returns deduplicated notes sorted by traversal weight.
    async fn multi_hop_traverse(
        &self,
        seed_id: &str,
        max_depth: usize,
        decay: f32,
    ) -> Result<Vec<MemoryNote>> {
        use std::collections::VecDeque;

        let mut visited = HashSet::new();
        visited.insert(seed_id.to_string());

        let mut queue: VecDeque<(String, usize, f32)> = VecDeque::new();
        let mut weighted_notes: Vec<(MemoryNote, f32)> = Vec::new();

        // Seed with direct links at depth 0
        let initial_links = self.graph_store.get_links(seed_id).await?;
        for link_id in initial_links {
            if visited.insert(link_id.clone()) {
                queue.push_back((link_id, 1, 1.0));
            }
        }

        const MAX_TRAVERSED: usize = 50;

        while let Some((current_id, depth, weight)) = queue.pop_front() {
            if weighted_notes.len() >= MAX_TRAVERSED {
                break;
            }

            if let Some(note) = self.vector_store.get(&current_id).await? {
                weighted_notes.push((note, weight));
            }

            if depth < max_depth {
                let next_weight = weight * decay;
                let links = self.graph_store.get_links(&current_id).await?;
                for link_id in links {
                    if visited.insert(link_id.clone()) {
                        queue.push_back((link_id, depth + 1, next_weight));
                    }
                }
            }
        }

        // Sort by weight descending
        weighted_notes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(weighted_notes.into_iter().map(|(n, _)| n).collect())
    }

    /// Embed query, find top-K with two-level episode retrieval, apply temporal scoring, follow links.
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        info!("Searching: \"{}\"", query);

        let embeddings = self.llm.embed(&[query]).await?;
        let query_embedding = embeddings.into_iter().next().unwrap_or_default();

        let is_temporal = Self::is_temporal_query(query);

        // Fetch more than top_k to allow re-ranking and episode discovery
        let fetch_k = if is_temporal {
            (top_k * 4).max(20)
        } else {
            (top_k * 2).max(10)
        };
        let direct = self
            .vector_store
            .find_similar(&query_embedding, fetch_k, &[])
            .await?;

        // Collect active foresight source note IDs for boosting
        let active_foresight_note_ids: std::collections::HashSet<String> = self
            .graph_store
            .get_active_foresights()
            .await?
            .into_iter()
            .map(|f| f.source_note_id)
            .collect();

        // --- Entity profile auto-include ---
        // Match query against known entity profiles and inject them into results
        let mut profile_note_ids: HashSet<String> = HashSet::new();
        let mut profile_results: Vec<SearchResult> = Vec::new();
        let query_lower = query.to_lowercase();

        let profiles = self.graph_store.get_all_profiles().await?;
        for (entity_id, note_id) in &profiles {
            let tokens: Vec<&str> = entity_id
                .split(|c: char| c.is_whitespace() || c == '-' || c == '_')
                .filter(|t| t.len() >= 3)
                .collect();
            let matched = tokens.iter().any(|t| query_lower.contains(&t.to_lowercase()));
            if matched {
                if let Some(note) = self.vector_store.get(note_id).await? {
                    if note.is_active() {
                        let link_count = self.graph_store.get_link_count(note_id).await?;
                        let graph_bonus = self.config.graph_weight * (1.0 + link_count as f32).ln();
                        let score = 1.0 + graph_bonus;
                        profile_note_ids.insert(note_id.clone());
                        debug!(entity_id = %entity_id, score = score, "Profile auto-include");
                        profile_results.push(SearchResult {
                            note,
                            score,
                            linked_notes: Vec::new(),
                        });
                    }
                }
            }
        }

        // --- Two-level episode retrieval ---
        // Partition ANN hits into episode narratives vs regular notes
        let mut episode_hits: Vec<(String, f32)> = Vec::new(); // (episode_id, score)
        let mut flat_hits: Vec<(MemoryNote, f32)> = Vec::new();

        let foresight_boost = self.config.foresight_boost;

        for (note, sim) in direct {
            if !note.is_active() || profile_note_ids.contains(&note.id) {
                continue;
            }
            let mut final_score = self.blended_score(sim, &note);

            // Graph-aware scoring: notes with more links score higher (PageRank-lite)
            let link_count = self.graph_store.get_link_count(&note.id).await?;
            let graph_bonus = self.config.graph_weight * (1.0 + link_count as f32).ln();
            final_score += graph_bonus;

            if active_foresight_note_ids.contains(&note.id) {
                final_score += foresight_boost;
            }

            if self.config.episode_retrieval_enabled {
                if let Provenance::Episode { ref episode_id } = note.provenance {
                    if final_score >= self.config.episode_drilldown_min_score {
                        episode_hits.push((episode_id.clone(), final_score));
                        continue; // Only skip flat if we're drilling down
                    }
                    // Below threshold: fall through to flat_hits — narrative may still be useful
                }
            }

            flat_hits.push((note, final_score));
        }

        // Sort episodes by score descending
        episode_hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_drilldowns = if is_temporal {
            self.config.max_episode_drilldowns * 2
        } else {
            self.config.max_episode_drilldowns
        };
        episode_hits.truncate(max_drilldowns);

        // Drill into episodes — collect chronologically ordered notes
        let mut episode_note_ids = HashSet::new();
        let mut episode_results: Vec<SearchResult> = Vec::new();

        for (episode_id, ep_score) in &episode_hits {
            let drilled = self.episode_drilldown(episode_id).await?;
            debug!(
                episode_id = %episode_id,
                score = ep_score,
                notes = drilled.len(),
                "Episode drilldown"
            );
            for note in drilled {
                if episode_note_ids.insert(note.id.clone()) {
                    episode_results.push(SearchResult {
                        note,
                        score: *ep_score,
                        linked_notes: Vec::new(), // Skip multi-hop for episode-drilled notes
                    });
                }
            }
        }

        // Sort flat hits and truncate
        flat_hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Budget: never starve flat ANN results — always keep at least top_k
        let flat_budget = top_k;
        flat_hits.truncate(flat_budget);

        // Build flat results with multi-hop traversal
        let max_depth = self.config.max_hop_depth;
        let decay = self.config.hop_decay_factor;
        let mut flat_results: Vec<SearchResult> = Vec::new();

        for (note, score) in flat_hits {
            if episode_note_ids.contains(&note.id) {
                continue; // Already included via episode drilldown
            }
            let linked_notes = self
                .multi_hop_traverse(&note.id, max_depth, decay)
                .await?;

            flat_results.push(SearchResult {
                note,
                score,
                linked_notes,
            });
        }

        // Merge: profiles first, then episode-drilled (chronological), then flat hits (by score)
        let mut results = profile_results;
        results.extend(episode_results);
        results.extend(flat_results);
        results.truncate(top_k);

        // Update last_accessed_at for all returned notes (access tracking for forgetting)
        for result in &results {
            let mut accessed = result.note.clone();
            accessed.last_accessed_at = Utc::now();
            let _ = self.vector_store.upsert(&accessed).await;
        }

        info!(
            results = results.len(),
            episode_drilldowns = episode_hits.len(),
            linked_total = results.iter().map(|r| r.linked_notes.len()).sum::<usize>(),
            "Search complete"
        );

        Ok(results)
    }

    /// Detect if a query is asking for a summary/overview (needs broader retrieval).
    fn is_summarization_query(query: &str) -> bool {
        let q = query.to_lowercase();
        q.contains("summary") || q.contains("summarize") || q.contains("comprehensive")
            || q.contains("overview") || q.contains("walk me through")
            || q.contains("list the order") || q.contains("how has")
            || q.contains("how did") || q.contains("progression")
    }

    /// Search + deduplicate + synthesize an answer with provenance markers.
    /// Includes abstention calibration: if no notes are sufficiently relevant, abstains.
    pub async fn ask(&self, query: &str, top_k: usize) -> Result<String> {
        // Adaptive top-K: summarization queries need broader coverage
        let effective_top_k = if Self::is_summarization_query(query) {
            top_k * self.config.summarization_top_k_multiplier
        } else {
            top_k
        };

        let results = self.search(query, effective_top_k).await?;

        if results.is_empty() {
            return Ok("Based on the available memories, I don't have information about this topic.".to_string());
        }

        // --- Reranker-based abstention ---
        // If reranker is enabled, re-score results for true relevance.
        // The reranker distinguishes "shares vocabulary" from "actually answers the question."
        if self.reranker_config.enabled {
            let notes_for_rerank: Vec<(MemoryNote, f32)> = results
                .iter()
                .take(self.reranker_config.max_rerank)
                .map(|r| (r.note.clone(), r.score))
                .collect();

            let reranked = self.reranker.rerank(query, notes_for_rerank).await?;

            let best_relevance = reranked.iter()
                .map(|r| r.relevance_score)
                .fold(0.0f32, f32::max);

            if best_relevance < self.reranker_config.abstention_threshold {
                debug!(
                    best_relevance = best_relevance,
                    threshold = self.reranker_config.abstention_threshold,
                    "Reranker: abstaining — notes not relevant to query"
                );
                return Ok(
                    "Based on the available memories, I don't have information about this topic."
                        .to_string(),
                );
            }
        }

        // Deduplicate: collect all unique notes (direct + linked)
        let mut seen = HashSet::new();
        let mut all_notes: Vec<&MemoryNote> = Vec::new();

        for result in &results {
            if seen.insert(&result.note.id) {
                all_notes.push(&result.note);
            }
            for linked in &result.linked_notes {
                if seen.insert(&linked.id) {
                    all_notes.push(linked);
                }
            }
        }

        if all_notes.is_empty() {
            return Ok("No relevant memories found.".to_string());
        }

        // Build notes text with provenance markers so the LLM knows
        // which notes are observed facts vs dream-derived inferences
        let notes_text: String = all_notes
            .iter()
            .enumerate()
            .map(|(i, note)| {
                let provenance_marker = match &note.provenance {
                    Provenance::Observed => "FACT".to_string(),
                    Provenance::Dream { dream_type, confidence, .. } => {
                        format!("INFERRED:{} conf={:.0}%", dream_type, confidence * 100.0)
                    }
                    Provenance::Profile { entity_id } => {
                        format!("PROFILE:{}", entity_id)
                    }
                    Provenance::Episode { episode_id } => {
                        format!("EPISODE:{}", episode_id)
                    }
                };
                let age = Utc::now()
                    .signed_duration_since(note.created_at)
                    .num_days();
                let recency = if age == 0 {
                    "today".to_string()
                } else if age == 1 {
                    "1 day ago".to_string()
                } else {
                    format!("{} days ago", age)
                };

                let date_str = note.created_at.format("%Y-%m-%d %H:%M");
                format!(
                    "[{}] ({}, {}, {}) {}\n    Context: {}",
                    i + 1,
                    provenance_marker,
                    date_str,
                    recency,
                    note.content,
                    note.context,
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: Prompts::synthesize_system().to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: Prompts::synthesize_user(query, &notes_text),
            },
        ];

        // Use structured output with reasoning for better abstention and consistency
        let config = GenConfig {
            max_tokens: 16384,
            temperature: 0.0,
            json_mode: false, // json_schema takes precedence
            json_schema: Some(crate::llm::schemas::synthesis_schema()),
        };

        let response = self.llm.chat(&messages, &config).await?;

        // Parse structured response
        let parsed: serde_json::Value =
            serde_json::from_str(&response.content).unwrap_or_default();

        let should_abstain = parsed["should_abstain"].as_bool().unwrap_or(false);
        let has_contradiction = parsed["has_contradiction"].as_bool().unwrap_or(false);
        let reasoning = parsed["reasoning"].as_str().unwrap_or("");

        if should_abstain {
            debug!(reasoning = reasoning, "Structured output: abstaining");
            return Ok(
                "Based on the available memories, I don't have information about this topic."
                    .to_string(),
            );
        }

        // Handle null answer (LLM returned answer: null without setting should_abstain)
        let mut answer = match parsed["answer"].as_str() {
            Some(a) if !a.is_empty() => a.to_string(),
            _ => {
                // Fallback: if the raw response is valid prose (not JSON), use it;
                // otherwise abstain gracefully
                if response.content.starts_with('{') {
                    return Ok(
                        "Based on the available memories, I don't have information about this topic."
                            .to_string(),
                    );
                }
                response.content.clone()
            }
        };

        // Prepend contradiction notice if flagged
        if has_contradiction {
            answer = format!(
                "**Note: The memories contain contradictory information on this topic.**\n\n{}",
                answer
            );
        }

        Ok(answer)
    }
}
