use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, info};

use crate::config::ReadConfig;
use crate::error::Result;
use crate::llm::{ChatMessage, GenConfig, LlmProvider, Prompts, Role};
use crate::note::{AskResult, MemoryNote, Provenance, SearchResult};
use crate::rerank::{Reranker, RerankerConfig};
use crate::store::{GraphStore, VectorStore};

/// Query classification for mode-specific retrieval behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryMode {
    /// Default: balanced similarity + recency.
    Standard,
    /// "current/latest/now" — aggressive recency weighting.
    Recency,
    /// "summarize/overview" — broad retrieval with high top_k.
    Breadth,
    /// "how many/difference/compare" — force multi-hop for cross-note computation.
    Computation,
    /// "order/sequence/steps" — sort by turn_index, maximize episode coverage.
    Temporal,
    /// "did I ever/is it true/still" — include contradiction dreams.
    Existence,
}

/// Prototype examples for embedding-based query classification.
/// Each mode has representative queries that define its embedding centroid.
const PROTOTYPES: &[(QueryMode, &[&str])] = &[
    (QueryMode::Temporal, &[
        "Can you list in order how I brought up different topics?",
        "What was the sequence of events in my project?",
        "In what order did I discuss these subjects?",
        "What was the chronological progression of my work?",
        "List the timeline of my activities",
        "What happened first, second, third?",
        "Walk me through the order of topics we covered",
    ]),
    (QueryMode::Recency, &[
        "What is my current status on the project?",
        "What's the latest update on my progress?",
        "What am I working on right now?",
        "What's the most recent version of my plan?",
        "What did I last decide about the design?",
        "What's my updated budget?",
    ]),
    (QueryMode::Breadth, &[
        "Give me a comprehensive summary of my project",
        "Can you summarize everything we've discussed?",
        "Provide an overview of all my activities",
        "Walk me through the full picture of what I've been doing",
        "How has my approach evolved over time?",
        "How did my project develop from start to finish?",
    ]),
    (QueryMode::Computation, &[
        "How many days between the start and the deadline?",
        "How many weeks did it take to finish?",
        "How long did it take me to complete the task?",
        "What's the total number of items I completed?",
        "How many more problems did I solve compared to last time?",
        "Calculate the time between event A and event B",
        "How much progress did I make between March and April?",
        "Which happened first, X or Y, and how far apart?",
        "How many hours did I spend studying?",
        "Days between my meeting and the deadline",
    ]),
    (QueryMode::Existence, &[
        "Did I ever mention using Excel for tracking?",
        "Is it true that I contradicted myself about the tool?",
        "Have I been inconsistent about my preferences?",
        "Does my earlier statement conflict with the later one?",
        "Was there a contradiction in what I said?",
    ]),
    (QueryMode::Standard, &[
        "What tools am I using for my project?",
        "What did I say about the API integration?",
        "What are my preferences for the UI design?",
        "Tell me about my debugging approach",
        "What feedback did I receive on my work?",
        "What are the main features I'm building?",
        "What constraints or requirements did I mention?",
    ]),
];

/// Embedding-based query classifier. Classifies by cosine similarity to prototype centroids.
struct QueryClassifier {
    centroids: Vec<(QueryMode, Vec<f32>)>,
}

impl QueryClassifier {
    /// Build classifier by embedding all prototypes and averaging per mode.
    async fn new(llm: &dyn LlmProvider) -> Self {
        let mut centroids = Vec::new();

        for (mode, examples) in PROTOTYPES {
            let refs: Vec<&str> = examples.to_vec();
            match llm.embed(&refs).await {
                Ok(embeddings) if !embeddings.is_empty() => {
                    let dim = embeddings[0].len();
                    let mut centroid = vec![0.0f32; dim];
                    for emb in &embeddings {
                        for (i, v) in emb.iter().enumerate() {
                            centroid[i] += v;
                        }
                    }
                    let n = embeddings.len() as f32;
                    for v in centroid.iter_mut() {
                        *v /= n;
                    }
                    centroids.push((*mode, centroid));
                }
                _ => {
                    tracing::warn!(mode = ?mode, "Failed to embed prototypes, skipping mode");
                }
            }
        }

        Self { centroids }
    }

    fn classify(&self, query_embedding: &[f32]) -> QueryMode {
        if self.centroids.is_empty() {
            return QueryMode::Standard;
        }

        let mut best_mode = QueryMode::Standard;
        let mut best_sim = f32::NEG_INFINITY;

        for (mode, centroid) in &self.centroids {
            let sim = cosine_similarity(query_embedding, centroid);
            if sim > best_sim {
                best_sim = sim;
                best_mode = *mode;
            }
        }

        best_mode
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Keyword-based fallback classifier (used when embeddings unavailable).
fn classify_query_keywords(query: &str) -> QueryMode {
    let q = query.to_lowercase();

    if q.contains("in what order") || q.contains("in order") || q.contains("sequence")
        || q.contains("timeline") || q.contains("chronolog")
        || q.contains("progression") || q.contains("history of")
        || q.contains("changed over") || q.contains("steps ")
        || (q.contains("list") && (q.contains("order") || q.contains("topic")))
    {
        return QueryMode::Temporal;
    }

    if q.contains("current") || q.contains("latest") || q.contains("most recent")
        || q.contains("right now") || q.contains("updated")
        || (q.contains("now") && !q.contains("know"))
    {
        return QueryMode::Recency;
    }

    if q.contains("summary") || q.contains("summarize") || q.contains("comprehensive")
        || q.contains("overview") || q.contains("walk me through")
        || q.contains("how has") || q.contains("how did")
    {
        return QueryMode::Breadth;
    }

    if q.contains("how many") || q.contains("how much") || q.contains("total")
        || q.contains("calculate") || q.contains("difference between")
        || q.contains("compare") || q.contains("days between")
        || q.contains("months between")
    {
        return QueryMode::Computation;
    }

    if q.contains("contradict") || q.contains("is it true") || q.contains("conflict")
        || q.contains("inconsisten")
    {
        return QueryMode::Existence;
    }

    QueryMode::Standard
}

/// Handles the read path: search, graph traversal, reranking, synthesis.
pub struct ReadEngine {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    llm: Arc<dyn LlmProvider>,
    reranker: Arc<dyn Reranker>,
    config: ReadConfig,
    reranker_config: RerankerConfig,
    classifier: tokio::sync::OnceCell<QueryClassifier>,
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
            classifier: tokio::sync::OnceCell::new(),
        }
    }

    /// Get or initialize the embedding-based query classifier.
    async fn get_classifier(&self) -> &QueryClassifier {
        self.classifier.get_or_init(|| async {
            info!("Initializing embedding-based query classifier...");
            QueryClassifier::new(self.llm.as_ref()).await
        }).await
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
    /// Accepts an explicit recency weight for mode-specific overrides.
    fn blended_score_with_weight(&self, similarity: f32, note: &MemoryNote, recency_weight: f32) -> f32 {
        let w = recency_weight.clamp(0.0, 1.0);
        let recency = self.recency_score(note);
        (1.0 - w) * similarity + w * recency
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

        // Chronological order: prefer turn_index > source_timestamp > created_at
        notes.sort_by(|a, b| {
            match (a.turn_index, b.turn_index) {
                (Some(ai), Some(bi)) => ai.cmp(&bi),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => match (a.source_timestamp, b.source_timestamp) {
                    (Some(at), Some(bt)) => at.cmp(&bt),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.created_at.cmp(&b.created_at),
                },
            }
        });
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

        // Embedding-based classification with keyword fallback
        let classifier = self.get_classifier().await;
        let mode = if classifier.centroids.is_empty() {
            classify_query_keywords(query)
        } else {
            classifier.classify(&query_embedding)
        };
        debug!(query_mode = ?mode, "Query classified");

        // Mode-specific fetch_k: wide pool for reranker, but Computation stays tight (precision > recall)
        let fetch_k = match mode {
            QueryMode::Temporal => (top_k * 4).max(20),
            QueryMode::Breadth => (top_k * 3).max(15),
            QueryMode::Computation => (top_k * 2).max(10),
            _ => (top_k * 4).max(20),
        };

        // Mode-specific recency weight override
        let effective_recency_weight = match mode {
            QueryMode::Recency => 0.60,
            _ => self.config.recency_weight,
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
            let mut final_score = self.blended_score_with_weight(sim, &note, effective_recency_weight);

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

        let max_drilldowns = match mode {
            QueryMode::Temporal | QueryMode::Breadth => self.config.max_episode_drilldowns * 2,
            _ => self.config.max_episode_drilldowns,
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

    /// Search + deduplicate + synthesize an answer with provenance markers.
    /// Includes abstention calibration: if no notes are sufficiently relevant, abstains.
    pub async fn ask(&self, query: &str, top_k: usize) -> Result<AskResult> {
        // Adaptive top-K based on query mode (keyword fallback; search() uses embedding classifier)
        let mode = classify_query_keywords(query);
        let effective_top_k = match mode {
            QueryMode::Breadth => top_k * self.config.summarization_top_k_multiplier,
            QueryMode::Temporal => top_k * 4,
            QueryMode::Computation => top_k * 2,
            _ => top_k,
        };

        let mode_str = format!("{:?}", mode);
        let mut reranker_best: Option<f32> = None;

        let mut results = self.search(query, effective_top_k).await?;

        if results.is_empty() {
            return Ok(AskResult {
                answer: "Based on the available memories, I don't have information about this topic.".to_string(),
                query_mode: mode_str,
                notes_used: 0,
                note_ids: Vec::new(),
                contradiction_injected: 0,
                has_contradiction: false,
                reranker_best_score: None,
            });
        }

        // --- Reranker: abstention gate + reorder results by cross-encoder relevance ---
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

            reranker_best = Some(best_relevance);

            if best_relevance < self.reranker_config.abstention_threshold {
                debug!(
                    best_relevance = best_relevance,
                    threshold = self.reranker_config.abstention_threshold,
                    "Reranker: abstaining — notes not relevant to query"
                );
                return Ok(AskResult {
                    answer: "Based on the available memories, I don't have information about this topic.".to_string(),
                    query_mode: mode_str,
                    notes_used: 0,
                    note_ids: Vec::new(),
                    contradiction_injected: 0,
                    has_contradiction: false,
                    reranker_best_score: reranker_best,
                });
            }

            // Reorder results by cross-encoder relevance, EXCEPT for Computation mode.
            // Computation needs factual completeness (both date notes), not topical precision.
            // Reranker pushes date-bearing notes down because they score low on topical relevance.
            if mode != QueryMode::Computation {
                let reranked_ids: HashSet<String> = reranked.iter().map(|r| r.note.id.clone()).collect();
                let mut reordered: Vec<SearchResult> = Vec::new();

                for rr in &reranked {
                    let linked = results.iter()
                        .find(|r| r.note.id == rr.note.id)
                        .map(|r| r.linked_notes.clone())
                        .unwrap_or_default();
                    reordered.push(SearchResult {
                        note: rr.note.clone(),
                        score: rr.relevance_score,
                        linked_notes: linked,
                    });
                }

                for r in &results {
                    if !reranked_ids.contains(&r.note.id) {
                        reordered.push(r.clone());
                    }
                }

                results = reordered;
                debug!(reranked_count = reranked.len(), "Reranker: reordered results by relevance");
            } else {
                debug!("Reranker: skipping reorder for Computation mode (preserving ANN order)");
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
            return Ok(AskResult {
                answer: "No relevant memories found.".to_string(),
                query_mode: mode_str,
                notes_used: 0,
                note_ids: Vec::new(),
                contradiction_injected: 0,
                has_contradiction: false,
                reranker_best_score: reranker_best,
            });
        }

        // Sort notes by conversation order (turn_index > source_timestamp > created_at).
        // LLMs perform better when notes arrive in chronological sequence, not relevance order.
        all_notes.sort_by(|a, b| {
            match (a.turn_index, b.turn_index) {
                (Some(ai), Some(bi)) => ai.cmp(&bi),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => match (a.source_timestamp, b.source_timestamp) {
                    (Some(at), Some(bt)) => at.cmp(&bt),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.created_at.cmp(&b.created_at),
                },
            }
        });

        // --- Contradiction force-retrieval ---
        // When a contradiction dream is among results (or a result note is linked to one),
        // force-include both source notes so the LLM sees both sides.
        let mut contradiction_inject_ids: HashSet<String> = HashSet::new();
        let seen_ids: HashSet<&String> = seen.iter().copied().collect();

        // 1. Scan for contradiction dreams in retrieved notes
        for note in all_notes.iter() {
            if let Provenance::Dream { dream_type, source_note_ids, .. } = &note.provenance {
                if dream_type == "contradiction" {
                    for sid in source_note_ids {
                        if !seen_ids.contains(sid) {
                            contradiction_inject_ids.insert(sid.clone());
                        }
                    }
                }
            }
        }

        // 2. Check if any retrieved note is linked to a contradiction dream (cap at 10 for latency)
        for note in all_notes.iter().take(10) {
            if let Ok(links) = self.graph_store.get_links_with_reasons(&note.id).await {
                for (linked_id, reason) in &links {
                    if reason.contains("contradiction dream") && !seen_ids.contains(linked_id) {
                        if let Ok(Some(dream_note)) = self.vector_store.get(linked_id).await {
                            if let Provenance::Dream { source_note_ids, .. } = &dream_note.provenance {
                                for sid in source_note_ids {
                                    if !seen_ids.contains(sid) {
                                        contradiction_inject_ids.insert(sid.clone());
                                    }
                                }
                            }
                            contradiction_inject_ids.insert(linked_id.clone());
                        }
                    }
                }
            }
        }

        // 3. Fetch injected notes (cap at 10 to bound LLM context growth)
        let mut inject_ids_vec: Vec<&str> = contradiction_inject_ids.iter().map(|s| s.as_str()).collect();
        inject_ids_vec.truncate(10);
        let inject_refs = inject_ids_vec;
        let contradiction_notes: Vec<MemoryNote> = if inject_refs.is_empty() {
            Vec::new()
        } else {
            debug!(count = inject_refs.len(), "Injecting contradiction source notes");
            self.vector_store.get_many(&inject_refs).await.unwrap_or_default()
        };

        // Build notes text with provenance markers so the LLM knows
        // which notes are observed facts vs dream-derived inferences.
        // Chain regular notes + contradiction-injected notes.
        let format_note = |i: usize, note: &MemoryNote, is_contradiction_source: bool| {
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
            // Use source_timestamp (real conversation date) if available, fall back to created_at
            let display_time = note.source_timestamp.unwrap_or(note.created_at);
            let age = Utc::now()
                .signed_duration_since(display_time)
                .num_days();
            let recency = if age == 0 {
                "today".to_string()
            } else if age == 1 {
                "1 day ago".to_string()
            } else {
                format!("{} days ago", age)
            };

            let date_str = display_time.format("%Y-%m-%d");
            let prefix = if is_contradiction_source { "[CONTRADICTION SOURCE] " } else { "" };
            format!(
                "[{}] {}({}, {}, {}) {}\n    Context: {}",
                i + 1,
                prefix,
                provenance_marker,
                date_str,
                recency,
                note.content,
                note.context,
            )
        };

        let mut note_entries: Vec<String> = Vec::new();
        for (i, note) in all_notes.iter().enumerate() {
            note_entries.push(format_note(i, note, false));
        }
        let base_count = all_notes.len();
        for (i, note) in contradiction_notes.iter().enumerate() {
            note_entries.push(format_note(base_count + i, note, true));
        }

        let notes_text = note_entries.join("\n\n");

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

        let has_contradiction = parsed["has_contradiction"].as_bool().unwrap_or(false);

        // Collect note IDs for metadata (including contradiction-injected notes)
        let mut collected_note_ids: Vec<String> = all_notes.iter().map(|n| n.id.clone()).collect();
        collected_note_ids.extend(contradiction_notes.iter().map(|n| n.id.clone()));
        let collected_count = all_notes.len() + contradiction_notes.len();
        let contradiction_count = contradiction_notes.len();

        // Extract answer — Gate 4: fallback for malformed JSON responses
        let mut answer = match parsed["answer"].as_str() {
            Some(a) if !a.is_empty() => a.to_string(),
            _ => {
                // Fallback: if the raw response is valid prose (not JSON), use it;
                // otherwise abstain gracefully
                if response.content.starts_with('{') {
                    return Ok(AskResult {
                        answer: "Based on the available memories, I don't have information about this topic.".to_string(),
                        query_mode: mode_str,
                        notes_used: collected_count,
                        note_ids: collected_note_ids,
                        contradiction_injected: contradiction_count,
                        has_contradiction,
                        reranker_best_score: reranker_best,
                    });
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

        Ok(AskResult {
            answer,
            query_mode: mode_str,
            notes_used: collected_count,
            note_ids: collected_note_ids,
            contradiction_injected: contradiction_count,
            has_contradiction,
            reranker_best_score: reranker_best,
        })
    }
}
