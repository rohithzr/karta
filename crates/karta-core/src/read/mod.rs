use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, info};

pub mod temporal;
pub mod resolve;

use crate::clock::ClockContext;
use crate::config::ReadConfig;
use crate::error::Result;
use crate::llm::{ChatMessage, GenConfig, LlmProvider, Prompts, Role};
use crate::note::{AskResult, FetchedMemories, MemoryNote, Provenance, SearchResult};
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

/// Lightweight regex/keyword pre-filter that flags queries containing temporal
/// indicators (relative phrases, weekday/month names, year tokens, ISO dates).
///
/// Used by the query classifier to set [`QueryClassification::temporal`] which
/// in turn enables interval-overlap SQL filtering at retrieval time. This is
/// purely lexical — semantic temporal intent is captured separately by
/// [`QueryMode::Temporal`] via embedding similarity.
pub(crate) fn has_temporal_indicator(query: &str) -> bool {
    let q = query.to_lowercase();

    let keywords = [
        "when", "last ", "next ", "yesterday", "today", "tomorrow",
        "before", "after", "during", "recently", "this week",
        "this month", "this year", "this quarter",
    ];
    if keywords.iter().any(|k| q.contains(k)) {
        return true;
    }

    let months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ];
    if months.iter().any(|m| q.contains(m)) {
        return true;
    }

    let days = [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    ];
    if days.iter().any(|d| q.contains(d)) {
        return true;
    }

    // 4-digit year (1900–2099) — also catches ISO dates like 2024-03-15.
    static YEAR_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = YEAR_RE.get_or_init(|| regex::Regex::new(r"\b(19|20)\d{2}\b").unwrap());
    re.is_match(&q)
}

/// Public, integration-test-visible wrapper around [`has_temporal_indicator`].
///
/// Crate-internal callers should prefer the `pub(crate)` helper; this wrapper
/// exists so tests under `crates/karta-core/tests/` (which see only `pub` API)
/// can verify the temporal pre-filter.
pub fn query_is_temporal(q: &str) -> bool {
    has_temporal_indicator(q)
}

/// Output of the query classifier: the mode bucket plus auxiliary flags that
/// downstream retrieval uses to gate behavior (e.g. interval-overlap SQL).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryClassification {
    pub mode: QueryMode,
    /// True when the query contains lexical temporal indicators (weekday/month
    /// names, year/ISO dates, "yesterday/last week/before/after/...").
    /// Set independently of `mode` — a query can be `QueryMode::Standard` and
    /// still be temporal (e.g. "did I deploy on March 15").
    pub temporal: bool,
}

/// Keyword-only classifier that returns the full [`QueryClassification`]
/// (mode + `temporal` flag). Embedding-based callers compose this with the
/// embedding-derived mode separately; see `ReadEngine::search_wide`.
pub fn classify_query(query: &str) -> QueryClassification {
    QueryClassification {
        mode: classify_query_keywords(query),
        temporal: has_temporal_indicator(query),
    }
}

/// Handles the read path: search, graph traversal, reranking, synthesis.
pub struct ReadEngine {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    llm: Arc<dyn LlmProvider>,
    synthesis_llm: Option<Arc<dyn LlmProvider>>,
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
        synthesis_llm: Option<Arc<dyn LlmProvider>>,
        reranker: Arc<dyn Reranker>,
        config: ReadConfig,
        reranker_config: RerankerConfig,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            llm,
            synthesis_llm,
            reranker,
            config,
            reranker_config,
            classifier: tokio::sync::OnceCell::new(),
        }
    }

    /// LLM used for the final answer-synthesis call. Defaults to `self.llm`
    /// (same as all internal Karta work) unless an explicit override was
    /// configured — lets the caller route only the user-facing answer step
    /// to a different model while keeping ingest/dream/retrieval on the
    /// primary LLM.
    fn synth_llm(&self) -> &dyn LlmProvider {
        self.synthesis_llm
            .as_deref()
            .unwrap_or_else(|| self.llm.as_ref())
    }

    /// Get or initialize the embedding-based query classifier.
    /// Falls back to empty classifier (keyword matching) on timeout or error.
    async fn get_classifier(&self) -> &QueryClassifier {
        self.classifier.get_or_init(|| async {
            eprintln!("[CLASSIFIER] Starting init (timeout: 60s)...");
            match tokio::time::timeout(
                std::time::Duration::from_secs(60),
                QueryClassifier::new(self.llm.as_ref()),
            ).await {
                Ok(c) => {
                    eprintln!("[CLASSIFIER] Initialized with {} centroids", c.centroids.len());
                    c
                }
                Err(_) => {
                    eprintln!("[CLASSIFIER] TIMEOUT — falling back to keyword matching");
                    QueryClassifier { centroids: Vec::new() }
                }
            }
        }).await
    }

    /// Compute a recency score for a note using exponential decay.
    /// Returns 1.0 for brand new notes, decaying toward 0.0 for old notes.
    /// Uses source_timestamp (the data's "now" at ingest) and the query's
    /// reference_time, NOT Utc::now() — replays must age relative to the
    /// query, not the wall clock.
    fn recency_score(&self, note: &MemoryNote, ctx: ClockContext) -> f32 {
        // Forward-date clamp (codex #2). If source_timestamp is past
        // reference_time (clock skew, future-dated import, bug), age_days
        // goes negative — clamp at 0.0 so recency = 1.0 (treated as
        // fresh-as-possible). Better than producing decay > 1 or NaN.
        let age_days = (ctx.reference_time() - note.source_timestamp)
            .num_seconds() as f64
            / 86400.0;
        let age_days = age_days.max(0.0);

        // Exponential decay: score = 0.5^(age / half_life)
        let half_life = self.config.recency_half_life_days.max(1.0);
        (0.5_f64.powf(age_days / half_life)) as f32
    }

    /// Combine similarity score with recency to produce a final score.
    /// Accepts an explicit recency weight for mode-specific overrides.
    fn blended_score_with_weight(
        &self,
        similarity: f32,
        note: &MemoryNote,
        recency_weight: f32,
        ctx: ClockContext,
    ) -> f32 {
        let w = recency_weight.clamp(0.0, 1.0);
        let recency = self.recency_score(note, ctx);
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
                (None, None) => a.source_timestamp.cmp(&b.source_timestamp),
            }
        });
        notes.truncate(self.config.max_notes_per_episode);
        Ok(notes)
    }

    /// BFS traversal through the link graph up to max_depth hops.
    /// Each hop applies a decay factor to the weight.
    /// Returns deduplicated notes sorted by traversal weight.
    ///
    /// Two-phase approach for latency: first traverse the graph (SQLite,
    /// sub-millisecond per hop) to collect all reachable IDs + weights,
    /// then fetch the notes in one batched vector-store call instead of
    /// N individual get() calls (which are expensive on Lance).
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
        let mut id_weights: Vec<(String, f32)> = Vec::new();

        let initial_links = self.graph_store.get_links(seed_id).await?;
        for link_id in initial_links {
            if visited.insert(link_id.clone()) {
                queue.push_back((link_id, 1, 1.0));
            }
        }

        const MAX_TRAVERSED: usize = 50;

        // Phase 1: graph-only BFS (SQLite, fast) — collect IDs + weights
        while let Some((current_id, depth, weight)) = queue.pop_front() {
            if id_weights.len() >= MAX_TRAVERSED {
                break;
            }

            id_weights.push((current_id.clone(), weight));

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

        if id_weights.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: batch fetch from vector store (one call instead of N)
        let id_refs: Vec<&str> = id_weights.iter().map(|(id, _)| id.as_str()).collect();
        let fetched = self.vector_store.get_many(&id_refs).await?;
        let note_map: std::collections::HashMap<String, MemoryNote> = fetched
            .into_iter()
            .map(|n| (n.id.clone(), n))
            .collect();

        let mut weighted_notes: Vec<(MemoryNote, f32)> = Vec::new();
        for (id, weight) in &id_weights {
            if let Some(note) = note_map.get(id) {
                weighted_notes.push((note.clone(), *weight));
            }
        }

        weighted_notes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(weighted_notes.into_iter().map(|(n, _)| n).collect())
    }

    /// Public search: returns exactly top_k results. Live default — anchors
    /// recency to Utc::now() via ClockContext::now().
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_clock(query, top_k, ClockContext::now()).await
    }

    /// Time-travel / replay query — recency anchored to ctx.reference_time().
    pub async fn search_with_clock(
        &self,
        query: &str,
        top_k: usize,
        ctx: ClockContext,
    ) -> Result<Vec<SearchResult>> {
        let (mut results, _mode) = self.search_wide(query, top_k, ctx).await?;
        results.truncate(top_k);

        // Access tracking (fire-and-forget — don't block the read path)
        let vs = Arc::clone(&self.vector_store);
        let ids: Vec<String> = results.iter().map(|r| r.note.id.clone()).collect();
        tokio::spawn(async move {
            for id in ids {
                if let Ok(Some(mut note)) = vs.get(&id).await {
                    note.last_accessed_at = Utc::now();
                    let _ = vs.upsert(&note).await;
                }
            }
        });

        Ok(results)
    }

    /// Internal search: returns the full expanded candidate pool (not truncated)
    /// plus the classified query mode. Used by ask() so the reranker can see the
    /// full pool before truncation, and ask() can use the same mode for top_k sizing.
    async fn search_wide(
        &self,
        query: &str,
        top_k: usize,
        ctx: ClockContext,
    ) -> Result<(Vec<SearchResult>, QueryMode)> {
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
        // Parallel search: notes + atomic facts (if enabled)
        let fact_k = fetch_k / 2;
        let (direct, fact_hits) = if self.config.fact_retrieval_enabled {
            let (direct_result, fact_hits_result) = tokio::join!(
                self.vector_store.find_similar(&query_embedding, fetch_k, &[]),
                self.vector_store.find_similar_facts(&query_embedding, fact_k, &[])
            );
            (direct_result?, fact_hits_result.unwrap_or_default())
        } else {
            let direct = self.vector_store.find_similar(&query_embedding, fetch_k, &[]).await?;
            (direct, Vec::new())
        };

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

            // Filter out dream/digest notes from direct ANN results.
            // They dilute top-K slots meant for original user notes.
            // Dreams are surfaced via contradiction force-retrieval.
            // Digests are surfaced via episode drilldown + episode link traversal.
            match &note.provenance {
                Provenance::Dream { .. } | Provenance::Digest { .. } | Provenance::Fact { .. } => continue,
                _ => {}
            }

            let mut final_score = self.blended_score_with_weight(sim, &note, effective_recency_weight, ctx);

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

        // --- Fact-to-note expansion ---
        // High-scoring facts: fetch parent notes and boost them into results
        let mut fact_expanded: Vec<SearchResult> = Vec::new();
        let mut seen_note_ids: HashSet<String> = HashSet::new();

        // Collect already-included note IDs
        for r in &profile_results { seen_note_ids.insert(r.note.id.clone()); }
        for r in &episode_results { seen_note_ids.insert(r.note.id.clone()); }
        for r in &flat_results { seen_note_ids.insert(r.note.id.clone()); }
        for id in &episode_note_ids { seen_note_ids.insert(id.clone()); }

        let fact_boost = self.config.fact_match_boost;
        for (fact, score) in &fact_hits {
            if *score < 0.3 { continue; }
            if seen_note_ids.insert(fact.source_note_id.clone()) {
                if let Ok(Some(parent)) = self.vector_store.get(&fact.source_note_id).await {
                    if parent.is_active() {
                        // Clone the parent and prepend the matched fact text so the
                        // synthesis LLM sees the exact value. The clone prevents
                        // the access-tracking upsert from corrupting stored content.
                        let mut annotated = parent.clone();
                        annotated.content = format!(
                            "[Matched fact: {}]\n\n{}",
                            fact.content, annotated.content
                        );
                        fact_expanded.push(SearchResult {
                            note: annotated,
                            score: score + fact_boost,
                            linked_notes: Vec::new(),
                        });
                    }
                }
            }
        }

        if !fact_expanded.is_empty() {
            debug!(count = fact_expanded.len(), "Fact-expanded parent notes");
        }

        // --- Episode link traversal ---
        // For episode-drilled results, follow episode links to find related episodes
        let mut linked_digest_results: Vec<SearchResult> = Vec::new();
        for (episode_id, _) in &episode_hits {
            if let Ok(links) = self.graph_store.get_episode_links(episode_id).await {
                for (linked_ep_id, _link_type, _entity) in &links {
                    if let Ok(Some(digest)) = self.graph_store.get_episode_digest(linked_ep_id).await {
                        if let Some(ref note_id) = digest.digest_note_id {
                            if seen_note_ids.insert(note_id.clone()) {
                                if let Ok(Some(digest_note)) = self.vector_store.get(note_id).await {
                                    if digest_note.is_active() {
                                        linked_digest_results.push(SearchResult {
                                            note: digest_note,
                                            score: 0.5, // Moderate score for linked digests
                                            linked_notes: Vec::new(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !linked_digest_results.is_empty() {
            debug!(count = linked_digest_results.len(), "Episode-linked digest notes");
        }

        // --- Structured digest query ---
        // For Computation/Breadth queries, search episode digests' structured data directly.
        // Digests contain pre-computed aggregations, entity counts, and date ranges
        // that answer "how many X" and "what is the current Y" questions directly.
        let mut digest_matched_results: Vec<SearchResult> = Vec::new();
        if matches!(mode, QueryMode::Computation | QueryMode::Breadth | QueryMode::Recency) {
            let all_digests = self.graph_store.get_all_episode_digests().await?;
            let query_lower = query.to_lowercase();

            for digest in &all_digests {
                // Check if digest entities or aggregations mention query terms
                let mut matched = false;

                // Search entities
                for entity in &digest.entities {
                    if query_lower.contains(&entity.name.to_lowercase())
                        || entity.name.to_lowercase().contains(&query_lower.split_whitespace()
                            .filter(|w| w.len() > 3)
                            .next().unwrap_or(""))
                    {
                        matched = true;
                        break;
                    }
                }

                // Search aggregations
                if !matched {
                    for agg in &digest.aggregations {
                        if query_lower.contains(&agg.label.to_lowercase()) {
                            matched = true;
                            break;
                        }
                        for item in &agg.items {
                            if query_lower.contains(&item.to_lowercase()) {
                                matched = true;
                                break;
                            }
                        }
                    }
                }

                // Search digest text for query keywords
                if !matched && !digest.digest_text.is_empty() {
                    let digest_lower = digest.digest_text.to_lowercase();
                    let query_words: Vec<&str> = query_lower.split_whitespace()
                        .filter(|w| w.len() > 3)
                        .collect();
                    let match_count = query_words.iter()
                        .filter(|w| digest_lower.contains(*w))
                        .count();
                    if match_count >= 2 || (query_words.len() <= 3 && match_count >= 1) {
                        matched = true;
                    }
                }

                if matched {
                    if let Some(ref note_id) = digest.digest_note_id {
                        if seen_note_ids.insert(note_id.clone()) {
                            if let Ok(Some(digest_note)) = self.vector_store.get(note_id).await {
                                if digest_note.is_active() {
                                    digest_matched_results.push(SearchResult {
                                        note: digest_note,
                                        score: 0.8, // High score: structured match is strong signal
                                        linked_notes: Vec::new(),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            if !digest_matched_results.is_empty() {
                debug!(count = digest_matched_results.len(), "Structurally matched episode digests");
            }
        }

        // Merge: profiles -> structurally matched digests -> linked digests -> episode-drilled -> fact-expanded -> flat hits
        // Do NOT truncate here. Return the full expanded pool so that ask() can
        // rerank the entire candidate set before truncating to final top_k.
        // Access tracking is done by the caller (search() or ask()) after truncation,
        // so only notes actually returned to the user get their access time bumped.
        let mut results = profile_results;
        results.extend(digest_matched_results);
        results.extend(linked_digest_results);
        results.extend(episode_results);
        results.extend(fact_expanded);
        results.extend(flat_results);

        info!(
            results = results.len(),
            episode_drilldowns = episode_hits.len(),
            linked_total = results.iter().map(|r| r.linked_notes.len()).sum::<usize>(),
            "Search complete"
        );

        Ok((results, mode))
    }

    /// Public retrieve-only API. Runs Karta's full retrieval pipeline —
    /// query classification, wide search, reranking, deduplication,
    /// chronological ordering, contradiction force-retrieval, and context
    /// assembly — then returns the assembled notes WITHOUT calling any LLM
    /// for answer composition.
    ///
    /// Karta's job is to find and organize the right memories; the caller
    /// decides what to do with them (run their own LLM, display them, pass
    /// them to an agent, etc.). Use `ask()` if you want Karta to also
    /// compose an answer via its configured answer-LLM.
    pub async fn fetch_memories(&self, query: &str, top_k: usize) -> Result<FetchedMemories> {
        self.fetch_memories_with_clock(query, top_k, ClockContext::now()).await
    }

    pub async fn fetch_memories_with_clock(
        &self,
        query: &str,
        top_k: usize,
        ctx: ClockContext,
    ) -> Result<FetchedMemories> {
        let wide_k = top_k * self.config.summarization_top_k_multiplier.max(4);
        let mut reranker_best: Option<f32> = None;

        let (mut results, mode) = self.search_wide(query, wide_k, ctx).await?;

        let effective_top_k = match mode {
            QueryMode::Breadth => top_k * self.config.summarization_top_k_multiplier,
            QueryMode::Temporal => top_k * 4,
            QueryMode::Computation => top_k * 2,
            _ => top_k,
        };
        let mode_str = format!("{:?}", mode);

        if results.is_empty() {
            return Ok(FetchedMemories {
                query: query.to_string(),
                context: String::new(),
                notes: Vec::new(),
                note_ids: Vec::new(),
                query_mode: mode_str,
                contradiction_injected: 0,
                reranker_best_score: None,
            });
        }

        // --- Reranker ---
        if self.reranker_config.enabled {
            let notes_for_rerank: Vec<(MemoryNote, f32)> = results
                .iter()
                .take(self.reranker_config.max_rerank)
                .map(|r| (r.note.clone(), r.score))
                .collect();

            let reranked = self.reranker.rerank(query, notes_for_rerank).await?;
            let best_relevance = reranked
                .iter()
                .map(|r| r.relevance_score)
                .fold(0.0f32, f32::max);
            reranker_best = Some(best_relevance);

            if mode != QueryMode::Computation {
                let reranked_ids: HashSet<String> =
                    reranked.iter().map(|r| r.note.id.clone()).collect();
                let mut reordered: Vec<SearchResult> = Vec::new();
                for rr in &reranked {
                    let linked = results
                        .iter()
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
            }
        }

        results.truncate(effective_top_k);

        // Dedup unique notes (direct + linked)
        let mut seen: HashSet<String> = HashSet::new();
        let mut all_notes: Vec<MemoryNote> = Vec::new();
        for result in &results {
            if seen.insert(result.note.id.clone()) {
                all_notes.push(result.note.clone());
            }
            for linked in &result.linked_notes {
                if seen.insert(linked.id.clone()) {
                    all_notes.push(linked.clone());
                }
            }
        }

        if all_notes.is_empty() {
            return Ok(FetchedMemories {
                query: query.to_string(),
                context: String::new(),
                notes: Vec::new(),
                note_ids: Vec::new(),
                query_mode: mode_str,
                contradiction_injected: 0,
                reranker_best_score: reranker_best,
            });
        }

        // Sort chronologically (turn_index > source_timestamp > created_at)
        all_notes.sort_by(|a, b| match (a.turn_index, b.turn_index) {
            (Some(ai), Some(bi)) => ai.cmp(&bi),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.source_timestamp.cmp(&b.source_timestamp),
        });

        // Contradiction force-retrieval
        let mut contradiction_inject_ids: HashSet<String> = HashSet::new();
        let seen_ids: HashSet<String> = all_notes.iter().map(|n| n.id.clone()).collect();

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

        let mut inject_ids_vec: Vec<&str> = contradiction_inject_ids
            .iter()
            .map(|s| s.as_str())
            .collect();
        inject_ids_vec.truncate(10);
        let contradiction_notes: Vec<MemoryNote> = if inject_ids_vec.is_empty() {
            Vec::new()
        } else {
            self.vector_store
                .get_many(&inject_ids_vec)
                .await
                .unwrap_or_default()
        };

        // Build notes text with provenance markers
        let mut note_entries: Vec<String> = Vec::new();
        for (i, note) in all_notes.iter().enumerate() {
            note_entries.push(Self::format_note_entry(i, note, false));
        }
        let base_count = all_notes.len();
        for (i, note) in contradiction_notes.iter().enumerate() {
            note_entries.push(Self::format_note_entry(base_count + i, note, true));
        }
        let joined_notes = note_entries.join("\n\n");

        let events_block = self.build_events_block(mode).await;
        let notes_text = if events_block.is_empty() {
            joined_notes
        } else {
            format!("{}\n\n{}", events_block, joined_notes)
        };

        let contradiction_injected = contradiction_notes.len();
        let mut ordered = all_notes;
        ordered.extend(contradiction_notes);
        let note_ids: Vec<String> = ordered.iter().map(|n| n.id.clone()).collect();

        Ok(FetchedMemories {
            query: query.to_string(),
            context: notes_text,
            notes: ordered,
            note_ids,
            query_mode: mode_str,
            contradiction_injected,
            reranker_best_score: reranker_best,
        })
    }

    /// Format a single retrieved note as a context entry with provenance
    /// markers, date, and recency annotation. Shared between the retrieval
    /// path (`fetch_memories`) and the retry path inside `ask`.
    fn format_note_entry(i: usize, note: &MemoryNote, is_contradiction_source: bool) -> String {
        let provenance_marker = match &note.provenance {
            Provenance::Observed => "FACT".to_string(),
            Provenance::Dream { dream_type, confidence, .. } => {
                format!("INFERRED:{} conf={:.0}%", dream_type, confidence * 100.0)
            }
            Provenance::Profile { entity_id } => format!("PROFILE:{}", entity_id),
            Provenance::Episode { episode_id } => format!("EPISODE:{}", episode_id),
            Provenance::Fact { source_note_id } => format!(
                "FACT:from-{}",
                &source_note_id[..8.min(source_note_id.len())]
            ),
            Provenance::Digest { episode_id } => {
                format!("DIGEST:{}", &episode_id[..8.min(episode_id.len())])
            }
        };
        let display_time = note.source_timestamp;
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
        let prefix = if is_contradiction_source {
            "[CONTRADICTION SOURCE] "
        } else {
            ""
        };
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
    }

    /// Search + deduplicate + synthesize an answer with provenance markers.
    /// Includes abstention calibration: if no notes are sufficiently relevant, abstains.
    pub async fn ask(&self, query: &str, top_k: usize) -> Result<AskResult> {
        self.ask_with_clock(query, top_k, ClockContext::now()).await
    }

    /// Time-travel ask — recency anchored to ctx.reference_time().
    pub async fn ask_with_clock(
        &self,
        query: &str,
        top_k: usize,
        ctx: ClockContext,
    ) -> Result<AskResult> {
        // Fetch a wide pool from search_wide(), which classifies the query using
        // the embedding classifier. We pass a generous top_k so the pool is large
        // enough for any mode, then truncate based on the actual classified mode.
        let wide_k = top_k * self.config.summarization_top_k_multiplier.max(4);
        let mut reranker_best: Option<f32> = None;

        let (mut results, mode) = self.search_wide(query, wide_k, ctx).await?;

        let effective_top_k = match mode {
            QueryMode::Breadth => top_k * self.config.summarization_top_k_multiplier,
            QueryMode::Temporal => top_k * 4,
            QueryMode::Computation => top_k * 2,
            _ => top_k,
        };
        let mode_str = format!("{:?}", mode);

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

            // Log low-relevance signal but do NOT abstain here.
            // Let the synthesis model decide whether to answer or abstain based on
            // the actual note content. Hard-gating on reranker score causes false
            // abstention on queries where notes are relevant but use different vocabulary.
            if best_relevance < self.reranker_config.abstention_threshold {
                debug!(
                    best_relevance = best_relevance,
                    threshold = self.reranker_config.abstention_threshold,
                    "Reranker: low relevance signal (proceeding to synthesis)"
                );
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

        // Now truncate to final top_k after reranking has reordered the full pool
        results.truncate(effective_top_k);

        // Access tracking (fire-and-forget — don't block the read path)
        {
            let vs = Arc::clone(&self.vector_store);
            let ids: Vec<String> = results.iter().map(|r| r.note.id.clone()).collect();
            tokio::spawn(async move {
                for id in ids {
                    if let Ok(Some(mut note)) = vs.get(&id).await {
                        note.last_accessed_at = Utc::now();
                        let _ = vs.upsert(&note).await;
                    }
                }
            });
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
                (None, None) => a.source_timestamp.cmp(&b.source_timestamp),
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
                Provenance::Fact { source_note_id } => {
                    format!("FACT:from-{}", &source_note_id[..8.min(source_note_id.len())])
                }
                Provenance::Digest { episode_id } => {
                    format!("DIGEST:{}", &episode_id[..8.min(episode_id.len())])
                }
            };
            // Use source_timestamp (real conversation date) if available, fall back to created_at
            let display_time = note.source_timestamp;
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

        let joined_notes = note_entries.join("\n\n");

        // For date-arithmetic / ordering queries, prepend a structured EVENTS
        // block extracted at dream time from per-episode and cross-episode
        // digests. Dates in the digest are LLM-extracted from note content,
        // which is much more reliable than expecting the synthesis model to
        // pick dates out of scattered note excerpts at query time.
        let events_block = self.build_events_block(mode).await;
        let notes_text = if events_block.is_empty() {
            joined_notes
        } else {
            format!("{}\n\n{}", events_block, joined_notes)
        };

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

        let response = self.synth_llm().chat(&messages, &config).await?;

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

        // --- A.3: Insufficient-info retry for Computation/Temporal modes ---
        // When the LLM admits it can't find specific data (dates, counts), retry with
        // wider retrieval and no reranker. Only for modes where missing specific facts
        // is the failure mode. Standard/Breadth/Existence abstentions are likely correct.
        let retry_eligible = matches!(mode, QueryMode::Computation | QueryMode::Temporal);
        if retry_eligible && collected_count > 0 && Self::answer_admits_insufficient_info(&answer) {
            info!(
                query_mode = ?mode,
                notes_used = collected_count,
                "Answer admits insufficient info — retrying with 3x wider retrieval"
            );

            // Retry: 3x top_k, skip reranker entirely
            let retry_top_k = effective_top_k * 3;
            let retry_results = self.search(query, retry_top_k).await?;

            if !retry_results.is_empty() {
                // Build notes text from retry results (same dedup + sort + format logic)
                let mut retry_seen = HashSet::new();
                let mut retry_notes: Vec<&MemoryNote> = Vec::new();
                for result in &retry_results {
                    if retry_seen.insert(&result.note.id) {
                        retry_notes.push(&result.note);
                    }
                    for linked in &result.linked_notes {
                        if retry_seen.insert(&linked.id) {
                            retry_notes.push(linked);
                        }
                    }
                }

                // Sort chronologically
                retry_notes.sort_by(|a, b| {
                    match (a.turn_index, b.turn_index) {
                        (Some(ai), Some(bi)) => ai.cmp(&bi),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => a.source_timestamp.cmp(&b.source_timestamp),
                    }
                });

                let joined_retry: String = retry_notes.iter().enumerate()
                    .map(|(i, note)| format_note(i, note, false))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                let retry_events_block = self.build_events_block(mode).await;
                let retry_notes_text = if retry_events_block.is_empty() {
                    joined_retry
                } else {
                    format!("{}\n\n{}", retry_events_block, joined_retry)
                };

                let retry_messages = vec![
                    ChatMessage {
                        role: Role::System,
                        content: Prompts::synthesize_system().to_string(),
                    },
                    ChatMessage {
                        role: Role::User,
                        content: Prompts::synthesize_user(query, &retry_notes_text),
                    },
                ];

                info!(
                    retry_notes = retry_notes.len(),
                    original_notes = collected_count,
                    "Retry: synthesizing with wider note set"
                );

                if let Ok(retry_response) = self.synth_llm().chat(&retry_messages, &config).await {
                    let retry_parsed: serde_json::Value =
                        serde_json::from_str(&retry_response.content).unwrap_or_default();
                    if let Some(retry_answer) = retry_parsed["answer"].as_str() {
                        if !retry_answer.is_empty() && !Self::answer_admits_insufficient_info(retry_answer) {
                            let retry_note_ids: Vec<String> = retry_notes.iter().map(|n| n.id.clone()).collect();
                            let retry_has_contradiction = retry_parsed["has_contradiction"].as_bool().unwrap_or(false);

                            let mut final_answer = retry_answer.to_string();
                            if retry_has_contradiction {
                                final_answer = format!(
                                    "**Note: The memories contain contradictory information on this topic.**\n\n{}",
                                    final_answer
                                );
                            }

                            info!("Retry produced confident answer, using it");
                            return Ok(AskResult {
                                answer: final_answer,
                                query_mode: mode_str,
                                notes_used: retry_notes.len(),
                                note_ids: retry_note_ids,
                                contradiction_injected: 0,
                                has_contradiction: retry_has_contradiction,
                                reranker_best_score: reranker_best,
                            });
                        }
                    }
                }
                info!("Retry also insufficient or failed, keeping original answer");
            }
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

    /// Build a structured EVENTS block from stored episode digests and the
    /// cross-episode digest. Returns empty string if the query mode does not
    /// benefit from dated events, or if no digests exist yet (pre-dream).
    ///
    /// Events are merged across per-episode and cross-episode digests, deduped
    /// by (description, date), and sorted chronologically with undated events
    /// at the end. The block is prepended to the synthesis context so the LLM
    /// has a clean ISO-dated table to compute against for "how many days
    /// between X and Y" and "in what order did I bring up X" questions.
    async fn build_events_block(&self, mode: QueryMode) -> String {
        // Only applicable to date/sequence-heavy modes. Other modes retain
        // the narrow note-only context to avoid noise.
        if !matches!(mode, QueryMode::Computation | QueryMode::Temporal) {
            return String::new();
        }

        let per_episode = self.graph_store.get_all_episode_digests().await.unwrap_or_default();
        let cross_level = self.graph_store.get_all_cross_episode_digests().await.unwrap_or_default();

        if per_episode.is_empty() && cross_level.is_empty() {
            return String::new();
        }

        // Dedup key: (lowercased description, date-or-empty). Events from the
        // cross-episode digest are already deduped by the LLM, but per-episode
        // events can overlap with each other.
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut merged: Vec<crate::note::TimedEvent> = Vec::new();

        let push_event = |ev: &crate::note::TimedEvent,
                          seen: &mut HashSet<(String, String)>,
                          merged: &mut Vec<crate::note::TimedEvent>| {
            let key = (
                ev.description.to_lowercase(),
                ev.date.clone().unwrap_or_default(),
            );
            if seen.insert(key) {
                merged.push(ev.clone());
            }
        };

        for d in &per_episode {
            for ev in &d.events {
                push_event(ev, &mut seen, &mut merged);
            }
        }
        for d in &cross_level {
            for ev in &d.events {
                push_event(ev, &mut seen, &mut merged);
            }
        }

        if merged.is_empty() {
            return String::new();
        }

        // Sort: dated events chronologically, undated last (by source_turn if present)
        merged.sort_by(|a, b| match (&a.date, &b.date) {
            (Some(ad), Some(bd)) => ad.cmp(bd),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.source_turn.unwrap_or(u32::MAX).cmp(&b.source_turn.unwrap_or(u32::MAX)),
        });

        // Cap to keep the context tight. 400 events ≈ 6-8k tokens; smoke
        // run showed conv 1 produces ~324 dated events across per-episode +
        // cross-level digests, so 400 keeps all dated events and a margin of
        // undated ones for ordering queries.
        const MAX_EVENTS: usize = 400;
        if merged.len() > MAX_EVENTS {
            merged.truncate(MAX_EVENTS);
        }

        let mut lines: Vec<String> = Vec::with_capacity(merged.len() + 2);
        lines.push("EVENTS IN THIS CONVERSATION (chronological, dream-extracted):".to_string());
        for ev in &merged {
            let date = ev.date.as_deref().unwrap_or("undated");
            lines.push(format!("- {}: {}", date, ev.description));
        }
        lines.push(
            "\nFor date-arithmetic questions, prefer the dates above over dates mentioned inside note text.".to_string()
        );
        lines.join("\n")
    }

    /// Check if an answer contains language indicating the LLM couldn't find enough information.
    fn answer_admits_insufficient_info(answer: &str) -> bool {
        let lower = answer.to_lowercase();
        lower.contains("don't have information")
            || lower.contains("notes do not")
            || lower.contains("notes don't")
            || lower.contains("can't find")
            || lower.contains("can't determine")
            || lower.contains("cannot determine")
            || lower.contains("not explicitly stated")
            || lower.contains("not mentioned in")
            || lower.contains("not in the notes")
            || lower.contains("not provided in")
            || lower.contains("i don't see any note")
    }
}
