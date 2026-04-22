use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::clock::ClockContext;
use crate::config::{EpisodeConfig, WriteConfig};
use crate::error::Result;
use crate::llm::{ChatMessage, GenConfig, LlmProvider, Prompts, Role};
use crate::note::{Episode, LinkDecision, MemoryNote, NoteAttributes, Provenance};
use crate::store::{GraphStore, VectorStore};
use crate::trace::{self, KnnCandidate, TraceEvent};

/// Handles the write path: index, link, evolve.
pub struct WriteEngine {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    llm: Arc<dyn LlmProvider>,
    config: WriteConfig,
    episode_config: EpisodeConfig,
}

impl WriteEngine {
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        llm: Arc<dyn LlmProvider>,
        config: WriteConfig,
        episode_config: EpisodeConfig,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            llm,
            config,
            episode_config,
        }
    }

    /// Live-default sugar over `add_note_with_clock`. Used by smoke tests
    /// and quick scripts that don't need to express a session, turn index, or
    /// reference time.
    pub async fn add_note(&self, content: &str) -> Result<MemoryNote> {
        self.add_note_with_clock(content, None, None, ClockContext::now())
            .await
    }

    /// Canonical write entrypoint. `session_id` carries through to episode
    /// segmentation; `turn_index` lets the read path order chronologically;
    /// `ctx.reference_time()` becomes the note's `source_timestamp` and the
    /// anchor for foresight TTL math at this turn.
    pub async fn add_note_with_clock(
        &self,
        content: &str,
        session_id: Option<&str>,
        turn_index: Option<u32>,
        ctx: ClockContext,
    ) -> Result<MemoryNote> {
        let note = self.add_note_inner(content, session_id, turn_index, ctx).await?;
        if session_id.is_some() {
            self.maybe_extend_episode(session_id.unwrap(), &note, ctx).await?;
        }
        Ok(note)
    }

    async fn add_note_inner(
        &self,
        content: &str,
        session_id: Option<&str>,
        turn_index: Option<u32>,
        ctx: ClockContext,
    ) -> Result<MemoryNote> {
        // Future-skew warning (codex #2). If the data's reference_time is
        // more than `future_skew_threshold_days` ahead of the wall clock,
        // log it but don't reject — production may have legitimate
        // client/server clock skew. Recency math clamps the negative-age
        // case at read time.
        let skew_days =
            (ctx.reference_time() - Utc::now()).num_seconds() as f64 / 86400.0;
        if skew_days > self.config.future_skew_threshold_days {
            warn!(
                skew_days,
                reference_time = %ctx.reference_time(),
                "ingest received future-dated reference_time"
            );
        }

        let preview_end = {
            let max = 60;
            let mut end = content.len().min(max);
            while end > 0 && !content.is_char_boundary(end) { end -= 1; }
            end
        };
        info!("Adding note: \"{}...\"", &content[..preview_end]);

        // 1. Generate attributes + embed raw content in parallel
        let content_owned = content.to_string();
        let attrs_fut = trace::stage("attrs", self.generate_attributes(content));
        let embed_fut = trace::stage("embed_raw", async {
            self.llm.embed(&[&content_owned]).await
        });
        let (attrs, raw_embeddings) = tokio::join!(attrs_fut, embed_fut);
        let attrs = attrs?;
        let raw_embedding = raw_embeddings?.into_iter().next().unwrap_or_default();
        debug!(context = %attrs.context, "Generated attributes");

        // 2. Create the note. source_timestamp = ctx.reference_time() —
        // every note carries the data's "now" at ingest. Live mode passes
        // ClockContext::now() so this matches Utc::now(); replay passes a
        // parsed source-clock instant.
        let mut note = MemoryNote::new(content.to_string());
        note.context = attrs.context;
        note.keywords = attrs.keywords;
        note.tags = attrs.tags;
        note.source_timestamp = ctx.reference_time();
        note.turn_index = turn_index;
        note.session_id = session_id.map(String::from);

        // 3. Use raw embedding for candidate search (fast path)
        let candidates = trace::stage("knn", async {
            let knn_start = std::time::Instant::now();
            let cands = self
                .vector_store
                .find_similar(&raw_embedding, self.config.top_k_candidates, &[&note.id])
                .await?;
            let wall_ms = knn_start.elapsed().as_millis() as u64;
            let heavy = trace::heavy();
            trace::try_emit(TraceEvent::KnnCandidates {
                ts: Utc::now(),
                turn_idx: trace::current_turn(),
                stage: trace::current_stage(),
                wall_ms,
                returned: cands.len(),
                candidates: if heavy {
                    Some(cands.iter().map(|(n, score)| KnnCandidate {
                        note_id: n.id.clone(),
                        score: *score,
                        content_preview: Some(n.content.chars().take(160).collect()),
                    }).collect())
                } else {
                    None
                },
            });
            Ok::<_, crate::error::KartaError>(cands)
        }).await?;

        // 4. Compute enriched embedding for storage (content + context + keywords)
        let embedding_text = format!(
            "{} {} {}",
            content,
            note.context,
            note.keywords.join(" ")
        );
        let enriched_embeddings = trace::stage("embed_enriched", async {
            self.llm.embed(&[&embedding_text]).await
        }).await?;
        note.embedding = enriched_embeddings.into_iter().next().unwrap_or_default();

        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|(_, score)| *score >= self.config.similarity_threshold)
            .collect();

        debug!(count = candidates.len(), "Found candidates above threshold");

        // 5. LLM decides which candidates to link
        let link_decisions = if !candidates.is_empty() {
            trace::stage("link", self.decide_links(content, &note.context, &candidates)).await?
        } else {
            Vec::new()
        };

        debug!(count = link_decisions.len(), "Link decisions made");

        // 6. Retroactive evolution (with drift protection)
        if self.config.evolve_linked_notes {
            for decision in &link_decisions {
                if let Some(existing) = self.vector_store.get(&decision.note_id).await? {
                    // Check evolution count — skip if over threshold (needs consolidation instead)
                    let evolution_history = self
                        .graph_store
                        .get_evolution_history(&existing.id)
                        .await?;

                    if evolution_history.len() >= self.config.max_evolutions_per_note {
                        debug!(
                            note_id = %existing.id,
                            evolutions = evolution_history.len(),
                            "Skipping evolution — note needs consolidation (drift protection)"
                        );
                        continue;
                    }

                    let prev_ctx = existing.context.clone();
                    let updated_context = trace::stage("evolve", self.evolve_context(
                        &existing.content,
                        &existing.context,
                        content,
                        &decision.reason,
                    )).await?;

                    // Record evolution in graph store
                    self.graph_store
                        .record_evolution(&existing.id, &note.id, &existing.context)
                        .await?;

                    let heavy = trace::heavy();
                    trace::try_emit(TraceEvent::Evolution {
                        ts: Utc::now(),
                        turn_idx: trace::current_turn(),
                        evolved_id: existing.id.clone(),
                        previous_context: if heavy { Some(prev_ctx) } else { None },
                        new_context: if heavy { Some(updated_context.clone()) } else { None },
                    });

                    // Update the note's context in vector store
                    let mut evolved = existing;
                    evolved.context = updated_context;
                    evolved.updated_at = chrono::Utc::now();
                    self.vector_store.upsert(&evolved).await?;

                    debug!(note_id = %evolved.id, "Evolved existing note");
                }
            }
        }

        // 7. Store the new note
        trace::stage("note_upsert", async {
            self.vector_store.upsert(&note).await
        }).await?;

        // 8. Store links (bidirectional)
        for decision in &link_decisions {
            self.graph_store
                .add_link(&note.id, &decision.note_id, &decision.reason)
                .await?;
            trace::try_emit(TraceEvent::LinkWritten {
                ts: Utc::now(),
                turn_idx: trace::current_turn(),
                from_id: note.id.clone(),
                to_id: decision.note_id.clone(),
                reason: decision.reason.clone(),
            });
        }

        note.links = link_decisions.iter().map(|d| d.note_id.clone()).collect();

        // 9. Store foresight signals extracted during attribute generation.
        // Both default TTL and DOA filter anchor to ctx.reference_time(), not
        // Utc::now() — a foresight extracted from a 2024 message must be
        // judged against 2024's "now", not the wall clock at replay time.
        let default_ttl = chrono::Duration::milliseconds(
            (self.config.foresight_default_ttl_days * 86_400_000.0) as i64,
        );
        let doa_window = chrono::Duration::milliseconds(
            (self.config.foresight_doa_threshold_days * 86_400_000.0) as i64,
        );
        let doa_horizon = ctx.reference_time() + doa_window;
        for signal in &attrs.foresight_signals {
            if !signal.content.is_empty() {
                let valid_until = signal
                    .valid_until
                    .as_deref()
                    .and_then(|s| {
                        chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                            .ok()
                            .and_then(|d| d.and_hms_opt(23, 59, 59))
                            .map(|dt| dt.and_utc())
                    })
                    .or_else(|| Some(ctx.reference_time() + default_ttl));

                // Foresight DOA filter (codex #2 partial): drop signals
                // whose valid_until is already inside the DOA window from
                // reference_time. Storing a foresight that's expired (or
                // about to expire) wastes a row and pollutes the active set
                // until the next dream cycle catches it.
                if let Some(until) = valid_until {
                    if until <= doa_horizon {
                        warn!(
                            valid_until = %until,
                            reference_time = %ctx.reference_time(),
                            "dropping DOA foresight"
                        );
                        continue;
                    }
                }

                let fs = crate::note::ForesightSignal::new(
                    signal.content.clone(),
                    note.id.clone(),
                    valid_until,
                );
                self.graph_store.upsert_foresight(&fs).await?;
                debug!(signal = %signal.content, "Stored foresight signal");
            }
        }

        // 10. Store atomic facts (each with its own embedding for fine-grained retrieval)
        if !attrs.atomic_facts.is_empty() {
            let fact_texts: Vec<&str> = attrs.atomic_facts.iter()
                .take(self.config.max_facts_per_note)
                .map(|f| f.content.as_str())
                .collect();

            let embed_result = trace::stage("embed_facts", async {
                self.llm.embed(&fact_texts).await
            }).await;

            match embed_result {
                Ok(fact_embeddings) => {
                    for (i, (extraction, embedding)) in attrs.atomic_facts.iter()
                        .take(5)
                        .zip(fact_embeddings)
                        .enumerate()
                    {
                        let mut fact = crate::note::AtomicFact::new(
                            extraction.content.clone(),
                            note.id.clone(),
                            i as u32,
                        );
                        fact.subject = extraction.subject.clone();
                        fact.embedding = embedding;
                        fact.created_at = note.created_at;

                        let fact_id = fact.id.clone();
                        let fact_content = fact.content.clone();
                        let _ = self.vector_store.upsert_fact(&fact).await;
                        let _ = self.graph_store.record_fact(
                            &fact.id, &note.id, i as u32, fact.subject.as_deref()
                        ).await;
                        let heavy = trace::heavy();
                        trace::try_emit(TraceEvent::FactWritten {
                            ts: Utc::now(),
                            turn_idx: trace::current_turn(),
                            fact_id,
                            source_note_id: note.id.clone(),
                            ordinal: i as u32,
                            content: if heavy { Some(fact_content) } else { None },
                        });
                    }
                    debug!(count = fact_texts.len(), note_id = %note.id, "Stored atomic facts");
                }
                Err(e) => {
                    debug!(error = %e, "Failed to embed atomic facts, skipping");
                }
            }
        }

        trace::try_emit(TraceEvent::NoteWritten {
            ts: Utc::now(),
            turn_idx: trace::current_turn(),
            note_id: note.id.clone(),
            link_count: note.links.len(),
        });

        info!(note_id = %note.id, links = note.links.len(), "Note stored");

        Ok(note)
    }

    /// Episode segmentation step. Called by `add_note_with_clock` when a
    /// session_id is supplied. Time-gap math anchors to `ctx.reference_time()`
    /// so replay sessions split into episodes the same way live ones do —
    /// without this, a 2024 message replayed in 2026 looks like a 2-year gap
    /// from any pre-existing episode and force-splits every turn.
    async fn maybe_extend_episode(
        &self,
        session_id: &str,
        note: &MemoryNote,
        ctx: ClockContext,
    ) -> Result<()> {
        if !self.episode_config.enabled {
            return Ok(());
        }

        let content = note.content.as_str();
        let episodes = self
            .graph_store
            .get_episodes_for_session(session_id)
            .await?;
        let current_episode = episodes.last();

        let should_new_episode = match current_episode {
            None => true,
            Some(ep) => {
                let time_gap = ctx
                    .reference_time()
                    .signed_duration_since(ep.end_time)
                    .num_seconds();

                if time_gap > self.episode_config.time_gap_threshold_secs {
                    true
                } else {
                    let last_note_content = if let Some(last_id) = ep.note_ids.last() {
                        self.vector_store
                            .get(last_id)
                            .await?
                            .map(|n| n.content.clone())
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };

                    if last_note_content.is_empty() {
                        false
                    } else {
                        trace::stage("episode_boundary", self.detect_episode_boundary(
                            &last_note_content,
                            content,
                            time_gap,
                        )).await?
                    }
                }
            }
        };

        if should_new_episode {
            let mut episode = Episode::new(session_id.to_string());
            episode.note_ids.push(note.id.clone());
            episode.start_time = ctx.reference_time();
            episode.end_time = ctx.reference_time();

            let (narrative, tags) = trace::stage("narrative_synth",
                self.synthesize_narrative(&[content])).await?;
            episode.narrative = narrative.clone();
            episode.topic_tags = tags;

            // Narrative-note source_timestamp = max(input.source_timestamp).
            // First note in episode → that's just this note's source_timestamp.
            let narrative_note = trace::stage("narrative_note",
                self.create_narrative_note(&narrative, &episode.id, note.source_timestamp)).await?;
            episode.narrative_note_id = Some(narrative_note.id.clone());

            self.graph_store.upsert_episode(&episode).await?;
            self.graph_store
                .add_note_to_episode(&note.id, &episode.id)
                .await?;

            debug!(episode_id = %episode.id, session = session_id, "Created new episode");
        } else if let Some(ep) = current_episode {
            self.graph_store
                .add_note_to_episode(&note.id, &ep.id)
                .await?;

            let all_note_ids = self
                .graph_store
                .get_notes_for_episode(&ep.id)
                .await?;
            let all_refs: Vec<&str> = all_note_ids.iter().map(|s| s.as_str()).collect();
            let all_notes = self.vector_store.get_many(&all_refs).await?;
            let contents: Vec<&str> = all_notes.iter().map(|n| n.content.as_str()).collect();

            // max(input.source_timestamp) over the episode's notes — bounded
            // claim for the narrative's effective freshness.
            let max_source_ts = all_notes
                .iter()
                .map(|n| n.source_timestamp)
                .max()
                .unwrap_or_else(|| note.source_timestamp);

            let (narrative, tags) = trace::stage("narrative_resynth",
                self.synthesize_narrative(&contents)).await?;

            let mut updated = ep.clone();
            updated.end_time = ctx.reference_time();
            updated.narrative = narrative.clone();
            updated.topic_tags = tags;

            if let Some(ref nar_id) = updated.narrative_note_id {
                if let Some(mut nar_note) = self.vector_store.get(nar_id).await? {
                    nar_note.content = narrative;
                    nar_note.updated_at = Utc::now();
                    nar_note.source_timestamp = max_source_ts;
                    let emb = trace::stage("embed_narrative", async {
                        self.llm.embed(&[&nar_note.content]).await
                    }).await?;
                    nar_note.embedding = emb.into_iter().next().unwrap_or_default();
                    self.vector_store.upsert(&nar_note).await?;
                }
            }

            self.graph_store.upsert_episode(&updated).await?;
            debug!(episode_id = %ep.id, notes = all_note_ids.len(), "Extended episode");
        }

        Ok(())
    }

    async fn detect_episode_boundary(
        &self,
        previous_content: &str,
        new_content: &str,
        time_gap_secs: i64,
    ) -> Result<bool> {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: Prompts::episode_boundary_system().to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: Prompts::episode_boundary_user(
                    previous_content,
                    new_content,
                    time_gap_secs,
                ),
            },
        ];

        let response = self.llm.chat(&messages, &GenConfig::default()).await?;
        let parsed: serde_json::Value =
            serde_json::from_str(&response.content).unwrap_or_default();

        // If sameEpisode is false, we need a new episode
        Ok(!parsed["sameEpisode"].as_bool().unwrap_or(true))
    }

    async fn synthesize_narrative(&self, note_contents: &[&str]) -> Result<(String, Vec<String>)> {
        let notes_text = note_contents
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n");

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: Prompts::episode_narrative_system().to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: Prompts::episode_narrative_user(&notes_text),
            },
        ];

        let response = self.llm.chat(&messages, &GenConfig::default()).await?;
        let parsed: serde_json::Value =
            serde_json::from_str(&response.content).unwrap_or_default();

        let narrative = parsed["narrative"]
            .as_str()
            .unwrap_or("Episode summary unavailable.")
            .to_string();
        let tags = parsed["topicTags"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        Ok((narrative, tags))
    }

    async fn create_narrative_note(
        &self,
        narrative: &str,
        episode_id: &str,
        source_timestamp: chrono::DateTime<Utc>,
    ) -> Result<MemoryNote> {
        let embeddings = trace::stage("embed_narrative", async {
            self.llm.embed(&[narrative]).await
        }).await?;
        let embedding = embeddings.into_iter().next().unwrap_or_default();

        // Narrative notes are inferences over a batch. source_timestamp =
        // max(input.source_timestamp) is the bounded claim — same shape as
        // dream notes (see dream/engine.rs persist_dream). Stamping with
        // Utc::now() would create the time-travel-confidence bug for
        // back-dated queries.
        let note = MemoryNote {
            id: Uuid::new_v4().to_string(),
            content: narrative.to_string(),
            context: format!("Episode narrative for episode {}", episode_id),
            keywords: vec!["episode".to_string(), "narrative".to_string()],
            tags: vec!["episode".to_string()],
            links: Vec::new(),
            embedding,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            evolution_history: Vec::new(),
            provenance: Provenance::Episode {
                episode_id: episode_id.to_string(),
            },
            confidence: 1.0,
            status: crate::note::NoteStatus::Active,
            last_accessed_at: Utc::now(),
            turn_index: None,
            source_timestamp,
            session_id: None,
        };

        self.vector_store.upsert(&note).await?;
        Ok(note)
    }

    async fn generate_attributes(&self, content: &str) -> Result<NoteAttributes> {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: Prompts::note_attributes_system().to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: Prompts::note_attributes_user(content),
            },
        ];

        let config = GenConfig {
            max_tokens: 2048,
            temperature: 0.0,
            json_mode: false,
            json_schema: Some(crate::llm::schemas::note_attributes_schema()),
        };

        let response = self.llm.chat(&messages, &config).await?;

        let parsed: serde_json::Value =
            serde_json::from_str(&response.content).unwrap_or_default();

        Ok(NoteAttributes {
            context: parsed["context"]
                .as_str()
                .unwrap_or(content)
                .to_string(),
            keywords: parsed["keywords"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            tags: parsed["tags"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            foresight_signals: parsed["foresight_signals"]
                .as_array()
                .or_else(|| parsed["foresightSignals"].as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| {
                            // Handle both old string format and new object format
                            if let Some(s) = v.as_str() {
                                Some(crate::note::ForesightExtraction {
                                    content: s.to_string(),
                                    valid_until: None,
                                })
                            } else {
                                Some(crate::note::ForesightExtraction {
                                    content: v["content"].as_str()?.to_string(),
                                    valid_until: v["valid_until"]
                                        .as_str()
                                        .map(String::from),
                                })
                            }
                        })
                        .collect()
                })
                .unwrap_or_default(),
            atomic_facts: parsed["atomic_facts"]
                .as_array()
                .or_else(|| parsed["atomicFacts"].as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| {
                            Some(crate::note::AtomicFactExtraction {
                                content: v["content"].as_str()?.to_string(),
                                subject: v["subject"].as_str().map(String::from),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default(),
        })
    }

    async fn decide_links(
        &self,
        new_content: &str,
        new_context: &str,
        candidates: &[(MemoryNote, f32)],
    ) -> Result<Vec<LinkDecision>> {
        let candidate_text: String = candidates
            .iter()
            .enumerate()
            .map(|(i, (note, _))| {
                format!(
                    "[{}] ID: {}\nContent: {}\nContext: {}",
                    i + 1,
                    note.id,
                    note.content,
                    note.context
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: Prompts::linking_system().to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: Prompts::linking_user(new_content, new_context, &candidate_text),
            },
        ];

        let response = self.llm.chat(&messages, &GenConfig::default()).await?;

        let parsed: serde_json::Value =
            serde_json::from_str(&response.content).unwrap_or_default();

        let links = parsed["links"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| {
                        Some(LinkDecision {
                            note_id: v["noteId"].as_str()?.to_string(),
                            reason: v["reason"].as_str().unwrap_or("").to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(links)
    }

    async fn evolve_context(
        &self,
        existing_content: &str,
        existing_context: &str,
        new_content: &str,
        link_reason: &str,
    ) -> Result<String> {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: Prompts::evolve_system().to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: Prompts::evolve_user(
                    existing_content,
                    existing_context,
                    new_content,
                    link_reason,
                ),
            },
        ];

        let response = self.llm.chat(&messages, &GenConfig::default()).await?;

        let parsed: serde_json::Value =
            serde_json::from_str(&response.content).unwrap_or_default();

        Ok(parsed["updatedContext"]
            .as_str()
            .unwrap_or(existing_context)
            .to_string())
    }
}
