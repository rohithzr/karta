use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::config::DreamConfig;
use crate::error::Result;
use crate::llm::{ChatMessage, GenConfig, LlmProvider, Prompts, Role};
use crate::note::{MemoryNote, Provenance};
use crate::store::{GraphStore, VectorStore};

use super::types::{DreamRecord, DreamRun, DreamType};

pub struct DreamEngine {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    llm: Arc<dyn LlmProvider>,
    config: DreamConfig,
}

impl DreamEngine {
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        llm: Arc<dyn LlmProvider>,
        config: DreamConfig,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            llm,
            config,
        }
    }

    pub async fn run(&self, scope_type: &str, scope_id: &str) -> Result<DreamRun> {
        let run_id = Uuid::new_v4().to_string();
        let started_at = Utc::now();
        let mut total_tokens: u64 = 0;
        let mut dreams: Vec<DreamRecord> = Vec::new();

        // Expire stale foresight signals
        let expired = self.graph_store.expire_foresights(Utc::now()).await?;
        if expired > 0 {
            debug!(expired = expired, "Expired foresight signals");
        }

        // Get dream cursor for incremental processing
        let cursor = self.graph_store.get_dream_cursor().await?;

        // Get all non-dream notes, filtered by cursor for incremental runs
        let all_notes: Vec<MemoryNote> = self
            .vector_store
            .get_all()
            .await?
            .into_iter()
            .filter(|n| !n.is_dream())
            .collect();

        let notes_to_process: Vec<&MemoryNote> = match cursor {
            Some(cursor_time) => {
                // Include notes created/updated after cursor AND their linked neighbors
                let mut relevant_ids: HashSet<String> = HashSet::new();

                for note in &all_notes {
                    if note.created_at > cursor_time || note.updated_at > cursor_time {
                        relevant_ids.insert(note.id.clone());
                        let links = self.graph_store.get_links(&note.id).await?;
                        relevant_ids.extend(links);
                    }
                }

                all_notes
                    .iter()
                    .filter(|n| relevant_ids.contains(&n.id))
                    .collect()
            }
            None => all_notes.iter().collect(),
        };

        let note_count = notes_to_process.len();
        info!(notes = note_count, "Starting dream pass");

        if notes_to_process.is_empty() {
            info!("No notes to dream about");
            return Ok(DreamRun {
                id: run_id,
                scope_type: scope_type.to_string(),
                scope_id: scope_id.to_string(),
                started_at,
                completed_at: Some(Utc::now()),
                notes_inspected: 0,
                dreams_attempted: 0,
                dreams_written: 0,
                dreams: Vec::new(),
                total_tokens_used: 0,
            });
        }

        // Build clusters from link graph
        let clusters = self.build_clusters(&notes_to_process).await?;
        debug!(clusters = clusters.len(), "Built clusters");

        let enabled: Vec<DreamType> = self
            .config
            .enabled_types
            .iter()
            .filter_map(|s| s.parse::<DreamType>().ok())
            .collect();

        // Per-cluster dreams (with dedup)
        for cluster in &clusters {
            if enabled.contains(&DreamType::Consolidation) && cluster.len() >= 3 {
                let (dream, tokens) = self.dream_consolidation(cluster).await?;
                total_tokens += tokens;
                if dream.would_write && !self.is_duplicate_dream(&dream).await? {
                    self.persist_dream(&dream).await?;
                }
                dreams.push(dream);
            }

            if enabled.contains(&DreamType::Deduction) && cluster.len() >= 2 {
                let (dream, tokens) = self.dream_deduction(cluster).await?;
                total_tokens += tokens;
                if dream.would_write && !self.is_duplicate_dream(&dream).await? {
                    self.persist_dream(&dream).await?;
                }
                dreams.push(dream);
            }

            if enabled.contains(&DreamType::Contradiction) && cluster.len() >= 2 {
                let (dream, tokens) = self.dream_contradiction(cluster).await?;
                total_tokens += tokens;
                if dream.would_write && !self.is_duplicate_dream(&dream).await? {
                    self.persist_dream(&dream).await?;
                }
                dreams.push(dream);
            }
        }

        // Cross-cluster dreams — run across sliding windows, not just first N
        let max = self.config.max_notes_per_prompt;

        if enabled.contains(&DreamType::Induction) && notes_to_process.len() >= 4 {
            // Run induction across multiple windows of notes
            for chunk in notes_to_process.chunks(max) {
                if chunk.len() < 4 {
                    continue;
                }
                let (dream, tokens) = self.dream_induction(chunk).await?;
                total_tokens += tokens;
                if dream.would_write && !self.is_duplicate_dream(&dream).await? {
                    self.persist_dream(&dream).await?;
                }
                dreams.push(dream);
            }
        }

        if enabled.contains(&DreamType::Abduction) && notes_to_process.len() >= 3 {
            for chunk in notes_to_process.chunks(max) {
                if chunk.len() < 3 {
                    continue;
                }
                let (dream, tokens) = self.dream_abduction(chunk).await?;
                total_tokens += tokens;
                if dream.would_write && !self.is_duplicate_dream(&dream).await? {
                    self.persist_dream(&dream).await?;
                }
                dreams.push(dream);
            }
        }

        // --- Episode Digests (Phase Next) ---
        if enabled.contains(&DreamType::EpisodeDigest) {
            let undigested = self.graph_store.get_undigested_episode_ids().await?;
            debug!(count = undigested.len(), "Undigested episodes found");

            for episode_id in &undigested {
                match self.graph_store.get_episode(episode_id).await {
                    Ok(Some(episode)) if episode.note_ids.is_empty() => {
                        let digest = crate::note::EpisodeDigest {
                            id: Uuid::new_v4().to_string(),
                            episode_id: episode.id.clone(),
                            entities: Vec::new(),
                            date_range: None,
                            aggregations: Vec::new(),
                            topic_sequence: Vec::new(),
                            digest_text: String::new(),
                            digest_note_id: None,
                            events: Vec::new(),
                            created_at: Utc::now(),
                        };
                        self.graph_store.upsert_episode_digest(&digest).await?;
                        debug!(episode_id = %episode_id, "Marked empty episode as digested");
                    }
                    Ok(Some(episode)) => {
                        let note_refs: Vec<&str> =
                            episode.note_ids.iter().map(|s| s.as_str()).collect();
                        match self.vector_store.get_many(&note_refs).await {
                            Ok(ep_notes) => {
                                match self.dream_episode_digest(&episode, &ep_notes).await {
                                    Ok((dream, tokens)) => {
                                        total_tokens += tokens;
                                        if dream.would_write {
                                            let _ = self.persist_dream(&dream).await;
                                        }
                                        dreams.push(dream);
                                    }
                                    Err(e) => {
                                        debug!(episode_id = %episode_id, error = %e, "Episode digest failed")
                                    }
                                }
                            }
                            Err(e) => {
                                debug!(episode_id = %episode_id, error = %e, "Failed to fetch notes for episode digest")
                            }
                        }
                    }
                    Ok(None) => debug!(episode_id = %episode_id, "Undigested episode id not found"),
                    Err(e) => {
                        debug!(episode_id = %episode_id, error = %e, "Failed to load episode for digest")
                    }
                }
            }
        }

        // --- Cross-Episode Digests (Phase Next) ---
        if enabled.contains(&DreamType::CrossEpisodeDigest) {
            let all_digests = self.graph_store.get_all_episode_digests().await?;
            if all_digests.len() >= 3 {
                for chunk in all_digests.chunks(10) {
                    if chunk.len() < 3 {
                        continue;
                    }
                    match self.dream_cross_episode_digest(chunk).await {
                        Ok((dream, tokens)) => {
                            total_tokens += tokens;
                            if dream.would_write
                                && !self.is_duplicate_dream(&dream).await.unwrap_or(false)
                            {
                                let _ = self.persist_dream(&dream).await;
                            }
                            dreams.push(dream);
                        }
                        Err(e) => debug!(error = %e, "Cross-episode digest failed"),
                    }
                }
            }
        }

        // Update dream cursor
        self.graph_store.set_dream_cursor(Utc::now()).await?;

        let dreams_written = dreams.iter().filter(|d| d.would_write).count();

        let run = DreamRun {
            id: run_id,
            scope_type: scope_type.to_string(),
            scope_id: scope_id.to_string(),
            started_at,
            completed_at: Some(Utc::now()),
            notes_inspected: note_count,
            dreams_attempted: dreams.len(),
            dreams_written,
            dreams,
            total_tokens_used: total_tokens,
        };

        // Record the run in graph store
        self.graph_store.record_dream_run(&run).await?;

        info!(
            attempted = run.dreams_attempted,
            written = run.dreams_written,
            tokens = total_tokens,
            "Dream pass complete"
        );

        Ok(run)
    }

    // --- Cluster building ---

    async fn build_clusters<'a>(
        &self,
        notes: &[&'a MemoryNote],
    ) -> Result<Vec<Vec<&'a MemoryNote>>> {
        // Union-find over the link graph
        let mut parent: HashMap<String, String> = HashMap::new();

        for note in notes {
            parent.insert(note.id.clone(), note.id.clone());
        }

        fn find(parent: &mut HashMap<String, String>, id: &str) -> String {
            let p = parent.get(id).cloned().unwrap_or_else(|| id.to_string());
            if p == id {
                return id.to_string();
            }
            let root = find(parent, &p);
            parent.insert(id.to_string(), root.clone());
            root
        }

        fn union(parent: &mut HashMap<String, String>, a: &str, b: &str) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent.insert(ra, rb.to_string());
            }
        }

        for note in notes {
            let links = self.graph_store.get_links(&note.id).await?;
            for link_id in &links {
                if parent.contains_key(link_id) {
                    union(&mut parent, &note.id, link_id);
                }
            }
        }

        // Group by root
        let mut clusters: HashMap<String, Vec<&'a MemoryNote>> = HashMap::new();
        for note in notes {
            let root = find(&mut parent, &note.id);
            clusters.entry(root).or_default().push(note);
        }

        // Only return clusters with 2+ notes
        Ok(clusters.into_values().filter(|c| c.len() >= 2).collect())
    }

    // --- Dream types ---

    fn format_notes(notes: &[&MemoryNote]) -> String {
        notes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                format!(
                    "[{}] ID: {}\nContent: {}\nContext: {}",
                    i + 1,
                    &n.id[..8.min(n.id.len())],
                    n.content,
                    n.context
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    async fn run_dream_prompt(&self, prompt: &str) -> Result<(serde_json::Value, u64)> {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt.to_string(),
        }];

        let config = GenConfig {
            max_tokens: 2000,
            ..Default::default()
        };

        let response = self.llm.chat(&messages, &config).await?;
        let parsed: serde_json::Value = serde_json::from_str(&response.content).unwrap_or_default();

        Ok((parsed, response.tokens_used))
    }

    async fn dream_deduction(&self, notes: &[&MemoryNote]) -> Result<(DreamRecord, u64)> {
        let capped: Vec<&MemoryNote> = notes
            .iter()
            .take(self.config.max_notes_per_prompt)
            .copied()
            .collect();
        let notes_text = Self::format_notes(&capped);
        let prompt = Prompts::dream_deduction(&notes_text);

        let (parsed, tokens) = self.run_dream_prompt(&prompt).await?;

        let reasoning = parsed["reasoning"].as_str().unwrap_or("").to_string();
        let conclusion = parsed["conclusion"].as_str().map(String::from);
        let confidence = parsed["confidence"].as_f64().unwrap_or(0.0) as f32;
        let would_write = conclusion.is_some() && confidence >= self.config.write_threshold;

        Ok((
            DreamRecord {
                id: Uuid::new_v4().to_string(),
                dream_type: DreamType::Deduction,
                source_note_ids: capped.iter().map(|n| n.id.clone()).collect(),
                reasoning,
                dream_content: conclusion.unwrap_or_else(|| "(no deduction possible)".into()),
                confidence,
                would_write,
                written_note_id: None,
                created_at: Utc::now(),
            },
            tokens,
        ))
    }

    async fn dream_induction(&self, notes: &[&MemoryNote]) -> Result<(DreamRecord, u64)> {
        let notes_text = Self::format_notes(notes);
        let prompt = Prompts::dream_induction(&notes_text);

        let (parsed, tokens) = self.run_dream_prompt(&prompt).await?;

        let reasoning = parsed["reasoning"].as_str().unwrap_or("").to_string();
        let generalisation = parsed["generalisation"].as_str().map(String::from);
        let confidence = parsed["confidence"].as_f64().unwrap_or(0.0) as f32;
        let would_write = generalisation.is_some() && confidence >= self.config.write_threshold;

        Ok((
            DreamRecord {
                id: Uuid::new_v4().to_string(),
                dream_type: DreamType::Induction,
                source_note_ids: notes.iter().map(|n| n.id.clone()).collect(),
                reasoning,
                dream_content: generalisation.unwrap_or_else(|| "(no pattern found)".into()),
                confidence,
                would_write,
                written_note_id: None,
                created_at: Utc::now(),
            },
            tokens,
        ))
    }

    async fn dream_abduction(&self, notes: &[&MemoryNote]) -> Result<(DreamRecord, u64)> {
        let notes_text = Self::format_notes(notes);
        let prompt = Prompts::dream_abduction(&notes_text);

        let (parsed, tokens) = self.run_dream_prompt(&prompt).await?;

        let reasoning = parsed["reasoning"].as_str().unwrap_or("").to_string();
        let hypothesis = parsed["hypothesis"].as_str().map(String::from);
        let confidence = parsed["confidence"].as_f64().unwrap_or(0.0) as f32;
        let would_write = hypothesis.is_some() && confidence >= self.config.write_threshold;

        let dream_id = Uuid::new_v4().to_string();

        // Emit foresight signal from hypothesis if it's forward-looking
        if let Some(ref h) = hypothesis
            && would_write
        {
            let fs = crate::note::ForesightSignal::new(
                h.clone(),
                dream_id.clone(),
                Some(chrono::Utc::now() + chrono::Duration::days(90)),
            );
            let _ = self.graph_store.upsert_foresight(&fs).await;
        }

        Ok((
            DreamRecord {
                id: dream_id,
                dream_type: DreamType::Abduction,
                source_note_ids: notes.iter().map(|n| n.id.clone()).collect(),
                reasoning,
                dream_content: hypothesis
                    .map(|h| format!("[HYPOTHESIS] {}", h))
                    .unwrap_or_else(|| "(no gap identified)".into()),
                confidence,
                would_write,
                written_note_id: None,
                created_at: Utc::now(),
            },
            tokens,
        ))
    }

    async fn dream_consolidation(&self, notes: &[&MemoryNote]) -> Result<(DreamRecord, u64)> {
        let capped: Vec<&MemoryNote> = notes
            .iter()
            .take(self.config.max_notes_per_prompt)
            .copied()
            .collect();
        let notes_text = Self::format_notes(&capped);
        let prompt = Prompts::dream_consolidation(&notes_text);

        let (parsed, tokens) = self.run_dream_prompt(&prompt).await?;

        let reasoning = parsed["reasoning"].as_str().unwrap_or("").to_string();
        let peer_card = parsed["peerCard"].as_str().map(String::from);
        let entity_id = parsed["entityId"].as_str().map(String::from);
        let confidence = parsed["confidence"].as_f64().unwrap_or(0.0) as f32;
        let would_write = peer_card.is_some() && confidence >= self.config.write_threshold;

        let dream_id = Uuid::new_v4().to_string();

        // Create or update entity profile if we have an entity ID and a valid peer card
        if would_write
            && let (Some(entity), Some(card)) = (&entity_id, &peer_card)
            && let Err(e) = self.upsert_profile(entity, card, &dream_id, &capped).await
        {
            debug!(error = %e, "Failed to upsert profile (non-fatal)");
        }

        Ok((
            DreamRecord {
                id: dream_id,
                dream_type: DreamType::Consolidation,
                source_note_ids: capped.iter().map(|n| n.id.clone()).collect(),
                reasoning,
                dream_content: peer_card.unwrap_or_else(|| "(consolidation failed)".into()),
                confidence,
                would_write,
                written_note_id: None,
                created_at: Utc::now(),
            },
            tokens,
        ))
    }

    /// Create or incrementally update an entity profile from a consolidation dream.
    async fn upsert_profile(
        &self,
        entity_id: &str,
        peer_card: &str,
        dream_id: &str,
        source_notes: &[&MemoryNote],
    ) -> Result<()> {
        let existing_profile_note_id = self.graph_store.get_profile_note_id(entity_id).await?;

        let profile_content = if let Some(ref existing_id) = existing_profile_note_id {
            // Merge with existing profile
            if let Some(existing_note) = self.vector_store.get(existing_id).await? {
                let merge_prompt = Prompts::profile_merge(&existing_note.content, peer_card);
                let messages = vec![ChatMessage {
                    role: Role::User,
                    content: merge_prompt,
                }];
                let config = GenConfig::default();
                let response = self.llm.chat(&messages, &config).await?;
                let parsed: serde_json::Value =
                    serde_json::from_str(&response.content).unwrap_or_default();
                parsed["updatedProfile"]
                    .as_str()
                    .unwrap_or(peer_card)
                    .to_string()
            } else {
                peer_card.to_string()
            }
        } else {
            peer_card.to_string()
        };

        // Create or update the profile note
        let profile_note_id = existing_profile_note_id.unwrap_or_else(|| dream_id.to_string());
        let embedding = self.llm.embed(&[&profile_content]).await?;

        let note = MemoryNote {
            id: profile_note_id.clone(),
            content: profile_content,
            context: format!("Entity profile for {}", entity_id),
            keywords: vec![entity_id.to_string(), "profile".to_string()],
            tags: vec!["profile".to_string()],
            links: Vec::new(),
            embedding: embedding.into_iter().next().unwrap_or_default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            evolution_history: Vec::new(),
            provenance: Provenance::Profile {
                entity_id: entity_id.to_string(),
            },
            confidence: 1.0,
            status: crate::note::NoteStatus::Active,
            last_accessed_at: Utc::now(),
            turn_index: None,
            source_timestamp: None,
        };

        self.vector_store.upsert(&note).await?;
        self.graph_store
            .upsert_profile(entity_id, &profile_note_id)
            .await?;

        // Link profile to source notes
        for source in source_notes {
            self.graph_store
                .add_link(&profile_note_id, &source.id, "profile source")
                .await?;
        }

        debug!(entity = entity_id, note_id = %profile_note_id, "Upserted entity profile");
        Ok(())
    }

    async fn dream_contradiction(&self, notes: &[&MemoryNote]) -> Result<(DreamRecord, u64)> {
        let capped: Vec<&MemoryNote> = notes
            .iter()
            .take(self.config.max_notes_per_prompt)
            .copied()
            .collect();
        let notes_text = Self::format_notes(&capped);
        let prompt = Prompts::dream_contradiction(&notes_text);

        let (parsed, tokens) = self.run_dream_prompt(&prompt).await?;

        let reasoning = parsed["reasoning"].as_str().unwrap_or("").to_string();
        let contradiction = parsed["contradiction"].as_str().map(String::from);
        let severity = parsed["severity"].as_str().unwrap_or("none").to_string();
        let confidence = parsed["confidence"].as_f64().unwrap_or(0.0) as f32;
        let is_real = contradiction.is_some() && severity != "none";
        let would_write = is_real && confidence >= self.config.write_threshold;

        Ok((
            DreamRecord {
                id: Uuid::new_v4().to_string(),
                dream_type: DreamType::Contradiction,
                source_note_ids: capped.iter().map(|n| n.id.clone()).collect(),
                reasoning,
                dream_content: contradiction
                    .map(|c| format!("[{}] {}", severity.to_uppercase(), c))
                    .unwrap_or_else(|| "(no contradiction)".into()),
                confidence: if is_real { confidence } else { 0.0 },
                would_write,
                written_note_id: None,
                created_at: Utc::now(),
            },
            tokens,
        ))
    }

    // --- Deduplication ---

    /// Check if a substantially similar dream already exists in the graph.
    /// Uses embedding similarity against existing dream notes of the same type.
    async fn is_duplicate_dream(&self, dream: &DreamRecord) -> Result<bool> {
        let embeddings = self.llm.embed(&[&dream.dream_content]).await?;
        let embedding = match embeddings.into_iter().next() {
            Some(e) => e,
            None => return Ok(false),
        };

        let candidates = self.vector_store.find_similar(&embedding, 5, &[]).await?;

        for (note, score) in &candidates {
            // High similarity + same dream type = duplicate
            if *score > 0.85
                && note.is_dream()
                && let Provenance::Dream { dream_type, .. } = &note.provenance
                && dream_type == dream.dream_type.as_str()
            {
                debug!(
                    dream_type = dream.dream_type.as_str(),
                    existing_id = %note.id,
                    similarity = score,
                    "Skipping duplicate dream"
                );
                return Ok(true);
            }
        }

        Ok(false)
    }

    // --- Persist dream as a note ---

    async fn persist_dream(&self, dream: &DreamRecord) -> Result<()> {
        let embedding_text = &dream.dream_content;
        let embeddings = self.llm.embed(&[embedding_text]).await?;
        let embedding = embeddings.into_iter().next().unwrap_or_default();

        let note = MemoryNote {
            id: dream.id.clone(),
            content: dream.dream_content.clone(),
            context: format!(
                "[{} dream, confidence {:.2}] {}",
                dream.dream_type.as_str(),
                dream.confidence,
                {
                    let max = 200;
                    let s = &dream.reasoning;
                    if s.len() <= max {
                        s
                    } else {
                        let mut end = max;
                        while end > 0 && !s.is_char_boundary(end) {
                            end -= 1;
                        }
                        &s[..end]
                    }
                }
            ),
            keywords: vec![
                dream.dream_type.as_str().to_string(),
                "dream".to_string(),
                "inference".to_string(),
            ],
            tags: vec!["dream".to_string(), dream.dream_type.as_str().to_string()],
            links: Vec::new(), // Links added below
            embedding,
            created_at: dream.created_at,
            updated_at: dream.created_at,
            evolution_history: Vec::new(),
            provenance: Provenance::Dream {
                dream_type: dream.dream_type.as_str().to_string(),
                source_note_ids: dream.source_note_ids.clone(),
                confidence: dream.confidence,
            },
            confidence: dream.confidence,
            status: crate::note::NoteStatus::Active,
            last_accessed_at: dream.created_at,
            turn_index: None,
            source_timestamp: None,
        };

        self.vector_store.upsert(&note).await?;

        // Link dream to source notes
        for source_id in &dream.source_note_ids {
            self.graph_store
                .add_link(
                    &dream.id,
                    source_id,
                    &format!("{} dream", dream.dream_type.as_str()),
                )
                .await?;
        }

        debug!(
            dream_type = dream.dream_type.as_str(),
            dream_id = %dream.id,
            "Persisted dream as note"
        );

        Ok(())
    }

    // --- Episode Digest Dreams (Phase Next) ---

    async fn dream_episode_digest(
        &self,
        episode: &crate::note::Episode,
        notes: &[MemoryNote],
    ) -> Result<(DreamRecord, u64)> {
        use crate::llm::Prompts;
        use crate::note::{
            AggregationEntry, DateRange, EntityMention, EpisodeDigest, NoteStatus, Provenance,
            TimedEvent,
        };

        let notes_text: String = notes
            .iter()
            .enumerate()
            .map(|(i, n)| format!("[{}] {}\n    Context: {}", i + 1, n.content, n.context))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = Prompts::episode_digest(&notes_text);
        let messages = vec![crate::llm::ChatMessage {
            role: crate::llm::Role::User,
            content: prompt,
        }];
        let config = crate::llm::GenConfig {
            max_tokens: 6144,
            temperature: 0.0,
            json_mode: true,
            json_schema: None,
        };

        let response = self.llm.chat(&messages, &config).await?;
        let tokens = response.tokens_used;
        let parsed: serde_json::Value = serde_json::from_str(&response.content).unwrap_or_default();

        let confidence = parsed["confidence"].as_f64().unwrap_or(0.7) as f32;
        let digest_text = parsed["digest_text"].as_str().unwrap_or("").to_string();

        // Parse structured fields
        let entities: Vec<EntityMention> = parsed["entities"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| {
                        Some(EntityMention {
                            name: v["name"].as_str()?.to_string(),
                            entity_type: v["type"].as_str().unwrap_or("other").to_string(),
                            count: v["count"].as_u64().unwrap_or(1) as u32,
                            latest_value: v["latest_value"].as_str().map(String::from),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let date_range = parsed["date_range"].as_object().and_then(|obj| {
            Some(DateRange {
                earliest: obj.get("earliest")?.as_str()?.to_string(),
                latest: obj.get("latest")?.as_str()?.to_string(),
            })
        });

        let aggregations: Vec<AggregationEntry> = parsed["aggregations"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| {
                        Some(AggregationEntry {
                            label: v["label"].as_str()?.to_string(),
                            count: v["count"].as_u64().unwrap_or(0) as u32,
                            items: v["items"]
                                .as_array()
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|i| i.as_str().map(String::from))
                                        .collect()
                                })
                                .unwrap_or_default(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let topic_sequence: Vec<String> = parsed["topic_sequence"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let events: Vec<TimedEvent> = parsed["timed_events"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| {
                        let description = v["description"].as_str()?.trim();
                        if description.is_empty() {
                            return None;
                        }
                        Some(TimedEvent {
                            description: description.to_string(),
                            date: v["date"].as_str().and_then(|s| {
                                let s = s.trim();
                                if s.is_empty() || s.eq_ignore_ascii_case("null") {
                                    None
                                } else {
                                    Some(s.to_string())
                                }
                            }),
                            source_turn: v["source_turn"].as_u64().map(|n| n as u32),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Create digest note for ANN searchability
        let mut digest_note_id = None;
        if !digest_text.is_empty() {
            let embedding = self
                .llm
                .embed(&[&digest_text])
                .await?
                .into_iter()
                .next()
                .unwrap_or_default();

            let note = MemoryNote {
                id: Uuid::new_v4().to_string(),
                content: digest_text.clone(),
                context: format!("Episode digest for episode {}", episode.id),
                keywords: vec!["digest".to_string(), "episode".to_string()],
                tags: vec!["digest".to_string()],
                links: Vec::new(),
                embedding,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                evolution_history: Vec::new(),
                provenance: Provenance::Digest {
                    episode_id: episode.id.clone(),
                },
                confidence,
                status: NoteStatus::Active,
                last_accessed_at: Utc::now(),
                turn_index: None,
                source_timestamp: None,
            };

            digest_note_id = Some(note.id.clone());
            self.vector_store.upsert(&note).await?;
        }

        // Store structured digest in SQLite
        let digest = EpisodeDigest {
            id: Uuid::new_v4().to_string(),
            episode_id: episode.id.clone(),
            entities,
            date_range,
            aggregations,
            topic_sequence,
            events,
            digest_text: digest_text.clone(),
            digest_note_id: digest_note_id.clone(),
            created_at: Utc::now(),
        };
        // Only store digest if the LLM produced meaningful content (guard against parse failures)
        if !digest_text.is_empty() {
            self.graph_store.upsert_episode_digest(&digest).await?;
        } else {
            debug!(episode_id = %episode.id, "Skipping empty digest (LLM parse failure)");
        }

        debug!(
            episode_id = %episode.id,
            entities = digest.entities.len(),
            aggregations = digest.aggregations.len(),
            events = digest.events.len(),
            "Episode digest created"
        );

        let dream = DreamRecord {
            id: Uuid::new_v4().to_string(),
            dream_type: DreamType::EpisodeDigest,
            source_note_ids: notes.iter().map(|n| n.id.clone()).collect(),
            reasoning: format!(
                "Digest for episode {} with {} notes",
                episode.id,
                notes.len()
            ),
            dream_content: digest_text,
            confidence,
            would_write: digest_note_id.is_some(),
            written_note_id: digest_note_id,
            created_at: Utc::now(),
        };

        Ok((dream, tokens))
    }

    async fn dream_cross_episode_digest(
        &self,
        digests: &[crate::note::EpisodeDigest],
    ) -> Result<(DreamRecord, u64)> {
        use crate::llm::Prompts;
        use crate::note::{
            AggregationEntry, CrossEpisodeDigest, EntityTimelineChange, EntityTimelineEntry,
            TimedEvent,
        };

        // Build mapping from display labels to real episode UUIDs
        // The LLM may return "Episode 1" or the UUID — we need to resolve both
        let mut label_to_uuid: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for (i, d) in digests.iter().enumerate() {
            label_to_uuid.insert(format!("Episode {}", i + 1), d.episode_id.clone());
            label_to_uuid.insert(d.episode_id.clone(), d.episode_id.clone()); // UUID maps to itself
        }

        // Pass the per-episode events through so the LLM can dedupe/merge them
        // instead of re-inferring from digest text alone.
        let digests_text: String = digests
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let events_line = if d.events.is_empty() {
                    String::new()
                } else {
                    let inline: Vec<String> = d
                        .events
                        .iter()
                        .take(20)
                        .map(|e| {
                            let date = e.date.as_deref().unwrap_or("?");
                            format!("{}: {}", date, e.description)
                        })
                        .collect();
                    format!("\n  Events: {}", inline.join(" | "))
                };
                format!(
                    "[Episode {} (id={})] {}\n  Entities: {}\n  Topics: {}{}",
                    i + 1,
                    d.episode_id,
                    d.digest_text,
                    d.entities
                        .iter()
                        .map(|e| format!("{}({})", e.name, e.count))
                        .collect::<Vec<_>>()
                        .join(", "),
                    d.topic_sequence.join(" → "),
                    events_line,
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = Prompts::cross_episode_digest(&digests_text);
        let messages = vec![crate::llm::ChatMessage {
            role: crate::llm::Role::User,
            content: prompt,
        }];
        let config = crate::llm::GenConfig {
            max_tokens: 6144,
            temperature: 0.0,
            json_mode: true,
            json_schema: None,
        };

        let response = self.llm.chat(&messages, &config).await?;
        let tokens = response.tokens_used;
        let parsed: serde_json::Value = serde_json::from_str(&response.content).unwrap_or_default();

        let confidence = parsed["confidence"].as_f64().unwrap_or(0.6) as f32;
        let digest_text = parsed["digest_text"].as_str().unwrap_or("").to_string();

        // Parse structured fields for persistent storage
        let entity_timeline: Vec<EntityTimelineEntry> = parsed["entity_timeline"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| {
                        let name = v["name"].as_str()?.to_string();
                        let entity_type = v["type"].as_str().unwrap_or("other").to_string();
                        let changes: Vec<EntityTimelineChange> = v["changes"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|c| {
                                        let raw = c["episode_id"].as_str()?;
                                        let resolved = label_to_uuid
                                            .get(raw)
                                            .cloned()
                                            .unwrap_or_else(|| raw.to_string());
                                        Some(EntityTimelineChange {
                                            episode_id: resolved,
                                            value: c["value"].as_str().unwrap_or("").to_string(),
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some(EntityTimelineEntry {
                            name,
                            entity_type,
                            changes,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let cross_aggregations: Vec<AggregationEntry> = parsed["cross_aggregations"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| {
                        Some(AggregationEntry {
                            label: v["label"].as_str()?.to_string(),
                            count: v["count"].as_u64().unwrap_or(0) as u32,
                            items: v["items"]
                                .as_array()
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|i| i.as_str().map(String::from))
                                        .collect()
                                })
                                .unwrap_or_default(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let cross_events: Vec<TimedEvent> = parsed["timed_events"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| {
                        let description = v["description"].as_str()?.trim();
                        if description.is_empty() {
                            return None;
                        }
                        Some(TimedEvent {
                            description: description.to_string(),
                            date: v["date"].as_str().and_then(|s| {
                                let s = s.trim();
                                if s.is_empty() || s.eq_ignore_ascii_case("null") {
                                    None
                                } else {
                                    Some(s.to_string())
                                }
                            }),
                            source_turn: v["source_turn"].as_u64().map(|n| n as u32),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let topic_progression: Vec<String> = parsed["topic_progression"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Create episode links from entity timelines
        for timeline in &entity_timeline {
            let episode_ids: Vec<String> = timeline
                .changes
                .iter()
                .map(|c| c.episode_id.clone())
                .collect();

            for window in episode_ids.windows(2) {
                if window.len() == 2 {
                    let has_value_change = timeline
                        .changes
                        .iter()
                        .filter(|c| c.episode_id == window[0] || c.episode_id == window[1])
                        .map(|c| c.value.as_str())
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        > 1;

                    let link_type = if has_value_change {
                        "value_update"
                    } else {
                        "entity_continuity"
                    };
                    let reason = format!("{} appears in both episodes", timeline.name);
                    let _ = self
                        .graph_store
                        .add_episode_link(
                            &window[0],
                            &window[1],
                            link_type,
                            Some(&timeline.name),
                            &reason,
                        )
                        .await;
                }
            }
        }

        // Persist structured cross-level digest so query-time retrieval can read it
        if !digest_text.is_empty() {
            let cross_digest = CrossEpisodeDigest {
                id: Uuid::new_v4().to_string(),
                scope_id: "default".to_string(),
                entity_timeline,
                cross_aggregations,
                events: cross_events.clone(),
                topic_progression,
                digest_text: digest_text.clone(),
                created_at: Utc::now(),
            };
            if let Err(e) = self
                .graph_store
                .upsert_cross_episode_digest(&cross_digest)
                .await
            {
                debug!(error = %e, "Failed to persist cross-episode digest");
            } else {
                debug!(
                    entities = cross_digest.entity_timeline.len(),
                    aggregations = cross_digest.cross_aggregations.len(),
                    events = cross_digest.events.len(),
                    "Cross-episode digest stored"
                );
            }
        }

        let source_ids: Vec<String> = digests
            .iter()
            .filter_map(|d| d.digest_note_id.clone())
            .collect();

        let dream = DreamRecord {
            id: Uuid::new_v4().to_string(),
            dream_type: DreamType::CrossEpisodeDigest,
            source_note_ids: source_ids,
            reasoning: format!(
                "Cross-episode digest across {} episodes ({} merged events)",
                digests.len(),
                cross_events.len()
            ),
            dream_content: digest_text,
            confidence,
            would_write: confidence >= self.config.write_threshold,
            written_note_id: None,
            created_at: Utc::now(),
        };

        Ok((dream, tokens))
    }
}
