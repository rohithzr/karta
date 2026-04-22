use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};

use super::traits::*;
use crate::error::Result;

/// A mock LLM provider for testing.
///
/// Generates deterministic responses by parsing the prompt content:
/// - Attribute generation: extracts keywords from content
/// - Linking decisions: links candidates that share entity names
/// - Evolution: appends new context information
/// - Synthesis: concatenates relevant note content
/// - Dream prompts: returns structured JSON based on dream type
///
/// Embeddings use a simple hash-based vector so notes with shared
/// words have higher cosine similarity.
pub struct MockLlmProvider {
    call_count: AtomicU64,
}

impl MockLlmProvider {
    pub fn new() -> Self {
        Self {
            call_count: AtomicU64::new(0),
        }
    }

    fn extract_words(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() > 3)
            .collect()
    }

    /// Simple hash-to-vector: projects words into a fixed-dim space
    /// so texts sharing words have higher cosine similarity.
    fn text_to_embedding(text: &str, dim: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; dim];
        for word in Self::extract_words(text) {
            let mut hash: u64 = 5381;
            for byte in word.bytes() {
                hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
            }
            let idx = (hash as usize) % dim;
            vec[idx] += 1.0;
            // Spread to neighbors for smoother similarity
            vec[(idx + 1) % dim] += 0.5;
            vec[(idx + 2) % dim] += 0.25;
        }
        // Normalize
        let mag: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            for v in &mut vec {
                *v /= mag;
            }
        }
        vec
    }

    /// Scan for a bare `YYYY-MM-DD` substring and parse it into `(y, m, d)`.
    /// Returns `None` if no valid ISO date appears. Matches the plan's
    /// deterministic rule: ISO date → 1-day interval @ confidence 1.0.
    fn find_iso_date(content: &str) -> Option<(i32, u32, u32)> {
        let bytes = content.as_bytes();
        if bytes.len() < 10 {
            return None;
        }
        for i in 0..=bytes.len() - 10 {
            // Require word-ish boundary on the left.
            if i > 0 {
                let prev = bytes[i - 1];
                if prev.is_ascii_digit() || prev == b'-' {
                    continue;
                }
            }
            let w = &bytes[i..i + 10];
            if w[4] != b'-' || w[7] != b'-' {
                continue;
            }
            if !w[0..4].iter().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if !w[5..7].iter().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if !w[8..10].iter().all(|b| b.is_ascii_digit()) {
                continue;
            }
            // Require word-ish boundary on the right too.
            if i + 10 < bytes.len() {
                let next = bytes[i + 10];
                if next.is_ascii_digit() || next == b'-' {
                    continue;
                }
            }
            let s = std::str::from_utf8(w).ok()?;
            let y: i32 = s[0..4].parse().ok()?;
            let m: u32 = s[5..7].parse().ok()?;
            let d: u32 = s[8..10].parse().ok()?;
            if chrono::NaiveDate::from_ymd_opt(y, m, d).is_some() {
                return Some((y, m, d));
            }
        }
        None
    }

    /// Build one `atomic_facts` entry that matches the real LLM schema.
    /// Deterministic: ISO-date fact → 1-day interval @ confidence 1.0;
    /// otherwise null bounds @ confidence 0.0.
    fn emit_fact_with_occurred(content: &str) -> serde_json::Value {
        if let Some((y, m, d)) = Self::find_iso_date(content) {
            let start = chrono::NaiveDate::from_ymd_opt(y, m, d)
                .expect("date validated in find_iso_date")
                .and_hms_opt(0, 0, 0)
                .expect("midnight is valid")
                .and_utc();
            let end = start + chrono::Duration::days(1);
            serde_json::json!({
                "content": content,
                "subject": null,
                "occurred_start": start.to_rfc3339(),
                "occurred_end": end.to_rfc3339(),
                "occurred_confidence": 1.0,
            })
        } else {
            serde_json::json!({
                "content": content,
                "subject": null,
                "occurred_start": null,
                "occurred_end": null,
                "occurred_confidence": 0.0,
            })
        }
    }

    fn handle_attributes(&self, user_msg: &str) -> String {
        // The attribute prompt is "Reference time: ...\n\nIndex this memory:\n\n{content}".
        // Strip the framing so fact extraction sees only the actual content —
        // otherwise the reference-time ISO date leaks into every fact's
        // `occurred_*` bounds via our YYYY-MM-DD scanner.
        let content: &str = user_msg
            .split("Index this memory:")
            .nth(1)
            .map(|s| s.trim_start_matches(|c: char| c == '\n' || c == ' '))
            .unwrap_or(user_msg);
        let words = Self::extract_words(content);
        let keywords: Vec<String> = words.iter().take(8).cloned().collect();

        // Generate tags based on content
        let mut tags = Vec::new();
        let lower = content.to_lowercase();
        if lower.contains("prefer") || lower.contains("want") || lower.contains("request") {
            tags.push("preference");
        }
        if lower.contains("require") || lower.contains("must") || lower.contains("mandate") {
            tags.push("constraint");
        }
        if lower.contains("automat") || lower.contains("workflow") || lower.contains("pipeline") {
            tags.push("workflow");
        }
        if lower.contains("policy") || lower.contains("compliance") || lower.contains("audit") {
            tags.push("decision");
        }
        let entity_words: Vec<&str> = content
            .split_whitespace()
            .filter(|w| w.chars().next().is_some_and(|c| c.is_uppercase()) && w.len() > 1)
            .take(3)
            .collect();
        if !entity_words.is_empty() {
            tags.push("entity");
        }
        if tags.is_empty() {
            tags.push("pattern");
        }

        // Generate a richer context
        let first_sentence = content.split('.').next().unwrap_or(content);
        let context = format!(
            "{} — this is significant because it establishes constraints and context that affect related decisions and workflows.",
            first_sentence.trim()
        );

        // Extract foresight signals (forward-looking statements)
        let mut foresight: Vec<&str> = Vec::new();
        let lower = content.to_lowercase();
        if lower.contains("deadline") || lower.contains("by ") || lower.contains("before ") {
            foresight.push("upcoming deadline detected");
        }
        if lower.contains("scheduled") || lower.contains("starts ") || lower.contains("launch") {
            foresight.push("scheduled event detected");
        }
        if lower.contains("plan") || lower.contains("will ") || lower.contains("going to") {
            foresight.push("future plan detected");
        }

        // Atomic facts: one per non-empty sentence (up to 5). Each fact
        // carries `occurred_*` fields per the STEP1.5 schema — ISO-date
        // content gets an explicit 1-day interval, everything else gets
        // null bounds + 0.0 confidence. The real write path validates
        // `occurred_*` and drops malformed facts; emitting the paired
        // null/0.0 shape keeps mock-backed tests honest.
        let fact_sentences: Vec<String> = content
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .take(5)
            .map(|s| s.to_string())
            .collect();
        let atomic_facts: Vec<serde_json::Value> = if fact_sentences.is_empty() {
            vec![Self::emit_fact_with_occurred(content)]
        } else {
            fact_sentences
                .iter()
                .map(|s| Self::emit_fact_with_occurred(s))
                .collect()
        };

        serde_json::json!({
            "context": context,
            "keywords": keywords,
            "tags": tags,
            "foresightSignals": foresight,
            "atomic_facts": atomic_facts,
        })
        .to_string()
    }

    fn handle_linking(&self, user_msg: &str) -> String {
        // Parse candidate IDs from the prompt and link those that share
        // meaningful words with the new memory content.
        let parts: Vec<&str> = user_msg.split("Candidates:").collect();
        let new_content = parts.first().unwrap_or(&"");
        let candidates_text = parts.get(1).unwrap_or(&"");

        let new_words: std::collections::HashSet<String> = Self::extract_words(new_content)
            .into_iter()
            .filter(|w| w.len() > 4)
            .collect();

        let mut links = Vec::new();

        // Parse each candidate block
        for block in candidates_text.split("[").skip(1) {
            // Extract the ID
            if let Some(id_line) = block.lines().find(|l| l.contains("ID:")) {
                let id = id_line
                    .split("ID:")
                    .nth(1)
                    .unwrap_or("")
                    .trim()
                    .to_string();

                if id.is_empty() {
                    continue;
                }

                let candidate_words: std::collections::HashSet<String> =
                    Self::extract_words(block)
                        .into_iter()
                        .filter(|w| w.len() > 4)
                        .collect();

                let shared: Vec<&String> = new_words.intersection(&candidate_words).collect();
                if shared.len() >= 2 {
                    links.push(serde_json::json!({
                        "noteId": id,
                        "reason": format!("shared context: {}", shared.iter().take(3).map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
                    }));
                }
            }
        }

        serde_json::json!({ "links": links }).to_string()
    }

    fn handle_evolution(&self, user_msg: &str) -> String {
        let existing_ctx = user_msg
            .split("Current context:")
            .nth(1)
            .and_then(|s| s.split("New related memory:").next())
            .unwrap_or("")
            .trim();

        let new_memory = user_msg
            .split("New related memory:")
            .nth(1)
            .and_then(|s| s.split("Link reason:").next())
            .unwrap_or("")
            .trim();

        let new_first_sentence = new_memory.split('.').next().unwrap_or(new_memory).trim();

        let updated = format!(
            "{} Additionally, this connects to the fact that {}.",
            existing_ctx, new_first_sentence
        );

        serde_json::json!({ "updatedContext": updated }).to_string()
    }

    fn handle_synthesis(&self, user_msg: &str) -> String {
        // Extract query and all note content, return a comprehensive answer
        let query = user_msg
            .split("Query:")
            .nth(1)
            .and_then(|s| s.split("Relevant memories:").next())
            .unwrap_or("")
            .trim();

        let notes_section = user_msg.split("Relevant memories:").nth(1).unwrap_or("");

        // Collect all note contents
        let mut note_contents = Vec::new();
        for block in notes_section.split("[").skip(1) {
            let content = block
                .split("Context:")
                .next()
                .unwrap_or(block)
                .trim()
                .trim_start_matches(|c: char| c.is_numeric() || c == ']' || c == ' ');
            if !content.is_empty() {
                note_contents.push(content.trim().to_string());
            }
        }

        // Build an answer that includes content from all notes
        let mut answer = format!("Based on the available memories regarding \"{}\": ", query);
        for (i, content) in note_contents.iter().enumerate() {
            answer.push_str(&format!("[{}] {} ", i + 1, content));
        }

        answer
    }

    fn handle_dream(&self, prompt: &str) -> String {
        let notes_section = prompt.split("Notes:").nth(1).unwrap_or("");
        let note_count = notes_section.matches("[").count();

        if prompt.contains("deductive") || prompt.contains("LOGICALLY NECESSARY") {
            let conclusion = if note_count >= 2 {
                "Based on the linked facts, these constraints are interconnected and must be satisfied together in any solution design."
            } else {
                return serde_json::json!({
                    "reasoning": "Insufficient linked notes for deduction",
                    "conclusion": null,
                    "confidence": 0.0
                }).to_string();
            };
            serde_json::json!({
                "reasoning": "Analyzing the linked notes reveals logically connected constraints that entail a combined conclusion.",
                "conclusion": conclusion,
                "confidence": 0.75
            })
            .to_string()
        } else if prompt.contains("inductive") || prompt.contains("REPEATED patterns") {
            serde_json::json!({
                "reasoning": "Multiple notes follow a pattern of enterprise requirements demanding structured, auditable, and compliant approaches.",
                "generalisation": "Enterprise customers consistently require audit trails, compliance verification, and structured notification patterns over ad-hoc approaches.",
                "confidence": 0.72,
                "supportingNoteCount": note_count
            })
            .to_string()
        } else if prompt.contains("gaps") || prompt.contains("CONSPICUOUSLY ABSENT") {
            serde_json::json!({
                "reasoning": "The notes describe requirements and constraints but lack explicit information about timeline commitments and resource allocation.",
                "hypothesis": "There may be unstated timeline or resource constraints that could affect the feasibility of meeting all stated requirements simultaneously.",
                "confidence": 0.68
            })
            .to_string()
        } else if prompt.contains("peer card") || prompt.contains("consolidation") {
            // Extract a plausible entity name from the notes
            let entity = notes_section
                .split_whitespace()
                .find(|w| w.len() > 2 && w.chars().next().is_some_and(|c| c.is_uppercase()))
                .unwrap_or("Unknown");
            serde_json::json!({
                "reasoning": "These notes cluster around the same entity or project with interconnected requirements.",
                "entityId": entity,
                "peerCard": format!("{} has multiple interconnected requirements spanning technical constraints, compliance needs, and operational preferences that must be addressed holistically.", entity),
                "confidence": 0.80
            })
            .to_string()
        } else if prompt.contains("CONTRADICT") || prompt.contains("consistency checker") {
            // Check if notes actually have contradictory signals
            let lower = notes_section.to_lowercase();
            let has_conflict = (lower.contains("eu") && lower.contains("us-east"))
                || (lower.contains("real-time") && lower.contains("batch"))
                || (lower.contains("nightly") && lower.contains("2 minute"));

            if has_conflict {
                serde_json::json!({
                    "reasoning": "The notes contain directly conflicting requirements or constraints.",
                    "contradiction": "There is a fundamental conflict between stated requirements that cannot both be satisfied simultaneously.",
                    "severity": "critical",
                    "confidence": 0.85
                })
                .to_string()
            } else {
                serde_json::json!({
                    "reasoning": "Notes are consistent with each other.",
                    "contradiction": null,
                    "severity": "none",
                    "confidence": 0.0
                })
                .to_string()
            }
        } else {
            serde_json::json!({
                "reasoning": "Unknown dream type",
                "conclusion": null,
                "confidence": 0.0
            })
            .to_string()
        }
    }
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn chat(&self, messages: &[ChatMessage], _config: &GenConfig) -> Result<ChatResponse> {
        self.call_count.fetch_add(1, Ordering::Relaxed);

        let system_msg = messages
            .iter()
            .find(|m| matches!(m.role, Role::System))
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let user_msg = messages
            .iter()
            .find(|m| matches!(m.role, Role::User))
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // For dream prompts (no system message, just user prompt)
        let full_prompt = if system_msg.is_empty() { user_msg } else { "" };

        let content = if system_msg.contains("Resolve the user's temporal query") {
            // Tier 2 temporal resolver. Deterministic mock: if the query
            // mentions "last spring" return a wide range at low
            // confidence; else return null bounds at confidence 0.0 so
            // the validator treats it as "no temporal content".
            let response = if user_msg.contains("last spring") {
                serde_json::json!({
                    "occurred_start": "2024-03-01T00:00:00Z",
                    "occurred_end":   "2024-06-01T00:00:00Z",
                    "occurred_confidence": 0.5
                })
            } else {
                serde_json::json!({
                    "occurred_start": null,
                    "occurred_end": null,
                    "occurred_confidence": 0.0
                })
            };
            response.to_string()
        } else if system_msg.contains("memory indexing system") {
            self.handle_attributes(user_msg)
        } else if system_msg.contains("should be linked") {
            self.handle_linking(user_msg)
        } else if system_msg.contains("Update the existing memory") {
            self.handle_evolution(user_msg)
        } else if system_msg.to_lowercase().contains("answer questions using only") {
            self.handle_synthesis(user_msg)
        } else if full_prompt.contains("updating an entity profile")
            || user_msg.contains("updating an entity profile")
        {
            // Profile merge
            let new_info = user_msg.split("New information:").nth(1).unwrap_or(user_msg);
            serde_json::json!({
                "updatedProfile": format!("Updated profile incorporating new information. {}", new_info.trim().chars().take(200).collect::<String>())
            })
            .to_string()
        } else if system_msg.contains("same conversational episode")
            || full_prompt.contains("same conversational episode")
        {
            // Episode boundary detection
            serde_json::json!({ "sameEpisode": true, "reason": "same topic" }).to_string()
        } else if system_msg.contains("narrative summary")
            || full_prompt.contains("narrative summary")
        {
            // Episode narrative synthesis
            serde_json::json!({
                "narrative": "This episode covers a series of related discussions.",
                "topicTags": ["general"]
            })
            .to_string()
        } else if full_prompt.contains("deductive")
            || full_prompt.contains("inductive")
            || full_prompt.contains("gaps")
            || full_prompt.contains("peer card")
            || full_prompt.contains("consistency checker")
            || full_prompt.contains("LOGICALLY NECESSARY")
            || full_prompt.contains("REPEATED patterns")
            || full_prompt.contains("CONSPICUOUSLY ABSENT")
            || full_prompt.contains("CONTRADICT")
        {
            self.handle_dream(if full_prompt.is_empty() { user_msg } else { full_prompt })
        } else {
            // Fallback
            serde_json::json!({
                "context": user_msg,
                "keywords": [],
                "tags": ["unknown"]
            })
            .to_string()
        };

        Ok(ChatResponse {
            content,
            tokens_used: 100,
            input_tokens: 60,
            output_tokens: 40,
        })
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| Self::text_to_embedding(t, 1536))
            .collect())
    }

    fn model_id(&self) -> &str {
        "mock-model"
    }

    fn embedding_model_id(&self) -> &str {
        "mock-embedding"
    }
}
