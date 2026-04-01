//! BEAM 100K full benchmark — real dataset from HuggingFace.
//!
//! Dataset: https://huggingface.co/datasets/Mohammadta/BEAM
//! Preprocessed to JSON via data/convert_beam.py
//!
//! This ingests 2866 user messages across 20 conversations and answers 400
//! probing questions across all 10 BEAM memory abilities.
//!
//! Run: cargo test --test beam_100k -- --ignored --nocapture
//!
//! WARNING: This takes several hours with a real LLM due to the volume
//! of ingestion calls. Each user message = ~5s (attributes + embed + link + evolve).

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use karta_core::config::KartaConfig;
use karta_core::Karta;

#[derive(serde::Deserialize)]
struct BeamDataset {
    split: String,
    num_conversations: usize,
    total_questions: usize,
    conversations: Vec<BeamConversation>,
}

#[derive(serde::Deserialize)]
struct BeamConversation {
    id: String,
    category: String,
    title: String,
    user_messages: Vec<BeamMessage>,
    total_turns: usize,
    questions: Vec<BeamQuestion>,
}

#[derive(serde::Deserialize)]
struct BeamMessage {
    role: String,
    content: String,
    time_anchor: String,
}

#[derive(serde::Deserialize)]
struct BeamQuestion {
    ability: String,
    question: String,
    reference_answer: String,
    rubric: serde_json::Value,
}

/// Parse BEAM time_anchor strings into DateTime<Utc>.
/// BEAM uses formats like "March-15-2024", "2024-03-15", "March 15, 2024", etc.
fn parse_time_anchor(anchor: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    use chrono::NaiveDate;

    let anchor = anchor.trim();

    // Try ISO format: "2024-03-15"
    if let Ok(d) = NaiveDate::parse_from_str(anchor, "%Y-%m-%d") {
        return d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());
    }

    // Try "March-15-2024"
    if let Ok(d) = NaiveDate::parse_from_str(anchor, "%B-%d-%Y") {
        return d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());
    }

    // Try "March 15, 2024"
    if let Ok(d) = NaiveDate::parse_from_str(anchor, "%B %d, %Y") {
        return d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());
    }

    // Try "15-March-2024"
    if let Ok(d) = NaiveDate::parse_from_str(anchor, "%d-%B-%Y") {
        return d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());
    }

    None
}

fn load_dataset(path: &str) -> BeamDataset {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Cannot read {}. Run: python3 data/convert_beam.py data/beam-100k.parquet data/beam-100k.json", path));
    serde_json::from_str(&data).expect("Invalid JSON in BEAM dataset")
}

/// BEAM official judge prompt (verbatim from github.com/mohammadtavakoli78/BEAM src/prompts.py).
/// Scores 1.0 / 0.5 / 0.0 per rubric item.
const BEAM_JUDGE_PROMPT: &str = r#"You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that"#;

/// LLM-as-judge using the exact BEAM official prompt and 3-tier scoring (1.0/0.5/0.0).
/// Returns score 0.0, 0.5, or 1.0.
async fn llm_judge_rubric(
    karta: &Karta,
    question: &str,
    answer: &str,
    rubric_item: &str,
) -> f64 {
    use karta_core::llm::{ChatMessage, GenConfig, Role};

    // Build the prompt exactly as BEAM does: substitute placeholders
    let prompt = BEAM_JUDGE_PROMPT
        .replace("<question>", question)
        .replace("<rubric_item>", rubric_item)
        .replace("<llm_response>", safe_truncate(answer, 12000));

    let messages = vec![ChatMessage {
        role: Role::User,
        content: prompt,
    }];

    let config = GenConfig {
        max_tokens: 256,
        temperature: 0.0,
        json_mode: true,
        json_schema: None,
    };

    match karta.llm_chat(&messages, &config).await {
        Ok(response) => {
            let parsed: serde_json::Value =
                serde_json::from_str(&response.content).unwrap_or_default();
            let score = parsed["score"].as_f64().unwrap_or(0.0);
            // Clamp to valid BEAM scores
            if score >= 0.75 { 1.0 }
            else if score >= 0.25 { 0.5 }
            else { 0.0 }
        }
        Err(e) => {
            eprintln!("    Judge error: {}", e);
            0.0
        }
    }
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key).ok().map(|s| s == "1" || s == "true").unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

async fn create_karta(conv_id: &str) -> Karta {
    let suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let data_dir = format!("/tmp/karta-beam100k-{}-{}", conv_id, suffix);
    let _ = std::fs::remove_dir_all(&data_dir);

    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;

    // All knobs configurable via env vars for experiment sweeps
    config.episode.enabled = env_bool("K_EPISODE", true);
    config.read.episode_retrieval_enabled = env_bool("K_EPISODE_RETRIEVAL", true);
    config.read.graph_weight = env_f32("K_GRAPH_WEIGHT", 0.0);
    config.read.foresight_boost = env_f32("K_FORESIGHT_BOOST", 0.1);
    config.read.max_episode_drilldowns = env_usize("K_MAX_DRILLDOWNS", 3);
    config.read.max_notes_per_episode = env_usize("K_MAX_NOTES_EPISODE", 10);
    config.read.episode_drilldown_min_score = env_f32("K_DRILLDOWN_MIN", 0.25);
    config.read.recency_weight = env_f32("K_RECENCY_WEIGHT", 0.15);
    config.read.summarization_top_k_multiplier = env_usize("K_SUMM_TOPK_MULT", 3);
    config.reranker.enabled = env_bool("K_RERANKER", true);
    config.reranker.abstention_threshold = env_f32("K_ABSTENTION_THRESH", 0.01);
    config.reranker.max_rerank = env_usize("K_MAX_RERANK", 10);
    config.write.foresight_default_ttl_days = env_usize("K_FORESIGHT_TTL", 90) as i64;

    let exp = std::env::var("K_EXPERIMENT").unwrap_or_else(|_| "default".to_string());
    println!("  Config [{}]: episode={}, ep_retrieval={}, graph={}, foresight={}, reranker={}, abstention_thresh={}",
        exp, config.episode.enabled, config.read.episode_retrieval_enabled,
        config.read.graph_weight, config.read.foresight_boost,
        config.reranker.enabled, config.reranker.abstention_threshold);

    Karta::with_defaults(config)
        .await
        .expect("Failed to create Karta — check .env credentials")
}

fn is_stop_word(w: &str) -> bool {
    matches!(
        w,
        "should" | "would" | "could" | "about" | "their"
            | "there" | "which" | "where" | "these" | "those"
            | "based" | "response" | "mention" | "state" | "related"
            | "information" | "provided"
    )
}

fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Run the full BEAM 100K benchmark on a single conversation.
///
/// Returns (questions_asked, checks_passed, checks_total, ability_scores)
async fn eval_conversation(
    conv: &BeamConversation,
) -> (usize, usize, usize, HashMap<String, (usize, usize)>) {
    println!("\n{}", "=".repeat(70));
    println!(
        "BEAM 100K — Conv {} [{}]: {} user msgs, {} questions",
        conv.id,
        conv.category,
        conv.user_messages.len(),
        conv.questions.len()
    );
    println!("{}", "=".repeat(70));

    let karta = create_karta(&conv.id).await;

    // --- Ingest phase (session-aware for episode creation) ---
    let ingest_start = Instant::now();
    let mut ingested = 0;
    let mut ingest_errors = 0;
    let mut current_session = 0usize;
    let mut last_anchor = String::new();

    for (i, msg) in conv.user_messages.iter().enumerate() {
        if msg.content.trim().is_empty() {
            continue;
        }

        // Derive session boundaries from time_anchor changes
        if !msg.time_anchor.is_empty() && msg.time_anchor != last_anchor {
            current_session += 1;
            last_anchor = msg.time_anchor.clone();
        }
        let session_id = format!("session-{}", current_session);

        // Parse time_anchor into structured timestamp instead of text prefix
        let source_timestamp = if msg.time_anchor.is_empty() {
            None
        } else {
            parse_time_anchor(&msg.time_anchor)
        };

        // Still include time_anchor as text prefix for LLM context
        let content = if msg.time_anchor.is_empty() {
            msg.content.clone()
        } else {
            format!("[{}] {}", msg.time_anchor, msg.content)
        };

        match karta.add_note_with_metadata(&content, &session_id, Some(i as u32), source_timestamp).await {
            Ok(note) => {
                ingested += 1;
                if (i + 1) % 20 == 0 || i == 0 {
                    println!("  Ingested {}/{} notes ({} links)", i + 1, conv.user_messages.len(), note.links.len());
                }
            }
            Err(e) => {
                ingest_errors += 1;
                if ingest_errors <= 3 {
                    eprintln!("  WARN: Note {} failed: {}", i + 1, e);
                }
            }
        }
    }

    let ingest_ms = ingest_start.elapsed().as_millis();
    println!(
        "  Ingested {} notes in {:.1}s ({:.1}s/note, {} errors)",
        ingested,
        ingest_ms as f64 / 1000.0,
        ingest_ms as f64 / 1000.0 / ingested.max(1) as f64,
        ingest_errors
    );

    // --- Optional: run dreaming ---
    let dream_start = Instant::now();
    match karta.run_dreaming("beam100k", &conv.id).await {
        Ok(run) => {
            println!(
                "  Dreaming: {} attempted, {} written ({:.1}s)",
                run.dreams_attempted,
                run.dreams_written,
                dream_start.elapsed().as_millis() as f64 / 1000.0
            );
        }
        Err(e) => eprintln!("  Dreaming failed: {}", e),
    }

    // --- Query phase ---
    let mut ability_scores: HashMap<String, (usize, usize)> = HashMap::new();
    let mut total_passed = 0;
    let mut total_checks = 0;

    // JSONL debug log — one line per question with full answer, scores, metadata
    let run_ts = std::env::var("BEAM_RUN_ID").unwrap_or_else(|_| {
        chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string()
    });
    let debug_path = std::env::var("BEAM_DEBUG_PATH")
        .unwrap_or_else(|_| format!(".results/beam-debug-{}-{}.jsonl", conv.id, run_ts));
    let mut debug_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&debug_path)
        .ok();

    for (qi, q) in conv.questions.iter().enumerate() {
        let query_start = Instant::now();
        let ask_result = match karta.ask(&q.question, 5).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  Query {} failed: {}", qi + 1, e);
                continue;
            }
        };
        let query_ms = query_start.elapsed().as_millis();
        let answer = &ask_result.answer;

        println!(
            "\n  [{}] Q{}: {}",
            q.ability,
            qi + 1,
            safe_truncate(&q.question, 80)
        );
        println!(
            "  A ({:.1}s): {}",
            query_ms as f64 / 1000.0,
            safe_truncate(answer, 200)
        );

        // Score against rubric items using LLM-as-judge (BEAM official methodology).
        let rubric_items: Vec<String> = match &q.rubric {
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            serde_json::Value::String(s) => vec![s.clone()],
            _ => Vec::new(),
        };

        let entry = ability_scores.entry(q.ability.clone()).or_insert((0, 0));
        let mut rubric_scores_debug: Vec<serde_json::Value> = Vec::new();
        let mut beam_score: f64 = 0.0;

        if rubric_items.is_empty() {
            entry.1 += 1;
            total_checks += 1;
            if answer.len() > 50 {
                total_passed += 1;
                entry.0 += 1;
                println!("    [PASS] (no rubric, non-trivial answer)");
                beam_score = 1.0;
            } else {
                println!("    [FAIL] (no rubric, trivial answer)");
                beam_score = 0.0;
            }
        } else {
            let mut rubric_score_sum = 0.0;

            for (ri, rubric) in rubric_items.iter().enumerate() {
                total_checks += 1;
                entry.1 += 1;

                // LLM-as-judge scores this rubric item
                let score = llm_judge_rubric(&karta, &q.question, answer, rubric).await;
                rubric_score_sum += score;

                let label = if score >= 1.0 { "FULL" } else if score >= 0.5 { "PART" } else { "FAIL" };
                // Count >= 0.5 as passed for the binary tally
                if score >= 0.5 {
                    total_passed += 1;
                    entry.0 += 1;
                }
                println!("    [{}] R{}: {:.1} ({})", label, ri + 1, score, safe_truncate(rubric, 70));

                rubric_scores_debug.push(serde_json::json!({
                    "item": rubric,
                    "score": score,
                    "grade": label,
                }));
            }

            beam_score = rubric_score_sum / rubric_items.len() as f64;
            println!("    → Q{} BEAM score: {:.2}", qi + 1, beam_score);
        }

        // Write JSONL debug line
        if let Some(ref mut f) = debug_file {
            use std::io::Write;
            let debug_entry = serde_json::json!({
                "conv_id": conv.id,
                "category": conv.category,
                "q_index": qi,
                "ability": q.ability,
                "question": q.question,
                "reference_answer": q.reference_answer,
                "system_answer": answer,
                "query_mode": ask_result.query_mode,
                "notes_used": ask_result.notes_used,
                "note_ids": ask_result.note_ids,
                "contradiction_injected": ask_result.contradiction_injected,
                "has_contradiction": ask_result.has_contradiction,
                "reranker_best_score": ask_result.reranker_best_score,
                "rubric_scores": rubric_scores_debug,
                "beam_score": beam_score,
                "query_time_ms": query_ms,
            });
            let _ = writeln!(f, "{}", debug_entry);
        }
    }

    (conv.questions.len(), total_passed, total_checks, ability_scores)
}

/// Run BEAM 100K on a single conversation (for quick testing).
/// Picks the first conversation.
///
/// Run: cargo test --test beam_100k beam_100k_single -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn beam_100k_single() {
    let dataset_path = std::env::var("BEAM_DATASET_PATH")
        .unwrap_or_else(|_| "data/beam-100k.json".to_string());

    if !Path::new(&dataset_path).exists() {
        eprintln!("BEAM dataset not found at {}.", dataset_path);
        eprintln!("Run: python3 data/convert_beam.py data/beam-100k.parquet data/beam-100k.json");
        return;
    }

    let dataset = load_dataset(&dataset_path);
    let conv_index: usize = std::env::var("BEAM_CONV_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let conv = &dataset.conversations[conv_index];

    let (questions, passed, total, ability_scores) = eval_conversation(conv).await;

    println!("\n{}", "=".repeat(70));
    println!("BEAM 100K — SINGLE CONVERSATION RESULT");
    println!("  Methodology: BEAM official unified_llm_judge_base_prompt");
    println!("  Scoring: 1.0 / 0.5 / 0.0 per rubric item (3-tier)");
    println!("  Judge: same model as system (disclosed)");
    println!("{}", "=".repeat(70));
    println!("  Questions: {}", questions);
    println!("  Rubric checks: {}/{} passed (>= 0.5)", passed, total);

    let pass_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
    println!("  Pass rate: {:.1}%", pass_rate * 100.0);

    println!("\n  Per-ability (rubric items >= 0.5):");
    let mut abilities: Vec<_> = ability_scores.into_iter().collect();
    abilities.sort_by_key(|(name, _)| name.clone());
    for (name, (p, t)) in &abilities {
        let rate = if *t > 0 { *p as f64 / *t as f64 * 100.0 } else { 0.0 };
        println!("    {:30} {}/{} ({:.0}%)", name, p, t, rate);
    }
}

/// Run BEAM 100K on ALL 20 conversations.
///
/// WARNING: This takes several hours with a real LLM.
///
/// Run: cargo test --test beam_100k beam_100k_full -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn beam_100k_full() {
    let dataset_path = std::env::var("BEAM_DATASET_PATH")
        .unwrap_or_else(|_| "data/beam-100k.json".to_string());

    if !Path::new(&dataset_path).exists() {
        eprintln!("BEAM dataset not found at {}.", dataset_path);
        eprintln!("Run: python3 data/convert_beam.py data/beam-100k.parquet data/beam-100k.json");
        return;
    }

    let dataset = load_dataset(&dataset_path);
    println!("BEAM 100K Full Benchmark: {} conversations, {} questions",
        dataset.num_conversations, dataset.total_questions);

    let concurrency: usize = std::env::var("BEAM_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    println!("  Concurrency: {} conversations at a time", concurrency);

    let full_start = Instant::now();
    let mut grand_total_q = 0;
    let mut grand_total_passed = 0;
    let mut grand_total_checks = 0;
    let mut grand_ability: HashMap<String, (usize, usize)> = HashMap::new();

    // Run conversations in parallel batches using tokio JoinSet
    for chunk in dataset.conversations.chunks(concurrency) {
        let mut set = tokio::task::JoinSet::new();
        for conv in chunk {
            // Clone data into owned types for 'static lifetime
            let conv_id = conv.id.clone();
            let conv_category = conv.category.clone();
            let msgs: Vec<(String, String, String)> = conv
                .user_messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone(), m.time_anchor.clone()))
                .collect();
            let questions: Vec<(String, String, String, serde_json::Value)> = conv
                .questions
                .iter()
                .map(|q| {
                    (
                        q.ability.clone(),
                        q.question.clone(),
                        q.reference_answer.clone(),
                        q.rubric.clone(),
                    )
                })
                .collect();

            set.spawn(async move {
                // Reconstruct the conv reference types
                let owned_msgs: Vec<BeamMessage> = msgs
                    .iter()
                    .map(|(r, c, t)| BeamMessage {
                        role: r.clone(),
                        content: c.clone(),
                        time_anchor: t.clone(),
                    })
                    .collect();
                let owned_qs: Vec<BeamQuestion> = questions
                    .iter()
                    .map(|(a, q, ra, rub)| BeamQuestion {
                        ability: a.clone(),
                        question: q.clone(),
                        reference_answer: ra.clone(),
                        rubric: rub.clone(),
                    })
                    .collect();
                let owned_conv = BeamConversation {
                    id: conv_id,
                    category: conv_category,
                    title: String::new(),
                    user_messages: owned_msgs,
                    total_turns: 0,
                    questions: owned_qs,
                };
                eval_conversation(&owned_conv).await
            });
        }

        while let Some(result) = set.join_next().await {
            let (questions, passed, total, ability_scores) = result.expect("task panicked");
            grand_total_q += questions;
            grand_total_passed += passed;
            grand_total_checks += total;

            for (ability, (p, t)) in ability_scores {
                let entry = grand_ability.entry(ability).or_insert((0, 0));
                entry.0 += p;
                entry.1 += t;
            }
        }
    }

    let total_secs = full_start.elapsed().as_secs();

    println!("\n{}", "=".repeat(70));
    println!("BEAM 100K FULL RESULTS");
    println!("{}", "=".repeat(70));
    println!("  Conversations: {}", dataset.num_conversations);
    println!("  Questions:     {}", grand_total_q);
    println!("  Passed:        {}/{}", grand_total_passed, grand_total_checks);

    let pass_rate = if grand_total_checks > 0 {
        grand_total_passed as f64 / grand_total_checks as f64
    } else {
        0.0
    };
    println!("  Pass rate:     {:.1}%", pass_rate * 100.0);
    println!("  Total time:    {}m {}s", total_secs / 60, total_secs % 60);

    println!("\n  Per-ability:");
    let mut abilities: Vec<_> = grand_ability.into_iter().collect();
    abilities.sort_by_key(|(name, _)| name.clone());
    for (name, (p, t)) in &abilities {
        let rate = if *t > 0 { *p as f64 / *t as f64 * 100.0 } else { 0.0 };
        println!("    {:30} {}/{} ({:.0}%)", name, p, t, rate);
    }
}
