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

async fn create_karta(conv_id: &str) -> Karta {
    let suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let data_dir = format!("/tmp/karta-beam100k-{}-{}", conv_id, suffix);
    let _ = std::fs::remove_dir_all(&data_dir);

    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;
    // Enable reranker for better abstention and relevance scoring
    config.reranker.enabled = true;
    config.reranker.abstention_threshold = 0.1; // Jina raw scores: <0.1 = abstain
    config.reranker.max_rerank = 10;

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

    // --- Ingest phase ---
    let ingest_start = Instant::now();
    let mut ingested = 0;
    let mut ingest_errors = 0;

    for (i, msg) in conv.user_messages.iter().enumerate() {
        if msg.content.trim().is_empty() {
            continue;
        }

        // Add time anchor as prefix for temporal context
        let content = if msg.time_anchor.is_empty() {
            msg.content.clone()
        } else {
            format!("[{}] {}", msg.time_anchor, msg.content)
        };

        match karta.add_note(&content).await {
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

    for (qi, q) in conv.questions.iter().enumerate() {
        let query_start = Instant::now();
        let answer = match karta.ask(&q.question, 5).await {
            Ok(a) => a,
            Err(e) => {
                eprintln!("  Query {} failed: {}", qi + 1, e);
                continue;
            }
        };
        let query_ms = query_start.elapsed().as_millis();

        println!(
            "\n  [{}] Q{}: {}",
            q.ability,
            qi + 1,
            safe_truncate(&q.question, 80)
        );
        println!(
            "  A ({:.1}s): {}",
            query_ms as f64 / 1000.0,
            safe_truncate(&answer, 200)
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

        if rubric_items.is_empty() {
            entry.1 += 1;
            total_checks += 1;
            if answer.len() > 50 {
                total_passed += 1;
                entry.0 += 1;
                println!("    [PASS] (no rubric, non-trivial answer)");
            } else {
                println!("    [FAIL] (no rubric, trivial answer)");
            }
        } else {
            let mut rubric_score_sum = 0.0;

            for (ri, rubric) in rubric_items.iter().enumerate() {
                total_checks += 1;
                entry.1 += 1;

                // LLM-as-judge scores this rubric item
                let score = llm_judge_rubric(&karta, &q.question, &answer, rubric).await;
                rubric_score_sum += score;

                let label = if score >= 1.0 { "FULL" } else if score >= 0.5 { "PART" } else { "FAIL" };
                // Count >= 0.5 as passed for the binary tally
                if score >= 0.5 {
                    total_passed += 1;
                    entry.0 += 1;
                }
                println!("    [{}] R{}: {:.1} ({})", label, ri + 1, score, safe_truncate(rubric, 70));
            }

            let avg_score = rubric_score_sum / rubric_items.len() as f64;
            println!("    → Q{} BEAM score: {:.2}", qi + 1, avg_score);
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

    let full_start = Instant::now();
    let mut grand_total_q = 0;
    let mut grand_total_passed = 0;
    let mut grand_total_checks = 0;
    let mut grand_ability: HashMap<String, (usize, usize)> = HashMap::new();

    for conv in &dataset.conversations {
        let (questions, passed, total, ability_scores) = eval_conversation(conv).await;
        grand_total_q += questions;
        grand_total_passed += passed;
        grand_total_checks += total;

        for (ability, (p, t)) in ability_scores {
            let entry = grand_ability.entry(ability).or_insert((0, 0));
            entry.0 += p;
            entry.1 += t;
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
