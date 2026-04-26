//! LongMemEval benchmark — long-term interactive memory evaluation.
//!
//! Paper: "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
//! Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
//! GitHub: https://github.com/xiaowu0162/LongMemEval
//!
//! Preprocessed to JSON via data/convert_longmem.py
//!
//! LongMemEval tests 5 core long-term memory abilities:
//!   1. Information Extraction (single-session-user, single-session-assistant, single-session-preference)
//!   2. Multi-Session Reasoning
//!   3. Knowledge Updates
//!   4. Temporal Reasoning
//!   5. Abstention (correctly refusing unanswerable questions)
//!
//! The oracle split contains only evidence sessions per question (~2-5 sessions each),
//! making it fastest for development. The S split has ~40 sessions (~115k tokens) and
//! M has ~500 sessions (~1.5M tokens) per question.
//!
//! Run single:  cargo test --test longmem longmem_single -- --ignored --nocapture
//! Run full:    cargo test --test longmem longmem_full -- --ignored --nocapture

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use karta_core::Karta;
use karta_core::config::KartaConfig;
use karta_core::llm::{ChatMessage, GenConfig, Role};

// ---------------------------------------------------------------------------
// Data model matching the JSON from convert_longmem.py
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct LongMemDataset {
    benchmark: String,
    split: String,
    total_questions: usize,
    questions: Vec<LongMemEntry>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct LongMemEntry {
    question_id: String,
    question_type: String,
    question: String,
    #[serde(deserialize_with = "deserialize_string_or_number")]
    answer: String,
    #[serde(default)]
    question_date: String,
    user_messages: Vec<LongMemMessage>,
    #[serde(default)]
    num_sessions: usize,
}

#[derive(serde::Deserialize)]
struct LongMemMessage {
    role: String,
    content: String,
    #[serde(default)]
    date: String,
    #[serde(default)]
    #[allow(dead_code)]
    session_id: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn deserialize_string_or_number<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    struct StringOrNumber;
    impl<'de> de::Visitor<'de> for StringOrNumber {
        type Value = String;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("string or number")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<String, E> {
            Ok(v)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<String, E> {
            Ok(v.to_string())
        }
    }
    deserializer.deserialize_any(StringOrNumber)
}

fn load_dataset(path: &str) -> LongMemDataset {
    let data = std::fs::read_to_string(path).unwrap_or_else(|_| {
        panic!(
            "Cannot read {}. Run: python3 data/convert_longmem.py oracle data/longmemeval-oracle.json",
            path
        )
    });
    serde_json::from_str(&data).expect("Invalid JSON in LongMemEval dataset")
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

async fn create_karta(label: &str) -> Karta {
    let suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let data_dir = format!("/tmp/karta-longmem-{}-{}", label, suffix);
    let _ = std::fs::remove_dir_all(&data_dir);

    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;
    config.reranker.enabled = true;
    config.reranker.abstention_threshold = 0.1; // Jina raw scores: <0.1 = abstain
    config.reranker.max_rerank = 10;

    Karta::with_defaults(config)
        .await
        .expect("Failed to create Karta — check .env credentials")
}

/// LLM-as-judge following LongMemEval's official evaluation methodology.
///
/// Uses per-question-type prompts from the paper's evaluate_qa.py:
/// - Standard questions: check if response contains the correct answer
/// - Temporal reasoning: allow off-by-one errors on day counts
/// - Knowledge update: accept responses that include updated answer alongside old info
/// - Preference: check if response satisfies the rubric/personalization
/// - Abstention: check if model correctly identifies question as unanswerable
///
/// Returns true if the judge says "yes".
async fn llm_judge(
    karta: &Karta,
    question_type: &str,
    question: &str,
    reference_answer: &str,
    model_response: &str,
    is_abstention: bool,
) -> bool {
    let prompt = if is_abstention {
        format!(
            "I will give you an unanswerable question, an explanation, and a response from a \
             model. Please answer yes if the model correctly identifies the question as \
             unanswerable. The model could say that the information is incomplete, or some \
             other information is given but the asked information is not.\n\n\
             Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n\
             Does the model correctly identify the question as unanswerable? Answer yes or no only.",
            question,
            reference_answer,
            safe_truncate(model_response, 3000)
        )
    } else {
        match question_type {
            "temporal-reasoning" => format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, \
                 answer no. If the response is equivalent to the correct answer or contains all \
                 the intermediate steps to get the correct answer, you should also answer yes. \
                 If the response only contains a subset of the information required by the \
                 answer, answer no. In addition, do not penalize off-by-one errors for the \
                 number of days. If the question asks for the number of days/weeks/months, \
                 etc., and the model makes off-by-one errors (e.g., predicting 19 days when \
                 the answer is 18), the model's response is still correct.\n\n\
                 Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question,
                reference_answer,
                safe_truncate(model_response, 3000)
            ),
            "knowledge-update" => format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, \
                 answer no. If the response contains some previous information along with an \
                 updated answer, the response should be considered as correct as long as the \
                 updated answer is the required answer.\n\n\
                 Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question,
                reference_answer,
                safe_truncate(model_response, 3000)
            ),
            "single-session-preference" => format!(
                "I will give you a question, a rubric for desired personalized response, and a \
                 response from a model. Please answer yes if the response satisfies the desired \
                 response. Otherwise, answer no. The model does not need to reflect all the \
                 points in the rubric. The response is correct as long as it recalls and \
                 utilizes the user's personal information correctly.\n\n\
                 Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question,
                reference_answer,
                safe_truncate(model_response, 3000)
            ),
            // single-session-user, single-session-assistant, multi-session
            _ => format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, \
                 answer no. If the response is equivalent to the correct answer or contains all \
                 the intermediate steps to get the correct answer, you should also answer yes. \
                 If the response only contains a subset of the information required by the \
                 answer, answer no.\n\n\
                 Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question,
                reference_answer,
                safe_truncate(model_response, 3000)
            ),
        }
    };

    let messages = vec![ChatMessage {
        role: Role::User,
        content: prompt,
    }];

    let config = GenConfig {
        max_tokens: 16,
        temperature: 0.0,
        json_mode: false,
        json_schema: None,
    };

    match karta.llm_chat(&messages, &config).await {
        Ok(response) => response.content.to_lowercase().contains("yes"),
        Err(e) => {
            eprintln!("    WARN: Judge call failed: {}", e);
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Core evaluation logic
// ---------------------------------------------------------------------------

/// Per-category results tracker.
struct CategoryResults {
    scores: HashMap<String, Vec<bool>>,
}

impl CategoryResults {
    fn new() -> Self {
        Self {
            scores: HashMap::new(),
        }
    }

    fn record(&mut self, category: &str, correct: bool) {
        self.scores
            .entry(category.to_string())
            .or_default()
            .push(correct);
    }

    fn merge(&mut self, other: &CategoryResults) {
        for (cat, scores) in &other.scores {
            self.scores.entry(cat.clone()).or_default().extend(scores);
        }
    }

    fn total_correct(&self) -> usize {
        self.scores
            .values()
            .flat_map(|v| v.iter())
            .filter(|&&b| b)
            .count()
    }

    fn total_count(&self) -> usize {
        self.scores.values().map(|v| v.len()).sum()
    }

    fn print_report(&self, header: &str) {
        println!("\n{}", "=".repeat(70));
        println!("{}", header);
        println!("{}", "=".repeat(70));

        let total = self.total_count();
        let correct = self.total_correct();
        let accuracy = if total > 0 {
            correct as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        println!("  Total:    {}/{} ({:.1}%)", correct, total, accuracy);
        println!("\n  Per-category:");

        let mut categories: Vec<_> = self.scores.keys().collect();
        categories.sort();

        for cat in categories {
            let scores = &self.scores[cat];
            let c = scores.iter().filter(|&&b| b).count();
            let t = scores.len();
            let rate = if t > 0 {
                c as f64 / t as f64 * 100.0
            } else {
                0.0
            };
            println!("    {:40} {}/{} ({:.0}%)", cat, c, t, rate);
        }
    }
}

/// Evaluate a single LongMemEval entry.
///
/// 1. Ingest all user messages from all sessions as notes (with date prefix for temporal context)
/// 2. Ask the evaluation question via karta.ask()
/// 3. Use LLM-as-judge to score the answer
async fn eval_entry(entry: &LongMemEntry, results: &mut CategoryResults) {
    println!(
        "\n  --- [{}] {} ---",
        entry.question_type, entry.question_id
    );

    let karta = create_karta(&entry.question_id).await;

    // --- Ingest phase: feed all user messages from all sessions ---
    let ingest_start = Instant::now();
    let mut ingested = 0;
    let mut errors = 0;

    for msg in &entry.user_messages {
        {
            if msg.role != "user" {
                continue;
            }
            if msg.content.trim().is_empty() {
                continue;
            }

            let content = if msg.date.is_empty() {
                msg.content.clone()
            } else {
                format!("[{}] {}", msg.date, msg.content)
            };

            match karta.add_note(&content).await {
                Ok(_) => ingested += 1,
                Err(e) => {
                    errors += 1;
                    if errors <= 2 {
                        eprintln!("    WARN: Ingest failed: {}", e);
                    }
                }
            }
        }
    }

    let ingest_ms = ingest_start.elapsed().as_millis();
    println!(
        "    Ingested {} notes in {:.1}s ({} errors)",
        ingested,
        ingest_ms as f64 / 1000.0,
        errors
    );

    // --- Query phase ---
    let query_start = Instant::now();
    let answer = match karta.ask(&entry.question, 5).await {
        Ok(result) => result.answer,
        Err(e) => {
            eprintln!("    Query failed: {}", e);
            results.record(&entry.question_type, false);
            return;
        }
    };
    let query_ms = query_start.elapsed().as_millis();

    println!("    Q: {}", safe_truncate(&entry.question, 100));
    println!("    Expected: {}", safe_truncate(&entry.answer, 100));
    println!(
        "    Got ({:.1}s): {}",
        query_ms as f64 / 1000.0,
        safe_truncate(&answer, 200)
    );

    // --- Judge phase ---
    let correct = llm_judge(
        &karta,
        &entry.question_type,
        &entry.question,
        &entry.answer,
        &answer,
        entry.question_type.contains("_abs"),
    )
    .await;

    let label = if correct { "PASS" } else { "FAIL" };
    println!("    [{}]", label);

    results.record(&entry.question_type, correct);
}

fn get_dataset_path() -> String {
    std::env::var("LONGMEM_DATASET_PATH")
        .unwrap_or_else(|_| "data/longmemeval-oracle.json".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Run LongMemEval on a single entry (first question) for quick testing.
///
/// Run: cargo test --test longmem longmem_single -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn longmem_single() {
    let dataset_path = get_dataset_path();

    if !Path::new(&dataset_path).exists() {
        eprintln!("LongMemEval dataset not found at {}.", dataset_path);
        eprintln!("Run: python3 data/convert_longmem.py oracle data/longmemeval-oracle.json");
        return;
    }

    let dataset = load_dataset(&dataset_path);
    println!(
        "LongMemEval [{}]: {} questions",
        dataset.split, dataset.total_questions
    );

    let entry = &dataset.questions[0];
    let mut results = CategoryResults::new();

    eval_entry(entry, &mut results).await;

    results.print_report("LONGMEMEVAL SINGLE RESULT");
}

/// Run LongMemEval on the first 10 questions for a quick sample.
///
/// Run: cargo test --test longmem longmem_ten -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn longmem_ten() {
    let dataset_path = get_dataset_path();

    if !Path::new(&dataset_path).exists() {
        eprintln!("LongMemEval dataset not found at {}.", dataset_path);
        return;
    }

    let dataset = load_dataset(&dataset_path);
    let count = dataset.questions.len().min(10);
    println!(
        "LongMemEval [{}]: running first {} of {} questions",
        dataset.split, count, dataset.total_questions
    );

    let mut results = CategoryResults::new();

    for (i, entry) in dataset.questions.iter().take(count).enumerate() {
        println!("\n{} [{}/{}]", "=".repeat(70), i + 1, count,);
        eval_entry(entry, &mut results).await;
    }

    results.print_report("LONGMEMEVAL 10-QUESTION RESULT");
}

/// Run LongMemEval on ALL questions (500 in the full dataset).
///
/// WARNING: This takes many hours with a real LLM. Each entry requires
/// ingesting multiple sessions of messages plus a query + judge call.
///
/// Run: cargo test --test longmem longmem_full -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn longmem_full() {
    let dataset_path = get_dataset_path();

    if !Path::new(&dataset_path).exists() {
        eprintln!("LongMemEval dataset not found at {}.", dataset_path);
        eprintln!("Run: python3 data/convert_longmem.py oracle data/longmemeval-oracle.json");
        return;
    }

    let dataset = load_dataset(&dataset_path);
    println!(
        "LongMemEval Full [{}]: {} questions",
        dataset.split, dataset.total_questions,
    );

    let full_start = Instant::now();
    let mut results = CategoryResults::new();

    for (i, entry) in dataset.questions.iter().enumerate() {
        println!(
            "\n{} [{}/{}] {}",
            "=".repeat(70),
            i + 1,
            dataset.total_questions,
            entry.question_id
        );

        let mut entry_results = CategoryResults::new();
        eval_entry(entry, &mut entry_results).await;
        results.merge(&entry_results);

        // Print running totals every 10 entries
        if (i + 1) % 10 == 0 {
            let c = results.total_correct();
            let t = results.total_count();
            let rate = if t > 0 {
                c as f64 / t as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "\n  Running total: {}/{} ({:.1}%) after {} entries",
                c,
                t,
                rate,
                i + 1
            );
        }
    }

    let total_secs = full_start.elapsed().as_secs();
    println!("\n  Total time: {}m {}s", total_secs / 60, total_secs % 60);

    results.print_report("LONGMEMEVAL FULL RESULTS");
}
