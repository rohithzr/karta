//! LOCOMO benchmark harness — long-term conversational memory evaluation.
//!
//! Dataset: Maharana et al., "LoCoMo: Long-Context Conversational Memory", ACL 2024
//!          https://github.com/snap-research/locomo
//!
//! Preprocessed to JSON via data/convert_locomo.py
//!
//! 5 question categories:
//!   1. single_hop    — factual recall from one turn
//!   2. multi_hop     — requires combining info across turns
//!   3. temporal      — time-based reasoning (dates, ordering)
//!   4. open_domain   — commonsense / open-ended
//!   5. adversarial   — deliberately misleading questions
//!
//! Run single:  cargo test --test locomo locomo_single -- --ignored --nocapture
//! Run full:    cargo test --test locomo locomo_full   -- --ignored --nocapture

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use karta_core::Karta;
use karta_core::config::KartaConfig;

// ---------------------------------------------------------------------------
// Data model (mirrors output of data/convert_locomo.py)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct LocomoDataset {
    benchmark: String,
    num_conversations: usize,
    total_questions: usize,
    total_messages: usize,
    conversations: Vec<LocomoConversation>,
}

#[derive(serde::Deserialize)]
struct LocomoConversation {
    id: String,
    speaker_a: String,
    speaker_b: String,
    num_sessions: usize,
    messages: Vec<LocomoMessage>,
    questions: Vec<LocomoQuestion>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct LocomoMessage {
    speaker: String,
    text: String,
    dia_id: String,
    session_index: usize,
    timestamp: String,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct LocomoQuestion {
    question: String,
    #[serde(deserialize_with = "deserialize_string_or_number")]
    answer: String,
    category: String,
    category_id: u32,
    evidence: Vec<String>,
    #[serde(default)]
    adversarial_answer: Option<String>,
}

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_dataset(path: &str) -> LocomoDataset {
    let data = std::fs::read_to_string(path).unwrap_or_else(|_| {
        panic!(
            "Cannot read {}. Download the raw data and run:\n  \
             curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
             -o data/locomo10_raw.json\n  \
             python3 data/convert_locomo.py data/locomo10_raw.json data/locomo.json",
            path
        )
    });
    serde_json::from_str(&data).expect("Invalid JSON in LOCOMO dataset")
}

async fn create_karta(conv_id: &str) -> Karta {
    let suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let data_dir = format!("/tmp/karta-locomo-{}-{}", conv_id, suffix);
    let _ = std::fs::remove_dir_all(&data_dir);

    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;

    Karta::with_defaults(config)
        .await
        .expect("Failed to create Karta — check .env credentials")
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

// ---------------------------------------------------------------------------
// LLM-as-judge scoring
// ---------------------------------------------------------------------------

/// Ask the LLM to judge whether the system's answer is correct given the
/// reference answer. Returns a score between 0.0 and 1.0.
///
/// For adversarial questions the judge also checks that the system did NOT
/// produce the adversarial (wrong) answer.
async fn llm_judge(
    karta: &Karta,
    question: &str,
    system_answer: &str,
    reference_answer: &str,
    adversarial_answer: Option<&str>,
) -> f64 {
    use karta_core::llm::{ChatMessage, GenConfig, Role};

    let adversarial_clause = match adversarial_answer {
        Some(adv) => format!(
            "\n\nIMPORTANT: This is an adversarial question. The WRONG answer \
             that a naive system might give is: \"{}\". If the LLM response \
             matches or is close to this wrong answer, score it 0. Only score 1 \
             if the response correctly avoids this trap and matches the reference.",
            adv
        ),
        None => String::new(),
    };

    let messages = vec![
        ChatMessage {
            role: Role::System,
            content: "You are a strict evaluation judge for a conversational memory system. \
                Score whether the system's answer is correct given a reference answer. \
                Respond with JSON only: {\"score\": 1} if the answer is substantively \
                correct (captures the key facts from the reference), {\"score\": 0} if not. \
                Be strict — vague or partially correct answers score 0."
                .to_string(),
        },
        ChatMessage {
            role: Role::User,
            content: format!(
                "Question: {}\n\n\
                 Reference Answer: {}\n\n\
                 System Response: {}{}\n\n\
                 Is the system response correct? Respond with JSON: {{\"score\": 0 or 1}}",
                question,
                reference_answer,
                safe_truncate(system_answer, 3000),
                adversarial_clause,
            ),
        },
    ];

    let config = GenConfig {
        max_tokens: 64,
        temperature: 0.0,
        json_mode: true,
        json_schema: None,
    };

    match karta.llm_chat(&messages, &config).await {
        Ok(response) => {
            let parsed: serde_json::Value =
                serde_json::from_str(&response.content).unwrap_or_default();
            parsed["score"].as_f64().unwrap_or(0.0)
        }
        Err(e) => {
            eprintln!("    Judge error: {}", e);
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Per-conversation evaluation
// ---------------------------------------------------------------------------

/// Scores for a single conversation.
struct ConvResult {
    questions_asked: usize,
    total_passed: usize,
    total_judged: usize,
    category_scores: HashMap<String, (usize, usize)>, // (passed, total)
    ingest_ms: u128,
    dream_ms: u128,
    query_ms: u128,
}

async fn eval_conversation(conv: &LocomoConversation) -> ConvResult {
    println!("\n{}", "=".repeat(70));
    println!(
        "LOCOMO — Conv {} ({} & {}): {} sessions, {} msgs, {} questions",
        conv.id,
        conv.speaker_a,
        conv.speaker_b,
        conv.num_sessions,
        conv.messages.len(),
        conv.questions.len(),
    );
    println!("{}", "=".repeat(70));

    let karta = create_karta(&conv.id).await;

    // --- Ingest phase ---
    // Each dialogue turn becomes a note, prefixed with session timestamp and
    // speaker name for temporal and entity context.
    let ingest_start = Instant::now();
    let mut ingested = 0;
    let mut ingest_errors = 0;

    for (i, msg) in conv.messages.iter().enumerate() {
        if msg.text.trim().is_empty() {
            continue;
        }

        let content = if msg.timestamp.is_empty() {
            format!("{}: {}", msg.speaker, msg.text)
        } else {
            format!("[{}] {}: {}", msg.timestamp, msg.speaker, msg.text)
        };

        match karta.add_note(&content).await {
            Ok(note) => {
                ingested += 1;
                if (i + 1) % 25 == 0 || i == 0 {
                    println!(
                        "  Ingested {}/{} notes ({} links)",
                        i + 1,
                        conv.messages.len(),
                        note.links.len()
                    );
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

    // --- Dream phase ---
    let dream_start = Instant::now();
    match karta.run_dreaming("locomo", &conv.id).await {
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
    let dream_ms = dream_start.elapsed().as_millis();

    // --- Query phase ---
    let query_start = Instant::now();
    let mut category_scores: HashMap<String, (usize, usize)> = HashMap::new();
    let mut total_passed = 0;
    let mut total_judged = 0;

    for (qi, q) in conv.questions.iter().enumerate() {
        let q_start = Instant::now();
        let answer = match karta.ask(&q.question, 5).await {
            Ok(result) => result.answer,
            Err(e) => {
                eprintln!("  Query {} failed: {}", qi + 1, e);
                continue;
            }
        };
        let q_ms = q_start.elapsed().as_millis();

        println!(
            "\n  [{}] Q{}: {}",
            q.category,
            qi + 1,
            safe_truncate(&q.question, 80)
        );
        println!("  Ref: {}", safe_truncate(&q.answer, 120));
        println!(
            "  Sys ({:.1}s): {}",
            q_ms as f64 / 1000.0,
            safe_truncate(&answer, 200)
        );

        // LLM-as-judge
        let adv = q.adversarial_answer.as_deref();
        let score = llm_judge(&karta, &q.question, &answer, &q.answer, adv).await;

        total_judged += 1;
        let entry = category_scores.entry(q.category.clone()).or_insert((0, 0));
        entry.1 += 1;

        if score >= 0.5 {
            total_passed += 1;
            entry.0 += 1;
            println!("    [PASS] score={:.0}", score);
        } else {
            println!("    [FAIL] score={:.0}", score);
        }
    }

    let query_ms = query_start.elapsed().as_millis();

    ConvResult {
        questions_asked: conv.questions.len(),
        total_passed,
        total_judged,
        category_scores,
        ingest_ms,
        dream_ms,
        query_ms,
    }
}

fn print_category_table(scores: &HashMap<String, (usize, usize)>) {
    let category_order = [
        "single_hop",
        "multi_hop",
        "temporal",
        "open_domain",
        "adversarial",
    ];
    for cat in &category_order {
        if let Some((p, t)) = scores.get(*cat) {
            let rate = if *t > 0 {
                *p as f64 / *t as f64 * 100.0
            } else {
                0.0
            };
            println!("    {:20} {}/{} ({:.0}%)", cat, p, t, rate);
        }
    }
    // Print any unknown categories
    let mut extras: Vec<_> = scores
        .iter()
        .filter(|(k, _)| !category_order.contains(&k.as_str()))
        .collect();
    extras.sort_by_key(|(k, _)| (*k).clone());
    for (name, (p, t)) in extras {
        let rate = if *t > 0 {
            *p as f64 / *t as f64 * 100.0
        } else {
            0.0
        };
        println!("    {:20} {}/{} ({:.0}%)", name, p, t, rate);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Run LOCOMO on a single conversation (the first one) for quick iteration.
///
/// Run: cargo test --test locomo locomo_single -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn locomo_single() {
    let dataset_path =
        std::env::var("LOCOMO_DATASET_PATH").unwrap_or_else(|_| "data/locomo.json".to_string());

    if !Path::new(&dataset_path).exists() {
        eprintln!("LOCOMO dataset not found at {}.", dataset_path);
        eprintln!("Download and convert:");
        eprintln!(
            "  curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o data/locomo10_raw.json"
        );
        eprintln!("  python3 data/convert_locomo.py data/locomo10_raw.json data/locomo.json");
        return;
    }

    let dataset = load_dataset(&dataset_path);
    let conv = &dataset.conversations[0];

    let result = eval_conversation(conv).await;

    println!("\n{}", "=".repeat(70));
    println!("LOCOMO SINGLE CONVERSATION RESULT");
    println!("{}", "=".repeat(70));
    println!("  Questions: {}", result.questions_asked);
    println!(
        "  Passed:    {}/{}",
        result.total_passed, result.total_judged
    );

    let pass_rate = if result.total_judged > 0 {
        result.total_passed as f64 / result.total_judged as f64
    } else {
        0.0
    };
    println!("  Pass rate: {:.1}%", pass_rate * 100.0);
    println!(
        "  Time:      ingest {:.0}s, dream {:.0}s, query {:.0}s",
        result.ingest_ms as f64 / 1000.0,
        result.dream_ms as f64 / 1000.0,
        result.query_ms as f64 / 1000.0,
    );

    println!("\n  Per-category:");
    print_category_table(&result.category_scores);
}

/// Run LOCOMO on ALL conversations in the dataset.
///
/// WARNING: This may take a long time with a real LLM depending on dataset size.
///
/// Run: cargo test --test locomo locomo_full -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn locomo_full() {
    let dataset_path =
        std::env::var("LOCOMO_DATASET_PATH").unwrap_or_else(|_| "data/locomo.json".to_string());

    if !Path::new(&dataset_path).exists() {
        eprintln!("LOCOMO dataset not found at {}.", dataset_path);
        eprintln!("Download and convert:");
        eprintln!(
            "  curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o data/locomo10_raw.json"
        );
        eprintln!("  python3 data/convert_locomo.py data/locomo10_raw.json data/locomo.json");
        return;
    }

    let dataset = load_dataset(&dataset_path);
    println!(
        "LOCOMO Full Benchmark: {} conversations, {} questions, {} messages",
        dataset.num_conversations, dataset.total_questions, dataset.total_messages
    );

    let full_start = Instant::now();
    let mut grand_questions = 0;
    let mut grand_passed = 0;
    let mut grand_judged = 0;
    let mut grand_categories: HashMap<String, (usize, usize)> = HashMap::new();
    let mut grand_ingest_ms = 0u128;
    let mut grand_dream_ms = 0u128;
    let mut grand_query_ms = 0u128;

    for conv in &dataset.conversations {
        let result = eval_conversation(conv).await;
        grand_questions += result.questions_asked;
        grand_passed += result.total_passed;
        grand_judged += result.total_judged;
        grand_ingest_ms += result.ingest_ms;
        grand_dream_ms += result.dream_ms;
        grand_query_ms += result.query_ms;

        for (cat, (p, t)) in result.category_scores {
            let entry = grand_categories.entry(cat).or_insert((0, 0));
            entry.0 += p;
            entry.1 += t;
        }
    }

    let total_secs = full_start.elapsed().as_secs();

    println!("\n{}", "=".repeat(70));
    println!("LOCOMO FULL RESULTS");
    println!("{}", "=".repeat(70));
    println!("  Conversations: {}", dataset.num_conversations);
    println!("  Questions:     {}", grand_questions);
    println!("  Passed:        {}/{}", grand_passed, grand_judged);

    let pass_rate = if grand_judged > 0 {
        grand_passed as f64 / grand_judged as f64
    } else {
        0.0
    };
    println!("  Pass rate:     {:.1}%", pass_rate * 100.0);
    println!(
        "  Total time:    {}m {}s (ingest {:.0}s, dream {:.0}s, query {:.0}s)",
        total_secs / 60,
        total_secs % 60,
        grand_ingest_ms as f64 / 1000.0,
        grand_dream_ms as f64 / 1000.0,
        grand_query_ms as f64 / 1000.0,
    );

    println!("\n  Per-category:");
    print_category_table(&grand_categories);
}
