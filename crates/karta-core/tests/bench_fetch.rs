//! Latency benchmark for `Karta::fetch_memories()` — retrieval-only overhead.
//!
//! Measures the full retrieval pipeline (embed query -> ANN search -> rerank ->
//! dedup -> context assembly) WITHOUT any LLM synthesis calls. Answers the
//! question: "how much latency does Karta add to an LLM message?"
//!
//! Requires existing BEAM 100K data dirs at `/tmp/karta-beam100k-{1..20}-*`
//! (created by a prior `beam_100k` ingest run) and valid `.env` credentials
//! for the embedding / reranker providers.
//!
//! Run:
//!   cargo test --release -p karta-core --test bench_fetch -- --ignored --nocapture

use std::time::Instant;

use karta_core::config::KartaConfig;
use karta_core::Karta;

// ---------------------------------------------------------------------------
// BEAM dataset types (mirrors beam_100k.rs)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct BeamDataset {
    split: String,
    num_conversations: usize,
    total_questions: usize,
    conversations: Vec<BeamConversation>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct BeamConversation {
    id: String,
    category: String,
    title: String,
    user_messages: Vec<BeamMessage>,
    total_turns: usize,
    questions: Vec<BeamQuestion>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct BeamMessage {
    role: String,
    content: String,
    time_anchor: String,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct BeamQuestion {
    ability: String,
    question: String,
    reference_answer: String,
    rubric: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Config helpers (same as beam_100k.rs)
// ---------------------------------------------------------------------------

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key).ok().map(|s| s == "1" || s == "true").unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn apply_config_env(config: &mut KartaConfig) {
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
    config.reranker.max_rerank = env_usize("K_MAX_RERANK", 20);
    config.write.foresight_default_ttl_days = env_usize("K_FORESIGHT_TTL", 90) as i64;
    config.write.extract_atomic_facts = env_bool("K_EXTRACT_FACTS", true);
    config.read.fact_retrieval_enabled = env_bool("K_FACT_RETRIEVAL", true);
    config.read.fact_match_boost = env_f32("K_FACT_BOOST", 0.1);
}

// ---------------------------------------------------------------------------
// Data dir discovery (same pattern as beam_100k.rs)
// ---------------------------------------------------------------------------

fn find_latest_data_dir(conv_id: &str) -> Option<String> {
    if let Ok(suffix) = std::env::var("BEAM_DATA_SUFFIX") {
        let path = format!("/tmp/karta-beam100k-{}-{}", conv_id, suffix);
        if std::path::Path::new(&path).is_dir() {
            return Some(path);
        }
    }

    let mut dirs: Vec<_> = std::fs::read_dir("/tmp")
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("karta-beam100k-{}-", conv_id))
                && e.path().is_dir()
        })
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            let created = meta.created().ok().or_else(|| meta.modified().ok())?;
            let lance_path = e.path().join("lance/notes.lance/data");
            if lance_path.exists() {
                Some((e.path().to_string_lossy().to_string(), created))
            } else {
                None
            }
        })
        .collect();
    dirs.sort_by(|a, b| b.1.cmp(&a.1)); // most recently created first
    dirs.into_iter().next().map(|(path, _)| path)
}

// ---------------------------------------------------------------------------
// Per-call measurement record
// ---------------------------------------------------------------------------

struct FetchSample {
    conv_num: usize,
    stored_note_count: usize,
    latency_ms: f64,
    notes_returned: usize,
    context_length_chars: usize,
    context_length_tokens_approx: usize,
}

// ---------------------------------------------------------------------------
// Percentile helpers
// ---------------------------------------------------------------------------

const CHARS_PER_TOKEN_APPROX: usize = 4;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn percentile_usize(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (p / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn bench_fetch_memories_latency() {
    let _ = dotenvy::dotenv();

    // Load BEAM dataset
    let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/beam-100k.json");

    let raw = std::fs::read_to_string(&dataset_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", dataset_path.display(), e));
    let dataset: BeamDataset = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("Failed to parse BEAM dataset: {}", e));

    println!("\n=== fetch_memories() latency benchmark ===");
    println!(
        "Dataset: {} conversations, {} total questions\n",
        dataset.num_conversations, dataset.total_questions
    );

    let top_k: usize = env_usize("BENCH_TOP_K", 5);
    println!("top_k = {}\n", top_k);

    let mut all_samples: Vec<FetchSample> = Vec::new();
    let mut convs_found = 0usize;

    for (i, conv) in dataset.conversations.iter().enumerate() {
        let conv_num = i + 1;
        let data_dir = match find_latest_data_dir(&conv_num.to_string()) {
            Some(d) => d,
            None => {
                println!("[conv {:>2}] no data dir found, skipping", conv_num);
                continue;
            }
        };

        println!(
            "[conv {:>2}] {} — {} questions, dir: {}",
            conv_num,
            conv.title,
            conv.questions.len(),
            data_dir
        );

        let mut config = KartaConfig::default();
        config.storage.data_dir = data_dir;
        apply_config_env(&mut config);

        let karta = match Karta::with_defaults(config).await {
            Ok(k) => k,
            Err(e) => {
                println!("[conv {:>2}] failed to open Karta: {}, skipping", conv_num, e);
                continue;
            }
        };

        let stored_note_count = karta.note_count().await.unwrap_or(0);
        convs_found += 1;

        for q in &conv.questions {
            let start = Instant::now();
            let result = match karta.fetch_memories(&q.question, top_k).await {
                Ok(r) => r,
                Err(e) => {
                    println!(
                        "  [WARN] fetch_memories failed for q='{}': {}",
                        &q.question[..q.question.len().min(60)],
                        e
                    );
                    continue;
                }
            };
            let elapsed = start.elapsed();

            let context_chars = result.context.len();
            let tokens_approx = context_chars / CHARS_PER_TOKEN_APPROX;

            all_samples.push(FetchSample {
                conv_num,
                stored_note_count,
                latency_ms: elapsed.as_secs_f64() * 1000.0,
                notes_returned: result.notes.len(),
                context_length_chars: context_chars,
                context_length_tokens_approx: tokens_approx,
            });
        }

        // Print per-conv summary
        let conv_samples: Vec<&FetchSample> = all_samples
            .iter()
            .filter(|s| s.conv_num == conv_num)
            .collect();
        if !conv_samples.is_empty() {
            let mut lats: Vec<f64> = conv_samples.iter().map(|s| s.latency_ms).collect();
            lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  notes={}, queries={}, p50={:.0}ms p95={:.0}ms max={:.0}ms",
                stored_note_count,
                conv_samples.len(),
                percentile(&lats, 50.0),
                percentile(&lats, 95.0),
                lats.last().unwrap_or(&0.0),
            );
        }
    }

    if all_samples.is_empty() {
        println!("\nNo data dirs found. Nothing to benchmark.");
        println!("Run the BEAM 100K ingest first, then re-run with BEAM_SKIP_INGEST=true.");
        return;
    }

    // -----------------------------------------------------------------------
    // Summary: latency by stored-note-count bucket
    // -----------------------------------------------------------------------

    println!("\n{}", "=".repeat(80));
    println!(
        "SUMMARY: {} calls across {} conversations\n",
        all_samples.len(),
        convs_found
    );

    let buckets: &[(&str, Box<dyn Fn(usize) -> bool>)] = &[
        ("<200 notes", Box::new(|n| n < 200)),
        ("200-400 notes", Box::new(|n| (200..=400).contains(&n))),
        ("400+ notes", Box::new(|n| n > 400)),
    ];

    println!(
        "{:<16} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Bucket", "N", "p50 ms", "p95 ms", "p99 ms", "max ms"
    );
    println!("{}", "-".repeat(64));

    for (label, pred) in buckets {
        let mut lats: Vec<f64> = all_samples
            .iter()
            .filter(|s| pred(s.stored_note_count))
            .map(|s| s.latency_ms)
            .collect();
        if lats.is_empty() {
            println!("{:<16} {:>6}", label, 0);
            continue;
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:<16} {:>6} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
            label,
            lats.len(),
            percentile(&lats, 50.0),
            percentile(&lats, 95.0),
            percentile(&lats, 99.0),
            lats.last().unwrap(),
        );
    }

    // Overall latency
    let mut all_lats: Vec<f64> = all_samples.iter().map(|s| s.latency_ms).collect();
    all_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{}", "-".repeat(64));
    println!(
        "{:<16} {:>6} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
        "ALL",
        all_lats.len(),
        percentile(&all_lats, 50.0),
        percentile(&all_lats, 95.0),
        percentile(&all_lats, 99.0),
        all_lats.last().unwrap(),
    );

    // -----------------------------------------------------------------------
    // Context / token stats
    // -----------------------------------------------------------------------

    println!("\n{}", "=".repeat(80));
    println!("CONTEXT SIZE (fetch_memories output)\n");

    let mut ctx_chars: Vec<usize> = all_samples.iter().map(|s| s.context_length_chars).collect();
    ctx_chars.sort();
    let mut ctx_tokens: Vec<usize> = all_samples
        .iter()
        .map(|s| s.context_length_tokens_approx)
        .collect();
    ctx_tokens.sort();

    let avg_chars: f64 = ctx_chars.iter().sum::<usize>() as f64 / ctx_chars.len() as f64;
    let avg_tokens: f64 = ctx_tokens.iter().sum::<usize>() as f64 / ctx_tokens.len() as f64;

    let mut notes_returned: Vec<usize> = all_samples.iter().map(|s| s.notes_returned).collect();
    notes_returned.sort();
    let avg_notes: f64 =
        notes_returned.iter().sum::<usize>() as f64 / notes_returned.len() as f64;

    println!(
        "{:<24} {:>8} {:>8} {:>8}",
        "", "avg", "p50", "p95"
    );
    println!("{}", "-".repeat(56));
    println!(
        "{:<24} {:>8.0} {:>8} {:>8}",
        "context_length_chars",
        avg_chars,
        percentile_usize(&ctx_chars, 50.0),
        percentile_usize(&ctx_chars, 95.0),
    );
    println!(
        "{:<24} {:>8.0} {:>8} {:>8}",
        "context_length_tokens",
        avg_tokens,
        percentile_usize(&ctx_tokens, 50.0),
        percentile_usize(&ctx_tokens, 95.0),
    );
    println!(
        "{:<24} {:>8.1} {:>8} {:>8}",
        "notes_returned",
        avg_notes,
        percentile_usize(&notes_returned, 50.0),
        percentile_usize(&notes_returned, 95.0),
    );

    println!("\n{}", "=".repeat(80));
    println!("Done. {} total fetch_memories() calls measured.", all_samples.len());
}
