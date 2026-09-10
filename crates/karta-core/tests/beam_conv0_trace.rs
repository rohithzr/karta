//! Per-stage ingestion trace for BEAM conv 0, first N turns.
//!
//! Goal: characterise where wall-time and tokens go on a single conversation
//! so we can reason about where to optimise (E1 / E2 in the experiment slate).
//!
//! Run:
//!   N=10 cargo test --release -p karta-core --test beam_conv0_trace \
//!     trace_conv0 -- --ignored --nocapture
//!
//! Env:
//!   BEAM_TRACE_TURNS         number of user messages to ingest (default 10)
//!   BEAM_TRACE_HEAVY         "1" to include prompts/completions/inputs in trace
//!                            (default "1" since the harness target is heavy mode)
//!   BEAM_TRACE_OUT           override output path; default
//!                            .results/conv0-trace-<ts>.jsonl
//!   AZURE_OPENAI_*           credentials (must be set; harness requires real LLM)

#![cfg(all(feature = "sqlite-vec", feature = "openai"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use karta_core::config::KartaConfig;
use karta_core::llm::{LlmProvider, OpenAiProvider, TracingLlmProvider};
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_core::trace::{self, TraceEvent, TraceWriter};
use karta_core::Karta;

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
    sessions: Vec<BeamSession>,
    total_turns: usize,
    total_user_turns: usize,
    questions: Vec<serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct BeamSession {
    session_index: usize,
    session_anchor: Option<String>,
    turns: Vec<BeamTurn>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct BeamTurn {
    turn_index: u32,
    role: String,
    content: String,
    time_anchor: Option<String>,
    effective_reference_time: Option<chrono::DateTime<chrono::Utc>>,
    question_type: Option<String>,
    raw_index: Option<String>,
}

fn resolve_dataset_path() -> String {
    if let Ok(explicit) = std::env::var("BEAM_DATASET_PATH") {
        return explicit;
    }
    let cwd_relative = "data/beam-100k.json";
    if Path::new(cwd_relative).exists() {
        return cwd_relative.to_string();
    }
    let workspace_relative = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/beam-100k.json");
    workspace_relative.to_string_lossy().into_owned()
}

/// Build an Azure-Azure provider (chat + embed both via Azure OpenAI).
/// Panics with a clear message if credentials aren't present, so the
/// trace harness fails fast rather than silently using a different path.
fn build_azure_provider() -> Arc<dyn LlmProvider> {
    let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
        .expect("AZURE_OPENAI_ENDPOINT not set — the trace harness requires Azure creds");
    let api_key = std::env::var("AZURE_OPENAI_API_KEY")
        .expect("AZURE_OPENAI_API_KEY not set");
    let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
        .unwrap_or_else(|_| "2025-04-01-preview".to_string());
    let chat_deployment = std::env::var("AZURE_OPENAI_CHAT_MODEL")
        .expect("AZURE_OPENAI_CHAT_MODEL not set");
    let embed_deployment = std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
        .unwrap_or_else(|_| "text-embedding-3-small".to_string());

    Arc::new(OpenAiProvider::azure(
        &endpoint,
        &api_key,
        &api_version,
        &chat_deployment,
        &embed_deployment,
    ))
}

async fn snapshot_counts(
    vector_store: &Arc<dyn VectorStore>,
    graph_store: &Arc<dyn GraphStore>,
) -> (usize, usize, usize) {
    let notes = vector_store.count().await.unwrap_or(0);
    // edges_total: sum of get_links per note. For 10 turns this is cheap.
    let mut edges = 0usize;
    if let Ok(all) = vector_store.get_all().await {
        for n in &all {
            if let Ok(links) = graph_store.get_links(&n.id).await {
                edges += links.len();
            }
        }
        // Bidirectional storage means each link is double-counted; not all
        // backends behave the same, but we report the raw graph_store count.
    }
    // facts: not all stores expose count_facts; approximate via get_facts_for_note
    let mut facts = 0usize;
    if let Ok(all) = vector_store.get_all().await {
        for n in &all {
            if let Ok(fs) = vector_store.get_facts_for_note(&n.id).await {
                facts += fs.len();
            }
        }
    }
    (notes, edges, facts)
}

#[tokio::test]
#[ignore]
async fn trace_conv0() {
    let _ = dotenvy::dotenv();

    let turns: usize = std::env::var("BEAM_TRACE_TURNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let heavy = std::env::var("BEAM_TRACE_HEAVY")
        .ok()
        .map(|s| s != "0" && !s.eq_ignore_ascii_case("false"))
        .unwrap_or(true);

    let dataset_path = resolve_dataset_path();
    let raw = std::fs::read_to_string(&dataset_path)
        .unwrap_or_else(|e| panic!("Cannot read BEAM dataset at {}: {}", dataset_path, e));
    let dataset: BeamDataset =
        serde_json::from_str(&raw).expect("Invalid JSON in BEAM dataset");

    let conv = dataset
        .conversations
        .first()
        .expect("BEAM dataset has no conversations");

    println!(
        "Tracing conv {} [{}]: {} sessions / {} total turns ({} user), ingesting first {} user turns",
        conv.id, conv.category, conv.sessions.len(), conv.total_turns, conv.total_user_turns, turns
    );

    // --- Output paths ---
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let results_dir = repo_root.join(".results");
    std::fs::create_dir_all(&results_dir).expect("create .results dir");

    let ts = Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let trace_path = std::env::var("BEAM_TRACE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            results_dir.join(format!("conv{}-trace-{}.jsonl", conv.id, ts))
        });

    println!("Trace output: {}", trace_path.display());
    println!("Heavy mode: {}", heavy);

    let writer = Arc::new(
        TraceWriter::new(&trace_path, heavy).expect("create trace writer"),
    );

    // --- Build Karta with traced LLM ---
    let suffix = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let data_dir = format!("/tmp/karta-trace-conv{}-{}", conv.id, suffix);
    let _ = std::fs::remove_dir_all(&data_dir);
    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;

    let raw_llm = build_azure_provider();
    let traced_llm: Arc<dyn LlmProvider> =
        Arc::new(TracingLlmProvider::new(Arc::clone(&raw_llm)));

    // Probe embedding dim once via the underlying provider so trace events
    // for that probe do not pollute the per-turn output.
    let probe = raw_llm
        .embed(&["karta-init-probe"])
        .await
        .expect("embed probe failed (Azure creds wrong?)");
    let embedding_dim = probe[0].len();
    println!("Embedding dim: {}", embedding_dim);

    let vec_store = SqliteVectorStore::new(&config.storage.data_dir, embedding_dim)
        .await
        .expect("open sqlite-vec store");
    let shared_conn = vec_store.connection();
    let vector_store: Arc<dyn VectorStore> = Arc::new(vec_store);
    let graph_store: Arc<dyn GraphStore> =
        Arc::new(SqliteGraphStore::with_connection(shared_conn));

    let karta = Karta::new(
        Arc::clone(&vector_store),
        Arc::clone(&graph_store),
        traced_llm,
        config,
    )
    .await
    .expect("Karta::new");

    // --- Flatten sessions to user turns; take first N. ---
    // The trace harness's intent is to characterise per-stage ingest cost
    // on user turns specifically. Assistant turns are present in the JSON
    // post-F8 but ingestion semantics for them are still TBD (Q9).
    let session_prefix = format!("trace-conv{}", conv.id);
    // HARNESS-LEVEL SKIP (F7-T9b): user echoes of AI suggestions carry no
    // originating claims. Skip before feeding the ingest loop so no attrs
    // LLM call fires for these turns.
    let user_turns: Vec<(usize, &BeamSession, &BeamTurn)> = conv
        .sessions
        .iter()
        .flat_map(|s| s.turns.iter().map(move |t| (s, t)))
        .filter(|(_, t)| {
            t.role == "user"
                && !t.content.trim().is_empty()
                && t.question_type.as_deref() != Some("answer_ai_question")
        })
        .take(turns)
        .enumerate()
        .map(|(i, (s, t))| (i, s, t))
        .collect();

    let overall_start = Instant::now();

    for (i, session, turn) in &user_turns {
        let session_id = format!("{}-s{}", session_prefix, session.session_index);
        let ctx = match turn.effective_reference_time {
            Some(t) => karta_core::clock::ClockContext::at(t),
            None => karta_core::clock::ClockContext::now(),
        };
        // No more harness-side `[time_anchor]` prefix — anchor lives in ctx.
        let content = turn.content.clone();

        let writer_clone = Arc::clone(&writer);
        let karta_ref = &karta;
        let content_for_event = if heavy { Some(content.clone()) } else { None };
        let turn_idx = *i as u32;

        let turn_start = Instant::now();
        writer_clone.emit(TraceEvent::TurnStart {
            ts: Utc::now(),
            turn_idx,
            content_len: content.len(),
            content: content_for_event,
        });

        let result = trace::with_trace(Some(Arc::clone(&writer)), turn_idx, async move {
            karta_ref
                .add_note_with_clock(&content, Some(&session_id), Some(turn_idx), ctx)
                .await
        })
        .await;

        let wall_ms = turn_start.elapsed().as_millis() as u64;
        let (notes_total, edges_total, facts_total) =
            snapshot_counts(&vector_store, &graph_store).await;

        match &result {
            Ok(note) => {
                println!(
                    "  turn {}/{}: {}ms, {} links, notes={} edges={} facts={}",
                    i + 1,
                    turns,
                    wall_ms,
                    note.links.len(),
                    notes_total,
                    edges_total,
                    facts_total
                );
            }
            Err(e) => {
                eprintln!("  turn {} FAILED: {}", i + 1, e);
            }
        }

        writer.emit(TraceEvent::TurnEnd {
            ts: Utc::now(),
            turn_idx,
            wall_ms,
            notes_total,
            edges_total,
            facts_total,
        });

        writer.flush();
    }

    let overall_ms = overall_start.elapsed().as_millis();
    println!(
        "\nDone. Ingested ~{} turns in {:.1}s. Trace at {}",
        turns,
        overall_ms as f64 / 1000.0,
        trace_path.display()
    );
}
