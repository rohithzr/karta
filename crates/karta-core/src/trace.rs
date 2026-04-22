//! Per-stage ingestion tracing.
//!
//! When a `TraceWriter` is set as the active task-local, every instrumented
//! stage in the write/dream paths emits structured JSONL events: stage
//! start/end, LLM chat/embed calls (with token counts and wall time), kNN
//! candidates, note/link/fact writes, and per-turn graph snapshots.
//!
//! Stages are scoped via `trace::stage(name, fut).await`. They nest correctly
//! across `tokio::join!` because each scope is bound to *polling* its inner
//! future, not to ambient state.

use std::fs::File;
use std::future::Future;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::task_local;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceEvent {
    TurnStart {
        ts: DateTime<Utc>,
        turn_idx: u32,
        content_len: usize,
        content: Option<String>,
    },
    TurnEnd {
        ts: DateTime<Utc>,
        turn_idx: u32,
        wall_ms: u64,
        notes_total: usize,
        edges_total: usize,
        facts_total: usize,
    },
    StageStart {
        ts: DateTime<Utc>,
        turn_idx: u32,
        stage: String,
    },
    StageEnd {
        ts: DateTime<Utc>,
        turn_idx: u32,
        stage: String,
        wall_ms: u64,
    },
    LlmChat {
        ts: DateTime<Utc>,
        turn_idx: u32,
        stage: String,
        model: String,
        wall_ms: u64,
        input_tokens: u64,
        output_tokens: u64,
        prompt: Option<String>,
        completion: Option<String>,
    },
    LlmEmbed {
        ts: DateTime<Utc>,
        turn_idx: u32,
        stage: String,
        model: String,
        wall_ms: u64,
        input_count: usize,
        total_chars: usize,
        inputs: Option<Vec<String>>,
    },
    KnnCandidates {
        ts: DateTime<Utc>,
        turn_idx: u32,
        stage: String,
        wall_ms: u64,
        returned: usize,
        candidates: Option<Vec<KnnCandidate>>,
    },
    NoteWritten {
        ts: DateTime<Utc>,
        turn_idx: u32,
        note_id: String,
        link_count: usize,
    },
    LinkWritten {
        ts: DateTime<Utc>,
        turn_idx: u32,
        from_id: String,
        to_id: String,
        reason: String,
    },
    Evolution {
        ts: DateTime<Utc>,
        turn_idx: u32,
        evolved_id: String,
        previous_context: Option<String>,
        new_context: Option<String>,
    },
    FactWritten {
        ts: DateTime<Utc>,
        turn_idx: u32,
        fact_id: String,
        source_note_id: String,
        ordinal: u32,
        content: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnCandidate {
    pub note_id: String,
    pub score: f32,
    pub content_preview: Option<String>,
}

pub struct TraceWriter {
    file: Mutex<BufWriter<File>>,
    heavy: bool,
}

impl TraceWriter {
    pub fn new(path: &Path, heavy: bool) -> std::io::Result<Self> {
        let f = File::create(path)?;
        Ok(Self {
            file: Mutex::new(BufWriter::new(f)),
            heavy,
        })
    }

    pub fn emit(&self, event: TraceEvent) {
        let json = serde_json::to_string(&event).expect("serialize trace event");
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{}", json);
        }
    }

    pub fn heavy(&self) -> bool {
        self.heavy
    }

    pub fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
    }
}

task_local! {
    pub static TRACE_WRITER: Arc<TraceWriter>;
    pub static TURN_IDX: u32;
    pub static STAGE: String;
}

/// Run `fut` with trace context bound. All instrumentation inside `fut`
/// (including transitive calls through stage spans and the LLM wrapper)
/// will emit to `writer`. No-op if `writer` is `None`.
pub async fn with_trace<F, T>(writer: Option<Arc<TraceWriter>>, turn_idx: u32, fut: F) -> T
where
    F: Future<Output = T>,
{
    match writer {
        Some(w) => {
            TRACE_WRITER
                .scope(
                    w,
                    TURN_IDX.scope(turn_idx, STAGE.scope("root".to_string(), fut)),
                )
                .await
        }
        None => fut.await,
    }
}

/// Wrap an async block in a stage span. Emits StageStart on entry, StageEnd
/// on completion with wall time. Sets STAGE for the polling duration so
/// nested LLM calls inherit the label. Composes safely with `tokio::join!`.
pub async fn stage<F, T>(name: &'static str, fut: F) -> T
where
    F: Future<Output = T>,
{
    let writer = TRACE_WRITER.try_with(|w| w.clone()).ok();
    let turn = current_turn();
    let start = Instant::now();
    if let Some(ref w) = writer {
        w.emit(TraceEvent::StageStart {
            ts: Utc::now(),
            turn_idx: turn,
            stage: name.to_string(),
        });
    }
    let result = STAGE.scope(name.to_string(), fut).await;
    if let Some(w) = writer {
        w.emit(TraceEvent::StageEnd {
            ts: Utc::now(),
            turn_idx: turn,
            stage: name.to_string(),
            wall_ms: start.elapsed().as_millis() as u64,
        });
    }
    result
}

pub fn try_emit(event: TraceEvent) {
    let _ = TRACE_WRITER.try_with(|w| w.emit(event));
}

pub fn current_turn() -> u32 {
    TURN_IDX.try_with(|t| *t).unwrap_or(u32::MAX)
}

pub fn current_stage() -> String {
    STAGE
        .try_with(|s| s.clone())
        .unwrap_or_else(|_| "no_ctx".to_string())
}

pub fn heavy() -> bool {
    TRACE_WRITER.try_with(|w| w.heavy()).unwrap_or(false)
}

pub fn flush() {
    let _ = TRACE_WRITER.try_with(|w| w.flush());
}
