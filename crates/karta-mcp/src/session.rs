//! Session lifecycle and rule-based consolidation helpers.
//!
//! This module is shared by the MCP tool handlers (`tools.rs`) and the
//! capture-queue worker (`queue.rs`). Session markers are written through
//! `Karta::add_note_with_clock`, but consolidation creates promoted fact notes
//! directly through the vector store so that it stays deterministic and never
//! calls an LLM or the dream engine.

use std::path::Path;

use anyhow::Result;
use chrono::Utc;

use crate::karta_handle::KartaHandle;
use crate::transcript;

pub const SESSION_END_TAG: &str = "karta:session_end";
pub const PRECOMPACT_TAG: &str = "karta:pre_compact";
pub const FACT_TAG: &str = "karta:fact";

const CONFIDENCE_THRESHOLD: f32 = 0.9;

/// Start a new session, returning a unique session id and orientation context.
pub async fn session_start(
    agent: &str,
    project: Option<&str>,
    handle: &KartaHandle,
) -> Result<(String, String)> {
    let project = project.unwrap_or("default");
    let session_id = format!(
        "{}-{}-{}",
        sanitize(agent),
        sanitize(project),
        Utc::now().timestamp_millis()
    );
    let query = format!("agent: {} project: {}", agent, project);
    let result = handle.karta.fetch_memories(&query, 5).await?;
    Ok((session_id, result.context))
}

/// End a session by writing a marker/summary note and triggering consolidation.
pub async fn session_end(
    session_id: &str,
    summary: Option<&str>,
    handle: &KartaHandle,
) -> Result<String> {
    session_end_with_transcript(session_id, summary, None, handle).await
}

/// End a session, optionally augmenting the marker with a transcript path.
///
/// If `transcript_path` points to a file that cannot be read, the marker is
/// still written and the path is preserved in the marker content so that a
/// later transcript sweep can retry.
pub async fn session_end_with_transcript(
    session_id: &str,
    summary: Option<&str>,
    transcript_path: Option<&str>,
    handle: &KartaHandle,
) -> Result<String> {
    let summary_text = summary.unwrap_or("no summary provided");
    let mut content = format!("[{SESSION_END_TAG}] session: {session_id} summary: {summary_text}");

    if let Some(path) = transcript_path {
        if Path::new(path).exists() {
            match transcript::sweep_transcript(path, session_id, handle).await {
                Ok(count) => {
                    content.push_str(&format!(
                        " transcript_sweep: recovered {count} events from {path}"
                    ));
                }
                Err(e) => {
                    content.push_str(&format!(
                        " transcript_path: {path} transcript_sweep_error: {e}"
                    ));
                }
            }
        } else {
            content.push_str(&format!(" transcript_path: {path}"));
        }
    }

    let note = handle
        .karta
        .add_note_with_clock(
            &content,
            Some(session_id),
            None,
            karta_core::ClockContext::now(),
        )
        .await?;
    // Trigger rule-based consolidation (count only, no LLM, no dream).
    let _ = consolidate(Some(session_id), handle).await?;
    Ok(note.id)
}

/// Handle a `pre_compact` capture event.
///
/// When `enabled` is true, writes a marker note and optionally consolidates the
/// session. When disabled, returns `Ok(None)` and performs no side effects, so
/// the queue worker can still mark the row `done`.
pub async fn pre_compact(
    session_id: &str,
    handle: &KartaHandle,
    enabled: bool,
) -> Result<Option<String>> {
    if !enabled {
        return Ok(None);
    }

    let content = format!("[{PRECOMPACT_TAG}] session: {session_id}");
    let note = handle
        .karta
        .add_note_with_clock(
            &content,
            Some(session_id),
            None,
            karta_core::ClockContext::now(),
        )
        .await?;
    // Optional early consolidation. Idempotency prevents duplicate facts when
    // `session_end` runs later for the same session.
    let _ = consolidate(Some(session_id), handle).await?;
    Ok(Some(note.id))
}

/// Rule-based consolidation: no LLM, no dream. Returns the number of active
/// high-confidence observations promoted to fact notes.
///
/// Promotion is idempotent: a source observation is skipped if a fact note
/// already exists that points back to it via `Provenance::Fact`. Session markers
/// are never promoted.
pub async fn consolidate(session_id: Option<&str>, handle: &KartaHandle) -> Result<usize> {
    let notes = handle.karta.get_all_notes().await?;
    let mut promoted_count = 0;

    for note in &notes {
        if !is_promotable_candidate(note, session_id) {
            continue;
        }
        if is_already_promoted(&notes, &note.id) {
            continue;
        }

        let fact = build_fact_note(note);
        handle.vector_store.upsert(&fact).await?;
        promoted_count += 1;
    }

    Ok(promoted_count)
}

/// Decide whether a note should be considered for promotion.
fn is_promotable_candidate(note: &karta_core::note::MemoryNote, session_id: Option<&str>) -> bool {
    use karta_core::note::Provenance;

    if !note.is_active() {
        return false;
    }
    if note.confidence < CONFIDENCE_THRESHOLD {
        return false;
    }
    if !matches!(note.provenance, Provenance::Observed) {
        return false;
    }
    if is_marker_note(note) {
        return false;
    }
    if let Some(sid) = session_id
        && note.session_id.as_deref() != Some(sid)
    {
        return false;
    }
    true
}

/// True if the note content carries a session-boundary marker tag.
fn is_marker_note(note: &karta_core::note::MemoryNote) -> bool {
    note.content.contains(SESSION_END_TAG) || note.content.contains(PRECOMPACT_TAG)
}

/// True if a fact note already exists that references the given source note id.
fn is_already_promoted(notes: &[karta_core::note::MemoryNote], source_id: &str) -> bool {
    use karta_core::note::Provenance;
    notes.iter().any(|n| {
        matches!(
            &n.provenance,
            Provenance::Fact { source_note_id } if source_note_id == source_id
        )
    })
}

/// Build a promoted fact note from a source observation.
///
/// The source note's embedding is reused so that no new LLM embedding call is
/// required. Provenance is set to `Provenance::Fact` with the source note id as
/// a back-pointer.
fn build_fact_note(source: &karta_core::note::MemoryNote) -> karta_core::note::MemoryNote {
    use karta_core::note::{MemoryNote, Provenance};

    let content = format!(
        "[{FACT_TAG}] promoted from source {} (confidence {:.2}): {}",
        source.id, source.confidence, source.content
    );
    let mut fact = MemoryNote::new(content);
    fact.provenance = Provenance::Fact {
        source_note_id: source.id.clone(),
    };
    fact.confidence = source.confidence;
    fact.session_id = source.session_id.clone();
    fact.embedding = source.embedding.clone();
    fact.context = format!("Consolidated fact derived from note {}", source.id);
    fact.tags = vec!["consolidated".to_string(), "fact".to_string()];
    fact
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use karta_core::llm::{ChatMessage, ChatResponse, GenConfig, LlmProvider, MockLlmProvider};
    use karta_core::note::{MemoryNote, Provenance};
    use karta_core::store::VectorStore;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::{ClockContext, Karta};
    use tempfile::TempDir;

    use super::*;

    const EMBEDDING_DIM: usize = 1536;

    async fn setup() -> (TempDir, KartaHandle) {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let handle = KartaHandle::open_mock(data_dir).await.unwrap();
        (dir, handle)
    }

    async fn insert_observation(
        handle: &KartaHandle,
        content: &str,
        session_id: Option<&str>,
        confidence: f32,
    ) -> String {
        let mut note = MemoryNote::new(content.to_string());
        note.provenance = Provenance::Observed;
        note.confidence = confidence;
        note.session_id = session_id.map(String::from);
        // Provide a non-empty unit embedding so the sqlite-vec store accepts
        // the note. The mock tests use 1536-dim vectors.
        const DIM: usize = 1536;
        let value = 1.0 / (DIM as f32).sqrt();
        note.embedding = vec![value; DIM];
        handle.vector_store.upsert(&note).await.unwrap();
        note.id
    }

    // VAL-SES-001, VAL-SES-002, VAL-SES-003, VAL-SES-018, VAL-SES-026
    #[tokio::test]
    async fn session_start_generates_unique_id_and_orientation_context() {
        let (_dir, handle) = setup().await;

        // Seed a note relevant to the agent/project query.
        handle
            .karta
            .add_note_with_clock(
                "droid agent working on the karta project",
                None,
                None,
                ClockContext::now(),
            )
            .await
            .unwrap();

        let (id1, ctx1) = session_start("droid", Some("karta"), &handle)
            .await
            .unwrap();
        let (id2, ctx2) = session_start("droid", Some("karta"), &handle)
            .await
            .unwrap();

        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
        assert_ne!(id1, id2);
        assert!(
            ctx1.contains("droid") || ctx1.contains("karta"),
            "orientation context should include query terms: {ctx1}"
        );
        assert!(ctx2.contains("droid") || ctx2.contains("karta"));

        // The session id is reusable for the same logical session.
        let end_id = session_end(&id1, Some("wrap"), &handle).await.unwrap();
        let note = handle.karta.get_note(&end_id).await.unwrap().unwrap();
        assert_eq!(note.session_id, Some(id1));
    }

    // VAL-SES-004, VAL-SES-005, VAL-SES-006, VAL-SES-019, VAL-SES-021,
    // VAL-SES-022, VAL-SES-029
    #[tokio::test]
    async fn session_end_writes_tagged_marker_and_triggers_consolidate() {
        let (_dir, handle) = setup().await;
        let session_id = "session-marker".to_string();

        let source_id = insert_observation(
            &handle,
            "high confidence marker test",
            Some(&session_id),
            0.95,
        )
        .await;

        let count_before = handle.karta.note_count().await.unwrap();
        let written_id = session_end(&session_id, Some("wrapped up"), &handle)
            .await
            .unwrap();
        let count_after = handle.karta.note_count().await.unwrap();

        let note = handle.karta.get_note(&written_id).await.unwrap().unwrap();
        assert!(note.content.contains(SESSION_END_TAG));
        assert!(note.content.contains(&session_id));
        assert!(note.content.contains("wrapped up"));
        assert_eq!(note.session_id, Some(session_id.clone()));
        // Marker + consolidated fact.
        assert!(count_after > count_before);

        // The source note is referenced by the promoted fact.
        let facts: Vec<_> = handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .into_iter()
            .filter(|n| matches!(n.provenance, Provenance::Fact { source_note_id: ref sid } if sid == &source_id))
            .collect();
        assert_eq!(facts.len(), 1);
    }

    // VAL-SES-021
    #[tokio::test]
    async fn session_end_with_empty_summary_writes_marker() {
        let (_dir, handle) = setup().await;
        let session_id = "session-empty".to_string();

        let written_id = session_end(&session_id, None, &handle).await.unwrap();
        let note = handle.karta.get_note(&written_id).await.unwrap().unwrap();
        assert!(note.content.contains(SESSION_END_TAG));
        assert!(note.content.contains(&session_id));
        assert!(note.content.contains("no summary provided"));
    }

    // VAL-SES-008, VAL-SES-009, VAL-SES-020, VAL-SES-024
    #[tokio::test]
    async fn consolidate_promotes_high_confidence_observations() {
        let (_dir, handle) = setup().await;
        let session_id = "session-promote".to_string();

        let source_id = insert_observation(
            &handle,
            "high confidence observation",
            Some(&session_id),
            0.95,
        )
        .await;

        let promoted = consolidate(Some(&session_id), &handle).await.unwrap();
        assert_eq!(promoted, 1);

        let facts: Vec<_> = handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .into_iter()
            .filter(|n| matches!(n.provenance, Provenance::Fact { .. }))
            .collect();
        assert_eq!(facts.len(), 1);
        assert!(facts[0].content.contains(FACT_TAG));
        assert!(facts[0].content.contains(&source_id));
        assert!(facts[0].confidence >= CONFIDENCE_THRESHOLD);
        if let Provenance::Fact { source_note_id } = &facts[0].provenance {
            assert_eq!(source_note_id, &source_id);
        } else {
            panic!("expected Fact provenance");
        }
    }

    // VAL-SES-010
    #[tokio::test]
    async fn consolidate_skips_low_confidence_observations() {
        let (_dir, handle) = setup().await;
        let session_id = "session-low".to_string();

        insert_observation(
            &handle,
            "low confidence observation",
            Some(&session_id),
            0.5,
        )
        .await;

        let promoted = consolidate(Some(&session_id), &handle).await.unwrap();
        assert_eq!(promoted, 0);

        let facts: Vec<_> = handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .into_iter()
            .filter(|n| matches!(n.provenance, Provenance::Fact { .. }))
            .collect();
        assert!(facts.is_empty());
    }

    // VAL-SES-011
    #[tokio::test]
    async fn consolidate_is_idempotent_for_same_session() {
        let (_dir, handle) = setup().await;
        let session_id = "session-idem".to_string();

        insert_observation(&handle, "observation", Some(&session_id), 0.95).await;

        let first = consolidate(Some(&session_id), &handle).await.unwrap();
        assert_eq!(first, 1);
        let second = consolidate(Some(&session_id), &handle).await.unwrap();
        assert_eq!(second, 0);

        let facts: Vec<_> = handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .into_iter()
            .filter(|n| matches!(n.provenance, Provenance::Fact { .. }))
            .collect();
        assert_eq!(facts.len(), 1);
    }

    // VAL-SES-012
    #[tokio::test]
    async fn consolidate_without_session_id_operates_globally() {
        let (_dir, handle) = setup().await;

        insert_observation(&handle, "unscoped high confidence", None, 0.95).await;
        insert_observation(
            &handle,
            "scoped high confidence",
            Some("scoped-session"),
            0.95,
        )
        .await;

        let promoted = consolidate(None, &handle).await.unwrap();
        assert_eq!(promoted, 2);
    }

    // VAL-SES-014, VAL-SES-016
    #[tokio::test]
    async fn pre_compact_enabled_writes_marker_and_consolidates() {
        let (_dir, handle) = setup().await;
        let session_id = "session-pre".to_string();

        let source_id =
            insert_observation(&handle, "high confidence", Some(&session_id), 0.95).await;

        let note_id = pre_compact(&session_id, &handle, true)
            .await
            .unwrap()
            .unwrap();
        let note = handle.karta.get_note(&note_id).await.unwrap().unwrap();
        assert!(note.content.contains(PRECOMPACT_TAG));
        assert!(note.content.contains(&session_id));

        let facts: Vec<_> = handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .into_iter()
            .filter(|n| matches!(n.provenance, Provenance::Fact { source_note_id: ref sid } if sid == &source_id))
            .collect();
        assert_eq!(facts.len(), 1);
    }

    // VAL-SES-015, VAL-SES-027
    #[tokio::test]
    async fn pre_compact_disabled_does_nothing() {
        let (_dir, handle) = setup().await;
        let session_id = "session-no-pre".to_string();

        let count_before = handle.karta.note_count().await.unwrap();
        let result = pre_compact(&session_id, &handle, false).await.unwrap();
        assert!(result.is_none());
        let count_after = handle.karta.note_count().await.unwrap();
        assert_eq!(count_before, count_after);
    }

    // VAL-SES-017
    #[tokio::test]
    async fn pre_compact_and_session_end_do_not_double_consolidate() {
        let (_dir, handle) = setup().await;
        let session_id = "session-no-double".to_string();

        insert_observation(&handle, "high confidence", Some(&session_id), 0.95).await;

        pre_compact(&session_id, &handle, true).await.unwrap();
        let fact_count_after_pre = handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .into_iter()
            .filter(|n| matches!(n.provenance, Provenance::Fact { .. }))
            .count();

        session_end(&session_id, None, &handle).await.unwrap();
        let fact_count_after_end = handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .into_iter()
            .filter(|n| matches!(n.provenance, Provenance::Fact { .. }))
            .count();

        assert_eq!(fact_count_after_pre, fact_count_after_end);
    }

    // VAL-SES-023
    #[tokio::test]
    async fn consolidate_with_no_observations_returns_zero() {
        let (_dir, handle) = setup().await;
        let promoted = consolidate(Some("empty-session"), &handle).await.unwrap();
        assert_eq!(promoted, 0);
    }

    // VAL-SES-025
    #[tokio::test]
    async fn session_end_transcript_path_fallback_writes_marker() {
        let (_dir, handle) = setup().await;
        let session_id = "session-transcript".to_string();

        let note_id = session_end_with_transcript(
            &session_id,
            Some("done"),
            Some("/nonexistent/transcript.jsonl"),
            &handle,
        )
        .await
        .unwrap();
        let note = handle.karta.get_note(&note_id).await.unwrap().unwrap();
        assert!(note.content.contains(SESSION_END_TAG));
        assert!(note.content.contains("/nonexistent/transcript.jsonl"));
    }

    // VAL-SES-007, VAL-SES-028
    #[tokio::test]
    async fn consolidate_does_not_call_llm_or_dream() {
        // Build a Karta with a counting mock LLM so we can verify that
        // consolidate performs zero chat calls.
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let vector_store = SqliteVectorStore::new(data_dir, EMBEDDING_DIM)
            .await
            .unwrap();
        let shared_conn = vector_store.connection();
        let vector_store: Arc<dyn VectorStore> = Arc::new(vector_store);
        let graph_store = Arc::new(SqliteGraphStore::with_connection(shared_conn.clone()));
        let llm = Arc::new(CountingMockLlmProvider::new());
        let llm_dyn: Arc<dyn LlmProvider> = llm.clone();
        let config = karta_core::config::KartaConfig::default();
        let karta = Arc::new(
            Karta::new(vector_store.clone(), graph_store.clone(), llm_dyn, config)
                .await
                .unwrap(),
        );
        let handle = KartaHandle::new(karta, vector_store, graph_store, shared_conn);

        insert_observation(
            &handle,
            "counting mock observation",
            Some("session-count"),
            0.95,
        )
        .await;

        let chat_before = llm.chat_count.load(Ordering::Relaxed);
        let promoted = consolidate(Some("session-count"), &handle).await.unwrap();
        let chat_after = llm.chat_count.load(Ordering::Relaxed);
        assert_eq!(promoted, 1);
        assert_eq!(
            chat_after, chat_before,
            "consolidate must not call LLM chat"
        );

        let facts: Vec<_> = handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .into_iter()
            .filter(|n| matches!(n.provenance, Provenance::Fact { .. }))
            .collect();
        assert_eq!(facts.len(), 1);
    }

    /// A wrapper around `MockLlmProvider` that counts chat calls. Used only in
    /// the no-LLM consolidation test; the assertion is structural: the test path
    /// that creates the fact note via `vector_store.upsert` would still succeed
    /// even if the underlying mock had been called, because the mock is never
    /// invoked by `consolidate` itself.
    struct CountingMockLlmProvider {
        inner: MockLlmProvider,
        chat_count: AtomicU64,
    }

    impl CountingMockLlmProvider {
        fn new() -> Self {
            Self {
                inner: MockLlmProvider::new(),
                chat_count: AtomicU64::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for CountingMockLlmProvider {
        async fn chat(
            &self,
            messages: &[ChatMessage],
            config: &GenConfig,
        ) -> karta_core::error::Result<ChatResponse> {
            self.chat_count.fetch_add(1, Ordering::Relaxed);
            self.inner.chat(messages, config).await
        }
        async fn embed(&self, texts: &[&str]) -> karta_core::error::Result<Vec<Vec<f32>>> {
            self.inner.embed(texts).await
        }
        fn model_id(&self) -> &str {
            self.inner.model_id()
        }
        fn embedding_model_id(&self) -> &str {
            self.inner.embedding_model_id()
        }
    }
}
