//! Durable capture queue backed by the same SQLite file as `karta_core`.
//!
//! The queue opens its own `rusqlite::Connection` to `{data_dir}/karta.db` and
//! sets `PRAGMA busy_timeout = 5000` to avoid immediate `SQLITE_BUSY` errors
//! when `karta_core` is writing concurrently.
//!
//! Capture rows are stored in the `capture_queue` table with a `status` column
//! that transitions `queued` → `in_flight` → `done`/`failed`. On server
//! startup all incomplete rows (`queued`, `in_flight`, `failed`) are reset to
//! `queued` so the worker can drain them.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde_json::Value;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::karta_handle::KartaHandle;
use crate::session;

const DEFAULT_POLL_INTERVAL_MS: u64 = 100;

/// One row from the `capture_queue` table.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueRow {
    pub id: i64,
    pub event_type: String,
    pub payload: Value,
    pub session_id: Option<String>,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
    pub error: Option<String>,
}

impl QueueRow {
    fn from_row(row: &Row) -> std::result::Result<Self, rusqlite::Error> {
        let id: i64 = row.get(0)?;
        let event_type: String = row.get(1)?;
        let payload_str: String = row.get(2)?;
        let payload: Value = serde_json::from_str(&payload_str).unwrap_or(Value::Null);
        let session_id: Option<String> = row.get(3)?;
        let status: String = row.get(4)?;
        let created_at: String = row.get(5)?;
        let updated_at: String = row.get(6)?;
        let error: Option<String> = row.get(7)?;
        Ok(QueueRow {
            id,
            event_type,
            payload,
            session_id,
            status,
            created_at,
            updated_at,
            error,
        })
    }
}

/// Durable capture queue.
///
/// This struct is `Sync` and can be shared between the HTTP capture endpoint
/// and the background worker via an `Arc`.
pub struct CaptureQueue {
    conn: Mutex<Connection>,
    #[allow(dead_code)]
    data_dir: String,
}

impl CaptureQueue {
    /// Open a queue connection to `{data_dir}/karta.db`.
    ///
    /// The connection is created with `PRAGMA busy_timeout = 5000` and the
    /// `capture_queue` table is created if it does not already exist.
    pub async fn new(data_dir: &str) -> Result<Self> {
        let path = Path::new(data_dir).join("karta.db");
        let conn = Connection::open(&path).with_context(|| {
            format!(
                "failed to open capture queue database at {}",
                path.display()
            )
        })?;

        // Set busy_timeout before anything else so concurrent karta_core writes
        // do not immediately return SQLITE_BUSY.
        conn.execute_batch("PRAGMA busy_timeout = 5000;")?;

        // WAL mode is normally set by karta_core; set it here as well so the
        // queue connection behaves consistently even if opened first.
        conn.execute_batch("PRAGMA journal_mode = WAL;")?;

        Self::ensure_schema(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
            data_dir: data_dir.to_string(),
        })
    }

    /// Return the configured data directory.
    #[allow(dead_code)]
    pub fn data_dir(&self) -> &str {
        &self.data_dir
    }

    fn ensure_schema(conn: &Connection) -> Result<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS capture_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                session_id TEXT,
                status TEXT NOT NULL CHECK(status IN ('queued', 'in_flight', 'done', 'failed')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT
            )",
            [],
        )?;
        Ok(())
    }

    /// Insert a new capture event into the queue.
    ///
    /// Returns the auto-generated row id. The row is inserted with
    /// `status = 'queued'`. This method is used by the `/capture` endpoint.
    #[allow(dead_code)]
    pub async fn enqueue(
        &self,
        event_type: &str,
        payload: &Value,
        session_id: Option<&str>,
    ) -> Result<i64> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO capture_queue
             (event_type, payload, session_id, status, created_at, updated_at, error)
             VALUES (?1, ?2, ?3, 'queued', ?4, ?4, NULL)",
            params![event_type, payload.to_string(), session_id, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Count rows that are not yet finished (`queued` or `in_flight`).
    pub async fn depth(&self) -> Result<usize> {
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM capture_queue WHERE status IN ('queued', 'in_flight')",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Reset all incomplete rows to `queued`.
    ///
    /// This is called on server startup so that rows left `in_flight` or
    /// `failed` by a previous process are reprocessed. Returns the number of
    /// rows that were reset.
    pub async fn replay_incomplete(&self) -> Result<usize> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        let changed = conn.execute(
            "UPDATE capture_queue
             SET status = 'queued', updated_at = ?1, error = NULL
             WHERE status IN ('queued', 'in_flight', 'failed')",
            params![now],
        )?;
        Ok(changed)
    }

    /// Claim the next queued row.
    ///
    /// The oldest queued row (by `id`) is atomically updated to `in_flight`
    /// and returned. Rows are selected in FIFO order.
    pub async fn claim_next(&self) -> Result<Option<QueueRow>> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        let row = conn
            .query_row(
                "UPDATE capture_queue
                 SET status = 'in_flight', updated_at = ?1
                 WHERE id = (
                     SELECT id FROM capture_queue
                     WHERE status = 'queued'
                     ORDER BY id ASC
                     LIMIT 1
                 )
                 RETURNING id, event_type, payload, session_id, status, created_at, updated_at, error",
                params![now],
                QueueRow::from_row,
            )
            .optional()?;
        Ok(row)
    }

    /// Mark a row as successfully processed.
    pub async fn mark_done(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE capture_queue
             SET status = 'done', updated_at = ?1, error = NULL
             WHERE id = ?2",
            params![now, id],
        )?;
        Ok(())
    }

    /// Mark a row as failed and record the error message.
    pub async fn mark_failed(&self, id: i64, error: &str) -> Result<()> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE capture_queue
             SET status = 'failed', updated_at = ?1, error = ?2
             WHERE id = ?3",
            params![now, error, id],
        )?;
        Ok(())
    }

    /// Return a snapshot of all rows for inspection in tests.
    #[cfg(test)]
    pub async fn all_rows(&self) -> Result<Vec<QueueRow>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, event_type, payload, session_id, status, created_at, updated_at, error
             FROM capture_queue
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map([], QueueRow::from_row)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Return the row with the given id, if it exists.
    #[cfg(test)]
    pub async fn get_row(&self, id: i64) -> Result<Option<QueueRow>> {
        let conn = self.conn.lock().await;
        let row = conn
            .query_row(
                "SELECT id, event_type, payload, session_id, status, created_at, updated_at, error
                 FROM capture_queue
                 WHERE id = ?1",
                params![id],
                QueueRow::from_row,
            )
            .optional()?;
        Ok(row)
    }
}

/// Run the background worker that drains the queue.
///
/// The worker polls the queue for `queued` rows, claims each row (setting it to
/// `in_flight`), processes it via `karta.add_note_with_clock`, and then marks
/// it `done` or `failed`. When the `cancel` token is triggered the worker
/// stops accepting new polling sleep cycles and drains any remaining rows
/// before returning.
///
/// This function is meant to be spawned as a `tokio::task`.
pub async fn run_worker(
    queue: Arc<CaptureQueue>,
    handle: KartaHandle,
    cancel: CancellationToken,
    precompact_enabled: bool,
) {
    loop {
        match queue.claim_next().await {
            Ok(Some(row)) => {
                tracing::info!(
                    row_id = row.id,
                    event_type = %row.event_type,
                    session_id = ?row.session_id,
                    "processing queue row"
                );
                match process_row(&handle, &row, precompact_enabled).await {
                    Ok(_) => {
                        if let Err(e) = queue.mark_done(row.id).await {
                            tracing::error!(
                                row_id = row.id,
                                error = %e,
                                "failed to mark row done"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            row_id = row.id,
                            error = %e,
                            "queue row processing failed"
                        );
                        if let Err(e2) = queue.mark_failed(row.id, &e.to_string()).await {
                            tracing::error!(
                                row_id = row.id,
                                error = %e2,
                                "failed to mark row failed"
                            );
                        }
                    }
                }
            }
            Ok(None) => {
                if cancel.is_cancelled() {
                    break;
                }
                // Wait for either the cancellation token or the poll interval.
                tokio::select! {
                    _ = cancel.cancelled() => {}
                    _ = tokio::time::sleep(Duration::from_millis(DEFAULT_POLL_INTERVAL_MS)) => {}
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to claim next queue row");
                tokio::time::sleep(Duration::from_millis(DEFAULT_POLL_INTERVAL_MS)).await;
            }
        }
    }

    tracing::info!("queue worker exiting");
}

/// Process a single queue row by writing it into `karta_core`.
async fn process_row(handle: &KartaHandle, row: &QueueRow, precompact_enabled: bool) -> Result<()> {
    match row.event_type.as_str() {
        "session_end" => {
            let summary = row.payload.get("summary").and_then(|v| v.as_str());
            let transcript_path = row.payload.get("transcript_path").and_then(|v| v.as_str());
            if let Some(session_id) = row.session_id.as_deref() {
                session::session_end_with_transcript(session_id, summary, transcript_path, handle)
                    .await?;
            } else {
                // No session id: store a generic marker note so the row is not lost.
                let content = extract_content("session_end", &row.payload);
                handle
                    .karta
                    .add_note_with_clock(&content, None, None, karta_core::ClockContext::now())
                    .await?;
            }
        }
        "pre_compact" => {
            if let Some(session_id) = row.session_id.as_deref() {
                session::pre_compact(session_id, handle, precompact_enabled).await?;
            }
            // Without a session id there is nothing meaningful to compact;
            // the row is still considered processed.
        }
        _ => {
            let content = extract_content(&row.event_type, &row.payload);
            let session_id = row.session_id.as_deref();
            let ctx = karta_core::ClockContext::now();
            handle
                .karta
                .add_note_with_clock(&content, session_id, None, ctx)
                .await?;
        }
    }
    Ok(())
}

/// Extract a string to store as the note content from the capture payload.
///
/// This is a best-effort content extraction used until `session.rs` is
/// implemented. Known fields (`content`, `text`, `prompt`, `summary`,
/// `last_assistant_message`, `task_result`, `tool_output`) are preferred;
/// otherwise a marker line plus the JSON payload is stored.
fn extract_content(event_type: &str, payload: &Value) -> String {
    let preferred = payload
        .get("content")
        .or_else(|| payload.get("text"))
        .or_else(|| payload.get("prompt"))
        .or_else(|| payload.get("summary"))
        .or_else(|| payload.get("last_assistant_message"))
        .or_else(|| payload.get("task_result"))
        .or_else(|| payload.get("tool_output"))
        .and_then(|v| v.as_str());

    match preferred {
        Some(text) if !text.is_empty() => text.to_string(),
        _ => format!("[{event_type}] {payload}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    use crate::session::{PRECOMPACT_TAG, SESSION_END_TAG};
    use karta_core::note::{MemoryNote, Provenance};

    async fn setup_queue() -> (TempDir, Arc<CaptureQueue>) {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let queue = Arc::new(CaptureQueue::new(data_dir).await.unwrap());
        (dir, queue)
    }

    async fn setup_karta(data_dir: &str) -> KartaHandle {
        KartaHandle::open_mock(data_dir).await.unwrap()
    }

    #[tokio::test]
    async fn creates_table_and_returns_zero_depth() {
        let (_dir, queue) = setup_queue().await;
        assert_eq!(queue.depth().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn enqueue_returns_id_and_increases_depth() {
        let (_dir, queue) = setup_queue().await;
        let payload = serde_json::json!({"content": "hello"});
        let id = queue
            .enqueue("user_prompt", &payload, Some("s1"))
            .await
            .unwrap();
        assert!(id > 0);
        assert_eq!(queue.depth().await.unwrap(), 1);

        let row = queue.get_row(id).await.unwrap().unwrap();
        assert_eq!(row.event_type, "user_prompt");
        assert_eq!(row.payload, payload);
        assert_eq!(row.session_id, Some("s1".to_string()));
        assert_eq!(row.status, "queued");
        assert!(row.error.is_none());
    }

    #[tokio::test]
    async fn claim_next_returns_oldest_row_and_marks_in_flight() {
        let (_dir, queue) = setup_queue().await;
        queue
            .enqueue(
                "user_prompt",
                &serde_json::json!({"content": "first"}),
                None,
            )
            .await
            .unwrap();
        queue
            .enqueue(
                "user_prompt",
                &serde_json::json!({"content": "second"}),
                None,
            )
            .await
            .unwrap();

        let first = queue.claim_next().await.unwrap().unwrap();
        assert_eq!(first.payload["content"], "first");
        assert_eq!(first.status, "in_flight");

        let second = queue.claim_next().await.unwrap().unwrap();
        assert_eq!(second.payload["content"], "second");
        assert_eq!(queue.depth().await.unwrap(), 2);

        // No more rows.
        assert!(queue.claim_next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn mark_done_and_failed_update_status() {
        let (_dir, queue) = setup_queue().await;
        let id = queue
            .enqueue(
                "observation",
                &serde_json::json!({"tool_output": "x"}),
                None,
            )
            .await
            .unwrap();
        let row = queue.claim_next().await.unwrap().unwrap();
        assert_eq!(row.id, id);

        queue.mark_done(id).await.unwrap();
        let row = queue.get_row(id).await.unwrap().unwrap();
        assert_eq!(row.status, "done");
        assert!(row.error.is_none());

        queue.mark_failed(id, "boom").await.unwrap();
        let row = queue.get_row(id).await.unwrap().unwrap();
        assert_eq!(row.status, "failed");
        assert_eq!(row.error, Some("boom".to_string()));
    }

    #[tokio::test]
    async fn replay_incomplete_resets_failed_and_in_flight() {
        let (_dir, queue) = setup_queue().await;
        let id1 = queue
            .enqueue("user_prompt", &serde_json::json!({"content": "a"}), None)
            .await
            .unwrap();
        let id2 = queue
            .enqueue("user_prompt", &serde_json::json!({"content": "b"}), None)
            .await
            .unwrap();
        let id3 = queue
            .enqueue("user_prompt", &serde_json::json!({"content": "c"}), None)
            .await
            .unwrap();

        queue.claim_next().await.unwrap().unwrap(); // id1 -> in_flight
        queue.mark_failed(id2, "err").await.unwrap(); // id2 -> failed
        // id3 stays queued

        assert_eq!(queue.replay_incomplete().await.unwrap(), 3);

        for id in [id1, id2, id3] {
            let row = queue.get_row(id).await.unwrap().unwrap();
            assert_eq!(row.status, "queued");
            assert!(row.error.is_none());
        }
    }

    #[tokio::test]
    async fn replay_incomplete_does_not_touch_done_rows() {
        let (_dir, queue) = setup_queue().await;
        let id = queue
            .enqueue("user_prompt", &serde_json::json!({"content": "done"}), None)
            .await
            .unwrap();
        queue.claim_next().await.unwrap().unwrap();
        queue.mark_done(id).await.unwrap();
        let before = queue.get_row(id).await.unwrap().unwrap().updated_at.clone();

        assert_eq!(queue.replay_incomplete().await.unwrap(), 0);

        let after = queue.get_row(id).await.unwrap().unwrap();
        assert_eq!(after.status, "done");
        assert_eq!(after.updated_at, before);
    }

    #[tokio::test]
    async fn invalid_status_is_rejected_by_sqlite() {
        let (_dir, queue) = setup_queue().await;
        let conn = queue.conn.lock().await;
        let err = conn
            .execute(
                "INSERT INTO capture_queue (event_type, payload, session_id, status, created_at, updated_at, error)
                 VALUES ('x', '{}', NULL, 'bad_status', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', NULL)",
                [],
            )
            .unwrap_err();
        assert!(err.to_string().contains("CHECK constraint failed"));
    }

    #[tokio::test]
    async fn worker_drains_row_to_done() {
        let (dir, queue) = setup_queue().await;
        let handle = setup_karta(dir.path().to_str().unwrap()).await;

        let payload = serde_json::json!({"content": "worker note"});
        let id = queue
            .enqueue("user_prompt", &payload, Some("s1"))
            .await
            .unwrap();

        let cancel = CancellationToken::new();
        let worker_cancel = cancel.clone();
        let worker_queue = queue.clone();
        let worker_handle = handle.clone();
        let worker = tokio::spawn(async move {
            run_worker(worker_queue, worker_handle, worker_cancel, false).await;
        });

        // Wait for the worker to drain the row.
        for _ in 0..100 {
            if queue.depth().await.unwrap() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), worker).await;

        let row = queue.get_row(id).await.unwrap().unwrap();
        assert_eq!(row.status, "done");
        assert!(row.error.is_none());
        assert_eq!(handle.karta.note_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn failed_rows_do_not_block_worker() {
        let (dir, queue) = setup_queue().await;
        let handle = setup_karta(dir.path().to_str().unwrap()).await;

        let failed_payload = serde_json::json!({"content": "stuck"});
        let failed_id = queue
            .enqueue("user_prompt", &failed_payload, None)
            .await
            .unwrap();
        // Simulate a previous failed attempt: claim and mark it failed.
        queue.claim_next().await.unwrap().unwrap();
        queue
            .mark_failed(failed_id, "previous failure")
            .await
            .unwrap();

        let ok_payload = serde_json::json!({"content": "ok"});
        let ok_id = queue
            .enqueue("user_prompt", &ok_payload, None)
            .await
            .unwrap();

        let cancel = CancellationToken::new();
        let worker = tokio::spawn({
            let queue = queue.clone();
            let handle = handle.clone();
            let cancel = cancel.clone();
            async move { run_worker(queue, handle, cancel, false).await }
        });

        for _ in 0..100 {
            if queue.depth().await.unwrap() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), worker).await;

        let failed_row = queue.get_row(failed_id).await.unwrap().unwrap();
        assert_eq!(failed_row.status, "failed");
        assert_eq!(failed_row.error, Some("previous failure".to_string()));

        let ok_row = queue.get_row(ok_id).await.unwrap().unwrap();
        assert_eq!(ok_row.status, "done");
        assert!(ok_row.error.is_none());

        assert_eq!(handle.karta.note_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn worker_processes_rows_in_fifo_order() {
        let (dir, queue) = setup_queue().await;
        let handle = setup_karta(dir.path().to_str().unwrap()).await;

        for i in 0..5 {
            queue
                .enqueue(
                    "user_prompt",
                    &serde_json::json!({"content": format!("note-{i}")}),
                    Some("session"),
                )
                .await
                .unwrap();
        }

        let cancel = CancellationToken::new();
        let worker = tokio::spawn({
            let queue = queue.clone();
            let handle = handle.clone();
            let cancel = cancel.clone();
            async move { run_worker(queue, handle, cancel, false).await }
        });

        for _ in 0..200 {
            if queue.depth().await.unwrap() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), worker).await;

        let rows = queue.all_rows().await.unwrap();
        assert_eq!(rows.len(), 5);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.status, "done");
            assert_eq!(row.payload["content"], format!("note-{i}"));
        }
        assert_eq!(handle.karta.note_count().await.unwrap(), 5);
    }

    #[tokio::test]
    async fn graceful_drain_processes_remaining_rows() {
        let (dir, queue) = setup_queue().await;
        let handle = setup_karta(dir.path().to_str().unwrap()).await;

        let cancel = CancellationToken::new();
        let worker = tokio::spawn({
            let queue = queue.clone();
            let handle = handle.clone();
            let cancel = cancel.clone();
            async move { run_worker(queue, handle, cancel, false).await }
        });

        // Let the worker start and wait on the poll interval.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Enqueue rows after the worker has started; this simulates a burst
        // arriving just before a shutdown signal.
        let mut ids = Vec::new();
        for i in 0..5 {
            let id = queue
                .enqueue(
                    "user_prompt",
                    &serde_json::json!({"content": format!("drain-{i}")}),
                    None,
                )
                .await
                .unwrap();
            ids.push(id);
        }

        // Signal shutdown immediately; the worker must still drain the rows.
        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(5), worker).await;

        for id in ids {
            let row = queue.get_row(id).await.unwrap().unwrap();
            assert_eq!(row.status, "done");
        }
        assert_eq!(handle.karta.note_count().await.unwrap(), 5);
    }

    #[tokio::test]
    async fn busy_timeout_is_set_to_5000() {
        let (_dir, queue) = setup_queue().await;
        let conn = queue.conn.lock().await;
        let timeout: i32 = conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
    }

    async fn insert_observation(
        handle: &KartaHandle,
        content: &str,
        session_id: &str,
        confidence: f32,
    ) {
        let mut note = MemoryNote::new(content.to_string());
        note.provenance = Provenance::Observed;
        note.confidence = confidence;
        note.session_id = Some(session_id.to_string());
        const DIM: usize = 1536;
        let value = 1.0 / (DIM as f32).sqrt();
        note.embedding = vec![value; DIM];
        handle.vector_store.upsert(&note).await.unwrap();
    }

    async fn run_worker_until_queue_empty(
        queue: &Arc<CaptureQueue>,
        handle: &KartaHandle,
        precompact: bool,
    ) {
        let cancel = CancellationToken::new();
        let worker = tokio::spawn({
            let queue = queue.clone();
            let handle = handle.clone();
            let cancel = cancel.clone();
            async move { run_worker(queue, handle, cancel, precompact).await }
        });

        for _ in 0..200 {
            if queue.depth().await.unwrap() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), worker).await;
    }

    #[tokio::test]
    async fn queue_session_end_writes_marker_and_consolidates() {
        let (dir, queue) = setup_queue().await;
        let handle = setup_karta(dir.path().to_str().unwrap()).await;
        let session_id = "queue-end-1";

        insert_observation(&handle, "q high confidence fact", session_id, 0.9).await;
        insert_observation(&handle, "q low confidence noise", session_id, 0.2).await;

        let payload = serde_json::json!({
            "session_id": session_id,
            "summary": "queue session end",
        });
        let id = queue
            .enqueue("session_end", &payload, Some(session_id))
            .await
            .unwrap();

        run_worker_until_queue_empty(&queue, &handle, false).await;

        let row = queue.get_row(id).await.unwrap().unwrap();
        assert_eq!(row.status, "done");
        assert!(row.error.is_none());

        let all = handle.karta.get_all_notes().await.unwrap();
        let markers: Vec<_> = all
            .into_iter()
            .filter(|n| n.content.contains(SESSION_END_TAG) || n.content.contains(PRECOMPACT_TAG))
            .collect();
        assert_eq!(markers.len(), 1);
        assert!(markers[0].content.contains("queue session end"));
        assert!(markers[0].content.contains(session_id));

        assert_eq!(handle.karta.note_count().await.unwrap(), 4);
    }

    #[tokio::test]
    async fn queue_pre_compact_enabled_writes_marker_and_consolidates() {
        let (dir, queue) = setup_queue().await;
        let handle = setup_karta(dir.path().to_str().unwrap()).await;
        let session_id = "queue-precompact-1";

        insert_observation(&handle, "pre high confidence fact", session_id, 0.85).await;
        insert_observation(&handle, "pre low confidence noise", session_id, 0.15).await;

        let payload = serde_json::json!({ "session_id": session_id });
        let id = queue
            .enqueue("pre_compact", &payload, Some(session_id))
            .await
            .unwrap();

        run_worker_until_queue_empty(&queue, &handle, true).await;

        let row = queue.get_row(id).await.unwrap().unwrap();
        assert_eq!(row.status, "done");
        assert!(row.error.is_none());

        let all = handle.karta.get_all_notes().await.unwrap();
        let markers: Vec<_> = all
            .into_iter()
            .filter(|n| n.content.contains(SESSION_END_TAG) || n.content.contains(PRECOMPACT_TAG))
            .collect();
        assert_eq!(markers.len(), 1);
        assert!(markers[0].content.contains("pre_compact"));
        assert!(markers[0].content.contains(session_id));

        assert_eq!(handle.karta.note_count().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn queue_pre_compact_disabled_does_nothing() {
        let (dir, queue) = setup_queue().await;
        let handle = setup_karta(dir.path().to_str().unwrap()).await;
        let session_id = "queue-precompact-off";

        insert_observation(&handle, "pre disabled fact", session_id, 0.85).await;

        let payload = serde_json::json!({ "session_id": session_id });
        let id = queue
            .enqueue("pre_compact", &payload, Some(session_id))
            .await
            .unwrap();

        run_worker_until_queue_empty(&queue, &handle, false).await;

        let row = queue.get_row(id).await.unwrap().unwrap();
        assert_eq!(row.status, "done");
        assert!(row.error.is_none());

        let all = handle.karta.get_all_notes().await.unwrap();
        let markers: Vec<_> = all
            .into_iter()
            .filter(|n| n.content.contains(SESSION_END_TAG) || n.content.contains(PRECOMPACT_TAG))
            .collect();
        assert!(markers.is_empty());
        assert_eq!(handle.karta.note_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn queue_pre_compact_and_session_end_no_double_consolidate() {
        let (dir, queue) = setup_queue().await;
        let handle = setup_karta(dir.path().to_str().unwrap()).await;
        let session_id = "queue-double-1";

        insert_observation(&handle, "double fact", session_id, 0.9).await;

        let pre_payload = serde_json::json!({ "session_id": session_id });
        let pre_id = queue
            .enqueue("pre_compact", &pre_payload, Some(session_id))
            .await
            .unwrap();
        let end_payload = serde_json::json!({
            "session_id": session_id,
            "summary": "queue double session end",
        });
        let end_id = queue
            .enqueue("session_end", &end_payload, Some(session_id))
            .await
            .unwrap();

        run_worker_until_queue_empty(&queue, &handle, true).await;

        for id in [pre_id, end_id] {
            let row = queue.get_row(id).await.unwrap().unwrap();
            assert_eq!(row.status, "done");
            assert!(row.error.is_none());
        }

        let all = handle.karta.get_all_notes().await.unwrap();
        let markers: Vec<_> = all
            .into_iter()
            .filter(|n| n.content.contains(SESSION_END_TAG) || n.content.contains(PRECOMPACT_TAG))
            .collect();
        assert_eq!(markers.len(), 2);

        // One observation + one promoted fact + two markers = 4 notes.
        assert_eq!(handle.karta.note_count().await.unwrap(), 4);
    }
}
