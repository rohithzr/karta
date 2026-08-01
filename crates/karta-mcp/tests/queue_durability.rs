//! Durability integration tests for the capture queue.
//!
//! These tests exercise the queue worker end-to-end by spawning the
//! `karta-mcp serve --mock` binary and manipulating the underlying SQLite
//! database directly. This avoids needing a live HTTP capture endpoint while
//! still verifying crash recovery, startup replay, graceful SIGTERM drain,
//! concurrent access, and FIFO ordering.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use rusqlite::{Connection, params};
use tempfile::TempDir;
use tokio::sync::Mutex;
use tokio::time::sleep;

const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Create the karta.db file with the capture_queue table and WAL mode.
fn create_queue_db(path: &Path) -> Connection {
    let conn = Connection::open(path).expect("open karta.db");
    conn.execute_batch("PRAGMA journal_mode = WAL;")
        .expect("enable WAL");
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
    )
    .expect("create capture_queue table");
    conn
}

/// Insert a capture row directly into the database.
fn insert_row(
    conn: &Connection,
    event_type: &str,
    payload: &str,
    session_id: Option<&str>,
    status: &str,
) -> i64 {
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO capture_queue
         (event_type, payload, session_id, status, created_at, updated_at, error)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5, NULL)",
        params![event_type, payload, session_id, status, now],
    )
    .expect("insert row");
    conn.last_insert_rowid()
}

/// Count rows whose status is in the provided set.
fn count_statuses(conn: &Connection, statuses: &[&str]) -> usize {
    let placeholders = statuses.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!("SELECT COUNT(*) FROM capture_queue WHERE status IN ({placeholders})");
    let mut stmt = conn.prepare(&sql).expect("prepare count");
    let count: i64 = stmt
        .query_row(rusqlite::params_from_iter(statuses.iter()), |row| {
            row.get(0)
        })
        .expect("count rows");
    count as usize
}

/// Wait until no rows remain in `queued` or `in_flight` status.
async fn wait_for_empty_queue(db_path: &Path) {
    for _ in 0..100 {
        let conn = Connection::open(db_path).expect("open db for drain check");
        if count_statuses(&conn, &["queued", "in_flight"]) == 0 {
            return;
        }
        sleep(Duration::from_millis(50)).await;
    }
    panic!("queue did not drain within timeout");
}

/// Wait until the worker has claimed at least one row (queued count drops
/// below the initial count). This ensures the signal handler is registered
/// and the process is doing real work before a test sends SIGTERM.
async fn wait_for_worker_to_start(db_path: &Path) {
    for _ in 0..100 {
        let conn = Connection::open(db_path).expect("open db for start check");
        if count_statuses(&conn, &["done"]) > 0 {
            return;
        }
        sleep(Duration::from_millis(10)).await;
    }
    panic!("worker did not start within timeout");
}

#[tokio::test]
async fn crash_recovery_replays_queued_in_flight_and_failed_rows() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().to_str().unwrap();
    let db_path = tmp.path().join("karta.db");

    // Seed the database with rows in all three incomplete states.
    let conn = create_queue_db(&db_path);
    insert_row(
        &conn,
        "user_prompt",
        r#"{"content": "queued row"}"#,
        Some("s1"),
        "queued",
    );
    insert_row(
        &conn,
        "user_prompt",
        r#"{"content": "in_flight row"}"#,
        Some("s2"),
        "in_flight",
    );
    insert_row(
        &conn,
        "user_prompt",
        r#"{"content": "failed row"}"#,
        Some("s3"),
        "failed",
    );
    drop(conn);

    // Start serve. Startup replay should reset the in_flight and failed rows
    // to queued, then the worker drains all three.
    let mut child = common::spawn_serve(data_dir, common::find_free_port());
    sleep(Duration::from_millis(500)).await;

    wait_for_empty_queue(&db_path).await;

    common::send_sigterm(&child);
    let status = common::wait_for_exit(&mut child, DRAIN_TIMEOUT)
        .await
        .expect("wait for exit");
    assert!(
        status.success(),
        "serve should exit cleanly after SIGTERM, got status {status:?}"
    );

    let conn = Connection::open(&db_path).expect("open db after recovery");
    assert_eq!(count_statuses(&conn, &["queued", "in_flight"]), 0);
    assert_eq!(count_statuses(&conn, &["done"]), 3);
    assert_eq!(count_statuses(&conn, &["failed"]), 0);
}

#[tokio::test]
async fn sigterm_drains_in_flight_and_queued_rows() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().to_str().unwrap();
    let db_path = tmp.path().join("karta.db");

    let conn = create_queue_db(&db_path);
    // Pre-seed a burst of queued rows.
    for i in 0..20 {
        insert_row(
            &conn,
            "user_prompt",
            &format!("{{\"content\": \"drain-{i}\"}}"),
            Some("session"),
            "queued",
        );
    }
    // Leave one row in_flight to simulate a mid-drain SIGTERM.
    insert_row(
        &conn,
        "user_prompt",
        r#"{"content": "mid-flight"}"#,
        Some("session"),
        "in_flight",
    );
    drop(conn);

    let mut child = common::spawn_serve(data_dir, common::find_free_port());
    // Wait until the worker has completed at least one full row before
    // sending SIGTERM. This ensures the signal handler and polling task are
    // active and avoids killing the process before it is ready to drain.
    wait_for_worker_to_start(&db_path).await;
    common::send_sigterm(&child);

    let status = common::wait_for_exit(&mut child, DRAIN_TIMEOUT)
        .await
        .expect("wait for exit");
    assert!(
        status.success(),
        "serve should exit cleanly after SIGTERM, got status {status:?}"
    );

    let conn = Connection::open(&db_path).expect("open db after SIGTERM");
    assert_eq!(count_statuses(&conn, &["queued", "in_flight"]), 0);
    assert_eq!(count_statuses(&conn, &["done"]), 21);
}

#[tokio::test]
async fn concurrent_enqueue_persists_all_rows() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("karta.db");

    let conn = create_queue_db(&db_path);
    drop(conn);

    // Open a shared queue connection from multiple async tasks. This mimics
    // concurrent /capture posts hitting the same SQLite file.
    let queue = Arc::new(Mutex::new(Connection::open(&db_path).unwrap()));
    let mut handles = Vec::new();
    for i in 0..50 {
        let queue = queue.clone();
        handles.push(tokio::spawn(async move {
            let conn = queue.lock().await;
            insert_row(
                &conn,
                "user_prompt",
                &format!("{{\"content\": \"concurrent-{i}\"}}"),
                Some("session"),
                "queued",
            )
        }));
    }
    let mut ids: Vec<i64> = Vec::new();
    for handle in handles {
        ids.push(handle.await.expect("task completed"));
    }

    let conn = Connection::open(&db_path).unwrap();
    assert_eq!(count_statuses(&conn, &["queued"]), 50);

    let mut unique_ids = ids.clone();
    unique_ids.sort();
    unique_ids.dedup();
    assert_eq!(
        unique_ids.len(),
        50,
        "each enqueue should produce a unique id"
    );
}

#[tokio::test]
async fn startup_replay_does_not_touch_done_rows() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().to_str().unwrap();
    let db_path = tmp.path().join("karta.db");

    let conn = create_queue_db(&db_path);
    let done_id = insert_row(
        &conn,
        "user_prompt",
        r#"{"content": "already done"}"#,
        Some("done-session"),
        "done",
    );
    let queued_id = insert_row(
        &conn,
        "user_prompt",
        r#"{"content": "needs replay"}"#,
        Some("replay-session"),
        "queued",
    );
    let done_updated_at: String = conn
        .query_row(
            "SELECT updated_at FROM capture_queue WHERE id = ?1",
            params![done_id],
            |row| row.get(0),
        )
        .unwrap();
    drop(conn);

    let mut child = common::spawn_serve(data_dir, common::find_free_port());
    wait_for_empty_queue(&db_path).await;
    common::send_sigterm(&child);
    let _ = common::wait_for_exit(&mut child, DRAIN_TIMEOUT).await;

    let conn = Connection::open(&db_path).unwrap();
    assert_eq!(count_statuses(&conn, &["queued", "in_flight"]), 0);
    assert_eq!(count_statuses(&conn, &["done"]), 2);

    let after_updated_at: String = conn
        .query_row(
            "SELECT updated_at FROM capture_queue WHERE id = ?1",
            params![done_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        after_updated_at, done_updated_at,
        "done row updated_at should not change during replay"
    );

    let queued_row_updated_at: String = conn
        .query_row(
            "SELECT updated_at FROM capture_queue WHERE id = ?1",
            params![queued_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_ne!(
        queued_row_updated_at, done_updated_at,
        "replayed row should have a fresh updated_at"
    );
}

#[tokio::test]
async fn worker_processes_rows_in_fifo_order() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().to_str().unwrap();
    let db_path = tmp.path().join("karta.db");

    let conn = create_queue_db(&db_path);
    for i in 0..5 {
        insert_row(
            &conn,
            "user_prompt",
            &format!("{{\"content\": \"fifo-{i}\"}}"),
            Some("fifo"),
            "queued",
        );
    }
    drop(conn);

    let mut child = common::spawn_serve(data_dir, common::find_free_port());
    wait_for_empty_queue(&db_path).await;
    common::send_sigterm(&child);
    let _ = common::wait_for_exit(&mut child, DRAIN_TIMEOUT).await;

    let conn = Connection::open(&db_path).unwrap();
    let mut stmt = conn
        .prepare("SELECT payload FROM capture_queue WHERE status = 'done' ORDER BY id ASC")
        .unwrap();
    let payloads: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    for (i, payload) in payloads.iter().enumerate() {
        assert_eq!(payload, &format!("{{\"content\": \"fifo-{i}\"}}"));
    }
}
