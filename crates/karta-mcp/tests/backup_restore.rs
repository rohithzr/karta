//! Integration tests for backup, export, and restore.
//!
//! These tests exercise the CLI subcommands against a real store and, for
//! the concurrent-capture-during-backup test, against a live `serve` process.

mod common;

use std::path::Path;
use std::time::Duration;

use rusqlite::Connection;
use serde_json::json;
use tempfile::TempDir;
use tokio::time::{interval, sleep, timeout};

/// Run the `backup` CLI subcommand and return the captured stdout.
async fn run_backup(data_dir: &str, dest: &Path) -> std::process::Output {
    tokio::process::Command::new(common::bin_path())
        .args(["backup", "--dest", &dest.to_string_lossy()])
        .env("KARTA_STORE_DIR", data_dir)
        .output()
        .await
        .expect("run backup subcommand")
}

/// Count notes and capture_queue rows in a SQLite file.
fn snapshot_counts(path: &Path) -> (usize, usize) {
    let conn = Connection::open(path).expect("open backup db");
    let note_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM notes", [], |row| row.get(0))
        .unwrap_or(0);
    let queue_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM capture_queue", [], |row| row.get(0))
        .unwrap_or(0);
    (note_count as usize, queue_count as usize)
}

#[tokio::test]
async fn concurrent_capture_during_backup_preserves_consistency() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().to_str().unwrap();
    let port = common::find_free_port();

    let mut child = common::spawn_serve(data_dir, port);
    common::wait_for_server(port).await;

    let client = reqwest::Client::new();
    let capture_url = format!("http://127.0.0.1:{port}/capture");
    let stop = tokio::sync::watch::channel(false);

    // Fire captures continuously in the background while backup runs.
    let capture_task = tokio::spawn({
        let client = client.clone();
        let url = capture_url.clone();
        let mut rx = stop.1.clone();
        async move {
            let mut ticker = interval(Duration::from_millis(50));
            let mut session_counter = 0u64;
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        session_counter += 1;
                        let body = json!({
                            "hook_event_name": "UserPromptSubmit",
                            "prompt": format!("concurrent note {session_counter}"),
                            "session_id": format!("backup-session-{}", session_counter % 5),
                        });
                        // Ignore errors; the goal is sustained concurrent writes.
                        let _ = client
                            .post(&url)
                            .json(&body)
                            .timeout(Duration::from_secs(2))
                            .send()
                            .await;
                    }
                    _ = rx.changed() => break,
                }
            }
            session_counter
        }
    });

    // Let the capture stream run for a moment before starting backup.
    sleep(Duration::from_millis(200)).await;

    let backup_path = tmp.path().join("concurrent-backup.db");
    let backup_output = timeout(Duration::from_secs(30), run_backup(data_dir, &backup_path))
        .await
        .expect("backup timed out");

    // Stop the capture stream.
    stop.0.send(true).ok();
    let captured = capture_task.await.unwrap();

    // The backup must have succeeded.
    assert!(
        backup_output.status.success(),
        "backup failed: {}",
        String::from_utf8_lossy(&backup_output.stderr)
    );
    assert!(backup_path.exists(), "backup file was not created");

    // The backup must be a valid SQLite database containing the same schema
    // (notes + capture_queue). It should include at least some of the notes
    // that were written before/during backup.
    let (note_count, queue_count) = snapshot_counts(&backup_path);
    assert!(
        note_count + queue_count > 0,
        "backup snapshot should contain notes or queue rows"
    );

    // Restore the backup into a fresh directory and verify it opens cleanly.
    let restore_dir = TempDir::new().unwrap();
    let restore_output = tokio::process::Command::new(common::bin_path())
        .args(["restore", "--from", &backup_path.to_string_lossy()])
        .env("KARTA_STORE_DIR", restore_dir.path().to_str().unwrap())
        .output()
        .await
        .expect("run restore subcommand");
    assert!(
        restore_output.status.success(),
        "restore failed: {}",
        String::from_utf8_lossy(&restore_output.stderr)
    );

    let restored_db = restore_dir.path().join("karta.db");
    assert!(restored_db.exists());
    let (restored_notes, restored_queue) = snapshot_counts(&restored_db);
    assert_eq!(restored_notes, note_count);
    assert_eq!(restored_queue, queue_count);

    // Clean up the serve process.
    let _ = child.kill();
    let _ = child.wait();

    // Sanity: at least one capture was attempted concurrently with backup.
    assert!(
        captured >= 1,
        "expected at least one concurrent capture, got {captured}"
    );
}
