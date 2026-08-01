//! Cross-area flow tests that exercise multiple karta-mcp subsystems together.

mod common;

use std::path::Path;
use std::time::Duration;

use karta_core::note::Provenance;
use karta_mcp::session;
use karta_mcp::tools::{
    ConsolidateParams, SessionEndParams, handle_consolidate, handle_session_end,
};
use rusqlite::Connection;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::time::{interval, sleep};

// ---------------------------------------------------------------------------
// In-process cross-area flows
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capture_to_fetch_memories_retrieves_just_captured_note() {
    let rt = common::TestRuntime::new().await;
    let keyword = "xyzzy-capture-fetch-keyword";

    let (status, _) = rt
        .post_capture(json!({
            "hook_event_name": "UserPromptSubmit",
            "session_id": "cross-fetch",
            "prompt": keyword
        }))
        .await;
    assert_eq!(status, reqwest::StatusCode::ACCEPTED);
    rt.drain().await;

    let result = rt.handle.karta.fetch_memories(keyword, 5).await.unwrap();
    assert!(
        !result.note_ids.is_empty(),
        "fetch_memories should return the just-captured note"
    );
    assert_eq!(rt.handle.karta.note_count().await.unwrap(), 1);

    rt.cleanup().await;
}

#[tokio::test]
async fn full_session_lifecycle_drains_to_zero_with_orientation_captures_and_consolidation() {
    let rt = common::TestRuntime::new().await;
    let (session_id, orientation) = session::session_start("droid", Some("cross-area"), &rt.handle)
        .await
        .unwrap();
    assert!(
        !session_id.is_empty(),
        "session_start must produce a session id"
    );
    assert!(
        orientation.contains("droid") || orientation.is_empty(),
        "orientation context should be a string"
    );

    let captures = [
        json!({
            "hook_event_name": "UserPromptSubmit",
            "session_id": session_id,
            "prompt": "session lifecycle user prompt"
        }),
        json!({
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "tool_name": "Edit",
            "tool_output": "session lifecycle observation"
        }),
        json!({
            "hook_event_name": "Stop",
            "session_id": session_id,
            "last_assistant_message": "session lifecycle summary"
        }),
    ];
    for payload in &captures {
        let (status, _) = rt.post_capture(payload.clone()).await;
        assert_eq!(status, reqwest::StatusCode::ACCEPTED);
    }
    rt.drain().await;

    let end_params = SessionEndParams {
        session_id: session_id.clone(),
        summary: Some("session ended".to_string()),
    };
    let end_response = handle_session_end(&rt.handle, end_params).await.unwrap();
    let end_value: Value = serde_json::from_str(&end_response).unwrap();
    let written_note_id = end_value["written_note_id"].as_str().unwrap();
    assert!(
        !written_note_id.is_empty(),
        "session_end must return a written note id"
    );

    let consolidate_params = ConsolidateParams {
        session_id: Some(session_id.clone()),
    };
    let consolidate_response = handle_consolidate(&rt.handle, consolidate_params)
        .await
        .unwrap();
    let consolidate_value: Value = serde_json::from_str(&consolidate_response).unwrap();
    assert_eq!(
        consolidate_value["promoted_count"].as_u64().unwrap(),
        0,
        "consolidate should be idempotent after session_end"
    );

    assert_eq!(
        rt.queue.depth().await.unwrap(),
        0,
        "queue depth must be zero after the full lifecycle"
    );

    let notes = rt.handle.karta.get_all_notes().await.unwrap();
    let fact_notes: Vec<_> = notes
        .iter()
        .filter(|n| matches!(n.provenance, Provenance::Fact { .. }))
        .collect();
    assert!(
        !fact_notes.is_empty(),
        "session_end must trigger consolidation and promote facts"
    );
    assert!(rt.handle.karta.note_count().await.unwrap() > 0);

    rt.cleanup().await;
}

#[tokio::test]
async fn first_visit_flow_returns_empty_orientation_then_populates_store() {
    let rt = common::TestRuntime::new().await;
    let query = "first visit harness";

    let (status, body) = rt.post_orient(json!({"query": query})).await;
    assert_eq!(status, reqwest::StatusCode::OK);
    let first_note_ids: Vec<String> = body["note_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(
        first_note_ids.is_empty(),
        "first orient on an empty store must return no note ids"
    );

    let (status, _) = rt
        .post_capture(json!({
            "hook_event_name": "UserPromptSubmit",
            "session_id": "first-visit",
            "prompt": "first visit harness note"
        }))
        .await;
    assert_eq!(status, reqwest::StatusCode::ACCEPTED);
    rt.drain().await;

    let (status, body) = rt.post_orient(json!({"query": query})).await;
    assert_eq!(status, reqwest::StatusCode::OK);
    let second_note_ids: Vec<String> = body["note_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(
        !second_note_ids.is_empty(),
        "second orient must find the captured note"
    );
    assert_eq!(rt.handle.karta.note_count().await.unwrap(), 1);

    rt.cleanup().await;
}

// ---------------------------------------------------------------------------
// Live-binary cross-area flows
// ---------------------------------------------------------------------------

async fn wait_for_empty_queue(db_path: &Path) {
    for _ in 0..200 {
        let conn = Connection::open(db_path).expect("open db for drain check");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM capture_queue WHERE status IN ('queued', 'in_flight')",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if count == 0 {
            return;
        }
        sleep(Duration::from_millis(50)).await;
    }
    panic!("queue did not drain within timeout");
}

async fn run_backup(data_dir: &str, dest: &Path) -> std::process::Output {
    tokio::process::Command::new(common::bin_path())
        .args(["backup", "--dest", &dest.to_string_lossy()])
        .env("KARTA_STORE_DIR", data_dir)
        .output()
        .await
        .expect("run backup subcommand")
}

#[tokio::test]
async fn backup_restore_roundtrip_preserves_fetch_results() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().to_str().unwrap();
    let port = common::find_free_port();

    let mut child = common::spawn_serve(data_dir, port);
    common::wait_for_server(port).await;

    let client = reqwest::Client::new();
    let capture_url = format!("http://127.0.0.1:{port}/capture");
    let orient_url = format!("http://127.0.0.1:{port}/orient");
    let query = "backup roundtrip sentinel keyword xyzzy";

    // Seed a unique note.
    let body = json!({
        "hook_event_name": "UserPromptSubmit",
        "session_id": "cross-backup",
        "prompt": query
    });
    let resp = client.post(&capture_url).json(&body).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);
    wait_for_empty_queue(&tmp.path().join("karta.db")).await;

    let orient_before = client
        .post(&orient_url)
        .json(&json!({"query": query}))
        .send()
        .await
        .unwrap();
    assert_eq!(orient_before.status(), reqwest::StatusCode::OK);
    let before_body: Value = orient_before.json().await.unwrap();
    let before_ids: Vec<String> = before_body["note_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(
        !before_ids.is_empty(),
        "seeded note must be retrievable before backup"
    );

    // Keep the server busy with unrelated captures while the backup runs.
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let noise_task = tokio::spawn({
        let client = client.clone();
        let url = capture_url.clone();
        let mut rx = stop_rx.clone();
        async move {
            let mut ticker = interval(Duration::from_millis(50));
            let mut i = 0u64;
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        i += 1;
                        let noise = json!({
                            "hook_event_name": "UserPromptSubmit",
                            "session_id": format!("noise-{i}"),
                            "prompt": format!("background noise {i}")
                        });
                        let _ = client
                            .post(&url)
                            .json(&noise)
                            .timeout(Duration::from_secs(2))
                            .send()
                            .await;
                    }
                    _ = rx.changed() => break,
                }
            }
            i
        }
    });

    sleep(Duration::from_millis(100)).await;

    let backup_path = tmp.path().join("cross-backup.db");
    let backup_output = run_backup(data_dir, &backup_path).await;
    assert!(
        backup_output.status.success(),
        "backup failed: {}",
        String::from_utf8_lossy(&backup_output.stderr)
    );
    assert!(backup_path.exists(), "backup file was not created");

    stop_tx.send(true).ok();
    let _ = noise_task.await;

    let _ = child.kill();
    let _ = child.wait();

    let restore_output = tokio::process::Command::new(common::bin_path())
        .args(["restore", "--from", &backup_path.to_string_lossy()])
        .env("KARTA_STORE_DIR", data_dir)
        .output()
        .await
        .expect("run restore subcommand");
    assert!(
        restore_output.status.success(),
        "restore failed: {}",
        String::from_utf8_lossy(&restore_output.stderr)
    );

    let port2 = common::find_free_port();
    let mut child2 = common::spawn_serve(data_dir, port2);
    common::wait_for_server(port2).await;

    let orient_after = client
        .post(format!("http://127.0.0.1:{port2}/orient"))
        .json(&json!({"query": query}))
        .send()
        .await
        .unwrap();
    assert_eq!(orient_after.status(), reqwest::StatusCode::OK);
    let after_body: Value = orient_after.json().await.unwrap();
    let after_ids: Vec<String> = after_body["note_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let seed_id = before_ids.first().expect("seed note must be retrievable");
    assert!(
        after_ids.contains(seed_id),
        "seed note {seed_id} must survive backup/restore round-trip; got {after_ids:?}"
    );

    let _ = child2.kill();
    let _ = child2.wait();
}

#[tokio::test]
async fn sigterm_during_active_capture_replays_queued_rows() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().to_str().unwrap();
    let port = common::find_free_port();
    let db_path = tmp.path().join("karta.db");

    let mut child = common::spawn_serve(data_dir, port);
    common::wait_for_server(port).await;

    let client = reqwest::Client::new();
    let capture_url = format!("http://127.0.0.1:{port}/capture");

    let mut accepted = 0usize;
    for i in 0..20 {
        let body = json!({
            "hook_event_name": "UserPromptSubmit",
            "session_id": "cross-sigterm",
            "prompt": format!("sigterm note {i}")
        });
        let resp = client.post(&capture_url).json(&body).send().await.unwrap();
        if resp.status() == reqwest::StatusCode::ACCEPTED {
            accepted += 1;
        }
    }
    assert!(
        accepted > 0,
        "at least one capture should be accepted before SIGTERM"
    );

    common::send_sigterm(&child);
    let status = common::wait_for_exit(&mut child, Duration::from_secs(10))
        .await
        .unwrap();
    assert!(status.success(), "serve should exit cleanly after SIGTERM");

    let port2 = common::find_free_port();
    let mut child2 = common::spawn_serve(data_dir, port2);
    common::wait_for_server(port2).await;

    wait_for_empty_queue(&db_path).await;

    let conn = Connection::open(&db_path).unwrap();
    let remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM capture_queue WHERE status IN ('queued', 'in_flight')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        remaining, 0,
        "no rows should remain queued or in_flight after restart"
    );

    let done: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM capture_queue WHERE status = 'done'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        done as usize, accepted,
        "all accepted captures must be replayed to done"
    );

    let handle = karta_mcp::karta_handle::KartaHandle::open_mock_for_data_dir(data_dir)
        .await
        .unwrap();
    assert_eq!(
        handle.karta.note_count().await.unwrap(),
        accepted,
        "every replayed capture must produce a note"
    );

    let _ = child2.kill();
    let _ = child2.wait();
}
