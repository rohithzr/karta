//! In-process integration tests for the HTTP capture endpoints.
//!
//! These tests construct the axum router directly with a mock `Karta` and
//! `CaptureQueue`, bind it to a random loopback port, and drive it with
//! `reqwest`. This avoids needing a live LLM endpoint or the full `serve`
//! binary lifecycle.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use karta_core::ClockContext;
use karta_core::Karta;
use karta_core::config::KartaConfig;
use karta_core::llm::LlmProvider;
use karta_core::llm::MockLlmProvider;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_mcp::capture::router;
use karta_mcp::queue::CaptureQueue;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

const EMBEDDING_DIM: usize = 1536;

struct TestServer {
    base_url: String,
    #[allow(dead_code)]
    handle: tokio::task::JoinHandle<()>,
}

async fn start_test_server() -> (TempDir, Arc<Karta>, Arc<CaptureQueue>, TestServer) {
    let dir = TempDir::new().unwrap();
    let data_dir = dir.path().to_str().unwrap();

    let vector_store = SqliteVectorStore::new(data_dir, EMBEDDING_DIM)
        .await
        .unwrap();
    let shared_conn = vector_store.connection();
    let vector_store: Arc<dyn VectorStore> = Arc::new(vector_store);
    let graph_store: Arc<dyn GraphStore> = Arc::new(SqliteGraphStore::with_connection(shared_conn));
    let llm: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new());
    let config = KartaConfig::default();
    let karta = Arc::new(
        Karta::new(vector_store, graph_store, llm, config)
            .await
            .unwrap(),
    );
    let queue = Arc::new(CaptureQueue::new(data_dir).await.unwrap());

    let app = router(karta.clone(), queue.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cancel = CancellationToken::new();
    let handle = tokio::spawn({
        let cancel = cancel.clone();
        async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(cancel.cancelled_owned())
                .await
                .unwrap();
        }
    });

    let base_url = format!("http://{}", addr);
    (dir, karta, queue, TestServer { base_url, handle })
}

#[derive(Debug)]
struct RowSnapshot {
    id: i64,
    event_type: String,
    payload: Value,
    session_id: Option<String>,
}

fn all_queue_rows(data_dir: &str) -> Vec<RowSnapshot> {
    let conn = rusqlite::Connection::open(Path::new(data_dir).join("karta.db")).unwrap();
    let mut stmt = conn
        .prepare("SELECT id, event_type, payload, session_id FROM capture_queue ORDER BY id ASC")
        .unwrap();
    let rows = stmt
        .query_map([], |r| {
            let payload_str: String = r.get(2)?;
            let payload: Value = serde_json::from_str(&payload_str).unwrap_or(Value::Null);
            Ok(RowSnapshot {
                id: r.get(0)?,
                event_type: r.get(1)?,
                payload,
                session_id: r.get(3)?,
            })
        })
        .unwrap();
    rows.map(|r| r.unwrap()).collect()
}

#[tokio::test]
async fn capture_user_prompt_returns_202_and_inserts_row() {
    let (dir, _karta, queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let body = json!({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "hello karta",
        "session_id": "s1"
    });
    let resp = client
        .post(format!("{}/capture", server.base_url))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 202);

    let resp_body: Value = resp.json().await.unwrap();
    assert_eq!(resp_body["status"], "queued");

    assert_eq!(queue.depth().await.unwrap(), 1);
    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].event_type, "user_prompt");
    assert_eq!(rows[0].payload["prompt"], "hello karta");
    assert_eq!(rows[0].session_id, Some("s1".to_string()));
}

#[tokio::test]
async fn capture_event_mapping_for_all_wired_events() {
    let (dir, _karta, _queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let cases = [
        ("UserPromptSubmit", "user_prompt"),
        ("PostToolUse", "observation"),
        ("Stop", "turn_summary"),
        ("SubagentStop", "subagent_result"),
        ("SessionEnd", "session_end"),
        ("PreCompact", "pre_compact"),
    ];

    for (hook, _) in &cases {
        let body = json!({
            "hook_event_name": hook,
            "session_id": "s1",
        });
        let resp = client
            .post(format!("{}/capture", server.base_url))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 202, "unexpected status for {hook}");
    }

    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert_eq!(rows.len(), cases.len());
    for (i, (_, expected)) in cases.iter().enumerate() {
        assert_eq!(rows[i].event_type, *expected);
    }
}

#[tokio::test]
async fn capture_session_start_returns_400_and_no_row() {
    let (dir, _karta, queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/capture", server.base_url))
        .json(&json!({"hook_event_name": "SessionStart"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    assert_eq!(queue.depth().await.unwrap(), 0);
    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert!(rows.is_empty());
}

#[tokio::test]
async fn capture_unknown_event_returns_400_and_no_row() {
    let (dir, _karta, queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/capture", server.base_url))
        .json(&json!({"hook_event_name": "Notification"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    assert_eq!(queue.depth().await.unwrap(), 0);
    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert!(rows.is_empty());
}

#[tokio::test]
async fn capture_invalid_json_returns_400() {
    let (dir, _karta, queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/capture", server.base_url))
        .body("not json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    assert_eq!(queue.depth().await.unwrap(), 0);
    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert!(rows.is_empty());
}

#[tokio::test]
async fn capture_missing_session_id_still_succeeds() {
    let (dir, _karta, _queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/capture", server.base_url))
        .json(&json!({
            "hook_event_name": "UserPromptSubmit",
            "prompt": "hi"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 202);
    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].event_type, "user_prompt");
    assert_eq!(rows[0].session_id, None);
}

#[tokio::test]
async fn capture_empty_payload_fields_succeed() {
    let (dir, _karta, _queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/capture", server.base_url))
        .json(&json!({
            "hook_event_name": "UserPromptSubmit",
            "prompt": "",
            "session_id": ""
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 202);
    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert_eq!(rows[0].payload["prompt"], "");
    assert_eq!(rows[0].session_id, Some("".to_string()));
}

#[tokio::test]
async fn capture_payload_preserves_nested_json() {
    let (dir, _karta, _queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let payload = json!({
        "hook_event_name": "PostToolUse",
        "session_id": "s1",
        "tool_input": {
            "path": "src/main.rs",
            "content": "fn main() {}"
        },
        "tool_output": {
            "success": true,
            "lines": [1, 2, 3]
        }
    });
    let resp = client
        .post(format!("{}/capture", server.base_url))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 202);
    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert_eq!(rows[0].payload, payload);
}

#[tokio::test]
async fn capture_concurrent_posts_all_persist() {
    let (dir, _karta, queue, server) = start_test_server().await;
    let client = reqwest::Client::new();
    let mut handles = Vec::new();
    for i in 0..50 {
        let client = client.clone();
        let url = format!("{}/capture", server.base_url);
        handles.push(tokio::spawn(async move {
            let body = json!({
                "hook_event_name": "UserPromptSubmit",
                "prompt": format!("concurrent-{i}"),
                "session_id": "concurrent"
            });
            client.post(url).json(&body).send().await.unwrap()
        }));
    }

    let mut statuses = Vec::new();
    for h in handles {
        statuses.push(h.await.unwrap().status());
    }
    assert!(
        statuses.iter().all(|s| s == &202),
        "all concurrent posts must return 202"
    );

    let rows = all_queue_rows(dir.path().to_str().unwrap());
    assert_eq!(rows.len(), 50, "expected 50 persisted rows");

    let unique_ids: HashSet<i64> = rows.iter().map(|r| r.id).collect();
    assert_eq!(unique_ids.len(), 50, "every row must have a unique id");

    let prompts: HashSet<String> = rows
        .iter()
        .map(|r| r.payload["prompt"].as_str().unwrap().to_string())
        .collect();
    for i in 0..50 {
        assert!(prompts.contains(&format!("concurrent-{i}")));
    }

    assert_eq!(queue.depth().await.unwrap(), 50);
}

#[tokio::test]
async fn orient_returns_200_with_context_and_note_ids() {
    let (_dir, karta, _queue, server) = start_test_server().await;
    let note = karta
        .add_note_with_clock(
            "the quick brown fox jumps over the lazy dog",
            None,
            None,
            ClockContext::now(),
        )
        .await
        .unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/orient", server.base_url))
        .json(&json!({"query": "fox"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["context"].is_string());
    let note_ids: Vec<String> = body["note_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(
        note_ids.contains(&note.id),
        "orient should return the seeded note"
    );
}

#[tokio::test]
async fn orient_query_derivation_from_droid_body() {
    let (_dir, karta, _queue, server) = start_test_server().await;
    let note = karta
        .add_note_with_clock("karta memory engine", None, None, ClockContext::now())
        .await
        .unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/orient", server.base_url))
        .json(&json!({
            "agent": "droid",
            "project": "karta",
            "cwd": "/workspace/karta"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let note_ids: Vec<String> = body["note_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(note_ids.contains(&note.id));
}

#[tokio::test]
async fn orient_query_derivation_from_claude_body() {
    let (_dir, karta, _queue, server) = start_test_server().await;
    let note = karta
        .add_note_with_clock("karta memory engine", None, None, ClockContext::now())
        .await
        .unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/orient", server.base_url))
        .json(&json!({
            "hook_event_name": "SessionStart",
            "agent": "claude-code",
            "project": "karta",
            "cwd": "/workspace/karta",
            "session_id": "abc"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let note_ids: Vec<String> = body["note_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(note_ids.contains(&note.id));
}

#[tokio::test]
async fn server_binds_to_loopback_only() {
    let (_dir, _karta, _queue, server) = start_test_server().await;
    let url = server.base_url.parse::<reqwest::Url>().unwrap();
    assert_eq!(url.host_str(), Some("127.0.0.1"));
}
