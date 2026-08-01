//! Integration smoke test for the stdio MCP server.
//!
//! Exercises the JSON-RPC handshake, `tools/list`, and one successful tool
//! call through a real `karta-mcp serve --mock` process. This complements the
//! unit tests in `src/tools.rs` by verifying the rmcp wiring end-to-end.

mod common;

use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

async fn spawn_serve(
    temp_dir: &TempDir,
    args: &[&str],
) -> (
    u16,
    Child,
    ChildStdin,
    BufReader<ChildStdout>,
    tokio::task::JoinHandle<()>,
) {
    let port = common::find_free_port();
    let mut child = Command::new(common::bin_path())
        .args(args)
        .env("KARTA_STORE_DIR", temp_dir.path().to_str().unwrap())
        .env("KARTA_CAPTURE_PORT", port.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn karta-mcp serve --mock");

    let stdin = child.stdin.take().expect("stdin pipe");
    let stdout = child.stdout.take().expect("stdout pipe");
    let stderr = child.stderr.take().expect("stderr pipe");
    let reader = BufReader::new(stdout);

    let stderr_handle = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            eprintln!("[serve stderr] {line}");
        }
    });

    // Give the server time to open the store and start listening.
    tokio::time::sleep(Duration::from_millis(300)).await;

    if let Some(status) = child.try_wait().expect("try_wait failed") {
        panic!(
            "karta-mcp serve exited early with status {status:?}; \
             the HTTP capture port may be in use"
        );
    }

    (port, child, stdin, reader, stderr_handle)
}

async fn read_message(reader: &mut BufReader<ChildStdout>) -> Option<Value> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).await.ok()? == 0 {
            return None;
        }
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            break;
        }
    }
    serde_json::from_str(line.trim()).ok()
}

async fn write_request(stdin: &mut ChildStdin, id: i64, method: &str, params: Value) {
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    });
    let msg = format!("{}\n", serde_json::to_string(&body).unwrap());
    stdin.write_all(msg.as_bytes()).await.unwrap();
    stdin.flush().await.unwrap();
}

async fn read_message_timeout(reader: &mut BufReader<ChildStdout>, label: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(10), read_message(reader))
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {label}"))
        .unwrap_or_else(|| panic!("server closed stdout before {label}"))
}

#[tokio::test]
async fn server_initializes_and_lists_tools() {
    let temp_dir = TempDir::new().unwrap();
    let (_port, mut child, mut stdin, mut reader, stderr_handle) =
        spawn_serve(&temp_dir, &["serve", "--mock"]).await;

    // Initialize handshake.
    write_request(
        &mut stdin,
        1,
        "initialize",
        json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0.0.1" },
        }),
    )
    .await;

    let init = read_message_timeout(&mut reader, "initialize response").await;
    assert!(
        init.get("result").is_some(),
        "initialize should return a result: {init}"
    );
    assert!(
        init.get("error").is_none(),
        "initialize should not return an error: {init}"
    );

    // Initialized notification.
    let notify = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    let msg = format!("{}\n", serde_json::to_string(&notify).unwrap());
    stdin.write_all(msg.as_bytes()).await.unwrap();
    stdin.flush().await.unwrap();

    // tools/list request.
    write_request(&mut stdin, 2, "tools/list", json!({})).await;
    let tools = read_message_timeout(&mut reader, "tools/list response").await;
    let result = tools
        .get("result")
        .expect("tools/list should have a result");
    let names: Vec<String> = result["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();

    let expected = [
        "karta_add_note",
        "karta_fetch_memories",
        "karta_run_dreaming",
        "karta_session_start",
        "karta_session_end",
        "karta_consolidate",
        "karta_status",
    ];
    for name in &expected {
        assert!(names.contains(&name.to_string()), "missing tool {name}");
    }
    assert_eq!(names.len(), 7, "expected exactly 7 tools, got {names:?}");

    let _ = child.kill().await;
    let _ = stderr_handle.await;
}

#[tokio::test]
async fn rejects_tool_call_with_unknown_field() {
    let temp_dir = TempDir::new().unwrap();
    let (_port, mut child, mut stdin, mut reader, stderr_handle) =
        spawn_serve(&temp_dir, &["serve", "--mock"]).await;

    // Initialize handshake.
    write_request(
        &mut stdin,
        1,
        "initialize",
        json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0.0.1" },
        }),
    )
    .await;

    let init = read_message_timeout(&mut reader, "initialize response").await;
    assert!(
        init.get("result").is_some(),
        "initialize should return a result: {init}"
    );
    assert!(
        init.get("error").is_none(),
        "initialize should not return an error: {init}"
    );

    // Initialized notification.
    let notify = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    let msg = format!("{}\n", serde_json::to_string(&notify).unwrap());
    stdin.write_all(msg.as_bytes()).await.unwrap();
    stdin.flush().await.unwrap();

    // Call karta_add_note with an unsupported extra parameter.
    write_request(
        &mut stdin,
        2,
        "tools/call",
        json!({
            "name": "karta_add_note",
            "arguments": {
                "content": "hello world",
                "extra_field": "not supported"
            }
        }),
    )
    .await;

    let response = read_message_timeout(&mut reader, "tools/call response").await;
    let result = response
        .get("result")
        .expect("tools/call should return a result object");
    assert!(
        result["isError"].as_bool().unwrap_or(false),
        "expected tool result to report an error for unknown field, got: {response}"
    );
    let content_text = result["content"][0]["text"]
        .as_str()
        .expect("error content should contain text");
    assert!(
        content_text.contains("unknown field"),
        "expected unknown field deserialization error, got: {content_text}"
    );

    let _ = child.kill().await;
    let _ = stderr_handle.await;
}

#[tokio::test]
async fn default_serve_with_global_mock_flag() {
    let temp_dir = TempDir::new().unwrap();
    // Spawn `karta-mcp --mock` with no explicit `serve` subcommand. The
    // global `--mock` flag must be forwarded to the default serve command.
    let (port, mut child, mut stdin, mut reader, stderr_handle) =
        spawn_serve(&temp_dir, &["--mock"]).await;

    // Initialize handshake.
    write_request(
        &mut stdin,
        1,
        "initialize",
        json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0.0.1" },
        }),
    )
    .await;

    let init = read_message_timeout(&mut reader, "initialize response").await;
    assert!(
        init.get("result").is_some(),
        "initialize should return a result: {init}"
    );
    assert!(
        init.get("error").is_none(),
        "initialize should not return an error: {init}"
    );

    // Initialized notification.
    let notify = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    let msg = format!("{}\n", serde_json::to_string(&notify).unwrap());
    stdin.write_all(msg.as_bytes()).await.unwrap();
    stdin.flush().await.unwrap();

    // Also verify the HTTP capture endpoint is up, which proves the serve
    // command was started rather than clap exiting with an error.
    let client = reqwest::Client::new();
    let orient_url = format!("http://127.0.0.1:{port}/orient");
    let response = client
        .post(&orient_url)
        .header("Content-Type", "application/json")
        .json(&json!({"hook_event_name": "SessionStart", "query": "healthcheck"}))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("orient request should reach the server");
    assert_eq!(
        response.status(),
        200,
        "orient should return 200 when serve is running with --mock"
    );

    let _ = child.kill().await;
    let _ = stderr_handle.await;
}

#[tokio::test]
async fn mcp_reachability_via_tool_list_and_call() {
    let temp_dir = TempDir::new().unwrap();
    let (_port, mut child, mut stdin, mut reader, stderr_handle) =
        spawn_serve(&temp_dir, &["serve", "--mock"]).await;

    // Initialize handshake.
    write_request(
        &mut stdin,
        1,
        "initialize",
        json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0.0.1" },
        }),
    )
    .await;

    let init = read_message_timeout(&mut reader, "initialize response").await;
    assert!(
        init.get("result").is_some(),
        "initialize should return a result: {init}"
    );

    let notify = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    let msg = format!("{}\n", serde_json::to_string(&notify).unwrap());
    stdin.write_all(msg.as_bytes()).await.unwrap();
    stdin.flush().await.unwrap();

    // tools/list request.
    write_request(&mut stdin, 2, "tools/list", json!({})).await;
    let tools = read_message_timeout(&mut reader, "tools/list response").await;
    let result = tools
        .get("result")
        .expect("tools/list should have a result");
    let names: Vec<String> = result["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"karta_status".to_string()));
    assert_eq!(names.len(), 7, "expected exactly 7 tools");

    // tools/call request for karta_status.
    write_request(
        &mut stdin,
        3,
        "tools/call",
        json!({
            "name": "karta_status",
            "arguments": {}
        }),
    )
    .await;
    let response = read_message_timeout(&mut reader, "tools/call response").await;
    let result = response
        .get("result")
        .expect("tools/call should return a result object");
    let content_text = result["content"][0]["text"]
        .as_str()
        .expect("tool result content should contain text");
    let payload: Value =
        serde_json::from_str(content_text).expect("karta_status content should be parseable JSON");
    assert!(payload["note_count"].is_number());
    assert!(payload["store_dir"].is_string());
    assert!(payload["embedding_model"].is_string());
    assert!(payload["capture_port"].is_number());
    assert!(payload["queue_depth"].is_number());

    let _ = child.kill().await;
    let _ = stderr_handle.await;
}
