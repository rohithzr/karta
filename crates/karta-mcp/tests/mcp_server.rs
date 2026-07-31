//! Integration smoke test for the stdio MCP server.
//!
//! Exercises the JSON-RPC handshake, `tools/list`, and one successful tool
//! call through a real `karta-mcp serve --mock` process. This complements the
//! unit tests in `src/tools.rs` by verifying the rmcp wiring end-to-end.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

/// Port range for spawned `serve` processes in these tests.
///
/// Each test gets a distinct loopback port so the tests can run in parallel
/// without colliding on the HTTP capture endpoint. The range starts well above
/// the default `3137` and the `31500` range used by the queue durability
/// integration tests.
static NEXT_PORT: AtomicU16 = AtomicU16::new(31700);

fn allocate_test_port() -> u16 {
    NEXT_PORT.fetch_add(1, Ordering::SeqCst)
}

fn bin_path() -> PathBuf {
    std::env::var("CARGO_BIN_EXE_karta-mcp")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Fallback: from the crate manifest go up to the workspace root,
            // then into the workspace target directory. This covers `cargo test`.
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| {
                std::env::current_dir()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            });
            PathBuf::from(manifest_dir)
                .join("..")
                .join("..")
                .join("target")
                .join("debug")
                .join("karta-mcp")
        })
}

async fn spawn_serve(
    temp_dir: &TempDir,
    port: u16,
    args: &[&str],
) -> (
    Child,
    ChildStdin,
    BufReader<ChildStdout>,
    tokio::task::JoinHandle<()>,
) {
    let mut child = Command::new(bin_path())
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
    (child, stdin, reader, stderr_handle)
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
    let port = allocate_test_port();
    let (mut child, mut stdin, mut reader, stderr_handle) =
        spawn_serve(&temp_dir, port, &["serve", "--mock"]).await;

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
    let port = allocate_test_port();
    let (mut child, mut stdin, mut reader, stderr_handle) =
        spawn_serve(&temp_dir, port, &["serve", "--mock"]).await;

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
    let port = allocate_test_port();
    // Spawn `karta-mcp --mock` with no explicit `serve` subcommand. The
    // global `--mock` flag must be forwarded to the default serve command.
    let (mut child, mut stdin, mut reader, stderr_handle) =
        spawn_serve(&temp_dir, port, &["--mock"]).await;

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
