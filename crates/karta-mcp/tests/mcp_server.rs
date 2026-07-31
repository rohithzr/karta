//! Integration smoke test for the stdio MCP server.
//!
//! Exercises the JSON-RPC handshake, `tools/list`, and one successful tool
//! call through a real `karta-mcp serve --mock` process. This complements the
//! unit tests in `src/tools.rs` by verifying the rmcp wiring end-to-end.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

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
) -> (
    Child,
    ChildStdin,
    BufReader<ChildStdout>,
    tokio::task::JoinHandle<()>,
) {
    let mut child = Command::new(bin_path())
        .arg("serve")
        .arg("--mock")
        .env("KARTA_STORE_DIR", temp_dir.path().to_str().unwrap())
        .env("KARTA_CAPTURE_PORT", "3137")
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
    let (mut child, mut stdin, mut reader, stderr_handle) = spawn_serve(&temp_dir).await;

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
        !init.get("error").is_some(),
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
