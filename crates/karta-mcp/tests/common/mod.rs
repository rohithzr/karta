//! Shared helpers for in-process karta-mcp integration tests.
#![allow(dead_code)]
//!
//! `TestRuntime` spins up the axum capture router, a mock-backed Karta, and a
//! queue worker on a random loopback port. Tests can POST to `/capture` and
//! `/orient` and then wait for the worker to drain the queue before inspecting
//! the store.
//!
//! Live-binary helpers are also provided so integration tests that spawn the
//! real `karta-mcp serve --mock` binary can find a free loopback port and
//! verify the child started successfully.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use karta_mcp::capture::router;
use karta_mcp::karta_handle::KartaHandle;
use karta_mcp::queue::{CaptureQueue, run_worker};
use reqwest::StatusCode;
use serde_json::Value;
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// A fully wired in-process test server.
pub struct TestRuntime {
    #[allow(dead_code)]
    dir: TempDir,
    pub handle: KartaHandle,
    pub queue: Arc<CaptureQueue>,
    pub base_url: String,
    cancel: CancellationToken,
    server: JoinHandle<()>,
    worker: JoinHandle<()>,
}

impl TestRuntime {
    /// Start a test runtime with `PreCompact` disabled.
    pub async fn new() -> Self {
        Self::with_precompact(false).await
    }

    /// Start a test runtime with an explicit `PreCompact` flag.
    pub async fn with_precompact(precompact: bool) -> Self {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let handle = KartaHandle::open_mock(data_dir).await.unwrap();
        let queue = Arc::new(CaptureQueue::new(data_dir).await.unwrap());

        let app = router(handle.karta.clone(), queue.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let cancel = CancellationToken::new();
        let server_cancel = cancel.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(server_cancel.cancelled_owned())
                .await
                .unwrap();
        });

        let worker_cancel = cancel.clone();
        let worker_queue = queue.clone();
        let worker_handle = handle.clone();
        let worker = tokio::spawn(async move {
            run_worker(worker_queue, worker_handle, worker_cancel, precompact).await;
        });

        let base_url = format!("http://{}", addr);
        Self {
            dir,
            handle,
            queue,
            base_url,
            cancel,
            server,
            worker,
        }
    }

    /// POST a JSON payload to `/capture` and return the status and body.
    pub async fn post_capture(&self, payload: Value) -> (StatusCode, Value) {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/capture", self.base_url))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = resp.status();
        let body = resp.json().await.unwrap_or(Value::Null);
        (status, body)
    }

    /// POST a JSON payload to `/orient` and return the status and body.
    pub async fn post_orient(&self, payload: Value) -> (StatusCode, Value) {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/orient", self.base_url))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = resp.status();
        let body = resp.json().await.unwrap_or(Value::Null);
        (status, body)
    }

    /// Wait until the queue contains no `queued` or `in_flight` rows.
    pub async fn drain(&self) {
        let start = tokio::time::Instant::now();
        while start.elapsed() < Duration::from_secs(10) {
            if self.queue.depth().await.unwrap() == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("queue did not drain within timeout");
    }

    /// Return the configured data directory for direct SQLite inspection.
    pub fn data_dir(&self) -> &str {
        self.dir.path().to_str().unwrap()
    }

    /// Cancel the worker and server tasks and wait for them to finish.
    pub async fn cleanup(self) {
        self.cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(5), self.worker).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), self.server).await;
    }
}

/// Return every row currently in the `capture_queue` table.
///
/// This is a synchronous convenience helper for tests that want to assert on
/// queue state without taking the async queue lock.
pub fn all_queue_rows(data_dir: &str) -> Vec<(i64, String, serde_json::Value, Option<String>)> {
    let path = Path::new(data_dir).join("karta.db");
    let conn = rusqlite::Connection::open(&path).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT id, event_type, payload, session_id
             FROM capture_queue
             ORDER BY id ASC",
        )
        .unwrap();
    let rows = stmt
        .query_map([], |r| {
            let payload_str: String = r.get(2)?;
            let payload: Value = serde_json::from_str(&payload_str).unwrap_or(Value::Null);
            Ok((r.get(0)?, r.get(1)?, payload, r.get(3)?))
        })
        .unwrap();
    rows.map(|r| r.unwrap()).collect()
}

// ---------------------------------------------------------------------------
// Live-binary helpers
// ---------------------------------------------------------------------------

/// Return the path to the `karta-mcp` binary under test.
///
/// Uses `CARGO_BIN_EXE_karta-mcp` when available (set by `cargo test`),
/// otherwise falls back to the workspace `target/debug/karta-mcp` path.
pub fn bin_path() -> PathBuf {
    std::env::var("CARGO_BIN_EXE_karta-mcp")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
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

/// Find a free loopback TCP port.
///
/// Binds a temporary socket to `127.0.0.1:0`, records the assigned port, and
/// drops the socket. This avoids collisions with other tests or stale
/// processes without relying on a fixed port range.
pub fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind free port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// Spawn `karta-mcp serve --mock` in the background.
///
/// Uses the pre-built binary directly so signals are delivered to the serve
/// process and not to a cargo wrapper. The child is verified to still be
/// alive shortly after spawn; if it exited early (e.g. due to a port
/// collision), the function panics with a clear message.
pub fn spawn_serve(data_dir: &str, port: u16) -> std::process::Child {
    let mut child = std::process::Command::new(bin_path())
        .args(["serve", "--mock"])
        .env("KARTA_STORE_DIR", data_dir)
        .env("KARTA_CAPTURE_PORT", port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn karta-mcp serve --mock");

    std::thread::sleep(Duration::from_millis(300));
    if let Some(status) = child.try_wait().expect("try_wait failed") {
        panic!(
            "karta-mcp serve exited early with status {status:?} on port {port}; \
             the port may be held by another process"
        );
    }
    child
}

/// Wait until the HTTP capture endpoint responds to `/orient`.
pub async fn wait_for_server(port: u16) {
    let client = reqwest::Client::new();
    for _ in 0..100 {
        if client
            .post(format!("http://127.0.0.1:{port}/orient"))
            .json(&serde_json::json!({"query":"ready"}))
            .send()
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server did not become ready in time");
}

/// Send SIGINT to a child process.
pub fn send_sigterm(child: &std::process::Child) {
    let pid = child.id() as i32;

    // Try using the kill command first (works on Unix systems)
    // Use SIGINT instead of SIGTERM as it's more reliably handled
    let status = std::process::Command::new("kill")
        .args(["-INT", &pid.to_string()])
        .status();

    // If kill command fails, the process might have already exited
    // This is not necessarily an error condition
    if let Err(e) = status {
        // On some systems/platforms, the kill command might not be available
        // or might fail for other reasons. We'll log it but not fail the test.
        tracing::debug!("Failed to send SIGINT via kill command: {}", e);
    }
}

/// Wait for the child to exit, killing it if it does not exit within the
/// supplied timeout.
pub async fn wait_for_exit(
    child: &mut std::process::Child,
    timeout: Duration,
) -> std::io::Result<std::process::ExitStatus> {
    let start = tokio::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if start.elapsed() >= timeout {
            // Try to kill the process more forcefully
            let _ = child.kill();
            // Give it a brief moment to terminate
            tokio::time::sleep(Duration::from_millis(100)).await;
            return child.wait();
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
