mod mcp;
mod tools;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use serde_json::Value;
use tracing::{debug, error, info};

use karta_core::{Karta, config::KartaConfig};

#[derive(Parser)]
#[command(name = "karta-cli", about = "MCP server for Karta memory system")]
struct Args {
    /// Data directory for LanceDB + SQLite storage
    #[arg(long, default_value = "~/Projects/karta/.karta")]
    data_dir: String,
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs_home() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[tokio::main]
async fn main() {
    // Log to stderr so stdout stays clean for JSON-RPC
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("karta_cli=info".parse().unwrap())
                .add_directive("karta_core=info".parse().unwrap()),
        )
        .with_writer(io::stderr)
        .init();

    let args = Args::parse();
    let data_dir = expand_tilde(&args.data_dir);

    // Load .env from next to the binary or from data_dir's parent.
    // Don't rely on CWD — Claude Code may launch us from anywhere.
    let env_path = std::env::current_exe()
        .ok()
        .and_then(|p| {
            // binary is at target/release/karta-cli, project root is 3 levels up
            p.ancestors().nth(3).map(|a| a.join(".env"))
        })
        .filter(|p| p.exists())
        .unwrap_or_else(|| data_dir.parent().unwrap_or(&data_dir).join(".env"));
    let _ = dotenvy::from_path(&env_path);

    info!(data_dir = %data_dir.display(), "Starting Karta MCP server");

    // Build config and init Karta
    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir.to_string_lossy().to_string();

    let karta = match Karta::with_defaults(config).await {
        Ok(k) => Arc::new(k),
        Err(e) => {
            error!(error = %e, "Failed to initialize Karta");
            std::process::exit(1);
        }
    };

    info!("Karta initialized, entering MCP stdio loop");

    // Stdio JSON-RPC loop
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                error!(error = %e, "Failed to read stdin");
                break;
            }
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        debug!(line = %line, "Received JSON-RPC message");

        let request: mcp::JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "Invalid JSON-RPC");
                let err = mcp::JsonRpcError::new(
                    Value::Null,
                    -32700,
                    format!("Parse error: {e}"),
                );
                let _ = writeln!(stdout, "{}", serde_json::to_string(&err).unwrap());
                let _ = stdout.flush();
                continue;
            }
        };

        let id = request.id.clone().unwrap_or(Value::Null);

        // Notifications (no id) don't get responses
        if request.id.is_none() {
            debug!(method = %request.method, "Notification (no response needed)");
            continue;
        }

        let response = match request.method.as_str() {
            "initialize" => mcp::handle_initialize(id),

            "ping" => mcp::handle_ping(id),

            "tools/list" => {
                let resp = mcp::JsonRpcResponse::new(id, tools::tool_schemas());
                serde_json::to_value(resp).unwrap()
            }

            "tools/call" => {
                let tool_name = request.params["name"]
                    .as_str()
                    .unwrap_or("");
                let arguments = &request.params["arguments"];
                let result = tools::dispatch(&karta, tool_name, arguments).await;
                let resp = mcp::JsonRpcResponse::new(id, result);
                serde_json::to_value(resp).unwrap()
            }

            _ => {
                let err = mcp::JsonRpcError::new(
                    id,
                    -32601,
                    format!("Method not found: {}", request.method),
                );
                serde_json::to_value(err).unwrap()
            }
        };

        let response_str = serde_json::to_string(&response).unwrap();
        debug!(response = %response_str, "Sending response");
        let _ = writeln!(stdout, "{}", response_str);
        let _ = stdout.flush();
    }
}
