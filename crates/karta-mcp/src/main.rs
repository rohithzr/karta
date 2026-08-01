//! `karta-mcp` binary entry point.
//!
//! Subcommands:
//! - `serve` (default): start the stdio MCP server + HTTP capture endpoint.
//! - `status`: print current store status.
//! - `backup --dest <path>`: online snapshot of the store.
//! - `export --dest <dir>`: markdown export of notes.
//! - `restore --from <path>`: restore store from a backup.
//!
//! stdout is reserved for MCP JSON-RPC traffic from the `serve` subcommand.
//! All logging goes to stderr via `tracing`.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use karta_mcp::{capture, config, karta_handle::KartaHandle, ops, queue, server};

use config::Config;

#[derive(Parser)]
#[command(name = "karta-mcp", version, about = "Karta MCP server + auto-capture")]
struct Cli {
    /// Use mock LLM and SQLite stores instead of a real LLM endpoint.
    ///
    /// This is a global flag so `karta-mcp --mock` works without an explicit
    /// `serve` subcommand, which is the default command.
    #[arg(long, global = true)]
    mock: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the MCP server and HTTP capture endpoint (default).
    #[command(name = "serve")]
    Serve,

    /// Print the current store status.
    #[command(name = "status")]
    Status,

    /// Create an online backup of the store.
    #[command(name = "backup")]
    Backup {
        #[arg(long, required = true)]
        dest: PathBuf,
    },

    /// Export notes to markdown.
    #[command(name = "export")]
    Export {
        #[arg(long, required = true)]
        dest: PathBuf,
    },

    /// Restore the store from a backup.
    #[command(name = "restore")]
    Restore {
        #[arg(long, required = true)]
        from: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Serve => serve(cli.mock).await,
        Commands::Status => status().await,
        Commands::Backup { dest } => backup(&dest).await,
        Commands::Export { dest } => export(&dest).await,
        Commands::Restore { from } => restore(&from).await,
    }
}

/// Initialise `tracing` so that all logs are written to stderr.
/// stdout must remain clean for MCP JSON-RPC traffic.
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();
}

async fn serve(mock: bool) -> Result<()> {
    let config = Config::from_env()?;
    let data_dir = config.store_dir().to_string();

    tracing::info!(
        store_dir = %config.store_dir(),
        capture_port = config.capture_port,
        precompact = config.precompact,
        "starting karta-mcp serve"
    );

    let handle = if mock {
        tracing::info!("using mock LLM provider");
        KartaHandle::open_mock(&data_dir).await?
    } else {
        let karta = Arc::new(karta_core::Karta::with_defaults(config.core.clone()).await?);
        let (vector_store, graph_store, shared_conn) =
            KartaHandle::open_stores_for_data_dir(&data_dir).await?;
        KartaHandle::new(karta, vector_store, graph_store, shared_conn)
    };

    let queue = Arc::new(queue::CaptureQueue::new(&data_dir).await?);
    let replayed = queue.replay_incomplete().await?;
    if replayed > 0 {
        tracing::info!(replayed_rows = replayed, "replayed incomplete queue rows");
    }

    let cancel = CancellationToken::new();
    let mut join_set = tokio::task::JoinSet::new();

    // stdio MCP server task. It is intentionally spawned outside the join set:
    // MCP clients keep the stdio transport open, and on SIGTERM we want to drain
    // the queue and exit without waiting for the transport to close gracefully.
    let server = server::KartaMcpServer::new(handle.clone(), queue.clone(), &config);
    let server_cancel = cancel.clone();
    tokio::spawn(async move {
        if let Err(e) = server::run_stdio_server(server, server_cancel).await {
            tracing::error!(error = %e, "MCP server exited early");
        }
    });

    // Queue worker task.
    let worker_queue = queue.clone();
    let worker_handle = handle.clone();
    let worker_cancel = cancel.clone();
    join_set.spawn(async move {
        queue::run_worker(
            worker_queue,
            worker_handle,
            worker_cancel,
            config.precompact,
        )
        .await;
        Ok::<(), anyhow::Error>(())
    });

    // HTTP capture endpoint task.
    let capture_router = capture::router(handle.karta.clone(), queue.clone());
    let socket = tokio::net::TcpSocket::new_v4()
        .with_context(|| "failed to create TCP socket for HTTP capture endpoint")?;
    socket
        .set_reuseaddr(true)
        .with_context(|| "failed to set SO_REUSEADDR on HTTP capture socket")?;
    let addr: std::net::SocketAddr = format!("127.0.0.1:{}", config.capture_port)
        .parse()
        .expect("127.0.0.1 with a valid port is a valid socket address");
    socket.bind(addr).with_context(|| {
        format!(
            "failed to bind HTTP capture endpoint to 127.0.0.1:{}",
            config.capture_port
        )
    })?;
    let listener = socket
        .listen(128)
        .with_context(|| "failed to listen on HTTP capture socket")?;
    let capture_addr = listener.local_addr()?;
    let capture_cancel = cancel.clone();
    join_set.spawn(async move {
        tracing::info!(addr = %capture_addr, "HTTP capture endpoint listening");
        let shutdown = capture_cancel.cancelled_owned();
        if let Err(e) = axum::serve(listener, capture_router)
            .with_graceful_shutdown(shutdown)
            .await
        {
            tracing::error!(error = %e, "HTTP capture server exited early");
        }
        Ok::<(), anyhow::Error>(())
    });

    // Shutdown signal handler.
    join_set.spawn(async move {
        wait_for_shutdown(cancel).await;
        Ok::<(), anyhow::Error>(())
    });

    // Wait for all tasks to finish. The worker exits only after it has drained
    // any remaining queue rows, so a SIGTERM/SIGINT first cancels the token and
    // then waits for the graceful drain to complete.
    while let Some(res) = join_set.join_next().await {
        res??;
    }

    tracing::info!("karta-mcp serve exiting");
    Ok(())
}

/// Wait for a shutdown signal and cancel the token when it fires.
///
/// On Unix, `signal-hook` is used instead of `tokio::signal` because the MCP
/// server task holds a background stdin reader; on some platforms (macOS)
/// `tokio::signal` leaves the default signal action enabled while async
/// handlers are pending, which causes the process to be killed with the
/// signal code before it can exit cleanly. `signal-hook` registers a real
/// signal handler that prevents the default action.
///
/// On non-Unix platforms, `tokio::signal::ctrl_c` is used as a portable
/// fallback. The function is `async` and awaits the signal, so it does not
/// cancel the token at startup.
#[cfg(unix)]
async fn wait_for_shutdown(cancel: CancellationToken) {
    use signal_hook::consts::{SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;

    let (tx, rx) = tokio::sync::oneshot::channel::<i32>();
    std::thread::spawn(move || {
        let mut signals = match Signals::new([SIGTERM, SIGINT]) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "failed to register signal handlers");
                return;
            }
        };
        if let Some(sig) = signals.wait().next() {
            let _ = tx.send(sig);
        }
    });

    match rx.await {
        Ok(sig) => tracing::info!(signal = sig, "received shutdown signal"),
        Err(_) => tracing::warn!("signal handler thread dropped without sending signal"),
    }
    cancel.cancel();
}

#[cfg(not(unix))]
async fn wait_for_shutdown(cancel: CancellationToken) {
    match tokio::signal::ctrl_c().await {
        Ok(()) => tracing::info!("received Ctrl-C signal"),
        Err(e) => tracing::error!(error = %e, "failed to listen for Ctrl-C signal"),
    }
    cancel.cancel();
}

async fn status() -> Result<()> {
    let config = Config::from_env()?;
    let output = ops::status(&config).await?;
    print!("{output}");
    Ok(())
}

async fn backup(dest: &Path) -> Result<()> {
    let config = Config::from_env()?;
    let source = Path::new(config.store_dir()).join("karta.db");
    ops::backup(&source, dest).await?;
    println!("backup: {} -> {}", source.display(), dest.display());
    Ok(())
}

async fn export(dest: &Path) -> Result<()> {
    let config = Config::from_env()?;
    let handle = KartaHandle::open_mock_for_data_dir(config.store_dir()).await?;
    let count = ops::export(&handle, dest).await?;
    println!("export: {count} notes -> {}", dest.display());
    Ok(())
}

async fn restore(from: &Path) -> Result<()> {
    let config = Config::from_env()?;
    let data_dir = Path::new(config.store_dir());
    ops::restore(from, data_dir).await?;
    println!(
        "restore: {} -> {}/karta.db",
        from.display(),
        data_dir.display()
    );
    println!("restart karta-mcp serve to use the restored store");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_is_default_subcommand() {
        // `karta-mcp` with no subcommand leaves `command` as `None`;
        // `main` maps that to `Commands::Serve`.
        let cli = Cli::parse_from(["karta-mcp"]);
        assert!(cli.command.is_none());
    }

    #[test]
    fn mock_flag_is_available_on_serve_subcommand() {
        let cli = Cli::parse_from(["karta-mcp", "serve", "--mock"]);
        assert!(matches!(cli.command, Some(Commands::Serve)));
        assert!(cli.mock);
    }

    #[test]
    fn global_mock_flag_works_without_subcommand() {
        let cli = Cli::parse_from(["karta-mcp", "--mock"]);
        assert!(cli.command.is_none());
        assert!(cli.mock);
    }
}
