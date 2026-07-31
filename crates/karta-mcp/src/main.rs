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

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use karta_mcp::{capture, config, karta_handle::KartaHandle, queue, server};

use config::Config;

#[derive(Parser)]
#[command(name = "karta-mcp", version, about = "Karta MCP server + auto-capture")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the MCP server and HTTP capture endpoint (default).
    #[command(name = "serve")]
    Serve(ServeArgs),

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

#[derive(Parser)]
struct ServeArgs {
    /// Use mock LLM and SQLite stores instead of a real LLM endpoint.
    #[arg(long)]
    mock: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    match cli
        .command
        .unwrap_or(Commands::Serve(ServeArgs { mock: false }))
    {
        Commands::Serve(args) => serve(args).await,
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

async fn serve(args: ServeArgs) -> Result<()> {
    let config = Config::from_env()?;
    let data_dir = config.store_dir().to_string();

    tracing::info!(
        store_dir = %config.store_dir(),
        capture_port = config.capture_port,
        precompact = config.precompact,
        "starting karta-mcp serve"
    );

    let handle = if args.mock {
        tracing::info!("using mock LLM provider");
        KartaHandle::open_mock(&data_dir).await?
    } else {
        let karta = Arc::new(karta_core::Karta::with_defaults(config.core.clone()).await?);
        let (vector_store, graph_store) = KartaHandle::open_stores_for_data_dir(&data_dir).await?;
        KartaHandle::new(karta, vector_store, graph_store)
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
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", config.capture_port))
        .await
        .with_context(|| {
            format!(
                "failed to bind HTTP capture endpoint to 127.0.0.1:{}",
                config.capture_port
            )
        })?;
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

/// Wait for SIGTERM or SIGINT and cancel the token when either fires.
///
/// Uses `signal-hook` instead of `tokio::signal` because the MCP server task
/// holds a background stdin reader; on some platforms (macOS) `tokio::signal`
/// leaves the default signal action enabled while async handlers are pending,
/// which causes the process to be killed with the signal code before it can
/// exit cleanly. `signal-hook` registers a real signal handler that prevents
/// the default action.
async fn wait_for_shutdown(cancel: CancellationToken) {
    let (tx, rx) = tokio::sync::oneshot::channel::<i32>();
    std::thread::spawn(move || {
        let mut signals = match signal_hook::iterator::Signals::new([
            signal_hook::consts::SIGTERM,
            signal_hook::consts::SIGINT,
        ]) {
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

async fn status() -> Result<()> {
    let config = Config::from_env()?;

    // Open the store with the mock provider so `status` works without a live
    // LLM endpoint. The note count and store directory are real; the
    // embedding_model label is taken from the environment when available.
    let karta = open_mock_karta(&config).await?;
    let note_count = karta.note_count().await?;
    let queue = queue::CaptureQueue::new(config.store_dir()).await?;
    let queue_depth = queue.depth().await?;
    let embedding_model = std::env::var("KARTA_EMBEDDING_MODEL")
        .unwrap_or_else(|_| config.core.llm.default.model.clone());

    println!("note_count: {note_count}");
    println!("store_dir: {}", config.store_dir());
    println!("embedding_model: {embedding_model}");
    println!("capture_port: {}", config.capture_port);
    println!("queue_depth: {queue_depth}");

    Ok(())
}

async fn backup(dest: &PathBuf) -> Result<()> {
    bail!("backup not yet implemented (destination: {dest:?})")
}

async fn export(dest: &PathBuf) -> Result<()> {
    bail!("export not yet implemented (destination: {dest:?})")
}

async fn restore(from: &PathBuf) -> Result<()> {
    bail!("restore not yet implemented (source: {from:?})")
}

/// Open a `Karta` instance using the mock LLM and SQLite stores.
///
/// This is used by operator subcommands (`status`, `backup`, etc.) that need to
/// inspect the store without requiring a live LLM endpoint.
async fn open_mock_karta(config: &Config) -> Result<karta_core::Karta> {
    use karta_core::llm::LlmProvider;
    use karta_core::llm::MockLlmProvider;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};

    const EMBEDDING_DIM: usize = 1536;

    let vector_store = SqliteVectorStore::new(config.store_dir(), EMBEDDING_DIM).await?;
    let shared_conn = vector_store.connection();
    let vector_store: Arc<dyn VectorStore> = Arc::new(vector_store);
    let graph_store: Arc<dyn GraphStore> = Arc::new(SqliteGraphStore::with_connection(shared_conn));
    let llm: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new());

    Ok(karta_core::Karta::new(vector_store, graph_store, llm, config.core.clone()).await?)
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
    fn mock_flag_is_available() {
        let cli = Cli::parse_from(["karta-mcp", "serve", "--mock"]);
        if let Some(Commands::Serve(args)) = cli.command {
            assert!(args.mock);
        } else {
            panic!("expected serve subcommand");
        }
    }
}
