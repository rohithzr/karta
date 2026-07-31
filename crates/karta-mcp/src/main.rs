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

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

mod config;

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

    tracing::info!(
        store_dir = %config.store_dir(),
        capture_port = config.capture_port,
        precompact = config.precompact,
        "starting karta-mcp serve"
    );

    if args.mock {
        tracing::info!("using mock LLM provider");
    }

    // TODO: spawn rmcp stdio server, axum HTTP endpoint, and queue worker.
    // For the foundational feature the skeleton just waits for shutdown.
    tokio::signal::ctrl_c().await?;
    tracing::info!("shutdown signal received");
    Ok(())
}

async fn status() -> Result<()> {
    let config = Config::from_env()?;

    // Open the store with the mock provider so `status` works without a live
    // LLM endpoint. The note count and store directory are real; the
    // embedding_model label is taken from the environment when available.
    let note_count = open_mock_karta(&config).await?.note_count().await?;
    let embedding_model = std::env::var("KARTA_EMBEDDING_MODEL")
        .unwrap_or_else(|_| config.core.llm.default.model.clone());

    println!("note_count: {note_count}");
    println!("store_dir: {}", config.store_dir());
    println!("embedding_model: {embedding_model}");
    println!("capture_port: {}", config.capture_port);
    println!("queue_depth: 0");

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
