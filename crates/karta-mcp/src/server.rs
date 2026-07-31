//! rmcp stdio MCP server with the seven Karta tools.

use std::sync::Arc;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};

use crate::config::Config;
use crate::karta_handle::KartaHandle;
use crate::queue::CaptureQueue;
use crate::tools;

/// The Karta MCP server state.
pub struct KartaMcpServer {
    handle: KartaHandle,
    queue: Arc<CaptureQueue>,
    store_dir: String,
    embedding_model: String,
    capture_port: u16,
    tool_router: ToolRouter<Self>,
}

impl KartaMcpServer {
    /// Build a new MCP server around a shared `KartaHandle` and queue.
    pub fn new(handle: KartaHandle, queue: Arc<CaptureQueue>, config: &Config) -> Self {
        let embedding_model = std::env::var("KARTA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| config.core.llm.default.model.clone());
        Self {
            handle,
            queue,
            store_dir: config.store_dir().to_string(),
            embedding_model,
            capture_port: config.capture_port,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl KartaMcpServer {
    #[tool(description = "Add a note to the Karta store")]
    async fn karta_add_note(
        &self,
        params: rmcp::handler::server::wrapper::Parameters<tools::AddNoteParams>,
    ) -> Result<String, rmcp::ErrorData> {
        tools::handle_add_note(&self.handle.karta, params.0).await
    }

    #[tool(description = "Retrieve relevant memories for a query")]
    async fn karta_fetch_memories(
        &self,
        params: rmcp::handler::server::wrapper::Parameters<tools::FetchMemoriesParams>,
    ) -> Result<String, rmcp::ErrorData> {
        tools::handle_fetch_memories(&self.handle.karta, params.0).await
    }

    #[tool(description = "Run the Karta dream engine over a scope")]
    async fn karta_run_dreaming(
        &self,
        params: rmcp::handler::server::wrapper::Parameters<tools::RunDreamingParams>,
    ) -> Result<String, rmcp::ErrorData> {
        tools::handle_run_dreaming(&self.handle.karta, params.0).await
    }

    #[tool(description = "Start a new session and return orientation context")]
    async fn karta_session_start(
        &self,
        params: rmcp::handler::server::wrapper::Parameters<tools::SessionStartParams>,
    ) -> Result<String, rmcp::ErrorData> {
        tools::handle_session_start(&self.handle, params.0).await
    }

    #[tool(description = "End a session with an optional summary")]
    async fn karta_session_end(
        &self,
        params: rmcp::handler::server::wrapper::Parameters<tools::SessionEndParams>,
    ) -> Result<String, rmcp::ErrorData> {
        tools::handle_session_end(&self.handle, params.0).await
    }

    #[tool(description = "Run rule-based consolidation (no LLM)")]
    async fn karta_consolidate(
        &self,
        params: rmcp::handler::server::wrapper::Parameters<tools::ConsolidateParams>,
    ) -> Result<String, rmcp::ErrorData> {
        tools::handle_consolidate(&self.handle, params.0).await
    }

    #[tool(description = "Return current store and queue status")]
    async fn karta_status(&self) -> Result<String, rmcp::ErrorData> {
        tools::handle_status(
            &self.handle.karta,
            self.queue.clone(),
            &self.store_dir,
            &self.embedding_model,
            self.capture_port,
        )
        .await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for KartaMcpServer {}

/// Run the stdio MCP server. This consumes the server and blocks until the
/// transport closes or the cancellation token fires.
pub async fn run_stdio_server(
    server: KartaMcpServer,
    cancel: tokio_util::sync::CancellationToken,
) -> anyhow::Result<()> {
    let mut running = Some(server.serve(rmcp::transport::io::stdio()).await?);
    tokio::select! {
        _ = cancel.cancelled() => {
            if let Some(running) = running.take() {
                running.cancellation_token().cancel();
                running.waiting().await
                    .map_err(|e| anyhow::anyhow!("MCP server task failed: {e}"))?;
            }
            Ok(())
        }
        result = async { running.take().unwrap().waiting().await } => {
            result.map_err(|e| anyhow::anyhow!("MCP server task failed: {e}"))?;
            Ok(())
        }
    }
}
