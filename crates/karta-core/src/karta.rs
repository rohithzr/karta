use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::config::KartaConfig;
use crate::dream::{DreamEngine, DreamRun};
use crate::error::Result;
use crate::llm::LlmProvider;
use crate::note::{MemoryNote, SearchResult};
use crate::read::ReadEngine;
use crate::rerank::{JinaReranker, LlmReranker, NoopReranker, Reranker};
use crate::store::{GraphStore, VectorStore};
use crate::write::WriteEngine;

/// Main entry point for the Karta memory system.
pub struct Karta {
    write_engine: WriteEngine,
    read_engine: ReadEngine,
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    llm: Arc<dyn LlmProvider>,
    config: KartaConfig,
}

impl Karta {
    /// Create a new Karta instance with explicit store and LLM implementations.
    pub async fn new(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        llm: Arc<dyn LlmProvider>,
        config: KartaConfig,
    ) -> Result<Self> {
        // Initialize graph store schema
        graph_store.init().await?;

        let write_engine = WriteEngine::new(
            Arc::clone(&vector_store),
            Arc::clone(&graph_store),
            Arc::clone(&llm),
            config.write.clone(),
            config.episode.clone(),
        );

        // Create reranker based on config + available credentials
        let reranker: Arc<dyn Reranker> = if config.reranker.enabled {
            if let Ok(jina_key) = std::env::var("JINA_API_KEY") {
                Arc::new(JinaReranker::new(&jina_key))
            } else {
                Arc::new(LlmReranker::new(Arc::clone(&llm)))
            }
        } else {
            Arc::new(NoopReranker)
        };

        let read_engine = ReadEngine::new(
            Arc::clone(&vector_store),
            Arc::clone(&graph_store),
            Arc::clone(&llm),
            reranker,
            config.read.clone(),
            config.reranker.clone(),
        );

        Ok(Self {
            write_engine,
            read_engine,
            vector_store,
            graph_store,
            llm,
            config,
        })
    }

    /// Create with default embedded stores (LanceDB + SQLite) and OpenAI-compatible LLM.
    ///
    /// Loads `.env` file if present (via dotenvy). Reads credentials from env vars:
    ///
    /// For standard OpenAI:
    ///   - `OPENAI_API_KEY` (read by async-openai automatically)
    ///
    /// For Azure OpenAI:
    ///   - `AZURE_OPENAI_API_KEY`
    ///   - `AZURE_OPENAI_ENDPOINT`
    ///   - `AZURE_OPENAI_API_VERSION` (optional, defaults to 2025-04-01-preview)
    ///
    /// For other OpenAI-compatible providers (Ollama, vLLM, Groq, Together):
    ///   - Set `llm.default.base_url` in config, or
    ///   - Set `OPENAI_API_BASE` env var
    ///
    /// Model names come from `config.llm.default.model` and can be overridden
    /// per-operation via `config.llm.overrides`.
    #[cfg(all(feature = "lance", feature = "sqlite", feature = "openai"))]
    pub async fn with_defaults(config: KartaConfig) -> Result<Self> {
        use crate::error::KartaError;
        use crate::llm::OpenAiProvider;
        use crate::store::lance::LanceVectorStore;
        use crate::store::sqlite::SqliteGraphStore;

        // Load .env if present (silently ignore if missing)
        let _ = dotenvy::dotenv();

        let lance_uri = config
            .storage
            .lance_uri
            .clone()
            .unwrap_or_else(|| format!("{}/lance", config.storage.data_dir));
        let vector_store =
            Arc::new(LanceVectorStore::new(&lance_uri).await?) as Arc<dyn VectorStore>;

        let graph_store =
            Arc::new(SqliteGraphStore::new(&config.storage.data_dir)?) as Arc<dyn GraphStore>;

        let model_ref = &config.llm.default;

        // Determine embedding model from config or env
        let embedding_model = std::env::var("KARTA_EMBEDDING_MODEL").unwrap_or_else(|_| {
            std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string())
        });

        // Build LLM provider based on what credentials are available
        let llm: Arc<dyn LlmProvider> = if let Some(ref base_url) = model_ref.base_url {
            // Explicit base URL in config (Ollama, vLLM, etc.)
            Arc::new(OpenAiProvider::with_base_url(
                &model_ref.model,
                &embedding_model,
                base_url,
            ))
        } else if let Ok(azure_key) = std::env::var("AZURE_OPENAI_API_KEY") {
            // Azure OpenAI — uses native AzureConfig for correct URL construction
            let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").map_err(|_| {
                KartaError::Config(
                    "AZURE_OPENAI_API_KEY is set but AZURE_OPENAI_ENDPOINT is missing".into(),
                )
            })?;
            let chat_model = std::env::var("AZURE_OPENAI_CHAT_MODEL")
                .unwrap_or_else(|_| model_ref.model.clone());
            let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
                .unwrap_or_else(|_| "2025-04-01-preview".to_string());

            Arc::new(OpenAiProvider::azure(
                &endpoint,
                &azure_key,
                &api_version,
                &chat_model,
                &embedding_model,
            ))
        } else {
            // Standard OpenAI (reads OPENAI_API_KEY from env automatically)
            Arc::new(OpenAiProvider::new(&model_ref.model, &embedding_model))
        };

        Self::new(vector_store, graph_store, llm, config).await
    }

    // --- Write ---

    pub async fn add_note(&self, content: &str) -> Result<MemoryNote> {
        self.write_engine.add_note(content).await
    }

    pub async fn add_note_with_session(
        &self,
        content: &str,
        session_id: &str,
    ) -> Result<MemoryNote> {
        self.write_engine
            .add_note_with_session(content, session_id)
            .await
    }

    /// Add a note with session context and optional temporal metadata.
    /// `turn_index`: position of this message within its conversation (0-indexed).
    /// `source_timestamp`: original timestamp from source data (distinct from ingestion time).
    pub async fn add_note_with_metadata(
        &self,
        content: &str,
        session_id: &str,
        turn_index: Option<u32>,
        source_timestamp: Option<DateTime<Utc>>,
    ) -> Result<MemoryNote> {
        let mut note = self
            .write_engine
            .add_note_with_session(content, session_id)
            .await?;

        if turn_index.is_some() || source_timestamp.is_some() {
            note.turn_index = turn_index;
            note.source_timestamp = source_timestamp;
            self.vector_store.upsert(&note).await?;
        }

        Ok(note)
    }

    // --- Read ---

    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.read_engine.search(query, top_k).await
    }

    pub async fn ask(&self, query: &str, top_k: usize) -> Result<crate::note::AskResult> {
        self.read_engine.ask(query, top_k).await
    }

    // --- Dream ---

    pub async fn run_dreaming(&self, scope_type: &str, scope_id: &str) -> Result<DreamRun> {
        let engine = DreamEngine::new(
            Arc::clone(&self.vector_store),
            Arc::clone(&self.graph_store),
            Arc::clone(&self.llm),
            self.config.dream.clone(),
        );
        engine.run(scope_type, scope_id).await
    }

    // --- Inspection ---

    pub async fn get_note(&self, id: &str) -> Result<Option<MemoryNote>> {
        self.vector_store.get(id).await
    }

    pub async fn get_all_notes(&self) -> Result<Vec<MemoryNote>> {
        self.vector_store.get_all().await
    }

    pub async fn note_count(&self) -> Result<usize> {
        self.vector_store.count().await
    }

    /// Get links for a note from the graph store.
    pub async fn get_links(&self, note_id: &str) -> Result<Vec<String>> {
        self.graph_store.get_links(note_id).await
    }

    /// Raw LLM chat access for evaluation/judge use cases.
    pub async fn llm_chat(
        &self,
        messages: &[crate::llm::ChatMessage],
        config: &crate::llm::GenConfig,
    ) -> Result<crate::llm::ChatResponse> {
        self.llm.chat(messages, config).await
    }

    // --- Health ---

    pub async fn health_check(&self) -> Result<KartaHealth> {
        let vector_ok = self.vector_store.count().await.is_ok();
        let graph_meta = self.graph_store.get_schema_meta().await;
        let graph_ok = graph_meta.is_ok();

        let (schema_version, pending, mut warnings) = match graph_meta {
            Ok(meta) => (
                Some(meta.schema_version.to_string()),
                meta.pending_migrations,
                meta.warnings,
            ),
            Err(ref e) => (
                None,
                vec![],
                vec![format!("Graph store schema meta unavailable: {e}")],
            ),
        };
        if !vector_ok {
            warnings.push("Vector store health check failed".into());
        }
        if !pending.is_empty() {
            warnings.push(format!(
                "{} pending migration(s): {}",
                pending.len(),
                pending.join(", ")
            ));
        }

        Ok(KartaHealth {
            vector_store_ok: vector_ok,
            graph_store_ok: graph_ok,
            schema_version,
            pending_migrations: pending,
            warnings,
        })
    }
}

/// Health status of a Karta instance.
#[derive(Debug, Clone, serde::Serialize)]
pub struct KartaHealth {
    pub vector_store_ok: bool,
    pub graph_store_ok: bool,
    pub schema_version: Option<String>,
    pub pending_migrations: Vec<String>,
    pub warnings: Vec<String>,
}
