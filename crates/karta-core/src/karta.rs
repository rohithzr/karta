use std::sync::Arc;

use crate::clock::ClockContext;
use crate::config::KartaConfig;
use crate::dream::{DreamEngine, DreamRun};
use crate::error::{KartaError, Result};
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
        Self::new_with_synthesis(vector_store, graph_store, llm, None, config).await
    }

    /// Create a Karta instance where the final answer-synthesis call can be
    /// routed to a separate LLM (e.g. a stronger model used only at answer
    /// time). All other Karta-internal calls — write, dream, rerank, query
    /// classification — still go through `llm`. Pass `None` for `synthesis_llm`
    /// to keep synthesis on the primary LLM (standard behavior).
    pub async fn new_with_synthesis(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        llm: Arc<dyn LlmProvider>,
        synthesis_llm: Option<Arc<dyn LlmProvider>>,
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
            synthesis_llm,
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
    /// Loads `.env` file if present (via dotenvy). Backend is chosen in this order:
    ///
    /// 1. **Explicit config**: `config.llm.default.base_url` set → OpenAI-compatible
    ///    endpoint (Ollama, vLLM, Groq, Together, …).
    /// 2. **`OPENAI_API_BASE` env var**: same OpenAI-compatible path. Wins over
    ///    `AZURE_OPENAI_API_KEY` so you can flip a single env var to redirect
    ///    Karta at a local Ollama during benchmarks without editing `.env`.
    /// 3. **`AZURE_OPENAI_API_KEY` env var**: Azure OpenAI via native `AzureConfig`.
    ///    Also reads `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` (default
    ///    `2025-04-01-preview`), `AZURE_OPENAI_CHAT_MODEL`, `AZURE_OPENAI_EMBEDDING_MODEL`.
    /// 4. **Fallback**: standard OpenAI (reads `OPENAI_API_KEY`).
    ///
    /// Chat model resolution, in order: `KARTA_CHAT_MODEL` env →
    /// `AZURE_OPENAI_CHAT_MODEL` env (Azure branch only) → `config.llm.default.model`.
    ///
    /// Embedding model resolution, in order: `KARTA_EMBEDDING_MODEL` env →
    /// `AZURE_OPENAI_EMBEDDING_MODEL` env → `"text-embedding-3-small"`.
    ///
    /// **Provider split:** if both `OPENAI_API_BASE` (chat, e.g. Ollama) *and*
    /// `AZURE_OPENAI_API_KEY` (embeddings) are set, chat goes to the
    /// OpenAI-compatible endpoint and embeddings go to Azure. This is the
    /// recommended BEAM config: local GPU for gen throughput, Azure for
    /// high-quality embeddings that match the P1 baseline's vector space.
    #[cfg(all(feature = "sqlite-vec", feature = "sqlite", feature = "openai", not(feature = "lance")))]
    pub async fn with_defaults(config: KartaConfig) -> Result<Self> {
        use crate::llm::{OpenAiProvider, SplitProvider};
        use crate::store::sqlite::SqliteGraphStore;
        use crate::store::sqlite_vec::SqliteVectorStore;

        // Load .env if present (silently ignore if missing)
        let _ = dotenvy::dotenv();

        let model_ref = &config.llm.default;

        let chat_model_base = std::env::var("KARTA_CORE_MODEL")
            .or_else(|_| std::env::var("KARTA_CHAT_MODEL"))
            .unwrap_or_else(|_| model_ref.model.clone());

        let embedding_model = std::env::var("KARTA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| {
                std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string())
            });

        let openai_base = model_ref
            .base_url
            .clone()
            .or_else(|| std::env::var("OPENAI_API_BASE").ok());
        let azure_creds = match (
            std::env::var("AZURE_OPENAI_API_KEY").ok(),
            std::env::var("AZURE_OPENAI_ENDPOINT").ok(),
        ) {
            (Some(key), Some(endpoint)) => Some((key, endpoint)),
            (Some(_), None) => {
                return Err(KartaError::Config(
                    "AZURE_OPENAI_API_KEY is set but AZURE_OPENAI_ENDPOINT is missing".into(),
                ));
            }
            _ => None,
        };

        let chat_llm: Arc<dyn LlmProvider> = if let Some(ref base_url) = openai_base {
            let api_key = std::env::var("OPENAI_API_KEY")
                .unwrap_or_else(|_| "ollama".to_string());
            Arc::new(OpenAiProvider::with_api_key(
                &chat_model_base,
                &embedding_model,
                &api_key,
                Some(base_url),
            ))
        } else if let Some((azure_key, endpoint)) = azure_creds.clone() {
            let chat_model = std::env::var("AZURE_OPENAI_CHAT_MODEL")
                .unwrap_or_else(|_| chat_model_base.clone());
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
            Arc::new(OpenAiProvider::new(&chat_model_base, &embedding_model))
        };

        let answer_model_opt = std::env::var("KARTA_ANSWER_MODEL").ok();
        let mut synthesis_llm: Option<Arc<dyn LlmProvider>> = None;

        let llm: Arc<dyn LlmProvider> = if openai_base.is_some() {
            if let Some((azure_key, endpoint)) = azure_creds.clone() {
                let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
                    .unwrap_or_else(|_| "2025-04-01-preview".to_string());
                let azure_chat_model = std::env::var("AZURE_OPENAI_CHAT_MODEL")
                    .unwrap_or_else(|_| chat_model_base.clone());
                let azure_embedding_model = std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string());
                let embed_llm: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::azure(
                    &endpoint,
                    &azure_key,
                    &api_version,
                    &azure_chat_model,
                    &azure_embedding_model,
                ));
                Arc::new(SplitProvider::new(chat_llm, embed_llm))
            } else {
                chat_llm
            }
        } else {
            chat_llm
        };

        if let Some(answer_model) = answer_model_opt {
            if let Ok(answer_base) = std::env::var("KARTA_ANSWER_BASE_URL") {
                let answer_key = std::env::var("KARTA_ANSWER_API_KEY")
                    .unwrap_or_else(|_| "placeholder".to_string());
                synthesis_llm = Some(Arc::new(OpenAiProvider::with_api_key(
                    &answer_model,
                    &embedding_model,
                    &answer_key,
                    Some(&answer_base),
                )));
            } else if let Some((azure_key, endpoint)) = azure_creds {
                let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
                    .unwrap_or_else(|_| "2025-04-01-preview".to_string());
                let azure_embedding_model = std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string());
                synthesis_llm = Some(Arc::new(OpenAiProvider::azure(
                    &endpoint,
                    &azure_key,
                    &api_version,
                    &answer_model,
                    &azure_embedding_model,
                )));
            }
        }

        const DEFAULT_DIM: usize = 1536;
        let embedding_dim = match llm.embed(&["karta-init-probe"]).await {
            Ok(vectors) if !vectors.is_empty() && !vectors[0].is_empty() => vectors[0].len(),
            _ => DEFAULT_DIM,
        };

        let sqlite_vec_store =
            SqliteVectorStore::new(&config.storage.data_dir, embedding_dim).await?;
        let shared_conn = sqlite_vec_store.connection();
        let vector_store = Arc::new(sqlite_vec_store) as Arc<dyn VectorStore>;
        let graph_store = Arc::new(SqliteGraphStore::with_connection(shared_conn)) as Arc<dyn GraphStore>;

        Self::new_with_synthesis(vector_store, graph_store, llm, synthesis_llm, config).await
    }

    #[cfg(all(feature = "lance", feature = "sqlite", feature = "openai"))]
    pub async fn with_defaults(config: KartaConfig) -> Result<Self> {
        use crate::llm::{OpenAiProvider, SplitProvider};
        use crate::store::lance::{DEFAULT_EMBEDDING_DIM, LanceVectorStore};
        use crate::store::sqlite::SqliteGraphStore;

        // Load .env if present (silently ignore if missing)
        let _ = dotenvy::dotenv();

        let model_ref = &config.llm.default;

        // Core model: `KARTA_CORE_MODEL` (preferred, names the model used
        // for all of Karta's *internal* LLM work — write, dream, link
        // analysis, query classification, rerank) wins over the legacy
        // `KARTA_CHAT_MODEL` name. Both fall back to the config default.
        // The Azure branch may further override with `AZURE_OPENAI_CHAT_MODEL`.
        let chat_model_base = std::env::var("KARTA_CORE_MODEL")
            .or_else(|_| std::env::var("KARTA_CHAT_MODEL"))
            .unwrap_or_else(|_| model_ref.model.clone());

        // Embedding model: KARTA_EMBEDDING_MODEL → AZURE_OPENAI_EMBEDDING_MODEL → default
        let embedding_model = std::env::var("KARTA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| {
                std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string())
            });

        let openai_base = model_ref
            .base_url
            .clone()
            .or_else(|| std::env::var("OPENAI_API_BASE").ok());
        let azure_creds = match (
            std::env::var("AZURE_OPENAI_API_KEY").ok(),
            std::env::var("AZURE_OPENAI_ENDPOINT").ok(),
        ) {
            (Some(key), Some(endpoint)) => Some((key, endpoint)),
            (Some(_), None) => {
                return Err(KartaError::Config(
                    "AZURE_OPENAI_API_KEY is set but AZURE_OPENAI_ENDPOINT is missing".into(),
                ));
            }
            _ => None,
        };

        // Chat backend: OPENAI_API_BASE > Azure > standard OpenAI
        let chat_llm: Arc<dyn LlmProvider> = if let Some(ref base_url) = openai_base {
            // OpenAI-compatible endpoint (Ollama, vLLM, Groq, Together, …).
            let api_key = std::env::var("OPENAI_API_KEY")
                .unwrap_or_else(|_| "ollama".to_string());
            Arc::new(OpenAiProvider::with_api_key(
                &chat_model_base,
                &embedding_model,
                &api_key,
                Some(base_url),
            ))
        } else if let Some((azure_key, endpoint)) = azure_creds.clone() {
            let chat_model = std::env::var("AZURE_OPENAI_CHAT_MODEL")
                .unwrap_or_else(|_| chat_model_base.clone());
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
            Arc::new(OpenAiProvider::new(&chat_model_base, &embedding_model))
        };

        // Three-model architecture:
        //   1. CORE (`KARTA_CORE_MODEL` / `chat_llm`) — powers every Karta-
        //      internal LLM call: write-side fact extraction, dream digest
        //      generation, link analysis, query classification, reranking.
        //      Pointed at by `OPENAI_API_BASE` (typically local Ollama) or
        //      falls back to Azure / OpenAI.
        //   2. ANSWER (`KARTA_ANSWER_MODEL`, optional) — powers ONLY the
        //      final answer-composition call in `read.rs::synthesize`. If
        //      set, Karta builds a dedicated answer backend and passes it
        //      through as `synthesis_llm`. If unset, the answer step uses
        //      the core LLM. This split lets you use a cheap/local core
        //      for the 97% of internal work and reserve a premium model
        //      for the 3% that's user-facing.
        //   3. JUDGE (`KARTA_JUDGE_MODEL`) — NOT built here; it lives in
        //      the BEAM benchmark harness because Karta itself never
        //      calls a judge in normal operation.
        //
        // Embedding backend: only split when chat is hitting an OpenAI-
        // compatible endpoint (Ollama) *and* Azure creds are available.
        // In split mode the embedding deployment comes from
        // `AZURE_OPENAI_EMBEDDING_MODEL`, NOT `KARTA_EMBEDDING_MODEL`
        // (which names the Ollama-native embedder and does not exist as
        // an Azure deployment).
        //
        // Answer backend resolution, in order of precedence:
        //   a) `KARTA_ANSWER_BASE_URL` set → use it as an OpenAI-compatible
        //      endpoint with `KARTA_ANSWER_MODEL` as the model name and
        //      `KARTA_ANSWER_API_KEY` (or placeholder) as the key.
        //   b) Azure creds present → treat `KARTA_ANSWER_MODEL` as an
        //      Azure deployment id and build an Azure client.
        //   c) No way to build a dedicated answer backend → fall back to
        //      the core LLM for synthesis.
        let answer_model_opt = std::env::var("KARTA_ANSWER_MODEL").ok();
        let mut synthesis_llm: Option<Arc<dyn LlmProvider>> = None;

        let llm: Arc<dyn LlmProvider> = if openai_base.is_some() {
            if let Some((azure_key, endpoint)) = azure_creds.clone() {
                let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
                    .unwrap_or_else(|_| "2025-04-01-preview".to_string());
                let azure_chat_model = std::env::var("AZURE_OPENAI_CHAT_MODEL")
                    .unwrap_or_else(|_| chat_model_base.clone());
                let azure_embedding_model = std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string());
                let embed_llm: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::azure(
                    &endpoint,
                    &azure_key,
                    &api_version,
                    &azure_chat_model,
                    &azure_embedding_model,
                ));
                Arc::new(SplitProvider::new(chat_llm, embed_llm))
            } else {
                chat_llm
            }
        } else {
            chat_llm
        };

        // Build the optional answer backend after the core llm so it can
        // be handed to ReadEngine alongside it. Independent of whether
        // the core path is split or not.
        if let Some(answer_model) = answer_model_opt {
            if let Ok(answer_base) = std::env::var("KARTA_ANSWER_BASE_URL") {
                let answer_key = std::env::var("KARTA_ANSWER_API_KEY")
                    .unwrap_or_else(|_| "placeholder".to_string());
                synthesis_llm = Some(Arc::new(OpenAiProvider::with_api_key(
                    &answer_model,
                    &embedding_model,
                    &answer_key,
                    Some(&answer_base),
                )));
            } else if let Some((azure_key, endpoint)) = azure_creds {
                let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
                    .unwrap_or_else(|_| "2025-04-01-preview".to_string());
                let azure_embedding_model = std::env::var("AZURE_OPENAI_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string());
                synthesis_llm = Some(Arc::new(OpenAiProvider::azure(
                    &endpoint,
                    &azure_key,
                    &api_version,
                    &answer_model,
                    &azure_embedding_model,
                )));
            }
            // else: no way to build a dedicated backend; leave None, answer
            // step falls through to the core LLM automatically.
        }

        // Probe the embedding model once to detect its dimensionality so the
        // LanceDB schema matches the provider (nomic-embed-text is 768, OpenAI
        // text-embedding-3-small is 1536, etc.). Falls back to the historical
        // default if the probe fails so we still surface a clearer error from
        // the real query path rather than failing the constructor silently.
        let embedding_dim = match llm.embed(&["karta-init-probe"]).await {
            Ok(vectors) if !vectors.is_empty() && !vectors[0].is_empty() => {
                vectors[0].len()
            }
            _ => DEFAULT_EMBEDDING_DIM,
        };

        let vector_store = Arc::new(
            LanceVectorStore::new(&config.storage.data_dir, embedding_dim).await?,
        ) as Arc<dyn VectorStore>;

        let graph_store = Arc::new(
            SqliteGraphStore::new(&config.storage.data_dir)?,
        ) as Arc<dyn GraphStore>;

        Self::new_with_synthesis(vector_store, graph_store, llm, synthesis_llm, config).await
    }

    // --- Write ---

    /// Live default — sugar over `add_note_with_clock(content, None, None,
    /// ClockContext::now())`. Intended for smoke tests, docs examples, and
    /// quick scripts. Production callers should prefer `add_note_with_clock`.
    pub async fn add_note(&self, content: &str) -> Result<MemoryNote> {
        self.add_note_with_clock(content, None, None, ClockContext::now()).await
    }

    /// Canonical ingest with full clock + session control. session_id is
    /// optional because not every note belongs to a session (a one-shot
    /// `add_note(content)` doesn't have one). turn_index is optional for
    /// non-conversational ingest paths.
    pub async fn add_note_with_clock(
        &self,
        content: &str,
        session_id: Option<&str>,
        turn_index: Option<u32>,
        ctx: ClockContext,
    ) -> Result<MemoryNote> {
        self.write_engine
            .add_note_with_clock(content, session_id, turn_index, ctx)
            .await
    }

    // --- Read ---

    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_clock(query, top_k, ClockContext::now()).await
    }

    pub async fn search_with_clock(
        &self,
        query: &str,
        top_k: usize,
        ctx: ClockContext,
    ) -> Result<Vec<SearchResult>> {
        self.read_engine.search_with_clock(query, top_k, ctx).await
    }

    pub async fn ask(&self, query: &str, top_k: usize) -> Result<crate::note::AskResult> {
        self.ask_with_clock(query, top_k, ClockContext::now()).await
    }

    pub async fn ask_with_clock(
        &self,
        query: &str,
        top_k: usize,
        ctx: ClockContext,
    ) -> Result<crate::note::AskResult> {
        self.read_engine.ask_with_clock(query, top_k, ctx).await
    }

    /// Retrieve-only entry point: runs the full Karta retrieval pipeline
    /// (classify → search → rerank → dedup → order → contradiction inject
    /// → assemble context) and returns the assembled memories **without**
    /// calling any LLM for answer composition.
    ///
    /// Karta's responsibility ends at "here are the relevant memories,
    /// pre-assembled into an LLM-ready context string". The caller composes
    /// the final prompt, picks their own model, and runs the generation
    /// step themselves. Use [`ask`] if you want Karta to also compose an
    /// answer via its configured answer-LLM.
    pub async fn fetch_memories(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<crate::note::FetchedMemories> {
        self.fetch_memories_with_clock(query, top_k, ClockContext::now()).await
    }

    pub async fn fetch_memories_with_clock(
        &self,
        query: &str,
        top_k: usize,
        ctx: ClockContext,
    ) -> Result<crate::note::FetchedMemories> {
        self.read_engine.fetch_memories_with_clock(query, top_k, ctx).await
    }

    // --- Dream ---

    pub async fn run_dreaming(
        &self,
        scope_type: &str,
        scope_id: &str,
    ) -> Result<DreamRun> {
        self.run_dreaming_with_clock(scope_type, scope_id, ClockContext::now()).await
    }

    pub async fn run_dreaming_with_clock(
        &self,
        scope_type: &str,
        scope_id: &str,
        ctx: ClockContext,
    ) -> Result<DreamRun> {
        let engine = DreamEngine::new(
            Arc::clone(&self.vector_store),
            Arc::clone(&self.graph_store),
            Arc::clone(&self.llm),
            self.config.dream.clone(),
        );
        engine.run_with_clock(scope_type, scope_id, ctx).await
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

    /// Get the atomic facts extracted from a note, in ordinal order.
    pub async fn get_facts_for_note(
        &self,
        note_id: &str,
    ) -> Result<Vec<crate::note::AtomicFact>> {
        self.vector_store.get_facts_for_note(note_id).await
    }

    /// Raw LLM chat access for evaluation/judge use cases.
    pub async fn llm_chat(
        &self,
        messages: &[crate::llm::ChatMessage],
        config: &crate::llm::GenConfig,
    ) -> Result<crate::llm::ChatResponse> {
        self.llm.chat(messages, config).await
    }
}
