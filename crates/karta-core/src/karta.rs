use std::sync::{Arc, Mutex};

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
    slot_ledger: Option<Arc<crate::store::slot_ledger::SlotLedger>>,
    entity_aliases: Arc<Mutex<crate::extract::entity_key::EntityAliases>>,
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

        let entity_aliases = write_engine.entity_aliases_handle();

        Ok(Self {
            write_engine,
            read_engine,
            vector_store,
            graph_store,
            llm,
            config,
            slot_ledger: None,
            entity_aliases,
        })
    }

    /// Attach a slot ledger for mutable-slot tracking. Wires the same
    /// `Arc` into the write engine (so ingest populates it) and keeps a
    /// copy here (so reads can query it via `slot_ledger_current`).
    pub(crate) fn attach_slot_ledger(&mut self, ledger: Arc<crate::store::slot_ledger::SlotLedger>) {
        self.write_engine.attach_slot_ledger(Arc::clone(&ledger));
        self.slot_ledger = Some(ledger);
    }

    /// Create with default embedded stores (sqlite-vec + SQLite) and OpenAI-compatible LLM.
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
    #[cfg(all(feature = "sqlite-vec", feature = "sqlite", feature = "openai"))]
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
        let graph_store = Arc::new(SqliteGraphStore::with_connection(shared_conn.clone())) as Arc<dyn GraphStore>;

        let mut karta = Self::new_with_synthesis(vector_store, graph_store, llm, synthesis_llm, config).await?;
        let ledger = Arc::new(crate::store::slot_ledger::SlotLedger::new(shared_conn)?);
        karta.attach_slot_ledger(ledger);
        Ok(karta)
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

    /// Current open ledger rows for a mutable slot. Canonicalizes `entity`
    /// via the same write-time normalization, then reads the slot_ledger.
    /// Empty when no ledger is attached (e.g. non-`with_defaults` construction).
    pub async fn slot_ledger_current(
        &self,
        entity: &str,
        predicate: &str,
    ) -> Result<Vec<crate::store::slot_ledger::LedgerRow>> {
        let Some(ledger) = &self.slot_ledger else {
            return Ok(Vec::new());
        };
        let norm = crate::extract::entity_key::normalize_entity(entity);
        ledger.current(&norm, predicate)
    }

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
