//! Tests for the retrieve-only `fetch_memories` API and the three-model
//! environment plumbing (CORE / ANSWER / JUDGE split).
//!
//! Uses `MockLlmProvider` + real LanceDB + real SQLite so the retrieval
//! pipeline is exercised end-to-end without any network calls.
//!
//! Run: cargo test --test fetch_memories

use std::sync::Arc;

use karta_core::config::KartaConfig;
use karta_core::llm::{LlmProvider, MockLlmProvider};
use karta_core::store::lance::LanceVectorStore;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_core::Karta;

/// Build a fresh Karta backed by MockLlmProvider + on-disk Lance/SQLite.
/// Wiping any previous data dir so the test starts from a clean slate.
async fn make_karta(tag: &str) -> Karta {
    let data_dir = format!("/tmp/karta-test-fetch-memories-{}", tag);
    let _ = std::fs::remove_dir_all(&data_dir);

    let vector_store = Arc::new(
        LanceVectorStore::new(
            &data_dir,
            karta_core::store::lance::DEFAULT_EMBEDDING_DIM,
        )
        .await
        .unwrap(),
    ) as Arc<dyn VectorStore>;

    let graph_store = Arc::new(SqliteGraphStore::new(&data_dir).unwrap()) as Arc<dyn GraphStore>;
    let llm = Arc::new(MockLlmProvider::new()) as Arc<dyn LlmProvider>;

    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;

    Karta::new(vector_store, graph_store, llm, config)
        .await
        .unwrap()
}

/// `fetch_memories` returns structured retrieval output WITHOUT calling
/// the LLM for synthesis. The context string should be non-empty, note
/// metadata should be populated, and the assembled context should include
/// content from the notes we ingested.
#[tokio::test]
async fn fetch_memories_returns_retrieved_context_without_llm_answer() {
    let karta = make_karta("basic").await;

    // Seed notes that will overlap the query's vocabulary so MockLlm's
    // hash-based embedder produces high cosine similarity.
    let notes = [
        "Alice prefers dark mode in the dashboard and asked to make it the default.",
        "The dashboard theme is currently light mode by default — Alice flagged this as a paper cut.",
        "Alice's team uses the dashboard for daily standups and wants Slack notifications too.",
        "Dark mode was scoped for the next sprint but not yet implemented.",
    ];
    for n in notes {
        karta.add_note(n).await.expect("ingest note");
    }

    let memories = karta
        .fetch_memories("What do we know about Alice's dashboard preferences?", 5)
        .await
        .expect("fetch_memories ok");

    // Struct invariants
    assert_eq!(memories.query, "What do we know about Alice's dashboard preferences?");
    assert!(
        !memories.notes.is_empty(),
        "expected at least one retrieved note, got zero"
    );
    assert_eq!(
        memories.notes.len(),
        memories.note_ids.len(),
        "notes and note_ids must be parallel"
    );
    assert!(
        !memories.context.is_empty(),
        "context string should be populated"
    );

    // The context should carry substance from the ingested notes — pick a
    // distinctive term that only appears in the notes we seeded.
    let ctx_lower = memories.context.to_lowercase();
    assert!(
        ctx_lower.contains("dark mode") || ctx_lower.contains("dashboard"),
        "context should include relevant note content, got: {}",
        memories.context
    );

    // query_mode is always a non-empty label (Standard / Breadth / Temporal / ...)
    assert!(
        !memories.query_mode.is_empty(),
        "query_mode classification should be set"
    );
}

/// Calling `fetch_memories` on an empty store returns an empty result
/// shape rather than erroring — callers can decide whether to abstain.
#[tokio::test]
async fn fetch_memories_empty_store_returns_empty_shape() {
    let karta = make_karta("empty").await;

    let memories = karta
        .fetch_memories("any question", 5)
        .await
        .expect("fetch_memories should not error on empty store");

    assert_eq!(memories.query, "any question");
    assert!(memories.notes.is_empty());
    assert!(memories.note_ids.is_empty());
    assert_eq!(memories.contradiction_injected, 0);
    // context may be empty when there's nothing to assemble
    assert!(memories.context.is_empty());
}

/// `fetch_memories` does NOT call the LLM for chat/synthesis. We prove
/// this by wrapping MockLlm with a counter and asserting zero chat calls
/// past the ingest phase.
#[tokio::test]
async fn fetch_memories_does_not_invoke_answer_llm() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use async_trait::async_trait;
    use karta_core::error::Result as KartaResult;
    use karta_core::llm::{ChatMessage, ChatResponse, GenConfig};

    struct CountingLlm {
        inner: MockLlmProvider,
        chat_calls: Arc<AtomicU64>,
    }

    #[async_trait]
    impl LlmProvider for CountingLlm {
        async fn chat(
            &self,
            messages: &[ChatMessage],
            config: &GenConfig,
        ) -> KartaResult<ChatResponse> {
            self.chat_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.chat(messages, config).await
        }

        async fn embed(&self, texts: &[&str]) -> KartaResult<Vec<Vec<f32>>> {
            self.inner.embed(texts).await
        }

        fn model_id(&self) -> &str {
            self.inner.model_id()
        }

        fn embedding_model_id(&self) -> &str {
            self.inner.embedding_model_id()
        }
    }

    let data_dir = "/tmp/karta-test-fetch-memories-counting".to_string();
    let _ = std::fs::remove_dir_all(&data_dir);

    let vector_store = Arc::new(
        LanceVectorStore::new(&data_dir, karta_core::store::lance::DEFAULT_EMBEDDING_DIM)
            .await
            .unwrap(),
    ) as Arc<dyn VectorStore>;
    let graph_store = Arc::new(SqliteGraphStore::new(&data_dir).unwrap()) as Arc<dyn GraphStore>;

    let chat_calls = Arc::new(AtomicU64::new(0));
    let llm = Arc::new(CountingLlm {
        inner: MockLlmProvider::new(),
        chat_calls: Arc::clone(&chat_calls),
    }) as Arc<dyn LlmProvider>;

    let mut config = KartaConfig::default();
    config.storage.data_dir = data_dir;
    let karta = Karta::new(vector_store, graph_store, llm, config)
        .await
        .unwrap();

    // Ingest one note (this WILL call chat for fact extraction; that's
    // internal Karta work, not the answer path).
    karta
        .add_note("Project Nimbus launches on 2026-05-01")
        .await
        .unwrap();

    // Snapshot the chat-call count after ingest. fetch_memories must not
    // increment it — the retrieval path only touches embed() + graph +
    // vector store, never chat().
    let before_fetch = chat_calls.load(Ordering::SeqCst);
    let _ = karta.fetch_memories("When does Nimbus launch?", 5).await.unwrap();
    let after_fetch = chat_calls.load(Ordering::SeqCst);

    assert_eq!(
        before_fetch, after_fetch,
        "fetch_memories must not invoke chat()/answer LLM (before={} after={})",
        before_fetch, after_fetch
    );
}
