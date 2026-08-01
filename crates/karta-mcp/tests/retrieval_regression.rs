//! Retrieval regression baseline: fixed probe queries against a wrapper-managed
//! store must return the same results as a direct `karta_core` Karta built from
//! the same underlying stores.

mod common;

use std::sync::Arc;

use karta_core::{
    ClockContext, Karta, config::KartaConfig, llm::LlmProvider, llm::MockLlmProvider,
};
use serde_json::json;

#[tokio::test]
async fn fixed_probe_queries_match_direct_karta_core() {
    let rt = common::TestRuntime::new().await;

    // Seed notes through both the direct add path and the capture path.
    rt.handle
        .karta
        .add_note_with_clock(
            "the quick brown fox jumps over the lazy dog",
            None,
            None,
            ClockContext::now(),
        )
        .await
        .unwrap();

    rt.post_capture(json!({
        "hook_event_name": "UserPromptSubmit",
        "session_id": "rr",
        "prompt": "karta memory engine harness"
    }))
    .await;
    rt.post_capture(json!({
        "hook_event_name": "PostToolUse",
        "session_id": "rr",
        "tool_name": "Edit",
        "tool_input": { "path": "src/lib.rs" },
        "tool_output": "contract replay harness"
    }))
    .await;
    rt.post_capture(json!({
        "hook_event_name": "Stop",
        "session_id": "rr",
        "last_assistant_message": "harness complete"
    }))
    .await;
    rt.drain().await;

    // Build a direct karta_core Karta from the same underlying stores to serve
    // as a regression reference. The wrapper must not alter retrieval behavior.
    let direct_llm: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new());
    let direct_karta = Karta::new(
        rt.handle.vector_store.clone(),
        rt.handle.graph_store.clone(),
        direct_llm,
        KartaConfig::default(),
    )
    .await
    .unwrap();

    let queries = ["fox", "karta", "harness"];
    for query in queries {
        let wrapper = rt.handle.karta.fetch_memories(query, 5).await.unwrap();
        let direct = direct_karta.fetch_memories(query, 5).await.unwrap();
        assert_eq!(
            wrapper.note_ids, direct.note_ids,
            "note_ids for query {query:?} differ between wrapper and direct karta_core"
        );
        assert_eq!(
            wrapper.query_mode, direct.query_mode,
            "query_mode for query {query:?} differs between wrapper and direct karta_core"
        );
    }

    rt.cleanup().await;
}
