//! MCP tool parameter structs and handlers for the seven Karta tools.

use std::sync::Arc;

use rmcp::schemars;
use serde::Deserialize;
use serde_json::json;

use crate::queue::CaptureQueue;
use crate::session;

/// Parameter struct for `karta_add_note`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AddNoteParams {
    pub content: String,
    pub session_id: Option<String>,
    pub turn_index: Option<u32>,
}

/// Parameter struct for `karta_fetch_memories`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FetchMemoriesParams {
    pub query: String,
    pub top_k: Option<usize>,
}

/// Parameter struct for `karta_run_dreaming`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RunDreamingParams {
    pub scope_type: Option<String>,
    pub scope_id: Option<String>,
}

/// Parameter struct for `karta_session_start`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SessionStartParams {
    pub agent: String,
    pub project: Option<String>,
}

/// Parameter struct for `karta_session_end`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SessionEndParams {
    pub session_id: String,
    pub summary: Option<String>,
}

/// Parameter struct for `karta_consolidate`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConsolidateParams {
    pub session_id: Option<String>,
}

/// Tool handler implementation.
///
/// The actual `#[tool]` methods live on the `KartaMcpServer` struct in
/// `server.rs`; this module provides the parameter schemas and the business
/// logic shared between tests and the server.
pub async fn handle_add_note(
    karta: &karta_core::Karta,
    params: AddNoteParams,
) -> Result<String, rmcp::ErrorData> {
    if params.content.trim().is_empty() {
        return Err(rmcp::ErrorData::invalid_params(
            "content must not be empty",
            None,
        ));
    }
    let note = karta
        .add_note_with_clock(
            &params.content,
            params.session_id.as_deref(),
            params.turn_index,
            karta_core::ClockContext::now(),
        )
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
    let response = json!({
        "note_id": note.id,
        "status": "ok",
    });
    serde_json::to_string(&response)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
}

pub async fn handle_fetch_memories(
    karta: &karta_core::Karta,
    params: FetchMemoriesParams,
) -> Result<String, rmcp::ErrorData> {
    let top_k = params.top_k.unwrap_or(5);
    let result = karta
        .fetch_memories(&params.query, top_k)
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
    let response = json!({
        "context": result.context,
        "note_ids": result.note_ids,
        "query_mode": result.query_mode,
        "contradiction_injected": result.contradiction_injected,
        "reranker_best_score": result.reranker_best_score,
    });
    serde_json::to_string(&response)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
}

pub async fn handle_run_dreaming(
    karta: &karta_core::Karta,
    params: RunDreamingParams,
) -> Result<String, rmcp::ErrorData> {
    let scope_type = params.scope_type.as_deref().unwrap_or("workspace");
    let scope_id = params.scope_id.as_deref().unwrap_or("default");
    let result = karta
        .run_dreaming(scope_type, scope_id)
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
    let response = json!({
        "dreams_attempted": result.dreams_attempted,
        "dreams_written": result.dreams_written,
        "notes_inspected": result.notes_inspected,
        "total_tokens_used": result.total_tokens_used,
    });
    serde_json::to_string(&response)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
}

pub async fn handle_session_start(
    karta: &karta_core::Karta,
    params: SessionStartParams,
) -> Result<String, rmcp::ErrorData> {
    if params.agent.trim().is_empty() {
        return Err(rmcp::ErrorData::invalid_params(
            "agent must not be empty",
            None,
        ));
    }
    let (session_id, orientation_context) =
        session::session_start(&params.agent, params.project.as_deref(), karta)
            .await
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
    let response = json!({
        "session_id": session_id,
        "orientation_context": orientation_context,
    });
    serde_json::to_string(&response)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
}

pub async fn handle_session_end(
    karta: &karta_core::Karta,
    params: SessionEndParams,
) -> Result<String, rmcp::ErrorData> {
    if params.session_id.trim().is_empty() {
        return Err(rmcp::ErrorData::invalid_params(
            "session_id must not be empty",
            None,
        ));
    }
    let written_note_id =
        session::session_end(&params.session_id, params.summary.as_deref(), karta)
            .await
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
    let response = json!({"written_note_id": written_note_id});
    serde_json::to_string(&response)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
}

pub async fn handle_consolidate(
    karta: &karta_core::Karta,
    params: ConsolidateParams,
) -> Result<String, rmcp::ErrorData> {
    let promoted_count = session::consolidate(params.session_id.as_deref(), karta)
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
    let response = json!({"promoted_count": promoted_count});
    serde_json::to_string(&response)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
}

pub async fn handle_status(
    karta: &karta_core::Karta,
    queue: Arc<CaptureQueue>,
    store_dir: &str,
    embedding_model: &str,
    capture_port: u16,
) -> Result<String, rmcp::ErrorData> {
    let note_count = karta
        .note_count()
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
    let queue_depth = queue
        .depth()
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
    let response = json!({
        "note_count": note_count,
        "store_dir": store_dir,
        "embedding_model": embedding_model,
        "capture_port": capture_port,
        "queue_depth": queue_depth,
    });
    serde_json::to_string(&response)
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use karta_core::Karta;
    use karta_core::llm::LlmProvider;
    use karta_core::llm::MockLlmProvider;
    use karta_core::store::sqlite::SqliteGraphStore;
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    use karta_core::store::{GraphStore, VectorStore};
    use serde_json::Value;
    use tempfile::TempDir;

    use super::*;

    const EMBEDDING_DIM: usize = 1536;

    async fn setup_karta() -> (TempDir, Karta) {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        let vector_store = SqliteVectorStore::new(data_dir, EMBEDDING_DIM)
            .await
            .unwrap();
        let shared_conn = vector_store.connection();
        let vector_store: Arc<dyn VectorStore> = Arc::new(vector_store);
        let graph_store: Arc<dyn GraphStore> =
            Arc::new(SqliteGraphStore::with_connection(shared_conn));
        let llm: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new());
        let config = karta_core::config::KartaConfig::default();
        let karta = Karta::new(vector_store, graph_store, llm, config)
            .await
            .unwrap();
        (dir, karta)
    }

    async fn setup_queue(data_dir: &str) -> Arc<CaptureQueue> {
        Arc::new(CaptureQueue::new(data_dir).await.unwrap())
    }

    fn parse_json(s: &str) -> Value {
        serde_json::from_str(s).expect("tool response should be valid JSON")
    }

    #[tokio::test]
    async fn add_note_creates_note_and_returns_id() {
        let (_dir, karta) = setup_karta().await;
        let params = AddNoteParams {
            content: "hello world".to_string(),
            session_id: Some("s1".to_string()),
            turn_index: Some(3),
        };
        let response = handle_add_note(&karta, params).await.unwrap();
        let value = parse_json(&response);
        assert_eq!(value["status"], "ok");
        assert!(
            value["note_id"].as_str().unwrap().len() > 0,
            "note_id should be non-empty"
        );
        assert_eq!(karta.note_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn add_note_rejects_empty_content() {
        let (_dir, karta) = setup_karta().await;
        let params = AddNoteParams {
            content: "   ".to_string(),
            session_id: None,
            turn_index: None,
        };
        let err = handle_add_note(&karta, params).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn fetch_memories_returns_empty_shape() {
        let (_dir, karta) = setup_karta().await;
        let params = FetchMemoriesParams {
            query: "karta".to_string(),
            top_k: None,
        };
        let response = handle_fetch_memories(&karta, params).await.unwrap();
        let value = parse_json(&response);
        assert!(value["context"].is_string());
        assert!(value["note_ids"].is_array());
        assert!(value["query_mode"].is_string());
        assert!(value["contradiction_injected"].is_number());
    }

    #[tokio::test]
    async fn fetch_memories_returns_just_added_note() {
        let (_dir, karta) = setup_karta().await;
        let add_params = AddNoteParams {
            content: "the quick brown fox jumps over the lazy dog".to_string(),
            session_id: None,
            turn_index: None,
        };
        let response = handle_add_note(&karta, add_params).await.unwrap();
        let note_id = parse_json(&response)["note_id"]
            .as_str()
            .unwrap()
            .to_string();

        let params = FetchMemoriesParams {
            query: "fox".to_string(),
            top_k: Some(5),
        };
        let response = handle_fetch_memories(&karta, params).await.unwrap();
        let value = parse_json(&response);
        let note_ids: Vec<String> = value["note_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(
            note_ids.contains(&note_id),
            "note_id should appear in results"
        );
    }

    #[tokio::test]
    async fn run_dreaming_returns_documented_shape() {
        let (_dir, karta) = setup_karta().await;
        let params = RunDreamingParams {
            scope_type: None,
            scope_id: None,
        };
        let response = handle_run_dreaming(&karta, params).await.unwrap();
        let value = parse_json(&response);
        assert!(value["dreams_attempted"].is_number());
        assert!(value["dreams_written"].is_number());
        assert!(value["notes_inspected"].is_number());
        assert!(value["total_tokens_used"].is_number());
    }

    #[tokio::test]
    async fn session_start_returns_session_and_orientation() {
        let (_dir, karta) = setup_karta().await;
        let params = SessionStartParams {
            agent: "droid".to_string(),
            project: Some("karta-mcp".to_string()),
        };
        let response = handle_session_start(&karta, params).await.unwrap();
        let value = parse_json(&response);
        assert!(
            value["session_id"].as_str().unwrap().len() > 0,
            "session_id should be non-empty"
        );
        assert!(value["orientation_context"].is_string());
    }

    #[tokio::test]
    async fn session_start_rejects_empty_agent() {
        let (_dir, karta) = setup_karta().await;
        let params = SessionStartParams {
            agent: "   ".to_string(),
            project: None,
        };
        let err = handle_session_start(&karta, params).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn session_end_writes_marker_and_returns_id() {
        let (_dir, karta) = setup_karta().await;
        let start_params = SessionStartParams {
            agent: "droid".to_string(),
            project: None,
        };
        let start_response = handle_session_start(&karta, start_params).await.unwrap();
        let session_id = parse_json(&start_response)["session_id"]
            .as_str()
            .unwrap()
            .to_string();

        let end_params = SessionEndParams {
            session_id: session_id.clone(),
            summary: Some("wrapped up".to_string()),
        };
        let response = handle_session_end(&karta, end_params).await.unwrap();
        let value = parse_json(&response);
        let written_note_id = value["written_note_id"].as_str().unwrap().to_string();
        assert!(!written_note_id.is_empty());

        let note = karta.get_note(&written_note_id).await.unwrap().unwrap();
        assert!(note.content.contains("wrapped up"));
        assert_eq!(note.session_id, Some(session_id));
    }

    #[tokio::test]
    async fn consolidate_counts_high_confidence_notes() {
        let (_dir, karta) = setup_karta().await;
        let add_params = AddNoteParams {
            content: "a very confident observation".to_string(),
            session_id: Some("session-a".to_string()),
            turn_index: None,
        };
        handle_add_note(&karta, add_params).await.unwrap();

        let params = ConsolidateParams {
            session_id: Some("session-a".to_string()),
        };
        let response = handle_consolidate(&karta, params).await.unwrap();
        let value = parse_json(&response);
        assert!(value["promoted_count"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn status_returns_all_fields() {
        let (dir, karta) = setup_karta().await;
        let queue = setup_queue(dir.path().to_str().unwrap()).await;
        let response = handle_status(&karta, queue, dir.path().to_str().unwrap(), "mock", 3137)
            .await
            .unwrap();
        let value = parse_json(&response);
        assert!(value["note_count"].is_number());
        assert!(value["store_dir"].is_string());
        assert!(value["embedding_model"].is_string());
        assert!(value["capture_port"].is_number());
        assert!(value["queue_depth"].is_number());
    }
}
