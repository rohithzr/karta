use std::sync::Arc;
use serde_json::{json, Value};
use karta_core::Karta;
use tracing::error;

/// Return the JSON array of tool schemas for `tools/list`.
pub fn tool_schemas() -> Value {
    json!({
        "tools": [
            {
                "name": "karta_add_note",
                "description": "Ingest a piece of knowledge into Karta. Triggers LLM attribute extraction, embedding, ANN linking, retroactive evolution, and atomic fact decomposition.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The text to ingest as a memory note"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID to group notes into episodes for thematic clustering"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "karta_ask",
                "description": "Query Karta with full synthesis. Classifies query mode, does ANN search, episode drilldown, fact retrieval, reranking, multi-hop graph traversal, and LLM synthesis.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to ask"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max notes to consider (default 5)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "karta_search",
                "description": "Raw retrieval without synthesis. Returns scored notes directly.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results (default 5)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "karta_dream",
                "description": "Run background reasoning over the knowledge graph. Produces new inferred notes via deduction, induction, abduction, consolidation, contradiction detection, and episode digests.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "scope_type": {
                            "type": "string",
                            "description": "Dream scope type (default 'workspace')"
                        },
                        "scope_id": {
                            "type": "string",
                            "description": "Scope identifier (default 'default')"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "karta_get_note",
                "description": "Retrieve a single note by ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Note UUID"
                        }
                    },
                    "required": ["id"]
                }
            },
            {
                "name": "karta_get_links",
                "description": "Get all note IDs linked to a given note.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "note_id": {
                            "type": "string",
                            "description": "Note UUID"
                        }
                    },
                    "required": ["note_id"]
                }
            },
            {
                "name": "karta_note_count",
                "description": "Total number of notes in the knowledge graph.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    })
}

/// Format a MemoryNote as JSON for tool responses (omit embedding, evolution_history).
fn format_note(note: &karta_core::note::MemoryNote) -> Value {
    json!({
        "id": note.id,
        "content": note.content,
        "context": note.context,
        "keywords": note.keywords,
        "tags": note.tags,
        "links": note.links,
        "provenance": serde_json::to_value(&note.provenance).unwrap(),
        "confidence": note.confidence,
        "status": serde_json::to_value(&note.status).unwrap(),
        "created_at": note.created_at.to_rfc3339(),
        "updated_at": note.updated_at.to_rfc3339(),
    })
}

/// Dispatch a `tools/call` request to the appropriate Karta method.
pub async fn dispatch(karta: &Arc<Karta>, name: &str, args: &Value) -> Value {
    match name {
        "karta_add_note" => {
            let content = match args["content"].as_str() {
                Some(c) => c,
                None => return tool_error("content is required"),
            };
            let result = match args["session_id"].as_str() {
                Some(sid) => karta.add_note_with_session(content, sid).await,
                None => karta.add_note(content).await,
            };
            match result {
                Ok(note) => tool_ok(json!({
                    "note_id": note.id,
                    "context": note.context,
                    "keywords": note.keywords,
                    "tags": note.tags,
                    "links_count": note.links.len(),
                })),
                Err(e) => {
                    error!(error = %e, "add_note failed");
                    tool_error(&format!("add_note failed: {e}"))
                }
            }
        }

        "karta_ask" => {
            let query = match args["query"].as_str() {
                Some(q) => q,
                None => return tool_error("query is required"),
            };
            let top_k = args["top_k"].as_u64().unwrap_or(5) as usize;
            match karta.ask(query, top_k).await {
                Ok(result) => tool_ok(json!({
                    "answer": result.answer,
                    "query_mode": result.query_mode,
                    "notes_used": result.notes_used,
                    "note_ids": result.note_ids,
                    "has_contradiction": result.has_contradiction,
                    "contradiction_injected": result.contradiction_injected,
                })),
                Err(e) => {
                    error!(error = %e, "ask failed");
                    tool_error(&format!("ask failed: {e}"))
                }
            }
        }

        "karta_search" => {
            let query = match args["query"].as_str() {
                Some(q) => q,
                None => return tool_error("query is required"),
            };
            let top_k = args["top_k"].as_u64().unwrap_or(5) as usize;
            match karta.search(query, top_k).await {
                Ok(results) => {
                    let notes: Vec<Value> = results
                        .iter()
                        .map(|r| {
                            let mut v = format_note(&r.note);
                            v.as_object_mut().unwrap().insert(
                                "score".to_string(),
                                json!(r.score),
                            );
                            v
                        })
                        .collect();
                    tool_ok(json!({ "results": notes }))
                }
                Err(e) => {
                    error!(error = %e, "search failed");
                    tool_error(&format!("search failed: {e}"))
                }
            }
        }

        "karta_dream" => {
            let scope_type = args["scope_type"].as_str().unwrap_or("workspace");
            let scope_id = args["scope_id"].as_str().unwrap_or("default");
            match karta.run_dreaming(scope_type, scope_id).await {
                Ok(run) => {
                    let types: Vec<String> = run
                        .dreams
                        .iter()
                        .filter(|d| d.would_write)
                        .map(|d| d.dream_type.as_str().to_string())
                        .collect();
                    tool_ok(json!({
                        "dreams_attempted": run.dreams_attempted,
                        "dreams_written": run.dreams_written,
                        "notes_inspected": run.notes_inspected,
                        "types_produced": types,
                        "total_tokens_used": run.total_tokens_used,
                    }))
                }
                Err(e) => {
                    error!(error = %e, "dream failed");
                    tool_error(&format!("dream failed: {e}"))
                }
            }
        }

        "karta_get_note" => {
            let id = match args["id"].as_str() {
                Some(id) => id,
                None => return tool_error("id is required"),
            };
            match karta.get_note(id).await {
                Ok(Some(note)) => tool_ok(json!({ "note": format_note(&note) })),
                Ok(None) => tool_ok(json!({ "note": null })),
                Err(e) => {
                    error!(error = %e, "get_note failed");
                    tool_error(&format!("get_note failed: {e}"))
                }
            }
        }

        "karta_get_links" => {
            let note_id = match args["note_id"].as_str() {
                Some(id) => id,
                None => return tool_error("note_id is required"),
            };
            match karta.get_links(note_id).await {
                Ok(links) => tool_ok(json!({ "linked_note_ids": links })),
                Err(e) => {
                    error!(error = %e, "get_links failed");
                    tool_error(&format!("get_links failed: {e}"))
                }
            }
        }

        "karta_note_count" => {
            match karta.note_count().await {
                Ok(count) => tool_ok(json!({ "count": count })),
                Err(e) => {
                    error!(error = %e, "note_count failed");
                    tool_error(&format!("note_count failed: {e}"))
                }
            }
        }

        _ => tool_error(&format!("unknown tool: {name}")),
    }
}

fn tool_ok(content: Value) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&content).unwrap()
        }]
    })
}

fn tool_error(message: &str) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": message
        }],
        "isError": true
    })
}
