//! JSON schemas for structured output. Used with providers that support
//! native structured output (OpenAI json_schema response_format).

use super::traits::JsonSchema;
use serde_json::json;

/// Schema for synthesis/ask responses — the core retrieval answer.
/// Forces reasoning before answering, with explicit abstention decision.
pub fn synthesis_schema() -> JsonSchema {
    JsonSchema {
        name: "synthesis_response".to_string(),
        schema: json!({
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning: 1) What topic does the question ask about? 2) Do ANY of the notes discuss this topic, even partially? 3) Are there contradictions? 4) What dates/times are relevant? Think through this before answering."
                },
                "should_abstain": {
                    "type": "boolean",
                    "description": "true ONLY if the notes are about a completely different topic than the question. If the notes contain ANY relevant information — even partial, indirect, or requiring synthesis — set to false and answer. Examples where should_abstain=false: notes mention a number the question asks about, notes discuss a related feature, notes contain facts that can be combined to answer. Examples where should_abstain=true: question asks about personal history but notes only discuss a project, question asks about a person never mentioned."
                },
                "has_contradiction": {
                    "type": "boolean",
                    "description": "true if the notes contain contradictory information relevant to the question."
                },
                "answer": {
                    "type": ["string", "null"],
                    "description": "The answer to the question based on the notes. null if should_abstain is true. Include code blocks when the notes contain code and the question asks for implementation details."
                },
                "cited_notes": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "description": "List of note numbers (1-indexed) that informed the answer."
                }
            },
            "required": ["reasoning", "should_abstain", "has_contradiction", "answer", "cited_notes"],
            "additionalProperties": false
        }),
    }
}

/// Schema for note attribute extraction.
pub fn note_attributes_schema() -> JsonSchema {
    JsonSchema {
        name: "note_attributes".to_string(),
        schema: json!({
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief analysis of what this note is about, what entities/dates/decisions it contains, and why it matters."
                },
                "context": {
                    "type": "string",
                    "description": "A rich 1-2 sentence description capturing deeper meaning, implications, and why this matters. Include any specific dates or deadlines mentioned."
                },
                "keywords": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "5 to 8 specific terms that would help find this note."
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "3 to 5 categorical labels: preference, decision, constraint, workflow, entity, pattern, temporal, code."
                },
                "foresight_signals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "The forward-looking statement including any time reference." },
                            "valid_until": { "type": ["string", "null"], "description": "ISO 8601 date (YYYY-MM-DD) when this prediction/deadline expires or becomes irrelevant. null if no time reference." }
                        },
                        "required": ["content", "valid_until"],
                        "additionalProperties": false
                    },
                    "description": "Forward-looking statements: deadlines, scheduled events, plans, predictions. Empty array if none."
                }
            },
            "required": ["reasoning", "context", "keywords", "tags", "foresight_signals"],
            "additionalProperties": false
        }),
    }
}

/// Schema for link decisions.
pub fn link_decision_schema() -> JsonSchema {
    JsonSchema {
        name: "link_decisions".to_string(),
        schema: json!({
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief analysis of which candidates have meaningful relationships with the new note and why."
                },
                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "noteId": { "type": "string" },
                            "reason": { "type": "string" }
                        },
                        "required": ["noteId", "reason"],
                        "additionalProperties": false
                    },
                    "description": "Notes to link. Only include meaningful relationships: same entity, causal, shared constraint, complementary context."
                }
            },
            "required": ["reasoning", "links"],
            "additionalProperties": false
        }),
    }
}
