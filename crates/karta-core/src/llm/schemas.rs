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
                "has_contradiction": {
                    "type": "boolean",
                    "description": "true ONLY if the notes contain mutually exclusive claims that cannot both be true simultaneously (e.g., 'I use Excel' vs 'I have never used Excel'). Mere updates or corrections (e.g., 'accuracy was 78%' then 'accuracy is now 92%') are NOT contradictions. If you flag true, you MUST present BOTH sides in the answer."
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to the question based on the notes. Include code blocks when the notes contain code and the question asks for implementation details. If the notes don't answer the question, explain what information is available and what is missing."
                },
                "cited_notes": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "description": "List of note numbers (1-indexed) that informed the answer."
                }
            },
            "required": ["reasoning", "has_contradiction", "answer", "cited_notes"],
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
                },
                "atomic_facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "A single atomic, independently verifiable statement." },
                            "subject": { "type": ["string", "null"], "description": "Primary entity or topic this fact is about. null if general." },
                            "occurred_start": {
                                "type": ["string", "null"],
                                "format": "date-time",
                                "description": "Inclusive lower bound (ISO 8601 UTC) of when the asserted event occurred. Null iff occurred_end is also null. For a date-only reference use 00:00:00Z. For true instants, 1 nanosecond before occurred_end."
                            },
                            "occurred_end": {
                                "type": ["string", "null"],
                                "format": "date-time",
                                "description": "Exclusive upper bound (ISO 8601 UTC). Use the next day for a date-only reference, next month for a month reference, next quarter for a quarter reference, occurred_start + 1ns for true instants."
                            },
                            "occurred_confidence": {
                                "type": "number",
                                "enum": [0.0, 0.5, 0.7, 0.8, 1.0],
                                "description": "Discrete band: 1.0 = explicit ISO date in source; 0.8 = natural-language absolute date (e.g. 'March 15, 2024'); 0.7 = relative reference with deterministic resolution (e.g. 'yesterday', 'next Friday'); 0.5 = vague reference with range chosen (e.g. 'recently', 'around March'); 0.0 = no temporal content. Must be exactly one of these values."
                            },
                            "temporal_evidence": {
                                "type": ["string", "null"],
                                "description": "EXACT verbatim quote from `content` containing the temporal phrase that justifies the bounds (e.g. 'April 15 deadline', 'yesterday', 'in March 2024'). MUST be null if and only if all three occurred_* fields are null/0.0. The validator will reject this fact's bounds if this string does not appear in `content` — so do not summarize, do not paraphrase, copy the substring."
                            }
                        },
                        "required": ["content", "subject", "occurred_start", "occurred_end", "occurred_confidence", "temporal_evidence"],
                        "additionalProperties": false
                    },
                    "description": "1-5 atomic facts. Each is a standalone, verifiable statement. Every fact MUST emit all four occurred_* fields plus temporal_evidence explicitly. Default is null bounds + 0.0 confidence + null evidence — only emit a non-null interval when a temporal phrase appears in the fact's own content."
                }
            },
            "required": ["reasoning", "context", "keywords", "tags", "foresight_signals", "atomic_facts"],
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
