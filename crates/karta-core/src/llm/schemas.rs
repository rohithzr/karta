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
                    "minItems": 0,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Single sentence stating one durable claim. Use ordinary-world language. Do not store benchmark or conversation jargon (e.g. 'time anchor', 'assistant', 'memory')."
                            },
                            "memory_kind": {
                                "type": "string",
                                "enum": [
                                    "durable_fact", "future_commitment", "preference",
                                    "decision", "constraint",
                                    "ephemeral_request", "speech_act", "echo"
                                ],
                                "description": "Admission classification. Use ephemeral_request for help-seeking turns, speech_act for greetings/acks, echo for restating the assistant. The validator drops these three."
                            },
                            "supporting_spans": {
                                "type": "array",
                                "items": { "type": "string" },
                                "minItems": 1,
                                "maxItems": 3,
                                "description": "1-3 verbatim substrings copied from the source MESSAGE that justify this fact. Each must be ≥4 characters. The validator rejects spans that aren't real substrings of the source message — do not paraphrase."
                            },
                            "facet": {
                                "type": "string",
                                "enum": [
                                    "deadline", "target_date", "preference", "tech_stack",
                                    "location", "ownership", "constraint", "event", "unknown"
                                ],
                                "description": "What aspect of the entity this fact describes. Use 'unknown' only when none apply."
                            },
                            "entity_type": {
                                "type": "string",
                                "enum": ["user", "project", "person", "org", "task", "unknown"],
                                "description": "Coarse type of the entity being described."
                            },
                            "entity_text": {
                                "type": ["string", "null"],
                                "description": "Surface form of the entity. Prefer the most specific name available in the message ('Coco', 'budget tracker'). Use 'project'/'user' only when no specific name appears."
                            },
                            "value_text": {
                                "type": ["string", "null"],
                                "description": "String value slot for the facet ('Flask 2.3.1', 'vegetarian')."
                            },
                            "value_date": {
                                "type": ["string", "null"],
                                "format": "date-time",
                                "description": "Date value slot for date-shaped facets like deadline / target_date. Distinct from occurred_* (which describes when the fact's event occurred)."
                            },
                            "occurred_start": {
                                "type": ["string", "null"],
                                "format": "date-time",
                                "description": "Inclusive lower bound of when the asserted event occurred. Most facts have null bounds. Use only when the fact text explicitly references a time."
                            },
                            "occurred_end": {
                                "type": ["string", "null"],
                                "format": "date-time",
                                "description": "Exclusive upper bound. Use start + 1 nanosecond for instants, next day for date-only references."
                            },
                            "occurred_confidence": {
                                "type": "number",
                                "enum": [0.0, 0.5, 0.7, 0.8, 1.0],
                                "description": "Discrete temporal-bound confidence. 0.0 paired with null bounds; 1.0 explicit ISO date; 0.8 NL absolute; 0.7 relative reference; 0.5 vague temporal word."
                            }
                        },
                        "required": [
                            "content", "memory_kind", "supporting_spans",
                            "facet", "entity_type", "entity_text",
                            "value_text", "value_date",
                            "occurred_start", "occurred_end", "occurred_confidence"
                        ],
                        "additionalProperties": false
                    },
                    "description": "Zero or more atomic facts. Empty array is the correct output when the message is a question, greeting, or pure speech act with no durable claim."
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
