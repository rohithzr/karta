//! Tier 2 temporal resolver — LLM fallback when tier 1 punts.
//!
//! Receives query + reference_time + last 3 user turns of the session for
//! context. Output passes through `validate_resolver_output` — schema
//! violations fall through to vector-only retrieval.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::json;

use crate::llm::{ChatMessage, GenConfig, JsonSchema, LlmProvider, Role};
use crate::read::temporal::{validate_resolver_output, ConfidenceBand, Interval, RawResolverOutput};

pub struct LlmResolver {
    llm: Arc<dyn LlmProvider>,
}

pub struct ResolverContext {
    /// Last 3 user turns of the same session, oldest first.
    pub recent_turns: Vec<String>,
}

fn tier2_schema() -> JsonSchema {
    JsonSchema {
        name: "temporal_resolution".to_string(),
        schema: json!({
            "type": "object",
            "properties": {
                "occurred_start": { "type": ["string", "null"], "format": "date-time" },
                "occurred_end":   { "type": ["string", "null"], "format": "date-time" },
                "occurred_confidence": { "type": "number", "enum": [0.0, 0.5, 0.7, 0.8, 1.0] }
            },
            "required": ["occurred_start", "occurred_end", "occurred_confidence"],
            "additionalProperties": false
        }),
    }
}

fn system_prompt() -> &'static str {
    r#"Resolve the user's temporal query into a half-open interval [occurred_start, occurred_end) and a discrete confidence band.

occurred_confidence MUST be one of {0.0, 0.5, 0.7, 0.8, 1.0}:
- 1.0: explicit ISO date in the query
- 0.8: natural-language absolute date
- 0.7: relative reference with deterministic resolution (uses reference_time)
- 0.5: vague reference, best-guess range
- 0.0: no temporal content (both bounds null)

Rules:
- Both bounds null AND confidence = 0.0, OR both bounds Some AND confidence in {0.5, 0.7, 0.8, 1.0}.
- occurred_end must be strictly greater than occurred_start.
- Date-only references: [day 00:00:00Z, next day 00:00:00Z).
- Respond with JSON only."#
}

fn user_prompt(query: &str, reference_time: DateTime<Utc>, ctx: &ResolverContext) -> String {
    let recent = if ctx.recent_turns.is_empty() {
        "(none)".to_string()
    } else {
        ctx.recent_turns
            .iter()
            .enumerate()
            .map(|(i, t)| format!("  {}: {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "Reference time: {}\n\nRecent user turns (context, oldest first):\n{}\n\nResolve: {}",
        reference_time.to_rfc3339(),
        recent,
        query
    )
}

impl LlmResolver {
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self { llm }
    }

    /// Returns the validated (Interval, ConfidenceBand), or None on any
    /// failure (timeout, parse error, schema violation, or a valid
    /// "no temporal content" resolution). Callers treat None as
    /// "fall through to vector-only retrieval".
    pub async fn resolve(
        &self,
        query: &str,
        reference_time: DateTime<Utc>,
        ctx: &ResolverContext,
    ) -> Option<(Interval, ConfidenceBand)> {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: system_prompt().to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: user_prompt(query, reference_time, ctx),
            },
        ];
        let config = GenConfig {
            max_tokens: 256,
            temperature: 0.0,
            json_mode: false,
            json_schema: Some(tier2_schema()),
        };
        let response = self.llm.chat(&messages, &config).await.ok()?;
        let parsed: serde_json::Value = serde_json::from_str(&response.content).ok()?;

        let start = parsed["occurred_start"]
            .as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        let end = parsed["occurred_end"]
            .as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        let conf = parsed["occurred_confidence"].as_f64()? as f32;

        let raw = RawResolverOutput {
            start,
            end,
            confidence_f32: conf,
        };
        let (iv, band) = validate_resolver_output(raw).ok()?;
        iv.map(|i| (i, band))
    }
}
