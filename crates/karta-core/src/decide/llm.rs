//! LLM-backed decider — the baseline every other backend is measured against.
//!
//! One structured-output call answers all questions. The schema constrains
//! each answer to the offered labels / bool / integer range, so the output is
//! as closed-set as a classifier's. An LLM gives no calibrated probabilities:
//! the chosen option gets 1.0 and yes/no maps to 1.0 or 0.0.

use async_trait::async_trait;
use std::sync::Arc;

use super::{validate, Decider, Decision, Decisions, Question, QuestionKind};
use crate::error::{KartaError, Result};
use crate::llm::{ChatMessage, GenConfig, JsonSchema, LlmProvider, Role};

pub struct LlmDecider {
    llm: Arc<dyn LlmProvider>,
}

impl LlmDecider {
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self { llm }
    }
}

fn answer_schema(questions: &[Question]) -> serde_json::Value {
    let mut props = serde_json::Map::new();
    for q in questions {
        let prop = match &q.kind {
            QuestionKind::Choice { options } => serde_json::json!({
                "type": "string",
                "enum": options.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
            }),
            QuestionKind::YesNo => serde_json::json!({ "type": "boolean" }),
            QuestionKind::Score { levels } => serde_json::json!({
                "type": "integer",
                "minimum": 0,
                "maximum": levels.len().saturating_sub(1),
            }),
        };
        props.insert(q.name.clone(), prop);
    }
    serde_json::json!({
        "type": "object",
        "properties": props,
        "required": questions.iter().map(|q| q.name.as_str()).collect::<Vec<_>>(),
        "additionalProperties": false,
    })
}

fn render_prompt(state: &str, questions: &[Question]) -> String {
    let mut out = String::from(
        "Answer each question about the input. Respond with JSON only, one key per question.\n\n",
    );
    out.push_str("Input:\n");
    out.push_str(state);
    out.push_str("\n\nQuestions:\n");
    for q in questions {
        out.push_str(&format!("- {}: {}\n", q.name, q.instructions));
        match &q.kind {
            QuestionKind::Choice { options } => {
                for (label, desc) in options {
                    out.push_str(&format!("    * {} — {}\n", label, desc));
                }
            }
            QuestionKind::YesNo => out.push_str("    (true or false)\n"),
            QuestionKind::Score { levels } => {
                for (i, desc) in levels.iter().enumerate() {
                    out.push_str(&format!("    {} = {}\n", i, desc));
                }
            }
        }
    }
    out
}

fn parse_answers(questions: &[Question], content: &str) -> Result<Decisions> {
    let parsed: serde_json::Value = serde_json::from_str(content)?;
    let mut out = Decisions::new();
    for q in questions {
        let v = &parsed[&q.name];
        let d = match &q.kind {
            QuestionKind::Choice { options } => {
                let label = v.as_str().unwrap_or_default().to_string();
                let probs = options
                    .iter()
                    .map(|(l, _)| (l.clone(), if *l == label { 1.0 } else { 0.0 }))
                    .collect();
                Decision::Choice { label, probs, confidence: 1.0 }
            }
            QuestionKind::YesNo => match v.as_bool() {
                Some(b) => Decision::YesNo { p_yes: if b { 1.0 } else { 0.0 } },
                None => continue,
            },
            QuestionKind::Score { .. } => match v.as_f64() {
                Some(n) => Decision::Score { value: n as f32, confidence: 1.0 },
                None => continue,
            },
        };
        out.insert(q.name.clone(), d);
    }
    Ok(out)
}

#[async_trait]
impl Decider for LlmDecider {
    async fn decide(&self, state: &str, questions: &[Question]) -> Result<Decisions> {
        if questions.is_empty() {
            return Ok(Decisions::new());
        }
        let messages = vec![ChatMessage {
            role: Role::User,
            content: render_prompt(state, questions),
        }];
        let config = GenConfig {
            max_tokens: 64 + 32 * questions.len() as u32,
            temperature: 0.0,
            json_mode: true,
            json_schema: Some(JsonSchema {
                name: "decisions".to_string(),
                schema: answer_schema(questions),
            }),
        };
        let response = self.llm.chat(&messages, &config).await?;
        let out = parse_answers(questions, &response.content)
            .map_err(|e| KartaError::Llm(format!("LlmDecider: bad answer JSON: {}", e)))?;
        validate(questions, &out)?;
        Ok(out)
    }

    fn id(&self) -> String {
        format!("llm:{}", self.llm.model_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qs() -> Vec<Question> {
        vec![
            Question::choice(
                "mode",
                "Which mode?",
                vec![("a".into(), "first".into()), ("b".into(), "second".into())],
            ),
            Question::yes_no("temporal", "Is it about time?"),
            Question::score(
                "relevance",
                "How relevant?",
                (0..6).map(|i| format!("level {}", i)).collect(),
            ),
        ]
    }

    #[test]
    fn schema_constrains_each_answer() {
        let s = answer_schema(&qs());
        assert_eq!(s["properties"]["mode"]["enum"], serde_json::json!(["a", "b"]));
        assert_eq!(s["properties"]["temporal"]["type"], "boolean");
        assert_eq!(s["properties"]["relevance"]["maximum"], 5);
        assert_eq!(s["required"], serde_json::json!(["mode", "temporal", "relevance"]));
    }

    #[test]
    fn parses_and_validates_answers() {
        let out = parse_answers(&qs(), r#"{"mode":"b","temporal":true,"relevance":3}"#).unwrap();
        validate(&qs(), &out).unwrap();
        assert_eq!(out["mode"].label(), Some("b"));
        assert_eq!(out["temporal"], Decision::YesNo { p_yes: 1.0 });
        assert_eq!(out["relevance"], Decision::Score { value: 3.0, confidence: 1.0 });
    }

    #[test]
    fn missing_or_unknown_answers_fail_validation() {
        let out = parse_answers(&qs(), r#"{"mode":"z","temporal":true,"relevance":3}"#).unwrap();
        assert!(validate(&qs(), &out).is_err());
        let out = parse_answers(&qs(), r#"{"mode":"a","relevance":3}"#).unwrap();
        assert!(validate(&qs(), &out).is_err());
    }
}
