//! TypeSafe Jev decider — System One API (closed-set, calibrated decisions).
//!
//! Jev answers choice / yes-no ("noul") / score questions about a state in a
//! single forward pass and never generates text. It is opt-in: construct it
//! explicitly or via [`JevDecider::from_env`] (`JEV_API_KEY`, optional
//! `JEV_BASE_URL`, `JEV_MODEL`).
//!
//! Wire format verified against the live API (model `jev-1.13.0`, 2026-10-02):
//!
//! ```text
//! POST /v1/systemone
//! { "model": "jev-latest", "state": "...",
//!   "questions": {
//!     "<name>": { "type": "choice", "instructions": "...", "criteria": { "<label>": "<description>", ... } },
//!     "<name>": { "type": "noul",   "instructions": "..." },
//!     "<name>": { "type": "score",  "instructions": "...", "criteria": ["<level 0>", "<level 1>", ...] } } }
//! ->
//! { "model": "jev-1.13.0",
//!   "answers": {
//!     "<name>": { "type": "choice", "choice": "<label>", "confidence": 0.95, "probabilities": { "<label>": 0.97, ... } },
//!     "<name>": { "type": "noul",   "noul": 0.05 },
//!     "<name>": { "type": "score",  "score": 1.6, "confidence": 0.0, "legend": {...}, "probabilities": {...} } },
//!   "usage": { "input_tokens": 396, "output_tokens": 63 } }
//! ```
//!
//! Score questions take at most 10 levels; the score is on `0..=levels-1`.
//! Choice handled 20 options without complaint. Choice criteria are sent as a
//! JSON object, so option order on the wire is serde_json's key order
//! (alphabetical) — stable across calls, but not the order in `Question`.

use async_trait::async_trait;

use super::{validate, Decider, Decision, Decisions, Question, QuestionKind};
use crate::error::{KartaError, Result};

/// Default endpoint (TypeSafe direct). OpenRouter also proxies Jev at
/// `https://openrouter.ai/api/v1/systemone` with an OpenRouter key.
pub const DEFAULT_JEV_URL: &str = "https://api.typesafe.ai/v1/systemone";
pub const DEFAULT_JEV_MODEL: &str = "jev-latest";
/// API limit on score levels.
pub const MAX_SCORE_LEVELS: usize = 10;

pub struct JevDecider {
    api_key: String,
    url: String,
    model: String,
    client: reqwest::Client,
}

impl JevDecider {
    pub fn new(api_key: &str) -> Self {
        Self::with_endpoint(api_key, DEFAULT_JEV_URL, DEFAULT_JEV_MODEL)
    }

    pub fn with_endpoint(api_key: &str, url: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            url: url.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Build from `JEV_API_KEY` (required), `JEV_BASE_URL`, `JEV_MODEL`.
    /// Returns `None` when no key is set so callers can fall back.
    pub fn from_env() -> Option<Self> {
        let key = std::env::var("JEV_API_KEY").ok().filter(|k| !k.is_empty())?;
        let url = std::env::var("JEV_BASE_URL").unwrap_or_else(|_| DEFAULT_JEV_URL.to_string());
        let model = std::env::var("JEV_MODEL").unwrap_or_else(|_| DEFAULT_JEV_MODEL.to_string());
        Some(Self::with_endpoint(&key, &url, &model))
    }
}

pub(crate) fn to_request(model: &str, state: &str, questions: &[Question]) -> Result<serde_json::Value> {
    let mut qmap = serde_json::Map::new();
    for q in questions {
        let body = match &q.kind {
            QuestionKind::Choice { options } => {
                let criteria: serde_json::Map<String, serde_json::Value> = options
                    .iter()
                    .map(|(l, d)| (l.clone(), serde_json::Value::String(d.clone())))
                    .collect();
                serde_json::json!({
                    "type": "choice",
                    "instructions": q.instructions,
                    "criteria": criteria,
                })
            }
            QuestionKind::YesNo => serde_json::json!({
                "type": "noul",
                "instructions": q.instructions,
            }),
            QuestionKind::Score { levels } => {
                if levels.len() < 2 || levels.len() > MAX_SCORE_LEVELS {
                    return Err(KartaError::Config(format!(
                        "Jev score question '{}' needs 2..={} levels, got {}",
                        q.name,
                        MAX_SCORE_LEVELS,
                        levels.len()
                    )));
                }
                serde_json::json!({
                    "type": "score",
                    "instructions": q.instructions,
                    "criteria": levels,
                })
            }
        };
        qmap.insert(q.name.clone(), body);
    }
    Ok(serde_json::json!({ "model": model, "state": state, "questions": qmap }))
}

fn get_f32(v: &serde_json::Value, key: &str) -> Option<f32> {
    v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32)
}

pub(crate) fn from_response(questions: &[Question], body: &serde_json::Value) -> Result<Decisions> {
    let answers = body
        .get("answers")
        .filter(|v| v.is_object())
        .ok_or_else(|| KartaError::Llm(format!("Jev: no answers object in response: {}", body)))?;

    let mut out = Decisions::new();
    for q in questions {
        let Some(a) = answers.get(&q.name) else { continue };
        let d = match &q.kind {
            QuestionKind::Choice { options } => {
                let Some(label) = a.get("choice").and_then(|x| x.as_str()) else { continue };
                let probs: Vec<(String, f32)> = options
                    .iter()
                    .map(|(l, _)| {
                        let p = a
                            .get("probabilities")
                            .and_then(|p| get_f32(p, l))
                            .unwrap_or(0.0);
                        (l.clone(), p)
                    })
                    .collect();
                Decision::Choice {
                    label: label.to_string(),
                    probs,
                    confidence: get_f32(a, "confidence").unwrap_or(0.0),
                }
            }
            QuestionKind::YesNo => match get_f32(a, "noul") {
                Some(p_yes) => Decision::YesNo { p_yes },
                None => continue,
            },
            QuestionKind::Score { .. } => match get_f32(a, "score") {
                Some(value) => Decision::Score {
                    value,
                    confidence: get_f32(a, "confidence").unwrap_or(0.0),
                },
                None => continue,
            },
        };
        out.insert(q.name.clone(), d);
    }
    Ok(out)
}

#[async_trait]
impl Decider for JevDecider {
    async fn decide(&self, state: &str, questions: &[Question]) -> Result<Decisions> {
        if questions.is_empty() {
            return Ok(Decisions::new());
        }
        let response = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&to_request(&self.model, state, questions)?)
            .send()
            .await
            .map_err(|e| KartaError::Llm(format!("Jev request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(KartaError::Llm(format!("Jev API error {}: {}", status, text)));
        }
        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| KartaError::Llm(format!("Jev response parse failed: {}", e)))?;

        let out = from_response(questions, &body)?;
        validate(questions, &out)?;
        Ok(out)
    }

    fn id(&self) -> String {
        self.model.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn levels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("level {}", i)).collect()
    }

    fn qs() -> Vec<Question> {
        vec![
            Question::choice(
                "mode",
                "Which mode?",
                vec![("a".into(), "first".into()), ("b".into(), "second".into())],
            ),
            Question::yes_no("temporal", "Is it about time?"),
            Question::score("relevance", "How relevant?", levels(6)),
        ]
    }

    #[test]
    fn request_matches_live_wire_format() {
        let r = to_request("jev-latest", "state", &qs()).unwrap();
        assert_eq!(r["model"], "jev-latest");
        assert_eq!(r["state"], "state");
        assert_eq!(r["questions"]["mode"]["type"], "choice");
        assert_eq!(r["questions"]["mode"]["criteria"]["b"], "second");
        assert_eq!(r["questions"]["temporal"]["type"], "noul");
        assert_eq!(r["questions"]["relevance"]["type"], "score");
        assert_eq!(r["questions"]["relevance"]["criteria"][5], "level 5");
    }

    #[test]
    fn request_rejects_too_many_score_levels() {
        let q = [Question::score("s", "?", levels(MAX_SCORE_LEVELS + 1))];
        assert!(to_request("jev-latest", "x", &q).is_err());
    }

    /// Body captured from the live API (jev-1.13.0), names adapted.
    #[test]
    fn response_parses_live_shape() {
        let body = serde_json::json!({
            "model": "jev-1.13.0",
            "answers": {
                "mode": { "type": "choice", "choice": "b", "confidence": 0.95,
                          "probabilities": { "a": 0.03, "b": 0.97 } },
                "temporal": { "type": "noul", "noul": 0.05 },
                "relevance": { "type": "score", "score": 1.6, "confidence": 0.0,
                               "legend": { "0": "level 0" },
                               "probabilities": { "0": 0.34, "1": 0.16 } }
            },
            "usage": { "input_tokens": 396, "output_tokens": 63 }
        });
        let out = from_response(&qs(), &body).unwrap();
        validate(&qs(), &out).unwrap();
        assert_eq!(out["mode"].label(), Some("b"));
        assert!((out["mode"].confidence() - 0.95).abs() < 1e-6);
        assert_eq!(
            out["mode"],
            Decision::Choice {
                label: "b".into(),
                probs: vec![("a".into(), 0.03), ("b".into(), 0.97)],
                confidence: 0.95,
            }
        );
        assert_eq!(out["temporal"], Decision::YesNo { p_yes: 0.05 });
        assert_eq!(out["relevance"], Decision::Score { value: 1.6, confidence: 0.0 });
    }

    #[test]
    fn response_without_answers_object_is_an_error() {
        let err = serde_json::json!({ "detail": "Too many score levels." });
        assert!(from_response(&qs(), &err).is_err());
    }
}
