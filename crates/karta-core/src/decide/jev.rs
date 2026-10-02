//! TypeSafe Jev decider — System One API (closed-set, calibrated decisions).
//!
//! Jev answers choice / yes-no ("noul") / score questions about a state in a
//! single forward pass and never generates text. It is opt-in: construct it
//! explicitly or via [`JevDecider::from_env`] (`JEV_API_KEY`, optional
//! `JEV_BASE_URL`, `JEV_MODEL`).
//!
//! WIRE FORMAT IS UNVERIFIED. Jev launched in early access on 2026-09-15 and
//! the public write-ups disagree on details (context window, option limits,
//! response field names). The request/response mapping is isolated in
//! `to_request` / `from_response` so it can be corrected in one place once we
//! have API access; the response parser accepts the common field-name
//! variants seen in third-party docs. Run the gated `decider_jev_live` test
//! against the real API before relying on this backend.

use async_trait::async_trait;

use super::{validate, Decider, Decision, Decisions, Question, QuestionKind};
use crate::error::{KartaError, Result};

/// Default endpoint (TypeSafe direct). OpenRouter also proxies Jev at
/// `https://openrouter.ai/api/v1/systemone` with an OpenRouter key.
pub const DEFAULT_JEV_URL: &str = "https://api.typesafe.ai/v1/systemone";
pub const DEFAULT_JEV_MODEL: &str = "jev-latest";
/// Jev's native score scale is 0–5; other `max` values are rescaled.
const JEV_SCORE_MAX: f32 = 5.0;

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

pub(crate) fn to_request(model: &str, state: &str, questions: &[Question]) -> serde_json::Value {
    let mut qmap = serde_json::Map::new();
    for q in questions {
        let body = match &q.kind {
            QuestionKind::Choice { options } => serde_json::json!({
                "type": "choice",
                "instructions": q.instructions,
                "options": options
                    .iter()
                    .map(|(l, d)| serde_json::json!({ "label": l, "description": d }))
                    .collect::<Vec<_>>(),
            }),
            QuestionKind::YesNo => serde_json::json!({
                "type": "noul",
                "instructions": q.instructions,
            }),
            QuestionKind::Score { .. } => serde_json::json!({
                "type": "score",
                "instructions": q.instructions,
            }),
        };
        qmap.insert(q.name.clone(), body);
    }
    serde_json::json!({ "model": model, "state": state, "questions": qmap })
}

fn first_f32(v: &serde_json::Value, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_f64())).map(|x| x as f32)
}

pub(crate) fn from_response(questions: &[Question], body: &serde_json::Value) -> Result<Decisions> {
    let answers = ["answers", "decisions", "results"]
        .iter()
        .find_map(|k| body.get(*k).filter(|v| v.is_object()))
        .ok_or_else(|| KartaError::Llm(format!("Jev: no answers object in response: {}", body)))?;

    let mut out = Decisions::new();
    for q in questions {
        let Some(a) = answers.get(&q.name) else { continue };
        let d = match &q.kind {
            QuestionKind::Choice { options } => {
                let label = ["choice", "selected", "option", "label"]
                    .iter()
                    .find_map(|k| a.get(*k).and_then(|x| x.as_str()))
                    .unwrap_or_default()
                    .to_string();
                let probs_obj = ["probabilities", "probs", "distribution"]
                    .iter()
                    .find_map(|k| a.get(*k));
                let probs: Vec<(String, f32)> = options
                    .iter()
                    .map(|(l, _)| {
                        let p = probs_obj
                            .and_then(|p| p.get(l))
                            .and_then(|x| x.as_f64())
                            .map(|x| x as f32)
                            .unwrap_or(if *l == label { 1.0 } else { 0.0 });
                        (l.clone(), p)
                    })
                    .collect();
                let confidence = first_f32(a, &["confidence"]).unwrap_or_else(|| {
                    probs.iter().find(|(l, _)| *l == label).map(|(_, p)| *p).unwrap_or(0.0)
                });
                Decision::Choice { label, probs, confidence }
            }
            QuestionKind::YesNo => match first_f32(a, &["probability", "p_yes", "p", "yes"]) {
                Some(p_yes) => Decision::YesNo { p_yes },
                None => continue,
            },
            QuestionKind::Score { max } => match first_f32(a, &["value", "score", "position"]) {
                Some(v) => Decision::Score {
                    value: v * (*max as f32) / JEV_SCORE_MAX,
                    confidence: first_f32(a, &["confidence"]).unwrap_or(0.0),
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
            .json(&to_request(&self.model, state, questions))
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

    fn qs() -> Vec<Question> {
        vec![
            Question::choice(
                "mode",
                "Which mode?",
                vec![("a".into(), "first".into()), ("b".into(), "second".into())],
            ),
            Question::yes_no("temporal", "Is it about time?"),
            Question::score("relevance", "How relevant?", 10),
        ]
    }

    #[test]
    fn request_maps_question_kinds() {
        let r = to_request("jev-latest", "state", &qs());
        assert_eq!(r["model"], "jev-latest");
        assert_eq!(r["questions"]["mode"]["type"], "choice");
        assert_eq!(r["questions"]["mode"]["options"][1]["label"], "b");
        assert_eq!(r["questions"]["temporal"]["type"], "noul");
        assert_eq!(r["questions"]["relevance"]["type"], "score");
    }

    #[test]
    fn response_parses_probabilities_and_rescales_score() {
        let body = serde_json::json!({
            "answers": {
                "mode": { "choice": "b", "probabilities": { "a": 0.2, "b": 0.8 }, "confidence": 0.8 },
                "temporal": { "probability": 0.9 },
                "relevance": { "value": 2.5, "confidence": 0.6 }
            }
        });
        let out = from_response(&qs(), &body).unwrap();
        validate(&qs(), &out).unwrap();
        assert_eq!(out["mode"].label(), Some("b"));
        assert!((out["mode"].confidence() - 0.8).abs() < 1e-6);
        assert_eq!(out["temporal"], Decision::YesNo { p_yes: 0.9 });
        // 2.5 on Jev's 0–5 scale is 5.0 on a 0–10 question.
        assert_eq!(out["relevance"], Decision::Score { value: 5.0, confidence: 0.6 });
    }

    #[test]
    fn response_without_answers_object_is_an_error() {
        assert!(from_response(&qs(), &serde_json::json!({ "error": "x" })).is_err());
    }
}
