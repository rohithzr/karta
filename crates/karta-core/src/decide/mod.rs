//! Decider module — typed, closed-set decisions (choice / yes-no / score).
//!
//! Many places in Karta make a small decision over a fixed label set: query
//! mode, which mutable-slot predicate a query asks about, whether two notes
//! should link, whether a context answers a question. Today these are done
//! with keyword lists, embedding centroids, or a full LLM call. A `Decider`
//! gives them one pluggable interface so a cheap "System One" classifier
//! (e.g. TypeSafe Jev) can stand in for the LLM where it is good enough.
//!
//! Architecture: trait-based, pluggable (same shape as `rerank`).
//! - LlmDecider: uses the configured LLM with a JSON schema (baseline, no new deps)
//! - JevDecider: TypeSafe Jev System One API (opt-in, needs an API key)
//! - MockDecider: scripted answers for tests
//!
//! A single `decide` call asks several named questions about one shared
//! `state`. Jev answers all questions in one forward pass, so batching
//! questions is the main lever for cost and latency.
//!
//! Nothing in the read or write path uses a Decider yet — see
//! `docs/decider-plan.md` for the rollout order.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{KartaError, Result};

mod jev;
mod llm;
pub mod questions;

pub use jev::JevDecider;
pub use llm::LlmDecider;

/// The kind of answer a question expects.
#[derive(Debug, Clone, PartialEq)]
pub enum QuestionKind {
    /// Pick exactly one of the options. Options are `(label, description)`.
    /// Keep option order stable across calls: closed-set classifiers flip a
    /// measurable fraction of answers under option reordering.
    Choice { options: Vec<(String, String)> },
    /// Yes / no — answered as a probability that the statement holds.
    YesNo,
    /// Ordinal score on `0..=levels.len()-1`; each entry describes one level,
    /// lowest first. Jev accepts at most 10 levels.
    Score { levels: Vec<String> },
}

/// One question to ask about a state.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    /// Stable key used to look up the answer in [`Decisions`].
    pub name: String,
    /// Natural-language instruction, e.g. "Which retrieval mode fits this query?"
    pub instructions: String,
    pub kind: QuestionKind,
}

impl Question {
    pub fn choice(name: &str, instructions: &str, options: Vec<(String, String)>) -> Self {
        Self {
            name: name.to_string(),
            instructions: instructions.to_string(),
            kind: QuestionKind::Choice { options },
        }
    }

    pub fn yes_no(name: &str, instructions: &str) -> Self {
        Self {
            name: name.to_string(),
            instructions: instructions.to_string(),
            kind: QuestionKind::YesNo,
        }
    }

    pub fn score(name: &str, instructions: &str, levels: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            instructions: instructions.to_string(),
            kind: QuestionKind::Score { levels },
        }
    }
}

/// The answer to one question.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Chosen option label plus a probability per option label.
    /// Backends that cannot produce probabilities report 1.0 for the choice.
    Choice {
        label: String,
        probs: Vec<(String, f32)>,
        confidence: f32,
    },
    /// Probability that the statement holds.
    YesNo { p_yes: f32 },
    /// Probability-weighted position on the scale, in `0..=levels.len()-1`.
    Score { value: f32, confidence: f32 },
}

impl Decision {
    /// Chosen label, if this is a choice.
    pub fn label(&self) -> Option<&str> {
        match self {
            Decision::Choice { label, .. } => Some(label),
            _ => None,
        }
    }

    /// Confidence in the answer, in `0.0..=1.0`. For yes/no this is the
    /// probability of the more likely side.
    pub fn confidence(&self) -> f32 {
        match self {
            Decision::Choice { confidence, .. } | Decision::Score { confidence, .. } => *confidence,
            Decision::YesNo { p_yes } => p_yes.max(1.0 - p_yes),
        }
    }
}

/// Answers keyed by [`Question::name`].
pub type Decisions = HashMap<String, Decision>;

/// Trait for answering typed questions about a state.
#[async_trait]
pub trait Decider: Send + Sync {
    /// Answer every question about `state`. Implementations must return an
    /// entry for each question or an error — never a partial map.
    async fn decide(&self, state: &str, questions: &[Question]) -> Result<Decisions>;

    /// Backend identifier for traces and benchmarks (e.g. "llm:gpt-5.4-mini", "jev-latest").
    fn id(&self) -> String;
}

/// Check a backend's answers against the questions asked: every question
/// answered, choice labels drawn from the offered options, scores in range.
pub(crate) fn validate(questions: &[Question], decisions: &Decisions) -> Result<()> {
    for q in questions {
        let d = decisions.get(&q.name).ok_or_else(|| {
            KartaError::Llm(format!("decider returned no answer for '{}'", q.name))
        })?;
        match (&q.kind, d) {
            (QuestionKind::Choice { options }, Decision::Choice { label, .. }) => {
                if !options.iter().any(|(l, _)| l == label) {
                    return Err(KartaError::Llm(format!(
                        "decider chose unknown option '{}' for '{}'",
                        label, q.name
                    )));
                }
            }
            (QuestionKind::YesNo, Decision::YesNo { p_yes }) => {
                if !(0.0..=1.0).contains(p_yes) {
                    return Err(KartaError::Llm(format!(
                        "decider p_yes {} out of range for '{}'",
                        p_yes, q.name
                    )));
                }
            }
            (QuestionKind::Score { levels }, Decision::Score { value, .. }) => {
                let max = levels.len().saturating_sub(1);
                if !(0.0..=max as f32).contains(value) {
                    return Err(KartaError::Llm(format!(
                        "decider score {} out of 0..={} for '{}'",
                        value, max, q.name
                    )));
                }
            }
            _ => {
                return Err(KartaError::Llm(format!(
                    "decider answer kind does not match question '{}'",
                    q.name
                )))
            }
        }
    }
    Ok(())
}

/// Scripted decider for tests. Answers each question from a fixed map keyed
/// by question name and records every `(state, question names)` call.
pub struct MockDecider {
    answers: Decisions,
    calls: Mutex<Vec<(String, Vec<String>)>>,
}

impl MockDecider {
    pub fn new(answers: Decisions) -> Self {
        Self {
            answers,
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<(String, Vec<String>)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl Decider for MockDecider {
    async fn decide(&self, state: &str, questions: &[Question]) -> Result<Decisions> {
        self.calls.lock().unwrap().push((
            state.to_string(),
            questions.iter().map(|q| q.name.clone()).collect(),
        ));
        let out: Decisions = questions
            .iter()
            .filter_map(|q| self.answers.get(&q.name).map(|d| (q.name.clone(), d.clone())))
            .collect();
        validate(questions, &out)?;
        Ok(out)
    }

    fn id(&self) -> String {
        "mock".to_string()
    }
}
