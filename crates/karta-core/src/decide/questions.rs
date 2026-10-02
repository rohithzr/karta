//! Ready-made questions for Karta's closed-set decisions, plus the mapping
//! from answers back to Karta types. These are the first candidates for a
//! Decider (see `docs/decider-plan.md`); the read path does not call them yet.

use super::{Decision, Question};
use crate::extract::slots::Predicate;
use crate::read::QueryMode;

pub const QUERY_MODE: &str = "query_mode";
pub const QUERY_PREDICATE: &str = "query_predicate";
pub const QUERY_TEMPORAL: &str = "query_temporal";

/// Label used for "no mutable slot" in the predicate question.
pub const NO_PREDICATE: &str = "none";

/// Option order is fixed: keep it stable across releases so decisions stay
/// comparable (closed-set classifiers are sensitive to option order).
const MODE_OPTIONS: &[(QueryMode, &str, &str)] = &[
    (QueryMode::Standard, "standard", "a specific fact, preference, tool or detail"),
    (QueryMode::Recency, "recency", "the current or latest value of something that may have changed"),
    (QueryMode::Breadth, "breadth", "a summary, overview, or how something evolved over time"),
    (QueryMode::Computation, "computation", "a count, total, duration or difference computed across several facts"),
    (QueryMode::Temporal, "temporal", "the order or sequence in which things happened or were discussed"),
    (QueryMode::Existence, "existence", "whether something was ever said, or whether statements contradict"),
];

pub fn query_mode_question() -> Question {
    Question::choice(
        QUERY_MODE,
        "What is this question to a personal memory assistant mainly asking for?",
        MODE_OPTIONS
            .iter()
            .map(|(_, l, d)| (l.to_string(), d.to_string()))
            .collect(),
    )
}

pub fn query_mode_from(decision: &Decision) -> Option<QueryMode> {
    let label = decision.label()?;
    MODE_OPTIONS.iter().find(|(_, l, _)| *l == label).map(|(m, _, _)| *m)
}

fn predicate_description(p: Predicate) -> &'static str {
    match p {
        Predicate::RoleTitle => "someone's job title or role",
        Predicate::Employer => "where someone works",
        Predicate::Location => "where someone or something is located or lives",
        Predicate::Status => "the status or state of a project or task",
        Predicate::Deadline => "when something is due",
        Predicate::ScheduledDate => "when something is scheduled to happen",
        Predicate::Count => "how many of something there are",
        Predicate::Amount => "a money amount, price, salary or budget",
        Predicate::TechChoice => "which tool, library or technology is used",
        Predicate::Preference => "what someone prefers or likes",
        Predicate::Ownership => "who owns or is responsible for something",
        Predicate::MetricValue => "the value of a metric such as accuracy, latency or score",
    }
}

/// 12 predicates + "none" = 13 options.
pub fn query_predicate_question() -> Question {
    let mut options: Vec<(String, String)> = Predicate::all()
        .iter()
        .map(|p| (p.as_str().to_string(), predicate_description(*p).to_string()))
        .collect();
    options.push((
        NO_PREDICATE.to_string(),
        "none of these — not asking for a single current value".to_string(),
    ));
    Question::choice(
        QUERY_PREDICATE,
        "Which kind of changeable value is this question asking for the current value of?",
        options,
    )
}

/// `None` for the "none" option or an unexpected label.
pub fn query_predicate_from(decision: &Decision) -> Option<Predicate> {
    Predicate::from_str(decision.label()?)
}

pub fn query_temporal_question() -> Question {
    Question::yes_no(
        QUERY_TEMPORAL,
        "Does answering this question depend on a specific date, time period, or when something happened?",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decide::{Decider, Decisions, MockDecider};

    fn choice(label: &str) -> Decision {
        Decision::Choice { label: label.into(), probs: vec![], confidence: 0.9 }
    }

    #[test]
    fn every_mode_round_trips() {
        let q = query_mode_question();
        let crate::decide::QuestionKind::Choice { options } = &q.kind else { panic!() };
        assert_eq!(options.len(), 6);
        for (mode, label, _) in MODE_OPTIONS {
            assert_eq!(query_mode_from(&choice(label)), Some(*mode));
        }
    }

    #[test]
    fn predicate_question_covers_all_predicates_plus_none() {
        let q = query_predicate_question();
        let crate::decide::QuestionKind::Choice { options } = &q.kind else { panic!() };
        assert_eq!(options.len(), Predicate::all().len() + 1);
        assert_eq!(query_predicate_from(&choice("employer")), Some(Predicate::Employer));
        assert_eq!(query_predicate_from(&choice(NO_PREDICATE)), None);
    }

    #[tokio::test]
    async fn one_call_answers_all_query_questions() {
        let mut answers = Decisions::new();
        answers.insert(QUERY_MODE.into(), choice("recency"));
        answers.insert(QUERY_PREDICATE.into(), choice("employer"));
        answers.insert(QUERY_TEMPORAL.into(), Decision::YesNo { p_yes: 0.1 });
        let decider = MockDecider::new(answers);

        let qs = [query_mode_question(), query_predicate_question(), query_temporal_question()];
        let out = decider.decide("Where does Sarah work now?", &qs).await.unwrap();

        assert_eq!(query_mode_from(&out[QUERY_MODE]), Some(QueryMode::Recency));
        assert_eq!(query_predicate_from(&out[QUERY_PREDICATE]), Some(Predicate::Employer));
        assert_eq!(decider.calls().len(), 1);
    }
}
