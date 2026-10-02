//! Per-fact validators for temporal slot population.
//!
//! Mirrors the supporting_spans cite-and-validate pattern. Real LLMs (esp.
//! smaller open models like Gemma) bias toward null fields under structured
//! output. Prose rules in the prompt lose to that bias. Structural validation
//! is what holds.
//!
//! Two gates:
//!  1. `validate_value_date_for_facet` — date-shaped facets MUST populate
//!     `value_date` (otherwise the fact carries no retrievable date).
//!  2. `validate_occurred_grounding` — `occurred_*` must be cited by at
//!     least one supporting_span containing a temporal marker (literal
//!     date or relative phrase). Inferred bounds get stripped.

use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use regex::Regex;

use crate::extract::facet::Facet;

#[derive(Debug, PartialEq, Eq)]
pub enum ValueDateOutcome {
    Keep,
    StripFact,
}

#[derive(Debug, PartialEq, Eq)]
pub enum OccurredOutcome {
    Keep,
    StripBounds,
}

/// If the facet is date-shaped (Deadline / TargetDate), `value_date` MUST
/// be populated. Otherwise the fact is contentless from a retrieval
/// standpoint — we'd carry "facet=deadline, value=null" with no way to
/// answer interval queries.
pub fn validate_value_date_for_facet(
    facet: Facet,
    value_date: Option<DateTime<Utc>>,
) -> ValueDateOutcome {
    match facet {
        Facet::Deadline | Facet::TargetDate => {
            if value_date.is_none() {
                ValueDateOutcome::StripFact
            } else {
                ValueDateOutcome::Keep
            }
        }
        _ => ValueDateOutcome::Keep,
    }
}

fn iso_date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap())
}

fn nl_month_day_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // matches "March 15", "March 15, 2024", "Mar 15", "15 March", etc.
        Regex::new(
            r"(?ix)
            \b(
                jan(uary)? | feb(ruary)? | mar(ch)? | apr(il)? | may | jun(e)? |
                jul(y)?   | aug(ust)?  | sep(tember)? | oct(ober)? |
                nov(ember)? | dec(ember)?
            )
            \s+\d{1,2}\b
            |
            \b\d{1,2}\s+(
                jan(uary)? | feb(ruary)? | mar(ch)? | apr(il)? | may | jun(e)? |
                jul(y)?   | aug(ust)?  | sep(tember)? | oct(ober)? |
                nov(ember)? | dec(ember)?
            )\b
            ",
        )
        .unwrap()
    })
}

fn year_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(19|20)\d{2}\b").unwrap())
}

fn bare_month_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        )
        .unwrap()
    })
}

const TEMPORAL_MARKERS: &[&str] = &[
    "yesterday",
    "today",
    "tomorrow",
    "last week",
    "last month",
    "last year",
    "next week",
    "next month",
    "next year",
    "ago",
    "recently",
    "just now",
    "this morning",
    "this afternoon",
    "this evening",
    "tonight",
    // W9 parity additions:
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "weekend",
    "last weekend",
    "over the weekend",
    "this weekend",
];

fn has_temporal_marker(span: &str) -> bool {
    let lower = span.to_lowercase();
    if TEMPORAL_MARKERS.iter().any(|m| lower.contains(m)) {
        return true;
    }
    if iso_date_re().is_match(&lower) {
        return true;
    }
    if nl_month_day_re().is_match(&lower) {
        return true;
    }
    if year_re().is_match(&lower) {
        return true;
    }
    if bare_month_re().is_match(&lower) {
        return true;
    }
    false
}

/// Derive a missing `occurred_end` from `occurred_start` using the same
/// convention the JSON schema documents: next day for date-only (midnight-
/// aligned) starts, start + 1ns for instants.
///
/// Small models reliably emit `occurred_start` but leave `occurred_end`
/// null — the prompt examples are start-centric and structured output
/// biases optional fields to null. Without derivation those facts die in
/// `validate_occurred` as unpaired bounds, which silently discards the
/// exact past-event facts the event_ordering ability depends on. A
/// model-provided end always wins; end-without-start is left untouched
/// for `validate_occurred` to reject.
pub fn derive_occurred_end(
    occurred_start: Option<DateTime<Utc>>,
    occurred_end: Option<DateTime<Utc>>,
) -> Option<DateTime<Utc>> {
    if occurred_end.is_some() {
        return occurred_end;
    }
    let start = occurred_start?;
    let is_midnight_aligned = start.time() == chrono::NaiveTime::MIN;
    if is_midnight_aligned {
        Some(start + chrono::Duration::days(1))
    } else {
        Some(start + chrono::Duration::nanoseconds(1))
    }
}

/// If a fact has `occurred_*` populated but no supporting_span contains a
/// temporal marker (literal date or relative phrase), the bounds were
/// inferred rather than grounded. Strip them.
///
/// Null occurred_* + `ConfidenceBand::None` (`occurred_confidence == 0.0`)
/// is the no-op case.
pub fn validate_occurred_grounding(
    occurred_start: Option<DateTime<Utc>>,
    occurred_confidence_f32: f32,
    supporting_spans: &[String],
) -> OccurredOutcome {
    if occurred_start.is_none() && occurred_confidence_f32 == 0.0 {
        return OccurredOutcome::Keep;
    }
    if supporting_spans.iter().any(|s| has_temporal_marker(s)) {
        OccurredOutcome::Keep
    } else {
        OccurredOutcome::StripBounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn t() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap()
    }

    // ---- value_date validator ----

    #[test]
    fn deadline_without_value_date_is_stripped() {
        assert_eq!(
            validate_value_date_for_facet(Facet::Deadline, None),
            ValueDateOutcome::StripFact
        );
    }

    #[test]
    fn deadline_with_value_date_passes() {
        assert_eq!(
            validate_value_date_for_facet(Facet::Deadline, Some(t())),
            ValueDateOutcome::Keep
        );
    }

    #[test]
    fn target_date_without_value_date_is_stripped() {
        assert_eq!(
            validate_value_date_for_facet(Facet::TargetDate, None),
            ValueDateOutcome::StripFact
        );
    }

    #[test]
    fn target_date_with_value_date_passes() {
        assert_eq!(
            validate_value_date_for_facet(Facet::TargetDate, Some(t())),
            ValueDateOutcome::Keep
        );
    }

    #[test]
    fn non_date_facet_is_unchanged_when_value_date_null() {
        for f in [
            Facet::TechStack,
            Facet::Preference,
            Facet::Constraint,
            Facet::Event,
            Facet::Location,
            Facet::Ownership,
            Facet::Unknown,
        ] {
            assert_eq!(
                validate_value_date_for_facet(f, None),
                ValueDateOutcome::Keep,
                "facet {:?} should pass with null value_date",
                f
            );
        }
    }

    // ---- occurred_* grounding ----

    #[test]
    fn null_occurred_passes() {
        let spans = vec!["Flask 2.3.1".to_string()];
        assert_eq!(
            validate_occurred_grounding(None, 0.0, &spans),
            OccurredOutcome::Keep
        );
    }

    #[test]
    fn occurred_with_iso_span_passes() {
        let spans = vec!["closed on 2024-03-29".to_string()];
        assert_eq!(
            validate_occurred_grounding(Some(t()), 1.0, &spans),
            OccurredOutcome::Keep
        );
    }

    #[test]
    fn occurred_with_relative_span_passes() {
        let spans = vec!["closed the auth ticket yesterday".to_string()];
        assert_eq!(
            validate_occurred_grounding(Some(t()), 0.7, &spans),
            OccurredOutcome::Keep
        );
    }

    #[test]
    fn occurred_with_nl_month_day_span_passes() {
        let spans = vec!["Sprint 1 ended March 29".to_string()];
        assert_eq!(
            validate_occurred_grounding(Some(t()), 0.8, &spans),
            OccurredOutcome::Keep
        );
    }

    #[test]
    fn occurred_without_temporal_span_is_stripped() {
        let spans = vec!["Flask is the web framework".to_string()];
        assert_eq!(
            validate_occurred_grounding(Some(t()), 0.7, &spans),
            OccurredOutcome::StripBounds
        );
    }

    #[test]
    fn occurred_with_no_spans_at_all_is_stripped() {
        let spans: Vec<String> = vec![];
        assert_eq!(
            validate_occurred_grounding(Some(t()), 0.7, &spans),
            OccurredOutcome::StripBounds
        );
    }

    #[test]
    fn case_insensitive_relative_marker() {
        let spans = vec!["YESTERDAY I closed it".to_string()];
        assert_eq!(
            validate_occurred_grounding(Some(t()), 0.7, &spans),
            OccurredOutcome::Keep
        );
    }

    // ---- occurred_end derivation ----

    #[test]
    fn derive_end_from_midnight_aligned_start_is_next_day() {
        let start = Utc.with_ymd_and_hms(2024, 3, 29, 0, 0, 0).unwrap();
        let want = Utc.with_ymd_and_hms(2024, 3, 30, 0, 0, 0).unwrap();
        assert_eq!(derive_occurred_end(Some(start), None), Some(want));
    }

    #[test]
    fn derive_end_from_instant_start_is_start_plus_1ns() {
        let start = Utc.with_ymd_and_hms(2024, 3, 29, 14, 30, 0).unwrap();
        let want = start + chrono::Duration::nanoseconds(1);
        assert_eq!(derive_occurred_end(Some(start), None), Some(want));
    }

    #[test]
    fn derive_end_keeps_model_provided_end() {
        let start = Utc.with_ymd_and_hms(2024, 3, 29, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 4, 2, 0, 0, 0).unwrap();
        assert_eq!(derive_occurred_end(Some(start), Some(end)), Some(end));
    }

    #[test]
    fn derive_end_without_start_stays_none() {
        assert_eq!(derive_occurred_end(None, None), None);
        // end-without-start is unpaired garbage — leave it for
        // validate_occurred to reject rather than inventing a start.
        let end = Utc.with_ymd_and_hms(2024, 4, 2, 0, 0, 0).unwrap();
        assert_eq!(derive_occurred_end(None, Some(end)), Some(end));
    }

    #[test]
    fn marker_parity_with_read_side() {
        assert!(has_temporal_marker("I deployed last Friday"));
        assert!(has_temporal_marker("we shipped over the weekend"));
        assert!(has_temporal_marker("the review is in March"));
        assert!(has_temporal_marker("this happened in 2024"));
        // negative control: no temporal cue
        assert!(!has_temporal_marker("the function returns a vector"));
    }
}
