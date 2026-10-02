use chrono::{TimeZone, Utc};
use karta_core::extract::dedup::dedup_extractions;
use karta_core::extract::entity_type::EntityType;
use karta_core::extract::facet::Facet;
use karta_core::extract::memory_kind::MemoryKind;
use karta_core::note::AtomicFactExtraction;
use karta_core::read::temporal::ConfidenceBand;

fn make(content: &str, ent: Option<&str>, facet: Facet, value_text: Option<&str>, value_date: Option<chrono::DateTime<Utc>>) -> AtomicFactExtraction {
    AtomicFactExtraction {
        content: content.into(),
        memory_kind: MemoryKind::DurableFact,
        supporting_spans: vec!["span".into()],
        facet,
        entity_type: if ent.is_some() { EntityType::Project } else { EntityType::Unknown },
        entity_text: ent.map(String::from),
        value_text: value_text.map(String::from),
        value_date,
        occurred_start: None,
        occurred_end: None,
        occurred_confidence: ConfidenceBand::None,
    }
}

#[test]
fn collapses_two_facts_same_slot_keeps_first() {
    let d = Utc.with_ymd_and_hms(2024, 4, 15, 0, 0, 0).unwrap();
    let facts = vec![
        make("April 15 deadline.", Some("Project"), Facet::Deadline, None, Some(d)),
        make("Deadline by April 15.", Some("project"), Facet::Deadline, None, Some(d)),
    ];
    let out = dedup_extractions(facts);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].content, "April 15 deadline.");
}

#[test]
fn keeps_distinct_slots() {
    let d = Utc.with_ymd_and_hms(2024, 4, 15, 0, 0, 0).unwrap();
    let facts = vec![
        make("April 15 deadline.", Some("Project"), Facet::Deadline, None, Some(d)),
        make("Project uses Flask.", Some("Project"), Facet::TechStack, Some("Flask"), None),
    ];
    let out = dedup_extractions(facts);
    assert_eq!(out.len(), 2);
}

#[test]
fn case_insensitive_entity_text() {
    let d = Utc.with_ymd_and_hms(2024, 4, 15, 0, 0, 0).unwrap();
    let facts = vec![
        make("a", Some("Coco"), Facet::Deadline, None, Some(d)),
        make("b", Some("coco"), Facet::Deadline, None, Some(d)),
    ];
    let out = dedup_extractions(facts);
    assert_eq!(out.len(), 1);
}

#[test]
fn empty_input_returns_empty() {
    assert_eq!(dedup_extractions(vec![]).len(), 0);
}

#[test]
fn collision_merges_supporting_spans_dedup_string_equality() {
    let d = Utc.with_ymd_and_hms(2024, 4, 15, 0, 0, 0).unwrap();
    let mut f1 = make("April 15 deadline.", Some("Project"), Facet::Deadline, None, Some(d));
    f1.supporting_spans = vec!["April 15 deadline".into()];
    let mut f2 = make("Due by the 15th.", Some("project"), Facet::Deadline, None, Some(d));
    f2.supporting_spans = vec!["due by the 15th".into(), "April 15 deadline".into()];
    let out = dedup_extractions(vec![f1, f2]);
    assert_eq!(out.len(), 1);
    // f1's "April 15 deadline" + f2's "due by the 15th" (the duplicate
    // "April 15 deadline" is dropped by string-equality dedup).
    assert_eq!(out[0].supporting_spans.len(), 2);
    assert!(out[0].supporting_spans.contains(&"April 15 deadline".to_string()));
    assert!(out[0].supporting_spans.contains(&"due by the 15th".to_string()));
}
