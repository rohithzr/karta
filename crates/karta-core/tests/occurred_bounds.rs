use chrono::{TimeZone, Utc};
use karta_core::note::AtomicFact;
use karta_core::read::temporal::ConfidenceBand;

fn sample(
    start: Option<chrono::DateTime<Utc>>,
    end: Option<chrono::DateTime<Utc>>,
    conf: ConfidenceBand,
) -> AtomicFact {
    AtomicFact {
        id: "f1".into(),
        content: "x".into(),
        source_note_id: "n1".into(),
        ordinal: 0,
        subject: None,
        embedding: vec![],
        created_at: Utc::now(),
        source_timestamp: Utc::now(),
        occurred_start: start,
        occurred_end: end,
        occurred_confidence: conf,
    }
}

#[test]
fn inv1_pairing_both_some() {
    let t = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let f = sample(Some(t), Some(t + chrono::Duration::days(1)), ConfidenceBand::Relative);
    assert!(f.validate_occurred().is_ok());
}

#[test]
fn inv1_pairing_both_none() {
    let f = sample(None, None, ConfidenceBand::None);
    assert!(f.validate_occurred().is_ok());
}

#[test]
fn inv1_pairing_mismatched_rejects() {
    let t = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let f_start_only = sample(Some(t), None, ConfidenceBand::Relative);
    let f_end_only = sample(None, Some(t), ConfidenceBand::Relative);
    assert!(f_start_only.validate_occurred().is_err());
    assert!(f_end_only.validate_occurred().is_err());
}

#[test]
fn inv2_end_must_be_after_start() {
    let t = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let f_equal = sample(Some(t), Some(t), ConfidenceBand::Relative);
    let f_inverted = sample(Some(t), Some(t - chrono::Duration::days(1)), ConfidenceBand::Relative);
    assert!(f_equal.validate_occurred().is_err());
    assert!(f_inverted.validate_occurred().is_err());
}

#[test]
fn inv2_instant_1ns_accepted() {
    let t = Utc.with_ymd_and_hms(2024, 3, 15, 14, 30, 0).unwrap();
    let instant = sample(Some(t), Some(t + chrono::Duration::nanoseconds(1)), ConfidenceBand::Explicit);
    assert!(instant.validate_occurred().is_ok());
}

#[test]
fn inv4_conf_none_with_bounds_rejects() {
    let t = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let f = sample(Some(t), Some(t + chrono::Duration::days(1)), ConfidenceBand::None);
    assert!(f.validate_occurred().is_err());
}

#[test]
fn inv4_conf_nonzero_without_bounds_rejects() {
    let f = sample(None, None, ConfidenceBand::Relative);
    assert!(f.validate_occurred().is_err());
}
