use chrono::{TimeZone, Utc};
use karta_core::read::temporal::{
    validate_resolver_output, RawResolverOutput, ResolverValidationError,
};

fn raw(start: Option<chrono::DateTime<Utc>>, end: Option<chrono::DateTime<Utc>>, conf: f32) -> RawResolverOutput {
    RawResolverOutput { start, end, confidence_f32: conf }
}

#[test]
fn valid_explicit_date_band() {
    let s = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let e = s + chrono::Duration::days(1);
    let (iv, band) = validate_resolver_output(raw(Some(s), Some(e), 1.0)).unwrap();
    assert!(iv.is_some());
    assert_eq!(band.as_f32(), 1.0);
}

#[test]
fn valid_null_bounds_zero_conf() {
    let (iv, band) = validate_resolver_output(raw(None, None, 0.0)).unwrap();
    assert!(iv.is_none());
    assert_eq!(band.as_f32(), 0.0);
}

#[test]
fn f7_t21a_confidence_0_3_rejected() {
    let s = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let e = s + chrono::Duration::days(1);
    let err = validate_resolver_output(raw(Some(s), Some(e), 0.3)).unwrap_err();
    assert!(matches!(err, ResolverValidationError::ConfidenceNotInClosedSet(_)));
}

#[test]
fn f7_t21b_confidence_neg_rejected() {
    let s = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let e = s + chrono::Duration::days(1);
    assert!(matches!(
        validate_resolver_output(raw(Some(s), Some(e), -0.1)).unwrap_err(),
        ResolverValidationError::ConfidenceNotInClosedSet(_)
    ));
}

#[test]
fn f7_t21c_confidence_above_one_rejected() {
    let s = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let e = s + chrono::Duration::days(1);
    assert!(matches!(
        validate_resolver_output(raw(Some(s), Some(e), 1.2)).unwrap_err(),
        ResolverValidationError::ConfidenceNotInClosedSet(_)
    ));
}

#[test]
fn f7_t21d_conf_with_null_bounds_rejected() {
    assert!(matches!(
        validate_resolver_output(raw(None, None, 0.7)).unwrap_err(),
        ResolverValidationError::ConfidenceBoundsMismatch { .. }
    ));
}

#[test]
fn f7_t21e_start_only_rejected() {
    let s = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    assert!(matches!(
        validate_resolver_output(raw(Some(s), None, 0.7)).unwrap_err(),
        ResolverValidationError::UnpairedBounds
    ));
}

#[test]
fn f7_t21f_end_before_start_rejected() {
    let s = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let e = s - chrono::Duration::days(1);
    assert!(matches!(
        validate_resolver_output(raw(Some(s), Some(e), 0.7)).unwrap_err(),
        ResolverValidationError::EndNotAfterStart
    ));
}
