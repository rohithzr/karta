use chrono::{TimeZone, Utc};
use karta_core::read::resolve::resolve_temporal_phrase;
use karta_core::read::temporal::ConfidenceBand;

fn ref_time() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2024, 4, 22, 10, 0, 0).unwrap()
}

#[test]
fn f7_t17_iso_date_resolves_to_day_interval() {
    let r = resolve_temporal_phrase("what happened on 2024-03-15", ref_time()).unwrap();
    assert_eq!(
        r.0.start.date_naive(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 15).unwrap()
    );
    assert_eq!(r.0.end, r.0.start + chrono::Duration::days(1));
    assert_eq!(r.1, ConfidenceBand::Explicit);
}

#[test]
fn f7_t17_nl_date_resolves_to_day_interval() {
    let r = resolve_temporal_phrase("what happened on March 15, 2024", ref_time()).unwrap();
    assert_eq!(
        r.0.start.date_naive(),
        chrono::NaiveDate::from_ymd_opt(2024, 3, 15).unwrap()
    );
    assert_eq!(r.1, ConfidenceBand::NLAbsolute);
}

#[test]
fn f7_t17_yesterday_resolves_to_1day_relative() {
    let r = resolve_temporal_phrase("what did I do yesterday", ref_time()).unwrap();
    assert_eq!(
        r.0.start.date_naive(),
        chrono::NaiveDate::from_ymd_opt(2024, 4, 21).unwrap()
    );
    assert_eq!(
        r.0.end.date_naive(),
        chrono::NaiveDate::from_ymd_opt(2024, 4, 22).unwrap()
    );
    assert_eq!(r.1, ConfidenceBand::Relative);
}

#[test]
fn f7_t17_last_week_resolves_to_7d_window() {
    // "last week" = [ref - 7d, ref) as a rolling 7-day window.
    let r = resolve_temporal_phrase("what happened last week", ref_time()).unwrap();
    assert_eq!(
        r.0.end.date_naive(),
        chrono::NaiveDate::from_ymd_opt(2024, 4, 22).unwrap()
    );
    assert_eq!(
        r.0.start.date_naive(),
        chrono::NaiveDate::from_ymd_opt(2024, 4, 15).unwrap()
    );
    assert_eq!(r.1, ConfidenceBand::Relative);
}

#[test]
fn f7_t18_ambiguous_returns_none() {
    assert!(resolve_temporal_phrase("what did we say last spring", ref_time()).is_none());
    assert!(resolve_temporal_phrase("before the deadline", ref_time()).is_none());
    assert!(resolve_temporal_phrase("end of Q2", ref_time()).is_none());
}

#[test]
fn f7_t18_no_temporal_indicator_returns_none() {
    assert!(resolve_temporal_phrase("how does Flask work", ref_time()).is_none());
}
