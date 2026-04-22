//! Tier 1 temporal resolver — Rust regex for high-confidence common phrases.
//!
//! Returns None for anything ambiguous. The only fallback is tier 2 (LLM).
//! Never silently guesses: if the phrase doesn't match a known pattern with
//! high confidence, we punt.

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use regex::Regex;

use super::temporal::{ConfidenceBand, Interval};

pub fn resolve_temporal_phrase(
    query: &str,
    reference_time: DateTime<Utc>,
) -> Option<(Interval, ConfidenceBand)> {
    let q = query.to_lowercase();

    // Punt immediately on known-ambiguous phrases so ordering doesn't matter.
    let ambiguous_patterns = [
        "last spring",
        "last summer",
        "last fall",
        "last autumn",
        "last winter",
        "before the deadline",
        "after the deadline",
        "end of q",
        "q2",
        "q3",
        "q4",
        "fiscal",
    ];
    if ambiguous_patterns.iter().any(|p| q.contains(p)) {
        return None;
    }

    // Order matters: more specific patterns first.
    if let Some(iv) = try_iso_date(&q) {
        return Some((iv, ConfidenceBand::Explicit));
    }
    if let Some(iv) = try_nl_absolute_date(&q) {
        return Some((iv, ConfidenceBand::NLAbsolute));
    }
    if let Some(iv) = try_day_relative(&q, reference_time) {
        return Some((iv, ConfidenceBand::Relative));
    }
    if let Some(iv) = try_week_relative(&q, reference_time) {
        return Some((iv, ConfidenceBand::Relative));
    }
    if let Some(iv) = try_month_relative(&q, reference_time) {
        return Some((iv, ConfidenceBand::Relative));
    }

    None
}

fn day_interval_at(date: NaiveDate) -> Interval {
    let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
    let end = start + Duration::days(1);
    Interval { start, end }
}

fn try_iso_date(q: &str) -> Option<Interval> {
    let re = Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").unwrap();
    let cap = re.captures(q)?;
    let y: i32 = cap[1].parse().ok()?;
    let m: u32 = cap[2].parse().ok()?;
    let d: u32 = cap[3].parse().ok()?;
    let date = NaiveDate::from_ymd_opt(y, m, d)?;
    Some(day_interval_at(date))
}

fn try_nl_absolute_date(q: &str) -> Option<Interval> {
    let months = [
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("may", 5),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
    ];
    for (name, m) in months {
        let re = Regex::new(&format!(r"{name}\s+(\d{{1,2}}),?\s+(\d{{4}})")).unwrap();
        if let Some(cap) = re.captures(q) {
            let d: u32 = cap[1].parse().ok()?;
            let y: i32 = cap[2].parse().ok()?;
            if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
                return Some(day_interval_at(date));
            }
        }
    }
    None
}

fn try_day_relative(q: &str, now: DateTime<Utc>) -> Option<Interval> {
    let today = now.date_naive();
    if q.contains("yesterday") {
        return Some(day_interval_at(today.pred_opt()?));
    }
    if q.contains("tomorrow") {
        return Some(day_interval_at(today.succ_opt()?));
    }
    if q.contains("today") {
        return Some(day_interval_at(today));
    }
    None
}

fn try_week_relative(q: &str, now: DateTime<Utc>) -> Option<Interval> {
    // "last week" = rolling 7-day window ending at `now.date_naive()`.
    // i.e. [now - 7d, now) aligned to day boundaries.
    if q.contains("last week") {
        let end_date = now.date_naive();
        let start_date = end_date - Duration::days(7);
        let start = start_date.and_hms_opt(0, 0, 0)?.and_utc();
        let end = end_date.and_hms_opt(0, 0, 0)?.and_utc();
        return Some(Interval { start, end });
    }
    if q.contains("this week") {
        // 7-day window centered on today, using Monday-start convention:
        // [Monday of this week, Monday of next week).
        let weekday = now.weekday().num_days_from_monday() as i64;
        let monday = now.date_naive() - Duration::days(weekday);
        let start = monday.and_hms_opt(0, 0, 0)?.and_utc();
        let end = start + Duration::days(7);
        return Some(Interval { start, end });
    }
    None
}

fn try_month_relative(q: &str, now: DateTime<Utc>) -> Option<Interval> {
    if q.contains("last month") {
        let first_of_this_month = NaiveDate::from_ymd_opt(now.year(), now.month(), 1)?;
        let first_of_last_month = if now.month() == 1 {
            NaiveDate::from_ymd_opt(now.year() - 1, 12, 1)?
        } else {
            NaiveDate::from_ymd_opt(now.year(), now.month() - 1, 1)?
        };
        let start = first_of_last_month.and_hms_opt(0, 0, 0)?.and_utc();
        let end = first_of_this_month.and_hms_opt(0, 0, 0)?.and_utc();
        return Some(Interval { start, end });
    }
    None
}
