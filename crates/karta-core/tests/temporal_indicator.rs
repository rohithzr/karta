//! F7-T20: regex/keyword pre-filter that flags temporally-indicated queries.
//!
//! Verifies the classifier output's `temporal: bool` field via the public
//! `query_is_temporal` wrapper exposed from `karta_core::read`.

use karta_core::read::{classify_query, query_is_temporal};

#[test]
fn f7_t20_temporal_indicator_detection_positive() {
    // Relative-time keywords
    assert!(query_is_temporal("when did I deploy"));
    assert!(query_is_temporal("what did I close last week"));
    assert!(query_is_temporal("what did we decide yesterday"));
    assert!(query_is_temporal("anything happening tomorrow"));
    assert!(query_is_temporal("notes from this month"));
    assert!(query_is_temporal("recently merged PRs"));
    assert!(query_is_temporal("before the launch"));
    assert!(query_is_temporal("after the standup"));

    // Month and weekday names
    assert!(query_is_temporal("meetings in March"));
    assert!(query_is_temporal("commits on Tuesday"));

    // Year tokens / ISO dates
    assert!(query_is_temporal("commits on 2024-03-15"));
    assert!(query_is_temporal("things from 1999"));
}

#[test]
fn f7_t20_temporal_indicator_detection_negative() {
    assert!(!query_is_temporal("how does Flask work"));
    assert!(!query_is_temporal("what is the database schema"));
    assert!(!query_is_temporal("user preferences"));
}

#[test]
fn f7_t20_classify_query_emits_temporal_flag() {
    // The flag rides on the classifier output independently of QueryMode.
    let temporal_query = classify_query("what did I close last week");
    assert!(temporal_query.temporal);

    let plain_query = classify_query("what is the database schema");
    assert!(!plain_query.temporal);
}
