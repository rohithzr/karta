//! STEP1.5 Task 13: calibration fixture loader.
//!
//! Validates the JSON fixture parses against the schema F7-T15 will
//! consume. Entry list may be empty pre-labeling — the loader test only
//! locks in shape/version/closed-set band values so a future schema drift
//! is caught before labels are wasted.

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Fixture {
    version: u32,
    #[allow(dead_code)]
    notes: String,
    entries: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
struct Entry {
    message_content: String,
    #[allow(dead_code)]
    reference_time: chrono::DateTime<chrono::Utc>,
    expected_facts: Vec<ExpectedFact>,
}

#[derive(Debug, Deserialize)]
struct ExpectedFact {
    content_pattern: String,
    expected_confidence_band: f32,
    expected_occurred_start: Option<chrono::DateTime<chrono::Utc>>,
    expected_occurred_end: Option<chrono::DateTime<chrono::Utc>>,
}

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../data/test/fixtures/confidence_calibration.json");
    p
}

#[test]
fn calibration_fixture_loads() {
    let raw = std::fs::read_to_string(fixture_path()).expect("fixture file must exist");
    let fx: Fixture = serde_json::from_str(&raw).expect("fixture must parse");
    assert_eq!(fx.version, 1);
    for e in &fx.entries {
        assert!(!e.message_content.is_empty());
        for f in &e.expected_facts {
            let valid_bands = [0.0, 0.5, 0.7, 0.8, 1.0];
            assert!(
                valid_bands.contains(&f.expected_confidence_band),
                "expected_confidence_band must be in closed set; got {}",
                f.expected_confidence_band
            );
            let _ = &f.content_pattern;
            let _ = &f.expected_occurred_start;
            let _ = &f.expected_occurred_end;
        }
    }
}

#[test]
fn calibration_readme_exists() {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../data/test/fixtures/README.md");
    assert!(p.exists(), "fixture README must exist at {:?}", p);
}
