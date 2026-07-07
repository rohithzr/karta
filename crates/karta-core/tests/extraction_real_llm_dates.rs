//! Real-LLM regression for temporal slot population.
//!
//! Validates that, given a message containing a date phrase, the production
//! LLM (read from .env via `Karta::with_defaults`) emits a fact with the
//! correct value_date or occurred_* slot. This is the gap exposed by BEAM
//! pre-dream conv 0 v1 (56.6%, temporal_reasoning 0/4) — the mock-LLM
//! `occurred_extraction.rs` test is deterministic and didn't catch the
//! prompt regression.
//!
//! Skipped unless `KARTA_REAL_LLM_TESTS=1`. Requires Ollama (or whatever
//! `.env` points to) to be reachable. Each fixture entry runs in its own
//! tempdir so failures don't pollute siblings.
//!
//! Run:
//!   KARTA_REAL_LLM_TESTS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
//!     cargo test --release --features sqlite-vec -p karta-core \
//!     --test extraction_real_llm_dates -- --nocapture

#![cfg(feature = "sqlite-vec")]

use std::path::Path;

use chrono::{DateTime, Utc};
use karta_core::clock::ClockContext;
use karta_core::config::KartaConfig;
use karta_core::note::AtomicFact;
use karta_core::Karta;
use serde::Deserialize;
use tempfile::TempDir;

#[derive(Debug, Deserialize)]
struct Fixture {
    id: String,
    input: String,
    ref_time_iso: String,
    expect_any_fact_with: ExpectedSlot,
}

#[derive(Debug, Deserialize)]
struct ExpectedSlot {
    facet: String,
    #[serde(default)]
    value_date_iso: Option<String>,
    #[serde(default)]
    value_text_substring: Option<String>,
    #[serde(default)]
    occurred_start_iso: Option<String>,
    #[serde(default)]
    occurred_confidence: Option<f64>,
}

async fn make_real_karta(dir: &Path) -> Karta {
    let mut config = KartaConfig::default();
    config.storage.data_dir = dir.to_string_lossy().to_string();
    Karta::with_defaults(config)
        .await
        .expect("Karta::with_defaults failed — check .env credentials")
}

fn parse_iso(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .expect("invalid RFC3339 in fixture")
        .to_utc()
}

fn matches_expectation(fact: &AtomicFact, exp: &ExpectedSlot) -> bool {
    let facet_serialized =
        serde_json::to_value(fact.facet).unwrap().as_str().unwrap().to_string();
    if facet_serialized != exp.facet {
        return false;
    }
    if let Some(ref iso) = exp.value_date_iso {
        let want = parse_iso(iso);
        if fact.value_date != Some(want) {
            return false;
        }
    }
    if let Some(ref needle) = exp.value_text_substring {
        match &fact.value_text {
            Some(v) if v.to_lowercase().contains(&needle.to_lowercase()) => {}
            _ => return false,
        }
    }
    if let Some(ref iso) = exp.occurred_start_iso {
        let want = parse_iso(iso);
        if fact.occurred_start != Some(want) {
            return false;
        }
    }
    if let Some(want_conf) = exp.occurred_confidence {
        let got = fact.occurred_confidence.as_f32() as f64;
        if (got - want_conf).abs() > 1e-6 {
            return false;
        }
    }
    true
}

#[tokio::test]
async fn real_llm_extracts_temporal_slots_from_dated_messages() {
    if std::env::var("KARTA_REAL_LLM_TESTS").unwrap_or_default() != "1" {
        eprintln!("[extraction_real_llm_dates] skipping — set KARTA_REAL_LLM_TESTS=1 to run");
        return;
    }

    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../data/test/fixtures/dated_messages.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing {}: {}", path.display(), e));
    let fixtures: Vec<Fixture> =
        serde_json::from_str(&raw).expect("malformed dated_messages.json");
    assert!(!fixtures.is_empty(), "fixture file is empty");

    let mut failures: Vec<String> = Vec::new();
    let mut detail: Vec<String> = Vec::new();

    for fx in &fixtures {
        let dir = TempDir::new().unwrap();
        let karta = make_real_karta(dir.path()).await;
        let ref_time = parse_iso(&fx.ref_time_iso);

        let note = match karta
            .add_note_with_clock(&fx.input, Some("fx"), Some(0), ClockContext::at(ref_time))
            .await
        {
            Ok(n) => n,
            Err(e) => {
                failures.push(format!("[{}] add_note error: {}", fx.id, e));
                continue;
            }
        };

        let facts = karta
            .get_facts_for_note(&note.id)
            .await
            .unwrap_or_else(|e| panic!("[{}] get_facts_for_note: {}", fx.id, e));

        let matched = facts.iter().any(|f| matches_expectation(f, &fx.expect_any_fact_with));

        if matched {
            detail.push(format!("[{}] OK ({} facts emitted)", fx.id, facts.len()));
        } else {
            failures.push(format!(
                "[{}] no fact matched expectation (facet={:?})",
                fx.id, fx.expect_any_fact_with.facet
            ));
            for (i, f) in facts.iter().enumerate() {
                detail.push(format!(
                    "    [{}#{}] facet={:?} entity={:?} vt={:?} vd={:?} occ_start={:?} conf={:?}",
                    fx.id, i, f.facet, f.entity_text, f.value_text, f.value_date,
                    f.occurred_start, f.occurred_confidence,
                ));
            }
            if facts.is_empty() {
                detail.push(format!("    [{}] (zero facts emitted)", fx.id));
            }
        }
    }

    eprintln!(
        "\n[extraction_real_llm_dates] {} / {} fixtures matched\n{}",
        fixtures.len() - failures.len(),
        fixtures.len(),
        detail.join("\n"),
    );

    if !failures.is_empty() {
        panic!(
            "{} / {} fixture(s) failed:\n  {}",
            failures.len(),
            fixtures.len(),
            failures.join("\n  ")
        );
    }
}
