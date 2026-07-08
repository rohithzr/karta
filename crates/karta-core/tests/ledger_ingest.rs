//! Real-LLM regression for slot-ledger population at ingest.
//!
//! Ingests two messages that both describe "first sprint deadline" via
//! `add_note_with_clock`, the second (later clock) superseding the first's
//! value. Verifies `Karta::slot_ledger_current` reflects a single open row
//! for the updated value, not the stale one.
//!
//! Skipped unless `KARTA_REAL_LLM_TESTS=1`. Requires `.env` credentials
//! (Azure gpt-5.4-mini in this worktree).
//!
//! Run:
//!   KARTA_REAL_LLM_TESTS=1 RUSTC_WRAPPER="" \
//!     cargo test --release --features sqlite-vec -p karta-core \
//!     --test ledger_ingest -- --nocapture

#![cfg(feature = "sqlite-vec")]

use chrono::{Duration, Utc};
use karta_core::clock::ClockContext;
use karta_core::config::KartaConfig;
use karta_core::Karta;
use tempfile::TempDir;

#[tokio::test]
async fn ledger_reflects_superseded_slot_value() {
    if std::env::var("KARTA_REAL_LLM_TESTS").unwrap_or_default() != "1" {
        eprintln!("[ledger_ingest] skipping — set KARTA_REAL_LLM_TESTS=1 to run");
        return;
    }

    let dir = TempDir::new().unwrap();
    let mut config = KartaConfig::default();
    config.storage.data_dir = dir.path().to_string_lossy().to_string();
    let karta = Karta::with_defaults(config)
        .await
        .expect("Karta::with_defaults failed — check .env credentials");

    let t0 = Utc::now();
    let t1 = t0 + Duration::minutes(10);

    karta
        .add_note_with_clock(
            "The first sprint deadline is April 1, 2024.",
            Some("conv"),
            Some(0),
            ClockContext::at(t0),
        )
        .await
        .expect("first ingest failed");

    karta
        .add_note_with_clock(
            "Actually the first sprint deadline moved to April 5, 2024.",
            Some("conv"),
            Some(1),
            ClockContext::at(t1),
        )
        .await
        .expect("second ingest failed");

    let rows = karta
        .slot_ledger_current("first sprint", "deadline")
        .await
        .expect("slot_ledger_current failed");

    eprintln!("[ledger_ingest] ledger rows: {:#?}", rows);

    assert_eq!(
        rows.len(),
        1,
        "expected exactly one open ledger row after supersession, got {}: {:#?}",
        rows.len(),
        rows
    );

    let value_lower = rows[0].value.to_lowercase();
    assert!(
        value_lower.contains("april 5") || rows[0].value.contains("5"),
        "expected open row to reflect April 5, got: {:?}",
        rows[0].value
    );

    assert!(
        rows[0].valid_to.is_none(),
        "expected the surviving row to remain open (valid_to = None), got: {:?}",
        rows[0].valid_to
    );
}
