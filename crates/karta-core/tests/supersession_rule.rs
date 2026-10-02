#![cfg(feature = "sqlite-vec")]
use chrono::{TimeZone, Utc};
use karta_core::extract::slots::{Predicate, SlotTuple};
use karta_core::store::slot_ledger::{LedgerOutcome, SlotLedger};
use karta_core::store::sqlite_vec::SqliteVectorStore;
use tempfile::TempDir;

async fn ledger() -> SlotLedger {
    let dir = TempDir::new().unwrap();
    let vs = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();
    // leak the TempDir so the file survives for the test body
    std::mem::forget(dir);
    SlotLedger::new(vs.connection()).unwrap()
}
fn tup(v: &str, ev: Option<i32>) -> SlotTuple {
    SlotTuple {
        entity_text: "first sprint".into(),
        predicate: Predicate::Deadline,
        value: v.into(),
        event_time: ev.map(|d| Utc.with_ymd_and_hms(2024, 4, d as u32, 0, 0, 0).unwrap()),
        source_span: format!("deadline {v}"),
    }
}

#[tokio::test]
async fn insert_then_corroborate_then_update() {
    let l = ledger().await;
    let m = Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap();
    // first insert
    assert!(matches!(
        l.apply("first sprint", &tup("April 1", None), "april 1", 1, m, "n1")
            .unwrap(),
        LedgerOutcome::Inserted(_)
    ));
    // same value again → corroboration
    assert!(matches!(
        l.apply("first sprint", &tup("April 1", None), "april 1", 2, m, "n2")
            .unwrap(),
        LedgerOutcome::Corroborated(_)
    ));
    // new value, later mention, no equal event_time → update
    let outcome = l
        .apply("first sprint", &tup("April 5", None), "april 5", 3, m, "n3")
        .unwrap();
    assert!(matches!(outcome, LedgerOutcome::Updated { .. }));
    let cur = l.current("first sprint", "deadline").unwrap();
    assert_eq!(cur.len(), 1);
    assert_eq!(cur[0].value, "April 5");
}

#[tokio::test]
async fn equal_event_time_different_value_is_conflict() {
    let l = ledger().await;
    let m = Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap();
    l.apply("first sprint", &tup("April 5", Some(5)), "april 5", 1, m, "n1")
        .unwrap();
    let outcome = l
        .apply("first sprint", &tup("April 6", Some(5)), "april 6", 2, m, "n2")
        .unwrap();
    assert!(matches!(outcome, LedgerOutcome::Conflict { .. }));
    let cur = l.current("first sprint", "deadline").unwrap();
    assert_eq!(cur.len(), 2, "both values stay open as a conflict");
    assert!(cur.iter().all(|r| r.conflict_group.is_some()));
}
