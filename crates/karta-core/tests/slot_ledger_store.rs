#![cfg(feature = "sqlite-vec")]
use chrono::{TimeZone, Utc};
use karta_core::store::slot_ledger::{NewLedgerRow, SlotLedger};
use karta_core::store::sqlite_vec::SqliteVectorStore;
use tempfile::TempDir;

#[tokio::test]
async fn insert_and_read_current() {
    let dir = TempDir::new().unwrap();
    let vec_store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();
    let ledger = SlotLedger::new(vec_store.connection()).unwrap();

    let mt = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let id = ledger
        .insert_open(&NewLedgerRow {
            entity_key: "dashboard api".into(),
            predicate: "metric_value".into(),
            value: "250ms".into(),
            value_norm: "250ms".into(),
            seq: 1,
            valid_from: None,
            mention_time: mt,
            note_id: "n1".into(),
            source_span: "response time is 250ms".into(),
        })
        .unwrap();
    assert!(id > 0);

    let cur = ledger.current("dashboard api", "metric_value").unwrap();
    assert_eq!(cur.len(), 1);
    assert_eq!(cur[0].value, "250ms");
    assert!(cur[0].valid_to.is_none());
}
