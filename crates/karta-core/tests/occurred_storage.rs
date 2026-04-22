// STEP1.5 Task 2: Round-trip storage tests for the new occurred_* columns
// and verification that the partial index `idx_facts_occurred` is created.

#![cfg(feature = "sqlite-vec")]

use chrono::{TimeZone, Utc};
use karta_core::note::AtomicFact;
use karta_core::read::temporal::ConfidenceBand;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::VectorStore;
use tempfile::TempDir;

fn embedding_of_dim(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 / n as f32).collect()
}

#[tokio::test]
async fn round_trip_occurred_fields() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();

    let start = Utc.with_ymd_and_hms(2024, 3, 15, 0, 0, 0).unwrap();
    let end = start + chrono::Duration::days(1);

    let fact = AtomicFact {
        id: "fact-1".into(),
        content: "User has a deadline on 2024-03-15".into(),
        source_note_id: "note-1".into(),
        ordinal: 0,
        subject: Some("deadline".into()),
        embedding: embedding_of_dim(1536),
        created_at: Utc::now(),
        source_timestamp: start,
        occurred_start: Some(start),
        occurred_end: Some(end),
        occurred_confidence: ConfidenceBand::Explicit,
    };

    store.upsert_fact(&fact).await.unwrap();
    let fetched = store.get_facts_for_note("note-1").await.unwrap();
    assert_eq!(fetched.len(), 1);
    let f = &fetched[0];
    assert_eq!(f.occurred_start, Some(start));
    assert_eq!(f.occurred_end, Some(end));
    assert_eq!(f.occurred_confidence, ConfidenceBand::Explicit);
    assert_eq!(f.source_timestamp, start);
}

#[tokio::test]
async fn round_trip_null_bounds() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();

    let fact = AtomicFact {
        id: "fact-2".into(),
        content: "Flask is used".into(),
        source_note_id: "note-1".into(),
        ordinal: 0,
        subject: Some("Flask".into()),
        embedding: embedding_of_dim(1536),
        created_at: Utc::now(),
        source_timestamp: Utc::now(),
        occurred_start: None,
        occurred_end: None,
        occurred_confidence: ConfidenceBand::None,
    };
    store.upsert_fact(&fact).await.unwrap();
    let fetched = store.get_facts_for_note("note-1").await.unwrap();
    assert_eq!(fetched.len(), 1);
    let f = &fetched[0];
    assert!(f.occurred_start.is_none());
    assert!(f.occurred_end.is_none());
    assert_eq!(f.occurred_confidence, ConfidenceBand::None);
}

#[tokio::test]
async fn partial_index_exists() {
    let dir = TempDir::new().unwrap();
    let _store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();
    // Open the same DB file directly to inspect schema.
    let path = format!("{}/karta.db", dir.path().to_str().unwrap());
    let conn = rusqlite::Connection::open(&path).unwrap();
    let sql: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'idx_facts_occurred'",
            [],
            |r| r.get(0),
        )
        .expect("idx_facts_occurred must exist");
    assert!(
        sql.contains("occurred_start"),
        "index must cover occurred_start"
    );
    assert!(sql.contains("WHERE"), "must be a partial index");
}
