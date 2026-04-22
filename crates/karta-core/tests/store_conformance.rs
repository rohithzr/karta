use chrono::Utc;
use tempfile::TempDir;

use karta_core::note::{MemoryNote, NoteStatus, Provenance};
use karta_core::store::VectorStore;

fn make_test_note(id: &str, content: &str, dim: usize) -> MemoryNote {
    let now = Utc::now();
    MemoryNote {
        id: id.to_string(),
        content: content.to_string(),
        context: format!("Context for {}", content),
        keywords: vec!["test".to_string()],
        tags: vec!["unit-test".to_string()],
        links: Vec::new(),
        embedding: vec![0.1; dim],
        created_at: now,
        updated_at: now,
        evolution_history: Vec::new(),
        provenance: Provenance::Observed,
        confidence: 0.9,
        status: NoteStatus::Active,
        last_accessed_at: now,
        turn_index: Some(1),
        source_timestamp: now,
        session_id: None,
    }
}

/// Generic conformance suite that any VectorStore impl should pass.
async fn conformance_suite(store: &dyn VectorStore, dim: usize) {
    // CRUD basics
    let note = make_test_note("c1", "conformance test", dim);
    store.upsert(&note).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);

    let fetched = store.get("c1").await.unwrap().unwrap();
    assert_eq!(fetched.content, "conformance test");
    assert_eq!(fetched.embedding.len(), dim);

    // Upsert overwrites
    let mut updated = make_test_note("c1", "updated content", dim);
    updated.embedding = vec![0.2; dim];
    store.upsert(&updated).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);
    let fetched = store.get("c1").await.unwrap().unwrap();
    assert_eq!(fetched.content, "updated content");

    // kNN ordering: closer vector should rank higher
    let mut n1 = make_test_note("knn1", "a", dim);
    n1.embedding = vec![1.0; dim];
    let mut n2 = make_test_note("knn2", "b", dim);
    n2.embedding = vec![0.0; dim];
    store.upsert(&n1).await.unwrap();
    store.upsert(&n2).await.unwrap();

    let query = vec![1.0; dim];
    let results = store.find_similar(&query, 2, &[]).await.unwrap();
    assert!(results.len() >= 2);
    assert_eq!(results[0].0.id, "knn1"); // exact match first
    assert!(results[0].1 > results[1].1); // higher score for closer vector

    // Score for exact match should be close to 1.0 (within tolerance)
    // 1/(1+0) = 1.0 for exact match with L2 distance
    assert!(
        (results[0].1 - 1.0).abs() < 0.05,
        "Exact match score {} not near 1.0",
        results[0].1
    );

    // Exclude IDs
    let results = store.find_similar(&query, 2, &["knn1"]).await.unwrap();
    assert!(results.iter().all(|(n, _)| n.id != "knn1"));

    // Delete removes from both storage and kNN
    store.delete("knn1").await.unwrap();
    let results = store.find_similar(&query, 10, &[]).await.unwrap();
    assert!(results.iter().all(|(n, _)| n.id != "knn1"));
    assert!(store.get("knn1").await.unwrap().is_none());

    // get_many with mixed existing/missing IDs
    let results = store.get_many(&["c1", "nonexistent"]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "c1");

    // get_all returns everything still in store
    let all = store.get_all().await.unwrap();
    assert!(all.len() >= 2); // c1 + knn2 at minimum
}

#[tokio::test]
#[cfg(feature = "sqlite-vec")]
async fn conformance_sqlite_vec() {
    use karta_core::store::sqlite_vec::SqliteVectorStore;
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4)
        .await
        .unwrap();
    conformance_suite(&store, 4).await;
}

#[tokio::test]
#[cfg(feature = "lance")]
async fn conformance_lance() {
    use karta_core::store::lance::LanceVectorStore;
    let dir = TempDir::new().unwrap();
    let store = LanceVectorStore::new(dir.path().to_str().unwrap(), 4)
        .await
        .unwrap();
    conformance_suite(&store, 4).await;
}
