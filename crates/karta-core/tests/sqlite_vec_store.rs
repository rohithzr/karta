use chrono::Utc;
use karta_core::note::{AtomicFact, MemoryNote, NoteStatus, Provenance};
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::VectorStore;
use tempfile::TempDir;

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

#[tokio::test]
async fn test_create_store() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();
    let count = store.count().await.unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_upsert_and_get() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    let note = make_test_note("n1", "Hello world", 4);
    store.upsert(&note).await.unwrap();
    let fetched = store.get("n1").await.unwrap().expect("note should exist");
    assert_eq!(fetched.id, "n1");
    assert_eq!(fetched.content, "Hello world");
    assert_eq!(fetched.embedding.len(), 4);
    assert_eq!(fetched.keywords, vec!["test"]);
    assert_eq!(fetched.confidence, 0.9);
    assert!(fetched.turn_index == Some(1));
}

#[tokio::test]
async fn test_upsert_overwrites() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    let mut note = make_test_note("n1", "Version 1", 4);
    store.upsert(&note).await.unwrap();
    note.content = "Version 2".to_string();
    store.upsert(&note).await.unwrap();
    let fetched = store.get("n1").await.unwrap().unwrap();
    assert_eq!(fetched.content, "Version 2");
    assert_eq!(store.count().await.unwrap(), 1);
}

#[tokio::test]
async fn test_get_missing_returns_none() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    assert!(store.get("nonexistent").await.unwrap().is_none());
}

#[tokio::test]
async fn test_get_many_mixed_ids() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    store.upsert(&make_test_note("n1", "exists", 4)).await.unwrap();
    let results = store.get_many(&["n1", "n999"]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "n1");
}

#[tokio::test]
async fn test_blob_round_trip() {
    let values = vec![0.0f32, -1.0, 1e-38, std::f32::consts::PI, 999.999];
    let blob = SqliteVectorStore::embedding_to_blob(&values);
    let back = SqliteVectorStore::blob_to_embedding(&blob);
    assert_eq!(values, back);
}

#[tokio::test]
async fn test_delete_and_count() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    store.upsert(&make_test_note("n1", "one", 4)).await.unwrap();
    store.upsert(&make_test_note("n2", "two", 4)).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 2);
    store.delete("n1").await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);
    assert!(store.get("n1").await.unwrap().is_none());
}

#[tokio::test]
async fn test_get_all() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    store.upsert(&make_test_note("n1", "one", 4)).await.unwrap();
    store.upsert(&make_test_note("n2", "two", 4)).await.unwrap();
    let all = store.get_all().await.unwrap();
    assert_eq!(all.len(), 2);
}

#[tokio::test]
async fn test_find_similar() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();

    let mut n1 = make_test_note("n1", "cats are great", 4);
    n1.embedding = vec![1.0, 0.0, 0.0, 0.0];
    store.upsert(&n1).await.unwrap();

    let mut n2 = make_test_note("n2", "dogs are great", 4);
    n2.embedding = vec![0.9, 0.1, 0.0, 0.0];
    store.upsert(&n2).await.unwrap();

    let mut n3 = make_test_note("n3", "fish are great", 4);
    n3.embedding = vec![0.0, 0.0, 1.0, 0.0];
    store.upsert(&n3).await.unwrap();

    let results = store.find_similar(&[1.0, 0.0, 0.0, 0.0], 2, &[]).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0.id, "n1");
    assert_eq!(results[1].0.id, "n2");
    assert!(results[0].1 > results[1].1);
}

#[tokio::test]
async fn test_find_similar_with_excludes() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();

    let mut n1 = make_test_note("n1", "cats", 4);
    n1.embedding = vec![1.0, 0.0, 0.0, 0.0];
    store.upsert(&n1).await.unwrap();

    let mut n2 = make_test_note("n2", "dogs", 4);
    n2.embedding = vec![0.9, 0.1, 0.0, 0.0];
    store.upsert(&n2).await.unwrap();

    let results = store.find_similar(&[1.0, 0.0, 0.0, 0.0], 2, &["n1"]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, "n2");
}

#[tokio::test]
async fn test_find_similar_empty_store() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    let results = store.find_similar(&[1.0, 0.0, 0.0, 0.0], 5, &[]).await.unwrap();
    assert!(results.is_empty());
}

fn make_test_fact(id: &str, content: &str, source_note_id: &str, ordinal: u32, dim: usize) -> AtomicFact {
    AtomicFact {
        id: id.to_string(),
        content: content.to_string(),
        source_note_id: source_note_id.to_string(),
        ordinal,
        subject: Some("test-entity".to_string()),
        embedding: vec![0.5; dim],
        created_at: Utc::now(),
    }
}

#[tokio::test]
async fn test_upsert_and_get_facts() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    let f1 = make_test_fact("f1", "Cats have four legs", "n1", 0, 4);
    let f2 = make_test_fact("f2", "Cats are mammals", "n1", 1, 4);
    store.upsert_fact(&f1).await.unwrap();
    store.upsert_fact(&f2).await.unwrap();
    let facts = store.get_facts_for_note("n1").await.unwrap();
    assert_eq!(facts.len(), 2);
    assert_eq!(facts[0].ordinal, 0);
    assert_eq!(facts[1].ordinal, 1);
}

#[tokio::test]
async fn test_find_similar_facts() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();
    let mut f1 = make_test_fact("f1", "fact one", "n1", 0, 4);
    f1.embedding = vec![1.0, 0.0, 0.0, 0.0];
    store.upsert_fact(&f1).await.unwrap();
    let mut f2 = make_test_fact("f2", "fact two", "n2", 0, 4);
    f2.embedding = vec![0.0, 1.0, 0.0, 0.0];
    store.upsert_fact(&f2).await.unwrap();
    // Search near f1, exclude n1's source
    let results = store.find_similar_facts(&[1.0, 0.0, 0.0, 0.0], 2, &["n1"]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, "f2");
}

#[tokio::test]
async fn test_delete_removes_from_knn() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 4).await.unwrap();

    let mut n1 = make_test_note("n1", "cat", 4);
    n1.embedding = vec![1.0, 0.0, 0.0, 0.0];
    store.upsert(&n1).await.unwrap();

    store.delete("n1").await.unwrap();

    let results = store.find_similar(&[1.0, 0.0, 0.0, 0.0], 5, &[]).await.unwrap();
    assert!(results.is_empty());
}
