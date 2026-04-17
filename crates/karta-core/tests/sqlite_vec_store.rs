use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::VectorStore;
use tempfile::TempDir;

#[tokio::test]
async fn test_create_store() {
    let dir = TempDir::new().unwrap();
    let store = SqliteVectorStore::new(dir.path().to_str().unwrap(), 1536)
        .await
        .unwrap();
    let count = store.count().await.unwrap();
    assert_eq!(count, 0);
}
