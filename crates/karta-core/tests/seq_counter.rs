#![cfg(feature = "sqlite-vec")]
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::VectorStore;
use tempfile::TempDir;

#[tokio::test]
async fn seq_is_monotonic_and_persists() {
    let dir = TempDir::new().unwrap();
    let data_dir = dir.path().to_str().unwrap();
    {
        let store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
        assert_eq!(store.next_seq().await.unwrap(), 1);
        assert_eq!(store.next_seq().await.unwrap(), 2);
        assert_eq!(store.next_seq().await.unwrap(), 3);
    }
    // reopen same dir — counter must persist, not reset
    let store2 = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
    assert_eq!(store2.next_seq().await.unwrap(), 4);
}
