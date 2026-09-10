#![cfg(feature = "sqlite-vec")]
use std::sync::Arc;
use karta_core::config::KartaConfig;
use karta_core::llm::MockLlmProvider;
use karta_core::store::sqlite::SqliteGraphStore;
use karta_core::store::sqlite_vec::SqliteVectorStore;
use karta_core::store::{GraphStore, VectorStore};
use karta_core::Karta;
use tempfile::TempDir;

#[tokio::test]
async fn notes_get_increasing_seq() {
    let dir = TempDir::new().unwrap();
    let data_dir = dir.path().to_str().unwrap();
    let vec_store = SqliteVectorStore::new(data_dir, 1536).await.unwrap();
    let conn = vec_store.connection();
    let graph = SqliteGraphStore::with_connection(conn);
    let vector_store = Arc::new(vec_store) as Arc<dyn VectorStore>;
    let graph_store = Arc::new(graph) as Arc<dyn GraphStore>;
    let llm = Arc::new(MockLlmProvider::new());
    let karta = Karta::new(vector_store.clone(), graph_store, llm, KartaConfig::default())
        .await.unwrap();

    let a = karta.add_note("first note").await.unwrap();
    let b = karta.add_note("second note").await.unwrap();
    let a2 = vector_store.get(&a.id).await.unwrap().unwrap();
    let b2 = vector_store.get(&b.id).await.unwrap().unwrap();
    assert!(b2.seq > a2.seq, "seq must increase: a={} b={}", a2.seq, b2.seq);
}
