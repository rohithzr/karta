//! Live smoke test for `Karta::with_defaults` against a local LM Studio server.
//!
//! This test is ignored by default because it requires LM Studio running on
//! `localhost:1234` with `text-embedding-nomic-embed-text-v1.5` loaded. Run it
//! explicitly with:
//!
//! ```bash
//! cargo test -p karta-mcp --test smoke_with_defaults -- --ignored
//! ```

use std::path::PathBuf;
use tempfile::TempDir;

#[tokio::test]
#[ignore = "requires LM Studio on localhost:1234"]
async fn with_defaults_smoke() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().to_string_lossy().into_owned();

    unsafe {
        std::env::set_var("OPENAI_API_BASE", "http://localhost:1234/v1");
        std::env::set_var("OPENAI_API_KEY", "lm-studio");
        std::env::set_var("KARTA_CORE_MODEL", "text-embedding-nomic-embed-text-v1.5");
        std::env::set_var(
            "KARTA_EMBEDDING_MODEL",
            "text-embedding-nomic-embed-text-v1.5",
        );
        std::env::set_var("KARTA_STORE_DIR", &data_dir);
    }

    let mut config = karta_core::config::KartaConfig::default();
    config.storage.data_dir = data_dir.clone();

    let karta = karta_core::Karta::with_defaults(config)
        .await
        .expect("Karta::with_defaults should succeed with LM Studio env");

    let count = karta
        .note_count()
        .await
        .expect("note_count should be readable");

    // The store should exist and be queryable even when empty.
    assert_eq!(count, 0);

    // Verify the expected database file was created.
    let db_path = PathBuf::from(data_dir).join("karta.db");
    assert!(
        db_path.exists(),
        "karta.db should exist after with_defaults"
    );
}
