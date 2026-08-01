//! Operator CLI actions: status, backup, export, restore.
//!
//! These functions are used by the `status`, `backup`, `export`, and `restore`
//! subcommands in `main.rs`. They are kept separate from `main.rs` so they
//! can be exercised directly by in-process integration tests.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rusqlite::Connection;

use crate::config::Config;
use crate::karta_handle::KartaHandle;

/// Return the five status fields as a formatted string.
///
/// The string is intended to be printed to stdout by the `status` subcommand.
/// No MCP server or HTTP endpoint is started.
pub async fn status(config: &Config) -> Result<String> {
    let handle = KartaHandle::open_mock_for_data_dir(config.store_dir())
        .await
        .context("failed to open store for status")?;
    let karta = handle.karta;
    let note_count = karta.note_count().await?;
    let queue = crate::queue::CaptureQueue::new(config.store_dir()).await?;
    let queue_depth = queue.depth().await?;

    let embedding_model = std::env::var("KARTA_EMBEDDING_MODEL")
        .unwrap_or_else(|_| config.core.llm.default.model.clone());

    let mut output = String::new();
    output.push_str(&format!("note_count: {note_count}\n"));
    output.push_str(&format!("store_dir: {}\n", config.store_dir()));
    output.push_str(&format!("embedding_model: {embedding_model}\n"));
    output.push_str(&format!("capture_port: {}\n", config.capture_port));
    output.push_str(&format!("queue_depth: {queue_depth}\n"));
    Ok(output)
}

/// Create an online snapshot of the store at `source` into `dest`.
///
/// Uses `VACUUM INTO` so the resulting SQLite file is a consistent
/// point-in-time copy that includes vectors, graph, slot ledger, and the
/// capture queue. `VACUUM INTO` is safe on a live WAL database.
pub async fn backup(source: &Path, dest: &Path) -> Result<()> {
    if !source.exists() {
        bail!("store file not found: {}", source.display());
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create destination directory: {}",
                parent.display()
            )
        })?;
    }

    // VACUUM INTO requires the destination to not exist.
    if dest.exists() {
        std::fs::remove_file(dest).with_context(|| {
            format!("failed to remove existing destination: {}", dest.display())
        })?;
    }

    let conn = Connection::open(source)
        .with_context(|| format!("failed to open store for backup: {}", source.display()))?;
    conn.execute_batch("PRAGMA busy_timeout = 5000;")
        .context("failed to set busy_timeout on backup connection")?;

    // Use a literal path because SQLite's VACUUM INTO does not accept bound
    // parameters. Escaping single quotes keeps the generated SQL safe.
    let escaped = escape_sql_string_literal(&dest.to_string_lossy());
    let sql = format!("VACUUM INTO '{escaped}'");
    conn.execute_batch(&sql)
        .with_context(|| format!("VACUUM INTO failed for destination: {}", dest.display()))?;

    Ok(())
}

/// Escape a path for use as a single-quoted SQL string literal.
fn escape_sql_string_literal(s: &str) -> String {
    s.replace('\'', "''")
}

/// Export all notes to markdown files under `dest`.
///
/// Returns the number of notes exported. One `.md` file is written per note.
/// The files include provenance (`FACT`/`INFERRED`), confidence, and source
/// back-pointers where applicable.
pub async fn export(handle: &KartaHandle, dest: &Path) -> Result<usize> {
    std::fs::create_dir_all(dest)
        .with_context(|| format!("failed to create export directory: {}", dest.display()))?;

    let notes = handle.karta.get_all_notes().await?;
    for note in &notes {
        let filename = format!("{}_{}.md", note.created_at.format("%Y%m%d-%H%M%S"), note.id);
        let path = dest.join(sanitize_filename(&filename));
        let markdown = note_to_markdown(note);
        std::fs::write(&path, markdown)
            .with_context(|| format!("failed to write export file: {}", path.display()))?;
    }

    Ok(notes.len())
}

/// Restore `karta.db` in `data_dir` from the backup file at `from`.
///
/// Requires that no other process (in particular `karta-mcp serve`) is holding
/// the store open. If the store appears to be locked, the restore aborts with
/// a clear error.
pub async fn restore(from: &Path, data_dir: &Path) -> Result<()> {
    if !from.exists() {
        bail!("backup file not found: {}", from.display());
    }

    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("failed to create store directory: {}", data_dir.display()))?;

    let store_path = data_dir.join("karta.db");

    if store_path.exists() {
        if is_store_locked(&store_path)? {
            bail!(
                "store is locked by another process (is karta-mcp serve running?). \
                 Stop serve before restore."
            );
        }
        std::fs::remove_file(&store_path).with_context(|| {
            format!("failed to remove existing store: {}", store_path.display())
        })?;
    }

    // Remove stale WAL files from the previous store; the backup is a fresh
    // consistent snapshot and does not need them.
    let _ = std::fs::remove_file(store_path.with_extension("db-wal"));
    let _ = std::fs::remove_file(store_path.with_extension("db-shm"));

    std::fs::copy(from, &store_path)
        .with_context(|| format!("failed to copy backup to store: {}", store_path.display()))?;

    Ok(())
}

/// Check whether the store is currently locked by another connection.
///
/// Tries to begin an immediate transaction with a short busy timeout. If the
/// database is busy, another process is likely holding the store open.
fn is_store_locked(path: &Path) -> Result<bool> {
    let conn = Connection::open(path)
        .with_context(|| format!("failed to open store for lock check: {}", path.display()))?;
    conn.execute_batch("PRAGMA busy_timeout = 1000;")
        .context("failed to set busy_timeout on lock check connection")?;
    match conn.execute_batch("BEGIN IMMEDIATE; ROLLBACK;") {
        Ok(()) => Ok(false),
        Err(e) if e.to_string().contains("database is locked") => Ok(true),
        Err(e) => Err(e.into()),
    }
}

/// Render a `MemoryNote` as reviewable markdown.
fn note_to_markdown(note: &karta_core::note::MemoryNote) -> String {
    use karta_core::note::Provenance;

    let provenance_label = match &note.provenance {
        Provenance::Observed | Provenance::Fact { .. } => "FACT",
        Provenance::Dream { .. }
        | Provenance::Profile { .. }
        | Provenance::Episode { .. }
        | Provenance::Digest { .. } => "INFERRED",
    };

    let provenance_detail = match &note.provenance {
        Provenance::Observed => "directly observed".to_string(),
        Provenance::Fact { source_note_id } => {
            format!("promoted from source note {source_note_id}")
        }
        Provenance::Dream {
            dream_type,
            source_note_ids,
            confidence,
        } => format!(
            "{dream_type} dream (confidence {confidence:.2}) from notes {}",
            source_note_ids.join(", ")
        ),
        Provenance::Profile { entity_id } => format!("profile for entity {entity_id}"),
        Provenance::Episode { episode_id } => format!("episode {episode_id}"),
        Provenance::Digest { episode_id } => format!("digest of episode {episode_id}"),
    };

    let tags = if note.tags.is_empty() {
        "none".to_string()
    } else {
        note.tags.join(", ")
    };

    let links = if note.links.is_empty() {
        "none".to_string()
    } else {
        note.links.join(", ")
    };

    let session = note.session_id.as_deref().unwrap_or("none");

    let context = if note.context.is_empty() {
        "_No generated context._".to_string()
    } else {
        note.context.clone()
    };

    format!(
        "# Note {id}

**Provenance:** {provenance_label} ({provenance_detail})
**Confidence:** {confidence:.2}
**Status:** {status:?}
**Created:** {created}
**Updated:** {updated}
**Session:** {session}
**Tags:** {tags}
**Links:** {links}

## Content

{content}

## Context

{context}
",
        id = note.id,
        provenance_label = provenance_label,
        provenance_detail = provenance_detail,
        confidence = note.confidence,
        status = note.status,
        created = note.created_at.to_rfc3339(),
        updated = note.updated_at.to_rfc3339(),
        session = session,
        tags = tags,
        links = links,
        content = note.content,
        context = context,
    )
}

/// Sanitize a filename by replacing filesystem-risky characters.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use karta_core::note::{MemoryNote, Provenance};

    async fn setup_handle() -> (TempDir, KartaHandle) {
        let dir = TempDir::new().unwrap();
        let handle = KartaHandle::open_mock(dir.path().to_str().unwrap())
            .await
            .unwrap();
        (dir, handle)
    }

    #[tokio::test]
    async fn status_reports_zero_fields_for_empty_store() {
        let (dir, _handle) = setup_handle().await;
        let mut config = Config::from_env().unwrap();
        config.core.storage.data_dir = dir.path().to_str().unwrap().to_string();
        let output = status(&config).await.unwrap();
        assert!(output.contains("note_count: 0"));
        assert!(output.contains(&format!("store_dir: {}", dir.path().to_str().unwrap())));
        assert!(output.contains("capture_port:"));
        assert!(output.contains("queue_depth: 0"));
        assert!(output.contains("embedding_model:"));
    }

    #[tokio::test]
    async fn backup_and_restore_round_trip() {
        let (dir, handle) = setup_handle().await;
        let data_dir = dir.path().to_str().unwrap();

        handle
            .karta
            .add_note_with_clock(
                "round-trip note",
                Some("s1"),
                None,
                karta_core::ClockContext::now(),
            )
            .await
            .unwrap();

        let source = Path::new(data_dir).join("karta.db");
        let backup_path = dir.path().join("backup.db");
        backup(&source, &backup_path).await.unwrap();
        assert!(backup_path.exists());

        let restore_dir = TempDir::new().unwrap();
        restore(&backup_path, restore_dir.path()).await.unwrap();

        let restored_handle =
            KartaHandle::open_mock_for_data_dir(restore_dir.path().to_str().unwrap())
                .await
                .unwrap();
        assert_eq!(restored_handle.karta.note_count().await.unwrap(), 1);
        let note = restored_handle
            .karta
            .get_all_notes()
            .await
            .unwrap()
            .pop()
            .unwrap();
        assert!(note.content.contains("round-trip note"));
        assert_eq!(note.session_id, Some("s1".to_string()));
    }

    #[tokio::test]
    async fn export_writes_markdown_with_provenance() {
        let (dir, handle) = setup_handle().await;
        let _data_dir = dir.path().to_str().unwrap();

        let mut note = MemoryNote::new("high confidence observation".to_string());
        note.provenance = Provenance::Observed;
        note.confidence = 0.95;
        note.session_id = Some("export-session".to_string());
        note.tags = vec!["tag1".to_string()];
        const DIM: usize = 1536;
        let value = 1.0 / (DIM as f32).sqrt();
        note.embedding = vec![value; DIM];
        handle.vector_store.upsert(&note).await.unwrap();

        let export_dir = dir.path().join("export");
        let count = export(&handle, &export_dir).await.unwrap();
        assert_eq!(count, 1);

        let entries: Vec<_> = std::fs::read_dir(&export_dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(entries.len(), 1);
        let content = std::fs::read_to_string(&entries[0]).unwrap();
        assert!(content.contains("high confidence observation"));
        assert!(content.contains("**Provenance:** FACT"));
        assert!(content.contains("**Confidence:** 0.95"));
        assert!(content.contains("export-session"));
        assert!(content.contains("tag1"));
    }

    #[tokio::test]
    async fn export_includes_source_back_pointer_for_fact() {
        let (dir, handle) = setup_handle().await;

        let mut source = MemoryNote::new("source observation".to_string());
        source.provenance = Provenance::Observed;
        source.confidence = 0.95;
        source.session_id = Some("fact-session".to_string());
        const DIM: usize = 1536;
        let value = 1.0 / (DIM as f32).sqrt();
        source.embedding = vec![value; DIM];
        handle.vector_store.upsert(&source).await.unwrap();

        let mut fact = MemoryNote::new("promoted fact".to_string());
        fact.provenance = Provenance::Fact {
            source_note_id: source.id.clone(),
        };
        fact.confidence = source.confidence;
        fact.session_id = source.session_id.clone();
        fact.embedding = source.embedding.clone();
        handle.vector_store.upsert(&fact).await.unwrap();

        let export_dir = dir.path().join("export-fact");
        let count = export(&handle, &export_dir).await.unwrap();
        assert_eq!(count, 2);

        let files: Vec<_> = std::fs::read_dir(&export_dir)
            .unwrap()
            .map(|e| std::fs::read_to_string(e.unwrap().path()).unwrap())
            .collect();
        let fact_file = files
            .iter()
            .find(|s| s.contains("promoted fact"))
            .expect("fact note exported");
        assert!(fact_file.contains("**Provenance:** FACT"));
        assert!(fact_file.contains(&source.id));
        assert!(fact_file.contains("promoted from source note"));
    }

    #[tokio::test]
    async fn restore_rejects_locked_store() {
        let (dir, _handle) = setup_handle().await;
        let _data_dir = dir.path().to_str().unwrap();
        let store_path = dir.path().join("karta.db");

        let backup_path = dir.path().join("backup.db");
        backup(&store_path, &backup_path).await.unwrap();

        // Open a connection and hold a transaction to lock the store.
        let conn = Connection::open(&store_path).unwrap();
        conn.execute_batch("BEGIN IMMEDIATE;").unwrap();

        let err = restore(&backup_path, dir.path()).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("locked") || msg.contains("serve running"),
            "expected lock error, got: {msg}"
        );
    }
}
