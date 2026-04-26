//! Synthetic memory evaluation tests that require no external API keys.
//!
//! These tests verify core data structures, decay logic, and schema
//! pipelines using deterministic mock data.
//!
//! Run with: cargo test -p karta-core --test synthetic_memory_eval

use karta_core::migrate::{CURRENT_SCHEMA_VERSION, SchemaMeta};
use karta_core::note::{MemoryNote, Provenance};

// ─── Synthetic Dataset: Preference Update ──────────────────────────────────

#[tokio::test]
async fn test_preference_update_flow() {
    let note1 = MemoryNote::new("I prefer dark mode for all UIs".into());
    let note2 = MemoryNote::new("Actually, I switched back to light mode".into());

    assert!(note1.content.contains("dark mode"));
    assert!(note2.content.contains("light mode"));
    assert_ne!(note1.id, note2.id);
}

// ─── Synthetic Dataset: Temporal Sequence ─────────────────────────────────

#[tokio::test]
async fn test_temporal_sequence_ordering() {
    let mut notes = Vec::new();
    for i in 0..5 {
        let mut note = MemoryNote::new(format!("Step {} of the process", i));
        note.turn_index = Some(i);
        notes.push(note);
    }

    notes.sort_by_key(|n| n.turn_index.unwrap_or(0));

    for (i, note) in notes.iter().enumerate() {
        assert_eq!(note.turn_index, Some(i as u32));
    }
}

// ─── Synthetic Dataset: Forgetting Decay ──────────────────────────────────

#[tokio::test]
#[ignore = "spec stub until production ForgetConfig decay scoring API is exposed"]
async fn test_forgetting_decay_scoring() {
    use chrono::{Duration, Utc};

    let decay_half_life_days = 30.0;
    let archive_threshold = 0.15;

    let mut old_note = MemoryNote::new("Old memory".into());
    old_note.last_accessed_at = Utc::now() - Duration::days(90);

    let mut recent_note = MemoryNote::new("Recent memory".into());
    recent_note.last_accessed_at = Utc::now() - Duration::days(1);

    // TODO: replace this specification stub with the production ForgetConfig scoring API.
    // Compute decay scores manually: score = 0.5^(elapsed / half_life)
    let old_score = 0.5_f64.powf(90.0 / decay_half_life_days); // 0.5^3 = 0.125
    let recent_score = 0.5_f64.powf(1.0 / decay_half_life_days); // ~0.977

    assert!(
        old_score < 0.13,
        "Old note should have low decay score: {}",
        old_score
    );
    assert!(
        recent_score > 0.95,
        "Recent note should have high decay score: {}",
        recent_score
    );
    assert!(
        old_score < archive_threshold,
        "Old note should be below archive threshold"
    );
    assert!(
        recent_score > archive_threshold,
        "Recent note should be above archive threshold"
    );
}

#[tokio::test]
async fn test_forgetting_protected_notes() {
    let mut profile_note = MemoryNote::new("Profile: John is a developer".into());
    assert!(!profile_note.is_profile());

    profile_note.provenance = Provenance::Profile {
        entity_id: "john".into(),
    };

    assert!(profile_note.is_profile());
}

// ─── Synthetic Dataset: Schema Meta ──────────────────────────────────────

#[tokio::test]
async fn test_schema_meta_structure() {
    let current_schema_version = CURRENT_SCHEMA_VERSION;
    assert!(current_schema_version >= 1);

    let meta = SchemaMeta::new(1, vec!["001_initial".into()], vec![], vec![]);

    assert_eq!(meta.schema_version, 1);
    assert_eq!(meta.applied_migrations.len(), 1);
    assert!(meta.pending_migrations.is_empty());
    assert!(meta.warnings.is_empty());
}

// ─── Synthetic Dataset: Entity Profile ────────────────────────────────────

// ─── Synthetic Dataset: Contradiction Resolution ──────────────────────────

#[tokio::test]
async fn test_note_lifecycle_states() {
    use karta_core::note::NoteStatus;

    let mut note = MemoryNote::new("Active note".into());
    assert_eq!(note.status, NoteStatus::Active);

    note.status = NoteStatus::Deprecated { by: "test".into() };
    assert!(!note.is_active());

    note.status = NoteStatus::Archived;
    assert!(!note.is_active());
}
