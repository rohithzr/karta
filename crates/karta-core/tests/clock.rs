//! T1, T2 — ClockContext constructor and accessor invariants.

use karta_core::clock::ClockContext;

#[test]
fn now_uses_current_wall_time() {
    let before = chrono::Utc::now();
    let ctx = ClockContext::now();
    let after = chrono::Utc::now();

    assert!(ctx.reference_time() >= before);
    assert!(ctx.reference_time() <= after);
}

#[test]
fn at_preserves_explicit_time() {
    let t = chrono::DateTime::parse_from_rfc3339("2024-03-15T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let ctx = ClockContext::at(t);
    assert_eq!(ctx.reference_time(), t);
}

#[test]
fn default_equals_now() {
    let before = chrono::Utc::now();
    let ctx = ClockContext::default();
    let after = chrono::Utc::now();
    assert!(ctx.reference_time() >= before);
    assert!(ctx.reference_time() <= after);
}

#[test]
fn copy_semantics_preserve_reference_time() {
    let t = chrono::DateTime::parse_from_rfc3339("2024-03-15T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let a = ClockContext::at(t);
    let b = a; // Copy
    assert_eq!(a.reference_time(), t);
    assert_eq!(b.reference_time(), t);
}
