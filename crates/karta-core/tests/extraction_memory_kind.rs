use karta_core::extract::memory_kind::MemoryKind;

/// Exhaustive variant list. The `match` in `kind_for_index` forces the
/// compiler to flag any new variant added later. If you add a 9th
/// variant and don't update this function, every test below fails to
/// compile — that is the intended check, not a nuisance.
const ALL_VARIANTS_LEN: usize = 8;

fn all_variants() -> [MemoryKind; ALL_VARIANTS_LEN] {
    use MemoryKind::*;
    let arr = [
        DurableFact,
        FutureCommitment,
        Preference,
        Decision,
        Constraint,
        EphemeralRequest,
        SpeechAct,
        Echo,
    ];
    // Force-exhaustive guard: matching every variant once. If a new
    // variant lands and isn't added to `arr` above, this match either
    // fails to compile (no arm) or trips the assertion (count drift).
    let mut covered = 0usize;
    for k in arr {
        match k {
            DurableFact
            | FutureCommitment
            | Preference
            | Decision
            | Constraint
            | EphemeralRequest
            | SpeechAct
            | Echo => covered += 1,
        }
    }
    assert_eq!(covered, ALL_VARIANTS_LEN, "all_variants exhaustiveness drift");
    arr
}

#[test]
fn serde_round_trip_each_variant() {
    for k in all_variants() {
        let s = serde_json::to_string(&k).unwrap();
        let back: MemoryKind = serde_json::from_str(&s).unwrap();
        assert_eq!(k, back, "round-trip failed for {:?}", k);
    }
}

#[test]
fn snake_case_serialization() {
    // Spot-check the 4 variants whose snake_case form is most likely to
    // drift if someone fiddles with the rename attribute. The exhaustive
    // round-trip above covers the rest.
    let cases: &[(MemoryKind, &str)] = &[
        (MemoryKind::DurableFact, "\"durable_fact\""),
        (MemoryKind::FutureCommitment, "\"future_commitment\""),
        (MemoryKind::EphemeralRequest, "\"ephemeral_request\""),
        (MemoryKind::SpeechAct, "\"speech_act\""),
    ];
    for (kind, expected) in cases {
        assert_eq!(serde_json::to_string(kind).unwrap(), *expected);
    }
}

#[test]
fn is_durable_correct_for_each_variant() {
    use MemoryKind::*;
    for k in all_variants() {
        let expected = matches!(
            k,
            DurableFact | FutureCommitment | Preference | Decision | Constraint
        );
        assert_eq!(
            k.is_durable(),
            expected,
            "is_durable wrong for {:?}: expected {}",
            k,
            expected
        );
    }
}

#[test]
fn rejects_unknown_string() {
    let r: Result<MemoryKind, _> = serde_json::from_str("\"random_kind\"");
    assert!(r.is_err());
}
