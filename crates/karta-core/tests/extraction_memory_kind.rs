use karta_core::extract::memory_kind::MemoryKind;

#[test]
fn serde_round_trip_each_variant() {
    for k in [
        MemoryKind::DurableFact,
        MemoryKind::FutureCommitment,
        MemoryKind::Preference,
        MemoryKind::Decision,
        MemoryKind::Constraint,
        MemoryKind::EphemeralRequest,
        MemoryKind::SpeechAct,
        MemoryKind::Echo,
    ] {
        let s = serde_json::to_string(&k).unwrap();
        let back: MemoryKind = serde_json::from_str(&s).unwrap();
        assert_eq!(k, back, "round-trip failed for {:?}", k);
    }
}

#[test]
fn snake_case_serialization() {
    assert_eq!(serde_json::to_string(&MemoryKind::DurableFact).unwrap(), "\"durable_fact\"");
    assert_eq!(serde_json::to_string(&MemoryKind::FutureCommitment).unwrap(), "\"future_commitment\"");
    assert_eq!(serde_json::to_string(&MemoryKind::EphemeralRequest).unwrap(), "\"ephemeral_request\"");
    assert_eq!(serde_json::to_string(&MemoryKind::SpeechAct).unwrap(), "\"speech_act\"");
}

#[test]
fn is_durable_correct_for_each_variant() {
    assert!(MemoryKind::DurableFact.is_durable());
    assert!(MemoryKind::FutureCommitment.is_durable());
    assert!(MemoryKind::Preference.is_durable());
    assert!(MemoryKind::Decision.is_durable());
    assert!(MemoryKind::Constraint.is_durable());
    assert!(!MemoryKind::EphemeralRequest.is_durable());
    assert!(!MemoryKind::SpeechAct.is_durable());
    assert!(!MemoryKind::Echo.is_durable());
}

#[test]
fn rejects_unknown_string() {
    let r: Result<MemoryKind, _> = serde_json::from_str("\"random_kind\"");
    assert!(r.is_err());
}
