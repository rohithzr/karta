use karta_core::extract::admission::{validate_admission, AdmissionError};
use karta_core::extract::memory_kind::MemoryKind;

#[test]
fn admits_durable_kinds() {
    for k in [
        MemoryKind::DurableFact,
        MemoryKind::FutureCommitment,
        MemoryKind::Preference,
        MemoryKind::Decision,
        MemoryKind::Constraint,
    ] {
        assert!(validate_admission(k).is_ok(), "{:?} should be admitted", k);
    }
}

#[test]
fn rejects_ephemeral_request() {
    let r = validate_admission(MemoryKind::EphemeralRequest);
    assert!(matches!(r, Err(AdmissionError::Ephemeral(MemoryKind::EphemeralRequest))));
}

#[test]
fn rejects_speech_act() {
    let r = validate_admission(MemoryKind::SpeechAct);
    assert!(matches!(r, Err(AdmissionError::Ephemeral(MemoryKind::SpeechAct))));
}

#[test]
fn rejects_echo() {
    let r = validate_admission(MemoryKind::Echo);
    assert!(matches!(r, Err(AdmissionError::Ephemeral(MemoryKind::Echo))));
}
