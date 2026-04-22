//! Admission gate: reject ephemeral memory kinds before storage.

use super::memory_kind::MemoryKind;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AdmissionError {
    #[error("ephemeral memory_kind {0:?} is not admitted to long-term storage")]
    Ephemeral(MemoryKind),
}

/// Returns `Ok(())` if the kind is durable; `Err` if it's a speech act,
/// echo, or request. Caller drops the fact on `Err`.
pub fn validate_admission(kind: MemoryKind) -> Result<(), AdmissionError> {
    if kind.is_durable() {
        Ok(())
    } else {
        Err(AdmissionError::Ephemeral(kind))
    }
}
