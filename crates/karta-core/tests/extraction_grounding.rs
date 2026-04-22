use karta_core::extract::grounding::{validate_supporting_spans, GroundingError};

const SOURCE: &str = "I'm working on a project with a Time Anchor of March 15, 2024, and I need to plan my tasks accordingly.";

#[test]
fn accepts_two_real_spans() {
    let spans = vec!["March 15, 2024".to_string(), "plan my tasks".to_string()];
    assert!(validate_supporting_spans(&spans, SOURCE).is_ok());
}

#[test]
fn rejects_empty_list() {
    assert_eq!(
        validate_supporting_spans(&[], SOURCE),
        Err(GroundingError::Empty),
    );
}

#[test]
fn rejects_span_not_in_source() {
    let spans = vec!["March 15, 2024".to_string(), "fabricated quote".to_string()];
    assert!(matches!(
        validate_supporting_spans(&spans, SOURCE),
        Err(GroundingError::NotInSource(_)),
    ));
}

#[test]
fn rejects_too_short_span() {
    let spans = vec!["on".to_string()];
    assert!(matches!(
        validate_supporting_spans(&spans, SOURCE),
        Err(GroundingError::TooShort(_)),
    ));
}

#[test]
fn rejects_whitespace_only_span() {
    let spans = vec!["   ".to_string()];
    assert!(matches!(
        validate_supporting_spans(&spans, SOURCE),
        Err(GroundingError::TooShort(_)),
    ));
}
