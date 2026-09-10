//! Grounding gate: each fact's `supporting_spans` must be real substrings
//! of the source NOTE content, each at least `MIN_SPAN_CHARS` long after
//! trimming.

/// Minimum span length in characters (post-trim).
///
/// **Why 4:** admits 4-character years like "2024" and short month/day
/// numbers like "Apr 5" (5 chars), while rejecting 1-3 char English
/// stopwords like "of", "the", "on", "in", "an". The LLM occasionally
/// picks a stopword as the "evidence" when it can't find a real temporal
/// phrase — this floor catches that without a semantic parser.
///
/// **Tradeoff:** 4 chars also admits non-temporal common words like
/// "used", "same", "have". The grounding gate is one of three filters
/// (admission + grounding + specificity); a 4-char common-word span
/// passing grounding doesn't mean the fact survives — it still needs a
/// typed entity_type or facet to clear specificity.
///
/// **If raising:** 6 would still admit "Apr 15", "March", "Monday";
/// would reject "2024" (4-char year) and "Apr 5" (5 chars). Don't raise
/// without checking calibration data — most temporal phrases actually
/// emitted in BEAM/LongMem are 6-25 chars.
const MIN_SPAN_CHARS: usize = 4;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GroundingError {
    #[error("supporting_spans is empty — cannot ground the fact")]
    Empty,
    #[error("span is too short (<{min} chars after trim): {0:?}", min = MIN_SPAN_CHARS)]
    TooShort(String),
    #[error("span not found verbatim in source: {0:?}")]
    NotInSource(String),
}

pub fn validate_supporting_spans(spans: &[String], source: &str) -> Result<(), GroundingError> {
    if spans.is_empty() {
        return Err(GroundingError::Empty);
    }
    for span in spans {
        let trimmed = span.trim();
        if trimmed.chars().count() < MIN_SPAN_CHARS {
            return Err(GroundingError::TooShort(span.clone()));
        }
        if !source.contains(trimmed) {
            return Err(GroundingError::NotInSource(span.clone()));
        }
    }
    Ok(())
}
