//! Slot-level dedup: collapse facts sharing
//! `(entity_text.to_lower(), facet, value_key)` to a single fact.
//! `value_key` is `value_date.date_naive()` if present, else
//! `value_text.to_lower()`, else "(none)".
//!
//! ## Keep-first with span-merge
//!
//! When two facts collapse to the same slot, the FIRST occurrence (by
//! original LLM emission order) is kept as the canonical row. The
//! `supporting_spans` from all dropped siblings are merged into the
//! kept fact's span list, deduplicated by string equality.
//!
//! This preserves the evidence trail: if the LLM emitted two valid
//! phrasings of the same claim ("April 15 deadline" vs "due by the 15th"),
//! both citations end up on the surviving row instead of one being
//! silently lost.
//!
//! Note: occurred_* / value_date / value_text from dropped siblings are
//! NOT merged today. If empirical traces show this dropping useful
//! temporal info, see Task 16 future-work for the promotion fix.

use std::collections::HashMap;

use crate::note::AtomicFactExtraction;

pub fn dedup_extractions(facts: Vec<AtomicFactExtraction>) -> Vec<AtomicFactExtraction> {
    // Map slot_key → index into `out` for the canonical fact.
    let mut slot_to_idx: HashMap<(String, String, String), usize> = HashMap::new();
    let mut out: Vec<AtomicFactExtraction> = Vec::with_capacity(facts.len());

    for f in facts {
        let key = slot_key(&f);
        if let Some(&idx) = slot_to_idx.get(&key) {
            // Collision: merge this sibling's spans into the canonical fact.
            for span in f.supporting_spans {
                if !out[idx].supporting_spans.contains(&span) {
                    out[idx].supporting_spans.push(span);
                }
            }
        } else {
            slot_to_idx.insert(key, out.len());
            out.push(f);
        }
    }

    out
}

fn slot_key(f: &AtomicFactExtraction) -> (String, String, String) {
    let entity = f.entity_text.as_deref().unwrap_or("(none)").to_lowercase();
    let facet = format!("{:?}", f.facet);
    let value_key = if let Some(d) = f.value_date {
        d.date_naive().to_string()
    } else if let Some(t) = &f.value_text {
        t.to_lowercase()
    } else {
        "(none)".to_string()
    };
    (entity, facet, value_key)
}
