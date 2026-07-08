use std::collections::{HashMap, HashSet};

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::llm::{ChatMessage, GenConfig, LlmProvider, Prompts, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Predicate {
    RoleTitle, Employer, Location, Status, Deadline, ScheduledDate,
    Count, Amount, TechChoice, Preference, Ownership, MetricValue,
}

impl Predicate {
    pub fn all() -> &'static [Predicate] {
        use Predicate::*;
        &[RoleTitle, Employer, Location, Status, Deadline, ScheduledDate,
          Count, Amount, TechChoice, Preference, Ownership, MetricValue]
    }
    pub fn as_str(&self) -> &'static str {
        use Predicate::*;
        match self {
            RoleTitle => "role_title", Employer => "employer", Location => "location",
            Status => "status", Deadline => "deadline", ScheduledDate => "scheduled_date",
            Count => "count", Amount => "amount", TechChoice => "tech_choice",
            Preference => "preference", Ownership => "ownership", MetricValue => "metric_value",
        }
    }
    pub fn from_str(s: &str) -> Option<Predicate> {
        Predicate::all().iter().copied().find(|p| p.as_str() == s)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotTuple {
    pub entity_text: String,
    pub predicate: Predicate,
    pub value: String,
    #[serde(default)]
    pub event_time: Option<DateTime<Utc>>,
    pub source_span: String,
}

/// Parse an `event_time` field that may be an RFC3339 timestamp, a bare
/// `YYYY-MM-DD` date, absent, or JSON null. Any parse failure defaults to
/// `None` rather than dropping the row.
fn parse_event_time(v: &serde_json::Value) -> Option<DateTime<Utc>> {
    let s = v.as_str()?;
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());
    }
    None
}

/// Parse the `{"slots": [...]}` payload emitted by the mutable-slot
/// extraction prompt into `SlotTuple`s. Rows with an unknown/missing
/// predicate, or missing required string fields, are silently dropped.
pub fn parse_slot_tuples(v: &serde_json::Value) -> Vec<SlotTuple> {
    let mut out = Vec::new();
    let Some(slots) = v["slots"].as_array() else {
        return out;
    };
    for obj in slots {
        let Some(predicate_str) = obj["predicate"].as_str() else {
            continue;
        };
        let Some(predicate) = Predicate::from_str(predicate_str) else {
            continue;
        };
        let Some(entity) = obj["entity"].as_str() else {
            continue;
        };
        let Some(value) = obj["value"].as_str() else {
            continue;
        };
        let Some(source_span) = obj["source_span"].as_str() else {
            continue;
        };
        let event_time = parse_event_time(&obj["event_time"]);
        out.push(SlotTuple {
            entity_text: entity.to_string(),
            predicate,
            value: value.to_string(),
            event_time,
            source_span: source_span.to_string(),
        });
    }
    out
}

/// Normalize a slot value for agreement comparison across voting runs.
/// Shared with the ledger wiring (Task 12), which uses the same
/// normalization when writing rows.
pub fn value_norm(value: &str) -> String {
    value.trim().to_lowercase()
}

/// Run the mutable-slot extraction prompt once at the given temperature.
/// Any failure (LLM error or malformed JSON) yields an empty result rather
/// than panicking — callers treat "no evidence this run" as a valid vote.
async fn extract_once(
    llm: &dyn LlmProvider,
    content: &str,
    reference_time: DateTime<Utc>,
    temperature: f32,
) -> Vec<SlotTuple> {
    let messages = [
        ChatMessage {
            role: Role::System,
            content: Prompts::mutable_slots_system().to_string(),
        },
        ChatMessage {
            role: Role::User,
            content: Prompts::mutable_slots_user(content, reference_time),
        },
    ];
    let cfg = GenConfig {
        json_schema: Some(crate::llm::schemas::mutable_slots_schema()),
        temperature,
        ..Default::default()
    };
    let Ok(resp) = llm.chat(&messages, &cfg).await else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&resp.content) else {
        return Vec::new();
    };
    parse_slot_tuples(&v)
}

/// Agreement key for comparing slot tuples across voting runs.
type SlotKey = (String, Predicate, String);

fn slot_key(t: &SlotTuple) -> SlotKey {
    (
        crate::extract::entity_key::normalize_entity(&t.entity_text),
        t.predicate,
        value_norm(&t.value),
    )
}

/// Adaptive 2×-with-tiebreak voting wrapper around mutable-slot extraction.
///
/// When `enabled` is false, performs a single deterministic (temperature 0)
/// extraction call. When enabled, runs the extraction twice at a non-zero
/// temperature; if both runs fully agree (same set of `(entity, predicate,
/// value)` keys), the agreeing tuples are returned (2 calls total). If the
/// two runs disagree, a third run is performed and every tuple whose key
/// appears in at least 2 of the 3 runs is kept (majority vote).
pub async fn vote_slots(
    llm: &dyn LlmProvider,
    content: &str,
    reference_time: DateTime<Utc>,
    enabled: bool,
) -> Vec<SlotTuple> {
    const VOTE_TEMPERATURE: f32 = 0.4;

    if !enabled {
        return extract_once(llm, content, reference_time, 0.0).await;
    }

    let run1 = extract_once(llm, content, reference_time, VOTE_TEMPERATURE).await;
    let run2 = extract_once(llm, content, reference_time, VOTE_TEMPERATURE).await;

    let keys1: HashSet<SlotKey> = run1.iter().map(slot_key).collect();
    let keys2: HashSet<SlotKey> = run2.iter().map(slot_key).collect();

    if keys1 == keys2 {
        return dedup_by_key(run1.into_iter().chain(run2), &keys1);
    }

    let run3 = extract_once(llm, content, reference_time, VOTE_TEMPERATURE).await;
    let keys3: HashSet<SlotKey> = run3.iter().map(slot_key).collect();

    let mut vote_count: HashMap<SlotKey, usize> = HashMap::new();
    for k in keys1.into_iter().chain(keys2).chain(keys3) {
        *vote_count.entry(k).or_insert(0) += 1;
    }

    let majority_keys: HashSet<SlotKey> = vote_count
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(k, _)| k)
        .collect();

    dedup_by_key(
        run1.into_iter().chain(run2).chain(run3),
        &majority_keys,
    )
}

/// Keep the first tuple seen for each key in `keep`, preserving first-seen
/// order across the chained runs.
fn dedup_by_key(tuples: impl Iterator<Item = SlotTuple>, keep: &HashSet<SlotKey>) -> Vec<SlotTuple> {
    let mut seen: HashSet<SlotKey> = HashSet::new();
    let mut out = Vec::new();
    for t in tuples {
        let k = slot_key(&t);
        if !keep.contains(&k) {
            continue;
        }
        if seen.insert(k) {
            out.push(t);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn predicate_roundtrips_snake_case() {
        assert_eq!(Predicate::from_str("role_title"), Some(Predicate::RoleTitle));
        assert_eq!(Predicate::TechChoice.as_str(), "tech_choice");
        assert_eq!(Predicate::from_str("not_a_predicate"), None);
    }
    #[test]
    fn there_are_twelve_predicates() {
        assert_eq!(Predicate::all().len(), 12);
    }
    #[test]
    fn parses_valid_slots_and_drops_unknown_predicate() {
        let v = serde_json::json!({"slots": [
            {"entity":"first sprint","predicate":"deadline","value":"April 5","event_time":null,"source_span":"deadline April 5"},
            {"entity":"x","predicate":"not_real","value":"y","source_span":"z"}
        ]});
        let out = parse_slot_tuples(&v);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].predicate, Predicate::Deadline);
        assert_eq!(out[0].value, "April 5");
    }
}
