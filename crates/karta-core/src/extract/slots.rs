use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
