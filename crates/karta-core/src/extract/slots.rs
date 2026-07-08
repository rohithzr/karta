use chrono::{DateTime, Utc};
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
}
