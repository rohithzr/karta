use karta_core::extract::facet::Facet;

#[test]
fn serde_round_trip_each_variant() {
    for f in [
        Facet::Deadline,
        Facet::TargetDate,
        Facet::Preference,
        Facet::TechStack,
        Facet::Location,
        Facet::Ownership,
        Facet::Constraint,
        Facet::Event,
        Facet::Unknown,
    ] {
        let s = serde_json::to_string(&f).unwrap();
        let back: Facet = serde_json::from_str(&s).unwrap();
        assert_eq!(f, back);
    }
}

#[test]
fn snake_case_serialization() {
    assert_eq!(serde_json::to_string(&Facet::TechStack).unwrap(), "\"tech_stack\"");
    assert_eq!(serde_json::to_string(&Facet::TargetDate).unwrap(), "\"target_date\"");
    assert_eq!(serde_json::to_string(&Facet::Unknown).unwrap(), "\"unknown\"");
}

#[test]
fn only_unknown_is_generic() {
    assert!(Facet::Unknown.is_generic());
    for f in [
        Facet::Deadline,
        Facet::TargetDate,
        Facet::Preference,
        Facet::TechStack,
        Facet::Location,
        Facet::Ownership,
        Facet::Constraint,
        Facet::Event,
    ] {
        assert!(!f.is_generic(), "{:?} should not be generic", f);
    }
}
