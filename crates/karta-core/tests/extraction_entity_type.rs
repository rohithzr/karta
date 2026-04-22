use karta_core::extract::entity_type::EntityType;

#[test]
fn serde_round_trip_each_variant() {
    for e in [
        EntityType::User,
        EntityType::Project,
        EntityType::Person,
        EntityType::Org,
        EntityType::Task,
        EntityType::Unknown,
    ] {
        let s = serde_json::to_string(&e).unwrap();
        let back: EntityType = serde_json::from_str(&s).unwrap();
        assert_eq!(e, back);
    }
}

#[test]
fn snake_case_serialization() {
    assert_eq!(serde_json::to_string(&EntityType::User).unwrap(), "\"user\"");
    assert_eq!(serde_json::to_string(&EntityType::Org).unwrap(), "\"org\"");
    assert_eq!(serde_json::to_string(&EntityType::Unknown).unwrap(), "\"unknown\"");
}

#[test]
fn only_unknown_is_generic() {
    assert!(EntityType::Unknown.is_generic());
    for e in [
        EntityType::User,
        EntityType::Project,
        EntityType::Person,
        EntityType::Org,
        EntityType::Task,
    ] {
        assert!(!e.is_generic(), "{:?} should not be generic", e);
    }
}
