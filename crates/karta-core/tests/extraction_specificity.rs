use karta_core::extract::admission::{validate_specificity, SpecificityError};
use karta_core::extract::entity_type::EntityType;
use karta_core::extract::facet::Facet;

#[test]
fn accepts_typed_entity_typed_facet() {
    assert!(validate_specificity(EntityType::Project, Facet::Deadline).is_ok());
}

#[test]
fn accepts_typed_entity_unknown_facet() {
    // Specificity only rejects when BOTH are generic.
    assert!(validate_specificity(EntityType::Project, Facet::Unknown).is_ok());
}

#[test]
fn accepts_unknown_entity_typed_facet() {
    assert!(validate_specificity(EntityType::Unknown, Facet::Deadline).is_ok());
}

#[test]
fn rejects_both_unknown() {
    assert_eq!(
        validate_specificity(EntityType::Unknown, Facet::Unknown),
        Err(SpecificityError::BothGeneric),
    );
}
