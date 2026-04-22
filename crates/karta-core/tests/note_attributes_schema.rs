use karta_core::llm::schemas::note_attributes_schema;

#[test]
fn atomic_facts_array_is_optional_zero_count() {
    let schema = note_attributes_schema();
    let v = schema.schema;
    let arr = &v["properties"]["atomic_facts"];
    // The array MUST allow zero items so admission can return [].
    assert!(arr["minItems"].is_null() || arr["minItems"].as_u64() == Some(0));
}

#[test]
fn atomic_fact_item_has_new_required_fields() {
    let schema = note_attributes_schema();
    let v = schema.schema;
    let req = v["properties"]["atomic_facts"]["items"]["required"]
        .as_array()
        .expect("required must be an array")
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect::<Vec<_>>();

    for expected in &[
        "content",
        "memory_kind",
        "supporting_spans",
        "facet",
        "entity_type",
        "entity_text",
        "value_text",
        "value_date",
        "occurred_start",
        "occurred_end",
        "occurred_confidence",
    ] {
        assert!(req.contains(&(*expected).to_string()), "missing required field: {}", expected);
    }
}

#[test]
fn deprecated_fields_absent() {
    let schema = note_attributes_schema();
    let v = schema.schema;
    let props = v["properties"]["atomic_facts"]["items"]["properties"]
        .as_object()
        .unwrap();
    assert!(!props.contains_key("subject"), "subject field must be removed");
    assert!(!props.contains_key("temporal_evidence"), "temporal_evidence field must be removed");
}

#[test]
fn memory_kind_enum_complete() {
    let schema = note_attributes_schema();
    let v = schema.schema;
    let mk = &v["properties"]["atomic_facts"]["items"]["properties"]["memory_kind"];
    let enum_vals = mk["enum"].as_array().expect("enum must exist");
    let strs: Vec<String> = enum_vals.iter().filter_map(|e| e.as_str().map(String::from)).collect();
    for expected in &[
        "durable_fact", "future_commitment", "preference", "decision",
        "constraint", "ephemeral_request", "speech_act", "echo",
    ] {
        assert!(strs.contains(&(*expected).to_string()), "memory_kind missing variant {}", expected);
    }
}
