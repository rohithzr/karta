# Confidence calibration fixture

Used by **F7-T15** (see `STEP1.5_FACT_TIME.md`).

**Size target:** ≥100 entries, ≥20 per confidence band (`0.0`, `0.5`, `0.7`, `0.8`, `1.0`).

**Format:**

```json
{
  "entries": [
    {
      "message_content": "Deadline is 2024-03-15 for the Flask milestone.",
      "reference_time": "2024-03-10T00:00:00Z",
      "expected_facts": [
        {
          "content_pattern": "deadline",
          "expected_confidence_band": 1.0,
          "expected_occurred_start": "2024-03-15T00:00:00Z",
          "expected_occurred_end":   "2024-03-16T00:00:00Z"
        }
      ]
    }
  ]
}
```

**Labeling rubric:** see the "Resolved decisions" section in
`STEP1.5_FACT_TIME.md`. Discrete bands:

- `1.0` — explicit ISO date in source
- `0.8` — natural-language absolute date
- `0.7` — relative reference, deterministic resolution
- `0.5` — vague reference, range chosen
- `0.0` — no temporal content

**Labeling process:**
1. Sample real BEAM conversations (`data/beam-100k.json`).
2. Run Karta's F7 extraction in trace mode to get the candidate facts.
3. Hand-label each fact's expected confidence band + expected bounds.
4. Flag ambiguous calls for re-review before appending.

**Maintenance:** when the rubric changes, re-label affected entries. This
fixture is a versioned artifact, not one-shot.
