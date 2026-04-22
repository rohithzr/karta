# Changelog

All notable changes to Karta will land here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely and
semantic versioning once we leave the experimental phase.

## [Unreleased]

### Added
- `rust-toolchain.toml` and `bacon.toml` for reproducible dev loop
- `benchmarks/` folder with BEAM 100K results, per-ability history, and
  reproduction commands
- `docs/landscape.md` — research survey of the AI memory space
- `docs/retrieval-plan.md` — open retrieval experiment backlog
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`

### Changed
- **BEAM 100K: 61.6%** (P1 retrieval fixes, 2026-04-14). Up from 53.0%
  P0 baseline (+8.6pp). Biggest wins: temporal_reasoning (+18pp),
  preference_following (+8pp), contradiction_resolution (+7pp).
- Retrieval path: expand-then-rerank, removed synthesis-level
  insufficient-info gate that was killing temporal/computation queries
- Dream engine: score candidates on source note timestamps instead of
  dream write time (accurate recency)
- README reframed to emphasize experimental research project status

### Fixed
- Read path prompt handling for recency-sensitive queries
- SQLite schema + trait adjustments for source timestamp propagation

### STEP1 — Clock context (2026-04-22)

Karta now takes an explicit reference time at every entry point instead
of reading `Utc::now()` deep in the write/read paths. Replay data with a
known anchor (BEAM, LongMem, LOCOMO) and live data both flow through the
same path.

- New `ClockContext { reference_time }` with `now()` and `at(t)` constructors
- All public API surfaces gained `*_with_clock` variants:
  `add_note_with_clock`, `run_dreaming_with_clock`, `search_with_clock`,
  `ask_with_clock`, `fetch_memories_with_clock`. Legacy wrappers without
  `_with_clock` keep working by passing `ClockContext::now()`.
- `MemoryNote.source_timestamp` is non-optional and always set from the
  caller's clock at write time. `session_id: Option<String>` added.
- BEAM converter (`data/convert_beam.py`) rewritten to emit
  `sessions[].turns[]` with a per-turn `effective_reference_time` so
  the harness no longer guesses time anchors from sparse `time_anchor`
  prefixes (3/5732 turns populated in the source).
- Dream-output `source_timestamp = max(input.source_timestamp)` (NOT
  `ctx.reference_time()`) — avoids the time-travel-confidence bug where
  a fresh-clocked dream over old notes would outrank actual-recent facts.

**Why:** Recency, foresight DOA, and "yesterday/last week" queries were
silently using wall-clock time during BEAM replay, which placed every
piece of replay content "now" relative to itself and broke ordering on
event_ordering (35%) and knowledge_update (43%) categories. Reference
time is now data, not ambient state.

### STEP1.5 — Fact `occurred_*` bounds + temporal resolver (2026-04-22)

Each atomic fact carries a half-open interval `[occurred_start, occurred_end)`
and a discrete confidence band, so "what did I close last week" is a SQL
range query against fact intervals instead of substring matching against
date strings inside fact text.

- `atomic_facts` schema gained `source_timestamp` (inherited from parent
  note), `occurred_start`, `occurred_end`, `occurred_confidence`. Partial
  composite index `idx_facts_occurred(occurred_start, occurred_end) WHERE
  occurred_start IS NOT NULL` for cheap interval-overlap scans.
- `ConfidenceBand` enum is a closed set: `{0.0, 0.5, 0.7, 0.8, 1.0}`
  mapped to `{None, Vague, Relative, NLAbsolute, Explicit}`. Continuous
  values are a schema violation and rejected.
- 4 invariants enforced via `AtomicFact::validate_occurred()`: bound
  pairing, end > start, closed-set confidence (type-level), confidence/
  bounds null-pairing.
- F7 prompt rewrite: `note_attributes_system` now requires per-fact
  occurred_start/end/confidence, forbids extracting requests-for-help as
  facts, and supplies the reference time in the user message preamble.
- Read-path two-tier resolver: tier 1 Rust regex (ISO/NL absolute/
  yesterday/last week/last month) → tier 2 LLM fallback for vague
  phrases ("last spring") with last-3-user-turns context. Both pass
  through `validate_resolver_output`. Validation failures fall through
  to vector-only retrieval (no silent guessing).
- Query classifier emits `temporal: bool`; when true and a resolver
  returns an interval, fact retrieval uses
  `find_similar_facts_in_interval` (the partial index path).
  Null-bound facts are excluded from temporal queries by design.
- BEAM harness skips `question_type='answer_ai_question'` turns at
  ingest time (these are user echoes of AI suggestions with no
  originating claims). Skip lives in the test harness, not Karta core.

**Why:** Pre-STEP1.5, "last week" matched on text similarity to fact
content. A fact saying "yesterday at 14:30" on Monday and the same
text on Friday were indistinguishable to the retrieval layer.
Structured bounds + interval-overlap SQL replace the text-match
hack with the right primitive.

### F7 prompt — evidence-grounded fact bounds (2026-04-22)

The first F7 prompt over-hedged: real-LLM traces returned
`Vague (0.5)` with wide ranges on most facts. The prose-only fix
made it worse: the LLM put `[ref_time-1d, ref_time)` bounds on
every fact, treating present-tense verbs ("uses Flask") as
"started yesterday."

Root cause: structured output models bias hard toward producing
non-null fields. Prose rules ("default to null") lose to that
bias.

Fix: required `temporal_evidence: string | null` field on each
fact. The LLM must quote the literal temporal phrase from the
fact's `content`. The write-path validator strips bounds
(downgrades to null/null/0.0) if:
  - `temporal_evidence` is null/empty, OR
  - the quote does not appear verbatim in `fact.content`.

This is grounded reasoning instead of inference. The LLM cannot
write a quote that isn't in the text, so it cannot smuggle
conversation-date intuitions into per-fact bounds.

**Trace impact (10-turn BEAM conv 0):**
- v1 (original):       0% null bounds (9/22 hallucinations)
- v2 (prose-only fix): 0% null bounds (41/41 hallucinations)
- v3 (grounding gate): 67% null bounds (29/43 correct nulls)

The validator stripped 29 of 44 LLM-emitted bounds (66%) for
ungrounded evidence quotes — including the LLM trying to use
"reference_time: 2024-03-15..." (the prompt prefix itself) and
"April 15" as evidence for unrelated facts.

**Known follow-up:** the 15 facts that survived the gate include
~6 borderline cases where the LLM included "March 15" in the
synthesized fact text and the gate accepts it. Per-band precision
needs the calibration fixture
(`data/test/fixtures/confidence_calibration.json`, scaffolded but
unlabeled — needs ≥100 hand-labeled entries before F7-T15 runs).

### STEP2 — Fact extraction redesign (2026-04-22)

Codex review (`docs/reviews/2026-04-22-codex-fact-extraction-review.md`)
identified six structural failures upstream of temporal grounding. The
LLM was producing prose paraphrases of input messages instead of typed,
admission-controlled retrieval rows. STEP2 rebuilds the schema, prompt,
and write path around the cite-and-validate pattern proven by
`temporal_evidence`.

**Schema changes** (atomic_facts table recreated, no migration):
- New typed enum fields: `memory_kind`, `facet`, `entity_type`
  (closed sets, snake_case serde rename, exhaustive variant guards in tests)
- New typed value slots: `entity_text`, `value_text`, `value_date`
- `supporting_spans: string[]` replaces single `temporal_evidence` field
  (1-3 per fact, each ≥4 chars, each a verbatim substring of source)
- Old `subject` field removed (subsumed by `entity_text` + `entity_type`)
- New canonical reader `row_to_fact` — every SELECT site routes through
  one function so future schema changes touch one place
- Shadow-DDL collision in `SqliteGraphStore` removed (the old `subject`
  index was clobbering schema on shared connections — silent data-shape
  bug fixed in passing)

**Validator gates** (in `WriteEngine::add_note_inner`, load-bearing order):
1. **Pre-filter (admission)** — drops `ephemeral_request | speech_act |
   echo` BEFORE dedup BEFORE embed. Saves embed cost and prevents
   ephemerals from claiming a slot dedup would assign to a real fact.
2. **Per-fact grounding** — every `supporting_span` ≥4 chars and a real
   substring of `note.content`. Runs FIRST per fact (cheap mechanical
   check, telemetry attribution honesty).
3. **Per-fact admission backstop** — defense-in-depth for the
   JSON-fallback parse path.
4. **Per-fact specificity** — rejects facts where both `entity_type`
   and `facet` are `unknown`.

**Slot-level dedup** runs BETWEEN pre-filter and embed. Collapses facts
sharing `(entity_text.lower(), facet, value_key)` to one row, MERGING
`supporting_spans` from siblings (string-equality dedup). Preserves the
evidence trail when the LLM emits two phrasings of the same claim.

**Mock pair:**
- `MockLlmProvider` — heuristic mock for trace harness convenience
  (deadline keywords, tech-stack tokens, pure-request detection)
- `ScriptedMockLlmProvider` — adversarial mock for validator tests
  (script `(needle, response_json)` pairs; falls back to heuristic
  for unscripted calls)

**Prompt rewrite:** admission becomes Section 1, then atomization,
grounding, normalization, temporal. Temporal is no longer "the one
big rule" — admission is. Anti-patterns A-G name 7 production failure
modes. Closer is "Properties of good output" (descriptive shape) rather
than "final checklist" (imperative validation) — the validator does
the mechanical checking, the prompt shapes intent.

**Empirical impact (10-turn BEAM conv 0):**

| Run | Facts | Null-bound | Typed (both axes) | Ephemeral persisted |
|---|---|---|---|---|
| v1 (original prompt) | 22 | 0% | n/a (no enum) | n/a |
| v2 (prose-only fix) | 41 | 0% | n/a | n/a |
| v3 (`temporal_evidence` gate) | 43 | 67% | n/a | n/a |
| **STEP2** | **17** | n/a (different model) | **100%** | **0** |

STEP2's 17 facts are correctly typed entities + facets:
- entity_type: project (10), org (5), task (1), person (1) — real types
- facet: tech_stack (13), target_date (2), ownership (1), preference (1)
- entity_text values: "Craig", "budget tracker", "app", "project" — real
  surface forms, not just generic "user"
- 16 durable_fact + 1 future_commitment, zero ephemeral_request /
  speech_act / echo

Pure requests now produce zero facts (admission pre-filter fires before
embed) — turn 1 of the trace ("I'm working on a project... can you
help me create a schedule") emitted 0 facts as designed.

**Test coverage:** 39 test binaries, ~131 tests in the STEP1.5+STEP2
regression bundle. New STEP2 test files: 10 (`extraction_*`,
`note_attributes_schema`, `mock_extraction_shape`, `extraction_failure_modes`).
The adversarial regression test
`ephemeral_collision_does_not_steal_durable_slot` uses the scriptable
mock to construct a slot collision — verifies the load-bearing
admission-before-dedup ordering doesn't regress.

**Deferred to follow-up:**
- Cross-note entity canonicalization (Phase 2 per codex; "project" vs
  "the project" still distinct entity_text values today)
- Negation/attribution slots (FM3-FM6 from codex review — mock can't
  drive these honestly, needs real LLM)
- Per-band calibration fixture extension to admission decisions
- Lance store backend mirror (currently feature-gated, returns
  Unknown defaults from row_to_fact)
- A few miscategorized facts in the trace (Bootstrap config tagged as
  `target_date`, port 5000 as `target_date`) — prompt tuning territory,
  not structural

## [0.1.0-experimental] — 2026-04-14

Initial public release of Karta as an experimental research project.
Core engine in Rust (`karta-core`), LanceDB + SQLite storage, dream
engine with 5 typed inference modes, cross-encoder reranking, structured
output with reasoning, and BEAM 100K benchmark harness.
