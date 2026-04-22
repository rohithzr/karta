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

**Known calibration gap:** real-LLM trace runs return mostly
`Vague (0.5)` with wide ranges even when content has explicit dates.
Prompt is over-hedging. Calibration fixture
(`data/test/fixtures/confidence_calibration.json`) scaffolded but
unlabeled — needs ≥100 hand-labeled entries before F7-T15 runs.

## [0.1.0-experimental] — 2026-04-14

Initial public release of Karta as an experimental research project.
Core engine in Rust (`karta-core`), LanceDB + SQLite storage, dream
engine with 5 typed inference modes, cross-encoder reranking, structured
output with reasoning, and BEAM 100K benchmark harness.
