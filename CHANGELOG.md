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
- `ClockContext` and `*_with_clock` variants on every public write/read
  entry point (`add_note`, `run_dreaming`, `search`, `ask`,
  `fetch_memories`). Replay data with a known anchor and live data flow
  through the same code path. Legacy non-clock wrappers continue to work
  by passing `ClockContext::now()`.
- Real-LLM extraction regression test gated by `KARTA_REAL_LLM_TESTS=1`.
  Runs an 8-entry dated-message fixture against the production LLM in
  ~2 minutes and asserts the right `value_date` / `occurred_*` slots
  are populated. Catches model-prompt drift before it leaks into a
  full benchmark run.

### Changed
- **BEAM 100K: 61.6%** (retrieval fixes, 2026-04-14). Up from 53.0%
  baseline (+8.6pp). Biggest wins: temporal_reasoning (+18pp),
  preference_following (+8pp), contradiction_resolution (+7pp).
- **BEAM 100K conv 0 retrieval-only: 71.2%** (extraction fixes,
  2026-04-22). Up from 56.6% mid-rebuild. Per-ability wins:
  event_ordering +46pp, multi_session_reasoning +33pp,
  summarization +30pp, contradiction_resolution +26pp.
- **Default storage backend is now SQLite + sqlite-vec only.** Lance
  remains in the tree behind a feature flag for benchmark/comparison
  purposes, but the production path uses SQLite for both graph and
  vector storage. Embedded-first, single-binary deployment, no
  external services to run.
- **Atomic facts are now typed retrieval rows.** Each fact carries
  closed-set `memory_kind` / `facet` / `entity_type` enums, typed
  `entity_text` / `value_text` / `value_date` slots, and 1-3
  `supporting_spans` (verbatim substrings of the source message).
  Replaces the prior single-string-content shape.
- **Temporal extraction is structurally enforced.** Date-shaped facets
  (`deadline`, `target_date`) must populate `value_date` or the fact
  is stripped. `occurred_*` bounds must be cited by a supporting span
  containing a literal temporal phrase or the bounds get nulled.
  Real LLMs (especially smaller open models) bias toward filling
  required fields under structured output; the validator is the only
  thing that holds.
- **Cite-and-validate write path.** Pre-admission filter drops
  ephemeral / speech-act / echo facts before embed. Per-fact gates
  enforce grounding (every span ≥4 chars and a real substring of the
  source), specificity (reject facts with both `entity_type=unknown`
  and `facet=unknown`), and temporal grounding. Slot-level dedup
  collapses facts sharing `(entity_text, facet, value)` and merges
  their supporting spans.
- **System prompt rewritten** around what the validator enforces:
  admission first (which kinds become memory), then atomization,
  grounding, normalization, temporal slots. Anti-patterns named after
  real production failure modes (wrapper-strip rule for "I want help
  configuring X", past events as durable facts, jargon leakage).
- **Read-path temporal resolver.** Two-tier resolution for "what did
  I close last week" style queries: tier 1 Rust regex, tier 2 LLM
  fallback for vague phrases ("last spring") with last-3-user-turns
  context. Validation failures fall through to vector-only retrieval
  (no silent guessing).
- Retrieval path: expand-then-rerank, removed synthesis-level
  insufficient-info gate that was killing temporal/computation queries
- Dream engine: score candidates on source note timestamps instead of
  dream write time (accurate recency)
- Dream output uses `max(input.source_timestamp)` rather than the
  current clock, so dreams over old notes don't outrank actual-recent
  facts.
- README reframed to emphasize experimental research project status

### Fixed
- Read path prompt handling for recency-sensitive queries
- SQLite schema + trait adjustments for source timestamp propagation
- "Yesterday" / "last week" queries against replay data no longer
  resolve to wall-clock time, which previously placed every replay
  memory "now" relative to itself.
- `MemoryNote.source_timestamp` is now non-optional and always set
  from the caller's clock at write time.

## [0.1.0-experimental] — 2026-04-14

Initial public release of Karta as an experimental research project.
Core engine in Rust (`karta-core`), LanceDB + SQLite storage, dream
engine with 5 typed inference modes, cross-encoder reranking, structured
output with reasoning, and BEAM 100K benchmark harness.
