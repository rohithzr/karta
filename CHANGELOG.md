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
- Real-LLM regression test for date-slot extraction
  (`crates/karta-core/tests/extraction_real_llm_dates.rs`), gated by
  `KARTA_REAL_LLM_TESTS=1`. Runs against a small fixture of dated
  messages and catches model-prompt drift in ~2 minutes — the gap that
  let temporal extraction silently regress until a 95-minute BEAM run
  exposed it.

### Changed
- **BEAM 100K: 61.6%** (retrieval fixes, 2026-04-14). Up from 53.0%
  baseline (+8.6pp). Biggest wins: temporal_reasoning (+18pp),
  preference_following (+8pp), contradiction_resolution (+7pp).
- Retrieval path: expand-then-rerank, removed synthesis-level
  insufficient-info gate that was killing temporal/computation queries
- Dream engine: score candidates on source note timestamps instead of
  dream write time (accurate recency)
- README reframed to emphasize experimental research project status

### Fixed
- Read path prompt handling for recency-sensitive queries
- SQLite schema + trait adjustments for source timestamp propagation

### Ingestion pipeline rebuild (2026-04-22)

Karta's write path now takes one declared reference time and turns each
incoming message into structured, typed retrieval rows with explicit
temporal anchors. The previous pipeline emitted prose paraphrases keyed
on similarity-only retrieval and silently used wall-clock time for
"yesterday" / "last week" queries. The new pipeline is structural at
every layer — schema, prompt, and validator — so failure modes get
caught at ingest instead of leaking into retrieval.

#### Caller-supplied reference time

Every write/read entry point now takes an explicit `ClockContext`
instead of reading `Utc::now()` internally. Replay data with a known
anchor (benchmarks, dataset re-runs) and live data flow through the
same code path.

- `ClockContext { reference_time }` with `now()` and `at(t)` constructors
- Public API: `add_note_with_clock`, `run_dreaming_with_clock`,
  `search_with_clock`, `ask_with_clock`, `fetch_memories_with_clock`
- `MemoryNote.source_timestamp` is non-optional, set from caller's clock
- `session_id: Option<String>` added for grouping related turns
- Dream output uses `max(input.source_timestamp)` rather than the
  current clock, so dreams over old notes don't outrank actual-recent
  facts on recency

**Impact:** "yesterday" / "last week" queries against replay data no
longer place every memory "now" relative to itself.

#### Typed atomic facts with structured slots

The `atomic_facts` table was rebuilt from a single-string-content shape
into typed retrieval rows.

- New typed enum fields: `memory_kind`, `facet`, `entity_type` (closed
  sets, snake_case serde, exhaustive variant guards)
- Typed value slots: `entity_text`, `value_text`, `value_date`
- `supporting_spans: string[]` (1-3 per fact, each ≥4 chars, each a
  verbatim substring of the source message)
- Half-open temporal interval `[occurred_start, occurred_end)` per fact
- `ConfidenceBand` is a closed set `{0.0, 0.5, 0.7, 0.8, 1.0}` mapped
  to `{None, Vague, Relative, NLAbsolute, Explicit}` — continuous values
  are schema violations
- Partial composite index `idx_facts_occurred` for cheap interval-overlap
  scans; null-bound facts excluded from temporal queries by design
- Canonical row reader `row_to_fact` — every SELECT site routes through
  one function

#### Cite-and-validate write path

Real LLMs (especially smaller open models) bias toward filling required
fields under structured output. Prose rules in the prompt lose to that
bias. The fix is to make the validator the source of truth and require
the LLM to *cite* the source text for every populated field.

Validator gates (in load-bearing order):

1. **Pre-admission filter** — drops `ephemeral_request | speech_act |
   echo` before dedup, before embed. Saves embed cost and prevents
   ephemeral facts from claiming slots that durable facts should fill.
2. **Per-fact grounding** — every `supporting_span` must be ≥4 chars
   and a real substring of the source message. Mechanical check, runs
   first per fact for honest telemetry attribution.
3. **Per-fact admission backstop** — defense-in-depth for the JSON
   fallback parse path.
4. **Per-fact specificity** — rejects facts where both `entity_type`
   and `facet` are `unknown`.
5. **Date-shaped facet → value_date required** — `facet ∈ {deadline,
   target_date}` with null `value_date` strips the entire fact.
6. **Temporal bounds → grounded by a span** — `occurred_*` populated
   without any supporting_span containing a literal temporal phrase
   (`yesterday`, `last week`, ISO date, month-day) strips the bounds
   to null.

**Slot-level dedup** runs between pre-admission and embed. Facts
sharing `(entity_text.lower(), facet, value_key)` collapse to one row
with merged `supporting_spans` — preserves the evidence trail when the
LLM emits two phrasings of the same claim.

#### Read-path temporal resolver

Two-tier resolution for "what did I close last week" style queries:

- Tier 1: Rust regex (ISO dates, NL absolute, "yesterday", "last week",
  "last month")
- Tier 2: LLM fallback for vague phrases ("last spring") with
  last-3-user-turns context
- Both pass through `validate_resolver_output`. Validation failures
  fall through to vector-only retrieval — no silent guessing.
- Query classifier emits `temporal: bool`; when true and a resolver
  returns an interval, fact retrieval uses the partial-index path
  `find_similar_facts_in_interval`.

#### Prompt structure

The system prompt is organized around what the validator enforces:

- Admission first (which kinds become memory)
- Atomization (one entity + one facet + one value per fact)
- Grounding (verbatim spans, including the temporal-phrase rule)
- Normalization (typed entity / facet / value slots)
- Temporal slots (when value_date is required vs when occurred_* is)
- Anti-patterns named after real production failure modes

The closer is "Properties of good output" (descriptive shape) rather
than imperative validation — the validator does the mechanical checking;
the prompt shapes intent.

#### Test infrastructure

- Two LLM mocks: `MockLlmProvider` (heuristic — deadline keywords,
  tech-stack tokens, pure-request detection) and `ScriptedMockLlmProvider`
  (adversarial scripted `(needle, response_json)` pairs; falls back to
  heuristic for unscripted calls)
- Real-LLM regression test gated by `KARTA_REAL_LLM_TESTS=1`. Runs an
  8-entry fixture of dated messages against the production LLM and
  asserts the right `value_date` / `occurred_*` slots get populated.
  Catches model-prompt drift in ~2 minutes — the gap that let temporal
  extraction silently regress until a 95-minute BEAM run exposed it.
- Adversarial regression test verifies that ephemeral facts cannot
  steal slots from durable facts (load-bearing admission-before-dedup
  ordering)

**Empirical impact (BEAM 100K conv 0, retrieval-only):**

| Stage | Pass rate | temporal_reasoning |
|---|---|---|
| Prior pipeline (unstructured ingest) | 65.9% | — |
| Mid-rebuild (typed slots, no value_date enforcement) | 56.6% | 0/4 |
| Final (with value_date + grounding validators) | **71.2%** | 0/4 |

Final result is back inside the prior three-model baseline range
(71.7-77.4%). Biggest per-ability wins: event_ordering +46pp,
multi_session_reasoning +33pp, summarization +30pp, contradiction
+26pp. `temporal_reasoning` failure mode shifted from "I can't
determine" (no data) to wrong arithmetic over the right facts —
extraction works; multi-fact synthesis is the next bottleneck.

**Known follow-up:**
- Cross-note entity canonicalization ("project" vs "the project" still
  distinct surface forms today)
- Multi-fact synthesis for date arithmetic (the remaining
  `temporal_reasoning` gap is downstream of extraction)
- Abstention discipline — richer fact retrieval surfaces unrelated
  context, and the synthesis side needs a "if facts unrelated, say so"
  rule
- Negation / attribution slots (mock can't drive these honestly; needs
  real-LLM fixtures)
- Lance store backend mirror (currently feature-gated, returns Unknown
  defaults from `row_to_fact`)

## [0.1.0-experimental] — 2026-04-14

Initial public release of Karta as an experimental research project.
Core engine in Rust (`karta-core`), LanceDB + SQLite storage, dream
engine with 5 typed inference modes, cross-encoder reranking, structured
output with reasoning, and BEAM 100K benchmark harness.
