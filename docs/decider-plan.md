# Decider plan — closed-set classifiers (Jev and friends)

> Status: groundwork landed (`crates/karta-core/src/decide/`), nothing wired
> into the read or write path yet. This doc is the rollout plan.

## Why

Karta makes about 20 small decisions over fixed label sets: query mode,
ledger predicate, temporal flag, link / no-link, episode boundary, rerank
relevance, "does this context answer the question". Today they are done with
keyword lists, embedding centroids, or full LLM calls. The keyword and
centroid ones are among the weakest parts of the read path:

- **Query mode** (`read/mod.rs` `QueryClassifier`): nearest-centroid over 5–10
  hand-written prototypes per mode, no confidence margin. Temporal-routed
  queries average 38.8% (62% fail), Computation 44.8% (47% fail). Mode
  controls fetch_k, top_k, recency weight, reranker skip, the EVENTS block
  and the retry.
- **Ledger predicate** (`read/mod.rs` `query_predicate`): substring matching
  ("own" matches "known", "due" matches "residue"). It gates the
  `[CURRENT]`/`[CONFLICT]` block that knowledge_update (43%, weakest ability
  after event_ordering) depends on.
- **Temporal flag** (`has_temporal_indicator`): lexical; "when", "before",
  "after" fire on almost anything.

A "System One" classifier answers these as typed questions with
probabilities, in one forward pass, without generating text.

## What landed

`karta_core::decide`:

| Item | Purpose |
|---|---|
| `Decider` trait | `decide(state, &[Question]) -> Decisions`. Several named questions per call, since batching is nearly free for Jev. |
| `Question` / `QuestionKind` | `Choice { options: (label, description) }`, `YesNo`, `Score { max }` |
| `Decision` | chosen label + per-option probabilities, `p_yes`, or score; `confidence()` |
| `LlmDecider` | Baseline. One structured-output call with an enum/bool/int schema. No new dependency. |
| `JevDecider` | TypeSafe Jev System One API. `JevDecider::from_env()` reads `JEV_API_KEY`, `JEV_BASE_URL`, `JEV_MODEL`. |
| `MockDecider` | Scripted answers for tests. |
| `decide::questions` | `query_mode_question`, `query_predicate_question` (12 predicates + none), `query_temporal_question`, and mappers back to `QueryMode` / `Predicate`. |
| `tests/decider_jev_live.rs` | Gated on `JEV_API_KEY`. Verifies the wire format and prints Jev vs keyword classifier on 18 hand-labeled queries. |

**The Jev wire format is unverified.** Public write-ups disagree (context
32K vs 64K, 2–16 vs up to 255 choice options, response field names). The
mapping lives in `to_request` / `from_response` in `decide/jev.rs`; fix it
there once the live test runs against a real key.

## Rollout order

Each step is behind a config flag defaulting to today's behavior, and is
measured on BEAM 100K (±3pp is noise, single runs) before flipping the default.

1. **Get access + verify wire format.** Run `decider_jev_live`, fix
   `jev.rs` if needed. Record latency and per-query cost.
2. **Query understanding (read path, one call per query).** Ask mode +
   predicate + temporal in a single `decide` call. Use the Decider answer
   when `confidence >= threshold`, else fall back to centroid / keywords.
   Wire via an `attach_decider` setter on `ReadEngine` (same pattern as
   `attach_slot_ledger`) so constructors don't change. Log both answers into
   the BEAM trace so disagreements can be inspected.
3. **Answerability gate.** YesNo "does this context contain the answer?"
   before synthesis; targets false abstention (18% of failures) and gives a
   calibrated abstain signal (the reranker threshold is logged, not enforced).
4. **Rerank scores.** `Score { max: 5 }` per candidate as an alternative
   `Reranker` impl; compare to Jina on the same runs.
5. **Write path.** Link / no-link and episode boundary as YesNo, replacing
   their LLM calls. Contradiction check on slot supersession and dream
   pre-filtering via an NLI-style question (entail / contradict / neutral).

Not planned: date resolution, value extraction, arithmetic. Jev's own docs
list dates, counting and arithmetic as weaknesses — those stay on the LLM or
regex resolvers.

## Keeping Karta embedded

A hosted API conflicts with "zero infrastructure", so Jev stays opt-in.
Two local paths for later:

- **Distil to a tiny classifier.** Once Jev / the LLM has labeled a few
  thousand real queries, train a small encoder for query mode + predicate.
  A public benchmark had a trained 22M-parameter classifier beat Jev on a
  77-way intent task (93.2% vs 80.1%) at ~8 ms on CPU. Fixed label sets
  like ours are where that wins.
- **Open weights.** `autotrust/JEV` (Apache-2.0, Qwen3.5-9B + adapter,
  1,024-token input) or TinyJev 0.6B/1.7B behind an OpenAI-compatible local
  server; point `JEV_BASE_URL` at it if the API shape matches.

## Risks

- **Option-order sensitivity.** ~11% of 16-option answers flip under
  reordering for the distilled model (7% for Jev). Option order in
  `decide::questions` is fixed; don't reorder casually.
- **Calibration.** One independent benchmark found Jev overconfident (88%
  stated vs 80% actual). Calibrate thresholds on BEAM traces, not vendor
  numbers.
- **Early access.** Pricing (vendor says it may be subsidised), rate limits
  and the API surface may change.
