# Decider plan — closed-set classifiers (Jev and friends)

> Status: step 2 (query understanding) is wired behind an opt-in
> `Karta::attach_decider`; default behavior is unchanged. Routing is
> evaluated on BEAM's 400 questions below; the end-to-end BEAM A/B is next.

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
| `Question` / `QuestionKind` | `Choice { options: (label, description) }`, `YesNo`, `Score { levels }` (2–10 level descriptions) |
| `Decision` | chosen label + per-option probabilities, `p_yes`, or score; `confidence()` |
| `LlmDecider` | Baseline. One structured-output call with an enum/bool/int schema. No new dependency. |
| `JevDecider` | TypeSafe Jev System One API. `JevDecider::from_env()` reads `JEV_API_KEY`, `JEV_BASE_URL`, `JEV_MODEL`. |
| `MockDecider` | Scripted answers for tests. |
| `decide::questions` | `query_mode_question`, `query_predicate_question` (12 predicates + none), `query_temporal_question`, and mappers back to `QueryMode` / `Predicate`. |
| `tests/decider_jev_live.rs` | Gated on `JEV_API_KEY`. Verifies the wire format and prints Jev vs keyword classifier on 18 hand-labeled queries. |

**Wire format verified** against the live API (`jev-1.13.0`, 2026-10-02);
the exact request/response shape is documented at the top of
`decide/jev.rs`. Score questions take 2–10 level descriptions; choice
handled 20 options.

## First live result (2026-10-02)

`tests/decider_jev_live.rs`, 18 hand-labeled queries, one `decide` call per
query asking mode + predicate + temporal together:

| Decision | Jev | Today's classifier |
|---|---|---|
| Query mode (6-way) | **17/18** | 11/18 (keyword fallback) |
| Ledger predicate (13-way) | **17/18** | — |
| Latency per call (3 questions) | ~150–250 ms | — |

- The one mode miss ("I'm known for being punctual, right?" → standard,
  want existence) came with confidence 0.62; every correct answer was
  ≥ 0.74, most ≥ 0.92. A confidence fallback would have caught it.
- The predicate "miss" ("How many days passed between…" → count) is
  arguably right.
- The temporal yes/no is not usable as worded: p_yes ranged 0.53–0.98 and
  barely separated temporal from non-temporal queries. It is not used by
  the read path; reword before relying on it.
- Caveats: 18 queries written by us, not BEAM; compared against the
  keyword classifier, not the embedding-centroid one used when embeddings
  are available. The BEAM A/B in step 2 is the real test.
- With uninformative options Jev picked the first option at 0.69 — it has
  a position prior. Choice criteria go over the wire as a JSON object in
  alphabetical key order, so order is stable.

## BEAM routing eval (2026-10-02)

`tests/decider_beam_routing.rs` runs Jev on all 400 BEAM 100K probing
questions (text only, no ingest). 0 errors, ~11 s at 8 concurrent calls.
BEAM has no mode labels, so four abilities with an obvious target mode are
scored; "keyword" is the built-in fallback router (the embedding centroid
needs a real embedding model, so it isn't in this table).

| Ability (target mode) | Keyword | Jev | Routed (Jev if conf ≥ 0.7, else keyword) |
|---|---|---|---|
| contradiction_resolution (Existence) | 0% | 95% | 93% |
| event_ordering (Temporal) | 98% | 100% | 100% |
| summarization (Breadth) | 90% | 100% | 100% |
| temporal_reasoning (Computation) | 90% | 98% | 98% |
| **Scored total** | **69%** | **98%** | **98%** |

Threshold sweep on the scored set: 0.5–0.7 → 98%, 0.8 → 95%, 0.9 → 91%.
Default `decider_min_confidence` is 0.7; at that setting 79 of 400 queries
get a different mode than the keyword router.

**knowledge_update is a predicate problem, not a mode problem.** Its
questions never say "current" ("How many sources are in my Zotero
library?"), so neither router sends them to Recency (2%). What matters is
whether the `[CURRENT]` ledger lookup fires:

| Ledger lookup fires on | Keyword | Jev (conf ≥ 0.8) |
|---|---|---|
| knowledge_update (want: yes) | 26/40 | **34/40** |
| abstention (want: no) | 7/40 | **0/40** |
| preference_following (want: mostly no) | 10/40 | **2/40** |

Keyword false fires are substring hits: "techniques" → tech_choice,
"confusing" → "using" → tech_choice, "agenda … sessions where" → location.

Open question for the A/B: Jev routes 17 knowledge_update questions to
Computation (keyword: 15). Computation skips the reranker and narrows
fetch_k, which may not suit "what is the latest count" questions.

## Rollout order

Each step is behind a config flag defaulting to today's behavior, and is
measured on BEAM 100K (±3pp is noise, single runs) before flipping the default.

1. ~~**Get access + verify wire format.**~~ Done 2026-10-02 (see above).
2. **Query understanding (read path, one call per query).** *Wired.*
   `Karta::attach_decider`; the BEAM harness enables it with
   `K_DECIDER=jev` (+ `K_DECIDER_MIN_CONF`). Asks mode + predicate in a
   single `decide` call, concurrently with the query embedding. Each answer
   is used when `confidence >= decider_min_confidence` (default 0.7), else
   the centroid / keyword classifiers decide; errors and a 10 s timeout fall
   back too. Both answers are logged (`Query routed with decider`).
   **Next:** BEAM 100K A/B — ingest once, then run the query phase twice on
   the same data (`BEAM_SKIP_INGEST=true`, with and without `K_DECIDER=jev`).
   Needs the Azure/OpenAI + Jina credentials.
3. **Answerability gate.** YesNo "does this context contain the answer?"
   before synthesis; targets false abstention (18% of failures) and gives a
   calibrated abstain signal (the reranker threshold is logged, not enforced).
4. **Rerank scores.** a 6-level `Score` per candidate as an alternative
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
