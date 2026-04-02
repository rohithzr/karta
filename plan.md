# Retrieval Quality Experiment Plan

Baseline: BEAM 100K = 51.3% (post-wiring optimal). Minimum Target: 63%+ (Honcho parity). Project Expectations: 90%+ (SOTA)
243 failures across 400 questions. This plan attacks them in priority order.

---

## P0: Foundation Fixes (must-do before any experiment makes sense)

These are not experiments. They're bugs and missing wiring that distort every measurement.

### P0.0 — Benchmark Debug Logging (JSONL)

**Problem:** Current benchmark only logs truncated answers and rubric scores to stdout. No way to debug retrieval failures, inspect full LLM answers, see which notes were retrieved, or reproduce issues. Every experiment requires re-running hours of LLM calls to investigate a single failure.

**Changes:**
1. Add structured JSONL output per question to `.results/beam-debug-{timestamp}.jsonl`
2. Each line records one question evaluation:
   ```json
   {
     "conv_id": "1",
     "conv_category": "Coding",
     "question_index": 5,
     "ability": "event_ordering",
     "question": "Can you list in order...",
     "reference_answer": "...",
     "rubric_items": ["item1", "item2"],
     "system_answer": "full LLM answer here",
     "rubric_scores": [{"item": "item1", "score": 1.0, "grade": "FULL"}, ...],
     "beam_score": 0.80,
     "notes_retrieved_count": 7,
     "notes_retrieved_ids": ["id1", "id2"],
     "query_mode": "Temporal",
     "contradiction_notes_injected": 0,
     "ingestion_time_ms": 450000,
     "query_time_ms": 23400,
     "dream_count": 12
   }
   ```
3. Modify `eval_conversation()` to return and log this data
4. Modify `ask()` in read.rs to optionally return retrieval metadata (notes used, query mode, injection count) alongside the answer

**Why this is P0:** Without this, every experiment is a black box. We can't tell if a failure is retrieval, synthesis, scoring, or judge noise without re-running the full pipeline.

**Files:** `tests/beam_100k.rs`, `read.rs` (return metadata), `note.rs` (SearchMetadata struct)

---

### P0.1 — BEAM Benchmark Parsing Fix

**Problem:** All notes in a conversation get `created_at` = ingestion time (within seconds of each other). Temporal sorting is meaningless. BEAM's `time_anchor` field has real timestamps but they're prepended as text prefixes `[March-15-2024]`, not stored as structured metadata.

**Changes:**
1. Add `turn_index: Option<u32>` field to `MemoryNote` in `note.rs`
2. Add `source_timestamp: Option<DateTime<Utc>>` field to `MemoryNote` (the real conversation time, distinct from ingestion `created_at`)
3. Extend `add_note_with_session()` API to accept optional `turn_index` and `source_timestamp`
4. In `bench_beam.rs`: parse `time_anchor` into `source_timestamp`, pass message position as `turn_index`
5. LanceDB schema migration: add both fields to the vector store table
6. Episode drilldown in `read.rs`: sort by `turn_index` when present, fall back to `source_timestamp`, fall back to `created_at`

**Validates:** Every temporal/ordering experiment downstream depends on this. Without it, we're measuring noise.

**Files:** `note.rs`, `read.rs`, `write.rs`, `store/lance.rs`, `store/mod.rs` (VectorStore trait), `tests/bench_beam.rs`, `config.rs`

### P0.2 — Wire Contradiction Force-Retrieval into Read Path

**Problem:** Contradiction dreams exist with `source_note_ids` tracking both sides. But the read path never uses this. When a query matches a contradiction dream, only one side gets retrieved via ANN. The other side (usually the "never" singleton) is lost.

**Changes:**
1. Add `get_notes_by_ids(ids: &[&str]) -> Vec<MemoryNote>` to VectorStore trait (batch fetch by ID)
2. In `read.rs` after ANN retrieval + merge: scan results for `Provenance::Dream { dream_type: "contradiction", source_note_ids, .. }`
3. For each contradiction dream found: fetch all `source_note_ids` via batch lookup
4. Inject source notes into result set with a `contradiction_source: true` marker
5. Pass marker to synthesis prompt: "Notes marked [CONTRADICTION SOURCE] are the two sides of a known contradiction. Present BOTH sides and ask which is correct."
6. Fix over-triggering: update synthesis prompt to only flag `has_contradiction=true` when actual mutually exclusive claims are present, not mere updates

**Also wire:** When any ANN-retrieved note is linked to a contradiction dream (check via `get_links_with_reasons()` for "{contradiction} dream" reason), pull in the dream AND its other source notes. This catches cases where the "I use Excel" note is retrieved but the contradiction dream itself isn't in top-K.

**Files:** `store/mod.rs` (VectorStore trait), `store/lance.rs`, `read.rs`, `llm/prompts.rs`

### P0.3 — Query-Type Router

**Problem:** Every query goes through identical retrieval. But "what's my current salary?" and "list the order of topics" need fundamentally different strategies.

**Changes:**
1. Add `QueryMode` enum: `Standard`, `Recency`, `Breadth`, `Computation`, `Temporal`, `Existence`
2. Add `classify_query(query: &str) -> QueryMode` function in `read.rs` — regex/keyword based, no LLM call
3. Each mode overrides specific config knobs:

| Mode | Detection | top_k | recency_weight | fetch_k multiplier | Special behavior |
|------|-----------|-------|----------------|--------------------|-|
| `Standard` | default | 5 | 0.15 | 2x | current behavior |
| `Recency` | "current/latest/now/updated/changed" | 5 | 0.60 | 2x | — |
| `Breadth` | "summarize/overview/all/everything" | 15 | 0.15 | 3x | already exists, formalize it |
| `Computation` | "how many more/days between/difference/compare" | 10 | 0.15 | 3x | force multi-hop |
| `Temporal` | "in what order/sequence/steps/first...then" | 20 | 0.0 | 1x | sort by turn_index, skip ANN scoring |
| `Existence` | "did I ever/have I/was there" | 5 | 0.15 | 2x | include contradiction dreams |

4. Log detected mode for experiment analysis

**Files:** `read.rs`, `config.rs` (add `QueryMode`)

### P0.4 — Kill Synthesis-Level Abstention

**Problem:** 4 abstention gates. Gate 3 (synthesis prompt "MUST abstain") is the most aggressive and least accurate. It triggers on tangentially related notes that DO answer the question. 18% of failures are false abstentions.

**Changes:**
1. Remove the `should_abstain` field from synthesis structured output
2. Remove Gate 3 logic in `read.rs` that checks `should_abstain`
3. Keep Gate 1 (empty results), Gate 2 (reranker threshold at 0.01), Gate 4 (null answer fallback)
4. Update synthesis prompt: remove "MUST abstain" language, replace with "If the notes don't answer the question, say so naturally in your answer rather than refusing to answer"

**Risk:** May increase hallucination slightly (currently 2.9%). Monitor in experiment.

**Files:** `read.rs`, `llm/prompts.rs`

---

## P1: Expand-then-Rerank

**Problem:** top-K=5 means 95% of notes in a 100+ note conversation are never considered. The reranker exists but only rescores the already-filtered top-5.

**Changes:**
1. Increase `fetch_k` to 20 (4x top_k) for all query modes
2. Run Jina cross-encoder reranker on all 20 candidates
3. Take top-K from reranked results (not from raw ANN scores)
4. Adjust `max_rerank` config to 20

**Cost:** ~15 extra Jina API calls per query (~50ms). Acceptable.

**Expected impact:** Directly fixes the "specific detail missed" sub-pattern (50 of 99 INCOMPLETE_RETRIEVAL failures). The detail IS in the top-20 but not in the top-5.

**Files:** `read.rs`, `config.rs`

---

## P2: Episode Narrative Ordering (EITHER/OR with P3)

> **EITHER P2 OR P3** — both attack EVENT_ORDERING (17.3% of failures) but from different angles. Run one, measure, then decide if the other is needed.

### P2: Embed Ordering in Episode Narratives

**Idea:** When synthesizing episode narratives, include a numbered chronological list of topics. For ordering queries, the LLM reads the ordered list directly from the narrative.

**Changes:**
1. Update `episode_narrative_system()` prompt to include: "End the narrative with a numbered chronological list of topics/events discussed, in conversation order"
2. Depends on P0.1 (turn_index) so the episode has real ordering to embed

**Effort:** Small — one prompt change + P0.1 dependency.

**Limitation:** Only helps queries that match an episode narrative. Pre-episode notes or cross-episode ordering won't benefit.

### P3: Temporal Chain Links (EITHER/OR with P2)

**Idea:** During write path, link each note to the previous note from the same session with a `follows` link type. Creates an ordered linked list per conversation.

**Changes:**
1. Track `last_note_id` per session in write path
2. After storing a new note, add link: `graph_store.add_link(prev_id, new_id, "follows")`
3. For `Temporal` query mode (P0.3): find any relevant note, traverse `follows` chain to reconstruct full sequence
4. Add `get_links_by_reason(note_id, reason)` to GraphStore trait for efficient chain traversal

**Effort:** Medium — new link type, traversal logic, GraphStore method.

**Advantage over P2:** Works at note level, not episode level. Can reconstruct micro-ordering within a conversation regardless of episode boundaries.

**Disadvantage:** More links in the graph, traversal cost for long conversations.

---

## P4: Ghost Queries (Paraphrase Expansion)

**Problem:** "salary offered" doesn't match "compensation package: $72K base" in embedding space. Single-query ANN misses vocabulary gaps.

**Changes:**
1. For each user query, generate 2 paraphrases (one LLM call, cheap model)
2. Run ANN against all 3 queries
3. Union results, deduplicate by note_id, keep highest score per note
4. Feed unified set to reranker (P1) and synthesis

**EITHER/OR with P5** — both solve vocabulary mismatch but differently.

**Cost:** 1 extra LLM call + 2 extra ANN queries per search (~100ms total).

**Files:** `read.rs`, `llm/prompts.rs` (paraphrase prompt)

---

## P5: Hybrid Keyword Search (EITHER/OR with P4)

**Problem:** Same vocabulary mismatch as P4, but solved at the index level instead of query level.

**Changes:**
1. Build lightweight inverted index on note `keywords` field (already stored, never searched)
2. During retrieval: extract query keywords, do keyword lookup in parallel with ANN
3. Union keyword hits + ANN hits, deduplicate, score, rerank

**Advantage over P4:** No extra LLM call. Keyword matching is exact and fast. Works for every query without generating paraphrases.

**Disadvantage:** Requires building and maintaining an inverted index. Keywords may not cover all vocabulary gaps (depends on write-path keyword extraction quality).

**Implementation options:**
- A) LanceDB full-text search (if supported natively) — zero extra infrastructure
- B) SQLite FTS5 table alongside vector store — well-understood, fast
- C) In-memory HashMap<String, Vec<NoteId>> — simplest, rebuilt on startup

---

## P6: Contradiction Magnet Links (write-time detection)

**Problem:** P0.2 catches contradictions that the dream engine has already found. But dream engine runs asynchronously, after ingestion. New contradictions aren't caught until the next dream cycle.

**Changes:**
1. During write path linking step: extend the LLM prompt to detect contradictions (not just semantic relationships)
2. When contradiction detected: create a special `contradiction` link type (distinct from `semantic`)
3. During retrieval: if any ANN hit has a `contradiction` link, pull in the linked note automatically

**Relationship to P0.2:** Complementary, not either/or. P0.2 uses existing dream infrastructure. P6 catches contradictions earlier (at write time) before dreams run. Together they provide full coverage.

**Effort:** Medium — prompt change in write path, new link type, retrieval hook.

**Files:** `write.rs`, `llm/prompts.rs`, `read.rs`, `store/mod.rs`

---

## P7: Meta-Instruction Pinning

**Problem:** Style directives ("always use bullets", "format as table") embed nowhere near content questions. They get lost in ANN for long conversations.

**Changes:**
1. During write path: detect meta-instructions (imperative patterns: "always", "never", "when I ask", "format as", "remember to")
2. Tag detected notes with `meta_instruction: true` in metadata
3. During retrieval: always include active meta-instructions in context regardless of similarity score
4. Use existing profile auto-include pattern in `read.rs` as template

**Effort:** Small — keyword detection in write path, auto-include in read path.

**Files:** `write.rs`, `read.rs`, `note.rs` (add field or tag)

---

## P8: Two-Pass Retrieval on Abstention

**Problem:** After P0.4 removes synthesis-level abstention, some queries will still get poor results on first pass. A retry with expanded parameters could catch them.

**Changes:**
1. After synthesis: if the answer contains "I don't have information" or similar no-answer phrases, trigger retry
2. Retry with: 2x top_k, relaxed reranker threshold (0.005), ghost queries (P4) if available
3. Only fully abstain if retry also fails

**Depends on:** P0.4 (new abstention flow). Optional dependency on P4 (ghost queries).

**Risk:** Doubles latency on abstention cases. But abstention cases are 18% of queries, and the retry only triggers on those.

**Files:** `read.rs`

---

## P9: Overwrite-on-Update (radical, high effort)

**Problem:** knowledge_update failures happen because multiple notes exist for the same fact at different points in time. ANN retrieves the older, more-linked version.

**Changes:**
1. Extend attribute generation prompt: add `updates_existing_note_id: Option<String>` field
2. If LLM identifies a candidate among top-K similar notes as "the same fact, updated": merge instead of create
3. Update existing note's content in-place. Append to evolution_history with the old content.
4. Re-embed the updated note (content changed, embedding must change)

**Advantage:** Eliminates knowledge_update failures at the source. Only ONE note per fact, always current.

**Disadvantage:** Destructive to original content (though evolution_history preserves it). Harder to reason about "what did I believe at time T?" Changes the data model assumption from append-only to mutable.

**EITHER/OR with Recency mode in P0.3** — P0.3's Recency mode is a retrieval-side fix (boost recent notes). P9 is a write-side fix (only one note exists). P0.3 is safer to try first; P9 is the nuclear option if recency boosting isn't enough.

---

## P-LLM: Per-Operation Model Routing

**Problem:** All LLM calls go to gpt-5.4-mini via OpenAI API. Expensive and slow for commodity operations (link decisions, evolution, episode boundaries). Critical operations (attribute extraction, synthesis, contradiction detection) need quality. Local OSS models via Ollama can handle the rest.

**Tiered model assignment:**

| Operation | Quality need | Model |
|---|---|---|
| write.attributes | HIGH | gpt-5.4-mini (OpenAI) |
| write.link_decision | MEDIUM | gpt-oss-120b (Ollama) |
| write.evolution | LOW | gpt-oss-120b (Ollama) |
| episode.boundary | LOW | gpt-oss-120b (Ollama) |
| episode.narrative | MEDIUM | gpt-oss-120b (Ollama) |
| dream.deduction | MEDIUM | gpt-oss-120b (Ollama) |
| dream.induction | MEDIUM | gpt-oss-120b (Ollama) |
| dream.abduction | MEDIUM | gpt-oss-120b (Ollama) |
| dream.consolidation | MEDIUM | gpt-oss-120b (Ollama) |
| dream.contradiction | HIGH | gpt-5.4-mini (OpenAI) |
| read.synthesis | HIGHEST | gpt-5.4-mini (OpenAI) |
| embedding | KEEP | text-embedding-3-small (OpenAI, 1536-dim) |
| reranking | KEEP | Jina or Cohere cross-encoder (separate API) |

**Changes:**
1. Wire `config.llm.model_for(operation)` into all ~10 call sites (write.rs, read.rs, dream/engine.rs, episode boundary/narrative)
2. Each call site passes its operation key (e.g., "write.attributes") to get the correct model
3. Add `KARTA_BASE_URL` env var support for Ollama endpoint
4. Config example with overrides in TOML
5. Benchmark with mixed models to validate quality doesn't regress on critical paths

**Cost impact:** ~60-70% API cost reduction. Local model handles ~7 of 11 call types.

**Files:** `config.rs`, `karta.rs`, `write.rs`, `read.rs`, `dream/engine.rs`, `llm/openai.rs`

---

## Phase Next: Atomic Facts + Dream-Powered Episode Intelligence

**The big architectural change.** Inspired by EverMemOS's MemCell decomposition but combined with Karta's unique dream engine for cross-episode reasoning.

### What EverMemOS Does (and what we learn from it)

EverMemOS decomposes conversations into MemCells containing three parallel extractions:
- **Episode narrative**: third-person summary of what happened
- **Atomic facts** (EventLog): discrete, verifiable statements with timestamps. "Flask 2.3.1", "budget $500", "deadline March 15" are separate searchable units.
- **Foresight signals**: predictions with time validity windows

MemCells are clustered into MemScenes (thematic groups). ProfileManager extracts user profiles from clusters. Retrieval queries against extracted facts/episodes, not raw conversations.

**What EverMemOS doesn't do:** reason across episodes. Their clusters are passive groupings. No deduction, induction, abduction, contradiction detection, or confidence propagation.

### What Karta Does Better

Two combined changes:

**B: Atomic Facts Layer (write-time)**
1. Extend attribute extraction prompt to also produce 1-5 atomic facts per note
2. Each atomic fact stored as a separate searchable unit with its own embedding
3. AtomicFact struct: `{ id, content, source_note_id, entity: Option<String>, entity_type: Option<String>, timestamp: Option<DateTime>, confidence: f32 }`
4. Episode metadata becomes a rollup of constituent facts: entity list, date range, counts
5. Facts are independently retrievable by ANN search

**C: Dream-Powered Episode Digests (async)**
1. New dream type: "episode_digest" — processes each episode after creation
2. Produces structured metadata: entities with types/counts, date ranges, aggregations, topic sequence
3. Stored as a Provenance::Dream note linked to the episode
4. Cross-episode digests (dream-of-dreams): "user mentioned movies in episodes 1, 3, 7... total: 13 unique"
5. Read path checks episode digests first. If the digest answers the query (aggregation, entity count, date range), no drilldown needed.

**Karta advantages over EverMemOS:**

| Capability | EverMemOS | Karta (with B+C) |
|---|---|---|
| Atomic facts | Yes | Yes |
| Cross-episode reasoning | No | **Dreams reason across episodes** |
| Contradiction detection | No | **Dream engine finds contradicting facts** |
| Retroactive evolution | No | **New facts update linked notes** |
| Confidence propagation | No | **Dream-derived facts carry confidence** |
| Forgetting/decay | No | **Phase 3 lifecycle management** |

**Expected impact:**
- event_ordering: 35% → 55%+ (facts preserve micro-ordering)
- temporal_reasoning: 37% → 50%+ (date ranges on episodes, facts with dates)
- knowledge_update: 31% → 50%+ (atomic facts replace old values)
- information_extraction: 60% → 70%+ (each fact is its own embedding)
- Overall: 56% → 65-70% (target: beat Honcho's 63%)

**Files:** `note.rs` (AtomicFact struct), `write.rs` (fact extraction), `store/lance.rs` (fact storage), `store/sqlite.rs` (fact-episode mapping), `dream/engine.rs` (episode_digest dream type), `read.rs` (two-level fact retrieval), `llm/prompts.rs` (extraction + digest prompts), `episode.rs` (EpisodeMetadata struct)

**Effort:** L (CC: ~6 hours for both B+C)

---

## Experiment Sequence

```
P0 (foundation fixes) [DONE: 56.8%]
  ↓
P1 (reranker reorder + mode-specific fetch_k) [DONE: 56.4%, noise range]
  ↓
A.1 (skip reranker for Computation) [DONE: 54.7%, noise]
A.2 (embedding classifier) + note sorting + source timestamps [RUNNING]
  ↓ benchmark
A.3 (generalized P8 retry on insufficient info)
  ↓ benchmark
Phase Next: Atomic Facts + Episode Digests (the big architectural change)
  ↓ benchmark (target: 65%+)
P-LLM (per-operation model routing — cost optimization, after quality stabilizes)
```

Each benchmark = full BEAM 100K (20 conversations, 400 questions). Log per-ability scores to track which failures each change actually fixes.

---

## Mutual Exclusivity Summary

| Group | Options | Pick criteria |
|-------|---------|---------------|
| Event ordering | **P2** (episode narratives) OR **P3** (temporal chain links) | Superseded by Phase Next (atomic facts provide micro-ordering) |
| Vocabulary mismatch | **P4** (ghost queries) OR **P5** (hybrid keyword search) | P4 if keyword extraction is unreliable. P5 if keywords are good and you want zero extra LLM calls. |
| Knowledge updates | **P0.3 Recency mode** OR **P9** (overwrite-on-update) | Try P0.3 first (retrieval fix). P9 only if recency boosting isn't enough (write-side nuclear option). |
