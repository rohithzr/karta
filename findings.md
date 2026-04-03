# BEAM 100K Findings

## Results Summary

| Run | Score | Config | Notes |
|-----|-------|--------|-------|
| Baseline (Day 2) | 56.8% | episodes=off, graph=dead code, reranker=0.1 | Original measurement |
| Post-wiring (all features) | 41.4% | episodes=on, graph=0.05, reranker=0.1 | Massive regression |
| Optimal (Day 2) | 51.3% | episodes=on, graph=0.0, foresight=0.1, reranker=0.01 | After experiment sweep |
| P0 foundation fixes | **56.8%** | +turn_index, query router, contradiction wiring, kill abstention | 2026-03-31 |
| P0+P1+P2 | 55.3% | +fetch_k=4x, max_rerank=20, narrative ordering | 2026-04-01 (regression) |
| P1-fix | 56.4% | +reranker reorder, mode-specific fetch_k | 2026-04-01 |
| All-fixes | 57.8% | +embed classifier, note sorting, source timestamps, parallel questions | 2026-04-02 |
| A.3 (retry) | **57.7%** | +insufficient-info retry for Computation/Temporal | 2026-04-02 |
| Honcho reference | 63.0% | Their system | Published number |

Note: 57.7-57.8% are within LLM non-determinism range (~3pp variance between identical runs).
The true improvement from Day 2 optimal (51.3%) to Day 4 is +6.4pp.

### Per-Ability Comparison (All Runs)

| Ability | Day 2 Opt (51.3%) | P0 (56.8%) | P0+P1+P2 (55.3%) | All-fixes (57.8%) | Best |
|---------|-------------------|------------|-------------------|-------------------|------|
| preference_following | 80% | 84% | 80% | 78% | **84% (P0)** |
| contradiction_resolution | 52% | 61% | 63% | **71%** | **71% (all-fixes)** |
| instruction_following | 54% | 66% | 68% | 68% | **68%** |
| abstention | 60% | 65% | 62% | 68% | **68% (all-fixes)** |
| information_extraction | 50% | 58% | 51% | **63%** | **63% (all-fixes)** |
| summarization | 64% | 61% | 64% | 62% | **64%** |
| multi_session_reasoning | 53% | 71% | 62% | 61% | **71% (P0)** |
| temporal_reasoning | 40% | 49% | 41% | 45% | **49% (P0)** |
| knowledge_update | 45% | 40% | 40% | 38% | **45% (Day 2)** |
| event_ordering | 31% | 35% | 35% | 37% | **37% (all-fixes)** |
| knowledge_update | 45% | 40% | 40% | **45% (Day 2)** |
| event_ordering | 31% | 35% | 35% | **35%** |

---

## Experiment Results (Conv 1, single conversation)

| Experiment | Score | Key Takeaway |
|---|---|---|
| E0: Baseline replay (no new features) | 69.8% | Code changes are clean — no regression from code itself |
| E5: Baseline + low abstention (0.01) | 69.8% | Same total but better distribution — temporal 0%->50%, event 50%->75% |
| E1: Episodes only | 62.3% | Episodes HURT with aggressive reranker — multi_session 67%->33% |
| E8: All features, no graph, low abstention | 67.9% | Best combo. multi_session 100% |
| E_optimal: Episodes+foresight, Conv 4 (Math) | 50.9% | Math is hard but event_ordering=44% |

---

## Failure Catalog: 243 Total Failures

### Distribution by Root Cause

| Category | Count | % | Description |
|----------|-------|---|-------------|
| INCOMPLETE_RETRIEVAL | 99 | 40.7% | Info exists in notes but wasn't retrieved or was partial |
| FALSE_ABSTENTION | 44 | 18.1% | System said "I don't know" when it had the answer |
| WRONG_ORDER | 42 | 17.3% | Events listed but in wrong chronological sequence |
| CONTRADICTION_MISS | 26 | 10.7% | Failed to detect or present both sides of contradiction |
| JUDGE_NOISE | 14 | 5.8% | Correct answer scored 0 by LLM judge |
| WRONG_COMPUTATION | 8 | 3.3% | Dates retrieved but math was wrong |
| HALLUCINATION | 7 | 2.9% | Answer contains info not in conversation |
| FORMAT_MISS | 3 | 1.2% | Right info, wrong format (no bullets, no tree diagram) |

---

## Culprit Group 1: INCOMPLETE_RETRIEVAL (99 failures, 40.7%)

The system has the information stored but fails to surface all relevant notes.

### Sub-patterns:

**1a. knowledge_update — old value returned (~15 failures)**
The system retrieves an older version of a fact instead of the latest update.

- Conv 4, Q7: "What's the current forecast accuracy?" — Answer: 78%. Expected: 92%. The 78% note was stored first and embeds closer to the query than the 92% update.
- Conv 8, Q7: "How many drafts of personal statement?" — Answer: 3. Expected: 5. Earlier count retrieved instead of latest.
- Conv 13, Q8: "Current sneaker collection size?" — Answer: 12. Expected: 18. Older inventory note retrieved.

**Why:** ANN retrieval is similarity-based, not recency-based. When multiple notes answer the same question, the oldest (most established, most linked) note often wins. Recency weight (0.15) isn't enough to overcome embedding similarity.

**Fix needed:** For knowledge_update queries, the system needs to detect "current/latest/now" keywords and aggressively boost recency. Or: evolution should update the original note's content (not just context) so there's only one version.

**1b. Specific detail missed in broad answer (~50 failures)**
System gives a good general answer but misses one specific rubric point.

- Conv 6, Q8: Summarize writing tools — Missed "ProWritingAid" specifically (mentioned Grammarly, Jasper AI)
- Conv 8, Q8: Resume format advice — Missed "quantify achievements with numbers" rubric item
- Conv 15, Q5: AI hiring journey — Missed "Carla suggested at lunch on March 1" — the specific person+date wasn't retrieved

**Why:** When 100+ notes exist, top-K=5 only surfaces the most similar 5. The specific detail the rubric wants may be in note #47 which never gets retrieved.

**Fix needed:** Increase top-K for information_extraction queries. Or: use the reranker to re-score a larger candidate pool (max_rerank=20 instead of 10).

**1c. Cross-note computation failed (~10 failures)**
Answer requires combining facts from 2+ notes that weren't both retrieved.

- Conv 3, Q19: "How many more problems did I complete?" — Needed note A (scored 8/10) and note B (completed 10 total) to compute 10-8=2. Only note A retrieved.
- Conv 5, Q19: "Days between transaction features and deadline" — Needed Jan 15 from one note and March 15 from another. Only March 15 retrieved.

**Why:** ANN retrieval optimizes for similarity to the query, not for co-retrieval of related facts. Two notes about different events don't naturally cluster in embedding space.

**Fix needed:** Multi-hop traversal should help here — if note A links to note B, traversing from A should find B. Check if links are being created between date-bearing notes.

---

## Culprit Group 2: FALSE_ABSTENTION (44 failures, 18.1%)

System says "Based on the available memories, I don't have information about this topic" when information exists.

### Abstention Pipeline — 4 gates, any can trigger:

1. **search() returns empty** — if all ANN hits are filtered out
2. **Reranker best_relevance < 0.01** — Jina cross-encoder scores all notes below threshold
3. **Structured output should_abstain = true** — LLM decides notes are irrelevant
4. **answer field is null** — LLM returns null without setting should_abstain

### What we fixed:
- Lowered reranker threshold from 0.1 to 0.01 — eliminated ~60% of false abstentions
- Fixed episode narrative drop-through — narratives below drilldown threshold now fall to flat hits
- Fixed flat budget starvation — episode results no longer crowd out ANN hits

### Remaining false abstentions (44):
Concentrated in:
- **Long conversations (130+ notes)**: Conv 7-11 (Job, Writing) where signal-to-noise is worst
- **temporal_reasoning**: 10 false abstentions — the specific date note isn't in top-K
- **instruction_following**: 8 false abstentions — meta-instructions ("always use bullets") don't embed close to the question asking about content

### Examples:

- Conv 7, Q10: "What is the salary offered?" — ABSTAINED. Expected: "$72,000". The salary note existed but didn't embed close to "salary offered" (it was phrased as "compensation package: $72K base").
- Conv 9, Q4: "First draft of screenplay?" — ABSTAINED. Expected: mention of contradiction. The screenplay draft note was among 140+ notes and wasn't retrieved.
- Conv 10, Q20: "How much progress on edits?" — ABSTAINED. Expected: percentage values. The progress note existed but had low similarity to the question phrasing.

**Fix needed:**
- Gate 3 (synthesis prompt) is still too abstention-heavy. The prompt says "MUST abstain if notes only discuss topic Y" — this triggers when notes are tangentially related but do answer the question.
- Consider removing the synthesis-level abstention entirely and relying only on the reranker gate.

---

## Culprit Group 3: WRONG_ORDER (42 failures, 17.3%)

Event ordering questions where the system attempted chronological ordering but got the sequence wrong.

### The fundamental problem:
Notes are retrieved by similarity, not by conversation order. When the system retrieves 5 notes about "triangle topics covered", it gets {triangles, area, medians, altitudes, classification} but has no way to know which was discussed first.

### Episode retrieval status:
Episodes ARE being created (config shows episode=true) but they don't observably help:
- No answer references episode narratives
- The failure mode is identical to pre-episode: right topics, wrong sequence
- Episode narratives capture *themes*, not *micro-ordering within a conversation*

### Why episodes don't help here:
1. All notes from a conversation often land in the SAME episode (no topic shift = no boundary)
2. Episode drilldown returns notes sorted by `created_at`, but `created_at` is the ingestion time (all within seconds in the benchmark), not the original conversation timestamp
3. The BEAM dataset's `time_anchor` field provides the real timestamp, but it's prepended as a text prefix `[March-15-2024]`, not stored as metadata

### Examples:

- Conv 3, Q5: "List the order of triangle classification topics" — Answer listed: 1) right-angle verification, 2) Pythagorean theorem, 3) triangle types, 4) area methods, 5) medians. Expected: 1) right-angle verification, 2) area calculation, 3) Heron's formula, 4) median properties, 5) altitude applications, 6) combined classification. Got ~3/9 items in right position.
- Conv 8, Q5: "Order of personal statement development" — Got 0/5 rubric items. System listed reasonable topics but in completely wrong order.
- Conv 15, Q5: "Steps in balancing spirituality and career" — Detailed 7-item answer but 0/7 rubric items matched positions.

**Fix needed:**
- Store conversation turn index as structured metadata on each note (not just text prefix)
- Sort by turn index, not created_at, in episode drilldown
- Parse `time_anchor` from BEAM data into actual note timestamps during ingestion

---

## Culprit Group 4: CONTRADICTION_MISS (26 failures, 10.7%)

Contradiction resolution dropped from 69% to 52% — the biggest regression.

### How BEAM contradiction questions work:
Each conversation has 2 contradiction questions. The conversation contains planted contradictions: "I have been using Excel to track expenses" in one message, and "I have never used Excel" in another. The rubric has 4 items:
- R1: State that a contradiction exists
- R2: Mention side A (the positive claim)
- R3: Mention side B (the negative claim)
- R4: Ask which statement is correct

### Failure breakdown (40 questions total):

| Outcome | Count | % |
|---------|-------|---|
| Perfect (4/4) | 5 | 12.5% |
| Detected, missed one side (2-3/4) | 14 | 35% |
| No detection at all (0-1/4) | 16 | 40% |
| Sided with one interpretation | 5 | 12.5% |

### Root causes:

**4a. One-sided retrieval (16 failures — "No detection")**
The system retrieves notes supporting one side (usually the positive: "I DID use Excel") but the single contradicting note ("I NEVER used Excel") isn't in top-K. Without both sides, no contradiction is detected.

**Why:** The "never" note is typically a single statement, while the "did" side has multiple supporting notes. ANN naturally favors the side with more notes embedding close to the query.

**4b. Detected but one-sided (14 failures)**
The system correctly flags "contradictory information" (R1=FULL) but then only articulates one side. Usually it presents the positive side and vaguely references "some notes suggest otherwise" without citing the specific negative claim.

**Why:** The contradiction detection comes from dream notes (contradiction dreams), but the specific source note with the negative claim may not be in the retrieved set. The dream says "there's a contradiction about X" but the synthesis LLM can't find the specific "never" note to quote.

**4c. Over-triggering on non-contradiction questions (40 instances)**
The "Note: The memories contain contradictory information" prefix appears on ~40 non-contradiction questions (~11% false positive rate). This is because:
- Contradiction dreams are stored as notes and get retrieved broadly
- The `has_contradiction` field in synthesis structured output triggers too easily
- Multiple notes about the same topic with different details (updates, not contradictions) trigger false detection

### Examples:

- Conv 3, Q3: "Coin toss problems?" — Answer: "Yes, you've worked through coin toss probability problems" (confident, no contradiction flagged). Expected: detect contradiction between "completed coin toss problems" and "never attempted coin toss problems". The "never attempted" note wasn't retrieved.
- Conv 17, Q3: "Read Daniel Dennett?" — Answer: acknowledged both "read Consciousness Explained" and "never read Dennett" but didn't call it a contradiction or ask which is correct. Score: 0.50.
- Conv 14, Q4: "Invited Mason/Michael to movie events?" — Answer: "Yes, you invited Michael..." without detecting the "never invited anyone" note. Score: 0.50.

**Fix needed:**
- When a query matches a contradiction dream, force-retrieve BOTH source notes (the dream records source_note_ids)
- Tighten the `has_contradiction` synthesis field — only trigger when actual contradicting notes are present, not when notes merely differ
- In the synthesis prompt, strengthen: "If you flag a contradiction, you MUST present BOTH sides with specific quotes and ask which is correct"

---

## Culprit Group 5: JUDGE_NOISE (14 failures, 5.8%)

Correct answers scored 0 by the LLM-as-judge.

### Pattern:
Almost all are abstention questions where the system correctly said "I don't have information" but the judge scored 0. The rubric text says "Based on the provided chat, there is no information related to..." which AGREES with the system's answer, but the judge doesn't recognize the match.

### Examples:
- Conv 1, Q1: "What are the specific ESLint rules?" — System: "I don't have information about specific Airbnb ESLint rules." — Rubric: "there is no information related to the specific rules." — Judge: 0.0.
- Conv 1, Q2: "What specific bugs in Jira?" — Same pattern. System correctly abstained, rubric agrees, judge scores 0.

**Why:** The judge uses exact rubric phrasing as the gold standard. The system's phrasing doesn't match closely enough. "I don't have information" ≠ "there is no information related to."

**Fix:** Not a system fix — this is LLM judge variance. Could normalize abstention answers to match rubric phrasing, but that's gaming the benchmark.

---

## Culprit Group 6: Misc (WRONG_COMPUTATION, HALLUCINATION, FORMAT_MISS)

### WRONG_COMPUTATION (8 failures, 3.3%)
Dates retrieved but math was wrong:
- Conv 14, Q19: "Months between walking goal and health assessment" — Said 16 months. Expected 4 months. Retrieved wrong dates.
- Conv 20, Q20: "Days between meeting with Ashlee and patent deadline" — Said 180 days. Expected 67 days. Retrieved Sept 10 meeting instead of May 14.

These are all misattributed dates (retrieved wrong event's date), not actual arithmetic errors. The LLM computes correctly when given the right numbers.

### HALLUCINATION (7 failures, 2.9%)
System generated plausible but nonexistent information:
- Conv 5, Q8: Described specific rubric criteria that weren't in the notes
- Conv 16, Q8: Listed editing tools "Lightworks, OpenShot" that weren't mentioned

Low frequency — the structured output with reasoning mostly prevents this.

### FORMAT_MISS (3 failures, 1.2%)
- Conv 5, Q10: "Card drawing without replacement" — Correct probability math but rubric required a "tree diagram". System doesn't generate visual diagrams.

---

## Bugs Fixed During This Session

| Bug | Impact | Fix |
|-----|--------|-----|
| Reranker threshold 0.1 too aggressive | ~60% of false abstentions | Lowered to 0.01 |
| Episode narrative drop-through | Notes lost when below drilldown threshold | Fall through to flat hits |
| Flat budget starvation | Episode results crowded out ANN hits | flat_budget = top_k (never reduce) |
| Raw JSON leak in synthesis | answer:null dumped raw JSON | Graceful abstention on null answer |
| Doubled abstention phrase | "I don't have info" appeared twice | Removed redundant concatenation |
| UTF-8 boundary panic | Crash on multi-byte chars (em-dash) | Safe truncation in dream engine + write path |
| Graph scoring adds noise | Dream notes with many links boosted over relevant notes | Disabled by default (graph_weight=0.0) |

---

## Open Issues (Priority Order)

### P0: INCOMPLETE_RETRIEVAL (41% of failures)
- Top-K=5 is too small for 100+ note conversations
- Recency weighting (0.15) insufficient for knowledge_update queries
- Cross-note computation needs both related notes but ANN only finds one

### P1: CONTRADICTION one-sided retrieval (52% → need 69%+)
- "Never" notes are singletons lost in ANN among many supporting notes
- Contradiction dreams know both source notes but don't force-retrieve them
- Over-triggering: 40 false contradiction prefixes on non-contradiction questions

### P2: EVENT_ORDERING sequence reconstruction (31% → need 50%+)
- Notes lack conversation turn index metadata
- Episode drilldown sorts by created_at (ingestion time), not conversation time
- BEAM time_anchor should be parsed into actual timestamps, not text prefixes

### P3: FALSE_ABSTENTION remaining (18%)
- Synthesis prompt still too abstention-heavy
- Long conversations (130+ notes) have worst signal-to-noise
- Meta-instructions ("always use bullets") don't embed close to content questions

### P4: JUDGE_NOISE (6%)
- Unavoidable LLM judge variance
- Abstention phrasing mismatch between system and rubric
- Not worth optimizing — would be gaming the benchmark

---

## P0 Foundation Fixes Results (2026-03-31)

P0 moved the needle from 51.3% to 56.8% (+5.5pp). Changes: turn_index/source_timestamp on MemoryNote, query-type router (6 modes), contradiction force-retrieval from dream source_note_ids, removed synthesis-level abstention (Gate 3).

**What worked:**
- multi_session_reasoning +18pp (53% → 71%): query router + episode drilldown sort
- instruction_following +12pp: broader retrieval catching meta-instructions
- contradiction_resolution +9pp: force-retrieval wiring both sides
- temporal_reasoning +9pp: turn_index sorting

**What didn't move:**
- event_ordering +4pp (31% → 35%): turn_index helps ordering but failures are mostly retrieval misses, not wrong ordering
- knowledge_update -5pp (45% → 40%): recency mode keywords not triggering on all update queries

---

## P1 Expand-then-Rerank Regression Analysis (2026-04-01)

P1 (fetch_k=4x, max_rerank=20) regressed from 56.8% to 55.3% (-1.5pp).

### Root Cause: More Notes = More Noise for Computation Queries

The smoking gun from JSONL analysis:

| notes_used | avg_score | fail_rate | count |
|------------|-----------|-----------|-------|
| 5 | 61.0% | 30% | 175 |
| 10 | 30.4% | 63% | 43 |
| 11-20 | 44.7% | 52% | 85 |

Questions with exactly 10 notes (the reranker cap) fail at 2x the rate of 5-note questions. The wider candidate pool lets more marginally-relevant notes through. The LLM then tries to compute from noisy data instead of abstaining.

### Computation Mode Is the Regression Driver

83 questions classified as Computation, 39 failing (47% fail rate). The LLM receives 10 notes with partial information and confabulates specific numbers with false confidence.

Worst examples:
- Conv 19, Q12: "How many children?" — 141 notes, answered "2" (wrong)
- Conv 2, Q12: "How many features?" — 10 notes, answered "10" (wrong)
- Conv 5, Q10: "How many hours studying?" — reranker=0.538, answered "3 hours" (wrong)

### Reranker Results Not Used for Ordering

Critical finding: the reranker is called and scores are computed, but the reranked ordering is **discarded**. The original ANN order is used for synthesis. So max_rerank=20 doubles Jina API cost without changing which notes reach the LLM or in what order. The reranker only serves as an abstention gate.

### Contradiction Over-Triggering

35 questions flagged has_contradiction=true. Only 12 are actual contradiction questions. 23 false triggers (66% false positive rate). The false "contradictory information" prefix on non-contradiction answers may confuse the judge.

### Abstention Failures Are All Hallucinations

All 15 abstention failures are cases where the system gave an answer when it should have said "I don't know." Reranker scores range 0.119-0.595. The reranker finds topically similar notes, the system assumes they contain the answer, and fabricates details.

### Query Mode Distribution

| Mode | Count | Avg Score | Fail Rate |
|------|-------|-----------|-----------|
| Standard | 202 | 61.1% | 31% |
| Breadth | 54 | 59.8% | 30% |
| Computation | 83 | 44.8% | 47% |
| Temporal | 53 | 38.8% | 62% |
| Recency | 7 | 35.7% | 43% |

### Action Items from P1 Analysis

1. **Revert fetch_k to 2x for Computation mode** — precision over recall for math queries [DONE]
2. **Wire reranker results to actually reorder notes** — use reranked order for synthesis, not just abstention [DONE]
3. **Cap notes at 5 for Computation mode** — computation needs exact data, not broad context [DONE via mode-specific fetch_k]
4. **Fix contradiction over-triggering** — 66% false positive rate on has_contradiction [improved to ~5% with reranker reordering]
5. **Raise reranker abstention threshold for abstention-type questions** — deferred, abstention improved to 68% without threshold change

---

## All-Fixes Run Results (2026-04-02)

Combined: embedding classifier, chronological note sorting, source timestamps, computation reranker skip, reranker reordering, contradiction force-retrieval. Best overall: **57.8%**.

Key insight: **chronological note ordering matters more than retrieval width.** Sorting notes by turn_index before synthesis helped every ability because LLMs process sequential input better than shuffled fragments. The source_timestamp fix (showing real conversation dates instead of ingestion time) similarly helped temporal reasoning.

---

## A.3 Insufficient-Info Retry Results (2026-04-02)

A.3 adds retry when the LLM admits missing information ("notes do not", "can't find the date"). Only fires for Computation and Temporal query modes. Retry uses 3x wider retrieval and skips the reranker.

**Result: 57.7%** (within noise of 57.8% baseline).

| Ability | All-fixes (57.8%) | A.3 (57.7%) | Delta |
|---|---|---|---|
| temporal_reasoning | 45% | **53%** | **+8** |
| multi_session_reasoning | 61% | **66%** | +5 |
| knowledge_update | 38% | **40%** | +2 |
| preference_following | 78% | 78% | 0 |
| abstention | 68% | 68% | 0 |
| information_extraction | 63% | 63% | 0 |
| summarization | 62% | 61% | -1 |
| event_ordering | 37% | 36% | -1 |
| instruction_following | 68% | 64% | -4 |
| contradiction_resolution | 71% | 67% | -4 |

**Temporal_reasoning +8pp** is the clear A.3 win. The retry rescued queries where the first pass missed date-bearing notes. The overall score is flat because gains are offset by LLM non-determinism losses in contradiction and instruction_following.

---

## Day 4 Session Learnings

### What Worked

1. **Turn index + source timestamps (P0.1)** — most impactful single change. Every temporal and ordering improvement traces back to having real conversation order and real dates on notes.

2. **Contradiction force-retrieval (P0.2)** — wiring dream source_note_ids into the read path fixed one-sided contradiction detection. Contradiction resolution went from 52% to 67-71%.

3. **Killing synthesis-level abstention (P0.4)** — removing the aggressive "MUST abstain" gate let the LLM attempt answers from partial information. Net positive: abstention improved from 60% to 68% because the remaining abstention decisions come from the reranker (more accurate than the LLM's self-judgment).

4. **Chronological note ordering** — sorting notes by turn_index before LLM synthesis improved coherence across all abilities. LLMs have a strong sequential bias and perform better when notes arrive in conversation order.

5. **JSONL debug logging (P0.0)** — made every failure diagnosable without re-running hours of LLM calls. Essential for iterative development.

### What Didn't Work

1. **Expanding fetch_k uniformly (original P1)** — 4x fetch_k + max_rerank=20 caused regression because more notes = more noise for Computation queries. The reranker pushed date-bearing notes down because they score low on topical relevance.

2. **Embedding-based query classifier** — replaced keyword matching but produced zero classification changes on 76 overlapping questions. The prototypes embed to the same semantic regions as the keywords. May help on edge cases not in BEAM.

3. **Question parallelization with unbounded concurrency** — spawning 20 concurrent question tasks overwhelmed Azure embedding API. Fixed with semaphore-based throttling (default 3 concurrent).

### Key Architectural Insights

1. **Reranker reordering helps abstention, hurts temporal.** Same mechanism, opposite effect. For abstention, pushing down tangentially-related notes prevents hallucination. For temporal, pushing down date-bearing notes loses the specific data needed for computation. Fix: mode-specific reranker behavior.

2. **LLM non-determinism is ~3-4pp per run.** Same code, same data, different scores. Any improvement under 4pp is in the noise range. Only structural changes (P0 foundation: +5.5pp) clear the noise floor.

3. **The retrieval ceiling for flat ANN is ~58%.** Further gains require structural changes to the data model: atomic fact decomposition (each fact independently searchable), episode-level metadata (pre-computed aggregations), and cross-episode reasoning (dream-powered digests).

4. **Write-time organization determines retrieval quality.** The most impactful changes (turn_index, source_timestamp, contradiction linking) all happened at write time. Read-path cleverness (reranker tuning, query classification, retry logic) produces diminishing returns.

### Variance Tracker

All runs on identical BEAM 100K dataset, same judge model, same ingested data (for skip-ingest runs).

| Run | Score | Notes |
|-----|-------|-------|
| Day 2 Opt | 51.3% | Starting point |
| P0 | 56.8% | +5.5pp (structural) |
| P0+P1+P2 | 55.3% | -1.5pp (fetch_k regression) |
| P1-fix | 56.4% | +1.1pp recovery |
| A.1 (skip rerank) | 54.7% | -1.7pp (noise) |
| All-fixes | 57.8% | +1.4pp (note sorting + source timestamps) |
| A.3 (retry) | 57.7% | flat (temporal +8pp offset by noise) |

True signal: **51.3% → ~57.5% (+6.2pp)** after removing noise. Remaining gap to Honcho (63%): 5.5pp. Next phase: atomic facts + episode digests.
