# Benchmarks

Karta is benchmarked against public long-horizon agent-memory evaluations.
These numbers track an experimental research project in active development —
they reflect a work in progress, not a stable release.

## Current headline scores

| Benchmark | Karta (current) | Status | Details |
|---|---|---|---|
| [BEAM 100K](./beam-100k.md) | **61.6%** (P1, 2026-04-14) | Active tracking | 20 conversations, 400 questions, single run, arithmetic mean of per-question rubric scores |
| LOCOMO | — | In progress | Query-phase hang being debugged; no validated score yet |
| LongMemEval-S | — | In progress | Oracle-split temporal-reasoning partial (50 qs, 50%); balanced full-set run pending |

## What each benchmark tests

### BEAM 100K — Beyond a Million Tokens

BEAM is an ICLR 2026 benchmark built around 20 multi-turn conversations
ranging from 100K to 10M tokens. Each conversation contains 20 probing
questions across 10 memory abilities: preference following, abstention,
contradiction resolution, multi-session reasoning, instruction following,
summarization, information extraction, temporal reasoning, knowledge
update, and event ordering. Scoring uses an LLM judge against a per-question
nugget rubric (0 / 0.5 / 1 per nugget, averaged).

Karta tracks the 100K tier (400 questions) as the primary development
benchmark because it's the smallest tier where ingest runs complete in
hours rather than days.

**Reproduction:**

```bash
# Requires .env with LLM credentials (OPENAI_API_KEY or AZURE_OPENAI_*)
BEAM_DATASET_PATH=data/beam-100k.json \
  cargo test --release -p karta-core --test beam_100k \
    beam_100k_full -- --ignored --nocapture
```

See [beam-100k.md](./beam-100k.md) for the full experiment log, failure
catalogue, per-ability history, and analysis of what moved the needle.

### LOCOMO

LoCoMo is a multi-turn long-conversation benchmark with single-hop,
multi-hop, open-domain, and temporal question types. Karta's LOCOMO run
currently hangs on the query phase after ingesting 419 notes + 139 dreams;
debugging is in progress.

### LongMemEval (S)

LongMemEval tests 5 memory abilities across 500 questions in sessions up
to 115K tokens (ICLR 2025). A partial run on the oracle split (first 50
questions, all temporal reasoning) scored 50%. A balanced full-set run
across all 6 question types is pending.

## Reproducibility notes

- **Single run per headline number.** Published scores are single runs, not
  seed averages. Treat deltas under ~3pp as within LLM non-determinism.
- **Judge model matters.** BEAM's absolute scores swing double-digits with
  judge choice. The reproduction commands above use the same judge prompt
  shipped in the BEAM dataset to stay comparable with published baselines.
- **Ingest is slow.** A full BEAM 100K run takes 10–12 hours on a
  well-provisioned machine due to the cost of real LLM calls during
  ingest + dreaming. Use the `.results/` directory (gitignored) for run
  logs.
- **Experimental, not frozen.** The benchmark harness, configuration
  defaults, and scoring pipeline can change between runs as Karta
  evolves. Historical runs in `beam-100k.md` note the config used for
  each data point.
