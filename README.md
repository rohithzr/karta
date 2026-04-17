# Karta

> ⚠️ **Experimental research project.** Karta is a novel, actively-evolving
> approach to AI agent memory. APIs, data formats, and benchmark numbers
> will change. Not production-ready. We're building this in the open because
> we think agent memory is one of the most important unsolved problems in
> AI — and our goal is to build the best AI memory system.

An agentic memory system that thinks, not just stores.

Karta is a novel approach that combines structured knowledge graphs with
active background reasoning to build memory that organizes itself at write
time, links associatively, evolves retroactively, and periodically "dreams"
to surface deductions, patterns, gaps, and contradictions that no retrieval
system could find because they were never explicitly stored.

## Features

- **Self-organizing note graph** — notes are enriched with LLM-generated context, semantically linked, and retroactively evolved when new information arrives
- **Dream engine** — 5 types of background inference: deduction, induction, abduction, consolidation, contradiction detection
- **Retrieve-only API** — `fetch_memories(query)` returns assembled, ready-to-use context without calling any LLM for composition. Bring your own model for the answering step.
- **Three-model architecture** — cleanly separates the *core* LLM (ingest / dream / retrieval internals) from the *answer* LLM (final composition). Use a cheap/local model for 97% of the work and a premium model only for user-facing answers.
- **Structured output with reasoning** — forces chain-of-thought before answers, enabling reliable abstention and contradiction flagging
- **Cross-encoder reranking** — Jina AI reranker for precise relevance scoring and intelligent abstention
- **Temporal awareness** — exponential decay scoring, foresight signals with validity windows
- **Provenance tracking** — every note tagged as FACT or INFERRED with confidence scores
- **Forgetting** — note lifecycle (Active → Deprecated → Superseded → Archived) with access-based decay
- **Embedded by default** — LanceDB + SQLite, zero infrastructure. `cargo add karta` and go.

## Quick Start

```rust
use karta_core::{Karta, config::KartaConfig};

#[tokio::main]
async fn main() {
    let config = KartaConfig::default();
    let karta = Karta::with_defaults(config).await.unwrap();

    // Add memories
    karta.add_note("Sarah prefers Slack notifications over email").await.unwrap();
    karta.add_note("Brightline requires all integrations on the approved vendor list").await.unwrap();

    // --- Retrieve only ---
    // Karta returns assembled context; you run your own LLM on top.
    let memories = karta.fetch_memories("What do I know about Sarah?", 5).await.unwrap();
    println!("Mode: {} | notes: {}", memories.query_mode, memories.notes.len());
    println!("Context:\n{}", memories.context);
    // Feed memories.context to whatever model / prompt / agent you want.

    // --- Or let Karta compose the answer too ---
    let result = karta.ask("What should I know about Sarah's preferences?", 5).await.unwrap();
    println!("{}", result.answer);

    // Run background reasoning
    let dream_run = karta.run_dreaming("workspace", "default").await.unwrap();
    println!("Dreams: {} attempted, {} written", dream_run.dreams_attempted, dream_run.dreams_written);
}
```

### Retrieve-only vs. ask()

Karta exposes both shapes on purpose:

| | `fetch_memories(query, top_k)` | `ask(query, top_k)` |
|---|---|---|
| Returns | `FetchedMemories { context, notes, note_ids, query_mode, reranker_best_score, … }` | `AskResult { answer, note_ids, has_contradiction, … }` |
| Calls the answer LLM | **No** — caller composes | Yes — uses the configured answer LLM |
| When to reach for it | Agents, RAG pipelines, custom prompts, displaying memories to users, routing to multiple models | Scripts, benchmarks, the "just give me the answer" path |

With `fetch_memories`, Karta's responsibility ends at "here are the relevant
memories, pre-assembled into a provenance-tagged context string" — the caller
decides what model to run, what prompt to wrap it in, and how to present the
result. This is the right shape for agent frameworks, RAG pipelines, and any
workflow that wants full control over the generation step.

## Architecture

```
Write Path:  content → LLM attributes → embed → ANN search → link → evolve → store
Read Path:   query → embed → ANN → rerank → multi-hop traverse → synthesize
Dream Path:  cluster notes → deduction/induction/abduction/consolidation/contradiction → persist
```

### Storage (trait-based, pluggable)

| Layer | Default | Production Options |
|-------|---------|-------------------|
| Vector + metadata | LanceDB (embedded) | pgvector, Qdrant |
| Graph + state | SQLite (WAL mode) | Postgres, Dolt |

### LLM Provider (trait-based)

| Provider | Status |
|----------|--------|
| OpenAI / Azure OpenAI | Built |
| Any OpenAI-compatible (Ollama, vLLM, Groq, Together) | Built |
| Anthropic | Planned |

### Three-model architecture

Karta separates the LLM it uses for *internal work* from the LLM it uses for
the *final user-facing answer*. This lets you use a cheap/local model for
the bulk of the work and reserve a premium model for the one call where
output quality directly reaches the user.

| Role | What it does | ~% of LLM calls | Typical choice |
|---|---|---|---|
| **Core** (`KARTA_CORE_MODEL`) | Write-side fact extraction, dream digests, link analysis, query classification, reranking | ~97% | Local Ollama (`gemma4:e4b`, `qwen3`, etc.) |
| **Answer** (`KARTA_ANSWER_MODEL`) | The final `synthesize` call in the read path — composes the user-facing answer from retrieved memories | ~3% | Hosted premium (e.g. `gpt-5.4-mini`) |
| **Judge** (`KARTA_JUDGE_MODEL`) | BEAM benchmark harness only — never used in normal operation | 0% at runtime | Held fixed for reproducibility |

`KARTA_ANSWER_MODEL` is optional. When unset, the core model handles answer
composition too (single-model mode, back-compat with earlier releases). When
set, Karta auto-detects the right backend: an explicit
`KARTA_ANSWER_BASE_URL` wins, otherwise Azure if credentials are present.

If you use `fetch_memories` instead of `ask`, you bypass the answer LLM
entirely — Karta's job ends at retrieval, and you run whatever you want.

### Reranker (trait-based)

| Provider | Status |
|----------|--------|
| Jina AI (cross-encoder) | Built |
| LLM-based (fallback) | Built |
| Noop (disabled) | Built |

## Configuration

```bash
cp .env.example .env
# Fill in your LLM credentials
```

### Environment variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_BASE` | OpenAI-compatible endpoint URL (Ollama, vLLM, Groq, Together, OpenAI). Wins over Azure when set. |
| `OPENAI_API_KEY` | API key for the above. Use any non-empty value (e.g. `ollama`) for endpoints that ignore it. |
| `KARTA_CORE_MODEL` | **Core LLM** — all internal work (write / dream / rerank / classify). Alias: `KARTA_CHAT_MODEL`. |
| `KARTA_EMBEDDING_MODEL` | Embedding model name. Dim is auto-probed at startup. |
| `KARTA_ANSWER_MODEL` | **Answer LLM** (optional) — the model used only for the final synthesis call. If unset, core handles answering too. |
| `KARTA_ANSWER_BASE_URL` | Optional explicit endpoint for the answer LLM. Defaults to Azure (if creds present) or core backend. |
| `KARTA_ANSWER_API_KEY` | Optional API key for the custom answer endpoint. |
| `AZURE_OPENAI_API_KEY` / `_ENDPOINT` / `_API_VERSION` | Azure credentials. When present alongside `OPENAI_API_BASE`, embeddings auto-route to Azure while chat stays on the OpenAI-compatible endpoint. |
| `AZURE_OPENAI_CHAT_MODEL` / `_EMBEDDING_MODEL` | Azure deployment names (not model names). |
| `KARTA_JUDGE_MODEL` | BEAM harness only — the model used to grade answers during benchmarks. |
| `JINA_API_KEY` | Optional — Jina reranker. Falls back to LLM reranker if unset. |

See [`.env.example`](./.env.example) for a fully-commented template.

### Config file (TOML) — per-operation model flexibility

```toml
[llm.default]
provider = "openai"
model = "gpt-4o-mini"

[llm.dream.abduction]
model = "claude-sonnet-4-6"

[reranker]
enabled = true
abstention_threshold = 0.1
```

## Benchmarks

**BEAM 100K** — 20 conversations, 400 questions, single run, arithmetic mean
of per-question rubric scores with the BEAM nugget LLM judge:

| Ability | P1 (gpt-5.4-mini) | P1-OSS ([gemma-4-e4b](https://huggingface.co/google/gemma-4-E4B)) |
|---|---|---|
| Preference Following | 92% | 79% |
| Instruction Following | 69% | 73% |
| Contradiction Resolution | 74% | 72% |
| Abstention | 62% | 72% |
| Multi-session Reasoning | 65% | 68% |
| Summarization | 66% | 62% |
| Information Extraction | 59% | 61% |
| Temporal Reasoning | 71% | 58% |
| Event Ordering | 35% | 45% |
| Knowledge Update | 43% | 41% |
| **Overall** | **61.6%** | **61.5%** |

P1-OSS reproduces the P1 result using fully open-source models for
Karta's core pipeline. All internal LLM work — fact extraction, linking,
dream consolidation, query classification, reranking — runs on
[gemma-4-e4b](https://huggingface.co/google/gemma-4-E4B) (8B MoE,
~4B active parameters) via Ollama on a single A10G GPU. Only the final
answer-composition step uses a hosted model (gpt-5.4-mini via Azure).
This proves Karta's memory system works with open-weight models — the
quality of stored memory is model-agnostic.

Up from a 53.0% P0 baseline (+8.6pp). Active development targeting 90%+
via retrieval improvements and dream quality. See
[`benchmarks/beam-100k.md`](./benchmarks/beam-100k.md) for the full
experiment log, failure catalogue, and per-ability history.

## Project Structure

```
karta/
├── crates/
│   ├── karta-core/          # Core engine (Rust)
│   │   ├── src/
│   │   │   ├── note.rs      # MemoryNote, Provenance, NoteStatus
│   │   │   ├── write.rs     # Write path (index, link, evolve)
│   │   │   ├── read.rs      # Search, multi-hop traversal, synthesis
│   │   │   ├── rerank.rs    # Cross-encoder reranking + abstention
│   │   │   ├── dream/       # Dream engine (5 inference types)
│   │   │   ├── store/       # LanceDB + SQLite implementations
│   │   │   └── llm/         # Provider trait + OpenAI + structured output
│   │   └── tests/           # Eval suites + BEAM/LOCOMO/LongMem harnesses
│   └── karta-cli/           # CLI (planned)
├── data/                    # Benchmark preprocessing scripts
├── Cargo.toml
└── README.md
```

## Development

```bash
# Run tests (mock LLM, no API keys needed)
cargo test

# Run real eval (requires .env credentials)
cargo test --test real_eval -- --ignored --nocapture

# Run BEAM benchmark
BEAM_DATASET_PATH=data/beam-100k.json cargo test --test beam_100k beam_100k_single -- --ignored --nocapture
```

## Documentation

- **[`benchmarks/`](./benchmarks/)** — benchmark results, reproduction commands, experiment logs
- **[`docs/landscape.md`](./docs/landscape.md)** — AI memory systems landscape research
- **[`docs/retrieval-plan.md`](./docs/retrieval-plan.md)** — open retrieval experiment backlog
- **[`CONTRIBUTING.md`](./CONTRIBUTING.md)** — how to contribute
- **[`CHANGELOG.md`](./CHANGELOG.md)** — release notes

## License

MIT
