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
- **First-class contradictions** — structured `Contradiction` objects with lifecycle (open → resolved/ignored), per-entity and per-scope queries, source-note protection from forgetting
- **Structured output with reasoning** — forces chain-of-thought before answers, enabling reliable abstention and contradiction flagging
- **Cross-encoder reranking** — Jina AI reranker for precise relevance scoring and intelligent abstention
- **Temporal awareness** — exponential decay scoring, foresight signals with validity windows
- **Provenance tracking** — every note tagged as FACT or INFERRED with confidence scores
- **Forgetting engine** — `Karta::run_forgetting()`/`Karta::preview_forgetting()`: archives stale low-activation notes using access-based exponential decay scoring with protected notes (profiles, episodes); lifecycle: Active → Deprecated → Superseded → Archived
- **Procedural memory** — `RuleEngine` with safe `ProceduralRule` DSL (query/session/contradiction conditions → prompt/retrieval actions only), fire-count tracking, note-sourced rules protected from forgetting
- **Evidence packets** — `AskResult` exposes an optional `EvidencePacket` slot for per-channel rank traces, fired rule IDs, contradiction IDs, and human-readable "why retrieved" explanations; current read paths return `evidence: None` until ACTIVATE populates it
- **Deterministic extractors** — `Extractor` trait with Markdown (headings, links, code fences), JSON (recursive paths), YAML (key-value), and Cargo.toml (metadata, dependency edges) extractors that run before LLM extraction
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

    // Query with synthesis
    let result = karta.ask("What should I know about Sarah's notification preferences?", 5).await.unwrap();
    println!("{}", result.answer);

    // Run background reasoning
    let dream_run = karta.run_dreaming("workspace", "default").await.unwrap();
    println!("Dreams: {} attempted, {} written", dream_run.dreams_attempted, dream_run.dreams_written);
}
```

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

```toml
# Per-operation model flexibility
[llm.default]
provider = "openai"
model = "gpt-4o-mini"

[llm.dream.abduction]
model = "claude-sonnet-4-6"

# Reranker
[reranker]
enabled = true
abstention_threshold = 0.1
```

## Benchmarks

**BEAM 100K** — 20 conversations, 400 questions, single run, arithmetic mean
of per-question rubric scores with the BEAM nugget LLM judge:

| Ability | P1 (2026-04-14) |
|---|---|
| Preference Following | 92% |
| Contradiction Resolution | 74% |
| Temporal Reasoning | 71% |
| Instruction Following | 69% |
| Summarization | 66% |
| Multi-session Reasoning | 65% |
| Abstention | 62% |
| Information Extraction | 59% |
| Knowledge Update | 43% |
| Event Ordering | 35% |
| **Overall** | **61.6%** |

Up from a 53.0% P0 baseline (+8.6pp). P1 fixes: expand-then-rerank,
killing the synthesis-level abstention gate, and recency scoring against
source note timestamps instead of dream write time.

Active development targeting 90%+ via atomic fact decomposition,
episode-aware retrieval, and improved write-time organization. See
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
│   └── karta-cli/           # CLI for local automation and agent integrations
├── data/                    # Benchmark preprocessing scripts
├── Cargo.toml
└── README.md
```

## Development

```bash
# Use the CLI (requires LLM credentials for commands that write/search/ask)
cargo run -p karta-cli -- --help
cargo run -p karta-cli -- --json add-note --content "Sarah prefers Slack notifications"
cargo run -p karta-cli -- --json search --query "Sarah notification preferences" --top-k 5

# Run tests (mock LLM, no API keys needed)
cargo test

# Run synthetic memory evals (zero API keys, deterministic)
cargo test -p karta-core --test synthetic_memory_eval

# Run real eval (requires .env credentials)
cargo test -p karta-core --features eval-harnesses --test real_eval -- --ignored --nocapture

# Run BEAM benchmark
BEAM_DATASET_PATH=data/beam-100k.json cargo test -p karta-core --features eval-harnesses --test beam_100k beam_100k_single -- --ignored --nocapture
```

CI gates on every PR: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo nextest run --workspace --no-fail-fast`.

## Pi Integration

This repository includes a project-local pi extension at `.pi/extensions/karta.ts`.
When you run pi from the repository root, the extension registers Karta memory tools
backed by the local CLI:

- `karta_add_note` — store durable memories
- `karta_search` — semantic memory search
- `karta_ask` — synthesized Q&A over memory
- `karta_get_note` — retrieve a note by ID
- `karta_note_count` — count stored notes
- `karta_health` — check embedded store health
- `karta_dream` — run background reasoning

By default the extension shells out to `cargo run -q -p karta-cli -- --json ...`,
so it works directly from a checkout. To use an installed binary instead:

```bash
cargo install --path crates/karta-cli
export KARTA_BIN=karta
pi
```

Useful environment variables:

```bash
export KARTA_DATA_DIR=.karta          # storage directory used by the CLI
export KARTA_LANCE_URI=.karta/lance   # optional LanceDB URI override
export KARTA_TIMEOUT_MS=120000        # extension command timeout
```

The pi extension also performs automatic memory recall before each agent turn.
It builds a search query from the user prompt, cwd, and exact-match tokens such
as paths, symbols, constants, issue IDs, and code spans, then injects a compact,
provenance-rich memory message. Configure it with:

```bash
export KARTA_AUTO_CONTEXT=0             # disable automatic recall
export KARTA_AUTO_CONTEXT_TOP_K=5       # retrieved memories, 1-20
export KARTA_AUTO_CONTEXT_MAX_CHARS=4000 # injected context budget
export KARTA_AUTO_CONTEXT_DISPLAY=1     # show injected memory messages in the TUI
```

Memory policy instructions for pi live in `.pi/APPEND_SYSTEM.md`. Automatic
recall is intentionally read-only; writes still happen through explicit memory
tools so durable memory promotion remains conservative.

## Documentation

- **[`benchmarks/`](./benchmarks/)** — benchmark results, reproduction commands, experiment logs
- **[`docs/landscape.md`](./docs/landscape.md)** — AI memory systems landscape research
- **[`docs/retrieval-plan.md`](./docs/retrieval-plan.md)** — open retrieval experiment backlog
- **[`CONTRIBUTING.md`](./CONTRIBUTING.md)** — how to contribute
- **[`CHANGELOG.md`](./CHANGELOG.md)** — release notes

## License

MIT
