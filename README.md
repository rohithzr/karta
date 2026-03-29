# Karta

An agentic memory system that thinks, not just stores.

Karta combines structured knowledge graphs with active background reasoning to build memory that organizes itself at write time, links associatively, evolves retroactively, and periodically "dreams" to surface deductions, patterns, gaps, and contradictions that no retrieval system could find because they were never explicitly stored.

## Features

- **Zettelkasten-inspired knowledge graph** — notes are enriched with LLM-generated context, semantically linked, and retroactively evolved when new information arrives
- **Dream engine** — 5 types of background inference: deduction, induction, abduction, consolidation, contradiction detection
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

    // Query with synthesis
    let answer = karta.ask("What should I know about Sarah's notification preferences?", 5).await.unwrap();
    println!("{}", answer);

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

BEAM 100K (20 conversations, 400 questions, official judge prompt):

| Ability | Score |
|---------|-------|
| Preference Following | 80% |
| Contradiction Resolution | 69% |
| Multi-session Reasoning | 69% |
| Summarization | 64% |
| Information Extraction | 63% |
| Knowledge Update | 62% |
| Temporal Reasoning | 61% |
| Instruction Following | 56% |
| Abstention | 52% |
| Event Ordering | 23% |
| **Overall** | **56.8%** |

> Day 2 baseline. Active development — episode-based retrieval, confidence propagation, and dream-of-dreams are next.

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

## License

MIT
