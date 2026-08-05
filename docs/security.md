# Security notes

Results of a security review (2026-08) of `karta-core` and `karta-cli`,
focused on PII and sensitive-data exfiltration for anyone integrating Karta
into agentic workflows. The review covered all library source, the test
suite, the vendored `sqlite-vec` copy, `Cargo.lock`, scripts, and CI.

**Bottom line:** no malicious code, no telemetry, no hidden egress. By
default the only network destination is `api.openai.com`. The findings
below are configuration and hygiene issues to know about before deploying.

## Network egress surface

| Destination | When active | Data sent |
|---|---|---|
| `api.openai.com` (chat + embeddings) | **default** | Conversation text, memory contents, queries, reference timestamp. `Authorization: Bearer OPENAI_API_KEY` |
| `AZURE_OPENAI_ENDPOINT` | Azure env vars set | Same payload classes, via `api-key` header |
| `api.jina.ai/v1/rerank` (hardcoded) | `reranker.enabled = true` **and** `JINA_API_KEY` present in env | Query + up to 20 note contents (500 chars each) |
| `OPENAI_API_BASE` / configured `base_url` | env/config override | Same as OpenAI — **with the configured API key attached** |
| `KARTA_ANSWER_BASE_URL` | answer-model env vars set | Synthesis prompts (query + retrieved memories) |

The only hardcoded third-party URL in the codebase is the Jina endpoint.
Prompts contain only user content plus a reference timestamp — no env
values, hostnames, or file contents are silently included. Environment
reads are limited to the documented `OPENAI_*` / `AZURE_*` / `KARTA_*` /
`JINA_API_KEY` variables; there is no reading of `~/.ssh`, `~/.aws`, or
similar paths anywhere in the library or test suite.

## Known findings

| Severity | Finding | Location |
|---|---|---|
| High | **Jina reranker selected by env-var presence.** When `reranker.enabled = true`, a `JINA_API_KEY` present in the environment silently selects `JinaReranker` over the local `LlmReranker`, sending every query plus up to 20 memory excerpts to `api.jina.ai` — a third party distinct from the configured LLM. | `karta-core/src/karta.rs:62`, `karta-core/src/rerank.rs:246` |
| Medium | **`OPENAI_API_KEY` forwarded to arbitrary base URLs.** Setting `OPENAI_API_BASE` (e.g. to a local Ollama/vLLM) attaches the configured API key as a Bearer credential to that endpoint, with no scheme validation — cleartext if the URL is `http://`. A LAN model server would receive (and typically log) the key. | `karta-core/src/llm/openai.rs:60-72`, `karta-core/src/karta.rs:154,169` |
| Medium | **PII in `info`-level logs.** Ingest logs the first 60 characters of every note; search logs every full query verbatim. These fire at typical `RUST_LOG=info` and reach whatever tracing sink the host application configures. | `karta-core/src/write.rs:113`, `karta-core/src/read/mod.rs:707` |
| Medium | **Memory DB created with default permissions.** The SQLite store holding all memories and embeddings is created with default umask permissions (typically world-readable). On shared/multi-user hosts, other local users can read the entire memory store, including `-wal`/`-shm` files. | `karta-core/src/store/sqlite_vec.rs:32`, `karta-core/src/store/sqlite.rs:22` |
| Low | **PII in `debug`-level logs.** Generated context, foresight signals, and verbatim grounding spans are emitted at `debug!`. Opt-in, but commonly enabled while troubleshooting. | `karta-core/src/write.rs` (multiple) |
| Low | **Heavy tracing writes full LLM traffic to a plain JSONL file.** When a consumer enables heavy tracing, prompts, completions, and fact contents land unencrypted at a caller-supplied path with default permissions. Off by default in the library; the `beam_conv0_trace` test defaults `BEAM_TRACE_HEAVY=1` and writes under `.results/`. | `karta-core/src/trace.rs:119`, `karta-core/src/llm/tracing_wrapper.rs` |
| Low | **Retry warnings log upstream error strings.** OpenAI sanitizes its errors, but arbitrary OpenAI-compatible backends may echo the offending input into error bodies, landing memory content in logs. | `karta-core/src/llm/openai.rs:213` |

## Integration checklist

- Keep `reranker.enabled = false` (the default), or make sure `JINA_API_KEY`
  is not set in the process environment unless you intend memory content to
  go to Jina.
- When overriding `OPENAI_API_BASE`, set `OPENAI_API_KEY` to a throwaway
  value (e.g. `ollama`) rather than a real key.
- Set `RUST_LOG=karta_core=warn` in any host application that ships logs
  off-box.
- `chmod 700` the data directory (default `.karta/`) on shared hosts.
- Avoid heavy tracing in production; scrub `.results/*.jsonl` after
  benchmark runs.

## Supply chain verification

- The vendored `sqlite-vec` (used via `[patch.crates-io]`) was diffed
  against upstream `asg017/sqlite-vec v0.1.10-alpha.3`: all C sources are
  **byte-identical**. Its `build.rs` only compiles the C via `cc` — no
  network, env access, or codegen.
- `Cargo.lock`: all dependencies resolve to the crates.io registry — no git
  dependencies, no typosquat candidates, single versions of
  rustls/openssl/hyper/reqwest.
- Test suite is safe to run with secrets in the environment: networked
  tests are `#[ignore]`d or gated behind `KARTA_REAL_LLM_TESTS=1`, and send
  only public eval fixtures (BEAM / LOCOMO / LongMemEval) to the configured
  endpoint.
