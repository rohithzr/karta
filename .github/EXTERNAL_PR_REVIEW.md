# External PR Review Standard

> Applied to every PR from contributors outside the core team.
> "Experimental" status does not lower the bar — it raises it. Unstable code with latent bugs becomes load-bearing faster than stable code.

---

## Review Checklist

Work through every section. Every BLOCKER and MUST-FIX must be resolved before merge. SHOULD-FIX items require explicit documented acceptance if deferred.

---

### 0. Context Check (before reading code)

- [ ] Does the PR touch an area that's in active development? (Check CLAUDE.md phase status.) If yes, coordination required.
- [ ] Is the PR implementing Phase 5+ work while core phases are incomplete? Flag it.
- [ ] Who is the author? Individual human, agent-assisted, or fully AI-generated? All are acceptable — but AI-generated PRs get stricter scrutiny because agents don't run their own code end-to-end.
- [ ] Does the PR description match the actual diff? Check every file changed.

---

### 1. Compilation

- [ ] Does `cargo check --all-targets --all-features` pass clean (zero warnings on new code)?
- [ ] Does `cargo check --no-default-features` pass?
- [ ] Are all imports used? All match arms exhaustive?
- [ ] Are new dependencies justified? Do they duplicate existing crates?
- [ ] Are new feature flags correctly gated — consistent with how existing features are gated in `llm/mod.rs`, `store/mod.rs`, etc.?

---

### 2. Correctness Against karta-core API

- [ ] Verify every public method called against `karta.rs` (grep `pub async fn`, `pub fn`).
- [ ] Verify every struct field accessed against the actual struct definition in `note.rs`, `dream/types.rs`, `read.rs`.
- [ ] Verify every enum variant used is real (don't trust the PR description — check the source).
- [ ] Verify every trait method signature matches `LlmProvider`, `VectorStore`, `GraphStore` in `store/mod.rs` and `llm/traits.rs`.
- [ ] If new trait impls are added: does the impl cover every required method? Are lifetimes/generics correct?

---

### 3. Concurrency and Async

- [ ] Is any blocking I/O (sync file read, `stdin.lock().lines()`, `std::thread::sleep`) used inside an `async fn` or `#[tokio::main]`? This blocks the Tokio worker thread.
- [ ] Are long-running operations (LLM calls, dream engine, full graph scans) wrapped in `tokio::time::timeout`?
- [ ] Is state shared across `.await` points? Does it implement `Send + Sync`? Are `Arc<Mutex<>>` locks held across `.await`?
- [ ] Are spawned tasks guaranteed to complete or are they fire-and-forget? Document the intent.

---

### 4. Error Handling

- [ ] No `unwrap()` or `expect()` on user input, external data, or network responses. Every `unwrap()` must be justified with a comment proving it can't fail.
- [ ] Errors must propagate or be explicitly handled — no silent `let _ = ...` on results that can fail meaningfully.
- [ ] Does an error in one operation leave shared state (DB, graph, vector store) in a consistent state?
- [ ] Are error messages useful for debugging? Do they include context (what was being attempted, what was received)?

---

### 5. Security and Trust Boundaries

- [ ] Does any input from an external caller (MCP tool args, HTTP body, CLI args, env vars) reach a file path, shell command, SQL query, or format string without validation?
- [ ] Are API keys, secrets, or tokens logged at any level?
- [ ] Are file paths constructed from user input? Check for path traversal (`../`, absolute paths).
- [ ] Does the new code expose internal implementation details (cursor IDs, internal state names, enum variant names as strings) in a public API? These become a compatibility surface.
- [ ] If the PR adds a network server: are there resource limits (max request size, max concurrent connections, per-request timeouts)?

---

### 6. Architecture Fit

- [ ] Does the PR respect the embedded-first principle? New dependencies must not require Docker, external services, or network access at build time.
- [ ] Does the PR add config that belongs in `KartaConfig` but was hardcoded instead?
- [ ] Does the PR add a heuristic that routes infrastructure decisions based on user-controlled strings (model names, deployment names)? These break silently in production. Use explicit config flags.
- [ ] Does the PR add new public API surface (new `pub` items, new tool schemas, new env vars)? Is this intentional? Does it match the documented roadmap?
- [ ] Does the PR rename or repurpose an existing crate or module? This must be explicitly discussed — don't change `karta-cli` from CLI to MCP-server-only without a plan for the CLI.

---

### 7. Parity with Existing Code

- [ ] If the PR adds a new implementation of a trait (e.g., a new `LlmProvider`): does it cover all fields in `GenConfig` (`temperature`, `max_tokens`, `json_mode`, `json_schema`)? Missing fields cause silent behavioral divergence.
- [ ] If the PR adds retry logic: does it match the retry patterns in `openai.rs` (`is_retryable`, backoff, log fields)?
- [ ] If the PR parses an LLM response: what happens on empty string, malformed JSON, content filter, refusal, or unexpected response shape? All must return `Err(...)`, not silently succeed with empty data.
- [ ] Does the PR add duplicated code that belongs in a shared helper? (Retry loops, error formatting, JSON extraction.)

---

### 8. Protocol Compliance (for MCP/API servers)

- [ ] Is the protocol version explicitly validated on `initialize`?
- [ ] Is there a state machine enforcing the correct message ordering (e.g., `initialized` before `tools/call`)?
- [ ] Are all notification types (messages with no `id`) handled — at minimum logged, at most driving state transitions?
- [ ] Are all tool schema fields (`type`, `description`, `inputSchema`, `required`) correct and complete?
- [ ] Are internal implementation details (cursor state, scope IDs) exposed as tool parameters? They shouldn't be.
- [ ] Are resource limits enforced on all tool input parameters (`top_k`, text length, array size)?

---

### 9. Tests

- [ ] Are there tests? For a PR of this size (>200 lines), at minimum:
  - Unit tests for any non-trivial data transformation
  - Deserialization/serialization round-trips for new data types
  - At least one error path test
- [ ] If no tests: explicitly document why (e.g., "requires live LLM, covered by real_eval.rs") and what the test plan is. Checked checkboxes in the PR description are not a test plan.
- [ ] Do existing tests still pass? (`cargo test --test eval` for mock sanity; real eval suite separately.)

---

### 10. Benchmark Safety

- [ ] Does the PR add any code path that could access reference answers, expected outputs, or benchmark ground truth during retrieval or synthesis? This is an immediate disqualifier (Design Principle #11).
- [ ] Does the PR change the write path, read path, or dream engine in a way that could affect BEAM scores? If yes, a benchmark run is required before merge.

---

## Severity Definitions

| Level | Definition | Merge policy |
|---|---|---|
| **BLOCKER** | Incorrect behavior, data loss, security issue, or protocol violation that will manifest in normal use | Must fix before merge, no exceptions |
| **MUST-FIX** | Silent failure, silent behavioral divergence, crash path, resource exhaustion, or missing safety bound | Must fix before merge |
| **SHOULD-FIX** | Inconsistency, poor debuggability, missing defensive check, or code that will cause operational pain | Fix or explicitly accept with documented rationale |
| **NIT** | Style, naming, unnecessary allocation, missing comment | Author's discretion; don't block merge |

---

## What "Correct" Looks Like

A PR is ready to merge when:
1. `cargo check --all-targets --all-features` is warning-free on new code
2. All BLOCKERs and MUST-FIXes are resolved
3. SHOULD-FIX items are either resolved or explicitly accepted with a comment
4. The PR description accurately reflects the diff
5. At least a token set of tests exists, or a documented reason why testing requires live infrastructure
6. No benchmark safety violations

---

## Notes on AI-Generated PRs

AI agents (OpenClaw, Codex, Cursor, etc.) frequently:
- Hallucinate method names — **always verify against actual source**
- Hard-code the author's machine paths into default values
- Copy-paste retry/error logic without adapting it
- Get protocol state machines wrong (MCP `initialized`, OAuth flows, etc.)
- Omit temperature and other GenConfig fields from LLM provider implementations
- Use model name strings as routing heuristics instead of explicit config
- Mark test plans as complete with checkboxes when no test code exists

These are not reasons to reject — they're a checklist of where to look first.
