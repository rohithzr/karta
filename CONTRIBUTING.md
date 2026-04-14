# Contributing to Karta

Karta is an experimental research project on AI agent memory. Contributions
are welcome — bug reports, benchmark runs on your own hardware, new
benchmarks, failure-case analyses, and code changes all count.

## Before you start

- **Read [`benchmarks/beam-100k.md`](./benchmarks/beam-100k.md)** if you're
  touching retrieval, dreaming, or the read path. It's the single source
  of truth for what's been tried, what worked, and why.
- **Read [`docs/retrieval-plan.md`](./docs/retrieval-plan.md)** for the
  open experiment backlog. If your idea is already on the list, great —
  coordinate on the relevant section.
- **Open an issue before a large PR.** For anything beyond a bug fix or
  small doc patch, open an issue first so we can agree on the approach.
  PRs that duplicate existing work or conflict with in-flight changes
  are the most painful to resolve.

## Development setup

```bash
# Prerequisites: Rust stable (1.85+), git, a real LLM API key for eval runs
rustup install stable
git clone https://github.com/rohithzr/karta
cd karta

# Build + cheap checks
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check

# Unit tests (mock LLM, no API keys needed)
cargo test

# Real eval (requires .env with OPENAI_API_KEY or AZURE_OPENAI_*)
cp .env.example .env
# Fill in credentials
cargo test --test real_eval -- --ignored --nocapture
```

A `bacon.toml` and `rust-toolchain.toml` are committed for a reproducible
dev loop. If you have [`bacon`](https://github.com/Canop/bacon) installed,
`bacon` in the project root runs `cargo check --all-targets` on every
save.

## Code style

- **Rust 2024 edition.** `cargo fmt` + `cargo clippy -- -D warnings` must
  pass before you push. CI will enforce this.
- **No mocked LLM calls in eval code.** All benchmarks, evals, and quality
  measurements must use a real LLM. The only mocked LLM test in the
  codebase is for build-sanity (`crates/karta-core/src/llm/mock.rs`).
- **Write-time organization beats read-time cleverness.** Changes to the
  write path (attribute extraction, linking, evolution) tend to matter
  more than changes to the read path (reranker tuning, retry logic).
  Budget your effort accordingly.
- **No bulk reformat / clippy-fix / refactor mixed with feature work.**
  Keep those PRs separate so the review signal stays clean.
- **Don't add comments that explain *what* the code does** — let
  identifiers do that. Comments should explain *why* something is
  non-obvious, or reference a hidden constraint.

## Commit messages

- Imperative mood, under 72 characters on the first line.
- Explain the *why*, not the *what* — the diff already shows what
  changed.
- One logical change per commit. Don't mix refactors with features or
  fixes.
- Include a BEAM / benchmark number in the body if the change moves the
  needle (e.g., `BEAM 100K: 61.6% (+0.8pp)`).

## Pull request checklist

Before marking a PR ready for review:

- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes
- [ ] `cargo test` passes
- [ ] If the change touches retrieval, dreaming, or scoring: include a
      short note on benchmark impact (run a single-conversation BEAM
      probe if you can)
- [ ] Docs updated if behavior, config, or APIs changed
- [ ] No benchmark numbers reported from incomplete / panicked runs
      (see the "Validate bench runs" standing rule in
      [`benchmarks/README.md`](./benchmarks/README.md))

Reviewers apply the checklist in
[`.github/EXTERNAL_PR_REVIEW.md`](./.github/EXTERNAL_PR_REVIEW.md) — if
you want a faster turnaround, pre-audit your PR against that file.

## Reporting bugs

Use GitHub Issues. Useful bug reports include:

- The exact command you ran
- The LLM provider + model
- A minimal repro if possible
- The relevant log snippet (redact anything sensitive)

For retrieval-quality bugs, a failing BEAM question with the debug JSONL
output is worth a thousand words.

## Code of Conduct

Participation in this project is governed by
[`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md). Be kind. Disagreements about
technical approaches are fine and expected; disagreements about whether
people deserve respect are not.
