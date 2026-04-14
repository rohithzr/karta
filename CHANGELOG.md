# Changelog

All notable changes to Karta will land here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely and
semantic versioning once we leave the experimental phase.

## [Unreleased]

### Added
- `rust-toolchain.toml` and `bacon.toml` for reproducible dev loop
- `benchmarks/` folder with BEAM 100K results, per-ability history, and
  reproduction commands
- `docs/landscape.md` — research survey of the AI memory space
- `docs/retrieval-plan.md` — open retrieval experiment backlog
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`

### Changed
- **BEAM 100K: 61.6%** (P1 retrieval fixes, 2026-04-14). Up from 53.0%
  P0 baseline (+8.6pp). Biggest wins: temporal_reasoning (+18pp),
  preference_following (+8pp), contradiction_resolution (+7pp).
- Retrieval path: expand-then-rerank, removed synthesis-level
  insufficient-info gate that was killing temporal/computation queries
- Dream engine: score candidates on source note timestamps instead of
  dream write time (accurate recency)
- README reframed to emphasize experimental research project status

### Fixed
- Read path prompt handling for recency-sensitive queries
- SQLite schema + trait adjustments for source timestamp propagation

## [0.1.0-experimental] — 2026-04-14

Initial public release of Karta as an experimental research project.
Core engine in Rust (`karta-core`), LanceDB + SQLite storage, dream
engine with 5 typed inference modes, cross-encoder reranking, structured
output with reasoning, and BEAM 100K benchmark harness.
