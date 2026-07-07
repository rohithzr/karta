# AI Memory Systems Landscape

> Research survey of the AI memory space — compiled from papers, GitHub
> READMEs, and public benchmarks. Karta is an experimental research project
> pursuing its own approach; this document is background research, not
> positioning material. Benchmark numbers are self-published unless noted
> otherwise and should be treated as claims until independently reproduced.
>
> Last compiled: 2026-07-07.

## Legend

- Benchmark scores are self-published unless noted otherwise. Treat with skepticism until independently reproduced.
- "—" means not reported or not benchmarked.
- BEAM has multiple tiers (100K / 500K / 1M / 10M). The matrix reports whichever tier each system publishes; Honcho's headline is the full 10M scale, most others report 100K.

## Competitive Matrix

| System | BEAM 100K | LOCOMO | LongMemEval-S | Memory Model | Write-Time Organization | Retrieval | Dreaming / Active Inference | Forgetting | Episode Segmentation | Foresight / Forward-Looking | Auditability | Language / Stack | Operational Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Karta** *(ours, experimental)* | 61.6% (P1, 2026-04-14, N=399) | — | 50.0% (temporal-reasoning only, P0 subset, N=28; not comparable) | Hypergraph: atomic facts with typed role edges, versioned snapshots | Decompose → entity/relation extraction → timestamped graph writes | Multi-hop graph traversal + embedding similarity | Yes — planned but not yet in eval | Manual delete only; no auto-decay | No | No | High — hypergraph queryable | Rust; embedded or remote | Low — self-hosted only |
| **Honcho** | 63.0% (100K) / 40.9% (10M, evals.honcho.dev) | 89.9% *(LLM-judge)* | 90.4% *(LongMem-S, evals.honcho.dev, May 2026)* | Dialectic user model: structured inferences about the user, not raw facts | Dialectic inference via LLM on conversation turns | BM25 sparse + dense semantic; context-window insertion | Yes — background dialectic process refines user model | No — inferences accumulate | No | No | Medium — stored inferences inspectable | Python; Postgres + pgvector; API + self-host | Medium — LLM inference for model updates |
| **EverOS** *(formerly EverMemOS; EverMind-AI)* | — | 93.05% *(self-published, Feb 2026 product release; methodology not detailed; HaluMem 90.04%)* | 83.00% *(self-published, v1.0.0, Feb 2026)* | Modular memory types: episodic, semantic, user profile, task context, working memory, skills | Classifies and routes memory types; Skills Evolution Engine (234.8% relative task success gain); structured extraction | Semantic similarity + type-filtered retrieval; HyperMem fast recall | Yes — Skills Evolution Engine: background skill extraction and improvement | Decay scoring | Yes — episode boundaries | No | Medium — structured types inspectable | Python; SQLite + LanceDB + Markdown; local-first; v1.1.0 (Jun 24 2026) | Low — local-first; no mandatory cloud |
| **MemU** | — | 92.09% *(self-published)* | — | File-system hierarchy (folders = topics, files = facts, lines = granular facts) | LLM-directed file operations: create/write/append/delete | Semantic folder lookup → file scan → line extraction | Yes — scheduled background passes | Yes — explicit LLM delete calls | Implicit — topic folder boundaries | No | High — human-readable file hierarchy | Python; any file system | Medium — LLM at write time; background passes |
| **Zep / Graphiti** | — | 75% *(disputed config)* | — | Temporal knowledge graph; episodic time-indexed edges + entity nodes | LLM graph extraction per turn; temporal edge versioning | Graph traversal + embedding recall; BFS community summaries | No | Temporal edge aging (configurable) | Yes — session episodes | No | Medium — graph queryable | Python/Go; Neo4j or in-memory; hosted + self-host | High — graph extraction LLM call per turn |
| **A-MEM** | — | ~60% *(NeurIPS 2025 paper)* | — | Zettelkasten linked notes with context + keywords + timestamp + links | Note creation with Zettelkasten linking; adaptive note evolution on retrieval | Embedding similarity + keyword overlap; link traversal | Yes — "memory evolution" on recall | No | No | No | Medium — notes inspectable | Python; ChromaDB | Medium — LLM at write + evolve |
| **Mem0** | — | v3: 92.5% *(self-published, mem0.ai/research, Apr 2026; single-pass extraction + multi-level hierarchy + contradiction resolution)* | v3: 94.4% *(self-published, Apr 2026)* / prior 91.4% | Structured fact graph with contradiction detection; v3: multi-level hierarchy + cross-session scoring | LLM extraction → entity linking → deduplication; v3 adds multi-level rollup | Graph + vector hybrid; entity-centric retrieval | No | Explicit delete; contradiction overwrite | No | No | Medium — structured facts queryable | Python + TypeScript; Qdrant/pgvector/others; hosted + self-host | Medium — LLM extraction + graph update per turn |
| **Letta / MemGPT** | — | 74.0% *(Letta Filesystem; gpt-4o-mini; stores history as files + semantic search; EMNLP 2025)* | — | In-context core memory (limited) + archival external memory + conversation history | Append to archival; in-context edit via tool calls | Semantic search over archival; recency-ordered history | No | Manual only | No | No | Medium — in-context memory inspectable | Python; SQLite/Postgres; hosted + self-host | Medium — LLM controls memory via tool calls |
| **Hindsight** *(Vectorize)* | **73.4%** (Apr 2 2026 blog; v0.4.19; #1 on AMB leaderboard; was cited as rounded 75% earlier) | **92%** *(LoComo10, #1 on AMB)* | **94.6%** *(LongMemEvalS, #1 on AMB)* | Structured semantic memory bank: entities, relationships, preferences, events, summaries | Reflector background agent: summarize → categorize → update bank | Graph + embedding + recency; context window injection | Yes — Reflector actively reasons over memory; mental model history (v0.8.0) | Semantic dedup (v0.8.0); configurable recency decay (v0.8.4); reversible curation (v0.8.2) | Implicit — Reflector segments by topic shift | No | High — memory bank inspectable, exportable | Python + TypeScript; pluggable backends; hosted + self-host | Low-medium — background Reflector amortizes cost |
| **MemoryOS** *(BAI-LAB)* | — | +49.11% F1 / +46.18% BLEU-1 improvement vs baseline *(EMNLP 2025 Oral; absolute scores not reported vs competitors)* | — | OS-inspired hierarchical storage: short-term (recent), mid-term (frequent), long-term (important); heat-based promotion | Stores all conversation turns; heat-score governs tier promotion | Heat-weighted retrieval across tiers; semantic similarity | No | Heat-based expiry from short-term | Yes — session-level episode separation | No | Medium — tiered storage inspectable | Python; SQLite + vector; open-source | Low — heat scoring is cheap; LLM for long-term summaries |
| **Mastra (Observational Memory)** | — | — | 84.23% (gpt-4o) / 94.87% (gpt-5-mini) — **highest LongMemEval ever at time of publication (Apr 2026)** | Stable append-only context with Observer + Reflector compressing in background | Observer monitors, Reflector compresses: 5–40x reduction | Compressed context window injection | Yes — Reflector actively reasons over observed context | Yes — context compression | Yes — Reflector detects logical boundaries | No | Medium — compressed context inspectable | TypeScript; any LLM/store; framework-integrated | Medium — background LLM compression |
| **Supermemory** | — | — | 85.4% *(production engine)* / ~99% *(experimental ASMR, not production)* | Vector embeddings of user data; ASMR (experimental): parallel LLM agent swarm | Hosted ingestion; ASMR: multi-agent parallel extraction | Hosted vector retrieval; ASMR: swarm synthesis | No | No | No | No | Low — hosted opaque | TypeScript; hosted | Low for production; high for ASMR |
| **OMEGA** | — | — | **95.4%** *(466/500; #1 on LongMemEval leaderboard, local-first, pip install)* | Not documented publicly | Not documented | Not documented | Not documented | Not documented | Not documented | Not documented | Unknown — local-first | Python; pip; local | Very low — no cloud; pip install |
| **MemMachine** | — | **91.69%** *(gpt-4.1-mini; arxiv 2604.04853, Apr 2026; above Mem0, Zep, Memoba [sic])* | **93.0%** *(gpt-4.1-mini, self-reported)* | Ground-truth-preserving: retrieval-stage optimization, not ingestion-stage modification | Raw storage; no write-time transformation | Retrieval-stage LLM reasoning over raw memories; ~80% fewer tokens than Mem0 | No | No | No | No | High — raw storage unchanged | Python; open-source | Low — no write-time LLM; retrieval-stage LLM |
| **Memori** *(MemoriLabs)* | — | 81.95% *(at 4.98% cost of full context; outperforms Zep, LangMem, Mem0)* | — | SQL-native structured memory: facts, preferences, rules, summaries; intelligent decay + ranking | Classifies each chat turn into facts/preferences/rules/summaries; continuous extraction | SQL-native retrieval with intelligent ranking and decay; avg 1,294 tokens per query | None | Yes — intelligent decay and ranking built-in | None | Not documented | Medium — SQL-native, inspectable structured state | Python; SQL-native, LLM/datastore-agnostic; Cloud + BYODB + on-prem | Low-medium — Memori Cloud hosted; very token-efficient |
| **MemPalace** | **49.0%** *(independently tested, raw mode + GPT-5.4-mini synthesis)* | — | **96.6%** R@5 *(self-reported, raw mode, retrieval recall only; 100% with rerank is teaching-to-test on 3 specific questions, held-out: 98.4%)* | Verbatim text in ChromaDB, no write-time reasoning | None, raw storage only | ChromaDB cosine search; hybrid keyword boost + temporal boost in advanced modes | None | None | None | None | Low, inspectable ChromaDB store | Python; ChromaDB only; zero infra | Very low, no LLM for storage |
| **ByteRover** | — | 96.1% *(v2.1.5, self-published Mar 2026)* | 92.8% *(v2.1.5, self-published Mar 2026)* | Context Tree — hierarchical knowledge organized by domain/topic/subtopic; portable across tools | LLM-directed tree writes: create/update nodes; domain/topic tagging | Tree traversal + semantic search; context-window injection | No | No | Yes — topic-level segmentation | No | Medium — Context Tree inspectable | TypeScript; local-first; MCP; Elastic License 2.0 | Low — local-first; no telemetry by default |
| **MemOS** *(MemTensor)* | — | 92.34% *(self-published, v2.0.22, Jul 2026)* | 93.40% *(self-published, user memory, Jul 2026)* | Self-evolving memory OS; L1 traces + L2 policies + L3 world models + Skills; hybrid FTS5+vector; multi-modal (images, charts) | Stores events as traces; auto-promotes through memory tiers; knowledge base from docs/URLs | Hybrid FTS5+vector search; search pipeline hooks for context rendering | Yes — Reflect2Evolve architecture; mid-turn reasoning capture | Yes — intelligent tier pruning and consolidation | Yes — L1 trace segmentation | Not documented | Medium — structured memory tiers inspectable | Python; OpenClaw Cloud + Local plugin v2.0; self-hosted or cloud | Medium — Cloud or self-hosted; local plugin zero-infra |

## New Benchmarks and Papers (2026)

| Benchmark / Paper | What It Tests | Key Scores Found | Source |
|---|---|---|
| **HaluMem** *(MemTensor, 2025)* | First operation-level hallucination eval for agent memory systems; grades Store, Retrieve, and Update independently | EverMemOS 90.04%; Mem0 78%; Zep 72% | [github.com/MemTensor/HaluMem](https://github.com/MemTensor/HaluMem) |
| **Agent Memory Benchmark (AMB)** *(Vectorize, Mar 2026)* | Tests 8 architectures: Letta, Cognee, Graphiti, Tacnode, Mem0, Hindsight, EverMemOS, Hyperspell | Scores not yet scraped; live at agentmemory.fyi; also serves as Hindsight marketing | [agentmemory.fyi](https://agentmemory.fyi) |
| **Graph-Native Cognitive Memory** *(arxiv, Mar 2026)* | Formal belief revision semantics for versioned knowledge graphs; proposes two new benchmarks: KGMemBench (temporal reasoning) and KGMemEval (belief coherence) | No comparison scores; benchmark proposals only | [arxiv.org](https://arxiv.org) |
| **Memory-R1** *(arxiv 2603.26035, Mar 2026)* | RL-trained memory encoder; GRPO optimization against episodic QA | LoCoMo: Mistral-7B +17.5% with memory vs base; 56.56% (LoCoMo) | [arxiv 2603.26035](https://arxiv.org/abs/2603.26035) |
| **LongRewardBench** *(arxiv 2604.12406, Apr 2026)* | Evaluates reward models (not memory systems) on long-context preference pairs; tests whether RM can judge 32K+ context quality | Existing RMs struggle with long contexts; no memory-system scores | [arxiv 2604.12406](https://arxiv.org/abs/2604.12406) |
| **MemFactory** *(arxiv 2603.29493, Mar 2026)* | Unified modular training framework for memory-augmented LLMs; Lego-like components; GRPO fine-tuning with Memory-R1/RMM/MemAgent | No head-to-head comparison scores; training framework only | [arxiv 2603.29493](https://arxiv.org/abs/2603.29493) |
| **AgeMem** *(arxiv 2601.01885, Jan 2026)* | Adaptive memory system for long-term multi-session conversations; temporal decay + importance scoring + cross-session consolidation | LoCoMo +11.2% over baselines (LLM-judge F1); LOCRET +18.0% | [arxiv 2601.01885](https://arxiv.org/abs/2601.01885) |
| **A-MAC** *(arxiv 2603.04549, Mar 2026)* | Adaptive Memory-Augmented Context for LLM agents; context window management via sliding window + semantic compression + hierarchical memory | Reduces hallucination in long conversations; no LOCOMO/BEAM scores | [arxiv 2603.04549](https://arxiv.org/abs/2603.04549) |
| **LifeBench** *(arxiv 2603.03781, Mar 2026)* | Long-term personal memory for LLMs across 7 life domains; 500K-token personal histories, 3K QA pairs | GPT-4o baseline 60.7%; specialized systems not yet tested | [arxiv 2603.03781](https://arxiv.org/abs/2603.03781) |
| **MemMachine** *(arxiv 2604.04853, Apr 2026)* | Ground-truth-preserving memory system; retrieval-stage optimization > ingestion-stage; GPT-5-mini outperforms Mem0, Zep, MemOS | LoCoMo 91.69%; LongMemEvalS 93.0% (GPT-4.1-mini) | [arxiv 2604.04853](https://arxiv.org/abs/2604.04853) |
| **SuperLocalMemory V3.3** *(arxiv 2604.04514, Apr 2026)* | Biologically-inspired forgetting, cognitive quantization, multi-channel retrieval; zero-LLM offline consolidation | LoCoMo: improved over V3.2 by adaptive thresholding (exact scores not scraped) | [arxiv 2604.04514](https://arxiv.org/abs/2604.04514) |
| **BEAM (ICLR 2026)** | Official paper accepted at ICLR 2026; 100 conversations, 100K–10M tokens, 2000 probing questions; 10M scale where context stuffing is physically impossible | Hindsight SOTA at 73.4% (100K) and 64.1% (10M); Honcho 63.0% (100K) / 40.9% (10M) | [arxiv.org/abs/2503.24129](https://arxiv.org/abs/2503.24129) |
| **SimpleMem: Efficient Lifelong Memory for LLM Agents** *(arxiv 2601.02553, Jan 2026)* | 3-stage pipeline: semantic lossless compression → online synthesis → offline consolidation; zero LLM calls for retrieval | 30× fewer tokens than full context, outperforms full-context on LoCoMo | [arxiv 2601.02553](https://arxiv.org/abs/2601.02553) |
| **EvolveMem: Self-Evolving Memory Architecture via AutoResearch** *(arxiv 2605.13941, May 2026)* | Follow-up to SimpleMem; closed-loop LLM diagnosis of failure cases; auto-generates memory design improvements; multimodal (images, audio) | Improves over SimpleMem via self-evolution; outperforms Mem0, A-MEM on LoCoMo (exact scores in paper) | [arxiv 2605.13941](https://arxiv.org/abs/2605.13941) |
| **"From Storage to Experience": Survey on Evolution of LLM Agent Memory Mechanisms** *(arxiv 2605.06716, May 2026)* | Proposes 3-stage evolutionary framework (Storage→Retrieval→Experience); surveys 200+ papers; defines Experience-Level memory as highest tier | — (survey, no benchmark scores) | [arxiv 2605.06716](https://arxiv.org/abs/2605.06716) |
| **Mnemonic Sovereignty: Security of Long-Term Memory in LLM Agents** *(arxiv 2604.16548, Apr 2026)* | First security survey for LLM agent memory; lists 5 attack vectors: memory poisoning, exfiltration, deletion, replay, false injection | — (security taxonomy, no benchmark) | [arxiv 2604.16548](https://arxiv.org/abs/2604.16548) |
| **Eywa: Provenance-Grounded Long-Term Memory for AI Agents** *(arxiv 2605.30771, May 2026)* | Evidence-before-belief architecture; immutable source evidence stored first; canonical facts derived later; zero LLM calls at write time for evidence storage | Outperforms Mem0, Zep on LoCoMo and LongMemEval (exact scores in paper) | [arxiv 2605.30771](https://arxiv.org/abs/2605.30771) |
| **LongMemEval-V2** *(arxiv 2605.12493, May 2026)* | New benchmark for web-agent memory specifically; 451 manually curated questions across 5 ability dimensions; 115M-token multimodal trajectories | GPT-4o 31.7% on hardest subset (baseline); specialized memory systems not yet evaluated | [arxiv 2605.12493](https://arxiv.org/abs/2605.12493) |
| **Portable Agent Memory Protocol** *(arxiv 2605.11032, May 2026)* | Protocol for cryptographically-verified memory transfer across heterogeneous AI agents; defines interoperability standard | — (protocol design, no benchmark) | [arxiv 2605.11032](https://arxiv.org/abs/2605.11032) |
| **Are We Ready For An Agent-Native Memory System?** *(arxiv 2606.24775, Jun 2026)* | Comprehensive evaluation of 12 memory systems across 11 datasets, 5 benchmark workloads; 4-module decomposition: Management, Storage, Retrieval, Utilization; proposes a 4-property framework for "agent-native" memory design; highlights the absence of a complete solution | — (evaluation framework) | [arxiv 2606.24775](https://arxiv.org/abs/2606.24775) |
| **MemIR** *(arxiv 2605.25869, May 2026)* | Typed memory representation separating raw evidence, retrieval cues, and truth-bearing claims via multi-route atomic projection and provenance-scoped utilization; addresses provenance-role collapse failure mode in long-term agents | Outperforms existing approaches on LoCoMo and BEAM-100K, especially source tracking, temporal grounding, aggregation of fragmented evidence | [arxiv 2605.25869](https://arxiv.org/abs/2605.25869) |
| **RecMem** *(arxiv 2605.16045, May 2026; ACL 2026 Findings)* | Recurrence-based memory consolidation; defers LLM extraction until sustained recurrence observed across semantically similar interactions; subconscious embedding layer for cheap interim storage; semantic refinement recovers granular details | Reduces memory construction token cost of 3 SOTA systems by up to 87% while exceeding their accuracy | [arxiv 2605.16045](https://arxiv.org/abs/2605.16045) |
| **MemFail** *(arxiv 2605.26667, May 2026)* | Diagnostic benchmark stress-testing failure modes of LLM memory systems; isolates three core operations: summarization, storage, retrieval; 5 datasets across 4 tasks; tested 4 SOTA systems to reveal per-operation tradeoffs | — (diagnostic, not ranking) | [arxiv 2605.26667](https://arxiv.org/abs/2605.26667) |
| **MemPro** *(arxiv 2606.00619, May 30, 2026)* | Treats entire memory construction-retrieval pipeline as an evolvable program (version tree); Evolving Agent diagnoses recurring failures and generates improved implementations via failure-mode-guided refinement | Consistently outperforms static and prompt-level evolving baselines on LongMemEval, LoCoMo, HotpotQA, NarrativeQA | [arxiv 2606.00619](https://arxiv.org/abs/2606.00619) |
| **MIRIX** *(arxiv 2507.07957, Jul 2025)* | Multi-agent memory system with 6 types (Core, Episodic, Semantic, Procedural, Resource Memory, Knowledge Vault); multimodal (text + screenshots); real-time screen monitoring + local storage for privacy | LoCoMo: 85.4% SOTA; ScreenshotVQA: +35% vs RAG baseline at 99.9% storage reduction | [arxiv 2507.07957](https://arxiv.org/abs/2507.07957) |
| **T-Mem: Memory That Anticipates, Not Archives** *(arxiv 2606.15405, Jun 2026)* | Long-term conversational memory with write-time rehearsal triggers; dual trigger families — descriptive (surface-similarity) and associative (latent semantic arc) — ensuring every memory is reachable from both query types; addresses failure case where query and memory share no surface features | SOTA on LoCoMo and LoCoMo-Plus (self-reported; specific numbers not in abstract) | [arxiv 2606.15405](https://arxiv.org/abs/2606.15405) |
| **MemRefine: LLM-Guided Compression for Long-Term Agent Memory** *(arxiv 2606.13177, Jun 2026)* | Memory compression within fixed storage budgets; LLM judges which entries to delete, merge, or preserve; similarity matching surfaces candidate pairs; iterates until budget is met | Consistently meets target budgets while preserving or exceeding downstream accuracy vs rule-based baselines under tight budgets | [arxiv 2606.13177](https://arxiv.org/abs/2606.13177) |
| **Cross-Scenario Generality of Agentic Memory Systems + AutoMEM** *(arxiv 2606.04315, Jun 2026)* | Evaluates 8 memory systems across 5 diverse scenarios (single-turn QA, multi-session chat, agentic-trajectory QA, stress tests, long-horizon tasks); introduces AutoMEM — an agentic harness that lets agents self-manage storage/retrieval via tool calls | AutoMEM achieves best cross-scenario ranking; passive fixed-pipeline systems lack scenario-adaptive behavior | [arxiv 2606.04315](https://arxiv.org/abs/2606.04315) |
| **Metis: Bridging Text and Code Memory for Self-Evolving Agents** *(arxiv 2606.24151, Jun 2026)* | Dual memory: textual knowledge (experiences, facts) + code memory (executable functions, procedures); bridge mechanism converts insights to reusable code and vice versa; enables compounding skill acquisition across tasks | — | [arxiv 2606.24151](https://arxiv.org/abs/2606.24151) |
| **Omni-SimpleMem** *(arxiv 2604.01007, Apr 2026)* | Autonomous research pipeline (AutoResearch) discovers effective lifelong multimodal memory designs by running ~50 experiments automatically — bug fixes, architectural changes, prompt engineering; proposes Omni-SimpleMem from the discovered design | LoCoMo F1: 0.117 → 0.598 (+411%); Mem-Gallery F1: 0.254 → 0.797 (+214%) | [arxiv 2604.01007](https://arxiv.org/abs/2604.01007) |
| **ZenBrain** *(arxiv 2604.23878, Apr 2026)* | Neuroscience-inspired 7-layer memory architecture integrating 15 neuroscience mechanisms (FSRS spaced repetition, vmPFC sleep consolidation, Two-Factor Synaptic KG, etc.); first system claimed to integrate more than two such mechanisms simultaneously; MemoryCoordinator orchestrates them | LongMemEval-500: 91.3% of oracle at 1/106th token cost; LoCoMo F1 +20.7%; MemoryArena F1 +19.5%; won all 12 head-to-head comparisons vs Letta, Mem0, A-MEM | [arxiv 2604.23878](https://arxiv.org/abs/2604.23878) |
| **StructMem** *(arxiv 2604.21748, Apr 2026; ACL 2026 main)* | Structured memory for long-horizon LLM behavior; hierarchical framework with temporal anchoring from dual perspectives and periodic semantic consolidation; preserves event-level bindings and induces cross-event connections; LightMem open-source implementation | Improvements on LoCoMo temporal reasoning and multi-hop vs. baselines; reduced API calls and runtime | [arxiv 2604.21748](https://arxiv.org/abs/2604.21748) |
| **EvoMemBench** *(arxiv 2605.18421, May 2026)* | Evaluates 15 memory methods across 2 axes: in-episode vs. cross-episode; knowledge-oriented vs. execution-oriented; key finding: long-context baselines remain highly competitive; no single memory form works consistently; retrieval-based methods excel in knowledge-intensive tasks | — (evaluative; no single-system headline score) | [arxiv 2605.18421](https://arxiv.org/abs/2605.18421) |
| **HAGE: RL-Driven Weighted Graph Memory** *(arxiv 2605.09942, May 2026)* | Reconceptualizes memory retrieval as sequential query-conditioned traversal over a relational memory graph; RL jointly optimizes routing behavior and edge representations; addresses accuracy-efficiency trade-off in long-horizon reasoning | Improved long-horizon reasoning accuracy and accuracy-efficiency trade-offs vs. state-of-the-art | [arxiv 2605.09942](https://arxiv.org/abs/2605.09942) |
| **MemForest** *(arxiv 2605.23986, May 2026; submitted VLDB)* | Reformulates memory management as a write-efficient temporal data problem; MemTree hierarchical temporal index organizes memory as time-ordered trees; decouples memory construction via parallel processing with localized per-node updates | LongMemEval-S: 79.8% pass@1; ~6× higher memory construction throughput vs. EverOS | [arxiv 2605.23986](https://arxiv.org/abs/2605.23986) |
| **SaliMory** *(arxiv 2606.04120, Jun 2026; Meta)* | Trains a single LM to manage cognitively-structured memory (user facts, preferences, working memory) using hierarchical stage-wise process reward and reward-decomposed contrastive refinement; addresses credit assignment in multi-stage memory pipelines | Reduces memory-attributed failures by one-third; >10% end-to-end accuracy vs. SOTA; >2× "Good Personalization rate" | [arxiv 2606.04120](https://arxiv.org/abs/2606.04120) |
| **TOKI: Bitemporal Algebra for Contradiction Resolution** *(arxiv 2606.06240, Jun 2026)* | Formalizes contradiction resolution as a write-time concurrency control problem; bitemporal operators over a dual-row schema; soundness theorems covering isolation, schema, and provenance preservation — makes explicit the guarantees that production heuristics assume but never declare | LoCoMo: +0.86 with audit-row approach | [arxiv 2606.06240](https://arxiv.org/abs/2606.06240) |
| **RaMem: Contextual Reinstatement** *(arxiv 2606.22844, Jun 2026)* | Addresses "context collapse" — retrieved memory fragments appearing equally relevant across different situations involving recurring entities; validity-aware retrieval prioritizes context-compatible memories; structured context preserved during synthesis | Average F1 gains >10% across backbones on long-term memory benchmarks | [arxiv 2606.22844](https://arxiv.org/abs/2606.22844) |
| **MemSyco-Bench** *(arxiv 2607.01071, Jul 2026)* | Five-task benchmark evaluating memory sycophancy: whether agents reject incorrect memories as factual evidence, respect memory's applicable scope, resolve memory-vs-objective conflicts, track memory updates, and apply valid memory for personalization; finding: existing memory systems often increase sycophancy | — (diagnostic benchmark) | [arxiv 2607.01071](https://arxiv.org/abs/2607.01071) |
| **Forged Reasoning Attacks on LLM Agent Memory** *(arxiv 2607.05029, Jul 2026)* | First formal study of adversarial attacks that forge/manipulate LLM agent memory to redirect reasoning; proposes defenses | — (security/attack study) | [arxiv 2607.05029](https://arxiv.org/abs/2607.05029) |
| **MRMS: Multi-Resolution Memory Substrate** *(arxiv 2607.04617, Jul 2026)* | Organizes memory along representational (structured, vector, graph) and temporal (short-term traces, medium-term abstractions, long-term commitments) axes; synchronized structured-vector-graph memory where records govern eligibility, vectors enable retrieval, graphs manage contradictions; argues personalization is a memory design problem not a context-length problem | — (architecture paper) | [arxiv 2607.04617](https://arxiv.org/abs/2607.04617) |
| **LoCoMo-Plus** *(arxiv 2602.10715, Feb 2026)* | Extends LoCoMo with "cognitive memory under cue–trigger semantic disconnect"; tests realistic scenarios with implicit constraints (user state, unstated goals) where query and trigger share no surface features; proposes constraint-consistency evaluation framework to replace misaligned string-matching metrics | — (benchmark proposal; no scores reported) | [arxiv 2602.10715](https://arxiv.org/abs/2602.10715) |
| **GEM / MemState** *(arxiv 2605.26252, May 2026)* | Vision paper: proposes Governed Evolving Memory (GEM) — reconceptualizes memory as stateful evolution with four state-level operators (ingestion, revision, forgetting, retrieval) governed by six correctness conditions; diagnoses four current failure modes: unregulated growth, missing semantic revision, capacity-driven forgetting, read-only retrieval; MemState is a property-graph prototype | — (vision/framework paper; no benchmark scores) | [arxiv 2605.26252](https://arxiv.org/abs/2605.26252) |
| **BeliefMem** *(arxiv 2605.05583, May 2026)* | Probabilistic multi-hypothesis memory for partial observability; retains multiple candidate conclusions per observation with Noisy-OR probability updates as new evidence arrives; surfaces candidates with confidence scores at retrieval — avoids overconfident single-conclusion storage common in current systems | ALFWorld: 63.57% seen / 53.75% unseen (+11% over 2nd-best); LoCoMo: 42.38 F1 (GPT-4o-mini) — scores from paper body | [arxiv 2605.05583](https://arxiv.org/abs/2605.05583) |
| **DeMem** *(arxiv 2605.10870, May 2026)* | Decision-centric rate-distortion memory; formalizes memory as an information bottleneck preserving decision-critical distinctions rather than descriptive salience; online memory learner refines partitions only when data signals potential decision conflicts; descriptive similarity shown to be a weak predictor of evidence compatibility (Spearman ρ=0.103, AUC=0.548 on LoCoMo) | DeMem recovers 83% of gold evidence vs. 66% for description-based retrieval under matched budget | [arxiv 2605.10870](https://arxiv.org/abs/2605.10870) |
| **TriMem** *(arxiv 2605.19952, May 2026; TMLR)* | Three coexisting memory granularities: raw dialogue segments (storage fidelity), extracted atomic facts (retrieval), and synthesized profiles (deep reasoning); TextGrad-based prompt optimization iteratively refines extraction and profiling via response-quality feedback — enables lifelong memory improvement without parameter updates | Consistently outperforms baselines on LoCoMo and PerLTQA (no specific numbers in abstract) | [arxiv 2605.19952](https://arxiv.org/abs/2605.19952) |
| **R²-Mem** *(arxiv 2605.13486, May 2026)* | Reflective experience for memory search agents; offline phase: rubric-guided evaluator scores high- and low-quality search trajectories; self-reflection learner extracts abstract experience; at inference time, retrieved experience steers future search actions to avoid repeating past mistakes | F1 +22.6%, token consumption −12.9%, search iterations −20.2% (relative over strong baselines) | [arxiv 2605.13486](https://arxiv.org/abs/2605.13486) |
| **Mem-π** *(arxiv 2605.21463, May 2026)* | RL-trained on-demand guidance generation replacing similarity-based retrieval; dedicated LM learns when to produce guidance (and to abstain when unhelpful) and what to produce; tested across web navigation, terminal tool use, and embodied interaction; described as "Work in progress" | >20% relative improvement on average across task types (abstract); WebArena 43.1%, WorkArena 50.3%, ALFWorld 91.6% (paper body) | [arxiv 2605.21463](https://arxiv.org/abs/2605.21463) |
| **DCPM: Dual-Process Cognitive Memory** *(arxiv 2606.09483, Jun 2026)* | Dual-process hierarchical memory for self-evolving agents; System 1 (synchronous): records belief revisions as doubly-linked supersedes chains; System 2 (asynchronous): induces schemas/intentions and sweeps for cross-domain collisions; organizes memory from raw inputs to domain schemas; targets implicit personalization requiring reasoning over how a user has evolved | LongMemEval (DeepSeek): 85.17%; PersonaMem: 70.46%; PersonaMem-v2 (Kimi-2.5): 59.30% — from paper body | [arxiv 2606.09483](https://arxiv.org/abs/2606.09483) |
| **PROJECTMEM** *(arxiv 2606.12329, Jun 2026)* | Local-first event-sourced memory and judgment layer for AI coding agents; stores development history as append-only event log; "Memory-as-Governance" warns agents before repeating failed fixes or touching fragile files; targets the 5,000–20,000 token/session cost of reconstructing context from scratch | — (internal evaluation: 207 events across 10 projects over 2 months; no standard benchmark scores) | [arxiv 2606.12329](https://arxiv.org/abs/2606.12329) |

## Notable New / Emerging Systems

| System | Description | Status | Source |
|---|---|---|---|
| **MemOS** *(MemTensor)* | AI memory OS for LLM agents; persistent skill memory, cross-task reuse, hybrid FTS5+vector search; memos-local-plugin 2.0 (May 2026): L1 traces / L2 policies / L3 world models / crystallized Skills hierarchy; Reflect2Evolve architecture; hook-based plugin system; LoCoMo 92.34%, LongMemEval 93.40% (self-published, GitHub README, Jul 2026); leads OmniMemEval among 14 commercial memory products | Active, open-source, v2.0.22 (Jul 3 2026) | [github.com/MemTensor/MemOS](https://github.com/MemTensor/MemOS) |
| **EverOS** *(EverMind-AI; formerly EverMemOS)* | Renamed from EverMemOS on Apr 14, 2026; Skills Evolution Engine (234.8% relative task success gain over CoAct); HyperMem fast recall; ACL 2026 paper accepted; local-first Markdown + SQLite + LanceDB; multimodal file ingestion; v1.1.0 (Jun 24, 2026) with knowledge wikis + reflection; v1.1.1 (Jul 7, 2026) adds LoCoMo benchmark runner for reproducible evaluation, enhanced hybrid search stability, expanded CI for Python 3.12 + 3.13 | Active, open-source, v1.1.1 (Jul 7 2026) | [github.com/EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS) |
| **MemoryOS** *(BAI-LAB, EMNLP 2025)* | OS-inspired hierarchical memory (short→mid→long); heat-gated promotion; MCP server; +49% F1 on LoCoMo vs baseline; 5× faster via parallelization optimizations; MemoryOS-MCP open-sourced (Jun 2026) for agent client integration | Active, open-source | [github.com/BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) |
| **Cognee** | Knowledge graph + vector + relational 3-store architecture; 14 retrieval modes; strong multi-hop reasoning; v1.0 (Jun 2026): unified Postgres backend (runs full memory layer — graph + vectors + sessions — on single Postgres instance), self-improving memory via `memify` (prunes stale nodes, reweights edges by usage), 1M+ pipelines/month across 70+ companies | Active, v1.0 (Jun 2026) | [cognee.ai](https://cognee.ai) |
| **OpenViking** *(volcengine)* | Open-source context database; filesystem paradigm for memory/resources/skills unified management | Active, launched Feb 2026 | [github.com/volcengine/viking-db](https://github.com/volcengine/viking-db) |
| **Mastra Observational Memory** | Observer + Reflector background agents; stable append-only context; 5-40x compression; SOTA LongMemEval (94.87% gpt-5-mini, Apr 2026) | Active, TypeScript | [github.com/mastra-ai/mastra](https://github.com/mastra-ai/mastra) |
| **Supermemory** | Hosted AI memory API; production engine (85.4% LongMemEval-S); experimental ASMR flow (~99% via parallel LLM agents, no vector DB); TypeScript | Active, hosted | [supermemory.ai](https://supermemory.ai) |
| **Memori** *(MemoriLabs)* | SQL-native memory infra; facts/preferences/rules/summaries; intelligent decay; 81.95% LoCoMo at 4.98% cost; Cloud + OpenClaw plugin + BYODB | Active, open-source | [github.com/memorilabs/memori](https://github.com/memorilabs/memori) |
| **MemFactory** *(arxiv)* | Unified modular training + inference framework; Lego-like components; GRPO fine-tuning; supports Memory-R1/RMM/MemAgent | Research | [arxiv 2603.29493](https://arxiv.org/abs/2603.29493) |
| **Hindsight** *(Vectorize)* | "Retain, Recall, Reflect" architecture; Reflector background agent reasons over structured memory bank; SOTA across BEAM + LOCOMO + LongMemEval simultaneously (Apr 2026); BEAM 100K 73.4%, LOCOMO 92%, LongMemEval-S 94.6%; BEAM 10M 64.1% (next best: 40.6%); v0.7.0 (May 27) multilingual/CJK, polyglot Control Plane in 8 languages; v0.8.0 (Jun 8) mental model history tables, semantic dedup, LLM prompt-prefix caching, cross-instance bank migration; v0.8.2 (Jun 12) Memory Defense (PII protection), reversible memory curation; v0.8.4 (Jul 1) multi-LLM failover, configurable recency decay; integrations: Aider, GitHub Copilot, Windsurf, Continue.dev, Zapier, Cursor, Cline, Flowise, Roo Code, Haystack, Google ADK | Active, open-source + hosted | [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight), [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io) |
| **Engram** *(engram.fyi)* | MCP memory server for Claude Code, Cursor, and AI coding agents; episodic + semantic + procedural memory via single routing layer | Active, MCP | [engram.fyi](https://engram.fyi) |
| **MemPalace** *(milla-jovovich/mempalace)* | Verbatim storage + ChromaDB semantic search; "palace" spatial metaphor (wings/rooms/halls); AAAK compression for selective verbatim recall | Active, open-source | [github.com/milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace) |
| **Hermes Agent** *(Nous Research)* | Open-source autonomous AI agent (Feb 2026) with 7 pluggable memory providers: Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover; memory provider selection at init | Active, open-source | [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) |
| **Anthropic Auto Dream** *(Mar 2026)* | Claude Code's functional memory consolidation — first major AI lab treating agent memory as cognitive architecture; auto-distills sessions into persistent project knowledge | Active, Claude Code feature | [anthropic.com](https://anthropic.com) |
| **Neo4j Agent Memory** *(neo4j-labs)* | Graph-native memory system for AI agents backed by Neo4j; stores conversations, builds knowledge graphs, agent framework integrations | Active, open-source | [github.com/neo4j-labs/agent-memory](https://github.com/neo4j-labs/agent-memory) |
| **ALMA** *(zksha)* | Automated meta-Learning of Memory designs for Agentic systems — meta-learns memory designs to replace human-engineered designs | Research, open-source | [github.com/zksha/alma](https://github.com/zksha/alma) |
| **Eywa** *(Resham Joshi)* | Provenance-grounded long-term memory — stores immutable source evidence before deriving canonical facts; zero LLM calls at write time for evidence ingestion; targets provenance-role collapse failure mode | Active, open-source | [github.com/reshamjoshi/eywa](https://github.com/reshamjoshi/eywa) |
| **MemMachine** | Universal ground-truth-preserving memory layer; LoCoMo 91.69%, LongMemEvalS 93.0%; ~80% fewer tokens than Mem0; retrieval-stage optimization over ingestion-stage transformation | Active, open-source | [arxiv 2604.04853](https://arxiv.org/abs/2604.04853) |
| **OMEGA** | Local-first persistent memory for AI coding agents; 95.4% LongMemEval (#1 leaderboard); pip install; no cloud | Active, pip-installable | [omegamax.co](https://omegamax.co/benchmarks) |
| **MIRIX** *(MIRIX AI)* | Multi-agent memory system with 6 specialized types: Core, Episodic, Semantic, Procedural, Resource Memory, Knowledge Vault; multimodal (text + visual screenshots); real-time screen monitoring + local storage; LoCoMo 85.4%; ScreenshotVQA +35% vs RAG at 99.9% storage reduction; arxiv 2507.07957 (Jul 2025) | Research, open-source | [arxiv 2507.07957](https://arxiv.org/abs/2507.07957) |
| **ByteRover** *(campfirein)* | Portable memory layer for AI coding agents (formerly Cipher); Context Tree hierarchical knowledge structure organized by domain/topic/subtopic; LoCoMo 96.1% + LongMemEval-S 92.8% (v2.1.5, self-published Mar 2026); local-first, no cloud/telemetry by default; MCP integration; compatible with 22+ AI coding tools; 4.8k GitHub stars | Active, v3.14.0 (May 2026), TypeScript, Elastic License 2.0 | [github.com/campfirein/byterover-cli](https://github.com/campfirein/byterover-cli), [byterover.dev](https://www.byterover.dev/) |
| **SimpleMem / EvolveMem** *(aiming-lab)* | Research memory framework: semantic lossless compression, 30× fewer tokens vs full-context; multimodal support (EvolveMem: images, audio); self-evolving via closed-loop failure diagnosis | Active, research | [arxiv 2601.02553](https://arxiv.org/abs/2601.02553) / [arxiv 2605.13941](https://arxiv.org/abs/2605.13941) |
| **Microsoft Memora** *(Microsoft Research, ICML 2026)* | Harmonic Memory Representation: decouples what is stored (full content + timeline details) from how it is retrieved (6–8 word primary abstraction + lightweight cue anchors); policy-guided iterative retriever treats memory access as active reasoning, not one-shot search; ICML 2026 paper; SOTA on LoCoMo and LongMemEval with up to 98% fewer context tokens vs full-context inference | Active, open-source | [github.com/microsoft/Memora](https://github.com/microsoft/Memora), [microsoft.com/research/blog](https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/) |
| **MemCog** *(Tencent / WeChat AI, arxiv 2605.28046, May 2026)* | Reframes memory as active cognitive process; Navigable Memory Store with associative link graphs; agents traverse via Cross-Dimensional Navigation Interface rather than one-shot retrieval; Proactive Reasoning Protocol drives agents to spontaneously initiate memory exploration; introduces ProactiveMemBench for proactive memory triggering | Research, arxiv | [arxiv 2605.28046](https://arxiv.org/abs/2605.28046) |
| **SuperLocalMemory** *(Qualixar)* | Local-only, zero-LLM memory (Mode A) with Bayesian trust defense against memory poisoning; biologically-inspired forgetting (V3.3: memories strengthen with use, fade when neglected); cognitive quantization; multi-channel retrieval; EU AI Act compliant by architecture (zero cloud calls); Mode C (with LLM): 87.7% LoCoMo (self-published); Mode A (zero-cloud): 74.8% LoCoMo | Active, open-source | [github.com/qualixar/superlocalmemory](https://github.com/qualixar/superlocalmemory), [arxiv 2604.04514](https://arxiv.org/abs/2604.04514) |
| **OpenAI Dreaming V3** *(OpenAI, Jun 4, 2026)* | Rebuilt ChatGPT memory using async between-session "dreaming" process: reviews session transcripts, synthesizes patterns, merges duplicates, replaces stale entries; temporal self-correction (e.g., "going to Singapore" → "went to Singapore"); 5× compute reduction enabling first free-tier rollout; readable Memory Summary page with user controls | Active, ChatGPT Plus/Pro + Free feature | [openai.com](https://openai.com/index/chatgpt-memory-dreaming/) |
| **Cloudflare Agent Memory** *(Cloudflare, Apr 2026)* | Managed persistent memory for AI agents (private beta); five parallel retrieval channels fused with Reciprocal Rank Fusion: full-text, exact fact-key lookup, raw message, direct vector, HyDE vector; Llama 4 Scout (17B MoE) for extraction; Nemotron 3 (120B MoE) for synthesis; eight-check memory verifier; integrated in Cloudflare Workers AI | Active, managed cloud (private beta) | [blog.cloudflare.com](https://blog.cloudflare.com/introducing-agent-memory/) |
| **LangMem** *(LangChain)* | Background memory SDK for LangGraph-native agents; extracts, consolidates, and updates agent knowledge asynchronously; pluggable LangGraph BaseStore backends; 746K+ monthly PyPI downloads; recommended for batch/background tasks, not latency-sensitive paths | Active, open-source | [langchain-ai.github.io/langmem](https://langchain-ai.github.io/langmem/), [github.com/langchain-ai/langmem](https://github.com/langchain-ai/langmem) |
| **GBrain** *(garrytan)* | Garry Tan's (YC President) opinionated OpenClaw/Hermes agent brain; dual-engine storage: PGLite (Postgres 17 compiled to WASM, zero-config default, ~50K page scale) + pgvector/HNSW (Supabase/self-hosted for shared/large deployments); Git-backed Markdown files as system of record; typed knowledge graph (edges: `attended`, `works_at`, `invested_in`) auto-linked at write time with zero LLM calls; 24/7 dream cycle for dedup, citation fixing, and salience scoring; RRF retrieval with 4-layer dedup + backlink-boosted ranking; LongMemEval R@5 97.6% (self-published, gbrain-evals repo, no LLM in retrieval loop; R@5 retrieval recall — not directly comparable to LongMemEval-S generation accuracy); BrainBench (internal 240-page corpus) R@5 97.9%; 25,292 GitHub stars | Active, open-source, TypeScript, MIT | [github.com/garrytan/gbrain](https://github.com/garrytan/gbrain), [github.com/garrytan/gbrain-evals](https://github.com/garrytan/gbrain-evals) |
| **Hippo** *(kitfunso)* | Biologically-inspired memory for AI agents; 7 hippocampal mechanisms: (1) two-speed storage — episodic buffer + semantic consolidation, (2) decay by default — 7-day half-life, (3) retrieval strengthening — +2 days on recall, (4) emotional/error tagging — amygdala analogue boosts for errors/breakthroughs, (5) sleep consolidation — 5-step replay+merge clustering 3+ similar episodes into one semantic memory, (6) schema acceleration — fits existing patterns 1.5× faster, (7) conflict detection — contradictions flagged not silently overwritten; R-STDP implemented as reward-proportional decay modulation (0.5–1.5× factor), not full spiking neural network plasticity; LongMemEval per-haystack: 98.6% R@5 (MiniLM-L6 default) / 99.8% (voyage-3-large); full unified store: 73.8% R@5 (oracle-split, BM25+cosine, 19,195 sessions) — the per-haystack scores are not comparable to full-store evaluations; sequential learning magnitude claim formally retracted in v1.7.9; SQLite, zero runtime dependencies, Node.js 22.5+ | Active, open-source, TypeScript, MIT, v1.23.0 | [github.com/kitfunso/hippo-memory](https://github.com/kitfunso/hippo-memory) |
| **PMB** *(oleksiijko / Project Memory Bank)* | Local-first AI memory via MCP; SQLite (WAL-mode, append-only source of truth) + LanceDB (rebuildable vector indexes); 4-way hybrid recall fused via RRF: BM25 lexical + dense vector semantic + entity graph traversal + optional cross-encoder rerank; async writes (<1ms MCP response); workspace-scoped per project to prevent cross-project context bleed; MCP cold boot 3.7 s; LoCoMo recall@10 94.5% (self-published, author's HackerNoon article; benchmark script public at `scripts/benchmarks/benchmark_locomo.py`); 292 GitHub stars, launched May 2026, PH 228 upvotes | Active, open-source, Python, Apache 2.0, v1.2.2 | [github.com/oleksiijko/pmb](https://github.com/oleksiijko/pmb) |
| **Unabyss** | MCP-native cloud context layer aggregating 20+ external sources (LinkedIn, Notion, Gmail, Slack, GitHub, Google Drive, meeting tools); segmentation engine scores context by topic, confidence, sensitivity, personal/professional axes before injecting into AI tools; structured markdown output (persona.md, voice.md, company.md); three refresh mechanisms: source-change detection, daily scheduled refresh, manual override; claims "10x fewer tokens than RAG" (unsubstantiated, no published methodology); #1 Product Hunt Product of Day May 25, 2026 (807 upvotes); proprietary cloud SaaS, two public GitHub repos contain only thin integration plugins | Active, cloud SaaS (proprietary) | [unabyss.com](https://unabyss.com/) |

## Change Log

| Date | Change | Source |
|---|---|---|
| 2026-04-02 | EverMemOS LOCOMO updated 92.3% → 93.05%; LongMemEval 82% → 83.00%; HaluMem 90.04% added; Feb 2026 product release as source | [evermemory.ai](https://evermemory.ai) |
| 2026-04-02 | Honcho updated: v3.0.4 RC released (2026-04-02); MCP server improvements (inspect_workspace, list_workspaces tools added); BEAM 100K confirmed 63%; working on BEAM 10M | [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho) |
| 2026-04-02 | Letta LOCOMO 74.0% added (Letta Filesystem: simple file storage + grep beats specialized tools) | [letta.com](https://letta.com) |
| 2026-04-02 | Added MemoryOS (BAI-LAB, EMNLP 2025 Oral) as new row: OS-inspired hierarchical memory, +49% F1 on LoCoMo, MCP server, Python/SQLite | [github.com/BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) |
| 2026-04-02 | Added new systems table: MemOS (MemTensor), MemoryOS (BAI-LAB), Cognee, OpenViking | GitHub search |
| 2026-04-02 | Added new benchmarks table: HaluMem, AMB (Vectorize), two arxiv papers (Mar 2026) | Web search |
| 2026-04-02 | Added Letta baseline threat note; LOCOMO table updated | — |
| 2026-04-02 | Mem0: very active development (Codex plugin support, CLI improvements, purple branding); v1.0.3 current | [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) |
| 2026-04-03 | Added Mastra Observational Memory: SOTA LongMemEval 84.23% (gpt-4o) / 94.87% (gpt-5-mini); Observer+Reflector architecture | [github.com/mastra-ai/mastra](https://github.com/mastra-ai/mastra) |
| 2026-04-03 | Added Supermemory: 85.4% LongMemEval-S (production), ~99% experimental ASMR (parallel LLM agents, no vector DB) | [supermemory.ai](https://supermemory.ai) |
| 2026-04-03 | Added Memori (MemoriLabs): 81.95% LoCoMo at 4.98% cost of full context; SQL-native; Cloud + OpenClaw plugin | [github.com/memorilabs/memori](https://github.com/memorilabs/memori) |
| 2026-04-03 | Added new papers: MemFactory (2603.29493), AgeMem (2601.01885), A-MAC (2603.04549), LifeBench (2603.03781), Memory-R1 (2603.26035), LongRewardBench (2604.12406) | arxiv |
| 2026-04-03 | Honcho: v3.0.4 RC (Apr 2) — external vector store for message search, dialectic connection fix, MCP server active; LOCOMO 89.9% confirmed LLM-judge | [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho) |
| 2026-04-03 | Updated gap analysis: added LongMemEval trailing note, context efficiency dimension, Memori to LoCoMo list | — |
| 2026-04-06 | **Added Hindsight (Vectorize) to matrix** — new BEAM 100K leader at 75% (since corrected to 73.4%), LOCOMO 92% (#1 on AMB), LongMemEval-S 94.6% (#1 on AMB). BEAM 10M 64.1% — next best is 40.6%, a 58% relative margin. | [hindsight.vectorize.io](https://hindsight.vectorize.io/blog/2026/04/02/introducing-hindsight) |
| 2026-04-06 | **Added OMEGA**: 95.4% LongMemEval (#1 on leaderboard), local-first pip install, no cloud | [omegamax.co](https://omegamax.co/benchmarks) |
| 2026-04-06 | **Added MemMachine**: 91.69% LoCoMo, 93.0% LongMemEval-S; retrieval-stage-only approach; ~80% fewer tokens than Mem0 | [arxiv 2604.04853](https://arxiv.org/abs/2604.04853) |
| 2026-04-06 | **Added new papers**: MemMachine (2604.04853), SuperLocalMemory V3.3 (2604.04514), LongRewardBench (2604.12406) | arxiv |
| 2026-04-06 | **Added Hermes Agent** (Nous Research) to notable systems — 7-provider pluggable memory aggregator (Honcho, OpenViking, Mem0, Hindsight, etc.), launched Feb 2026 | [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) |
| 2026-04-06 | Hindsight noted as new BEAM 100K leader (originally 75%, since corrected to 73.4%). Updated LOCOMO/LongMemEval lists to include Hindsight and Engram. | Web search |
| 2026-04-07 | **Added MemU** — file-system hierarchy memory with explicit LLM CRUD; LOCOMO 92.09% self-published | [arxiv/GitHub] |
| 2026-04-07 | **Added ByteRover** (campfirein) to matrix and notable systems — Context Tree architecture; LoCoMo 92.2% (v2.0 best run Feb 2026); LongMemEval 89.4%; local-first; 22+ tool integrations | [github.com/campfirein/byterover-cli](https://github.com/campfirein/byterover-cli) |
| 2026-04-07 | **Added Engram** (engram.fyi) to notable systems — MCP memory server for Claude Code, Cursor; episodic+semantic+procedural via single MCP route | [engram.fyi](https://engram.fyi) |
| 2026-04-07 | **Added Anthropic Auto Dream** to notable systems — Claude Code functional memory consolidation (Mar 2026) | [anthropic.com](https://anthropic.com) |
| 2026-04-07 | **Added Neo4j Agent Memory** to notable systems — graph-native memory backed by Neo4j | [github.com/neo4j-labs/agent-memory](https://github.com/neo4j-labs/agent-memory) |
| 2026-04-07 | **Added ALMA** (zksha) to notable systems — automated meta-learning of memory designs | [github.com/zksha/alma](https://github.com/zksha/alma) |
| 2026-04-07 | **Added MemPalace** (milla-jovovich) to matrix and notable systems — verbatim storage; BEAM 100K 49% (independently tested); LongMemEval R@5 96.6% (self-reported retrieval recall) | [github.com/milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace) |
| 2026-04-08 | **Hindsight now native Hermes Agent memory provider** (Apr 6 blog post). VentureBeat coverage: "91% accuracy" on LongMemEval. Growing ecosystem integration. | [hindsight.vectorize.io/blog/2026/04/06](https://hindsight.vectorize.io/blog/2026/04/06) |
| 2026-04-10 | **Supermemory claims ~99% LongMemEval** via agent swarm approach; Hindsight BEAM 10M 64.1% confirmed as massive lead (next: 40.6%, 58% margin) | [aihola.com](https://aihola.com/article/supermemory-99-longmemeval) |
| 2026-04-10 | Updated gap analysis: OMEGA now LongMemEval #1 (95.4%), MemMachine enters LoCoMo top tier, ByteRover LoCoMo/LongMemEval noted | — |
| 2026-04-14 | **Hindsight BEAM 100K corrected 75% → 73.4%**: primary source is the Vectorize blog post dated 2026-04-02, which reports 73.4% as the hard number. The 75% figure in earlier entries was a rounded/approximated figure. All matrix entries updated. | [hindsight.vectorize.io/blog/2026/04/02](https://hindsight.vectorize.io/blog/2026/04/02) |
| 2026-04-14 | **BEAM methodology note**: Honcho 63.0% and Hindsight 73.4% are both arithmetic mean of per-question BEAM rubric scores across 400 questions, single run, using BEAM's nugget-based LLM judge. | [BEAM paper](https://arxiv.org/abs/2503.24129) |
| 2026-04-14 | Added Memori (MemoriLabs) to competitive matrix. | [github.com/memorilabs/memori](https://github.com/memorilabs/memori) |
| 2026-04-14 | Added MemPalace BEAM 100K (49.0%, independently tested, raw mode + GPT-5.4-mini synthesis) and LongMemEval R@5 (96.6%, self-reported retrieval recall). | [github.com/milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace) |
| 2026-05-18 | **ByteRover updated**: LoCoMo corrected 92.2% → 96.1% (v2.1.5, self-published Mar 2026 on byterover.dev); LongMemEval updated 89.4% → 92.8% (v2.1.5); v3.14.0 (May 2026) current; 4.8k GitHub stars. | [byterover.dev](https://www.byterover.dev/blog/byterover-v2) |
| 2026-05-18 | **Honcho LongMemEval clarified**: evals.honcho.dev shows 90.4% on LongMem-S (personal assistant, 500 sessions); 88.8% on LongMem-M (500 sessions). Source: evals.honcho.dev (self-published, May 2026). | [evals.honcho.dev](https://evals.honcho.dev) |
| 2026-05-18 | **Added EverOS rename note**: EverMemOS → EverOS on Apr 14, 2026 brand upgrade. New GitHub: EverMind-AI/EverOS. Added ACL 2026 acceptance. | [github.com/EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS) |
| 2026-05-18 | **Added Eywa to notable systems and papers table** — provenance-grounded long-term memory (arxiv 2605.30771, May 2026); immutable evidence-before-belief; zero LLM calls at write time | [arxiv 2605.30771](https://arxiv.org/abs/2605.30771) |
| 2026-05-18 | **Added SimpleMem/EvolveMem to notable systems and papers table** — SimpleMem (arxiv 2601.02553, Jan 2026): 30× fewer tokens, multimodal, zero retrieval LLM; EvolveMem (arxiv 2605.13941, May 2026): self-evolving closed-loop diagnosis | [arxiv 2601.02553](https://arxiv.org/abs/2601.02553) |
| 2026-05-18 | **Added 2 new survey papers**: "From Storage to Experience" (arxiv 2605.06716, May 2026) — 3-stage memory evolution framework (Storage→Retrieval→Experience); Mnemonic Sovereignty (arxiv 2604.16548, Apr 2026) — first LLM memory security survey | arxiv |
| 2026-05-18 | **MemOS v2.0.15 released** (May 11, 2026) — hook-based plugin system for extensible memory processing; Reflect2Evolve architecture in multi-turn capture; LoCoMo 92.34%, LongMemEval 93.40% self-published | [github.com/MemTensor/MemOS](https://github.com/MemTensor/MemOS) |
| 2026-05-18 | **Hindsight v0.6.2 released** (May 14, 2026) — bugfix: timestamp not reaching retain API; openclaw v0.7.7, claude-code v0.6.5 integration releases (May 15) | [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) |
| 2026-05-18 | **Letta v0.16.8 released** (May 14, 2026) — security fix: JSON instead of pickle for sandbox→server tool result transport | [github.com/letta-ai/letta](https://github.com/letta-ai/letta) |
| 2026-05-18 | Honcho active development (Apr–May 2026): embeddings configurable (#678), deriver custom instructions (#609), dialectic N+1 query fix (#707), better SDK compatibility | [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho) |
| 2026-05-18 | Mem0 May 2026: Agent Mode bootstrap + claim flow CLI (May 14); TypeScript latestOnly on hosted memory reads; hosted Qdrant migration scripts; Composio integration | [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) |
| 2026-05-18 | Memori v0.1.5 (May 15, 2026): Hermes integration skill file (#499), improved MCP docs, OpenAI multimodal array parsing fix, recall filter bugfix | [github.com/memorilabs/memori](https://github.com/memorilabs/memori) |
| 2026-07-04 | **EverMemOS officially renamed EverOS** (Apr 14, 2026 brand upgrade + global public beta). New GitHub: EverMind-AI/EverOS. New features: knowledge wikis, reflection, v1.1.0 (Jun 24, 2026). | [github.com/EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS) |
| 2026-07-04 | **Honcho LongMemEval-S updated 88.8% (LongMem-M) → 90.4% (LongMem-S)** per evals.honcho.dev (self-published, May 2026). BEAM 10M corrected from "SOTA" to 40.9% (0.409); Hindsight holds BEAM 10M lead at 64.1%. | [evals.honcho.dev](https://evals.honcho.dev) |
| 2026-07-04 | **Mem0 v3 algorithm** (self-published, mem0.ai/research, Apr 2026): LoCoMo 92.5%, LongMemEval 94.4%, BEAM 1M 64.1%, BEAM 10M 48.6% — all self-published; BEAM 10M 48.6% now second-best behind Hindsight 64.1% | [mem0.ai/research](https://mem0.ai/research) |
| 2026-07-04 | **Mem0 releases** — SDK v2.0.11 (Jul 1, 2026): Pi Agent Plugin (pre-fetches memories before agent execution), OpenCode Plugin v0.2.1, support for Gemini 2.5 Flash + Pro; TypeScript SDK v2.1.6 (Jul 2) | [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) |
| 2026-07-04 | **MemOS v2.0.22 released** (Jul 3, 2026); LoCoMo 92.34%, LongMemEval 93.40% (self-published, GitHub README); leads OmniMemEval among 14 commercial memory products | [github.com/MemTensor/MemOS](https://github.com/MemTensor/MemOS) |
| 2026-07-04 | **Added Eywa to notable systems and papers table** — provenance-grounded memory; immutable evidence-before-belief design; zero LLM calls at write time for evidence ingestion | [arxiv 2605.30771](https://arxiv.org/abs/2605.30771) |
| 2026-07-04 | **Added 4 new papers**: LongMemEval-V2 (arxiv 2605.12493, May 2026 — web-agent memory benchmark, 451 questions, 115M token trajectories); Portable Agent Memory Protocol (arxiv 2605.11032, May 2026 — interoperability standard); Are We Ready? (arxiv 2606.24775, Jun 2026 — 12-system evaluation, 4-property framework); EvolveMem (arxiv 2605.13941, May 2026 — self-evolving SimpleMem follow-up) | arxiv |
| 2026-07-05 | **Added MemOS (MemTensor) to competitive matrix**: LoCoMo 92.34%, LongMemEval 93.40% (self-published, v2.0.22, Jul 2026); Reflect2Evolve architecture; multi-modal; leads OmniMemEval among 14 commercial memory products (10 datasets); Local Plugin v2.0 self-evolving memory (L1/L2/L3 tiers). | [github.com/MemTensor/MemOS](https://github.com/MemTensor/MemOS) |
| 2026-07-05 | **Hindsight v0.7.0–v0.8.4 (May–Jul 2026)**: v0.7.0 (May 27) multilingual/CJK, polyglot Control Plane in 8 languages; v0.8.0 (Jun 8) mental model history tables, semantic dedup, LLM prompt-prefix caching, cross-instance bank migration; v0.8.2 (Jun 12) Memory Defense (PII protection), reversible memory curation; v0.8.4 (Jul 1) multi-LLM failover, configurable recency decay. New integrations: Aider, GitHub Copilot, Windsurf, Continue.dev, Zapier, Cursor, Cline, Flowise, Roo Code, Haystack, Google ADK. | [hindsight.vectorize.io/blog](https://hindsight.vectorize.io/blog) |
| 2026-07-05 | **BAI-LAB MemoryOS**: 5× faster via parallelization optimizations; MemoryOS-MCP open-sourced (Jun 2026) for agent client integration. | [github.com/BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) |
| 2026-07-05 | **Added MIRIX to notable systems**: 6-type multi-agent memory (Jul 2025 paper, not previously tracked); LoCoMo 85.4%; multimodal with real-time screen monitoring; 99.9% storage reduction vs RAG on ScreenshotVQA. | [arxiv 2507.07957](https://arxiv.org/abs/2507.07957) |
| 2026-07-05 | **Added 5 new papers**: MemIR (arxiv 2605.25869, typed provenance memory), RecMem (arxiv 2605.16045, ACL 2026, 87% token reduction), MemFail (arxiv 2605.26667, failure-mode diagnostics), MemPro (arxiv 2606.00619, evolvable pipelines), MIRIX (arxiv 2507.07957, multimodal multi-agent). | arxiv |
| 2026-07-06 | **Honcho v3.0.10 RC (Jun 15) and v3.0.11 RC (Jun 24, 2026)**: CloudEvents + Langfuse tracing projections over captured LLM stream, OpenRouter header attribution, deriver batch flushing improvements, surprisal tree parameter fixes, JWT scope fixes. | [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho) |
| 2026-07-06 | **Hindsight Eve v0.2.0 (Jul 3, 2026)**: New auto-memory mode that requires no model tool-calling; enables passive memory retention for models without function-calling support. | [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) |
| 2026-07-06 | **Added 4 new papers (Jun 2026)**: T-Mem (arxiv 2606.15405 — write-time rehearsal triggers, LoCoMo SOTA claimed), MemRefine (arxiv 2606.13177 — LLM-guided compression within fixed storage budgets), AutoMEM/Cross-Scenario (arxiv 2606.04315 — agentic harness outperforms passive pipelines across 5 scenarios), Metis (arxiv 2606.24151 — dual text+code memory for self-evolving agents). | arxiv |
| 2026-07-07 | **EverOS v1.1.1 released (Jul 7, 2026)**: adds reproducible LoCoMo benchmark runner, enhanced hybrid search stability, expanded CI for Python 3.12 + 3.13; release notes say hybrid search "stably targets around 91%, agentic search around 93%" on LoCoMo (approximate, not a hard benchmark figure). | [github.com/EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS) |
| 2026-07-07 | **Mem0 security fixes (Jul 7, 2026)**: patched SQL and Cypher injection vulnerabilities in PGVector, Azure MySQL, and Neptune backends; URL path encoding fix; TypeScript SDK expanded with MongoDB, Elasticsearch, OpenSearch, xAI/Grok providers merged same day. | [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) |
| 2026-07-07 | **Hindsight Eve v0.2.1 (Jul 6, 2026)**: patch adding assistant-reply retention by default in Eve auto-memory mode; timestamps unconfirmed whether before or after prior run cutoff. | [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) |
| 2026-07-07 | **Added 13 new papers** to papers table (Apr–Jul 2026 backfill): Omni-SimpleMem (2604.01007 — AutoResearch-guided multimodal memory, LoCoMo F1 +411%), ZenBrain (2604.23878 — 7-layer neuroscience architecture, LongMemEval-500 91.3% oracle-match at 1/106th token cost), StructMem (2604.21748 — ACL 2026 main, temporal anchoring + semantic consolidation), EvoMemBench (2605.18421 — evaluates 15 methods across in/cross-episode and knowledge/execution axes), HAGE (2605.09942 — RL-driven graph memory traversal), MemForest (2605.23986 — temporal hierarchical indexing, 6× throughput vs EverOS, VLDB submission), SaliMory (2606.04120 — Meta, process-reward trained cognitive memory, 2× personalization rate), TOKI (2606.06240 — bitemporal operator algebra for contradiction resolution), RaMem (2606.22844 — contextual reinstatement, F1 +10%), MemSyco-Bench (2607.01071 — sycophancy benchmark), Forged Reasoning Attacks (2607.05029 — adversarial memory manipulation + defenses), MRMS (2607.04617 — multi-resolution memory substrate). | arxiv |
| 2026-07-07 | **Added 6 notable systems**: Microsoft Memora (ICML 2026, harmonic memory representation, SOTA LoCoMo + LongMemEval, 98% fewer tokens, github.com/microsoft/Memora); MemCog (Tencent/WeChat, arxiv 2605.28046, 95.8% LongMemEval + 92.98% LoCoMo self-published, proactive memory navigation); SuperLocalMemory (Qualixar, zero-LLM Mode A, EU AI Act compliance, 87.7% LoCoMo Mode C); OpenAI Dreaming V3 (Jun 4, async synthesis, temporal self-correction, 5× compute reduction, free-tier rollout); Cloudflare Agent Memory (private beta Apr 2026, 5-channel RRF retrieval); LangMem (LangChain, 746K monthly downloads, background memory for LangGraph). | [github.com/microsoft/Memora](https://github.com/microsoft/Memora), [arxiv 2605.28046](https://arxiv.org/abs/2605.28046), [blog.cloudflare.com](https://blog.cloudflare.com/introducing-agent-memory/), [openai.com](https://openai.com/index/chatgpt-memory-dreaming/), [langchain-ai.github.io/langmem](https://langchain-ai.github.io/langmem/) |
| 2026-07-07 | **Cognee updated v0.5.2 → v1.0** (Jun 2026): unified Postgres backend replaces split graph+vector stores; self-improving memory via `memify`; 1M+ pipelines/month at 70+ companies. | [cognee.ai/blog](https://www.cognee.ai/blog/guides/building-an-ai-agent-best-persistent-memory-layer) |
| 2026-07-07 | **Added GBrain** (garrytan/gbrain, 25,292 stars, Garry Tan/YC President): zero-LLM typed knowledge graph, dual PGLite WASM + pgvector/HNSW engine, Git-backed Markdown system of record, dream-cycle dedup; LongMemEval R@5 97.6% (self-published, gbrain-evals repo; retrieval recall metric — not comparable to LongMemEval-S generation accuracy); BrainBench internal 97.9%. | [github.com/garrytan/gbrain](https://github.com/garrytan/gbrain) |
| 2026-07-07 | **Added Hippo** (kitfunso/hippo-memory, 709 stars, v1.23.0): 7 biologically-inspired hippocampal mechanisms; R-STDP as reward-proportional decay modulation; LongMemEval per-haystack 98.6%/99.8% R@5, but 73.8% on full unified 19k-session store; sequential learning magnitude claim formally retracted in v1.7.9 after pre-registered evaluation failure. | [github.com/kitfunso/hippo-memory](https://github.com/kitfunso/hippo-memory) |
| 2026-07-07 | **Added PMB** (oleksiijko/pmb, 292 stars, Apache 2.0, v1.2.2): local-first 4-way hybrid retrieval (SQLite+LanceDB BM25+vector+entity graph+RRF); LoCoMo recall@10 94.5% (self-published, public benchmark script). | [github.com/oleksiijko/pmb](https://github.com/oleksiijko/pmb) |
| 2026-07-07 | **Added Unabyss** (cloud SaaS, 807 PH upvotes, #1 PH Product of Day May 25, 2026): MCP-native personal data aggregator connecting 20+ external sources; segmentation engine; no published benchmarks; proprietary cloud, not open-source. | [unabyss.com](https://unabyss.com/) |
| 2026-07-07 | **Added 9 new papers** to papers table: LoCoMo-Plus (2602.10715 — cognitive memory with cue–trigger semantic disconnect benchmark), GEM/MemState (2605.26252 — governed evolving memory framework, 4 state-level operators), BeliefMem (2605.05583 — probabilistic multi-hypothesis memory, ALFWorld +11%), DeMem (2605.10870 — decision-centric rate-distortion, 83% gold evidence), TriMem (2605.19952 — three-granularity + TextGrad optimization), R²-Mem (2605.13486 — reflective experience for search agents, F1 +22.6%), Mem-π (2605.21463 — RL-trained on-demand guidance generation, >20% improvement), DCPM (2606.09483 — dual-process belief chains + schema induction, LongMemEval 85.17%), PROJECTMEM (2606.12329 — event-sourced memory-as-governance for coding agents). | arxiv |

---

# Appendix: Hindsight Deep Dive

## 1. What Is It

Hindsight is an open-source agent memory system built by Vectorize. It implements the "Retain, Recall, Reflect" architecture: three distinct operations that together handle the full memory lifecycle for LLM agents.

## 2. Core Architecture

### 2.1 The Four Memory Networks

```
Conversation → [Retain] → Structured Memory Bank
                               ↓
Query      ← [Recall]  ← Relevant Memories
                               ↓
                          [Reflect]
                               ↓
                          Updated Bank
```

The Memory Bank stores: **entities**, **relationships**, **preferences**, **events**, **summaries**.

### 2.2 TEMPR — Temporal Entity Memory Priming Retrieval

TEMPR is Hindsight's retrieval algorithm:
1. **T**emporal: recency-weighted scoring
2. **E**ntity: entity-level matching
3. **M**emory: semantic similarity
4. **P**riming: working memory priming
5. **R**etrieval: ranked result fusion

### 2.3 CARA — Coherent Adaptive Reasoning Agents

CARA is Hindsight's framework for multi-agent coordination with shared memory banks. Multiple agents read/write the same bank; coherence maintained via versioned writes.

### 2.4 Storage Backend

Hindsight supports pluggable backends:
- **Default**: in-memory (zero infra)
- **Persistent**: PostgreSQL + pgvector, Qdrant, Weaviate, Pinecone
- **Hosted**: Vectorize cloud (managed)

### 2.5 LLM Provider Configuration

```yaml
memory:
  llm:
    provider: openai  # or anthropic, azure, together, fireworks
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
```

# Supported providers
# openai, anthropic, azure, together, fireworks, litellm

## 3. The Three Operations in Practice

### Retain

```python
from hindsight import Hindsight

hd = Hindsight(api_key="...")
hd.retain(
    messages=[{"role": "user", "content": "I prefer dark mode"}],
    user_id="alice"
)
```

### Recall

```python
memories = hd.recall(
    query="user interface preferences",
    user_id="alice"
)
# Returns: [{"memory": "Prefers dark mode", "score": 0.94, ...}]
```

### Reflect

```python
hd.reflect(user_id="alice")  # background; usually scheduled
# Consolidates memories, resolves contradictions, updates summaries
```

### LLM Wrapper (2-line integration)

```python
from hindsight import HindsightLLM

llm = HindsightLLM(base_llm=openai_client, user_id="alice")
response = llm.chat(messages)  # retain + recall happen automatically
```

# All subsequent LLM calls automatically retain/recall memories

## 4. Benchmark Results

### 4.1 LongMemEval (S) — Per-Category Breakdown

| Category | Score |
|---|---|
| Single-session QA | 96.2% |
| Cross-session QA | 93.1% |
| Temporal reasoning | 92.4% |
| Preference tracking | 97.8% |
| Entity tracking | 94.6% |
| **Overall (S)** | **94.6%** |

### 4.2 LoCoMo — Full Table

LoCoMo10 (10-session) results from the AMB leaderboard (self-reported by Vectorize):

| System | F1 | BLEU-1 | ROUGE-1 | METEOR | **Avg** |
|---|---|---|---|---|---|
| **Hindsight (GPT-4o-mini)** | 94.33 | 88.77 | 96.26 | 91.80 | **92.79%** |
| **Hindsight (GPT-4o)** | 94.14 | 87.00 | 97.19 | 91.05 | **92.35%** |
| **Hindsight (Claude 3.5 Sonnet)** | 93.60 | 87.29 | 96.59 | 89.48 | **91.74%** |
| **Hindsight (OSS-20B)** | 95.7% | 66.7% | 84.6% | 79.7% | 79.7% | **83.6%** |
| **Hindsight (OSS-120B)** | — | — | — | — | — | **89.0%** |
| **Hindsight (Gemini-3 Pro)** | — | — | — | — | — | **91.4%** |

### 4.3 Agent Memory Benchmark (AMB) — All Datasets

| Dataset | Hindsight Score | Notes |
|---|---|---|
| LoCoMo10 (GPT-4o-mini) | 92.79% | #1 on AMB |
| LoCoMo10 (GPT-4o) | 92.35% | |
| LongMemEval-S | 94.6% | #1 on AMB |
| BEAM 100K | 73.4% | #1 on AMB; corrected from 75% |
| BEAM 10M | 64.1% | Next best: 40.6% |

### 4.4 BEAM Benchmark Deep Dive

| System | BEAM 100K | BEAM 500K | BEAM 1M | BEAM 10M | **Avg (BEAM)** |
|---|---|---|---|---|---|
| **Hindsight** | 73.4% | — | — | 64.1% | — |
| **Honcho** | 63.0% | — | — | 40.9% | — |
| **Mem0 v3** | — | — | 64.1% | 48.6% | — |
| **Hindsight (OSS-20B)** | 74.11 | 64.58 | 90.96 | 76.32 | **83.18%** |
| **Hindsight (OSS-120B)** | 76.79 | 62.50 | 93.68 | 79.44 | **85.67%** |
| **Hindsight (Gemini-3 Pro)** | 86.17 | 70.83 | 95.12 | 83.80 | **89.61%** |

## 5. Benchmark Validity Assessment

### 5.1 LongMemEval — Most Credible

- Independent benchmark (not run by Vectorize)
- Standardized eval harness
- Multiple systems benchmarked
- Hindsight score: 94.6% (self-run against public benchmark)

### 5.2 LoCoMo — Questionable

- AMB leaderboard run by Vectorize (conflict of interest)
- Other systems' LoCoMo scores come from their own papers, not AMB
- Comparison is apples-to-oranges: different LLMs, configs, dates

| System | LoCoMo Score | Source | LLM Used |
|---|---|---|---|
| ByteRover | 96.1% | byterover.dev (self-pub, v2.1.5, Mar 2026) | Not disclosed |
| EverOS | 93.05% | evermemory.ai (self-pub, Feb 2026) | Not disclosed |
| MemOS | 92.34% | github.com/MemTensor/MemOS (self-pub, Jul 2026) | Not disclosed |
| Mem0 v3 | 92.5% | mem0.ai/research (self-pub, Apr 2026) | Not disclosed |
| Hindsight | 92% | AMB (Vectorize, conflict of interest) | GPT-4o-mini |
| Memori | 81.95% | memorilabs (self-pub) | Not disclosed |
| Honcho | 89.9% | Honcho blog (LLM-judge, not standard metrics) | Not disclosed |
| MemU | 92.09% | self-pub | Not disclosed |
| Letta | 74.0% | EMNLP 2025 paper (Letta Filesystem) | gpt-4o-mini |
| Zep | 75% | disputed config | Not disclosed |
| A-MEM | ~60% | NeurIPS 2025 paper | Not disclosed |
| LoCoMo (AMB) | Independent dataset | Partial | Medium | ⚠️ Medium — Hindsight itself disclaims |

### 5.3 AMB (Agent Memory Benchmark) — Conflict of Interest

- Built and run by Vectorize (Hindsight's creator)
- Hindsight tops every dataset it's on
- Cannot be treated as independent validation

### 5.4 Summary Validity Table

| Benchmark | Independence | Hindsight Score | Caution Level |
|---|---|---|---|
| LongMemEval-S | High — Stanford/independent | 94.6% | Low |
| BEAM 100K | High — ICLR 2026 paper | 73.4% | Low |
| BEAM 10M | High | 64.1% | Low |
| LoCoMo (AMB) | Low — Vectorize-run | 92% | High |

## 6. Paper Summary

Hindsight does not have an arxiv paper. The primary references are:
- [Vectorize blog: Introducing Hindsight](https://hindsight.vectorize.io/blog/introducing-hindsight)
- [BEAM benchmark paper](https://arxiv.org/abs/2503.24129)
- [AMB leaderboard](https://agentmemory.fyi)

## 7. Sources

- [hindsight.vectorize.io](https://hindsight.vectorize.io)
- [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)
- [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io)
- [agentmemory.fyi](https://agentmemory.fyi)
- [vectorize.io/blog/introducing-hindsight](https://vectorize.io/blog/introducing-hindsight-agent-memory-that-works-like-human-memory)