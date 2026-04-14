# AI Memory Systems Landscape

> Research survey of the AI memory space — compiled from papers, GitHub
> READMEs, and public benchmarks. Karta is an experimental research project
> pursuing its own approach; this document is background research, not
> positioning material. Benchmark numbers are self-published unless noted
> otherwise and should be treated as claims until independently reproduced.
>
> Last compiled: 2026-04-14.

## Legend

- Benchmark scores are self-published unless noted otherwise. Treat with skepticism until independently reproduced.
- "—" means not reported or not benchmarked.
- BEAM has multiple tiers (100K / 500K / 1M / 10M). The matrix reports whichever tier each system publishes; Honcho's headline is the full 10M scale, most others report 100K.

## Competitive Matrix

| System | BEAM 100K | LOCOMO | LongMemEval-S | Memory Model | Write-Time Organization | Retrieval | Dreaming / Active Inference | Forgetting | Episode Segmentation | Foresight / Forward-Looking | Auditability | Language / Stack | Operational Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Karta** *(ours, experimental)* | 61.6% (P1, 2026-04-14, N=399) | — | 50.0% (temporal-reasoning only, 50 qs, oracle split) | Zettelkasten linked notes + async dream pass (5 types) + episodes + foresight signals | LLM attributes + semantic linking + retroactive evolution + turn_index ordering | Cosine → Jina cross-encoder rerank → multi-hop BFS → chronological sort → structured synthesis w/ reasoning | Yes — deduction, induction, abduction, consolidation, contradiction; incremental cursor; cross-cluster sliding windows | Planned (Phase 3) — NoteStatus lifecycle, access-based decay, dream-driven deprecation | In progress (Phase 2B) — boundary detection + narrative synthesis, wiring next | In progress (Phase 2B) — ForesightSignal with validity windows, abduction-generated | Very high — every note, link, evolution, dream is auditable JSON in-process | Rust; LanceDB + SQLite embedded; zero infra | Low — in-process library, no Docker |
| **Honcho** | 63.0% (100K) / SOTA at 10M | 89.9% *(LLM-judge)* | 88.8% (LongMem-M) | Peer Representations per user/agent; formal logic-based; v3 architecture | Dialectic agentic loop over full history | Agentic loop — full history re-processing per query; external vector store option (v3.0.4) | Yes — async dreaming: deduction, induction, abduction, summaries, peer cards | Not exposed | Not documented | Not documented | Low — dreaming chain-of-thought is opaque | Python; hosted API ($2/M tokens ingested); MCP server (v3); v3.0.6 RC (Apr 10); self-hosting now provider-agnostic (any OpenAI-compatible endpoint) | Medium-high — dreaming charged separately |
| **EverMemOS** | — | 93.05% *(self-published, Feb 2026 product release; beats full-context; +19.7% multi-hop, +16.1% temporal vs prior SOTA)* | 83.00% | Engram lifecycle: encode → consolidate → forget; event-boundary segmentation | Event-boundary segmentation at write time | Dual-mode: fast embed + multi-hop agentic reasoning | Yes — consolidation, lifecycle transitions, active forgetting | Yes — core design principle; intelligent attention filter | Yes — MemCell event-boundary segmentation is their core innovation | Not documented | Low — HTTP service, internals opaque | Python + Docker + Vectorize.io; REST API; EverMemOS Cloud launched Feb 4, 2026 | High — separate service with own infra; Cloud option reduces ops |
| **MemU** | — | 92.09% *(self-published)* | — | File-system hierarchy (folders = topics, files = facts); dual-mode retrieval | Topic-folder organization at write time | Fast embedding for monitoring + LLM reasoning for queries | Yes — proactive background monitoring, fires agent actions autonomously | Not documented | Not documented | Yes — proactive monitoring fires predictions/actions | Low — proactive loop is internal | Python; PostgreSQL + pgvector | Medium — always-on daemon, Docker |
| **Zep / Graphiti** | — | 75% *(disputed config)* | — | Temporal knowledge graph; episodic time-indexed nodes | Entity + relationship extraction into temporal graph | Graph traversal + embedding hybrid | None | Episodic decay by recency | Implicit — temporal graph nodes have time boundaries | Not documented | Medium — graph inspectable, prompts less so | Python; hosted or self-host Docker | Medium |
| **A-MEM** | — | ~60% *(NeurIPS 2025 paper)* | — | Zettelkasten linked notes with context + keywords; retroactive evolution | LLM attributes + semantic linking + retroactive evolution | Cosine search → link graph traversal → LLM synthesis | None in paper | None — accumulates indefinitely | None | None | Very high — every note, link, evolution is inspectable JSON | Python pip | Low — in-process library |
| **Mem0** | — | 66.9% *(arxiv 2504.19413, Apr 2026; Mem0g graph variant: 68.4%)*; 26% relative gain over OpenAI Memory | — | Flat key-value facts, vector-indexed; graph variant (Mem0g) builds directed labeled KG alongside vector store | Minimal — fact extraction only | Embedding similarity; reranker support (Cohere, ZeroEntropy, HF, Sentence Transformers, LLM-based) added v1.0.0; Mem0g adds graph traversal | None | Manual or TTL | None | None | High — simple API, no hidden logic | Python SDK; REST API; Skill Graph w/ Vercel AI SDK (Apr 2026) | Low — hosted API |
| **Letta / MemGPT** | — | 74.0% *(Letta Filesystem; gpt-4o-mini; stores history as files + semantic search + grep)* | — | In-context + external paged memory; agent manages its own state; Letta Filesystem (2026): raw files as memory | Agent self-organizes via tool calls (expensive, non-deterministic); Filesystem: full history in files | Agent self-retrieves via tool calls; Filesystem: semantic search + grep over stored files | Agent self-reflects (expensive, full agent turn per reflection) | Agent-controlled context eviction | None — flat paged memory | None | Medium — agent actions logged | Python (open-source); v0.16.7 (Mar 31); Letta Code desktop app (Apr 6) — memory-first coding agent, #1 model-agnostic on Terminal-Bench | High — full agent runtime per query |
| **Hindsight** *(Vectorize)* | **73.4%** (Apr 2 2026 blog; v0.4.19; #1 on AMB leaderboard; was cited as rounded 75% earlier) | **92%** *(LoComo10, #1 on AMB)* | **94.6%** *(LongMemEvalS, #1 on AMB)* | "Retain, Recall, Reflect" — structured queryable memory bank built from conversation streams; Reflector layer adds cross-session reasoning | Append-only conversational extraction → structured memory bank | Semantic + BM25 hybrid retrieval + Reflector reasoning layer | Yes — Reflector agent reasons across memories; v0.4.19 claims #1 across all AMB datasets | Not documented | Not documented | Not documented | Medium — Reflector chain-of-thought logged but internals semi-opaque | Python; open-source (github.com/vectorize-io/hindsight); hosted API | Low-medium — hosted or self-host |
| **MemoryOS** *(BAI-LAB)* | — | +49.11% F1 / +46.18% BLEU-1 improvement vs baseline *(EMNLP 2025 Oral; absolute scores not reported vs competitors)* | — | OS-inspired hierarchical storage: short-term → mid-term (heat-gated promotion) → long-term persona; automated user profile; per-user memory | Heat-gated promotion: interactions accumulate "heat", high-heat segments trigger profile updates | Short-term + mid-term + profile retrieved per query | None — lifecycle promotions are consolidation but no explicit dream pass | Yes — mid-term promotion is implicit consolidation; soft retirement | None | Not documented | Medium — structured per-user storage, MCP server available | Python; SQLite + local vectors (BGE-M3, Chroma) | Low — in-process library, Docker available |
| **Mastra (Observational Memory)** | — | — | 84.23% (gpt-4o) / 94.87% (gpt-5-mini) — **highest LongMemEval score ever recorded** | Observer + Reflector background agents maintain dense observation log replacing raw message history | Append-only observation log — Observer watches conversations, extracts structured observations | Observations injected into context window; no explicit retrieval — stable, prompt-cacheable context | Yes — Reflector agent performs background reasoning over observations | Not documented | Not documented | Not documented | Medium — observation log is inspectable but agent internals less so | TypeScript; open-source (Mastra framework) | Low-medium — 5-40x compression, high prompt cache hit rates reduce token costs 4-10x |
| **Supermemory** | — | — | 85.4% *(production engine)* / ~99% *(experimental ASMR, not production)* | Hybrid: production uses vector + structured retrieval; experimental ASMR uses parallel LLM agents reading raw history | Production: structured extraction; ASMR: no vector DB, parallel LLM agents search raw history | Production: vector + structured; ASMR: agentic search + memory retrieval (parallel LLM reasoning) | ASMR experimental flow uses multi-agent reasoning | Not documented | Not documented | Not documented | Low — proprietary service | TypeScript; hosted API | Medium — hosted service; ASMR is very expensive (parallel LLM reads) |
| **OMEGA** | — | — | **95.4%** *(466/500; #1 on LongMemEval leaderboard, local-first, pip install)* | Persistent memory for AI coding agents; local-first, session-spanning | Not documented | Semantic retrieval, local inference | Not documented | Not documented | Not documented | Not documented | Low — pip install, local-only | Python; local-first, no cloud retrieval | Very low — fully local, M1 MacBook tested |
| **MemMachine** | — | **91.69%** *(gpt-4.1-mini; arxiv 2604.04853, Apr 2026; above Mem0, Zep, Memobase, LangMem, OpenAI baseline)* | **93.0%** *(LongMemEvalS overall)* | Ground-truth-preserving memory; universal memory layer; scalable extensible interoperable | Retrieval-stage optimized: depth tuning, context formatting, search prompt design | Multi-stage: retrieval depth tuning (+4.2%), context formatting (+2.0%), search prompt design (+1.8%), query bias correction (+1.4%) | Not documented | Not documented | Not documented | Not documented | Medium | Python; open-source (github.com/MemMachine/MemMachine) | Low — ~80% fewer input tokens than Mem0 |
| **Memori** *(MemoriLabs)* | — | 81.95% *(at 4.98% cost of full context; outperforms Zep, LangMem, Mem0)* | — |
| **MemPalace** | **49.0%** *(independently tested, raw mode + GPT-5.4-mini synthesis)* | — | **96.6%** R@5 *(self-reported, raw mode, retrieval recall only; 100% with rerank is teaching-to-test on 3 specific questions, held-out: 98.4%)* | Verbatim text in ChromaDB, no write-time reasoning | None, raw storage only | ChromaDB cosine search; hybrid keyword boost + temporal boost in advanced modes | None | None | None | None | Low, inspectable ChromaDB store | Python; ChromaDB only; zero infra | Very low, no LLM for storage | SQL-native structured memory: facts, preferences, rules, summaries; intelligent decay + ranking | Classifies each chat turn into facts/preferences/rules/summaries; continuous extraction | SQL-native retrieval with intelligent ranking and decay; avg 1,294 tokens per query | None | Yes — intelligent decay and ranking built-in | None | Not documented | Medium — SQL-native, inspectable structured state | Python; SQL-native, LLM/datastore-agnostic; Cloud + BYODB + on-prem | Low-medium — Memori Cloud hosted; very token-efficient |

## New Benchmarks and Papers (2026)

| Benchmark / Paper | What It Tests | Key Scores Found | Source |
|---|---|---|---|
| **HaluMem** *(MemTensor, 2025)* | First operation-level hallucination eval for agent memory systems | EverMemOS: 90.04% | [github.com/MemTensor/HaluMem](https://github.com/MemTensor/HaluMem) |
| **Agent Memory Benchmark (AMB)** *(Vectorize, Mar 2026)* | Tests 8 architectures: Letta, Cognee, Graphiti, Tacnode, Mem0, Hindsight, EverMemOS, Hyperspell | Scores not yet scraped; live at agentmemorybenchmark.ai | [Vectorize manifesto](https://hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark) |
| **Graph-Native Cognitive Memory** *(arxiv, Mar 2026)* | Formal belief revision semantics for versioned memory; graph-based fact invalidation | — theory paper | [arxiv 2603.17244](https://arxiv.org/html/2603.17244v1) |
| **Multi-Layered Memory Architectures** *(arxiv, Mar 2026)* | Experimental eval of long-term context retention across architectures | — experimental | [arxiv 2603.29194](https://arxiv.org/html/2603.29194) |
| **MemFactory** *(arxiv, Apr 2026)* | First unified modular training + inference framework for memory agents; Lego-like plug-and-play components; GRPO fine-tuning; supports Memory-R1, RMM, MemAgent | Up to 14.8% relative gains over base models | [arxiv 2603.29493](https://arxiv.org/abs/2603.29493) |
| **AgeMem: Agentic Memory** *(arxiv, Jan 2026)* | RL-based unified LTM + STM management; memory operations as tool-based actions (store, retrieve, update, summarize, discard) | — | [arxiv 2601.01885](https://arxiv.org/abs/2601.01885) |
| **A-MAC: Adaptive Memory Admission Control** *(arxiv, Mar 2026)* | Treats memory admission as structured decision problem, not implicit byproduct of generation | — | [arxiv 2603.04549](https://arxiv.org/abs/2603.04549) |
| **LifeBench** *(arxiv, Mar 2026)* | Long-horizon multi-source memory benchmark | — | [arxiv 2603.03781](https://arxiv.org/html/2603.03781) |
| **ICLR 2026 MemAgents Workshop** | Workshop on Memory for LLM-Based Agentic Systems; accepted at ICLR 2026 | — | [OpenReview](https://openreview.net/pdf?id=U51WxL382H) |
| **MemoryAgentBench** *(ICLR 2026, HUST-AI-HYZ)* | Four competencies for memory agents: accurate retrieval, test-time learning, long-range understanding, and **selective forgetting**; two new datasets: EventQA + FactConsolidation; ICLR 2026 accepted; reveals current methods fall short on all 4 competencies simultaneously | — | [arxiv 2507.05257](https://arxiv.org/abs/2507.05257), [github.com/HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) |
| **AMA-Bench** *(arxiv, Feb 2026)* | Evaluates long-horizon memory for agentic applications | — | [arxiv 2602.22769](https://arxiv.org/html/2602.22769) |
| **Mem0 paper** *(arxiv 2504.19413, Apr 2026)* | First formal paper on Mem0; LOCOMO evaluation: 66.9% (base), 68.4% (Mem0g graph); 26% relative gain over OpenAI Memory; 91% faster + 90% fewer tokens vs full-context; reranker layer support | Mem0: 66.9% LOCOMO; Mem0g: 68.4% LOCOMO | [arxiv 2504.19413](https://arxiv.org/abs/2504.19413) |
| **CAST: Character-and-Scene Episodic Memory** *(arxiv 2602.06051, Feb 2026)* | 3D scene-based episodic memory (time/place/topic) organized into character profiles; inspired by dramatic theory | +8.11% F1, +10.21% J(LLM-Judge) improvement over baselines, especially on time-sensitive conversational questions | [arxiv 2602.06051](https://arxiv.org/abs/2602.06051) |
| **Memoria** *(arxiv 2512.12686)* | Modular memory framework: dynamic session-level summarization + weighted knowledge graph for user modeling | — | [arxiv 2512.12686](https://arxiv.org/abs/2512.12686) |
| **Memory for Autonomous LLM Agents** *(arxiv 2603.07670, Mar 2026)* | Survey formalizing agent memory as write-manage-read loop; 3D taxonomy (temporal scope, representational substrate, control policy); 5 mechanism families | — survey | [arxiv 2603.07670](https://arxiv.org/html/2603.07670v1) |
| **Memory in the LLM Era** *(arxiv 2604.01707, Apr 2026)* | Modular architectures and strategies in unified framework; experiment + analysis + benchmark | — | [arxiv 2604.01707](https://arxiv.org/html/2604.01707) |
| **Mastra Observational Memory** *(research, 2026)* | Observer + Reflector agents with stable context windows; 5-40x compression; prompt-cacheable | 84.23% LongMemEval (gpt-4o), 94.87% (gpt-5-mini) — SOTA | [mastra.ai/research](https://mastra.ai/research/observational-memory) |
| **Supermemory ASMR** *(experimental, 2026)* | Agentic Search and Memory Retrieval — parallel LLM agents reading raw history, no vector DB | ~99% LongMemEval-S (experimental, not production) | [supermemory.ai/blog](https://supermemory.ai/blog/we-broke-the-frontier-in-agent-memory-introducing-99-sota-memory-system/) |
| **MemMachine** *(arxiv 2604.04853, Apr 2026)* | Ground-truth-preserving memory system; retrieval-stage optimization > ingestion-stage; GPT-5-mini outperforms GPT-5 with optimized prompts | LoCoMo: 91.69% (gpt-4.1-mini); LongMemEvalS: 93.0%; ~80% fewer tokens than Mem0 | [arxiv 2604.04853](https://arxiv.org/abs/2604.04853) |
| **SuperLocalMemory V3.3** *(arxiv 2604.04514, Apr 2026)* | Biologically-inspired forgetting, cognitive quantization, multi-channel retrieval; zero-LLM agent memory | — | [arxiv 2604.04514](https://arxiv.org/html/2604.04514) |
| **BEAM (ICLR 2026)** | Official paper accepted at ICLR 2026; 100 conversations, 100K–10M tokens, 2000 probing questions; 10M scale where context stuffing is physically impossible | Hindsight SOTA at 64.1% (10M), next best 40.6% | [github.com/mohammadtavakoli78/BEAM](https://github.com/mohammadtavakoli78/BEAM), [arxiv 2510.27246](https://arxiv.org/html/2510.27246) |

## Notable New / Emerging Systems

| System | Description | Status | Source |
|---|---|---|---|
| **MemOS** *(MemTensor)* | AI memory OS for LLM agents; persistent skill memory, cross-task reuse, hybrid FTS5+vector search; OpenClaw Cloud + Local plugins (Mar 2026) | Active, open-source | [github.com/MemTensor/MemOS](https://github.com/MemTensor/MemOS) |
| **MemoryOS** *(BAI-LAB, EMNLP 2025)* | OS-inspired hierarchical memory (short→mid→long); heat-gated promotion; MCP server; +49% F1 on LoCoMo vs baseline | Active, open-source | [github.com/BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) |
| **Cognee** | Knowledge graph + vector + relational 3-store architecture; 14 retrieval modes; strong multi-hop reasoning | Active, v0.5.2 | [cognee.ai](https://www.cognee.ai/research-and-evaluation-results) |
| **OpenViking** *(volcengine)* | Open-source context database; filesystem paradigm for memory/resources/skills unified management | Active, launched Jan 2026 | [github.com/volcengine/OpenViking](https://github.com/volcengine/OpenViking) |
| **Mastra Observational Memory** | Observer + Reflector background agents; stable append-only context; 5-40x compression; SOTA LongMemEval (94.87% gpt-5-mini) | Active, open-source (TypeScript) | [mastra.ai/docs/memory/observational-memory](https://mastra.ai/docs/memory/observational-memory) |
| **Supermemory** | Hosted AI memory API; production engine (85.4% LongMemEval-S); experimental ASMR flow (~99% via parallel LLM agents, no vector DB) | Active, hosted service | [supermemory.ai/research](https://supermemory.ai/research/) |
| **Memori** *(MemoriLabs)* | SQL-native memory infra; facts/preferences/rules/summaries; intelligent decay; 81.95% LoCoMo at 4.98% cost; Cloud + OpenClaw plugin (Mar 2026) | Active, Cloud launched Mar 2026 | [github.com/MemoriLabs/Memori](https://github.com/MemoriLabs/Memori) |
| **MemFactory** *(arxiv)* | Unified modular training + inference framework; Lego-like components; GRPO fine-tuning; supports Memory-R1/RMM/MemAgent | Research, open-source | [arxiv 2603.29493](https://arxiv.org/abs/2603.29493) |
| **Hindsight** *(Vectorize)* | "Retain, Recall, Reflect" architecture; Reflector background agent reasons over structured memory bank; SOTA across BEAM + LOCOMO + LongMemEval simultaneously (Apr 2026); BEAM 100K 73.4%, LOCOMO 92%, LongMemEval-S 94.6%; BEAM 10M 64.1% (next best: 40.6%) | Active, open-source + hosted | [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight), [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io) |
| **Engram** *(engram.fyi)* | MCP memory server for Claude Code, Cursor, and AI coding agents; episodic + semantic + procedural memory via single router/retriever; single binary + SQLite, npm install; 80% LOCOMO at 96.6% fewer tokens; 2.5K installs; ENGRAM paper at arxiv 2511.12960 | Active, open-source | [engram.fyi](https://www.engram.fyi/), [arxiv 2511.12960](https://arxiv.org/abs/2511.12960) |
| **MemPalace** *(milla-jovovich/mempalace)* | Verbatim storage + ChromaDB semantic search; "palace" spatial metaphor (wings/rooms/halls); AAAK compression dialect (experimental, lossy); 96.6% LongMemEval R@5 raw mode (no LLM); 10K GitHub stars; no write-time reasoning or active inference; BEAM 100K: 49% (independently tested) | Active, open-source | [github.com/milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace) |
| **Hermes Agent** *(Nous Research)* | Open-source autonomous AI agent (Feb 2026) with 7 pluggable memory providers: Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover; memory provider ecosystem aggregator; auto-migrates configs. Hindsight now native provider (Apr 6). Setup wizard for memory config. | Active, open-source | [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), [hindsight.vectorize.io/blog/2026/04/06/hermes-native-memory-provider](https://hindsight.vectorize.io/blog/2026/04/06/hermes-native-memory-provider) |
| **Anthropic Auto Dream** *(Mar 2026)* | Claude Code's functional memory consolidation — first major AI lab treating agent memory as cognitive architecture problem, not storage problem. Sessions become cumulative. | Shipped in Claude Code (Mar 2026) | [dev.to/max_quimby](https://dev.to/max_quimby/ai-agent-memory-in-2026-auto-dream-context-files-and-what-actually-works-39m8) |
| **Neo4j Agent Memory** *(neo4j-labs)* | Graph-native memory system for AI agents backed by Neo4j; stores conversations, builds knowledge graphs, agents learn from own reasoning | Active, open-source, launched Jan 2026 | [github.com/neo4j-labs/agent-memory](https://github.com/neo4j-labs/agent-memory) |
| **ALMA** *(zksha)* | Automated meta-Learning of Memory designs for Agentic systems — meta-learns memory designs to replace human-engineered designs | Research, open-source, Jan 2026 | [github.com/zksha/alma](https://github.com/zksha/alma) |
| **MemMachine** | Universal ground-truth-preserving memory layer; LoCoMo 91.69%, LongMemEvalS 93.0%; ~80% fewer tokens than Mem0; retrieval-stage optimization dominates | Active, open-source, arxiv Apr 2026 | [github.com/MemMachine/MemMachine](https://github.com/MemMachine/MemMachine), [arxiv 2604.04853](https://arxiv.org/abs/2604.04853) |
| **OMEGA** | Local-first persistent memory for AI coding agents; 95.4% LongMemEval (#1 leaderboard); pip install; no cloud | Active, pip-installable | [omegamax.co](https://omegamax.co/benchmarks) |

## Change Log

| Date | Change | Source |
|---|---|---|
| 2026-04-02 | EverMemOS LOCOMO updated 92.3% → 93.05%; LongMemEval 82% → 83.00%; HaluMem 90.04% added; Feb 2026 product release noted | [evermind.ai/blogs/evermemos-hits-sota-performance-on-locomo](https://evermind.ai/blogs/evermemos-hits-sota-performance-on-locomo) |
| 2026-04-02 | Honcho updated: v3.0.4 RC released (2026-04-02); MCP server improvements (inspect_workspace, list_workspaces tools); external vector store for search | [github.com/plastic-labs/honcho commits](https://github.com/plastic-labs/honcho/commits/main) |
| 2026-04-02 | Letta LOCOMO 74.0% added (Letta Filesystem: simple file storage + grep beats specialized tools) | [letta.com/blog/benchmarking-ai-agent-memory](https://www.letta.com/blog/benchmarking-ai-agent-memory) |
| 2026-04-02 | Added MemoryOS (BAI-LAB, EMNLP 2025 Oral) as new row: OS-inspired hierarchical memory, +49% F1 on LoCoMo, MCP server, Python/SQLite | [github.com/BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) |
| 2026-04-02 | Added new systems table: MemOS (MemTensor), MemoryOS (BAI-LAB), Cognee, OpenViking | GitHub search |
| 2026-04-02 | Added new benchmarks table: HaluMem, AMB (Vectorize), two arxiv papers (Mar 2026) | Web search |
| 2026-04-02 | Added Letta baseline threat note; LOCOMO table updated | — |
| 2026-04-02 | Mem0: very active development (Codex plugin support, CLI improvements, purple branding); v1.0.3 current | [github.com/mem0ai/mem0 commits](https://github.com/mem0ai/mem0/commits/main) |
| 2026-04-03 | Added Mastra Observational Memory: SOTA LongMemEval 84.23% (gpt-4o) / 94.87% (gpt-5-mini); Observer+Reflector agents, stable context, 5-40x compression, open-source TypeScript | [mastra.ai/research/observational-memory](https://mastra.ai/research/observational-memory), [VentureBeat coverage](https://venturebeat.com/data/observational-memory-cuts-ai-agent-costs-10x-and-outscores-rag-on-long) |
| 2026-04-03 | Added Supermemory: 85.4% LongMemEval-S (production), ~99% experimental ASMR (parallel LLM agents, no vector DB) | [supermemory.ai/research](https://supermemory.ai/research/), [supermemory.ai/blog](https://supermemory.ai/blog/we-broke-the-frontier-in-agent-memory-introducing-99-sota-memory-system/) |
| 2026-04-03 | Added Memori (MemoriLabs): 81.95% LoCoMo at 4.98% cost of full context; SQL-native; Cloud + OpenClaw plugin launched Mar 2026 | [github.com/MemoriLabs/Memori](https://github.com/MemoriLabs/Memori), [prweb.com](https://www.prweb.com/releases/memori-labs-outperforms-every-other-memory-system-with-81-95-accuracy-at-4-98-the-cost-of-full-context-302719406.html) |
| 2026-04-03 | Added new papers: MemFactory (2603.29493), AgeMem (2601.01885), A-MAC (2603.04549), LifeBench (2603.03781), ICLR 2026 MemAgents Workshop | arxiv, OpenReview |
| 2026-04-03 | Honcho: v3.0.4 RC (Apr 2) — external vector store for message search, dialectic connection fix, MCP server improvements continue | [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho) |
| 2026-04-03 | Letta: v0.16.7 (Mar 31) — summarizer prompt now remembers plan files/GH PRs/ticket IDs; memfs fix; using "Letta Code" AI for commits | [github.com/letta-ai/letta](https://github.com/letta-ai/letta) |
| 2026-04-03 | Mem0: OpenClaw v1.0.1 released, Codex plugin support added, CLI improvements, security scanner fixes | [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) |
| 2026-04-03 | EverMemOS: attempted rename to EverOS (Mar 29), reverted same day (Mar 30); Memory Sparse Attention docs added; OpenClaw plugin refinements | [github.com/EverMind-AI/EverMemOS](https://github.com/EverMind-AI/EverMemOS) |
| 2026-04-03 | Updated gap analysis: added LongMemEval trailing note, context efficiency dimension, Memori to LoCoMo list | Web search |
| 2026-04-03 | Initial table created from project findings and CLAUDE.md | Manual compilation |
| 2026-04-06 | **Added Hindsight (Vectorize) to matrix** — new BEAM 100K leader at 75% (since corrected to 73.4%), LOCOMO 92% (#1 on AMB), LongMemEval-S 94.6% (#1 on AMB). BEAM 10M 64.1% — next best is 40.6%. Open-source Python + hosted API. | [hindsight.vectorize.io/blog/2026/04/02/beam-sota](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota), [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io) |
| 2026-04-06 | **Mem0 formal paper published** — arxiv 2504.19413 (Apr 2026). LOCOMO score updated 67% → 66.9% (base); Mem0g graph variant: 68.4%. 26% relative gain over OpenAI Memory (52.9%). Reranker support added v1.0.0. | [arxiv 2504.19413](https://arxiv.org/abs/2504.19413) |
| 2026-04-06 | Mem0 Apr 4–6 activity: Skill Graph feature with dedicated CLI + Vercel AI SDK skills (Apr 6); AGENTS.md for AI coding agent instructions; OpenClaw agent plugin fix; groq model fix | [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) |
| 2026-04-06 | Honcho Apr 3 activity: retry on more httpx exceptions (#467); explicit rollback in transactions (#486); OpenClaw integration docs improved (#462) — v3.0.4 RC still in progress | [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho) |
| 2026-04-06 | Letta: no new releases since v0.16.7 (Mar 31). Most recent commit: Mar 31. Stable. | [github.com/letta-ai/letta](https://github.com/letta-ai/letta) |
| 2026-04-06 | **Added MemoryAgentBench (ICLR 2026)** to papers table — four memory competencies including selective forgetting; EventQA + FactConsolidation datasets; current methods fall short on all 4 simultaneously | [arxiv 2507.05257](https://arxiv.org/abs/2507.05257) |
| 2026-04-06 | **Added AMA-Bench** (arxiv 2602.22769) to papers table — long-horizon memory benchmark for agentic applications | [arxiv 2602.22769](https://arxiv.org/html/2602.22769) |
| 2026-04-06 | **Added Engram** to notable systems — MCP memory server, 80% LOCOMO at 96.6% fewer tokens, single binary + SQLite, 2.5K installs, coding-agent focused | [engram.fyi](https://www.engram.fyi/) |
| 2026-04-06 | **Added Hermes Agent** (Nous Research) to notable systems — 7-provider pluggable memory aggregator (Honcho, OpenViking, Mem0, Hindsight, etc.), launched Feb 2026 | [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) |
| 2026-04-06 | Hindsight noted as new BEAM 100K leader (originally 75%, since corrected to 73.4%). Updated LOCOMO/LongMemEval lists to include Hindsight and Engram. | Web search |
| 2026-04-07 | **Added MemPalace** to competitive matrix. Independently ran BEAM 100K: 49.0% (20 convs, 400 qs, GPT-5.4-mini synthesis + judge). Raw ChromaDB retrieval + LLM synthesis. Strong on preference following (80%), weak on contradiction (40%), summarization (35%), event ordering (32%). Filed issue + PR on their repo (milla-jovovich/mempalace#125, #168). Code audit: no answer leakage but LoCoMo 100% is top-k > corpus (trivial), LongMemEval 100% is 3 targeted fixes for 3 failing questions (held-out: 98.4%). | Independent benchmarking |
| 2026-04-07 | **Added MemoryBench (supermemoryai/memorybench)** — pluggable benchmark framework with 11 datasets, MemScore triple (accuracy/latency/tokens), existing baselines for Mem0/Zep/Supermemory. | Research |
| 2026-04-08 | Honcho Apr 6-8 activity: Zo Computer memory skill integration (#495); self-hosting docs overhaul — now provider-agnostic, any OpenAI-compatible endpoint (#510); transaction scope tightening for performance (#525); v3 API path | [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho) |
| 2026-04-08 | Mem0 Apr 6-8 activity: ChatDev integration guide (#4751); guard against LLM-hallucinated IDs in temp_uuid_mapping (#4674); Azure OpenAI + DeepSeek response_format forwarding (#4689, #4688); AGENTS.md added | [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) |
| 2026-04-08 | Letta Apr 7-8 activity: anti-spam issue guard with AI disclosure policy (adapted from Ghostty); Letta Code desktop app launched Apr 6 — memory-first coding agent, #1 model-agnostic on Terminal-Bench; v0.16.7 still latest release | [github.com/letta-ai/letta](https://github.com/letta-ai/letta), [letta.com/blog/introducing-the-letta-code-app](https://www.letta.com/blog/introducing-the-letta-code-app) |
| 2026-04-08 | **Hindsight now native Hermes Agent memory provider** (Apr 6 blog post). VentureBeat coverage: "91% accuracy" on LongMemEval. Growing ecosystem integration. | [hindsight.vectorize.io/blog/2026/04/06/hermes-native-memory-provider](https://hindsight.vectorize.io/blog/2026/04/06/hermes-native-memory-provider), [VentureBeat](https://venturebeat.com/data/with-91-accuracy-open-source-hindsight-agentic-memory-provides-20-20-vision) |
| 2026-04-08 | **EverMemOS Cloud launched** Feb 4, 2026 — production-grade cloud memory infra. "Memory Genesis 2026" hackathon with $80K+ prize pool, supported by OpenAI. | [manilatimes.net](https://www.manilatimes.net/2026/02/04/tmt-newswire/pr-newswire/end-the-agentic-amnesia-evermind-launches-evermemos-cloud-and-kicks-off-memory-genesis-2026-global-developer-hackathon-supported-by-openai/2271016) |
| 2026-04-08 | **Anthropic Auto Dream** noted as emerging system — Claude Code shipped functional memory consolidation (Mar 2026); first major AI lab treating agent memory as cognitive architecture. ETH Zurich found context files reduced task success in 5/8 tests. | [dev.to/max_quimby](https://dev.to/max_quimby/ai-agent-memory-in-2026-auto-dream-context-files-and-what-actually-works-39m8) |
| 2026-04-08 | Added 5 new papers: CAST episodic memory (2602.06051), Memoria framework (2512.12686), "Memory for Autonomous LLM Agents" survey (2603.07670), "Memory in the LLM Era" unified framework (2604.01707, Apr 2026) | arxiv |
| 2026-04-08 | **ICLR 2026 MemAgents workshop** and DeepLearning.AI Agent Memory course signal field mainstreaming. Spring AI AutoMemoryTools (Apr 7) brings agent memory to Java ecosystem. | [OpenReview](https://openreview.net/pdf?id=U51WxL382H), [spring.io](https://spring.io/blog/2026/04/07/spring-ai-agentic-patterns-6-memory-tools/), [DeepLearning.AI](https://www.deeplearning.ai/short-courses/agent-memory-building-memory-aware-agents/) |
| 2026-04-10 | Honcho v3.0.6 RC released (Apr 10); self-hosting docs overhaul merged (#510) — now fully provider-agnostic; Zo Computer memory skill integration (#495); transaction scope tightening for perf (#525) | [github.com/plastic-labs/honcho](https://github.com/plastic-labs/honcho) |
| 2026-04-10 | Mem0 Apr 9-10 activity: DeepSeek LLM support added (deepseek.ts + unit tests); OpenClaw plugin config refactored; SEO redirect chain fixes (~222 eliminated); session_id → run_id doc migration | [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) |
| 2026-04-10 | Letta: anti-spam issue guard merged (AI disclosure policy, adapted from Ghostty); v0.16.7 still latest release | [github.com/letta-ai/letta](https://github.com/letta-ai/letta) |
| 2026-04-10 | **Added MemMachine to competitive matrix and notable systems** — LoCoMo 91.69% (gpt-4.1-mini), LongMemEvalS 93.0%, ~80% fewer tokens than Mem0; ground-truth-preserving architecture; retrieval-stage optimization dominates ingestion-stage | [arxiv 2604.04853](https://arxiv.org/abs/2604.04853) |
| 2026-04-10 | **Added OMEGA to competitive matrix** — 95.4% LongMemEval (#1 leaderboard); local-first, pip install, no cloud; M1 MacBook tested with GPT-4.1 judge | [omegamax.co/benchmarks](https://omegamax.co/benchmarks) |
| 2026-04-10 | **Added Neo4j Agent Memory** (neo4j-labs) to notable systems — graph-native memory backed by Neo4j, launched Jan 2026 | [github.com/neo4j-labs/agent-memory](https://github.com/neo4j-labs/agent-memory) |
| 2026-04-10 | **Added ALMA** (zksha) to notable systems — meta-learning memory designs for agentic systems, replaces human-engineered designs | [github.com/zksha/alma](https://github.com/zksha/alma) |
| 2026-04-10 | **Added 3 new papers**: MemMachine (2604.04853), SuperLocalMemory V3.3 (2604.04514 — biologically-inspired forgetting, zero-LLM), BEAM ICLR 2026 acceptance | [arxiv](https://arxiv.org) |
| 2026-04-10 | Updated gap analysis: OMEGA now LongMemEval #1 (95.4%), MemMachine enters LoCoMo top tier (91.69%). LoCoMo field increasingly competitive with 4 systems above 90%. | Web search |
| 2026-04-10 | **Supermemory claims ~99% LongMemEval** via agent swarm approach; Hindsight BEAM 10M 64.1% confirmed as massive lead (next: 40.6%, 58% margin) | [aihola.com](https://aihola.com/article/supermemory-99-longmemeval-agentic-memory), [hindsight.vectorize.io](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota) |
| 2026-04-14 | **Hindsight BEAM 100K corrected 75% → 73.4%**: primary source is the Vectorize blog post dated 2026-04-02, which reports 73.4% as the hard number. The 75% figure in earlier entries was rounded from their dashboard. | [hindsight.vectorize.io/blog/2026/04/02/beam-sota](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota) |
| 2026-04-14 | **BEAM methodology note**: Honcho 63.0% and Hindsight 73.4% are both arithmetic mean of per-question BEAM rubric scores across 400 questions, single run, using BEAM's nugget-based LLM judge (0/0.5/1 per nugget, averaged). | BEAM paper arxiv 2510.27246; Honcho artifact `a1d689b/beam_100K_20251215_151001.json` |
| 2026-04-14 | **Reframed as research survey**: dropped "Competitive" framing; added header clarifying this is background research compiled from public information. Removed positioning content. | — |

---

# Appendix: Hindsight Deep Dive

> Research deep-dive on Hindsight compiled 2026-04-06 from the paper
> (arxiv 2512.12818), official benchmarks repo, and Vectorize blog posts.
> Retained as an appendix to the landscape survey.


> Research compiled: 2026-04-06. Based on: paper arxiv 2512.12818 (Dec 2025), GitHub README, official benchmarks repo, AMB manifesto, BEAM blog post, and coverage from VentureBeat, Open Source For You, and Vectorize team.

---

## 1. What Is It

**Paper**: "Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects"
**arXiv**: [2512.12818](https://arxiv.org/abs/2512.12818) — December 2025
**Authors**: Chris Latimer (CEO, Vectorize) + 6 co-authors including Andrew Neeser (Washington Post Applied ML) and Naren Ramakrishnan (Virginia Tech Sanghani Center for AI)
**Code**: [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) — MIT license
**Deployment**: Docker container (PostgreSQL inside) + Python/Node/REST SDKs
**Venue**: arXiv preprint — not yet accepted to a peer-reviewed conference as of Apr 2026

Hindsight positions itself as a "compositional memory architecture" that treats memory as a **structured substrate for factual, biographical, and opinionated reasoning**, not just a retrieval layer. The core claim is that existing systems either (a) lose temporal/entity coherence or (b) can't form evolving opinions — Hindsight is designed to do both.

---

## 2. Core Architecture

The system is built around **two main subsystems** (TEMPR and CARA) operating over **four epistemically-distinct memory networks** via **three public operations** (Retain, Recall, Reflect).

### 2.1 The Four Memory Networks

| Network | What It Stores | Epistemic Status |
|---|---|---|
| **World** | Objective facts about the external environment | Ground truth / factual |
| **Experience** | Agent's own actions and events, written in first-person | Biographical / observed |
| **Opinion** | Subjective judgments with evolving confidence scores | Inferred / dynamic |
| **Observation** | Preference-neutral entity summaries synthesized from underlying facts | Derived / consolidated |

The distinction matters: most systems (Mem0, A-MEM, Graphiti) mix these together in a flat fact store. Hindsight routes each piece of information to the correct epistemic bucket, which changes how it's retrieved and reasoned over. Opinions can be updated as confidence evolves; world facts are stable; experiences are immutable first-person records.

### 2.2 TEMPR — Temporal Entity Memory Priming Retrieval

TEMPR implements the **Retain** and **Recall** operations.

#### Retain Pipeline (Write Path)

```
Raw Input
  → LLM coarse chunking (narrative units, not sentence fragments)
  → Fact extraction (LLM — generates discrete facts per chunk)
  → Embedding generation (vector representation per fact)
  → Entity resolution (canonical entity normalization, cross-memory reference unification)
  → Link construction (4 link types, see below)
  → Persist to PostgreSQL + vector index
```

**Key design decision**: TEMPR uses *narrative-unit chunking* rather than sentence-level splitting. This preserves cross-turn rationale and prevents facts from losing context. Entity resolution happens across the entire memory bank — so "the user" and "John" get unified into the same canonical entity across all prior conversations.

**Graph link types constructed at write time:**

| Link Type | Condition | Notes |
|---|---|---|
| **Entity** | Two facts share the same canonical entity | Core of multi-hop traversal |
| **Temporal** | Facts close in time | Weighted by exponential decay |
| **Semantic** | High embedding cosine similarity | Connects related concepts across entities |
| **Causal** | LLM detects cause-effect relationship | Explicit causal chain for reasoning |

These links form a **temporal entity knowledge graph** in PostgreSQL. It's not a vector DB like most competitors — it's a relational graph with vector indexes on top.

#### Recall Pipeline (Read Path)

```
Query
  → 4-way parallel retrieval:
      • Semantic: vector similarity search
      • Keyword: BM25 exact/fuzzy matching
      • Graph: traversal via entity/temporal/causal links
      • Temporal: time-range filtering
  → Merge results: Reciprocal Rank Fusion (RRF)
  → Cross-encoder reranking (neural reranker for final precision)
  → Token-budget trimming
  → Return to caller
```

This is more sophisticated than most other systems in this space. Mem0 is embedding-only. Graphiti is graph+embedding but no BM25. Honcho re-processes full history. Hindsight adds BM25 and graph traversal to its candidate pool and uses a cross-encoder reranker for final precision.

### 2.3 CARA — Coherent Adaptive Reasoning Agents

CARA implements the **Reflect** operation.

Reflect is the active reasoning pass: given the memory bank, CARA synthesizes new connections, forms opinions, and updates confidence scores. Unlike a simple RAG summarization, Reflect produces **opinions that persist** in the Opinion network and evolve over time.

#### Disposition Parameters (CARA's "personality knobs")

CARA takes a configurable behavioral profile that modulates how it reasons over memories:

| Parameter | Range | Effect |
|---|---|---|
| **Skepticism** | 1–5 | Higher = more cautious evaluation; emphasis on evidence quality; reluctant to accept unsupported claims. Lower = more exploratory, trusting. |
| **Literalism** | 1–5 | Higher = strict adherence to exact wording. Lower = reads between the lines, infers intent. |
| **Empathy** | 1–5 | Higher = weighs feelings and relationships in reasoning. Lower = purely analytical. |
| **Bias-strength** | 0–1 | How strongly the disposition influences the output vs. evidence. |

The paper argues these parameters allow *different agents operating on the same memory bank* to form different but internally consistent opinions — analogous to how two humans with different personalities interpret the same facts differently. This is primarily a product feature (different agents can have different personas) rather than a research contribution, but it's notable that no competing system exposes this.

#### Confidence Score Evolution (Opinion Network)

Opinions stored in the Opinion network carry confidence scores that evolve:
- Supporting evidence from new memories **increases** confidence
- Contradictions **decrease** confidence with a **doubled penalty** vs. support

The doubled contradiction penalty is designed to prevent the system from holding two contradictory opinions with equal confidence indefinitely. Unlike a background batch reconciliation pass, contradiction handling is tightly integrated with the Opinion network at write time.

### 2.4 Storage Backend

| Component | Backend |
|---|---|
| Memory persistence | **PostgreSQL** (relational, not a graph DB) |
| Vector indexes | PostgreSQL extensions (pgvector implied) |
| Deployment | Docker container; `$HOME/.hindsight-docker` for data persistence |
| Embedding provider | Configurable — same LLM provider as language model |

**Important**: Hindsight uses PostgreSQL, not a dedicated graph DB (Neo4j) or vector-only store (LanceDB, Qdrant). The graph is modeled relationally. This has performance implications at scale (no native graph traversal optimizations) but avoids the operational complexity of running multiple storage systems.

### 2.5 LLM Provider Configuration

```bash
# Supported providers
HINDSIGHT_API_LLM_PROVIDER = openai | anthropic | gemini | groq | ollama | lmstudio | minimax
# Also supports: Azure OpenAI, Together AI, Fireworks, LiteLLM (100+ providers)
```

Hindsight uses a **single LLM provider** for all operations (retain fact extraction, recall reasoning, reflect). There is no per-operation model routing — the same provider handles every step. You can, however, swap the entire provider per deployment.

---

## 3. The Three Operations in Practice

### Retain
```python
client.retain("User mentioned they prefer Python over JavaScript for backend work")
```
Internally: chunk → extract facts → embed → resolve entities → build links → store.

### Recall
```python
memories = client.recall("What are the user's programming preferences?")
```
Internally: 4-way parallel retrieval → RRF → rerank → trim → return.

### Reflect
```python
insights = client.reflect("What patterns do you notice in how this user makes decisions?")
```
Internally: CARA loads relevant memories → applies disposition profile → synthesizes opinions → persists new Opinion nodes with confidence scores.

### LLM Wrapper (2-line integration)
```python
from hindsight_client import HindsightWrapper
client = HindsightWrapper(openai_client, hindsight_api_key)
# All subsequent LLM calls automatically retain/recall memories
```
This is a key UX differentiator — most memory systems require explicit calls; the wrapper makes memory transparent.

---

## 4. Benchmark Results

### 4.1 LongMemEval (S) — Per-Category Breakdown

LongMemEval (ICLR 2025) tests 5 abilities across 500 questions in sessions up to 115K tokens. Judge: GPT-4o (>97% agreement with human experts).

| System | Single-Session User | Single-Session Pref | Knowledge Update | Temporal Reasoning | Multi-Session | **Overall** |
|---|---|---|---|---|---|---|
| Full-context OSS-20B (baseline) | 88.0% | 26.0% | 62.0% | 31.6% | 21.1% | 39.0% |
| Full-context GPT-4o | 81.4% | 20.0% | 78.2% | 45.1% | 44.3% | 60.2% |
| Supermemory (GPT-4o) | — | — | — | — | — | 81.6% |
| Supermemory (GPT-5) | — | — | — | — | — | 84.6% |
| **Hindsight (OSS-20B)** | 95.7% | 66.7% | 84.6% | 79.7% | 79.7% | **83.6%** |
| **Hindsight (OSS-120B)** | — | — | — | — | — | **89.0%** |
| **Hindsight (Gemini-3 Pro)** | — | — | — | — | — | **91.4%** |
| Mastra Observational Memory (gpt-5-mini) | — | — | — | — | — | 94.87% |

**Key insight from the per-category data**: Hindsight's biggest gains are in exactly the hardest categories — *multi-session* (21% → 80%) and *temporal reasoning* (32% → 80%) and *single-session preference* (20% → 67%). These require memory across conversations and structured temporal/preference tracking. The single-session-user category (basic factual recall) shows only modest gains because full-context already works well there.

**Notable gap**: Mastra Observational Memory (94.87%) leads Hindsight (91.4%) by 3.5pp using gpt-5-mini. Mastra's approach is different — a stable append-only observation log rather than structured fact extraction, which makes it highly cache-efficient. Worth investigating why Mastra leads despite simpler design.

### 4.2 LoCoMo — Full Table

LoCoMo is a multi-turn, long-conversation dataset with 4 question types. **Important caveat**: Hindsight themselves state in their benchmarks README that they "do not consider [LoCoMo] to be a reliable indicator of memory system quality" due to missing ground truth, ambiguous questions, and insufficient conversation length.

| System | Single-Hop | Multi-Hop | Open Domain | Temporal | **Overall** |
|---|---|---|---|---|---|
| Memobase | — | — | — | — | 75.78% |
| Zep | — | — | — | — | 75.14% |
| **Hindsight (OSS-20B)** | 74.11 | 64.58 | 90.96 | 76.32 | **83.18%** |
| **Hindsight (OSS-120B)** | 76.79 | 62.50 | 93.68 | 79.44 | **85.67%** |
| **Hindsight (Gemini-3 Pro)** | 86.17 | 70.83 | 95.12 | 83.80 | **89.61%** |
| Backboard | — | — | — | — | 90.00% |
| EverMemOS | — | — | — | — | 93.05% |
| MemU | — | — | — | — | 92.09% |

**Multi-hop is the weak spot** (62–70% vs 74–86% for other categories). This is structurally significant — multi-hop requires chaining facts across multiple entity links, and Hindsight's graph traversal is weaker here than EverMemOS's MemCell-based approach.

### 4.3 Agent Memory Benchmark (AMB) — All Datasets

AMB was created by Vectorize (Hindsight's makers). Hindsight v0.4.19 with Gemini-3 Pro:

| Dataset | Hindsight Score | Notes |
|---|---|---|
| **LongMemEvalS** | **94.6%** | #1 on leaderboard |
| **LoComo10** | **92%** | #1 on leaderboard |
| **PersonaMem32K** | **86.6%** | #1 on leaderboard |
| **BEAM 100K** | **75%** | #1 on leaderboard |
| **BEAM 500K** | 71.1% | #1 on leaderboard |
| **BEAM 1M** | 73.9% | Improves vs 500K — unusual |
| **BEAM 10M** | 64.1% | #1; next-best is 40.6% (58% margin) |
| **LifeBenchEN** | 71.5% | #1 on leaderboard |

### 4.4 BEAM Benchmark Deep Dive

BEAM ("Beyond a Million Tokens") is designed to require genuine memory — at 10M tokens, context-stuffing is physically impossible (even 1M-token context windows can't hold the full dataset). This is the key benchmark Vectorize argues actually validates whether a memory system works.

**Model config for BEAM**: The BEAM runs use Hindsight's backend (fact extraction, graph, retrieval) powered by **GPT-OSS-120B**. Gemini-3 Pro is used as the final answer generator in Hindsight's top configuration. The LLM-as-judge is also GPT-OSS-120B consistently across all systems compared.

**BEAM score progression with scale**:
```
100K tokens  → 75%
500K tokens  → 71.1%  (slight dip)
1M tokens    → 73.9%  (recovers — counter-intuitive)
10M tokens   → 64.1%  (next best: 40.6%)
```

The 1M > 500K trend is unusual and not explained in the available materials. Possibly noise, possibly the larger corpus enables better entity co-reference resolution.

---

## 5. Benchmark Validity Assessment

This is important context for interpreting Hindsight's results.

### 5.1 LongMemEval — Most Credible

- **Dataset origin**: Created by independent researchers (Wu et al., ICLR 2025), not Hindsight
- **Judge model**: GPT-4o with >97% agreement to human experts
- **External validation**: Virginia Tech (Naren Ramakrishnan) and The Washington Post (Andrew Neeser) co-authored the paper and reproduced results
- **Context window concern**: LongMemEval was designed for 32K windows; 1M-token models can now partially "cheat" by dumping full history. However, Hindsight's OSS-20B result (83.6%) beats full-context GPT-4o (60.2%), which controls for this.
- **Verdict**: ✅ Credible, externally validated, controls for context stuffing

### 5.2 LoCoMo — Questionable

- **Hindsight's own assessment** (from their benchmarks README): "We do not consider this benchmark to be a reliable indicator of memory system quality" — citing missing ground truth, ambiguous questions, insufficient conversation length, data quality issues
- **Yet they report 89.61%** on this benchmark they distrust
- **Adversarial category excluded** by Hindsight due to evaluation reliability concerns
- **EverMemOS leads here** (93.05%) — and they self-published their score without these caveats
- **Verdict**: ⚠️ Treat with skepticism; Hindsight's own disclaimer makes it inconsistent to cite their score

### 5.3 AMB (Agent Memory Benchmark) — Conflict of Interest

- **Created by Vectorize** (Hindsight's parent company)
- **Scoring formula**: 60% "Fast Benchmark" (speed/cost/reliability) + 40% quality — this means the headline leaderboard score is not purely about answer accuracy
- **Self-submission**: All scores are submitted by vendors; Hindsight submitted their own scores
- **Open methodology**: The harness is public; anyone can run it. Hindsight claims results are reproducible.
- **BEAM specifically**: Designed to prevent context stuffing; tests at scales no model can handle in context. This is the most structurally sound dataset on AMB.
- **Verdict**: ⚠️ Conflict of interest; open methodology partially mitigates this. BEAM results are the most trustworthy portion because the methodology (10M tokens > any context window) is sound regardless of who designed it.

### 5.4 Summary Validity Table

| Benchmark | Dataset Origin | External Validation | Context Stuffing Risk | Trust Level |
|---|---|---|---|---|
| LongMemEval (paper) | Independent (ICLR 2025) | Yes (Virginia Tech, WaPo) | Low (OSS-20B beats GPT-4o full-context) | ✅ High |
| LoCoMo (AMB) | Independent dataset | Partial | Medium | ⚠️ Medium — Hindsight itself disclaims |
| BEAM (AMB) | Vectorize-created | None external | None (10M > any context window) | ✅ Medium-High (sound methodology, creator bias) |
| AMB leaderboard rank | Vectorize-created | None external | N/A | ⚠️ Low for rank claims; see per-dataset |

---

## 6. Paper Summary

**Title**: Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects
**arXiv**: 2512.12818
**Authors**: Chris Latimer + 6 co-authors (Vectorize, Virginia Tech, The Washington Post)
**Published**: December 2025
**Venue**: arXiv preprint (not peer-reviewed as of Apr 2026)

**Core Claim**: Existing memory systems are either retrieval-only (fail at reasoning) or context-dump-based (fail at scale). Hindsight proposes a compositional architecture that (a) structures memory into epistemically-typed networks, (b) retrieves via four parallel strategies fused with RRF, and (c) reasons over the memory bank with configurable disposition to produce evolving opinions.

**Key Contribution**: The formal distinction between World/Experience/Opinion/Observation networks and the TEMPR + CARA decomposition. The empirical contribution is achieving 91.4% on LongMemEval-S with an open-source 120B model backbone — at the time of publication, the highest score on that benchmark with OSS models.

**Experimental Setup**:
- LongMemEval-S: 500 questions, 5 categories, sessions up to 115K tokens; GPT-4o judge
- Baselines: full-context with same backbone, full-context GPT-4o, Supermemory
- Model configs: OSS-20B, OSS-120B, Gemini-3 Pro as answer generator
- Memory system (TEMPR operations): always GPT-OSS-120B for fact extraction and graph construction

**Results Summary**: LongMemEval-S 83.6% (OSS-20B) → 89.0% (OSS-120B) → 91.4% (Gemini-3 Pro). Largest gains in multi-session (+58pp) and temporal reasoning (+48pp) over full-context baseline. Knowledge update: full-context GPT-4o 78.2% vs Hindsight OSS-20B 84.6%.

**Limitations Acknowledged** (from paper/README):
- LoCoMo dataset quality issues (their own assessment)
- Multi-hop recall is the weakest category (62–70%)
- PostgreSQL dependency limits embedded use cases
- No forgetting mechanism

---

## 7. Sources

- [arxiv 2512.12818](https://arxiv.org/abs/2512.12818) — original paper
- [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) — code, README, installation
- [github.com/vectorize-io/hindsight-benchmarks](https://github.com/vectorize-io/hindsight-benchmarks) — benchmark methodology and leaderboard
- [hindsight.vectorize.io/blog/2026/04/02/beam-sota](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota) — BEAM #1 blog post (Apr 2, 2026)
- [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io) — live AMB leaderboard
- [prnewswire.com — Vectorize Breaks 90% on LongMemEval](https://www.prnewswire.com/news-releases/vectorize-breaks-90-on-longmemeval-with-open-source-ai-agent-memory-system-302643146.html)
- [opensourceforu.com — Hindsight Beats RAG](https://www.opensourceforu.com/2025/12/agentic-memory-hindsight-beats-rag-in-long-term-ai-reasoning/)
- [vectorize.io/blog/introducing-hindsight](https://vectorize.io/blog/introducing-hindsight-agent-memory-that-works-like-human-memory)
