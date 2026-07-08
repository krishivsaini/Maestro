# Maestro — Multi-Agent Orchestration System (LangGraph Supervisor) — Complete Build Plan

> **Audience:** A coding agent (Claude Code, Cursor, etc.) executing this build end-to-end with minimal human intervention. Also readable by Krishiv as a reference doc.
> **Goal:** Ship a genuine multi-agent system — a supervisor/orchestrator agent that dynamically decomposes a complex goal into subtasks at runtime and delegates each to a role-specialized subagent, running independent subtasks with bounded parallelism, recovering visibly when a subagent fails, and composing the results into a cited output — all orchestrated as an explicit LangGraph state graph whose every transition is observable and replayable. In 15–17 days, ~3 hours/day, ₹0 to run on the Gemini free tier.
> **Headline framing:** This is *multi-agent orchestration done honestly*. Not renamed nodes in a pipeline — a real supervisor that plans at runtime, delegates to specialists with their own contexts, lets a critic subagent reject and return work, parallelizes what's independent, and recovers when a subagent fails. The differentiator is **observable, recoverable multi-agent coordination**: you can watch the supervisor delegate, watch subagents work in parallel, watch the critic send work back, and replay the whole run node-by-node afterward. Lead external artifacts with "supervisor-orchestrated multi-agent system with visible delegation and recovery," never "autonomous research agent."
> **Reading order:** Read top to bottom once before writing any code. Source of truth throughout.

> **Standalone project (self-contained — no external docs needed):** This doc is complete on its own. Everything needed to build Maestro is in this file; there are no references to other repos or docs.

> **⚠️ Honest-claims anchor (the whole reason this project exists in this form):** The phrase "multi-agent" is easy to overclaim and interviewers probe it hard. This project is designed so the claim is *true*: subagents have separate prompts and separate contexts, the supervisor delegates real subtasks it decided at runtime, and at least one subagent (the Critic) can disagree and send work back. If any of those three stops being true during the build, the claim weakens — protect them. See §18.

---

## Table of Contents

1. [Mission & Non-Negotiables](#1-mission--non-negotiables)
2. [Scope Boundary: What This Project Is Not](#2-scope-boundary)
3. [Glossary](#3-glossary)
4. [Repository Layout](#4-repository-layout)
5. [Environment & Dependencies](#5-environment--dependencies)
6. [The Task This System Performs](#6-the-task)
7. [The Agents (Supervisor + Four Specialists)](#7-the-agents)
8. [State Model (The Shared Blackboard)](#8-state-model)
9. [The Graph (Supervisor Routing + Subagent Nodes)](#9-the-graph)
10. [Dynamic Decomposition (Runtime Planning)](#10-dynamic-decomposition)
11. [Bounded Parallelism & Dependency Handling](#11-bounded-parallelism)
12. [The Critic Loop (Real Disagreement)](#12-the-critic-loop)
13. [Tool Registry](#13-tool-registry)
14. [Memory (Short-Term + Long-Term)](#14-memory)
15. [Error Recovery, Rate Limits & Loop Control](#15-error-recovery)
16. [The API Service (FastAPI + Streaming)](#16-the-api-service)
17. [Observability (Replayable Runs)](#17-observability)
18. [Honest Claims Discipline (Read This Twice)](#18-honest-claims-discipline)
19. [Lightweight Evaluation](#19-lightweight-evaluation)
20. [Day-by-Day Execution Plan](#20-day-by-day-execution-plan)
21. [Acceptance Criteria (Definition of Done)](#21-acceptance-criteria)
22. [Common Failure Modes for the Coding Agent](#22-common-failure-modes)
23. [Output Artifacts Checklist](#23-output-artifacts-checklist)

---

## 1. Mission & Non-Negotiables

### 1.1 What we are building

A multi-agent system in the **supervisor pattern**: one orchestrator agent receives a complex goal, decomposes it into subtasks *at runtime*, delegates each subtask to a role-specialized subagent, coordinates their execution (parallel where independent, sequential where dependent), handles subagent failure, and composes the results into a final cited output.

- **The supervisor (headline)** — an orchestrator that uses the LLM to plan: given a goal, it produces a set of subtasks and assigns each to a specialist role. It then routes work, collects results, and decides when the goal is satisfied.
- **Four specialist subagents** — `Researcher`, `Analyst`, `Critic`, `Writer` (§7), each with its own system prompt and its own working context. They are genuinely distinct agents, not renamed stages.
- **Dynamic decomposition** — the subtask breakdown is decided per-goal at runtime by the supervisor, not hardcoded (§10).
- **Bounded parallelism** — independent subtasks run concurrently (capped at 2–3 at once); dependent subtasks are sequenced (§11).
- **The critic loop (the vivid part)** — the Critic can reject the Analyst's or Writer's output and send it back with feedback; they iterate until the Critic passes it or a ceiling is hit (§12). This is what makes "agents can disagree" real.
- **Visible recovery** — when a subagent fails (tool error, empty result, rate-limit), the supervisor recovers visibly: retry, re-delegate, or degrade — recorded in the trace (§15).
- **Observability** — every delegation, subagent result, critic decision, and recovery is a logged, replayable trace event (§17).
- **The service** — FastAPI streaming the supervisor's delegations and subagents' progress in real time (§16).

### 1.2 Non-negotiables (do not negotiate these away)

1. **Genuine multi-agent, not renamed nodes.** Subagents have separate system prompts and separate contexts. The supervisor delegates subtasks it decided at runtime. If it collapses into "one shared context with role labels," the core claim is false — stop and fix it (§18).
2. **Dynamic decomposition.** The supervisor plans subtasks at runtime from the goal. A hardcoded fixed pipeline is the thing this project exists to *not* be (§10).
3. **A real critic that can disagree.** The Critic must be able to reject work and send it back, and that rejection-and-retry must be demonstrable on demand (§12). A critic that always approves proves nothing.
4. **Bounded parallelism that actually parallelizes.** At least one run must execute independent subtasks concurrently (capped). Dependency-aware: dependent subtasks wait (§11).
5. **Visible recovery.** At least one subagent must be able to fail, and the supervisor must recover via an explicit, logged path (§15). A demo where nothing fails proves nothing.
6. **Loop & cost control.** Hard ceilings on total steps, critic iterations, and recovery attempts. Multi-agent systems loop and multiply cost; bounding them is a required feature, not an afterthought (§15).
7. **Rate-limit resilience.** Exponential backoff with jitter on every LLM/tool call, and a concurrency cap, because the free tier *will* return 429s under parallel bursts (§5, §15). This is non-negotiable and is itself an engineering signal.
8. **Replayable runs.** Every delegation, subagent result, critic verdict, and recovery is logged with enough detail to reconstruct the run afterward (§17).
9. **Streamed visibility.** The API streams supervisor delegations and subagent progress so a viewer can watch the system coordinate in real time (§16).
10. **Reproducible in <10 minutes from a clean clone.**

### 1.3 Scope cuts (explicitly out of scope)

- **No production-grade RAG.** If a subagent retrieves, use a *simple* retriever. Hybrid retrieval, reranking, and formal retrieval eval are out of scope. The signal here is orchestration, not retrieval quality.
- **No formal trajectory-evaluation harness.** Evaluation here is lightweight (§19) — enough to show the system behaves, including under failure, not a metrics framework.
- **No human-in-the-loop approval UI.** Streamed visibility is read-only. No interactive mid-run intervention.
- **No fine-tuning, no cost-optimization layer.** Track cost; don't optimize it.
- **No more than 4 specialist roles.** A sprawling 10-agent swarm is harder to defend, harder to keep under rate limits, and easier to break than a tight 4-role system done well. Depth over breadth.
- **No unbounded fan-out.** Parallelism is capped (§11). Unbounded concurrency both blows the rate limit and is bad design.
- **No fancy frontend.** The streamed-coordination view is dense and functional. Tailwind core only if you build a viewer at all; a clean terminal stream plus a simple web view is plenty.

---

## 2. Scope Boundary

Maestro is a **multi-agent orchestration** project. Its signal is a supervisor that dynamically decomposes goals and coordinates specialist subagents, whose coordination is observable and whose failures recover visibly. Keeping that boundary sharp is what keeps it focused and defensible.

**What this project is:** a supervisor-pattern multi-agent system — runtime decomposition, role-specialized subagents with separate contexts, bounded parallel execution, a critic that can reject and return work, visible recovery, and replayable runs — built as an explicit LangGraph state graph.

**What this project is deliberately not:**
- Not a production RAG system — any retrieval is deliberately simple; hybrid retrieval and reranking are out of scope.
- Not a formal evaluation harness — evaluation is lightweight, enough to demonstrate behavior.
- Not a fine-tuning or cost-optimization project — cost is tracked, not optimized.
- Not an agent swarm — exactly one supervisor and four specialists, done well.

**Why LangGraph is used here (stands on its own):** the supervisor pattern *is* a graph — a controller node that routes to specialist nodes, with conditional edges for delegation, the critic loop, and recovery, over a shared typed state. Once you need runtime routing between multiple agents, a critic loop that sends work back, bounded parallel branches, and recovery edges, hand-rolling that coordination means reinventing a graph/state engine — badly. LangGraph is purpose-built for exactly this: `StateGraph` with named nodes, typed shared state, conditional edges, and parallel branches. The interesting engineering here is not the framework; it is making the multi-agent coordination **observable, recoverable, and honestly multi-agent**. LangGraph handles the graph mechanics so the build concentrates on those hard, rare parts. Because the project genuinely needs and uses LangGraph, and you can explain *why*, listing it as a skill is earned rather than padded.

---

## 3. Glossary

| Term | Definition |
|---|---|
| **Supervisor / orchestrator** | The controlling agent that decomposes the goal, delegates subtasks, collects results, and decides completion. |
| **Subagent / specialist** | A role-specialized agent (Researcher, Analyst, Critic, Writer) with its own system prompt and its own working context. |
| **Dynamic decomposition** | The supervisor deciding, at runtime via the LLM, what subtasks exist and which specialist handles each — not a hardcoded pipeline. |
| **Subtask** | A unit of work the supervisor delegates to one specialist, with a dependency list (which subtasks must finish first). |
| **Bounded parallelism** | Running independent subtasks concurrently up to a fixed cap (2–3), while dependent subtasks wait. |
| **Critic loop** | The Critic reviewing another subagent's output and either passing it or rejecting it back for revision, bounded by a max-iteration ceiling. |
| **Shared state** | The typed object (Pydantic / `TypedDict`) that flows through the graph and holds the plan, subtask results, critic verdicts, and trace. |
| **Working memory** | Short-term state held in the graph state for the duration of a run. |
| **Long-term memory** | Persisted memory (vector store) queryable across steps / within a thread. |
| **Thread / session** | An identifier grouping a multi-turn interaction and its long-term memory. |
| **Recovery** | The explicit, logged handling of a failed subagent: retry (with backoff), re-delegate, or degrade. |
| **Run trace** | The ordered, replayable record of delegations, subagent results, critic verdicts, and recoveries for one execution. |

---

## 4. Repository Layout

```
maestro/                                # repo root
├── README.md                              # orchestration-first; the hiring-decision artifact (§21)
├── LICENSE                                # MIT
├── .gitignore                             # .env, *.db, __pycache__, .next, node_modules, memory_store/
├── .env.example                           # GOOGLE_API_KEY= (primary, Gemini free tier), search API key
├── pyproject.toml                         # uv-managed Python project
├── ARCHITECTURE.md                        # the rendered graph diagram + the "why LangGraph / why supervisor" rationale
├── graph.png                              # exported LangGraph diagram (regenerated by a script)
│
├── maestro/                            # the package
│   ├── __init__.py
│   ├── state.py                           # the shared State model incl. subtasks + verdicts (§8)
│   ├── graph.py                           # StateGraph: supervisor routing + subagent nodes + critic loop (§9)
│   ├── supervisor.py                      # decomposition + delegation + completion logic (§10)
│   ├── agents/
│   │   ├── base.py                        # shared subagent scaffolding (own prompt, own context, backoff)
│   │   ├── researcher.py                  # gathers info via search/retrieve
│   │   ├── analyst.py                     # synthesizes findings into analysis
│   │   ├── critic.py                      # reviews analysis/writing; can REJECT and return (§12)
│   │   └── writer.py                      # composes final cited output
│   ├── scheduler.py                       # bounded-parallel, dependency-aware subtask execution (§11)
│   ├── tools/
│   │   ├── registry.py                    # tool registry + schemas (§13)
│   │   ├── web_search.py                  # real search tool (with a fault-injection switch for the demo)
│   │   └── retrieve.py                    # SIMPLE vector retrieval (deliberately basic, not production-grade)
│   ├── memory/
│   │   ├── working.py                     # short-term, in-state helpers
│   │   └── longterm.py                    # vector-store-backed long-term memory
│   ├── resilience.py                      # exponential backoff + jitter + rate-limit handling (§15)
│   ├── serve.py                           # FastAPI: /run (streams delegations + subagent progress), /healthz, thread id
│   ├── trace.py                           # run-trace recording + export (§17)
│   ├── logging_config.py                  # structured logging
│   └── config.py                          # caps: max steps, max critic iters, max recovery, parallel cap, model names
│
├── viewer/                                # OPTIONAL minimal web viewer for the streamed trace (Tailwind core)
│   └── ...
│
├── tests/
│   ├── test_graph.py                      # graph compiles; happy-path run composes a final answer
│   ├── test_decomposition.py              # supervisor produces subtasks with dependencies for a goal
│   ├── test_parallelism.py                # independent subtasks run concurrently; cap is respected
│   ├── test_critic_loop.py                # critic rejects once, revision happens, then passes; ceiling halts
│   ├── test_recovery.py                   # injected subagent failure → recovery → run still completes
│   └── test_loop_control.py               # max-step / max-iter ceilings halt deliberately runaway cases
│
├── scripts/
│   ├── render_graph.py                    # regenerates graph.png from the StateGraph
│   └── demo_cases.py                      # the handful of showcase goals (§19)
│
└── data/
    └── eval_cases.jsonl                   # ~15 cases for lightweight eval (§19)
```

---

## 5. Environment & Dependencies

- **Python** via `uv`; pin versions in `pyproject.toml`.
- **Core:** `langgraph`, `langchain-core` (message/tool types), `langchain-google-genai` (Gemini binding), a search tool (Tavily / SerpAPI / DuckDuckGo — pick one with a free tier), a vector store for long-term memory (FAISS local), `fastapi` + `uvicorn`, `pydantic`, `tenacity` (retries/backoff), `sse-starlette` or native FastAPI streaming.
- **Model — Gemini 3.5 Flash on the free tier (primary).** Use a local embedding model (`sentence-transformers`, e.g. `bge-small-en`) for long-term memory so that part is fully free. Keep a single provider binding; the code should let the model ID be swapped via `config.py`.

> **⚠️ §5 RATE-LIMIT REALITY — architect around this, it is not optional.**
> The Gemini free tier for Flash models is roughly **10–15 RPM, ~250K–1M TPM, ~1,500 RPD** (verify current numbers — see note below). A multi-agent run *multiplies* calls: one goal → 1 planning call + several subagent calls + critic iterations + a compose call = easily 6–10+ calls per run, and **parallel bursts hit the RPM ceiling fastest**. Therefore:
> - **Exponential backoff with jitter is mandatory** on every model/tool call (`resilience.py`, via `tenacity`). Build it before you need it — 429s appear immediately under parallel load.
> - **Cap concurrency at 2–3** (`config.py` `MAX_PARALLEL`). Queue the rest. Never fan out unbounded.
> - **Bound critic iterations and recovery attempts** so a single run can't spiral into dozens of calls.
> - RPD (1,500/day) is *not* your constraint for building/demoing — you'll make tens of runs, not thousands. **RPM during parallel bursts is the real friction**, and backoff + the concurrency cap solve it.
> - Document a "scales to higher concurrency on a paid tier" boundary in the README — an honest, mature framing of the free-tier limit.

> **§5 note to coding agent — verify live before building:**
> - Gemini free-tier rate limits change often (Google tightened them more than once in 2026). **Check `https://ai.google.dev/gemini-api/docs/rate-limits` for the current Flash free-tier RPM/TPM/RPD before setting `MAX_PARALLEL` and backoff parameters.** The exact RPM number determines how aggressively you can parallelize.
> - **Confirm the current Gemini Flash model ID** (e.g. `gemini-3.5-flash` or the then-current identifier) on the pricing/models page — do not hardcode from memory; model IDs and free-tier eligibility shift.
> - LangGraph's API surface (StateGraph, checkpointers, streaming, parallel/fan-out edges) changes between versions. **Check LangGraph's current docs for the installed version before writing `graph.py` and `scheduler.py`.** Do not rely on memory for the parallel-branch / `Send` API — verify it.
> - When unsure about any provider or framework API, check docs rather than guessing.

---

## 6. The Task

The system's goal must genuinely require decomposition into subtasks handled by *different specialists*, or the multi-agent structure has nothing to show. **Pick one concrete task domain and commit on Day 1.** Recommended domain: **research/analysis synthesis.**

Canonical task: **"Produce a cited analytical brief on a non-trivial question"** — e.g. *"Compare approaches X and Y for problem Z and recommend one, with evidence."* This naturally decomposes into:
- research subtasks (gather evidence on X; gather evidence on Y) — **independent → parallelizable**
- an analysis subtask (synthesize a comparison) — **depends on the research**
- a critique subtask (check the analysis for gaps/unsupported claims) — **depends on the analysis, can send it back**
- a writing subtask (compose the final cited brief) — **depends on a passed critique**

Why this domain: the dependency structure is real (some subtasks parallelize, some must sequence), the critic role is natural (reviewing analysis quality), recovery is natural (a search returns nothing → re-delegate or degrade), and it's legible in a 90-second demo.

Criteria the chosen task must meet: requires ≥3 subtasks across ≥2 specialist roles on a typical run; has at least one pair of independent subtasks (to show parallelism); has a natural critic-rejection point; and has an injectable failure (search empty / source unreadable) to demonstrate recovery.

> Document the chosen task in the README's first paragraph. The whole demo hangs on it being legibly non-trivial and genuinely decomposable.

---

## 7. The Agents

One supervisor, four specialists. **Each specialist has its own system prompt and its own working context** — this separation is what makes "multi-agent" honest (§18).

| Agent | Role | Own context sees | Produces |
|---|---|---|---|
| **Supervisor** | Decompose goal → subtasks with dependencies; assign each to a specialist; collect results; decide completion | The goal, the subtask list + statuses, subagent results (summarized) | Delegations; completion decision |
| **Researcher** | Gather evidence for an assigned sub-question via search/retrieval | Its assigned sub-question + tool results (not other agents' internals) | `Evidence` items with sources |
| **Analyst** | Synthesize gathered evidence into a structured analysis/comparison | The research evidence relevant to its subtask | An analysis draft with claims tied to evidence |
| **Critic** | Review the analysis (or final writing) for gaps, unsupported claims, contradictions; **pass or reject with feedback** | The analysis draft + the evidence it cites | A verdict: PASS, or REJECT + specific feedback |
| **Writer** | Compose the final cited brief from the passed analysis | The passed analysis + evidence | Final cited output |

Key design rules:
- Subagents communicate **through the shared state's structured fields and via the supervisor**, not by reading each other's raw context. The Researcher's scratch reasoning is not visible to the Writer; only its produced `Evidence` is.
- The **Critic is the disagreement mechanism** — it must be able to send work back (§12). Without it, "multi-agent" is just parallel workers.
- Keep each subagent's prompt genuinely role-scoped. If two subagents' prompts are near-identical, they're not really distinct agents — collapse or differentiate them.

---

## 8. State Model (`state.py`)

The shared state is the blackboard the supervisor and subagents coordinate through. Define it explicitly (Pydantic / `TypedDict`) before writing nodes. Sketch (coding agent fills fields):

- `goal: str`
- `subtasks: list[Subtask]` — each `{id, description, role, depends_on: list[id], status, result?}` (status: pending/running/done/failed/degraded)
- `evidence: list[Evidence]` — accumulated researcher findings, each with source + content + subtask_id
- `analysis: Optional[AnalysisDraft]` — the Analyst's current draft
- `critic_verdicts: list[Verdict]` — history of PASS/REJECT + feedback (bounded, §12)
- `critic_iterations: int` — for the critic-loop ceiling
- `final_output: Optional[Answer]` — the Writer's cited composition
- `step_count: int` — global loop control
- `recovery_attempts: int` — bounded
- `memory_hits: list[MemoryItem]` — what long-term memory returned this run
- `trace: list[TraceEvent]` — appended by every node/agent (§17)

The `Subtask` model with its `depends_on` list is the spine of the whole system — it's what makes decomposition dynamic (§10) and scheduling dependency-aware (§11). Get it right before writing the graph.

---

## 9. The Graph (`graph.py`)

A `StateGraph` in the supervisor pattern. Conceptual shape (the coding agent verifies the exact LangGraph parallel/routing API against current docs — §5):

```
START → supervisor(plan) → scheduler ──▶ [Researcher ∥ Researcher ∥ …]   (independent, bounded-parallel)
                                │                     │
                                │                     ▼
                                └──────────────▶  Analyst  (after research deps met)
                                                       │
                                                       ▼
                                                    Critic ──▶ (REJECT) ──▶ back to Analyst   (bounded loop, §12)
                                                       │
                                                    (PASS)
                                                       ▼
                                                    Writer ──▶ (optional final Critic pass) ──▶ END

any subagent failure ──▶ supervisor(recover) ──▶ retry / re-delegate / degrade   (bounded, §15)
```

- **supervisor(plan)** — dynamic decomposition (§10): produces `subtasks` with dependencies and role assignments.
- **scheduler** — dispatches ready subtasks (dependencies satisfied) with bounded parallelism (§11).
- **specialist nodes** — Researcher / Analyst / Critic / Writer, each its own agent (§7).
- **critic edge** — conditional: REJECT routes back to the Analyst (or Writer) with feedback; PASS advances. Bounded by `MAX_CRITIC_ITERS` (§12).
- **recovery edge** — any subagent failure routes to the supervisor's recover logic (§15).
- Export the compiled graph to `graph.png` via `scripts/render_graph.py` for the README and ARCHITECTURE.md.

---

## 10. Dynamic Decomposition (`supervisor.py`)

This is what makes the "orchestrator that breaks down goals" claim true rather than cosmetic.

- The supervisor prompts the LLM with the goal and the available specialist roles, and asks for a **structured plan**: a list of subtasks, each with a description, an assigned role, and a `depends_on` list referencing other subtasks. Use structured output (a Pydantic schema) so the plan is parseable, not free text.
- Different goals must produce different plans. `test_decomposition.py` runs two distinct goals and asserts the subtask sets differ and dependencies are well-formed (no cycles, valid references).
- The supervisor re-plans on recovery when a subtask fails in a way that changes the plan (e.g. an evidence source is unavailable → add an alternative research subtask).
- **Bound the plan size** (e.g. ≤6 subtasks) in `config.py` — both for rate-limit sanity and because a good decomposition is tight, not sprawling.

> Do not hardcode the subtask list. A fixed pipeline with an LLM call bolted on top is exactly the non-dynamic version this project rejects.

---

## 11. Bounded Parallelism & Dependency Handling (`scheduler.py`)

- The scheduler looks at `subtasks`, finds those whose `depends_on` are all `done`, and dispatches them.
- **Independent ready subtasks run concurrently, capped at `MAX_PARALLEL` (2–3).** Queue the rest. This is the parallelism signal — and the cap is the rate-limit protection (§5).
- **Dependent subtasks wait** until their dependencies complete. This is the dependency-awareness signal.
- `test_parallelism.py` asserts (a) two independent subtasks actually run concurrently, and (b) the concurrency cap is never exceeded, and (c) a dependent subtask does not start before its dependency is `done`.
- Implementation note: LangGraph has native fan-out/parallel-branch support — **verify the current API (e.g. `Send` / conditional parallel edges) against installed-version docs (§5)** before building this; the API for parallel branches has changed across versions.

> The honest framing for the README: "independent subtasks execute concurrently (bounded at N); dependent subtasks are sequenced; higher concurrency is available on a paid tier." True, and mature.

---

## 12. The Critic Loop (`agents/critic.py`)

This is what makes "agents can disagree" real and demoable — the single most vivid part of the multi-agent claim.

- After the Analyst produces a draft, the Critic reviews it against the gathered evidence and returns a structured **verdict: PASS or REJECT + specific feedback** (what's unsupported, what's missing, what contradicts).
- On REJECT, the draft goes **back to the Analyst** with the feedback; the Analyst revises; the Critic reviews again.
- **Bounded by `MAX_CRITIC_ITERS`** (e.g. 3). If the ceiling is hit without a PASS, degrade gracefully: proceed with the best draft, flagged as "not fully validated." Never loop unbounded.
- `test_critic_loop.py` proves: a deliberately weak first draft is rejected, a revision happens, and either a later draft passes or the ceiling halts it — and the whole exchange is in the trace.
- **The demo money-shot:** in the Loom, show the Critic rejecting the Analyst's first draft with concrete feedback, the Analyst revising, and the second draft passing. That single sequence is what proves this is multi-agent, not a pipeline.

> A Critic that always passes is theater. It must be *able* to reject, and you must be able to trigger a rejection on demand.

---

## 13. Tool Registry (`tools/registry.py`)

- A small set (2–3) of well-described tools: web search, simple retrieval.
- Each tool has a typed schema and returns either a result or a **structured failure** (never raises raw into the graph — the supervisor decides what failures mean).
- **One tool must support fault injection** (a config/env switch that makes it fail or return empty) so the recovery path is demonstrable on demand. This produces the recovery Loom moment.

> Keep retrieval deliberately simple (§1.3). Production-grade retrieval is out of scope.

---

## 14. Memory (`memory/`)

- **Working memory** — lives in the graph state for the run (`evidence`, `subtasks`, `analysis`, etc.). No persistence.
- **Long-term memory** — vector store (local embeddings, free). On completion (or per validated finding), write distilled findings tagged with the thread id. On a later turn in the same thread, the supervisor queries it during planning and results land in `memory_hits`, informing decomposition and synthesis.
- **Non-negotiable:** long-term memory must be *read back and used*, not just written (§1.2). Cleanest demonstration: a two-turn thread where turn 2's plan/answer provably uses something stored in turn 1.

---

## 15. Error Recovery, Rate Limits & Loop Control (`resilience.py`, `supervisor.py`, `config.py`)

This is a rare, high-signal cluster. Most multi-agent demos have no real recovery, no rate-limit handling, and loop forever.

**Rate-limit resilience (mandatory — §5):**
- `resilience.py` wraps every LLM/tool call in exponential backoff with jitter (tenacity). A 429 triggers backoff-and-retry, not a crash.
- The concurrency cap (`MAX_PARALLEL`) prevents most 429s at the source.
- Rate-limit retries are logged to the trace (visible, not silent) — "handles provider rate limits gracefully" is a legitimate signal you can point to.

**Subagent recovery:**
- When a subagent reports a structured failure, the supervisor's recover logic: (a) retries with backoff up to a bound, then (b) re-delegates / re-plans around the failed subtask (e.g. alternative source), then (c) degrades gracefully (marks that subtask "could not complete," proceeds with partial evidence, flags it in the output). Each choice is an explicit, logged transition.

**Loop & cost control:**
- Hard `MAX_STEPS` ceiling halts any run that exceeds it.
- `MAX_CRITIC_ITERS` bounds the critic loop (§12).
- `MAX_RECOVERY_ATTEMPTS` bounds recovery.
- A loop detector notices repeated identical delegations and forces completion.
- `test_loop_control.py` proves the ceilings halt deliberately runaway cases.

Every recovery decision, rate-limit retry, and ceiling trigger is recorded in the trace so it's visible, not silent.

---

## 16. The API Service (`serve.py`)

- `POST /run` — accepts `{goal, thread_id?}`; **streams supervisor delegations and subagent progress** as they happen (SSE): plan produced → subtasks dispatched → each subagent's status → critic verdicts → final output. A viewer watches the system coordinate in real time.
- `GET /healthz` — liveness.
- Thread identity: `thread_id` groups turns and long-term memory.
- Structured logging: every request logs run id, the plan, the delegation path, critic iterations, recoveries, rate-limit retries, and final outcome.
- Pydantic validation throughout.

> Streaming is the difference between a legible 90-second demo and an opaque one. Watching the supervisor delegate and the critic reject is the whole pitch. Required (§1.2.9).

---

## 17. Observability — Replayable Runs (`trace.py`)

- Every meaningful event appends a `TraceEvent`: `{event_type, agent, timestamp, summary, subtask_id?, tool_call?, error?, critic_verdict?, recovery_decision?, rate_limit_retry?}`. Event types include: plan_produced, subtask_dispatched, subagent_result, critic_pass, critic_reject, recovery, rate_limit_backoff, degraded, completed.
- On completion, the full trace is persisted (SQLite or JSON) keyed by run id and exportable, so a reader (or the viewer) can **replay the whole coordination node-by-node afterward** — including which subtasks ran in parallel, where the critic pushed back, and where recovery happened.
- This is the production-thinking signal: a full audit log of a multi-agent run you can reconstruct after the fact.

---

## 18. Honest Claims Discipline (Read This Twice)

"Multi-agent" is a loaded, frequently-overclaimed term, and interviewers at the level worth targeting will probe it in one or two questions. This project is designed so the claim survives probing — **but only if you protect three things during the build:**

1. **Separate contexts.** Each subagent has its own system prompt and sees only its own task + relevant structured inputs — not one shared scratchpad every node writes to. If during the build it collapses into a single shared context with role labels, the claim "multi-agent" becomes false. Stop and restore the separation.
2. **Real runtime delegation.** The supervisor decides subtasks at runtime and delegates them. If the "plan" quietly becomes a fixed pipeline, you have multi-step, not multi-agent — call it accurately or fix it.
3. **A critic that can actually disagree.** If the Critic always passes, "agents coordinate and disagree" is theater. You must be able to trigger a rejection live.

Claims you may make honestly if the above hold: *"supervisor-orchestrated multi-agent system," "dynamic runtime decomposition," "role-specialized subagents with separate contexts," "a critic subagent that can reject and return work," "bounded-parallel dependency-aware execution," "visible recovery from subagent failure."*

Claims to avoid unless literally true: *"fully autonomous agents," "emergent agent behavior," "agents negotiate"* (they coordinate through a supervisor and shared state — say that). And every number (subtask counts, critic iterations, parallel width, latency) must come from real runs you can reproduce on demand.

The interview value of this project is that you can **drive it live** — open the stream, give it a goal, watch it decompose, watch subagents run in parallel, watch the critic reject and the analyst revise, inject a failure and watch recovery. Build toward that demo being real, and every claim takes care of itself. A precise, smaller, demoable claim beats an impressive one that collapses under one follow-up question.

---

## 19. Lightweight Evaluation (`data/eval_cases.jsonl`, `scripts/demo_cases.py`)

Deliberately *light* — formal evaluation is out of scope (§1.3). The point is to *demonstrate the system behaves*, including under failure — not to produce a metrics framework.

- ~15 cases covering: typical decomposable goals; at least 2 with independent subtasks (to show parallelism); at least 2 that trigger a critic rejection; at least 2 that trigger recovery (via fault injection); at least 1 that exercises a ceiling; and at least 1 two-turn thread exercising long-term memory recall.
- For each, record: did it complete, number of subtasks, max parallel width reached, critic iterations, recoveries triggered, rate-limit retries, and a hand-judged pass/fail on the final output (hand-judged is fine at this scale — say so honestly).

---

## 20. Day-by-Day Execution Plan

15–17 days × ~3 hours. If something slips, **cut the optional web viewer and the code-exec tool — never the critic loop, the parallelism, the recovery, or the rate-limit handling**, because those are the differentiators and the honesty anchors.

| Day | Focus | End-of-day artifact |
|---|---|---|
| 1 | Verify live Gemini free-tier limits + model ID + LangGraph API (§5); pick task (§6); repo scaffold; `config.py`, `state.py` (incl. `Subtask`/deps) | Limits confirmed; task committed; state model with dependencies defined |
| 2 | `resilience.py` (backoff + jitter) first; tool registry + web search + simple retrieve; fault-injection switch | Every call is backoff-wrapped; one tool can fail on demand |
| 3 | Supervisor dynamic decomposition (§10); structured plan output | Two different goals produce two different valid subtask plans; `test_decomposition.py` green |
| 4 | `scheduler.py`: dependency resolution + bounded parallelism (§11) | Independent subtasks run concurrently under cap; deps wait; `test_parallelism.py` green |
| 5 | Researcher + Analyst subagents with separate contexts (§7) | Research→analysis path produces a draft from real evidence |
| 6 | Critic subagent + critic loop (§12): reject → revise → pass, bounded | Deliberately weak draft gets rejected then revised; `test_critic_loop.py` green |
| 7 | Writer subagent; full happy-path plan→research∥→analyze→critique→write | End-to-end cited brief for a real goal |
| 8 | Supervisor recovery (§15): retry → re-delegate → degrade on injected failure | Injected failure recovers visibly; `test_recovery.py` green |
| 9 | Loop/cost control: MAX_STEPS, MAX_CRITIC_ITERS, MAX_RECOVERY; loop detector | `test_loop_control.py` green; runs can't spiral |
| 10 | Long-term memory: write on completion, read in planning; two-turn thread | Turn 2 provably uses turn 1's stored finding |
| 11 | `trace.py`: record every event type; persist + export replayable runs | A completed run is fully replayable, incl. parallelism + critic + recovery |
| 12 | FastAPI `/run` streaming delegations + subagent progress; `/healthz`; thread id | Watchable real-time coordinated run end-to-end |
| 13 | `render_graph.py` → `graph.png`; ARCHITECTURE.md; eval_cases (~15) | Graph diagram + lightweight eval results |
| 14 | Rate-limit hardening pass under real parallel load; tune MAX_PARALLEL/backoff; reproducibility pass | Clean-clone run <10 min; no unhandled 429s under demo load |
| 15 | (Optional) minimal viewer; otherwise polish streaming + logging | Coordination is legible to a viewer |
| 16 | README + Loom (decompose → parallel → critic reject → recover) + resume bullet | All §23 artifacts exist |
| 17 | Buffer / overflow for the hardest parts (parallelism API, critic loop) | Slack day — multi-agent builds run long |

---

## 21. Acceptance Criteria (Definition of Done)

### 21.1 Multi-agent orchestration
- [ ] Supervisor produces a **dynamic** subtask plan (different goals → different plans); `test_decomposition.py` proves it.
- [ ] Subagents have **separate system prompts and separate contexts** (§18.1) — verifiable in code.
- [ ] `StateGraph` compiles; happy-path run decomposes, delegates, and composes a cited final output.
- [ ] The graph is exported as `graph.png` and embedded in README + ARCHITECTURE.md.

### 21.2 Parallelism & dependencies
- [ ] Independent subtasks run concurrently; concurrency never exceeds `MAX_PARALLEL`; dependent subtasks wait; `test_parallelism.py` proves all three.

### 21.3 Critic loop
- [ ] Critic can REJECT with feedback; a rejected draft is revised and re-reviewed; ceiling halts runaway; `test_critic_loop.py` proves it.
- [ ] A rejection is triggerable on demand for the demo.

### 21.4 Recovery, rate limits & loop control
- [ ] Injected subagent failure routes to recovery and the run still completes; `test_recovery.py` proves it.
- [ ] Every LLM/tool call is backoff-wrapped; 429s are handled, not fatal; retries are logged.
- [ ] `MAX_STEPS`, `MAX_CRITIC_ITERS`, `MAX_RECOVERY_ATTEMPTS` all enforced; `test_loop_control.py` proves it.

### 21.5 Memory
- [ ] Long-term memory is written and **read back and used** in a later turn; a two-turn thread demonstrates it.

### 21.6 Service & observability
- [ ] `/run` streams delegations and subagent progress in real time.
- [ ] Every run is a replayable trace: parallelism, critic verdicts, and recovery all reconstructable after completion.

### 21.7 README + Loom
- [ ] README opens with the orchestration headline, not "autonomous research agent":

> **Maestro: a supervisor-orchestrated multi-agent system with visible delegation and recovery.**
>
> A supervisor agent decomposes a goal into subtasks at runtime and delegates each to a role-specialized subagent (Researcher, Analyst, Critic, Writer), each with its own context. Independent subtasks run in bounded parallel; a Critic subagent can reject and return work until it passes; subagent failures recover visibly (retry → re-delegate → degrade); and every run is replayable node-by-node. Built on LangGraph, runs on the Gemini free tier with exponential backoff and a concurrency cap.
>
> [Live demo / Loom →] [Graph diagram →] [Architecture →]

- [ ] The rendered graph diagram is in the README.
- [ ] "Why LangGraph for this orchestration" subsection — two sentences on why supervisor-pattern multi-agent coordination is exactly what a graph framework is for (§2).
- [ ] A concrete **critic-rejection** example shown (draft → reject+feedback → revision → pass) — a killer screenshot.
- [ ] A concrete **recovery** example shown (injected failure → recovery) — a second killer screenshot.
- [ ] "Limitations" section: simple retrieval (not production RAG), lightweight eval, single task domain, hand-judged output quality, free-tier concurrency cap (higher on paid). Framed as deliberate scope choices.
- [ ] "Reproducing" section with literal commands.
- [ ] 90-second Loom: give a goal → watch decomposition → watch parallel research → watch the Critic reject and the Analyst revise → inject a failure → watch recovery → final cited brief.

### 21.8 Repo hygiene
- [ ] `.gitignore` excludes `.env`, `*.db`, `memory_store/`, `__pycache__`, `.next`, `node_modules`.
- [ ] No secrets in history; the Gemini key is via env only.
- [ ] `uv sync` succeeds from clean clone; LICENSE is MIT.

---

## 22. Common Failure Modes for the Coding Agent

1. **Do not collapse subagents into one shared context with role labels.** Separate prompts, separate contexts — or the "multi-agent" claim is false (§18.1, §7).
2. **Do not hardcode the subtask plan.** Decomposition is dynamic and runtime (§10). A fixed pipeline is the anti-goal.
3. **Do not build a critic that always passes.** It must be able to reject and return work, triggerably (§12).
4. **Do not fan out unbounded.** Cap concurrency at `MAX_PARALLEL`; queue the rest (§11) — for both correctness and rate limits.
5. **Do not skip backoff.** Every LLM/tool call is wrapped; 429s appear immediately under parallel load (§5, §15). Build `resilience.py` on Day 2, before the parallelism.
6. **Do not omit the ceilings.** MAX_STEPS / MAX_CRITIC_ITERS / MAX_RECOVERY are required; multi-agent systems loop and multiply cost (§15).
7. **Do not build production RAG here.** Simple vector search only (§1.3).
8. **Do not add a fifth specialist or a swarm.** Four roles, done well (§1.3).
9. **Do not swallow errors in try/except.** Tools return structured failures; the *supervisor* decides what they mean (§13, §15).
10. **Do not rely on memory for the LangGraph parallel/`Send` API or the Gemini model ID / limits.** Verify against current docs before building (§5).
11. **Do not write README placeholders.** Real run traces, real graph diagram, real critic-rejection and recovery screenshots. No invented numbers (§18).

---

## 23. Output Artifacts Checklist

By the final day, all of the following must exist and be publicly accessible:

| # | Artifact | Location | Required for |
|---|---|---|---|
| 1 | Public GitHub repo `maestro` | `github.com/<user>/maestro` | Resume link, outreach |
| 2 | Orchestration-first README with the embedded graph diagram | repo root | Recruiter first impression |
| 3 | `ARCHITECTURE.md` + `graph.png` (rendered from the StateGraph) | repo root | Engineering depth signal |
| 4 | Passing tests incl. `test_decomposition`, `test_parallelism`, `test_critic_loop`, `test_recovery`, `test_loop_control` | `tests/` + CI tab | Proof the hard parts work |
| 5 | A captured **critic-rejection** trace (draft → reject → revise → pass) | README screenshot | The "it's really multi-agent" evidence |
| 6 | A captured **recovery** trace (injected failure → recovery) | README screenshot | The resilience evidence |
| 7 | Replayable run traces (persisted/exportable), showing parallelism | repo / demo | Observability signal |
| 8 | 90-second Loom (decompose → parallel → critic reject → recover) | Loom public link | LinkedIn post, outreach |
| 9 | Two-turn thread demonstrating long-term memory recall | demo / README | Memory claim is real |
| 10 | "Why LangGraph / why supervisor pattern" rationale | README + ARCHITECTURE.md | The judgment signal |
| 11 | Resume bullet with honest, demoable multi-agent claims | resume file | Linked from outreach |

When all exist, this is a complete, defensible multi-agent orchestration project: a supervisor that dynamically decomposes goals and coordinates specialist subagents with visible delegation, real disagreement, bounded parallelism, and visible recovery — a system you can *drive live* in an interview, which almost no fresher portfolio can honestly claim.

---

## Final note to the coding agent executing this

This document is the source of truth. When the agent's instinct conflicts with this document, this document wins. When the agent thinks "it would be simpler to share one context across agents / hardcode the plan / make the critic always pass / fan out unbounded / skip backoff," the agent should stop — those are exactly the shortcuts that turn this from a genuine, defensible multi-agent system into a renamed pipeline that collapses under one interview question.

The goal is not the most capable research system. The goal is a focused, honest, *live-demoable* multi-agent orchestration system whose supervisor visibly decomposes and delegates, whose critic visibly disagrees, and whose failures visibly recover — all on the free tier, all reproducible. Optimize for that.

Good build, Krishiv.
