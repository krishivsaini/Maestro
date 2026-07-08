# Maestro — Requirements Specification

> **Status:** Draft v1 · Derived from `MAESTRO_BUILD_PLAN.md` (source of truth)
> **Project:** Maestro — a supervisor-orchestrated multi-agent system with visible delegation and recovery.
> **Purpose of this doc:** Enumerate *what* Maestro must do (functional) and *how well* it must do it (non-functional), with traceable acceptance criteria. The build plan holds the narrative rationale; this document holds the checklist a reviewer signs off against.

---

## 1. Overview

Maestro is a multi-agent system built in the **supervisor pattern**. A single orchestrator agent receives a complex goal, decomposes it into subtasks *at runtime*, delegates each to a role-specialized subagent, coordinates their execution (parallel where independent, sequential where dependent), recovers visibly when a subagent fails, and composes results into a cited output — all as an explicit LangGraph state graph that is observable and replayable.

The system runs on the **Gemini free tier at ₹0**, which makes rate-limit resilience a first-class requirement rather than an afterthought.

### 1.1 The one-line claim this project must keep true

> A supervisor agent decomposes a goal into subtasks at runtime and delegates each to a role-specialized subagent (Researcher, Analyst, Critic, Writer), each with its own context. Independent subtasks run in bounded parallel; a Critic subagent can reject and return work until it passes; subagent failures recover visibly; and every run is replayable node-by-node.

Every requirement below exists to make that sentence literally true and demonstrable on demand.

---

## 2. Stakeholders & Users

| Stakeholder | Interest |
|---|---|
| **Krishiv (builder/owner)** | A defensible, live-demoable portfolio project that survives interview probing on the "multi-agent" claim. |
| **Interviewer / reviewer** | Wants to watch the system decompose, delegate, disagree (critic), and recover — live — and probe whether "multi-agent" is honest. |
| **Operator (demo driver)** | Runs a goal through the API, injects a failure, and watches the stream. |
| **Coding agent (implementer)** | Needs unambiguous, testable requirements to build against. |

**Primary use case:** Given a non-trivial analytical question, produce a cited analytical brief by orchestrating specialist subagents — while the whole coordination is watchable in real time and replayable afterward.

---

## 3. Functional Requirements

Each requirement has an ID (`FR-n`), a priority (**MUST** / **SHOULD** / **MAY**), and a verification hook (the test or artifact that proves it).

### 3.1 Supervisor & Dynamic Decomposition

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-1 | MUST | The supervisor shall accept a free-text goal and produce a **structured plan**: a list of subtasks, each with a description, an assigned specialist role, and a `depends_on` list. | `test_decomposition.py` |
| FR-2 | MUST | The plan shall be produced **at runtime by the LLM**, not read from a hardcoded template. Two distinct goals shall produce two distinct, valid subtask sets. | `test_decomposition.py` |
| FR-3 | MUST | The plan's dependency graph shall be well-formed: valid references, no cycles. | `test_decomposition.py` |
| FR-4 | MUST | Plan size shall be bounded (≤ configurable `MAX_SUBTASKS`, e.g. 6). | `config.py` + test |
| FR-5 | MUST | The supervisor shall decide when the goal is satisfied and route to completion. | `test_graph.py` |
| FR-6 | SHOULD | On a recovery that changes the plan (e.g. a source is unavailable), the supervisor shall re-plan around the failed subtask. | `test_recovery.py` |

### 3.2 Specialist Subagents

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-7 | MUST | The system shall implement exactly four specialists: **Researcher, Analyst, Critic, Writer**. No fifth specialist, no swarm. | Code review |
| FR-8 | MUST | Each specialist shall have its **own system prompt** and its **own working context**; a subagent sees only its assigned task plus relevant structured inputs — never one shared scratchpad. | Code review (§18.1 of plan) |
| FR-9 | MUST | Subagents shall communicate **through structured state fields and via the supervisor**, not by reading each other's raw context. | Code review |
| FR-10 | MUST | The Researcher shall gather evidence for its assigned sub-question via search/retrieval and emit `Evidence` items with sources. | `test_graph.py` |
| FR-11 | MUST | The Analyst shall synthesize relevant evidence into a structured analysis draft whose claims tie to evidence. | `test_graph.py` |
| FR-12 | MUST | The Writer shall compose the final cited brief from the passed analysis. | `test_graph.py` |

### 3.3 The Critic Loop (Real Disagreement)

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-13 | MUST | The Critic shall review the Analyst's draft against gathered evidence and return a structured **verdict: PASS or REJECT + specific feedback**. | `test_critic_loop.py` |
| FR-14 | MUST | On REJECT, the draft shall route **back to the Analyst** with feedback; the Analyst revises; the Critic reviews again. | `test_critic_loop.py` |
| FR-15 | MUST | The critic loop shall be bounded by `MAX_CRITIC_ITERS` (e.g. 3). On hitting the ceiling without PASS, the system shall degrade gracefully — proceed with the best draft flagged "not fully validated." | `test_critic_loop.py`, `test_loop_control.py` |
| FR-16 | MUST | A critic rejection shall be **triggerable on demand** for the demo (e.g. a deliberately weak first draft / test hook). | `test_critic_loop.py` |

### 3.4 Bounded Parallelism & Dependency Handling

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-17 | MUST | The scheduler shall dispatch subtasks whose dependencies are all `done`. | `test_parallelism.py` |
| FR-18 | MUST | Independent ready subtasks shall run **concurrently, capped at `MAX_PARALLEL` (2–3)**; the remainder queue. | `test_parallelism.py` |
| FR-19 | MUST | Concurrency shall **never exceed** `MAX_PARALLEL`. | `test_parallelism.py` |
| FR-20 | MUST | A dependent subtask shall **not start** before its dependencies are `done`. | `test_parallelism.py` |

### 3.5 Error Recovery

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-21 | MUST | A subagent shall be able to fail (tool error, empty result, rate limit) and report a **structured failure** (never raise raw into the graph). | `test_recovery.py` |
| FR-22 | MUST | On failure, the supervisor's recovery logic shall follow an explicit, logged path: (a) retry with backoff → (b) re-delegate / re-plan → (c) degrade gracefully. | `test_recovery.py` |
| FR-23 | MUST | An **injected** subagent failure shall route to recovery and the run shall still complete. | `test_recovery.py` |
| FR-24 | MUST | At least one tool shall support **fault injection** via a config/env switch so recovery is demonstrable on demand. | `tools/` + demo |

### 3.6 Loop & Cost Control

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-25 | MUST | A hard `MAX_STEPS` ceiling shall halt any run that exceeds it. | `test_loop_control.py` |
| FR-26 | MUST | `MAX_CRITIC_ITERS` and `MAX_RECOVERY_ATTEMPTS` shall be enforced. | `test_loop_control.py` |
| FR-27 | SHOULD | A loop detector shall notice repeated identical delegations and force completion. | `test_loop_control.py` |

### 3.7 Memory

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-28 | MUST | **Working memory** shall live in the graph state for the run's duration (`evidence`, `subtasks`, `analysis`, …). | Code review |
| FR-29 | MUST | **Long-term memory** shall persist distilled findings to a local vector store, tagged with a thread id, on completion. | Code review |
| FR-30 | MUST | Long-term memory shall be **read back and used** in a later turn — not merely written. A two-turn thread shall provably use a turn-1 finding in turn 2. | Two-turn demo |

### 3.8 Service (API)

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-31 | MUST | `POST /run` shall accept `{goal, thread_id?}` and **stream** supervisor delegations and subagent progress (SSE): plan → dispatch → subagent status → critic verdicts → final output. | Manual + demo |
| FR-32 | MUST | `GET /healthz` shall report liveness. | Manual |
| FR-33 | MUST | `thread_id` shall group turns and long-term memory. | Two-turn demo |
| FR-34 | SHOULD | Requests/responses shall be validated with Pydantic; each request shall log run id, plan, delegation path, critic iterations, recoveries, rate-limit retries, outcome. | Code review |

### 3.9 Observability & Replayable Runs

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-35 | MUST | Every meaningful event shall append a `TraceEvent` (plan_produced, subtask_dispatched, subagent_result, critic_pass, critic_reject, recovery, rate_limit_backoff, degraded, completed). | Code review |
| FR-36 | MUST | On completion, the full trace shall persist (SQLite or JSON) keyed by run id and be exportable, enabling **node-by-node replay** — including parallelism, critic pushback, and recovery. | `trace.py` + demo |

### 3.10 Graph & Diagram

| ID | Priority | Requirement | Verified by |
|---|---|---|---|
| FR-37 | MUST | The system shall be an explicit LangGraph `StateGraph` that compiles and runs the happy path end-to-end. | `test_graph.py` |
| FR-38 | MUST | The compiled graph shall be exported to `graph.png` via a script and embedded in README + ARCHITECTURE.md. | `scripts/render_graph.py` |

---

## 4. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | **Rate-limit resilience** | Every LLM/tool call shall be wrapped in exponential backoff with jitter (tenacity). A 429 triggers backoff-and-retry, not a crash. Retries are logged to the trace. |
| NFR-2 | **Concurrency** | Global concurrency shall be capped (`MAX_PARALLEL` = 2–3). No unbounded fan-out. |
| NFR-3 | **Cost** | The system shall run at ₹0 on the Gemini free tier; embeddings shall use a local model (`sentence-transformers`, e.g. `bge-small-en`). Cost is *tracked, not optimized*. |
| NFR-4 | **Reproducibility** | A clean clone shall be runnable in **< 10 minutes**; `uv sync` shall succeed from a clean clone. |
| NFR-5 | **Configurability** | Model IDs and all ceilings (`MAX_STEPS`, `MAX_CRITIC_ITERS`, `MAX_RECOVERY_ATTEMPTS`, `MAX_PARALLEL`, `MAX_SUBTASKS`) shall be settable in `config.py`. |
| NFR-6 | **Observability** | Structured logging throughout; the run trace is the audit log. |
| NFR-7 | **Security / hygiene** | No secrets in git history; the Gemini key via env only; `.gitignore` excludes `.env`, `*.db`, `memory_store/`, `__pycache__`, `.next`, `node_modules`. LICENSE is MIT. |
| NFR-8 | **Portability** | Single provider binding; model swappable via config. Python via `uv`, pinned versions. |
| NFR-9 | **Honest claims** | Every reported number (subtask counts, critic iterations, parallel width, latency) shall come from a reproducible real run. No invented numbers in the README. |

---

## 5. Constraints & Assumptions

- **Model:** Gemini Flash on the free tier (primary). Exact model ID and current free-tier RPM/TPM/RPD **must be verified live** before setting `MAX_PARALLEL` and backoff parameters (see `MAESTRO_BUILD_PLAN.md` §5).
- **Framework:** LangGraph. Its parallel/fan-out API (`Send` / conditional parallel edges) **must be verified against the installed version's docs** before implementing `graph.py` / `scheduler.py`.
- **Rate limits are the real friction:** RPM during parallel bursts, not RPD, is the binding constraint at demo scale. Backoff + concurrency cap solve it.
- **Single task domain:** research/analysis synthesis, committed on Day 1.
- **Read-only visibility:** no human-in-the-loop mid-run intervention.

---

## 6. Out of Scope (Explicit Non-Requirements)

- ❌ Production-grade RAG (hybrid retrieval, reranking, retrieval eval). Retrieval is deliberately simple.
- ❌ Formal trajectory-evaluation harness. Evaluation is lightweight (~15 hand-judged cases).
- ❌ Human-in-the-loop approval UI. Streaming is read-only.
- ❌ Fine-tuning or a cost-optimization layer.
- ❌ More than four specialist roles; any agent swarm.
- ❌ Unbounded fan-out.
- ❌ A fancy frontend (an optional minimal Tailwind viewer at most).

---

## 7. Acceptance Criteria (Definition of Done)

The project is done when **all** of the following hold and each is reproducible on demand:

- [ ] **Dynamic decomposition:** different goals → different valid plans (FR-1–4).
- [ ] **Separate contexts:** each subagent has its own prompt + context, verifiable in code (FR-8).
- [ ] **Graph compiles & runs:** happy path decomposes, delegates, composes a cited output (FR-37).
- [ ] **Graph diagram:** `graph.png` rendered and embedded (FR-38).
- [ ] **Parallelism:** independent subtasks run concurrently, cap respected, deps wait (FR-17–20).
- [ ] **Critic loop:** reject → revise → re-review; ceiling halts; triggerable on demand (FR-13–16).
- [ ] **Recovery:** injected failure → recovery → run completes (FR-21–24).
- [ ] **Rate limits:** every call backoff-wrapped; 429s handled; retries logged (NFR-1).
- [ ] **Ceilings:** `MAX_STEPS` / `MAX_CRITIC_ITERS` / `MAX_RECOVERY_ATTEMPTS` enforced (FR-25–26).
- [ ] **Memory:** written and read-back-and-used across a two-turn thread (FR-28–30).
- [ ] **Streaming service:** `/run` streams delegations + progress; `/healthz` live (FR-31–33).
- [ ] **Replayable trace:** parallelism, verdicts, recovery all reconstructable after the fact (FR-35–36).
- [ ] **All tests green:** `test_decomposition`, `test_parallelism`, `test_critic_loop`, `test_recovery`, `test_loop_control`, `test_graph`.
- [ ] **README + Loom:** orchestration-first README, embedded diagram, captured critic-rejection and recovery traces, limitations, reproduction commands, 90-second Loom.
- [ ] **Hygiene:** `.gitignore`, no secrets, `uv sync` clean, MIT license (NFR-4, NFR-7).

---

## 8. Traceability Summary

| Requirement area | FR / NFR IDs | Test / artifact |
|---|---|---|
| Decomposition | FR-1…6, FR-4 | `test_decomposition.py` |
| Subagents & contexts | FR-7…12, FR-8 | code review, `test_graph.py` |
| Critic loop | FR-13…16 | `test_critic_loop.py` |
| Parallelism | FR-17…20 | `test_parallelism.py` |
| Recovery | FR-21…24, FR-6 | `test_recovery.py` |
| Loop control | FR-25…27 | `test_loop_control.py` |
| Memory | FR-28…30 | two-turn demo |
| Service | FR-31…34 | manual/demo |
| Observability | FR-35…36 | `trace.py` + demo |
| Graph | FR-37…38 | `test_graph.py`, `render_graph.py` |
| Rate limits / cost / repro | NFR-1…9 | resilience code, README |
