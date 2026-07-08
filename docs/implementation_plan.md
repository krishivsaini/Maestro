# Maestro — Implementation Plan

> **Status:** Draft v1 · Derived from `MAESTRO_BUILD_PLAN.md` (source of truth)
> **Scope of this doc:** The *build order* — phases, per-module sequencing, the day-by-day plan, the testing strategy, and risk handling. Requirements live in `requirement.md`; structure in `architecture.md`.
> **Budget:** 15–17 days × ~3 hours/day, ₹0 on the Gemini free tier.

---

## 1. Guiding Principles for the Build

1. **Build resilience before parallelism.** `resilience.py` (backoff + jitter) exists before anything fans out — 429s appear immediately under parallel load.
2. **Verify live before coding the risky parts.** Gemini free-tier limits + model ID, and the LangGraph parallel/`Send` API, are checked against current docs *before* `config.py`, `graph.py`, `scheduler.py`.
3. **Protect the three anchors continuously** (separate contexts, runtime delegation, a critic that can disagree). If a shortcut breaks one, stop and fix it.
4. **If time slips, cut the optional viewer and any extra tool — never** the critic loop, parallelism, recovery, or rate-limit handling. Those are the differentiators.
5. **Every claim is backed by a reproducible run.** No invented numbers.

---

## 2. Phases (Milestone View)

The 17-day plan groups into six phases, each ending in a demonstrable capability.

| Phase | Days | Milestone (demonstrable) | Key modules |
|---|---|---|---|
| **P0 — Foundation** | 1–2 | Limits verified; task committed; state model with dependencies; every call backoff-wrapped; one tool can fail on demand. | `config.py`, `state.py`, `resilience.py`, `tools/` |
| **P1 — Orchestration core** | 3–4 | Two goals → two different valid plans; independent subtasks run concurrently under cap, deps wait. | `supervisor.py`, `scheduler.py`, `graph.py` |
| **P2 — The specialists** | 5–7 | Full happy path: plan → research∥ → analyze → critique → write → cited brief. | `agents/*` |
| **P3 — Resilience & control** | 8–9 | Injected failure recovers visibly; ceilings halt runaway runs. | `supervisor.py` (recover), `config.py`, loop detector |
| **P4 — Memory & observability** | 10–11 | Two-turn thread uses a stored finding; a completed run is fully replayable. | `memory/*`, `trace.py` |
| **P5 — Service, docs & demo** | 12–17 | Watchable streaming run; graph diagram; eval cases; hardening; README + Loom. | `serve.py`, `scripts/*`, docs, viewer (optional) |

---

## 3. Module Build Order & Contracts

Build modules in dependency order. Each entry lists what it must expose and its "done" signal.

1. **`config.py`** — caps (`MAX_STEPS`, `MAX_CRITIC_ITERS`, `MAX_RECOVERY_ATTEMPTS`, `MAX_PARALLEL`, `MAX_SUBTASKS`), model IDs, backoff params. *Done:* all tunables in one place, no magic numbers elsewhere.
2. **`state.py`** — `State`, `Subtask` (with `depends_on`, `status`), `Evidence`, `AnalysisDraft`, `Verdict`, `Answer`, `MemoryItem`, `TraceEvent`. *Done:* the `Subtask`/deps spine is right before any node is written.
3. **`resilience.py`** — backoff + jitter wrapper (tenacity) around every LLM/tool call; 429 → retry, logged to trace. *Done:* a forced 429 retries instead of crashing.
4. **`tools/registry.py` + `web_search.py` + `retrieve.py`** — typed schemas; tools return a result or a **structured failure**; one tool has a fault-injection switch. *Done:* the fault switch makes a tool fail on demand.
5. **`supervisor.py` (plan)** — structured decomposition via a Pydantic schema. *Done:* `test_decomposition.py` green.
6. **`scheduler.py`** — dependency resolution + bounded parallel dispatch (verify LangGraph `Send`/parallel-edge API first). *Done:* `test_parallelism.py` green.
7. **`graph.py`** — `StateGraph`: supervisor → scheduler → specialists → critic loop → writer → END; recovery edges. *Done:* compiles; happy path runs.
8. **`agents/base.py`** — shared subagent scaffolding (own prompt, own context, backoff).
9. **`agents/researcher.py`, `analyst.py`** — research → analysis draft from real evidence.
10. **`agents/critic.py`** — structured PASS/REJECT + feedback; reject routes back; bounded. *Done:* `test_critic_loop.py` green.
11. **`agents/writer.py`** — final cited composition. *Done:* end-to-end brief for a real goal.
12. **`supervisor.py` (recover)** — retry → re-delegate → degrade, each logged. *Done:* `test_recovery.py` green.
13. **Loop control** — `MAX_STEPS`, ceilings, loop detector. *Done:* `test_loop_control.py` green.
14. **`memory/working.py`, `longterm.py`** — in-state helpers; FAISS vector store, write on completion, read in planning. *Done:* two-turn recall works.
15. **`trace.py`** — record every event type; persist + export. *Done:* a run replays node-by-node.
16. **`serve.py`** — `POST /run` (SSE stream), `GET /healthz`, thread id, structured logging. *Done:* watchable real-time run.
17. **`scripts/render_graph.py`, `demo_cases.py`; `data/eval_cases.jsonl`** — diagram + ~15 eval cases.
18. **`viewer/` (optional)** — minimal streamed-trace view.

---

## 4. Day-by-Day Plan

| Day | Focus | End-of-day artifact |
|---|---|---|
| 1 | **Verify live** Gemini limits + model ID + LangGraph API; pick task; repo scaffold; `config.py`, `state.py` (incl. `Subtask`/deps). | Limits confirmed; task committed; state model defined. |
| 2 | `resilience.py` (backoff + jitter) **first**; tool registry + web search + simple retrieve; fault-injection switch. | Every call backoff-wrapped; one tool fails on demand. |
| 3 | Supervisor dynamic decomposition; structured plan output. | Two goals → two different valid plans; `test_decomposition.py` green. |
| 4 | `scheduler.py`: dependency resolution + bounded parallelism. | Independent subtasks run concurrently under cap; deps wait; `test_parallelism.py` green. |
| 5 | Researcher + Analyst with separate contexts. | Research → analysis path produces a draft from real evidence. |
| 6 | Critic subagent + critic loop: reject → revise → pass, bounded. | Weak draft rejected then revised; `test_critic_loop.py` green. |
| 7 | Writer subagent; full happy-path plan→research∥→analyze→critique→write. | End-to-end cited brief for a real goal. |
| 8 | Supervisor recovery: retry → re-delegate → degrade on injected failure. | Injected failure recovers visibly; `test_recovery.py` green. |
| 9 | Loop/cost control: `MAX_STEPS`, `MAX_CRITIC_ITERS`, `MAX_RECOVERY`; loop detector. | `test_loop_control.py` green; runs can't spiral. |
| 10 | Long-term memory: write on completion, read in planning; two-turn thread. | Turn 2 provably uses turn 1's stored finding. |
| 11 | `trace.py`: record every event type; persist + export replayable runs. | A completed run is fully replayable (parallelism + critic + recovery). |
| 12 | FastAPI `/run` streaming delegations + progress; `/healthz`; thread id. | Watchable real-time coordinated run end-to-end. |
| 13 | `render_graph.py` → `graph.png`; ARCHITECTURE.md; eval cases (~15). | Graph diagram + lightweight eval results. |
| 14 | Rate-limit hardening under real parallel load; tune `MAX_PARALLEL`/backoff; reproducibility pass. | Clean-clone run < 10 min; no unhandled 429s under demo load. |
| 15 | (Optional) minimal viewer; else polish streaming + logging. | Coordination legible to a viewer. |
| 16 | README + Loom (decompose → parallel → critic reject → recover) + resume bullet. | All output artifacts exist. |
| 17 | Buffer for the hardest parts (parallelism API, critic loop). | Slack day. |

---

## 5. Testing Strategy

Tests are the proof the hard parts work — they map 1:1 to the differentiators.

| Test | Proves | Assertions |
|---|---|---|
| `test_graph.py` | Graph compiles; happy path composes a cited final answer. | Compiles; run reaches `completed` with a non-empty cited output. |
| `test_decomposition.py` | Decomposition is dynamic. | Two goals → different subtask sets; deps well-formed (valid refs, no cycles). |
| `test_parallelism.py` | Bounded parallel + dependency-aware. | (a) two independent subtasks run concurrently; (b) cap never exceeded; (c) dependent subtask waits for `done` deps. |
| `test_critic_loop.py` | Real disagreement, bounded. | Weak first draft rejected; revision happens; later pass **or** ceiling halts; exchange in the trace. |
| `test_recovery.py` | Visible recovery. | Injected failure → recovery path → run completes. |
| `test_loop_control.py` | Ceilings bite. | Deliberately runaway cases halt at `MAX_STEPS` / `MAX_CRITIC_ITERS` / `MAX_RECOVERY_ATTEMPTS`. |

**Conventions:** deterministic where possible (fault injection and "weak draft" hooks make critic/recovery reproducible without depending on model whims); concurrency assertions use timing/overlap instrumentation from the trace; avoid live-LLM flakiness by asserting on structure, not exact wording.

---

## 6. Risk Register & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Gemini free-tier 429s under parallel bursts | High | High | Backoff+jitter built Day 2; `MAX_PARALLEL` cap; queue overflow; hardening pass Day 14. |
| LangGraph parallel/`Send` API differs from memory | High | Medium | Verify current docs before Days 4/7; the plan explicitly forbids relying on memory here. |
| Critic collapses into "always passes" | Medium | High (kills the claim) | On-demand rejection hook; `test_critic_loop.py` asserts a real reject. |
| Subagents collapse into one shared context | Medium | High (kills the claim) | Separate prompts/contexts enforced in `agents/base.py`; code-review checkpoint. |
| Plan silently becomes a fixed pipeline | Medium | High | `test_decomposition.py` requires distinct plans per goal. |
| Runs spiral in cost/loops | Medium | Medium | `MAX_STEPS` + ceilings + loop detector, tested. |
| Model ID / limits changed since plan written | Medium | Medium | Live verification Day 1 before hardcoding anything. |
| Scope creep (5th agent, production RAG, fancy UI) | Medium | Medium | Explicit non-goals; cut order defined (§1.4). |

---

## 7. Cut Order (If Time Runs Short)

Cut from the top; never cross the line:

1. Optional web `viewer/`.
2. Any tool beyond search + simple retrieve.
3. Extra eval cases (below the ~15 floor's non-essential ones).

**Never cut:** the critic loop · bounded parallelism · visible recovery · rate-limit handling. These are the differentiators and the honesty anchors.

---

## 8. Output Artifacts Checklist (Ship List)

| # | Artifact | Location |
|---|---|---|
| 1 | Public GitHub repo `maestro` | `github.com/<user>/maestro` |
| 2 | Orchestration-first README with embedded graph diagram | repo root |
| 3 | `ARCHITECTURE.md` + `graph.png` | repo root |
| 4 | Passing tests (decomposition, parallelism, critic, recovery, loop control) | `tests/` + CI |
| 5 | Captured critic-rejection trace (draft → reject → revise → pass) | README screenshot |
| 6 | Captured recovery trace (injected failure → recovery) | README screenshot |
| 7 | Replayable run traces showing parallelism | repo / demo |
| 8 | 90-second Loom | public link |
| 9 | Two-turn long-term-memory recall demo | demo / README |
| 10 | "Why LangGraph / why supervisor" rationale | README + ARCHITECTURE.md |
| 11 | Resume bullet with honest, demoable claims | resume file |
