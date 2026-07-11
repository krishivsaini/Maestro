# Maestro

> **A supervisor-orchestrated multi-agent system with visible delegation and recovery.**
>
> A supervisor agent decomposes a goal into subtasks at runtime and delegates each to a
> role-specialized subagent (Researcher, Analyst, Critic, Writer), each with its own context.
> Independent subtasks run in bounded parallel; a Critic subagent can reject and return work until
> it passes; subagent failures recover visibly (retry → re-delegate → degrade); and every run is
> replayable node-by-node. Built on LangGraph, runs on the Gemini free tier with exponential
> backoff and a concurrency cap.

**[▶ Live demo / Loom](#)** &nbsp;·&nbsp; **[Graph diagram](#the-graph)** &nbsp;·&nbsp; **[Architecture](ARCHITECTURE.md)**

<!-- Loom link goes in the badge above once recorded. -->

The task it performs — *"produce a cited analytical brief on a non-trivial question"* (e.g. *"Compare
PostgreSQL and MongoDB for a startup's first product, and recommend one"*) — is deliberately chosen
because it genuinely **requires** decomposition into different specialists: independent research that
parallelizes, an analysis that depends on it, a critique that can send the analysis back, and a final
composition. The multi-agent structure isn't decoration; the task doesn't reduce to a single prompt.

---

## The graph

![Maestro StateGraph](graph.png)

```
START → supervisor → research ─┬─▶ analyze ─▶ critique ─┬─(PASS / ceiling)─▶ write → END
                               │                        │
                    (failed)   ├─▶ recover ⟳ (retry)    └─(REJECT & under ceiling)─▶ analyze
                               │       │
                    (exhausted)└─▶ degrade ─▶ analyze
```

Every edge is a real LangGraph edge. The `research` node fans out to N researcher subtasks
concurrently at runtime via a bounded thread pool (`MAX_PARALLEL`), then fans in — it renders as a
single node because the parallelism is internal to the node, not a graph-level `Send` branch.

### Why LangGraph for this orchestration

The supervisor pattern **is** a graph: a controller node routing to specialist nodes over a shared
typed state, with conditional edges for delegation, the critic loop, and recovery. Once a system
needs runtime routing between agents, a critic loop that sends work back, bounded parallel branches,
and recovery edges, hand-rolling that coordination means reinventing a graph/state engine — badly;
LangGraph is purpose-built for exactly that, so the build concentrates on the hard parts (making the
coordination observable, recoverable, and honestly multi-agent). See **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## What it actually does (real runs)

All traces below are **real output from the system**, not mock-ups.

### A live run — dynamic decomposition + parallel research + memory

A live Gemini free-tier run. The supervisor decomposed the goal at runtime (note the goal-specific
subtask ids), the two researchers ran concurrently, the critic passed, and distilled findings were
written to long-term memory for later turns:

```
goal: Should a startup use PostgreSQL or MongoDB for its first product?
status=completed  subtasks=5  critic_iters=1  recoveries=0

  plan_produced       supervisor    planned 5 subtasks: ['research_postgres', 'research_mongodb', 'analyze_comparison', 'critique_analysis', 'write_brief']
  subtask_dispatched  researcher    (research_postgres)   dispatched research_postgres
  subtask_dispatched  researcher    (research_mongodb)    dispatched research_mongodb
  subagent_result     researcher    (research_mongodb)    research_mongodb -> done      # ran concurrently,
  subagent_result     researcher    (research_postgres)   research_postgres -> done     # max_concurrency=2
  subagent_result     analyst       (analyze_comparison)  analysis draft rev 0
  critic_pass         critic        (critique_analysis)   review 1: PASS
  subagent_result     writer        (write_brief)         final brief composed
  memory_write        supervisor    stored 5 finding(s) to long-term memory
  completed           supervisor    run completed (completed)
```

### The critic can disagree — reject → revise → pass

The Critic reviews the draft against the evidence and can **REJECT with feedback**; the `critique →
analyze` edge routes it back to the Analyst, which revises, and the Critic reviews again (bounded by
`MAX_CRITIC_ITERS`). This is the "it's really multi-agent" signal — one agent overrules another:

```
status=completed  subtasks=5  critic_iters=2  recoveries=0

  ...
  subagent_result   analyst   (analyze)    analysis draft rev 0
  critic_reject     critic    (critique)   review 1: REJECT      verdict=REJECT
  subagent_result   analyst   (analyze)    analysis draft rev 1   ← revised on the critic's feedback
  critic_pass       critic    (critique)   review 2: PASS         verdict=PASS
  subagent_result   writer    (write)      final brief composed
  completed         supervisor             run completed (completed)
```

### Failures recover visibly — retry → re-delegate → degrade

When a researcher reports a **structured failure** (tools never raise raw into the graph), the
supervisor climbs an explicit, logged ladder. Here a search keeps failing; after the recovery
ceiling the subtask degrades and the run **still completes on partial evidence**, flagged honestly:

```
status=degraded  subtasks=5  critic_iters=1  recoveries=2

  subagent_result     researcher  (research_1)   research_1 -> failed
  recovery            supervisor  (research_1)   recovery attempt 1 (retry)        decision=retry
  subagent_result     researcher  (research_1)   research_1 -> failed
  recovery            supervisor  (research_1)   recovery attempt 2 (re-delegate)  decision=re-delegate
  subagent_result     researcher  (research_1)   research_1 -> failed
  degraded            supervisor  (research_1)   degraded after exhausting recovery; proceeding on partial evidence
  subagent_result     analyst     (analyze)      analysis draft rev 0
  critic_pass         critic      (critique)     review 1: PASS
  completed           supervisor                 run completed (degraded)
```

Both the critic-rejection and recovery paths are triggerable on demand for a demo
(`MAESTRO_FORCE_CRITIC_REJECT=1`, `MAESTRO_FAULT_INJECTION=true`).

---

## Design highlights

| Concern | How it's handled |
|---|---|
| **Dynamic decomposition** | LLM returns a structured Pydantic `Plan` (subtasks + roles + `depends_on`), validated for cycles/dupes/size. Different goals → structurally different plans. |
| **Separate contexts** | Each subagent has its own system prompt and sees only its task + relevant structured inputs — the only cross-agent hand-off is structured data via the supervisor. |
| **Bounded parallelism** | Independent researchers run concurrently in a thread pool capped at `MAX_PARALLEL` (default 2); dependents wait. Concurrency is *observed* so the claim is measurable. |
| **Critic loop** | Real REJECT-with-feedback that routes work back, bounded by `MAX_CRITIC_ITERS`; degrades gracefully at the ceiling. |
| **Visible recovery** | retry → re-delegate → degrade, bounded by `MAX_RECOVERY_ATTEMPTS`; every rung is a trace event. |
| **Rate-limit resilience** | Every LLM/tool call is wrapped in exponential backoff + jitter (tenacity), retrying 429s **and** transient 5xx/network errors — added after a live Gemini 503. |
| **Loop & cost control** | `MAX_STEPS` backstop + stall/repeat detectors so a multi-agent run can't spiral in calls or cost. |
| **Long-term memory** | FAISS inner-product index; findings written on completion, recalled during planning on a later turn of the same thread. Swappable embedder (local `bge-small`, or a no-torch hashing embedder for tests). |
| **Observability** | Every transition is a `TraceEvent`; each run is persisted to SQLite and **replayable node-by-node**, exportable as JSON. |
| **Service + viewer** | FastAPI `POST /run` streams the coordination as SSE; a minimal web viewer renders it live and replays past runs. |

---

## Reproducing

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
git clone https://github.com/krishivsaini/Maestro.git
cd Maestro
uv sync                       # deterministic install from uv.lock

# run the full test suite (offline, no API key) — 62 passed, 1 skipped
uv run pytest

# see the coordination in the browser
cp .env.example .env          # then add your GOOGLE_API_KEY (free tier)
uv run uvicorn maestro.serve:create_app --factory
# open http://127.0.0.1:8000 and give it a goal

# or run the lightweight eval cases (typical / parallelism / critic / recovery / ceiling / memory)
uv run python scripts/demo_cases.py list
uv run python scripts/demo_cases.py run crit-01     # watch a forced critic rejection
```

The model id is swappable in `maestro/config.py` (or `MAESTRO_MODEL_ID`); the free tier is
`gemini-3.5-flash` with `gemini-3.1-flash-lite` as a higher-throughput fallback.

---

## Limitations (deliberate scope choices)

- **Retrieval is simple**, not production RAG — web/knowledge lookups + a small local corpus, enough
  to produce evidence-backed briefs, not a tuned retrieval stack.
- **Evaluation is lightweight** — ~15 scripted cases that demonstrate behavior (including failure and
  recovery), not a metrics framework; final output quality is hand-judged.
- **One task domain** (analytical briefs) — chosen because it genuinely decomposes across specialists;
  the orchestration generalizes, the prompts are domain-specific.
- **Free-tier concurrency cap** — `MAX_PARALLEL=2` because the Gemini free tier is ~10 RPM and one run
  makes 6–10 calls; the daily request cap means ~2 live runs/model/day. The parallelism is real and
  **scales to higher concurrency on a paid tier** by raising the cap — the design already bounds and
  backs off; only the number changes.

---

## Tests

`uv run pytest` → **62 passed, 1 skipped**, offline (LLM stubbed, no key/quota). The hard parts each
have a proof: `test_decomposition`, `test_parallelism` (independent subtasks overlap, cap never
exceeded, dependents wait), `test_critic_loop`, `test_recovery`, `test_loop_control`, plus memory,
trace-replay, service, and writer-citation tests.
