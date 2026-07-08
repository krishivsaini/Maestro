# Maestro — Architecture

> **Status:** Draft v1 · Derived from `MAESTRO_BUILD_PLAN.md` (source of truth)
> **Scope of this doc:** The *technical structure* — tech stack and rationale, the graph, the state model, concurrency, resilience, memory, and observability. The product intent lives in `product_design.md`; the build order in `implementation_plan.md`.
> **Note:** This planning doc feeds the eventual repo-root `ARCHITECTURE.md`, which additionally embeds the rendered `graph.png`.

---

## 1. System Overview

Maestro is a **supervisor-pattern multi-agent system** implemented as an explicit **LangGraph `StateGraph`**. A controller node (the supervisor) plans at runtime and routes work to role-specialized subagent nodes over a shared typed state. Conditional edges implement delegation, the critic loop, and recovery; parallel branches implement bounded fan-out.

```
                         ┌─────────────────────────────────────────────┐
   goal ──▶  FastAPI /run │  StateGraph (LangGraph)                     │──▶ streamed
   thread_id              │                                             │    trace (SSE)
                          │   Supervisor ─ Scheduler ─ Specialists      │
                          │        ▲            │           │           │
                          │        └── recovery ┘   critic loop         │
                          └───────────────┬─────────────────────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                     ▼
              Tool registry        Long-term memory        Trace store
           (search, retrieve)      (FAISS + local emb.)   (SQLite / JSON)
```

---

## 2. Tech Stack & Rationale

| Concern | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph `StateGraph`** | The supervisor pattern *is* a graph: named nodes, typed shared state, conditional edges, parallel branches. See §3. |
| LLM | **Gemini Flash (free tier)** | ₹0 to run; single provider binding, model ID swappable via `config.py`. Verify live model ID + limits. |
| LLM binding | `langchain-google-genai` | Gemini binding; message/tool types via `langchain-core`. |
| Retries | **`tenacity`** | Exponential backoff + jitter on every LLM/tool call. |
| Structured output | **Pydantic** | Parseable plans, verdicts, evidence — not free text. |
| Search | one free-tier tool (Tavily / SerpAPI / DuckDuckGo) | Evidence gathering; one tool carries a fault-injection switch. |
| Long-term memory | **FAISS** + local embeddings (`sentence-transformers`, e.g. `bge-small-en`) | Fully free; no API cost for the memory layer. |
| Service | **FastAPI + uvicorn**, SSE streaming | Streams delegations + subagent progress in real time. |
| Persistence (trace) | **SQLite or JSON** keyed by run id | Replayable runs. |
| Packaging | **`uv`**, pinned in `pyproject.toml` | Reproducible clean-clone install. |

### 2.1 Why LangGraph (why this isn't over-engineering)

Once the system needs *runtime routing between multiple agents, a critic loop that sends work back, bounded parallel branches, and recovery edges over a shared typed state*, hand-rolling that coordination means reinventing a graph/state engine — badly. LangGraph is purpose-built for exactly this. The interesting engineering is not the framework; it's making the coordination **observable, recoverable, and honestly multi-agent**. Because the project genuinely needs and uses LangGraph — and the "why" is explainable — listing it as a skill is *earned*, not padded.

> **Verify before coding:** LangGraph's parallel/fan-out API (`Send` / conditional parallel edges), checkpointer, and streaming surface change across versions. Check the installed version's docs before writing `graph.py` and `scheduler.py`.

---

## 3. The Graph (`graph.py`)

Conceptual shape (verify exact parallel/routing API against current LangGraph docs):

```
START → supervisor(plan) → scheduler ──▶ [Researcher ∥ Researcher ∥ …]   (independent, bounded-parallel)
                                │                     │
                                │                     ▼
                                └──────────────▶  Analyst   (after research deps met)
                                                       │
                                                       ▼
                                                    Critic ──▶ (REJECT) ──▶ back to Analyst   (bounded loop)
                                                       │
                                                    (PASS)
                                                       ▼
                                                    Writer ──▶ (optional final Critic pass) ──▶ END

any subagent failure ──▶ supervisor(recover) ──▶ retry / re-delegate / degrade   (bounded)
```

| Node / edge | Responsibility |
|---|---|
| **supervisor(plan)** | Dynamic decomposition (§5): produces `subtasks` with dependencies + role assignments. |
| **scheduler** | Dispatches ready subtasks (deps satisfied) with bounded parallelism (§6). |
| **specialist nodes** | Researcher / Analyst / Critic / Writer, each its own agent with its own prompt + context. |
| **critic edge** (conditional) | REJECT → back to Analyst (or Writer) with feedback; PASS → advance. Bounded by `MAX_CRITIC_ITERS`. |
| **recovery edge** (conditional) | Any subagent failure → supervisor recover logic. Bounded by `MAX_RECOVERY_ATTEMPTS`. |

The compiled graph is exported to `graph.png` via `scripts/render_graph.py` for the README and ARCHITECTURE.md.

---

## 4. State Model (`state.py`) — The Shared Blackboard

The typed state is what the supervisor and subagents coordinate through. Defined explicitly (Pydantic / `TypedDict`) before any node is written.

| Field | Type | Purpose |
|---|---|---|
| `goal` | `str` | The user's goal. |
| `subtasks` | `list[Subtask]` | The plan spine — see below. |
| `evidence` | `list[Evidence]` | Researcher findings: `{source, content, subtask_id}`. |
| `analysis` | `Optional[AnalysisDraft]` | The Analyst's current draft. |
| `critic_verdicts` | `list[Verdict]` | History of PASS/REJECT + feedback (bounded). |
| `critic_iterations` | `int` | Critic-loop ceiling counter. |
| `final_output` | `Optional[Answer]` | The Writer's cited composition. |
| `step_count` | `int` | Global loop control. |
| `recovery_attempts` | `int` | Recovery ceiling counter. |
| `memory_hits` | `list[MemoryItem]` | What long-term memory returned this run. |
| `trace` | `list[TraceEvent]` | Appended by every node/agent (§9). |

**`Subtask`** — the spine of the whole system:

```
Subtask {
  id: str
  description: str
  role: "researcher" | "analyst" | "critic" | "writer"
  depends_on: list[str]     # ids that must be `done` first
  status: "pending" | "running" | "done" | "failed" | "degraded"
  result: Optional[...]
}
```

`depends_on` is what makes decomposition *dynamic* (§5) and scheduling *dependency-aware* (§6). Get it right before writing the graph.

---

## 5. Dynamic Decomposition (`supervisor.py`)

- The supervisor prompts the LLM with the goal + available roles and asks for a **structured plan** (Pydantic schema): a list of subtasks each with description, assigned role, and `depends_on`.
- **Different goals must produce different plans.** The plan graph must be well-formed: valid references, no cycles.
- Plan size is bounded (`MAX_SUBTASKS`, e.g. 6) for rate-limit sanity and because a good decomposition is tight.
- On a recovery that changes the plan (e.g. an unavailable source), the supervisor re-plans around the failed subtask.

> **Anti-goal:** a hardcoded subtask list with an LLM call bolted on top. That is the non-dynamic version this project rejects.

---

## 6. Concurrency & Scheduling (`scheduler.py`)

- The scheduler finds subtasks whose `depends_on` are all `done` and dispatches them.
- **Independent ready subtasks run concurrently, capped at `MAX_PARALLEL` (2–3);** the rest queue. This is *both* the parallelism signal and the rate-limit protection.
- **Dependent subtasks wait** until dependencies complete — the dependency-awareness signal.
- Implementation uses LangGraph's native fan-out (verify the current `Send` / conditional-parallel-edge API against installed-version docs before building).

**Invariants (`test_parallelism.py`):** (a) two independent subtasks actually overlap; (b) concurrency never exceeds the cap; (c) a dependent subtask never starts before its deps are `done`.

---

## 7. Resilience, Rate Limits & Loop Control (`resilience.py`, `supervisor.py`, `config.py`)

A rare, high-signal cluster — most multi-agent demos have none of this.

**Rate-limit resilience (mandatory):**
- `resilience.py` wraps every LLM/tool call in exponential backoff + jitter (tenacity). A 429 → backoff-and-retry, not a crash. Retries are logged to the trace (visible, not silent).
- The `MAX_PARALLEL` cap prevents most 429s at the source.

**Subagent recovery (explicit, logged transitions):**
1. **Retry** with backoff up to a bound.
2. **Re-delegate / re-plan** around the failed subtask (e.g. an alternative source).
3. **Degrade gracefully** — mark the subtask "could not complete," proceed with partial evidence, flag it in the output.

**Loop & cost control:**
- `MAX_STEPS` halts any run that exceeds it.
- `MAX_CRITIC_ITERS` bounds the critic loop.
- `MAX_RECOVERY_ATTEMPTS` bounds recovery.
- A loop detector notices repeated identical delegations and forces completion.

Tools return **structured failures**, never raw exceptions into the graph — the *supervisor* decides what a failure means.

---

## 8. The Critic Loop (`agents/critic.py`)

- The Critic reviews the Analyst's draft against gathered evidence → structured **verdict: PASS or REJECT + specific feedback** (what's unsupported / missing / contradictory).
- On REJECT, the draft returns to the Analyst with feedback; the Analyst revises; the Critic re-reviews.
- Bounded by `MAX_CRITIC_ITERS`. On hitting the ceiling without PASS, degrade: proceed with the best draft, flagged "not fully validated." Never loop unbounded.
- A rejection must be **triggerable on demand** (weak-draft/test hook) so the demo money-shot is reproducible.

---

## 9. Memory Architecture (`memory/`)

| Layer | Lifetime | Backing | Role |
|---|---|---|---|
| **Working memory** | One run | Graph state (`evidence`, `subtasks`, `analysis`, …) | Coordination during the run; no persistence. |
| **Long-term memory** | Across turns in a thread | FAISS + local embeddings | On completion, write distilled findings tagged with `thread_id`. On a later turn, the supervisor queries it during planning; results land in `memory_hits` and inform decomposition + synthesis. |

**Non-negotiable:** long-term memory must be **read back and used**, not just written — proven by a two-turn thread where turn 2's plan/answer provably uses a turn-1 finding.

---

## 10. Observability — Replayable Runs (`trace.py`)

- Every meaningful event appends a `TraceEvent`:
  `{event_type, agent, timestamp, summary, subtask_id?, tool_call?, error?, critic_verdict?, recovery_decision?, rate_limit_retry?}`.
- **Event types:** `plan_produced`, `subtask_dispatched`, `subagent_result`, `critic_pass`, `critic_reject`, `recovery`, `rate_limit_backoff`, `degraded`, `completed`.
- On completion, the full trace persists (SQLite or JSON) keyed by run id and is exportable — enabling **node-by-node replay**: which subtasks ran in parallel, where the critic pushed back, where recovery happened. This is the production-thinking signal: a full audit log you can reconstruct after the fact.

---

## 11. The Service (`serve.py`)

| Endpoint | Behavior |
|---|---|
| `POST /run` | Accepts `{goal, thread_id?}`; **streams** (SSE) supervisor delegations and subagent progress: plan produced → subtasks dispatched → each subagent's status → critic verdicts → final output. |
| `GET /healthz` | Liveness. |

- `thread_id` groups turns and long-term memory.
- Pydantic validation throughout; structured logging records run id, plan, delegation path, critic iterations, recoveries, rate-limit retries, and final outcome.
- Streaming is what makes a 90-second demo legible instead of opaque — it is a required capability, not a nicety.

---

## 12. Configuration Surface (`config.py`)

All tunables live here — no magic numbers elsewhere:

| Setting | Purpose |
|---|---|
| `MODEL_ID` | Gemini Flash model id (verify live). |
| `EMBEDDING_MODEL` | Local embedding model for long-term memory. |
| `MAX_PARALLEL` (2–3) | Concurrency cap. |
| `MAX_SUBTASKS` (~6) | Plan-size bound. |
| `MAX_STEPS` | Global loop ceiling. |
| `MAX_CRITIC_ITERS` (~3) | Critic-loop ceiling. |
| `MAX_RECOVERY_ATTEMPTS` | Recovery ceiling. |
| backoff params | Base delay, max delay, jitter for tenacity. |
| fault-injection switch | Enables on-demand tool failure for the recovery demo. |

---

## 13. Repository Layout (Target)

```
maestro/
├── README.md                 # orchestration-first; the hiring-decision artifact
├── ARCHITECTURE.md           # rendered graph diagram + why-LangGraph/why-supervisor
├── graph.png                 # exported LangGraph diagram (regenerated by a script)
├── pyproject.toml            # uv-managed
├── .env.example              # GOOGLE_API_KEY=, search API key
│
├── maestro/
│   ├── state.py              # shared State incl. subtasks + verdicts
│   ├── graph.py              # StateGraph: supervisor + subagent nodes + critic loop
│   ├── supervisor.py         # decomposition + delegation + completion + recovery
│   ├── agents/               # base, researcher, analyst, critic, writer
│   ├── scheduler.py          # bounded-parallel, dependency-aware execution
│   ├── tools/                # registry, web_search (fault-injectable), retrieve (simple)
│   ├── memory/               # working (in-state), longterm (FAISS)
│   ├── resilience.py         # backoff + jitter + rate-limit handling
│   ├── serve.py              # FastAPI streaming + healthz + thread id
│   ├── trace.py              # run-trace recording + export
│   ├── logging_config.py     # structured logging
│   └── config.py             # all caps + model names
│
├── viewer/                   # OPTIONAL minimal web viewer (Tailwind core)
├── tests/                    # graph, decomposition, parallelism, critic_loop, recovery, loop_control
├── scripts/                  # render_graph.py, demo_cases.py
└── data/                     # eval_cases.jsonl (~15 cases)
```

---

## 14. Architectural Invariants (Do Not Violate)

1. **Separate contexts** — each subagent has its own prompt and sees only its task + relevant structured inputs. Never one shared scratchpad. *(Breaks the "multi-agent" claim if violated.)*
2. **Runtime delegation** — the plan is LLM-produced per goal, never a hardcoded pipeline.
3. **A critic that can disagree** — rejection is real and triggerable on demand.
4. **Bounded parallelism** — concurrency capped; never fan out unbounded.
5. **Backoff on every call** — no LLM/tool call is unwrapped.
6. **Ceilings enforced** — `MAX_STEPS` / `MAX_CRITIC_ITERS` / `MAX_RECOVERY_ATTEMPTS`.
7. **Structured failures** — tools never raise raw into the graph; the supervisor decides meaning.
8. **Everything traced** — no coordination decision is invisible.
