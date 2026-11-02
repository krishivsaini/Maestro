# Maestro

[![tests](https://github.com/krishivsaini/Maestro/actions/workflows/tests.yml/badge.svg)](https://github.com/krishivsaini/Maestro/actions/workflows/tests.yml)

> **A supervisor-orchestrated multi-agent system with visible delegation and recovery.**
>
> A supervisor agent decomposes a goal into subtasks at runtime and delegates each to a
> role-specialized subagent (Researcher, Analyst, Critic, Writer), each with its own context.
> Independent subtasks run in bounded parallel; a Critic subagent can reject and return work until
> it passes; subagent failures recover visibly (retry → re-delegate → degrade); and every run is
> replayable node-by-node. Built on LangGraph, runs on the Gemini free tier with exponential
> backoff and a concurrency cap.

https://github.com/user-attachments/assets/4956f815-f2f3-4dc4-a0ac-8a024fb712ee

**[Graph diagram](#the-graph)** &nbsp;·&nbsp; **[Architecture](ARCHITECTURE.md)**

The task it performs — *"produce a cited analytical brief on a non-trivial question"* (e.g. *"Compare
PostgreSQL and MongoDB for a startup's first product, and recommend one"*) — is deliberately chosen
because it genuinely **requires** decomposition into different specialists: independent research that
parallelizes, an analysis that depends on it, a critique that can send the analysis back, and a final
composition. The multi-agent structure isn't decoration; the task doesn't reduce to a single prompt.

---

## The graph

![Maestro StateGraph](graph.png)

```
START → supervisor → research ─┬─▶ analyze ─▶ critique ─┬─(PASS / ceiling)─────────────▶ write → END
                               │       ▲                ├─(REJECT, rewrite can fix it)─▶ analyze
                    (failed)   ├─▶ recover ⟳ (retry)    └─(REJECT, evidence missing)──▶ replan
                               │       │                                                  │
                    (exhausted)└─▶ degrade ─▶ analyze ◀───── new research, then revise ───┘
```

Every edge is a real LangGraph edge. When the critic finds a gap that only new evidence can fill, it
attaches a research question; `replan` adds researcher subtasks for it mid-run (streamed as a
`plan_replanned` event), appends their evidence, and the analyst revises with it. One round per run by
default (`MAX_REPLANS`, at most `MAX_RESEARCH_REQUESTS` questions), so a run's cost stays bounded. The `research` node fans out to N researcher subtasks
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
| **Critic loop** | A structured review, not a vibe check. Code first checks what needs no model (uncited claims, citations to evidence that doesn't exist, figures that aren't in the evidence they cite, no recommendation when one is asked). The model then checks every claim against its cited evidence (supported, partial, overstated, miscited, unsupported, contradicted) and raises issues by severity; **the verdict is decided in code** from those findings. A re-review must confirm each fix and may only add new blockers, and a revision that fixes nothing stops the loop. Bounded by `MAX_CRITIC_ITERS`; degrades gracefully at the ceiling. |
| **Grounded agents** | The Analyst cites evidence numbers on every claim and revises from its previous draft against numbered issues. Researchers search with 1-3 queries the planner wrote for them, read the top pages rather than only search snippets, and keep the passages relevant to their question; findings are dropped if their source isn't one of the results they were shown, and near-duplicates are removed. The Writer leaves out claims the critic found unsupported and states unresolved issues as caveats. |
| **Visible recovery** | retry → re-delegate → degrade, bounded by `MAX_RECOVERY_ATTEMPTS`; every rung is a trace event. |
| **Rate-limit resilience** | Every LLM/tool call is wrapped in exponential backoff + jitter (tenacity), retrying 429s **and** transient 5xx/network errors — added after a live Gemini 503. |
| **Loop & cost control** | `MAX_STEPS` backstop + stall/repeat detectors so a multi-agent run can't spiral in calls or cost. |
| **Long-term memory** | Findings are written on completion and recalled during planning on a later turn of the same thread, from the same database as the runs (SQLite, or Postgres when deployed). Recall is **by meaning** (Gemini embeddings, findings and questions embedded as documents vs queries) with a word-matching fallback, so an embedding outage degrades recall instead of failing the run. On 12 follow-ups phrased in different words from their findings (`scripts/memory_eval.py`), meaning recall ranked the right finding first 12/12, word matching 1/12. |
| **Follow-up guardrail** | A follow-up must be about its thread's research. A free term-overlap check passes clear cases; otherwise one temperature-0 model call decides, and steering attempts ("ignore your instructions") always get the model check. Off-topic follow-ups are refused before any agent runs, with the reason and an offer to ask them as a new question. |
| **Observability** | Every transition is a `TraceEvent`; each run is persisted (SQLite, or Postgres when deployed) and **replayable node-by-node**, exportable as JSON. Every run also records **what it cost**: each model call's agent, tokens, latency and success (failed retries included) plus embedding calls, streamed with the result, kept with the run, and shown in the viewer's step details (a typical run: 6 calls, ~11k tokens). A dollar estimate appears only if you set per-token prices. |
| **Human in the loop** | Tick *Review the plan before research starts* and the run pauses after planning (a LangGraph interrupt): edit a researcher's question or searches, remove one, or add one, then start research. Every node is checkpointed (SQLite, or Postgres when deployed), so a paused run survives a restart, and a run that fails partway (say, quota spent at the analyst) offers **Resume**, continuing from its last finished step without redoing planning or research. `POST /runs/{id}/resume` does both. |
| **Service + viewer** | FastAPI `POST /run` streams the coordination as SSE; a minimal web viewer renders it live and replays past runs. |

---

## Reproducing

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
git clone https://github.com/krishivsaini/Maestro.git
cd Maestro
uv sync                       # deterministic install from uv.lock

# run the full test suite (offline, no API key) — 176 passed, 8 skipped
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
`gemini-3.5-flash-lite` with `gemini-3.6-flash` as the fuller-model alternative. The free tier meters
requests per model per day, so the alternative is a genuinely separate budget, switchable from the
viewer mid-run. The lite model leads because a shared public demo exhausts the full-size Flash quota
first, and it finishes a run in ~40s rather than minutes.

---

## Deploying

The service ships as a container: [`Dockerfile`](Dockerfile) builds it, [`render.yaml`](render.yaml)
is a one-click [Render](https://render.com) Blueprint for the free tier.

```bash
# build and run locally, exactly as the host will
docker build -t maestro .
docker run --rm -p 8000:8000 -e GOOGLE_API_KEY=... maestro
# open http://127.0.0.1:8000
```

**On Render:** push this repo to GitHub, then Dashboard → **New** → **Blueprint** → select the repo.
Render reads `render.yaml` and prompts for `GOOGLE_API_KEY` (stored encrypted, never committed).
No credit card required.

### Hosting the viewer separately

The viewer is a single static file with no build step, so it can be served from a CDN while the
API stays on Render. The page then loads instantly even when the backend has spun down, and says
so plainly instead of hanging:

> ● warming the orchestra… **12s** — the free tier sleeps after 15 min idle

1. Deploy `viewer/` to any static host (Cloudflare Pages, Netlify — both read `viewer/_headers`).
2. Set the API origin in `viewer/config.js`:
   ```js
   window.MAESTRO_API = "https://maestro-xxxx.onrender.com";
   ```
3. Point the API back at it — set `MAESTRO_CORS_ORIGINS` to the static host's origin in the Render
   dashboard. Empty by default, which leaves CORS off; that is correct while FastAPI serves the
   viewer itself, since same-origin needs no headers.

`?api=<origin>` overrides `config.js`, which is how you point a local copy of the page at a
deployed backend without editing anything.

None of this is required — FastAPI still serves the viewer at `/`, so the single container remains
self-contained.

The image is **about 130MB compressed** (about 600MB unpacked) because it omits the `embeddings` extra — `sentence-transformers` pulls torch
(~670MB installed), and long-term memory defaults to `HashingEmbedder`, so the deployed service never
needs it. Install it locally with `uv sync --extra embeddings` if you want real semantic embeddings.

Anything hosting this needs a **long-lived container**, not serverless functions: `POST /run` streams
SSE for the length of a multi-agent run, which function-style hosts (Vercel, Netlify) cut short.

What the free tier costs you, concretely:

| | |
|---|---|
| **512MB RAM / 0.1 CPU** | A measured run peaks at ~187MB, so memory is fine; the thin CPU slows cold start, not runs (the work is I/O-bound on Gemini). |
| **15-min idle spin-down** | The first visitor after a quiet spell waits ~1 min for the container to wake. Health checks don't keep it up. |
| **Ephemeral disk** | Without a database, past runs and thread memory live in `/tmp` and reset on every restart and deploy. Set `DATABASE_URL` (below) and both are stored in Postgres instead. |
| **Shared quota** | The server key is the free Gemini tier shared by every visitor, so runs are limited before any model call: 6 per visitor per hour and 60 per day on the shared key (`MAESTRO_RUN_LIMIT_PER_IP`, `MAESTRO_SHARED_DAILY_RUNS`). A visitor can add their own key in the viewer: it is used for that run only, never stored or logged, and their runs don't count toward the daily cap. |

### Keeping runs and memory across restarts

Create a free Postgres database (for example a [Neon](https://neon.tech) project; Supabase works too),
copy its connection string, and set it as `DATABASE_URL` in the Render dashboard. On Supabase, use the
**Session pooler** (or Transaction pooler) string from *Connect*, not the direct connection: the direct
one is IPv6-only on the free plan and Render can't reach it. `GET /healthz` reports which storage is live
(`"storage": "postgres"`), so a deploy can be checked without reading its logs; if the database can't be
reached, the service falls back to local storage and says so there instead of failing to start. Maestro creates its
tables on first start and stores runs and thread memory there; with no `DATABASE_URL` it uses SQLite
and files as before. Recall needs no vector extension (a thread holds a handful of findings, ranked in
Python), so any Postgres works.

### Safety on a public demo

- **Usage limits** as above, per visitor IP (behind Render's proxy, from `X-Forwarded-For`; set
  `MAESTRO_TRUST_PROXY=true` only behind a proxy, since anyone can send that header).
- **Validated input**: questions up to 500 characters, thread ids and keys checked, with messages a
  visitor can act on.
- **Web text is data**: researchers read third-party pages, so sentences addressing an AI ("ignore
  previous instructions…") are removed before a researcher sees them, the pages are marked as
  untrusted in its prompt, a finding that still carries instructions is dropped, and every agent that
  handles evidence treats it as data, not instructions.
- **Off-topic follow-ups** are refused before any agent runs (see the follow-up guardrail).

---

## Limitations (deliberate scope choices)

- **Retrieval is web search plus page reading**, not production RAG. Researchers search with planned
  queries and read the top pages (stdlib HTML extraction, relevant passages kept), but pages that block
  automated readers, render only with JavaScript, or are PDFs fall back to their search snippet; in a
  sample of 29 real result pages, 20 were readable.
- **Evaluation is lightweight** — ~15 scripted cases that demonstrate behavior (including failure and
  recovery), plus a 24-draft seeded-flaw set for the critic (below); not a metrics framework, and final
  output quality is hand-judged.
- **One task domain** (analytical briefs) — chosen because it genuinely decomposes across specialists;
  the orchestration generalizes, the prompts are domain-specific.
- **Free-tier concurrency cap** — `MAX_PARALLEL=2` because the Gemini free tier is ~10 RPM and one run
  makes 6–10 calls; the daily request cap means ~2 live runs/model/day. The parallelism is real and
  **scales to higher concurrency on a paid tier** by raising the cap — the design already bounds and
  backs off; only the number changes.

---

## Tests

`uv run pytest` → **176 passed, 8 skipped**, offline (LLM stubbed, no key/quota); the 5 Postgres variants of the storage tests run when `MAESTRO_TEST_DATABASE_URL` is set, as in CI. The hard parts each
have a proof: `test_decomposition`, `test_parallelism` (independent subtasks overlap, cap never
exceeded, dependents wait), `test_critic_loop`, `test_critic_review`, `test_recovery`, `test_loop_control`,
plus memory, trace-replay, service, guardrail and writer-citation tests.

### Recorded runs, replayed

Stubbed tests prove the mechanics, but not that the code still behaves on *real* model output.
`data/cassettes` holds recorded live runs: every model response and search result, and what the run
produced. `tests/test_replays.py` replays each one offline through the current code in about a second:
planning validation, scheduling, grounding and code checks, the critic's decision, routing, citations
all run for real; only the model and the search engine are played back. If the code now decides
differently on the same output, the test says exactly what changed.

```bash
python scripts/record_run.py "Compare solar and wind ..." --name solar-wind   # record (live)
python scripts/record_run.py --refresh solar-wind                            # accept a deliberate change
```

Every push runs the whole suite, replays included, in GitHub Actions: no key, no quota.

### Critic evaluation

`scripts/critic_eval.py` runs a critic over `data/critic_eval_cases.json`: per topic, a clean draft that is
correct against its evidence and five copies with one planted flaw each. A good critic rejects every flawed
draft and passes every clean one. The critic before the structured review is reproduced in the script, shown
the same citations, so the comparison measures the reviewing rather than access to information.

| Critic (model) | Flaws caught | Clean drafts passed |
|---|---|---|
| Before: PASS/REJECT verdict (`gemini-3.5-flash-lite`) | 12 / 20 | 4 / 4 |
| Structured review (`gemini-3.5-flash-lite`, the default) | 20 / 20 | 4 / 4 |
| Structured review (`gemini-3.6-flash`) | 19 / 20 | 4 / 4 |

The old critic caught none of the wrong citations. One topic (`k8s-swarm`) was written after the critic
was tuned on the other three and run once, to check the result isn't fitted to the tuning cases: the new
critic caught 5/5 there, the old one 3/5. Each run is 24 live calls, paced for the free tier's
15 requests/minute.
