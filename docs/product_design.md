# Maestro — Product Design

> **Status:** Draft v1 · Derived from `MAESTRO_BUILD_PLAN.md` (source of truth)
> **Scope of this doc:** The *product* — what Maestro is, who it's for, the experience it delivers, the task it performs, and the design principles that keep the "multi-agent" claim honest. Technical structure lives in `architecture.md`; the build order lives in `implementation_plan.md`.

---

## 1. Product Vision

**Maestro is multi-agent orchestration done honestly.**

Not renamed nodes in a pipeline — a real supervisor that plans at runtime, delegates to specialists with their own contexts, lets a critic subagent reject and return work, parallelizes what's independent, and recovers when a subagent fails.

The differentiator is **observable, recoverable multi-agent coordination**: you can *watch* the supervisor delegate, *watch* subagents work in parallel, *watch* the critic send work back, and *replay* the whole run node-by-node afterward.

> **Positioning line (lead every external artifact with this):**
> *"A supervisor-orchestrated multi-agent system with visible delegation and recovery."*
> Never *"autonomous research agent."*

---

## 2. Target User & Core Use Case

| | |
|---|---|
| **Primary audience** | A technical interviewer/reviewer evaluating whether the builder can design and defend a genuine multi-agent system. |
| **Operator** | Someone driving the demo: submits a goal, watches the stream, injects a failure. |
| **Core job-to-be-done** | "Show me — live — that this is really multi-agent: it decomposes, delegates to distinct specialists, disagrees via a critic, runs work in parallel, and recovers from failure." |

The product succeeds when a skeptical viewer, after a 90-second demo, believes the multi-agent claim *and* the honesty of its framing.

---

## 3. The Task the Product Performs

**Domain (committed): research / analysis synthesis.**

**Canonical task:** *Produce a cited analytical brief on a non-trivial question* — e.g. *"Compare approaches X and Y for problem Z and recommend one, with evidence."*

This naturally decomposes into:

- **Research subtasks** — gather evidence on X; gather evidence on Y → *independent, parallelizable*.
- **Analysis subtask** — synthesize a comparison → *depends on research*.
- **Critique subtask** — check the analysis for gaps / unsupported claims → *depends on analysis; can send it back*.
- **Writing subtask** — compose the final cited brief → *depends on a passed critique*.

**Why this task earns the architecture:**
- The dependency structure is *real* — some subtasks parallelize, some must sequence.
- The critic role is *natural* — reviewing analysis quality.
- Recovery is *natural* — a search returns nothing → re-delegate or degrade.
- It's *legible in 90 seconds*.

**A valid task must:** require ≥3 subtasks across ≥2 roles on a typical run; have ≥1 pair of independent subtasks; have a natural critic-rejection point; and have an injectable failure.

---

## 4. The Cast — Supervisor + Four Specialists

One conductor, four players. Each specialist has **its own system prompt and its own working context** — the separation that makes "multi-agent" honest.

| Agent | Role | Sees | Produces |
|---|---|---|---|
| 🎼 **Supervisor** | Decompose goal → subtasks with deps; assign roles; collect results; decide completion | Goal, subtask list + statuses, summarized results | Delegations; completion decision |
| 🔍 **Researcher** | Gather evidence for one sub-question via search/retrieval | Its sub-question + tool results | `Evidence` items with sources |
| 🧩 **Analyst** | Synthesize evidence into a structured comparison | Relevant research evidence | Analysis draft, claims tied to evidence |
| ⚖️ **Critic** | Review the analysis for gaps / unsupported claims / contradictions | Draft + cited evidence | Verdict: **PASS**, or **REJECT + feedback** |
| ✍️ **Writer** | Compose the final cited brief | Passed analysis + evidence | Final cited output |

**Design rules that protect the claim:**
1. Subagents talk **through structured state and the supervisor**, not each other's raw scratchpads.
2. The **Critic is the disagreement mechanism** — without a critic that can reject, this is just parallel workers.
3. Prompts must be genuinely role-scoped — near-identical prompts mean the agents aren't really distinct.

---

## 5. Key Experiences (The Demo Is the Product)

The interview value is that Maestro can be **driven live**. The streamed run *is* the product surface. Five experiences matter:

### 5.1 Watch it decompose
Submit a goal → the supervisor's plan streams in: subtasks, roles, dependencies. The viewer sees a *thinking* orchestrator, not a script.

### 5.2 Watch it parallelize
Two independent research subtasks dispatch and run **concurrently** (capped). The stream shows overlapping progress; dependent subtasks visibly wait.

### 5.3 Watch the critic disagree — **the money shot** 💥
The Critic rejects the Analyst's first draft with *concrete* feedback → the Analyst revises → the second draft passes. This single sequence is what proves "multi-agent, not pipeline."

### 5.4 Watch it recover 🛟
Inject a tool failure (search returns empty). The supervisor recovers **visibly**: retry → re-delegate → degrade — each an explicit logged transition. The run still completes.

### 5.5 Replay it 🔁
After the run, the persisted trace reconstructs the whole coordination node-by-node — which subtasks ran in parallel, where the critic pushed back, where recovery happened.

> **90-second Loom storyboard:** give a goal → watch decomposition → watch parallel research → watch the Critic reject and the Analyst revise → inject a failure → watch recovery → final cited brief.

---

## 6. Design Principles

1. **Honesty over impressiveness.** A precise, smaller, demoable claim beats an impressive one that collapses under one follow-up question.
2. **Observable by construction.** If a coordination decision isn't in the trace and the stream, it doesn't count.
3. **Bounded everything.** Steps, critic iterations, recovery attempts, and parallel width are all capped — bounding is a *feature*.
4. **Depth over breadth.** Four roles done well beats a ten-agent swarm.
5. **Recover, don't crash.** Tools return structured failures; the supervisor decides what they mean.
6. **Free-tier native.** Backoff + concurrency cap are part of the design, framed maturely as "scales to higher concurrency on a paid tier."

---

## 7. The Three Things That Must Stay True (Honest-Claims Anchor)

If any of these stops being true during the build, the "multi-agent" claim weakens — **stop and fix it**:

1. **Separate contexts** — each subagent has its own prompt and sees only its own task + relevant structured inputs, never one shared scratchpad.
2. **Real runtime delegation** — the supervisor decides subtasks at runtime; the plan never quietly becomes a fixed pipeline.
3. **A critic that can actually disagree** — a rejection is triggerable live; a critic that always passes is theater.

**Claims allowed if the above hold:** *"supervisor-orchestrated multi-agent system," "dynamic runtime decomposition," "role-specialized subagents with separate contexts," "a critic subagent that can reject and return work," "bounded-parallel dependency-aware execution," "visible recovery from subagent failure."*

**Claims to avoid unless literally true:** *"fully autonomous agents," "emergent behavior," "agents negotiate."* They *coordinate through a supervisor and shared state* — say that.

---

## 8. Success Metrics (Product-Level)

These are demonstration outcomes, not a metrics framework (evaluation is deliberately lightweight):

| Signal | Target |
|---|---|
| Live demo believability | A skeptical viewer accepts the multi-agent claim after 90s. |
| Critic rejection captured | A real draft → reject+feedback → revision → pass, screenshotted. |
| Recovery captured | A real injected failure → recovery → completion, screenshotted. |
| Parallelism visible | A run where two research subtasks provably overlap under the cap. |
| Memory recall | A two-turn thread where turn 2 provably uses a turn-1 finding. |
| Reproducibility | Clean clone → running demo in < 10 minutes. |

---

## 9. Scope Boundaries (Product View)

**Maestro is:** a supervisor-pattern multi-agent system — runtime decomposition, role-specialized subagents with separate contexts, bounded parallel execution, a critic that can reject and return work, visible recovery, replayable runs.

**Maestro is deliberately not:** a production RAG system, a formal eval harness, a fine-tuning/cost-optimization project, or an agent swarm. These are *deliberate scope choices*, framed as such in the README's "Limitations" section — a maturity signal, not a gap.
