"""Supervisor — dynamic decomposition (§10) and (later) delegation/completion/recovery.

This is what makes the "orchestrator that breaks down goals" claim true rather
than cosmetic. The supervisor asks the LLM for a **structured plan** (a Pydantic
schema, not free text): a list of subtasks each with a description, an assigned
specialist role, and a ``depends_on`` list. Different goals produce different
plans; the plan is validated (unique ids, valid references, no cycles, size bound)
before it ever enters the graph.

A deterministic ``heuristic_planner`` is provided for offline runs and tests — it
is a fallback, NOT the headline. The default production path is the LLM planner.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .logging_config import get_logger
from .resilience import OnRetry, resilient_call
from .state import Role, Subtask

log = get_logger("supervisor")

# One-line role guide injected into the planning prompt.
ROLE_GUIDE: dict[Role, str] = {
    Role.researcher: "gathers evidence for one specific sub-question via web search/retrieval; independent research subtasks can run in parallel",
    Role.analyst: "synthesizes the gathered evidence into a structured analysis/comparison; depends on the relevant researcher subtasks",
    Role.critic: "reviews the analysis for gaps, unsupported claims, and contradictions; can reject and return it; depends on the analyst",
    Role.writer: "composes the final cited brief; depends on a passed critique/analysis",
}

DECOMP_SYSTEM = """You are the Supervisor of a multi-agent research/analysis system.
Given a goal, decompose it into a tight set of subtasks and assign each to one specialist role.

Available specialist roles:
{roles}

Rules:
- Produce between 3 and {max_subtasks} subtasks. A good decomposition is tight, not sprawling.
- Give each subtask a short unique snake_case id (e.g. "research_x", "analyze", "write").
- `depends_on` lists the ids that must finish first. Independent research subtasks have an empty `depends_on`.
- Prefer 2+ INDEPENDENT researcher subtasks (so they can run in parallel), then one analyst that
  depends on them, then a critic that depends on the analyst, then a writer that depends on the critic.
- Do not create cycles. Every `depends_on` id must be a real subtask id.
Return ONLY the structured plan."""


# --------------------------------------------------------------------------- #
# Structured plan schema (LLM output target)
# --------------------------------------------------------------------------- #
class PlanSubtask(BaseModel):
    id: str
    description: str
    role: Role
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    subtasks: list[PlanSubtask]


class PlanValidationError(ValueError):
    """Raised when a produced plan is malformed (bad refs, cycle, size, dupes)."""


# planner signature: (goal, roles, max_subtasks) -> Plan
Planner = Callable[[str, list[Role], int], Plan]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _has_cycle(graph: dict[str, list[str]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}

    def dfs(n: str) -> bool:
        color[n] = GRAY
        for m in graph[n]:
            if color[m] == GRAY:
                return True
            if color[m] == WHITE and dfs(m):
                return True
        color[n] = BLACK
        return False

    return any(color[n] == WHITE and dfs(n) for n in graph)


def validate_and_build(plan: Plan, max_subtasks: int) -> list[Subtask]:
    """Validate a plan and convert it to ``Subtask`` objects (status=pending).

    Raises ``PlanValidationError`` on: empty plan, oversize plan, duplicate ids,
    unknown/self references, or a dependency cycle.
    """
    subs = plan.subtasks
    if not subs:
        raise PlanValidationError("plan has no subtasks")
    if len(subs) > max_subtasks:
        raise PlanValidationError(f"plan has {len(subs)} subtasks, exceeds max {max_subtasks}")

    ids = [s.id for s in subs]
    if len(set(ids)) != len(ids):
        raise PlanValidationError(f"duplicate subtask ids: {ids}")

    idset = set(ids)
    graph: dict[str, list[str]] = {}
    for s in subs:
        for d in s.depends_on:
            if d == s.id:
                raise PlanValidationError(f"subtask '{s.id}' depends on itself")
            if d not in idset:
                raise PlanValidationError(f"subtask '{s.id}' depends on unknown id '{d}'")
        graph[s.id] = list(s.depends_on)

    if _has_cycle(graph):
        raise PlanValidationError("plan dependency graph contains a cycle")

    return [
        Subtask(id=s.id, description=s.description, role=s.role, depends_on=list(s.depends_on))
        for s in subs
    ]


# --------------------------------------------------------------------------- #
# Planners
# --------------------------------------------------------------------------- #
def build_llm_planner(
    settings: Optional[Settings] = None,
    *,
    on_retry: Optional[OnRetry] = None,
) -> Planner:
    """The production planner: prompts Gemini for a structured Plan (backoff-wrapped)."""
    cfg = settings or get_settings()

    def planner(goal: str, roles: list[Role], max_subtasks: int) -> Plan:
        from .llm import get_chat_model  # lazy

        model = get_chat_model(cfg)
        structured = model.with_structured_output(Plan)
        role_lines = "\n".join(f"- {r.value}: {ROLE_GUIDE[r]}" for r in roles)
        system = DECOMP_SYSTEM.format(roles=role_lines, max_subtasks=max_subtasks)
        messages = [("system", system), ("human", f"Goal: {goal}")]
        return resilient_call(structured.invoke, messages, on_retry=on_retry)

    return planner


def _split_angles(goal: str) -> list[str]:
    """Derive two research angles from a goal (deterministic, goal-sensitive)."""
    m = re.search(
        r"(?:compare|between)\s+(.+?)\s+(?:and|vs\.?|versus|or)\s+(.+?)(?:\s+for\b|\s+on\b|[?.:]|$)",
        goal,
        re.I,
    )
    if not m:
        m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.:]|$)", goal, re.I)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    g = goal.rstrip("?.").strip()
    return [f"evidence supporting: {g}", f"evidence and counterpoints on: {g}"]


def heuristic_planner(goal: str, roles: list[Role], max_subtasks: int) -> Plan:
    """Deterministic, LLM-free planner for offline runs and tests.

    Not the headline — a fallback. Produces a realistic dependency shape:
    two independent researchers -> analyst -> critic -> writer.
    """
    angles = _split_angles(goal)
    subs: list[PlanSubtask] = []
    research_ids: list[str] = []
    for i, angle in enumerate(angles, start=1):
        rid = f"research_{i}"
        subs.append(
            PlanSubtask(id=rid, description=f"Gather evidence on {angle}", role=Role.researcher)
        )
        research_ids.append(rid)
    subs.append(
        PlanSubtask(
            id="analyze",
            description=f"Synthesize a structured comparison for: {goal}",
            role=Role.analyst,
            depends_on=research_ids,
        )
    )
    subs.append(
        PlanSubtask(
            id="critique",
            description="Review the analysis for gaps, unsupported claims, and contradictions",
            role=Role.critic,
            depends_on=["analyze"],
        )
    )
    subs.append(
        PlanSubtask(
            id="write",
            description=f"Compose the final cited brief for: {goal}",
            role=Role.writer,
            depends_on=["critique"],
        )
    )
    return Plan(subtasks=subs)


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def decompose(
    goal: str,
    *,
    planner: Optional[Planner] = None,
    settings: Optional[Settings] = None,
    on_retry: Optional[OnRetry] = None,
) -> list[Subtask]:
    """Decompose a goal into validated subtasks.

    Uses the LLM planner by default; pass ``planner=heuristic_planner`` (or any
    stub) for offline/deterministic runs.
    """
    cfg = settings or get_settings()
    plan_fn = planner or build_llm_planner(cfg, on_retry=on_retry)
    plan = plan_fn(goal, list(Role), cfg.max_subtasks)
    subtasks = validate_and_build(plan, cfg.max_subtasks)
    log.info("decomposed goal into %d subtasks: %s", len(subtasks), [s.id for s in subtasks])
    return subtasks
