"""Day 3 — dynamic decomposition (§10, FR-1..FR-4).

Proves the supervisor produces *dynamic* plans: different goals yield different
valid subtask sets, and malformed plans are rejected. Offline tests use the
deterministic heuristic planner; a live test (skipped without a key) exercises
the real Gemini planner.
"""

import os

import pytest

from maestro.state import Role, SubtaskStatus
from maestro.supervisor import (
    Plan,
    PlanSubtask,
    PlanValidationError,
    decompose,
    heuristic_planner,
    validate_and_build,
)

GOAL_A = "Compare solar and wind energy for grid reliability, and recommend one."
GOAL_B = "Should a startup use PostgreSQL or MongoDB for its first product? Recommend one with evidence."


def test_decompose_produces_valid_plan():
    subs = decompose(GOAL_A, planner=heuristic_planner)
    assert len(subs) >= 3
    assert len({s.role for s in subs}) >= 2  # >=2 specialist roles
    assert any(s.depends_on == [] for s in subs)  # >=1 independent subtask
    assert all(s.status == SubtaskStatus.pending for s in subs)
    idset = {s.id for s in subs}
    for s in subs:  # every dependency references a real subtask
        assert set(s.depends_on).issubset(idset)


def test_different_goals_produce_different_plans():
    a = decompose(GOAL_A, planner=heuristic_planner)
    b = decompose(GOAL_B, planner=heuristic_planner)
    assert [s.description for s in a] != [s.description for s in b]


def test_dependency_structure_is_sane():
    subs = decompose(GOAL_A, planner=heuristic_planner)
    by_id = {s.id: s for s in subs}
    analysts = [s for s in subs if s.role == Role.analyst]
    assert analysts, "expected an analyst subtask"
    analyst = analysts[0]
    assert analyst.depends_on, "analyst should depend on research"
    for dep in analyst.depends_on:
        assert by_id[dep].role == Role.researcher


def test_rejects_cycle():
    plan = Plan(
        subtasks=[
            PlanSubtask(id="a", description="x", role=Role.researcher, depends_on=["b"]),
            PlanSubtask(id="b", description="y", role=Role.analyst, depends_on=["a"]),
        ]
    )
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_and_build(plan, max_subtasks=6)


def test_rejects_unknown_reference():
    plan = Plan(
        subtasks=[PlanSubtask(id="a", description="x", role=Role.researcher, depends_on=["ghost"])]
    )
    with pytest.raises(PlanValidationError, match="unknown"):
        validate_and_build(plan, max_subtasks=6)


def test_rejects_self_reference():
    plan = Plan(
        subtasks=[PlanSubtask(id="a", description="x", role=Role.researcher, depends_on=["a"])]
    )
    with pytest.raises(PlanValidationError, match="itself"):
        validate_and_build(plan, max_subtasks=6)


def test_rejects_oversize_plan():
    subs = [PlanSubtask(id=f"s{i}", description="x", role=Role.researcher) for i in range(7)]
    with pytest.raises(PlanValidationError, match="exceeds max"):
        validate_and_build(Plan(subtasks=subs), max_subtasks=6)


def test_rejects_duplicate_ids():
    subs = [
        PlanSubtask(id="dup", description="x", role=Role.researcher),
        PlanSubtask(id="dup", description="y", role=Role.analyst),
    ]
    with pytest.raises(PlanValidationError, match="duplicate"):
        validate_and_build(Plan(subtasks=subs), max_subtasks=6)


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="needs live Gemini key")
def test_llm_decomposition_live():
    """Opt-in: real Gemini planner produces a valid, role-diverse plan."""
    subs = decompose(GOAL_A)  # default planner = LLM
    assert len(subs) >= 3
    assert {s.role for s in subs} & {Role.researcher, Role.analyst}
