"""Day 7 — the full LangGraph StateGraph, end-to-end (§9, FR-37).

Runs the whole supervisor -> research -> analyze <-> critique -> write graph with
stub-modeled agents (offline, deterministic). Proves: it compiles; the happy path
composes a cited final answer; the critic loop routes reject->revise->pass through
the graph; and the ceiling degrades gracefully instead of looping forever.
"""

import pytest

from maestro.agents import Analyst, Critic, Researcher, Writer
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import CriticOutput
from maestro.agents.researcher import ResearchFinding, ResearchFindings
from maestro.agents.writer import WriterOutput
from maestro.config import Settings
from maestro.graph import MaestroAgents, build_graph, run_goal
from maestro.state import CriticDecision, EventType, RunStatus, SubtaskStatus
from maestro.supervisor import heuristic_planner
from maestro.tools import DEFAULT_REGISTRY

GOAL = "Compare solar and wind energy for grid reliability, and recommend one."

CORPUS = [
    {"source": "src_solar", "content": "Solar PV capacity factor ~25%; output peaks midday; needs storage for evenings."},
    {"source": "src_wind", "content": "Wind capacity factor ~35%; variable; stronger at night and in winter."},
    {"source": "src_cost", "content": "Grid battery storage costs fell ~80% over the decade."},
]


def build_stub_agents(make_stub, critic_decisions):
    counter = {"ci": 0}

    def responder(schema, messages):
        name = getattr(schema, "__name__", "")
        if name == "ResearchFindings":
            return ResearchFindings(
                findings=[
                    ResearchFinding(source="src_solar", content="Solar peaks midday; needs storage."),
                    ResearchFinding(source="src_wind", content="Wind is stronger at night."),
                ]
            )
        if name == "AnalysisModel":
            human = messages[-1][1]
            tag = "revised" if "critic feedback" in human.lower() else "initial"
            return AnalysisModel(content=f"{tag} analysis of solar vs wind", claims=["c1", "c2"])
        if name == "CriticOutput":
            d = critic_decisions[min(counter["ci"], len(critic_decisions) - 1)]
            counter["ci"] += 1
            fb = "" if d == CriticDecision.passed else "claim c2 is unsupported; add a source"
            return CriticOutput(decision=d, feedback=fb)
        if name == "WriterOutput":
            return WriterOutput(content="FINAL BRIEF: solar and wind are complementary.", citations=["src_solar", "src_wind"])
        raise AssertionError(f"unexpected schema {name}")

    model = make_stub(responder)
    cfg = Settings(max_critic_iters=3, max_parallel=2)
    return MaestroAgents(
        researcher=Researcher(model=model, settings=cfg),
        analyst=Analyst(model=model, settings=cfg),
        critic=Critic(model=model, settings=cfg),
        writer=Writer(model=model, settings=cfg),
        settings=cfg,
        planner=heuristic_planner,
        tools=DEFAULT_REGISTRY,
        corpus=CORPUS,
    )


def test_graph_compiles(make_stub):
    graph = build_graph(build_stub_agents(make_stub, [CriticDecision.passed]))
    assert hasattr(graph, "invoke")


def test_happy_path_composes_cited_answer(make_stub):
    agents = build_stub_agents(make_stub, [CriticDecision.passed])
    final = run_goal(GOAL, agents=agents)

    assert final["status"] == RunStatus.completed.value
    answer = final["final_output"]
    assert answer is not None and answer.validated and answer.content
    assert answer.citations  # cited output
    assert len(final["evidence"]) > 0  # research produced evidence
    assert all(s.status == SubtaskStatus.done for s in final["subtasks"])  # all subtasks completed

    etypes = {e.event_type for e in final["trace"]}
    assert EventType.plan_produced in etypes
    assert EventType.critic_pass in etypes
    assert EventType.completed in etypes


def test_critic_loop_routes_through_graph(make_stub):
    agents = build_stub_agents(make_stub, [CriticDecision.rejected, CriticDecision.passed])
    final = run_goal(GOAL, agents=agents)

    assert final["status"] == RunStatus.completed.value
    assert final["final_output"].validated
    decisions = [v.decision for v in final["critic_verdicts"]]
    assert decisions == [CriticDecision.rejected, CriticDecision.passed]  # reject then pass
    assert final["analysis"].revision == 1  # analyst revised once inside the graph


def test_critic_ceiling_degrades_in_graph(make_stub):
    agents = build_stub_agents(make_stub, [CriticDecision.rejected])  # always rejects
    final = run_goal(GOAL, agents=agents)

    assert final["status"] == RunStatus.degraded.value
    answer = final["final_output"]
    assert answer is not None and not answer.validated and answer.notes  # produced but flagged
    assert final["critic_iterations"] == 3  # bounded at the ceiling
    assert sum(v.decision == CriticDecision.rejected for v in final["critic_verdicts"]) == 3
