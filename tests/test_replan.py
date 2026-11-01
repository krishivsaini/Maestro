"""Replanning: the critic can send work back to research, not only to the analyst.

Some gaps no rewrite can fix — a dimension the evidence never covers. The critic
attaches a research question to such an issue; the supervisor adds researcher
subtasks for it, their evidence is appended, and the analyst revises with it. One
round per run by default (MAX_REPLANS), at most MAX_RESEARCH_REQUESTS questions.
Offline: stub models, a small corpus.
"""

import re

from maestro.agents import Analyst, Critic, Researcher, Writer
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import CriticOutput, IssueOut
from maestro.agents.researcher import ResearchFinding, ResearchFindings
from maestro.agents.writer import WriterOutput
from maestro.config import Settings
from maestro.graph import MaestroAgents, run_goal
from maestro.state import CriticDecision, EventType, Role, RunStatus, Subtask
from maestro.supervisor import heuristic_planner
from maestro.tools import DEFAULT_REGISTRY

GOAL = "Compare solar and wind for grid reliability"
CORPUS = [
    {"source": "src_solar", "content": "Solar PV capacity factor ~25%; output peaks midday."},
    {"source": "src_wind", "content": "Wind capacity factor ~35%; stronger at night."},
    {"source": "src_cost", "content": "Grid battery storage costs fell ~80% over the decade."},
]
GAP = IssueOut(severity="blocker", criterion="coverage", problem="the draft never covers storage cost",
               research="How much does grid battery storage cost?")


def build(make_stub, reviews, *, invent_extra=False, cfg=None):
    """``reviews`` is a list of callables (messages) -> CriticOutput, one per review."""
    seen = {"analyst": [], "critic": 0}

    def responder(schema, messages):
        name, human = getattr(schema, "__name__", ""), messages[-1][1]
        if name == "ResearchFindings":
            question = re.search(r"Sub-question: (.*)", human).group(1)
            src = re.search(r"source=(\S+)", human).group(1)
            if invent_extra and "storage cost" in question:
                src = "https://invented.example"  # not among the results: the grounding check drops it
            return ResearchFindings(findings=[ResearchFinding(source=src, content=f"finding for: {question}")])
        if name == "AnalysisModel":
            seen["analyst"].append(human)
            return AnalysisModel(content="analysis", claims=["c1 [1]"])
        if name == "CriticOutput":
            n = seen["critic"]; seen["critic"] += 1
            return reviews[min(n, len(reviews) - 1)](messages)
        if name == "WriterOutput":
            return WriterOutput(content="BRIEF", citations=["src_solar"])
        raise AssertionError(name)

    model = make_stub(responder)
    cfg = cfg or Settings(max_critic_iters=3, max_parallel=2)
    agents = MaestroAgents(researcher=Researcher(model=model, settings=cfg), analyst=Analyst(model=model, settings=cfg),
                           critic=Critic(model=model, settings=cfg), writer=Writer(model=model, settings=cfg),
                           settings=cfg, planner=heuristic_planner, tools=DEFAULT_REGISTRY, corpus=CORPUS)
    return agents, seen


def ask(*issues):
    return lambda messages: CriticOutput.scripted(CriticDecision.passed, "", messages).model_copy(
        update={"issues": list(issues)})


def approve(messages):
    return CriticOutput.scripted(CriticDecision.passed, "", messages)


def test_a_gap_only_new_evidence_can_fix_triggers_more_research(make_stub):
    agents, seen = build(make_stub, [ask(GAP), approve])
    final = run_goal(GOAL, agents=agents)
    types = [e.event_type for e in final["trace"]]

    assert final["status"] == RunStatus.completed.value
    assert final["replans"] == 1 and final["critic_iterations"] == 2
    replan = next(e for e in final["trace"] if e.event_type == EventType.plan_replanned)
    assert replan.data["plan"] == [{"id": "research_more_1", "role": "researcher",
                                    "description": GAP.research, "depends_on": []}]
    extra = next(s for s in final["subtasks"] if s.id == "research_more_1")
    assert extra.status.value == "done"
    # the new evidence was appended (earlier evidence numbers unchanged) and the revision saw it
    assert final["evidence"][-1].content == f"finding for: {GAP.research}"
    assert f"finding for: {GAP.research}" in seen["analyst"][-1] and f"finding for: {GAP.research}" not in seen["analyst"][0]
    assert types.index(EventType.plan_replanned) < types.index(EventType.critic_pass)
    review = next(e for e in final["trace"] if e.event_type == EventType.critic_reject).data["review"]
    assert review["research"] == [GAP.research]


def test_only_one_extra_research_round_by_default(make_stub):
    second = IssueOut(severity="blocker", criterion="coverage", problem="no data on grid inertia",
                      research="How does solar affect grid inertia?")
    agents, seen = build(make_stub, [ask(GAP), ask(second), approve])
    final = run_goal(GOAL, agents=agents)
    assert final["replans"] == 1 and final["critic_iterations"] == 3
    assert sum(e.event_type == EventType.plan_replanned for e in final["trace"]) == 1
    assert not any(s.id.startswith("research_more_2") for s in final["subtasks"])  # second gap went to a plain revision


def test_extra_research_that_finds_nothing_degrades_the_run(make_stub):
    agents, _ = build(make_stub, [ask(GAP), approve], invent_extra=True)
    final = run_goal(GOAL, agents=agents)
    extra = next(s for s in final["subtasks"] if s.id == "research_more_1")
    assert extra.status.value == "degraded"
    assert final["status"] == RunStatus.degraded.value
    assert "partial evidence" in (final["final_output"].notes or "")


def test_no_research_rounds_when_the_budget_is_zero(make_stub):
    agents, _ = build(make_stub, [ask(GAP), approve], cfg=Settings(max_critic_iters=3, max_parallel=2, max_replans=0))
    final = run_goal(GOAL, agents=agents)
    assert final["replans"] == 0 and not any(e.event_type == EventType.plan_replanned for e in final["trace"])


def test_only_serious_issues_can_ask_for_research_and_requests_are_capped(make_stub):
    out = CriticOutput(issues=[
        IssueOut(severity="minor", criterion="coverage", problem="nice to have", research="q-minor"),
        IssueOut(severity="major", criterion="coverage", problem="gap a", research="q-a"),
        IssueOut(severity="major", criterion="coverage", problem="gap b", research="q-b"),
        IssueOut(severity="blocker", criterion="coverage", problem="gap c", research="q-c"),
    ])
    from maestro.state import AnalysisDraft, Evidence

    critic = Critic(model=make_stub(lambda s, m: out), settings=Settings(max_research_requests=2))
    _, v = critic.run(Subtask(id="critique", description="r", role=Role.critic), GOAL,
                      AnalysisDraft(content="x", claims=["c"], refs=[[1]]),
                      [Evidence(subtask_id="r", source="s", content="fact")])
    assert v.decision == CriticDecision.rejected and v.research == ["q-a", "q-b"]


def test_a_major_gap_with_a_research_question_is_researched_while_budget_remains(make_stub):
    major_gap = GAP.model_copy(update={"severity": "major"})  # alone, a major would pass
    agents, _ = build(make_stub, [ask(major_gap), approve])
    final = run_goal(GOAL, agents=agents)
    assert final["replans"] == 1 and final["critic_iterations"] == 2
    first = final["critic_verdicts"][0]
    assert first.decision == CriticDecision.rejected and first.research == [GAP.research]


def test_once_the_budget_is_spent_a_major_gap_passes_as_usual(make_stub):
    major_gap = GAP.model_copy(update={"severity": "major"})
    agents, _ = build(make_stub, [ask(major_gap), approve],
                      cfg=Settings(max_critic_iters=3, max_parallel=2, max_replans=0))
    final = run_goal(GOAL, agents=agents)
    assert final["replans"] == 0 and final["critic_iterations"] == 1
    assert final["critic_verdicts"][0].decision == CriticDecision.passed
