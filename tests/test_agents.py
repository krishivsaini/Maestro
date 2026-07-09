"""Day 5 — Researcher + Analyst subagents with separate contexts (§7, §18.1).

Uses the StubModel from conftest so the LLM step is deterministic and offline.
Proves: the researcher turns tool output into sourced evidence; tool failure /
empty results yield a failed subtask (no raw raise); the analyst produces a draft
from evidence; and the two agents have distinct prompts + scoped contexts.
"""

import pytest

from maestro.agents import Analyst, Researcher
from maestro.agents.analyst import AnalysisModel
from maestro.agents.researcher import ResearchFinding, ResearchFindings
from maestro.state import Evidence, Role, Subtask, SubtaskStatus

CORPUS = [
    {"source": "src_solar", "content": "Solar PV capacity factor ~25%; output peaks midday; needs storage for evenings."},
    {"source": "src_wind", "content": "Wind capacity factor ~35%; variable; often stronger at night and in winter."},
    {"source": "src_cost", "content": "Grid battery storage costs have fallen roughly 80% over the decade."},
]


def research_responder(schema, messages):
    assert schema is ResearchFindings
    human = messages[-1][1]
    assert "Sub-question:" in human and "Raw results:" in human  # scoped researcher context
    return ResearchFindings(
        findings=[
            ResearchFinding(source="src_solar", content="Solar peaks midday; needs storage."),
            ResearchFinding(source="src_cost", content="Battery storage costs fell ~80%."),
        ]
    )


def analyst_responder(schema, messages):
    assert schema is AnalysisModel
    human = messages[-1][1]
    assert "Evidence:" in human
    return AnalysisModel(
        content="Solar and wind are complementary; storage bridges solar's evening gap.",
        claims=["storage is the swing factor", "wind has a higher capacity factor"],
    )


# --- Researcher -------------------------------------------------------------
def test_researcher_gathers_sourced_evidence(make_stub):
    r = Researcher(model=make_stub(research_responder))
    st = Subtask(id="r1", description="solar grid reliability", role=Role.researcher)
    updated, evidence = r.run(st, corpus=CORPUS)
    assert updated.status == SubtaskStatus.done
    assert len(evidence) == 2
    assert all(e.subtask_id == "r1" for e in evidence)
    assert all(e.source and e.content for e in evidence)
    assert updated.attempts == 1


def test_researcher_fails_on_empty_results(make_stub):
    r = Researcher(model=make_stub(research_responder))
    st = Subtask(id="r1", description="query with no corpus match", role=Role.researcher)
    updated, evidence = r.run(st, corpus=[])  # retrieve -> empty
    assert updated.status == SubtaskStatus.failed
    assert evidence == []
    assert updated.attempts == 1


def test_researcher_fails_on_tool_failure(make_stub, monkeypatch):
    from maestro.config import reset_settings_cache

    monkeypatch.setenv("MAESTRO_FAULT_INJECTION", "true")
    monkeypatch.setenv("MAESTRO_FAULT_INJECTION_TOOL", "retrieve")
    reset_settings_cache()  # pick up the injected fault
    r = Researcher(model=make_stub(research_responder))
    updated, evidence = r.run(
        Subtask(id="r1", description="x", role=Role.researcher), corpus=CORPUS
    )
    assert updated.status == SubtaskStatus.failed
    assert "injected_fault" in (updated.error or "")
    assert evidence == []


# --- Analyst ----------------------------------------------------------------
def test_analyst_produces_draft_from_evidence(make_stub):
    a = Analyst(model=make_stub(analyst_responder))
    st = Subtask(id="analyze", description="compare solar vs wind", role=Role.analyst, depends_on=["r1", "r2"])
    evidence = [
        Evidence(subtask_id="r1", source="src_solar", content="Solar peaks midday"),
        Evidence(subtask_id="r2", source="src_wind", content="Wind stronger at night"),
    ]
    updated, draft = a.run(st, "Compare solar and wind", evidence, revision=0)
    assert updated.status == SubtaskStatus.done
    assert draft.content and len(draft.claims) >= 1
    assert draft.revision == 0


def test_analyst_passes_feedback_into_context(make_stub):
    captured = {}

    def responder(schema, messages):
        captured["human"] = messages[-1][1]
        return AnalysisModel(content="revised", claims=["c"])

    a = Analyst(model=make_stub(responder))
    a.run(
        Subtask(id="analyze", description="d", role=Role.analyst),
        "goal",
        [Evidence(subtask_id="r1", source="s", content="e")],
        feedback="claim 2 is unsupported",
        revision=1,
    )
    assert "claim 2 is unsupported" in captured["human"]  # critic feedback reaches the analyst


# --- Separate contexts (§18.1) ---------------------------------------------
def test_agents_have_distinct_prompts():
    assert Researcher().system_prompt != Analyst().system_prompt
    assert Researcher().system_prompt.strip() and Analyst().system_prompt.strip()


def test_analyst_context_is_scoped_to_structured_evidence(make_stub):
    captured = {}

    def responder(schema, messages):
        captured["system"] = messages[0][1]
        captured["human"] = messages[-1][1]
        return AnalysisModel(content="x", claims=["c"])

    a = Analyst(model=make_stub(responder))
    ev = [Evidence(subtask_id="r1", source="src_solar", content="UNIQUE_EVIDENCE_TOKEN")]
    a.run(Subtask(id="analyze", description="d", role=Role.analyst), "goal", ev)
    assert "Analyst" in captured["system"]  # its own prompt
    assert "UNIQUE_EVIDENCE_TOKEN" in captured["human"]  # evidence handed via structured field
