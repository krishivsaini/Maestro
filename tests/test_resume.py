"""Checkpoints: approve or edit the plan before research, and resume a failed run.

With a checkpointer, the run's state is saved after every node. That lets a run pause
after planning for a person to review the plan, and lets a run that failed partway
continue from its last finished node instead of starting over.
"""

import re

import pytest
from langgraph.checkpoint.memory import MemorySaver

from maestro.agents import Analyst, Critic, Researcher, Writer
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import CriticOutput
from maestro.agents.researcher import ResearchFinding, ResearchFindings
from maestro.agents.writer import WriterOutput
from maestro.config import Settings
from maestro.graph import MaestroAgents, apply_plan_edits, build_graph
from maestro.state import CriticDecision, EventType, Role, RunStatus, Subtask, SubtaskStatus, new_state
from maestro.supervisor import heuristic_planner
from maestro.tools import DEFAULT_REGISTRY

GOAL = "Compare solar and wind for grid reliability"
CORPUS = [{"source": "src_solar", "content": "Solar capacity factor ~25%; peaks midday."},
          {"source": "src_wind", "content": "Wind capacity factor ~35%; stronger at night."},
          {"source": "src_cost", "content": "Battery storage costs fell sharply."}]


def build(make_stub, *, analyst_down=False):
    calls = {"plan": 0, "research": [], "analyst": 0, "down": analyst_down}

    def planner(goal, roles, n):
        calls["plan"] += 1
        return heuristic_planner(goal, roles, n)

    def responder(schema, messages):
        name, human = getattr(schema, "__name__", ""), messages[-1][1]
        if name == "ResearchFindings":
            q = re.search(r"Sub-question: (.*)", human).group(1)
            calls["research"].append(q)
            src = re.search(r"source=(\S+)", human).group(1)
            return ResearchFindings(findings=[ResearchFinding(source=src, content=f"finding for {q}")])
        if name == "AnalysisModel":
            calls["analyst"] += 1
            if calls["down"]:  # a failure retrying can't fix (quota and 5xx wording would be retried)
                raise ValueError("the model returned an invalid response")
            return AnalysisModel(content="We recommend wind.", claims=["c1 [1]"])
        if name == "CriticOutput":
            return CriticOutput.scripted(CriticDecision.passed, "", messages)
        if name == "WriterOutput":
            return WriterOutput(content="BRIEF", citations=["1"])
        raise AssertionError(name)

    model = make_stub(responder)
    cfg = Settings(max_critic_iters=3, max_parallel=2, backoff_max_attempts=1)
    agents = MaestroAgents(researcher=Researcher(model=model, settings=cfg), analyst=Analyst(model=model, settings=cfg),
                           critic=Critic(model=model, settings=cfg), writer=Writer(model=model, settings=cfg),
                           settings=cfg, planner=planner, tools=DEFAULT_REGISTRY, corpus=CORPUS)
    return agents, calls


def config(run_id):
    return {"configurable": {"thread_id": run_id}, "recursion_limit": 60}


def test_a_reviewed_run_pauses_before_research_and_resumes_with_the_edits(make_stub):
    agents, calls = build(make_stub)
    graph = build_graph(agents=agents, checkpointer=MemorySaver(), review_plan=True)
    graph.invoke(new_state(GOAL, run_id="r1"), config("r1"))

    paused = graph.get_state(config("r1"))
    assert paused.next == ("research",) and calls["research"] == []   # nothing researched yet
    researchers = [s for s in paused.values["subtasks"] if s.role == Role.researcher]
    keep, drop = researchers[0], researchers[1]

    changed, event = apply_plan_edits(paused.values["subtasks"], {
        "subtasks": [{"id": keep.id, "description": "Wind capacity factor in winter"},
                     {"id": drop.id, "remove": True}],
        "add": [{"description": "How cheap has battery storage become?"}],
    }, max_subtasks=8)
    graph.update_state(config("r1"), {"subtasks": changed, "trace": [event]})
    final = graph.invoke(None, config("r1"))

    assert final["status"] == RunStatus.completed.value
    assert sorted(calls["research"]) == ["How cheap has battery storage become?", "Wind capacity factor in winter"]
    status = {s.id: s.status for s in final["subtasks"]}
    assert status[drop.id] == SubtaskStatus.skipped and status["research_added_1"] == SubtaskStatus.done
    approved = next(e for e in final["trace"] if e.event_type == EventType.plan_approved)
    assert approved.summary == f"plan approved: edited {keep.id}, removed {drop.id}, added research_added_1"


def test_a_failed_run_resumes_from_its_last_finished_node(make_stub):
    agents, calls = build(make_stub, analyst_down=True)
    graph = build_graph(agents=agents, checkpointer=MemorySaver())
    with pytest.raises(ValueError, match="invalid response"):
        graph.invoke(new_state(GOAL, run_id="r2"), config("r2"))
    stuck = graph.get_state(config("r2"))
    assert stuck.next == ("analyze",)
    research_done = list(calls["research"])

    calls["down"] = False  # the model is back
    final = graph.invoke(None, config("r2"))
    assert final["status"] == RunStatus.completed.value
    assert calls["plan"] == 1 and calls["research"] == research_done  # planning and research not redone
    assert calls["analyst"] == 2                                        # the failed step, then the resumed one


def test_plan_edits_are_checked():
    subs = [Subtask(id="r1", description="a", role=Role.researcher),
            Subtask(id="analyze", description="b", role=Role.analyst, depends_on=["r1"])]
    with pytest.raises(ValueError, match="only research subtasks"):
        apply_plan_edits(subs, {"subtasks": [{"id": "analyze", "remove": True}]}, 8)
    with pytest.raises(ValueError, match="at least one research"):
        apply_plan_edits(subs, {"subtasks": [{"id": "r1", "remove": True}]}, 8)
    with pytest.raises(ValueError, match="no subtask"):
        apply_plan_edits(subs, {"subtasks": [{"id": "nope", "remove": True}]}, 8)
    with pytest.raises(ValueError, match="at most 3"):
        apply_plan_edits(subs, {"add": [{"description": "x"}, {"description": "y"}]}, 3)
    changed, event = apply_plan_edits(subs, {}, 8)
    assert changed == [] and event.summary == "plan approved as planned"


# --- through the server ---------------------------------------------------------------------
def server(make_stub, **kw):
    import json

    from fastapi.testclient import TestClient

    from maestro.serve import create_app
    from maestro.trace import TraceStore

    agents, calls = build(make_stub, **kw)
    client = TestClient(create_app(agents=agents, trace_store=TraceStore(":memory:")))

    def frames(resp):
        return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    return client, calls, frames


def test_a_reviewed_run_waits_for_approval_and_finishes_with_the_edits(make_stub):
    client, calls, frames = server(make_stub)
    f = frames(client.post("/run", json={"goal": GOAL, "review_plan": True}))
    assert f[-1]["type"] == "paused" and f[-1]["next"] == ["research"]
    run_id = f[-1]["run_id"]
    researchers = [p for p in f[-1]["plan"] if p["role"] == "researcher"]
    assert calls["research"] == [] and client.get("/runs").json()["runs"] == []   # nothing run or saved yet

    edits = {"subtasks": [{"id": researchers[1]["id"], "remove": True}],
             "add": [{"description": "How cheap has battery storage become?", "queries": ["battery cost 2024"]}]}
    f = frames(client.post(f"/runs/{run_id}/resume", json={"plan": edits}))
    types = [x["type"] for x in f]
    assert types[0] == "resumed" and types[1] == "plan_approved" and types[-1] == "final"
    assert "run_started" not in types and types.count("plan_produced") == 0  # nothing from before the pause repeated
    assert f[-1]["status"] == "completed" and f[1]["edits"] == [f"removed {researchers[1]['id']}", "added research_added_1"]
    assert sorted(calls["research"]) == sorted([researchers[0]["description"], "How cheap has battery storage become?"])
    assert [r["run_id"] for r in client.get("/runs").json()["runs"]] == [run_id]
    # a finished run's checkpoints are gone; it is replayed from the trace store instead
    assert client.post(f"/runs/{run_id}/resume", json={}).status_code == 404


def test_an_edit_that_cannot_run_is_refused_and_the_run_stays_paused(make_stub):
    client, calls, frames = server(make_stub)
    run_id = frames(client.post("/run", json={"goal": GOAL, "review_plan": True}))[-1]["run_id"]
    r = client.post(f"/runs/{run_id}/resume", json={"plan": {"subtasks": [{"id": "analyze", "remove": True}]}})
    assert r.status_code == 422 and "only research subtasks" in r.json()["error"]
    f = frames(client.post(f"/runs/{run_id}/resume", json={}))   # approve as planned
    assert f[1]["type"] == "plan_approved" and f[-1]["status"] == "completed"


def test_a_failed_run_resumes_through_the_server(make_stub):
    client, calls, frames = server(make_stub, analyst_down=True)
    f = frames(client.post("/run", json={"goal": GOAL}))
    final = f[-1]
    assert final["status"] == "error" and final["resumable"] is True
    assert client.get("/runs").json()["runs"] == []                # a failed run isn't saved as finished
    research_done = list(calls["research"])

    calls["down"] = False
    r = client.post(f"/runs/{final['run_id']}/resume", json={"plan": {"add": [{"description": "too late"}]}})
    assert r.status_code == 409                                     # past planning: the plan can't change
    f = frames(client.post(f"/runs/{final['run_id']}/resume", json={}))
    assert f[0]["from"] == "analyze" and f[-1]["status"] == "completed"
    assert calls["plan"] == 1 and calls["research"] == research_done


def test_resuming_a_run_that_does_not_exist(make_stub):
    client, _, _ = server(make_stub)
    assert client.post("/runs/run_00000000/resume", json={}).status_code == 404
    assert client.post("/runs/../etc/resume", json={}).status_code == 404
    assert client.post("/runs/not-a-run/resume", json={}).status_code == 404
