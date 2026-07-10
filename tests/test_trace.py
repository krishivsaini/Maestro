"""Day 11 — replayable run traces (§17, FR-35..FR-36).

A completed run is persisted and can be reconstructed after the fact: metadata,
final output, and the ordered event stream (which subtasks ran, where the critic
rejected, where recovery happened). Node-by-node replay + JSON export + audit query.
"""

import json

from maestro.state import (
    Answer,
    EventType,
    Role,
    Subtask,
    SubtaskStatus,
    TraceEvent,
    new_state,
)
from maestro.trace import RunRecord, TraceStore, record_from_state


def make_state() -> dict:
    s = new_state("Compare X and Y", thread_id="th1", run_id="run1")
    s["subtasks"] = [
        Subtask(id="r1", description="research", role=Role.researcher, status=SubtaskStatus.done),
        Subtask(id="analyze", description="analyze", role=Role.analyst, status=SubtaskStatus.done, depends_on=["r1"]),
    ]
    s["trace"] = [
        TraceEvent(event_type=EventType.plan_produced, agent="supervisor", summary="planned 2"),
        TraceEvent(event_type=EventType.subtask_dispatched, agent="researcher", summary="dispatched r1", subtask_id="r1"),
        TraceEvent(event_type=EventType.subagent_result, agent="researcher", summary="r1 -> done", subtask_id="r1"),
        TraceEvent(event_type=EventType.critic_reject, agent="critic", summary="review 1: REJECT"),
        TraceEvent(event_type=EventType.critic_pass, agent="critic", summary="review 2: PASS"),
        TraceEvent(event_type=EventType.completed, agent="supervisor", summary="run completed (completed)"),
    ]
    s["final_output"] = Answer(content="the brief", citations=["s1"], validated=True)
    s["status"] = "completed"
    s["step_count"] = 6
    s["critic_iterations"] = 2
    return s


def test_save_and_get_run(tmp_path):
    store = TraceStore(str(tmp_path / "runs.db"))
    rid = store.save_run(make_state())
    assert rid == "run1"
    rec = store.get_run("run1")
    assert isinstance(rec, RunRecord)
    assert rec.goal == "Compare X and Y" and rec.status == "completed"
    assert rec.critic_iterations == 2
    assert len(rec.events) == 6
    assert rec.final_output.content == "the brief"
    assert len(rec.subtasks) == 2


def test_replay_preserves_order(tmp_path):
    store = TraceStore(str(tmp_path / "runs.db"))
    store.save_run(make_state())
    types = [e.event_type for e in store.replay("run1")]
    assert types[0] == EventType.plan_produced
    assert types[-1] == EventType.completed
    assert EventType.critic_reject in types and EventType.critic_pass in types


def test_list_and_query_events(tmp_path):
    store = TraceStore(str(tmp_path / "runs.db"))
    store.save_run(make_state())
    runs = store.list_runs()
    assert any(r["run_id"] == "run1" and r["event_count"] == 6 for r in runs)
    rejects = store.query_events(run_id="run1", event_type="critic_reject")
    assert len(rejects) == 1 and rejects[0]["agent"] == "critic"


def test_export_json_file(tmp_path):
    store = TraceStore(str(tmp_path / "runs.db"))
    store.save_run(make_state())
    out = tmp_path / "run1.json"
    store.export_json_file("run1", str(out))
    data = json.loads(out.read_text())
    assert data["run_id"] == "run1"
    assert len(data["events"]) == 6
    assert data["final_output"]["validated"] is True


def test_persists_across_connections(tmp_path):
    db = str(tmp_path / "runs.db")
    TraceStore(db).save_run(make_state())
    # a fresh connection reconstructs the full run
    rec = TraceStore(db).get_run("run1")
    assert rec is not None and len(rec.events) == 6


def test_record_from_state_handles_missing_output():
    rec = record_from_state(new_state("goal only", run_id="r0"))
    assert rec.run_id == "r0" and rec.final_output is None and rec.events == []
