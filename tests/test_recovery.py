"""Day 8 — visible recovery from subagent failure (§15, FR-21..FR-24).

Injects a researcher failure and proves the supervisor recovers along an explicit,
logged path: retry succeeds -> run completes fully; or recovery is exhausted ->
the subtask degrades and the run STILL completes on partial evidence, flagged in
the output. Every recovery decision is a trace event.
"""

import threading

from maestro.agents import Analyst, Critic, Writer
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import CriticOutput
from maestro.agents.writer import WriterOutput
from maestro.config import Settings
from maestro.graph import MaestroAgents, run_goal
from maestro.state import CriticDecision, EventType, Evidence, RunStatus, Subtask, SubtaskStatus
from maestro.supervisor import heuristic_planner

GOAL = "Compare solar and wind energy for grid reliability, and recommend one."


class FlakyResearcher:
    """Fake researcher: fails specified subtasks (once, or always) to drive recovery."""

    def __init__(self, fail_first=(), fail_always=()):
        self.fail_first = set(fail_first)
        self.fail_always = set(fail_always)
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()

    def run(self, subtask: Subtask, *, tools=None, corpus=None, on_retry=None):
        with self._lock:
            n = self._seen.get(subtask.id, 0)
            self._seen[subtask.id] = n + 1
        fail = subtask.id in self.fail_always or (subtask.id in self.fail_first and n == 0)
        if fail:
            failed = subtask.model_copy(
                update={"status": SubtaskStatus.failed, "error": "injected failure", "attempts": subtask.attempts + 1}
            )
            return failed, []
        done = subtask.model_copy(
            update={"status": SubtaskStatus.done, "result": "ok", "attempts": subtask.attempts + 1}
        )
        return done, [Evidence(subtask_id=subtask.id, source=f"src_{subtask.id}", content=f"finding for {subtask.id}")]


def build_agents(make_stub, researcher):
    def responder(schema, messages):
        name = getattr(schema, "__name__", "")
        if name == "AnalysisModel":
            return AnalysisModel(content="analysis of solar vs wind", claims=["c1"])
        if name == "CriticOutput":
            return CriticOutput(decision=CriticDecision.passed, feedback="")
        if name == "WriterOutput":
            return WriterOutput(content="FINAL BRIEF", citations=["src"])
        raise AssertionError(f"unexpected schema {name}")

    model = make_stub(responder)
    cfg = Settings(max_recovery_attempts=2, max_critic_iters=3, max_parallel=2)
    return MaestroAgents(
        researcher=researcher,
        analyst=Analyst(model=model, settings=cfg),
        critic=Critic(model=model, settings=cfg),
        writer=Writer(model=model, settings=cfg),
        settings=cfg,
        planner=heuristic_planner,
    )


def test_no_failure_completes_without_recovery(make_stub):
    final = run_goal(GOAL, agents=build_agents(make_stub, FlakyResearcher()))
    assert final["status"] == RunStatus.completed.value
    assert final.get("recovery_attempts", 0) == 0
    assert not any(e.event_type == EventType.recovery for e in final["trace"])
    assert len(final["evidence"]) == 2


def test_recovery_retry_succeeds(make_stub):
    final = run_goal(GOAL, agents=build_agents(make_stub, FlakyResearcher(fail_first={"research_1"})))

    assert final["status"] == RunStatus.completed.value  # recovered fully
    assert final["recovery_attempts"] == 1
    recoveries = [e for e in final["trace"] if e.event_type == EventType.recovery]
    assert recoveries and recoveries[0].recovery_decision == "retry"

    by_id = {s.id: s for s in final["subtasks"]}
    assert by_id["research_1"].status == SubtaskStatus.done  # failed then recovered
    assert len(final["evidence"]) == 2  # both researchers ultimately produced evidence


def test_recovery_degrades_after_ceiling(make_stub):
    final = run_goal(GOAL, agents=build_agents(make_stub, FlakyResearcher(fail_always={"research_1"})))

    assert final["status"] == RunStatus.degraded.value  # couldn't complete research_1
    assert final["recovery_attempts"] == 2  # bounded by MAX_RECOVERY_ATTEMPTS

    by_id = {s.id: s for s in final["subtasks"]}
    assert by_id["research_1"].status == SubtaskStatus.degraded
    assert by_id["research_2"].status == SubtaskStatus.done

    # run STILL completed, on partial evidence, flagged in the output
    answer = final["final_output"]
    assert answer is not None
    assert "partial evidence" in (answer.notes or "")
    assert len(final["evidence"]) == 1  # only research_2's evidence

    recoveries = [e for e in final["trace"] if e.event_type == EventType.recovery]
    assert len(recoveries) == 2  # two bounded attempts, both logged
    degraded_events = [e for e in final["trace"] if e.event_type == EventType.degraded]
    assert any("research_1" in e.summary for e in degraded_events)
