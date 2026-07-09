"""Day 9 — loop & cost control (§15, FR-25..FR-27).

Proves the ceilings halt deliberately runaway runs: the global MAX_STEPS backstop
halts a spinning critic loop, and the loop detector forces completion when the
critic gives identical feedback with no progress. Plus unit tests for the
detectors themselves.
"""

from maestro.agents import Analyst, Critic, Writer
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import CriticOutput
from maestro.agents.writer import WriterOutput
from maestro.config import Settings
from maestro.graph import MaestroAgents, run_goal
from maestro.loop_control import repeated_delegation, step_ceiling_reached, stuck_in_critic_loop
from maestro.state import (
    CriticDecision,
    CriticVerdict,
    EventType,
    Evidence,
    RunStatus,
    Subtask,
    SubtaskStatus,
    TraceEvent,
)
from maestro.supervisor import heuristic_planner

GOAL = "Compare solar and wind energy for grid reliability, and recommend one."


class OkResearcher:
    def run(self, subtask, *, tools=None, corpus=None, on_retry=None):
        done = subtask.model_copy(update={"status": SubtaskStatus.done, "result": "ok"})
        return done, [Evidence(subtask_id=subtask.id, source="s", content="e")]


def build_agents(make_stub, cfg, critic_decision):
    def responder(schema, messages):
        name = getattr(schema, "__name__", "")
        if name == "AnalysisModel":
            return AnalysisModel(content="analysis", claims=["c1", "c2"])
        if name == "CriticOutput":
            fb = "" if critic_decision == CriticDecision.passed else "claim c2 is unsupported"
            return CriticOutput(decision=critic_decision, feedback=fb)
        if name == "WriterOutput":
            return WriterOutput(content="BRIEF", citations=["s"])
        raise AssertionError(name)

    model = make_stub(responder)
    return MaestroAgents(
        researcher=OkResearcher(),
        analyst=Analyst(model=model, settings=cfg),
        critic=Critic(model=model, settings=cfg),
        writer=Writer(model=model, settings=cfg),
        settings=cfg,
        planner=heuristic_planner,
    )


# --- MAX_STEPS global backstop ----------------------------------------------
def test_max_steps_halts_runaway(make_stub):
    # critic ceiling is huge; MAX_STEPS must bite first
    cfg = Settings(max_steps=4, max_critic_iters=20, max_parallel=2)
    final = run_goal(GOAL, agents=build_agents(make_stub, cfg, CriticDecision.rejected))
    assert final["status"] == RunStatus.halted.value
    assert final["critic_iterations"] == 1  # halted after the first review, not 20
    assert "step ceiling" in (final["final_output"].notes or "").lower()
    assert any(e.event_type == EventType.halted for e in final["trace"])


# --- loop detector (no-progress critic) -------------------------------------
def test_loop_detector_halts_stalled_critic(make_stub):
    # plenty of headroom on steps + critic iters; the stall detector must fire
    cfg = Settings(max_steps=100, max_critic_iters=20, max_parallel=2)
    final = run_goal(GOAL, agents=build_agents(make_stub, cfg, CriticDecision.rejected))
    assert final["status"] == RunStatus.halted.value
    assert final["critic_iterations"] == 3  # stopped at the stall threshold, not 20
    assert "loop detector" in (final["final_output"].notes or "").lower()


# --- control: a normal run is not halted ------------------------------------
def test_normal_run_completes_not_halted(make_stub):
    cfg = Settings(max_steps=40, max_critic_iters=3, max_parallel=2)
    final = run_goal(GOAL, agents=build_agents(make_stub, cfg, CriticDecision.passed))
    assert final["status"] == RunStatus.completed.value
    assert final["step_count"] < cfg.max_steps


# --- unit tests for the detectors -------------------------------------------
def test_step_ceiling_reached_unit():
    assert step_ceiling_reached({"step_count": 5}, Settings(max_steps=5))
    assert not step_ceiling_reached({"step_count": 4}, Settings(max_steps=5))


def test_stuck_in_critic_loop_unit():
    same = [CriticVerdict(decision=CriticDecision.rejected, feedback="x", iteration=i) for i in range(1, 4)]
    assert stuck_in_critic_loop({"critic_verdicts": same})  # 3 identical rejects
    varied = [
        CriticVerdict(decision=CriticDecision.rejected, feedback="a"),
        CriticVerdict(decision=CriticDecision.rejected, feedback="b"),
        CriticVerdict(decision=CriticDecision.rejected, feedback="c"),
    ]
    assert not stuck_in_critic_loop({"critic_verdicts": varied})


def test_repeated_delegation_unit():
    trace = [
        TraceEvent(event_type=EventType.subtask_dispatched, agent="researcher", summary="", subtask_id="r1")
        for _ in range(5)
    ]
    assert repeated_delegation(trace, 5) == "r1"
    assert repeated_delegation(trace, 6) is None
