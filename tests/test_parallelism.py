"""Day 4 — bounded parallelism + dependency handling (§11, FR-17..FR-20).

Proves the three parallelism guarantees:
  (a) independent subtasks actually run concurrently,
  (b) concurrency never exceeds the cap,
  (c) a dependent subtask does not start before its dependencies are done.
"""

import threading
import time

from maestro.scheduler import (
    ScheduleReport,
    blocked_subtasks,
    is_complete,
    next_batch,
    ready_subtasks,
    run_schedule,
)
from maestro.state import Evidence, Role, Subtask, SubtaskStatus

EPS = 1e-3


def make_plan() -> list[Subtask]:
    return [
        Subtask(id="r1", description="research 1", role=Role.researcher),
        Subtask(id="r2", description="research 2", role=Role.researcher),
        Subtask(id="analyze", description="analyze", role=Role.analyst, depends_on=["r1", "r2"]),
        Subtask(id="critique", description="critique", role=Role.critic, depends_on=["analyze"]),
        Subtask(id="write", description="write", role=Role.writer, depends_on=["critique"]),
    ]


def sleepy_worker(delay: float = 0.05):
    def worker(st: Subtask) -> Subtask:
        time.sleep(delay)
        return st.model_copy(update={"status": SubtaskStatus.done, "result": f"done:{st.id}"})

    return worker


# --- readiness primitives ---------------------------------------------------
def test_readiness_and_next_batch():
    subs = make_plan()
    assert {s.id for s in ready_subtasks(subs)} == {"r1", "r2"}  # only independents ready
    assert len(next_batch(subs, 2)) == 2
    assert len(next_batch(subs, 1)) == 1  # cap limits the batch
    # once research is done, analyze becomes ready — critique/write still blocked
    subs2 = [
        s.model_copy(update={"status": SubtaskStatus.done}) if s.id in ("r1", "r2") else s
        for s in subs
    ]
    assert {s.id for s in ready_subtasks(subs2)} == {"analyze"}


# --- (a) + (b): concurrency happens, and the cap is respected ---------------
def test_independent_run_concurrently_under_cap2():
    final, report = run_schedule(make_plan(), sleepy_worker(0.05), cap=2)
    assert is_complete(final)
    assert all(s.status == SubtaskStatus.done for s in final)
    # (a) peak-concurrency counter proves two workers were live simultaneously
    assert report.max_concurrency == 2, report
    # (b) peak never exceeds the cap
    assert report.max_concurrency <= report.cap
    # (a, direct) the two researchers' wall-clock intervals actually OVERLAP
    (r1s, r1e), (r2s, r2e) = report.start_end["r1"], report.start_end["r2"]
    assert r1s < r2e and r2s < r1e, "r1 and r2 intervals did not overlap"


def test_cap_of_one_serializes():
    _, report = run_schedule(make_plan(), sleepy_worker(0.02), cap=1)
    assert report.max_concurrency == 1


def test_cap_never_exceeded_with_more_independents():
    three = [Subtask(id=f"r{i}", description="x", role=Role.researcher) for i in range(3)]
    _, r_cap2 = run_schedule(three, sleepy_worker(0.05), cap=2)
    assert r_cap2.max_concurrency == 2  # 3 independents, cap 2 -> at most 2 at once

    three_again = [Subtask(id=f"r{i}", description="x", role=Role.researcher) for i in range(3)]
    _, r_cap3 = run_schedule(three_again, sleepy_worker(0.05), cap=3)
    assert r_cap3.max_concurrency == 3  # cap 3 -> all three overlap


# --- (c): dependents wait for their dependencies ----------------------------
def test_dependent_starts_after_dependencies_finish():
    _, report = run_schedule(make_plan(), sleepy_worker(0.03), cap=2)
    se = report.start_end
    # analyze must start only after BOTH researchers finished
    assert se["analyze"][0] >= max(se["r1"][1], se["r2"][1]) - EPS
    # the sequential tail must also respect order
    assert se["critique"][0] >= se["analyze"][1] - EPS
    assert se["write"][0] >= se["critique"][1] - EPS


# --- blocked detection (used by Day 8 recovery) -----------------------------
def test_blocked_subtasks_when_dependency_failed():
    subs = [
        Subtask(id="r1", description="x", role=Role.researcher, status=SubtaskStatus.failed),
        Subtask(id="analyze", description="y", role=Role.analyst, depends_on=["r1"]),
    ]
    blocked = blocked_subtasks(subs)
    assert {s.id for s in blocked} == {"analyze"}
    assert not is_complete(subs)  # analyze is still pending, blocked


# ===========================================================================
# Graph-level integration: prove the REAL `research` node fans out at runtime.
# graph.png renders `research` as a single node; these tests confirm that node
# actually runs the researchers concurrently (bounded) inside itself.
# ===========================================================================
class _ConcurrencyResearcher:
    """A researcher stub that sleeps and records peak concurrent invocations."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self._lock = threading.Lock()
        self._live = 0
        self.peak = 0
        self.ends: dict[str, float] = {}

    def run(self, subtask, *, tools=None, corpus=None, on_retry=None):
        with self._lock:
            self._live += 1
            self.peak = max(self.peak, self._live)
        time.sleep(self.delay)
        with self._lock:
            self._live -= 1
            self.ends[subtask.id] = time.perf_counter()
        done = subtask.model_copy(update={"status": SubtaskStatus.done, "result": "ok"})
        return done, [Evidence(subtask_id=subtask.id, source="s", content="e")]


def _graph_agents(researcher, make_stub, max_parallel):
    from maestro.agents import Analyst, Critic, Writer
    from maestro.agents.analyst import AnalysisModel
    from maestro.agents.critic import CriticOutput
    from maestro.agents.writer import WriterOutput
    from maestro.config import Settings
    from maestro.graph import MaestroAgents
    from maestro.state import CriticDecision
    from maestro.supervisor import heuristic_planner

    timings: dict[str, float] = {}

    def responder(schema, messages):
        name = getattr(schema, "__name__", "")
        if name == "AnalysisModel":
            timings["analyst_start"] = time.perf_counter()
            return AnalysisModel(content="analysis", claims=["c1"])
        if name == "CriticOutput":
            return CriticOutput(decision=CriticDecision.passed, feedback="")
        if name == "WriterOutput":
            return WriterOutput(content="brief", citations=["s"])
        raise AssertionError(name)

    model = make_stub(responder)
    cfg = Settings(max_parallel=max_parallel, max_critic_iters=3)
    agents = MaestroAgents(
        researcher=researcher,
        analyst=Analyst(model=model, settings=cfg),
        critic=Critic(model=model, settings=cfg),
        writer=Writer(model=model, settings=cfg),
        settings=cfg,
        planner=heuristic_planner,
    )
    return agents, timings


def test_graph_research_node_fans_out_concurrently(make_stub):
    from maestro.graph import run_goal

    researcher = _ConcurrencyResearcher(delay=0.05)
    agents, timings = _graph_agents(researcher, make_stub, max_parallel=2)
    run_goal("Compare A and B for reliability", agents=agents)

    # (a) the two researchers ran concurrently INSIDE the single `research` node
    assert researcher.peak == 2, f"research node did not fan out (peak={researcher.peak})"
    # (b) never exceeded MAX_PARALLEL
    assert researcher.peak <= 2
    # (c) the dependent analyst stage started only after both researchers finished
    assert timings["analyst_start"] >= max(researcher.ends.values()) - EPS


def test_graph_max_parallel_bounds_fanout(make_stub):
    from maestro.graph import run_goal

    researcher = _ConcurrencyResearcher(delay=0.03)
    agents, _ = _graph_agents(researcher, make_stub, max_parallel=1)
    run_goal("Compare A and B for reliability", agents=agents)

    # cap=1 -> the real graph serializes the researchers (peak never reaches 2)
    assert researcher.peak == 1, f"MAX_PARALLEL=1 not enforced in graph (peak={researcher.peak})"
