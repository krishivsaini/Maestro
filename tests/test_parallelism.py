"""Day 4 — bounded parallelism + dependency handling (§11, FR-17..FR-20).

Proves the three parallelism guarantees:
  (a) independent subtasks actually run concurrently,
  (b) concurrency never exceeds the cap,
  (c) a dependent subtask does not start before its dependencies are done.
"""

import time

from maestro.scheduler import (
    ScheduleReport,
    blocked_subtasks,
    is_complete,
    next_batch,
    ready_subtasks,
    run_schedule,
)
from maestro.state import Role, Subtask, SubtaskStatus

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
    assert report.max_concurrency == 2, report  # r1 and r2 overlapped
    assert report.max_concurrency <= report.cap  # never exceeded the cap


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
