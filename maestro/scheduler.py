"""Bounded-parallel, dependency-aware scheduling (§11).

Two layers:

1. **Readiness primitives** — pure functions over a subtask list: which subtasks
   are ready (deps satisfied), the next capped batch, completion/blocked checks.
   These are reused by the LangGraph routing on Day 7, so the graph and this
   standalone executor share one definition of "ready".
2. **A concrete rolling executor** (``run_schedule``) — runs ready subtasks
   concurrently via a thread pool, **capped at ``MAX_PARALLEL``**, refilling as
   tasks finish, while dependent subtasks wait. It observes actual concurrency so
   the parallelism claim is measurable, and emits per-subtask events for streaming.

The cap is both the parallelism signal *and* the rate-limit protection (§5): with
the Gemini free tier at ~10 RPM, unbounded fan-out would hit 429s immediately.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import Settings, get_settings
from .logging_config import get_logger
from .state import Subtask, SubtaskStatus

log = get_logger("scheduler")

# A worker takes a (running) subtask and returns it updated to a terminal status.
Worker = Callable[[Subtask], Subtask]
# Optional streaming hook: (phase, subtask) where phase in {"dispatched", "completed"}.
OnEvent = Callable[[str, Subtask], None]


# --------------------------------------------------------------------------- #
# Readiness primitives (shared with the graph routing)
# --------------------------------------------------------------------------- #
def _done_ids(subtasks: list[Subtask]) -> set[str]:
    return {s.id for s in subtasks if s.status == SubtaskStatus.done}


def ready_subtasks(subtasks: list[Subtask]) -> list[Subtask]:
    """Pending subtasks whose dependencies are all ``done``."""
    done = _done_ids(subtasks)
    return [
        s
        for s in subtasks
        if s.status == SubtaskStatus.pending and set(s.depends_on) <= done
    ]


def next_batch(subtasks: list[Subtask], cap: int) -> list[Subtask]:
    """Up to ``cap`` ready subtasks (the dispatch unit for one wave)."""
    return ready_subtasks(subtasks)[: max(0, cap)]


TERMINAL = (SubtaskStatus.done, SubtaskStatus.failed, SubtaskStatus.degraded)


def is_complete(subtasks: list[Subtask]) -> bool:
    """True when every subtask has reached a terminal status."""
    return all(s.status in TERMINAL for s in subtasks)


def blocked_subtasks(subtasks: list[Subtask]) -> list[Subtask]:
    """Pending subtasks that can never run because a dependency failed/degraded."""
    bad = {s.id for s in subtasks if s.status in (SubtaskStatus.failed, SubtaskStatus.degraded)}
    return [
        s for s in subtasks if s.status == SubtaskStatus.pending and bad.intersection(s.depends_on)
    ]


# --------------------------------------------------------------------------- #
# Concurrent executor
# --------------------------------------------------------------------------- #
@dataclass
class ScheduleReport:
    cap: int
    max_concurrency: int = 0
    dispatch_order: list[str] = field(default_factory=list)
    # id -> (start, end) monotonic timestamps, for dependency-ordering assertions
    start_end: dict[str, tuple[float, float]] = field(default_factory=dict)


def run_schedule(
    subtasks: list[Subtask],
    worker: Worker,
    *,
    settings: Optional[Settings] = None,
    cap: Optional[int] = None,
    on_event: Optional[OnEvent] = None,
) -> tuple[list[Subtask], ScheduleReport]:
    """Execute subtasks with bounded parallelism, honoring dependencies.

    Rolling dispatch: keep up to ``cap`` subtasks in flight; as each finishes,
    refill from whatever became ready. Returns the updated subtasks and a report
    with the maximum concurrency actually observed (never exceeds ``cap``).
    """
    if cap is None:
        cap = (settings or get_settings()).max_parallel
    cap = max(1, cap)

    report = ScheduleReport(cap=cap)
    by_id: dict[str, Subtask] = {s.id: s for s in subtasks}

    lock = threading.Lock()
    live = 0

    def tracked(st: Subtask) -> Subtask:
        nonlocal live
        with lock:
            live += 1
            report.max_concurrency = max(report.max_concurrency, live)
        start = time.perf_counter()
        try:
            result = worker(st)
        except Exception as exc:  # a worker should return a failed subtask, not raise
            result = st.model_copy(
                update={"status": SubtaskStatus.failed, "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            end = time.perf_counter()
            with lock:
                live -= 1
                report.start_end[st.id] = (start, end)
        return result

    with ThreadPoolExecutor(max_workers=cap) as pool:
        in_flight: dict = {}  # future -> id
        while True:
            # Fill up to the cap with newly-ready subtasks.
            while len(in_flight) < cap:
                ready = next_batch(list(by_id.values()), cap - len(in_flight))
                if not ready:
                    break
                for s in ready:
                    running = s.model_copy(update={"status": SubtaskStatus.running})
                    by_id[s.id] = running
                    report.dispatch_order.append(s.id)
                    if on_event:
                        on_event("dispatched", running)
                    fut = pool.submit(tracked, running)
                    in_flight[fut] = s.id

            if not in_flight:
                break

            done_set, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done_set:
                res = fut.result()
                by_id[res.id] = res
                del in_flight[fut]
                if on_event:
                    on_event("completed", res)

    final = list(by_id.values())
    if not is_complete(final):
        stuck = [s.id for s in final if s.status == SubtaskStatus.pending]
        log.warning("schedule halted with unfinished subtasks (blocked by failed deps): %s", stuck)
    return final, report
