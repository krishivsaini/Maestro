"""The shared state — the blackboard the supervisor and subagents coordinate through.

Two layers live here:

1. **Domain models** (Pydantic): ``Subtask``, ``Evidence``, ``AnalysisDraft``,
   ``CriticVerdict``, ``Answer``, ``MemoryItem``, ``TraceEvent``.
2. **The graph state** (``MaestroState``, a ``TypedDict``) that flows through the
   LangGraph ``StateGraph``. List channels that receive concurrent writes from
   bounded-parallel branches carry **reducers** so parallel updates merge instead
   of clobbering each other.

The ``Subtask`` model with its ``depends_on`` list is the spine: it is what makes
decomposition dynamic (§10) and scheduling dependency-aware (§11).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str = "st") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    researcher = "researcher"
    analyst = "analyst"
    critic = "critic"
    writer = "writer"


class SubtaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    degraded = "degraded"


class CriticDecision(str, Enum):
    passed = "PASS"
    rejected = "REJECT"


class RunStatus(str, Enum):
    running = "running"
    completed = "completed"
    degraded = "degraded"
    halted = "halted"  # a ceiling forced termination


class EventType(str, Enum):
    plan_produced = "plan_produced"
    plan_replanned = "plan_replanned"
    subtask_dispatched = "subtask_dispatched"
    subagent_result = "subagent_result"
    critic_pass = "critic_pass"
    critic_reject = "critic_reject"
    memory_recall = "memory_recall"
    memory_write = "memory_write"
    recovery = "recovery"
    rate_limit_backoff = "rate_limit_backoff"
    degraded = "degraded"
    halted = "halted"
    completed = "completed"
    error = "error"


# --------------------------------------------------------------------------- #
# Domain models
# --------------------------------------------------------------------------- #
class Subtask(BaseModel):
    """A unit of work the supervisor delegates to one specialist."""

    id: str = Field(default_factory=lambda: new_id("st"))
    description: str
    role: Role
    depends_on: list[str] = Field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.pending
    result: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0  # for recovery bounding


class Evidence(BaseModel):
    """A researcher finding, tied back to the subtask that produced it."""

    subtask_id: str
    source: str  # URL or corpus id
    content: str
    retrieved_at: str = Field(default_factory=_utcnow)


class AnalysisDraft(BaseModel):
    """The Analyst's current draft; ``revision`` increments on each critic cycle."""

    content: str
    claims: list[str] = Field(default_factory=list)
    revision: int = 0


class CriticVerdict(BaseModel):
    """A critic decision: PASS, or REJECT with specific feedback."""

    decision: CriticDecision
    feedback: str = ""
    iteration: int = 0
    created_at: str = Field(default_factory=_utcnow)


class Answer(BaseModel):
    """The Writer's final cited composition."""

    content: str
    citations: list[str] = Field(default_factory=list)
    validated: bool = True  # False when the critic ceiling was hit (degraded)
    notes: Optional[str] = None


class MemoryItem(BaseModel):
    """A distilled finding written to / recalled from long-term memory."""

    thread_id: str
    content: str
    score: Optional[float] = None
    created_at: str = Field(default_factory=_utcnow)


class TraceEvent(BaseModel):
    """One replayable event. Every meaningful transition appends one (§17)."""

    event_type: EventType
    agent: str
    summary: str
    timestamp: str = Field(default_factory=_utcnow)
    subtask_id: Optional[str] = None
    tool_call: Optional[str] = None
    error: Optional[str] = None
    critic_verdict: Optional[str] = None
    recovery_decision: Optional[str] = None
    rate_limit_retry: Optional[int] = None
    data: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Reducers — merge concurrent writes from bounded-parallel branches
# --------------------------------------------------------------------------- #
def merge_subtasks(left: list[Subtask], right: list[Subtask]) -> list[Subtask]:
    """Merge subtask updates by id, preserving the original ordering.

    The supervisor sets the full plan (left empty, right = full list). Parallel
    specialist nodes each return only their own updated subtask; this reducer
    merges them by id so concurrent status updates do not clobber the plan.
    """
    if not left:
        return list(right or [])
    if not right:
        return list(left)
    by_id: dict[str, Subtask] = {s.id: s for s in left}
    order: list[str] = [s.id for s in left]
    for s in right:
        if s.id not in by_id:
            order.append(s.id)
        by_id[s.id] = s
    return [by_id[i] for i in order]


def append_list(left: Optional[list], right: Optional[list]) -> list:
    """Concatenate list channels (evidence, verdicts, trace) across branches."""
    return (left or []) + (right or [])


# --------------------------------------------------------------------------- #
# The graph state
# --------------------------------------------------------------------------- #
class MaestroState(TypedDict, total=False):
    """The typed object that flows through the StateGraph.

    ``total=False`` lets nodes return partial updates. Scalar counters
    (``step_count``, ``critic_iterations``, ``recovery_attempts``) are written by
    single (non-parallel) nodes, so they use last-write-wins; concurrently-written
    list channels use reducers above.
    """

    # identity
    goal: str
    thread_id: str
    run_id: str

    # plan + working memory
    subtasks: Annotated[list[Subtask], merge_subtasks]
    evidence: Annotated[list[Evidence], append_list]
    analysis: Optional[AnalysisDraft]
    critic_verdicts: Annotated[list[CriticVerdict], append_list]
    memory_hits: list[MemoryItem]

    # counters (loop/cost control, §15)
    step_count: int
    critic_iterations: int
    recovery_attempts: int

    # outputs
    final_output: Optional[Answer]
    status: str  # RunStatus value

    # observability (§17)
    trace: Annotated[list[TraceEvent], append_list]


def new_state(
    goal: str,
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> MaestroState:
    """Construct the initial state for a run."""
    return MaestroState(
        goal=goal,
        thread_id=thread_id or new_id("thread"),
        run_id=run_id or new_id("run"),
        subtasks=[],
        evidence=[],
        analysis=None,
        critic_verdicts=[],
        memory_hits=[],
        step_count=0,
        critic_iterations=0,
        recovery_attempts=0,
        final_output=None,
        status=RunStatus.running.value,
        trace=[],
    )
