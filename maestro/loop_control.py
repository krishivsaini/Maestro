"""Loop & cost control (§15) — the safety nets that keep a run bounded.

Multi-agent systems loop and multiply cost, so bounding them is a required
feature, not an afterthought. Three explicit bounds live in the graph's routers:
``MAX_CRITIC_ITERS`` (critic loop), ``MAX_RECOVERY_ATTEMPTS`` (recovery), and the
global ``MAX_STEPS`` backstop here. Two loop detectors catch pathologies the
per-loop counters might miss: a critic that keeps giving identical feedback with
no progress, and a subtask that keeps getting re-dispatched.
"""

from __future__ import annotations

from typing import Optional

from .config import Settings
from .state import CriticDecision, EventType

# Consecutive identical REJECT verdicts that count as "no progress".
CRITIC_STALL_THRESHOLD = 3


def step_ceiling_reached(state: dict, settings: Settings) -> bool:
    """True once the run has executed ``MAX_STEPS`` nodes (global cost backstop)."""
    return state.get("step_count", 0) >= settings.max_steps


def stuck_in_critic_loop(state: dict, threshold: int = CRITIC_STALL_THRESHOLD) -> bool:
    """True if the last ``threshold`` critic verdicts are identical REJECTs.

    Identical feedback repeated means the analyst is not making progress and the
    loop would spin uselessly to the ceiling — force completion instead.
    """
    rejects = [v for v in state.get("critic_verdicts", []) if v.decision == CriticDecision.rejected]
    if len(rejects) < threshold:
        return False
    recent = rejects[-threshold:]
    first = recent[0].feedback.strip()
    return bool(first) and all(v.feedback.strip() == first for v in recent)


def repeated_delegation(trace: list, threshold: int) -> Optional[str]:
    """Return a subtask id that has been dispatched >= ``threshold`` times, else None."""
    counts: dict[str, int] = {}
    for e in trace:
        if e.event_type == EventType.subtask_dispatched and e.subtask_id:
            counts[e.subtask_id] = counts.get(e.subtask_id, 0) + 1
            if counts[e.subtask_id] >= threshold:
                return e.subtask_id
    return None
