"""Critic subagent + the critic loop (§12) — the vivid "agents disagree" moment.

The Critic reviews the Analyst's draft AGAINST the gathered evidence and returns a
structured verdict: PASS, or REJECT + specific feedback. On REJECT the draft goes
back to the Analyst, which revises using the feedback; the Critic reviews again.
The loop is bounded by ``MAX_CRITIC_ITERS`` — on hitting the ceiling without a PASS
it degrades gracefully (proceed with the best draft, flagged "not fully validated"),
never looping unbounded.

A genuine LLM critic can reject on its own. ``force_critic_reject`` (config) makes a
rejection reliably triggerable on demand for the demo (§21.3) without faking the
whole thing — it only forces the first N reviews to REJECT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

from ..resilience import OnRetry
from ..state import (
    AnalysisDraft,
    CriticDecision,
    CriticVerdict,
    Evidence,
    Role,
    Subtask,
    SubtaskStatus,
)
from .base import Subagent

CRITIC_PROMPT = """You are a Critic subagent in a multi-agent research system.
You review an Analyst's draft AGAINST the evidence it is supposed to rest on.
Decide one of:
- PASS: claims are supported by the evidence, the analysis addresses the goal, and there
  are no major gaps, contradictions, or unsupported assertions.
- REJECT: there are unsupported claims, missing key considerations, contradictions, or the
  goal is not addressed. When you REJECT, give specific, actionable feedback naming the
  problems so the Analyst can fix them.
Be a genuine reviewer: do not rubber-stamp. If the draft is weak, REJECT it."""


class CriticOutput(BaseModel):
    decision: CriticDecision
    feedback: str = ""


def _render_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no evidence)"
    return "\n".join(f"[{i}] ({e.source}) {e.content}" for i, e in enumerate(evidence, start=1))


class Critic(Subagent):
    role = Role.critic
    name = "critic"
    system_prompt = CRITIC_PROMPT

    def run(
        self,
        subtask: Subtask,
        goal: str,
        draft: AnalysisDraft,
        evidence: list[Evidence],
        *,
        iteration: int = 1,
        on_retry: Optional[OnRetry] = None,
    ) -> tuple[Subtask, CriticVerdict]:
        """Return (updated subtask, verdict). ``iteration`` is the 1-based review number."""
        done = subtask.model_copy(update={"status": SubtaskStatus.done, "result": f"review {iteration}"})

        # Demo affordance: force the first N reviews to REJECT (still real feedback text).
        if iteration <= self.settings.force_critic_reject:
            verdict = CriticVerdict(
                decision=CriticDecision.rejected,
                feedback="[forced-reject demo] Tighten claim-to-evidence grounding; several claims lack a cited source.",
                iteration=iteration,
            )
            self.log.info("critic review %d -> REJECT (forced demo)", iteration)
            return done, verdict

        claims = "\n".join(f"- {c}" for c in draft.claims) or "(none listed)"
        human = (
            f"Goal: {goal}\n\n"
            f"Analysis draft:\n{draft.content}\n\n"
            f"Claims made:\n{claims}\n\n"
            f"Evidence the draft must rest on:\n{_render_evidence(evidence)}"
        )
        out = self._structured(CriticOutput, human, on_retry=on_retry)
        verdict = CriticVerdict(decision=out.decision, feedback=out.feedback, iteration=iteration)
        self.log.info("critic review %d -> %s", iteration, verdict.decision.value)
        return done, verdict


# --------------------------------------------------------------------------- #
# The bounded critic loop
# --------------------------------------------------------------------------- #
@dataclass
class CriticLoopResult:
    draft: AnalysisDraft
    verdicts: list[CriticVerdict] = field(default_factory=list)
    iterations: int = 0
    passed: bool = False
    degraded: bool = False


def run_critic_loop(
    *,
    analyst,
    critic: Critic,
    analysis_subtask: Subtask,
    critic_subtask: Subtask,
    goal: str,
    evidence: list[Evidence],
    draft: AnalysisDraft,
    max_iters: int,
    on_retry: Optional[OnRetry] = None,
    on_event=None,
) -> CriticLoopResult:
    """Run reject->revise->pass, bounded by ``max_iters``.

    Does up to ``max_iters`` critic reviews, with the Analyst revising between
    rejections. Returns PASS as soon as it happens; if the ceiling is reached
    without a PASS, returns ``degraded=True`` with the best (last) draft.
    """
    verdicts: list[CriticVerdict] = []
    current = draft

    for n in range(1, max_iters + 1):
        _, verdict = critic.run(critic_subtask, goal, current, evidence, iteration=n, on_retry=on_retry)
        verdicts.append(verdict)
        if on_event:
            on_event("critic_verdict", verdict)

        if verdict.decision == CriticDecision.passed:
            return CriticLoopResult(draft=current, verdicts=verdicts, iterations=n, passed=True, degraded=False)

        # REJECT: revise unless we've exhausted the ceiling
        if n < max_iters:
            _, current = analyst.run(
                analysis_subtask, goal, evidence, feedback=verdict.feedback, revision=n, on_retry=on_retry
            )
            if on_event:
                on_event("revised", current)

    # ceiling reached without a PASS -> degrade gracefully
    degraded_draft = current.model_copy(update={"revision": current.revision})
    return CriticLoopResult(
        draft=degraded_draft, verdicts=verdicts, iterations=max_iters, passed=False, degraded=True
    )
