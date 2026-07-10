"""Analyst subagent (§7).

Synthesizes the gathered evidence into a structured analysis/comparison that
addresses the goal, grounding claims in the evidence. Its context is the goal,
its analysis task, and the evidence handed to it — not the researchers' raw
reasoning. The optional ``feedback`` argument is how the critic loop (Day 6)
feeds a rejection back for revision.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ..resilience import OnRetry
from ..state import AnalysisDraft, Evidence, MemoryItem, Role, Subtask, SubtaskStatus
from .base import Subagent

ANALYST_PROMPT = """You are an Analyst subagent in a multi-agent research system.
You are given a goal and a set of evidence items gathered by researchers.
Your job:
- Synthesize a structured analysis/comparison that DIRECTLY addresses the goal.
- Ground every claim in the provided evidence; reference sources where possible.
- If the evidence is thin or conflicting, say so explicitly rather than overclaiming.
- If critic feedback is provided, address each point in this revision.
Return a structured draft: a coherent analysis plus the list of key claims you make."""


class AnalysisModel(BaseModel):
    content: str
    claims: list[str]


def _render_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no evidence gathered)"
    return "\n".join(f"[{i}] ({e.source}) {e.content}" for i, e in enumerate(evidence, start=1))


class Analyst(Subagent):
    role = Role.analyst
    name = "analyst"
    system_prompt = ANALYST_PROMPT

    def run(
        self,
        subtask: Subtask,
        goal: str,
        evidence: list[Evidence],
        *,
        feedback: Optional[str] = None,
        revision: int = 0,
        memory_hits: Optional[list[MemoryItem]] = None,
        on_retry: Optional[OnRetry] = None,
    ) -> tuple[Subtask, AnalysisDraft]:
        """Return (updated subtask, analysis draft). ``revision`` tracks critic cycles."""
        human = (
            f"Goal: {goal}\n\n"
            f"Analysis task: {subtask.description}\n\n"
            f"Evidence:\n{_render_evidence(evidence)}"
        )
        if memory_hits:
            recalled = "\n".join(f"- {h.content}" for h in memory_hits)
            human += f"\n\nRelevant prior findings recalled from this thread's long-term memory:\n{recalled}"
        if feedback:
            human += f"\n\nAddress this critic feedback in your revision:\n{feedback}"

        out = self._structured(AnalysisModel, human, on_retry=on_retry)
        draft = AnalysisDraft(content=out.content, claims=list(out.claims), revision=revision)
        done = subtask.model_copy(
            update={"status": SubtaskStatus.done, "result": f"analysis draft rev {revision}"}
        )
        self.log.info("analyst %s -> draft rev %d, %d claims", subtask.id, revision, len(draft.claims))
        return done, draft
