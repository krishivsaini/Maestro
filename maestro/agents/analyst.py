"""Analyst subagent.

Synthesizes the gathered evidence into a structured analysis/comparison that
addresses the goal, grounding claims in the evidence. Its context is the goal,
its analysis task, and the evidence handed to it — not the researchers' raw
reasoning. The optional ``feedback`` argument is how the critic loop
feeds a rejection back for revision.

Every claim carries the evidence numbers it rests on, so the Critic can check each
one against its source rather than judge an overall impression. A revision starts
from the previous draft and the numbered open issues, and reports how it addressed
each, instead of rewriting from scratch and losing what was already right.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, model_validator

from ..resilience import OnRetry
from ..state import AnalysisDraft, Evidence, MemoryItem, ReviewIssue, Role, Subtask, SubtaskStatus
from .base import Subagent

ANALYST_PROMPT = """You are an Analyst subagent in a multi-agent research system.
You are given a goal and a set of evidence items gathered by researchers.
Your job:
- Synthesize a structured analysis/comparison that DIRECTLY addresses the goal.
- Ground every claim in the provided evidence; reference sources where possible.
- Organize the analysis around the dimensions that actually decide the question,
  with a short paragraph per dimension rather than a list of one-liners. Weigh the
  evidence on each — where the options genuinely differ, and by how much.
- Do not stop at describing both options; reach a comparative judgement and say what
  it rests on.
- If the goal asks for a recommendation, commit to one, and name the condition under
  which it would change.
- If the evidence is thin or conflicting, say so explicitly rather than overclaiming.
  Keep qualifiers the evidence uses ("up to", "in some regions", "one study found").

CLAIMS: list the 4-8 key claims your analysis rests on, one specific fact or judgement
each; split a claim that bundles several facts. Each claim gives the numbers of the
evidence items that state it. Cite only items that actually say it, not items on
the same topic; a claim with nothing to cite does not belong in the analysis.

Before you answer, check your own draft: every claim cited and supported by what it
cites; no claim stronger than its evidence; no contradiction between sections; the
goal fully answered. Fix anything that fails.

If you are revising after a review, start from your previous draft: keep what was
right, fix every open issue listed, and in `addressed` say what you changed for each
issue number.
Return a structured draft: a coherent analysis plus the list of key claims you make."""


class ClaimOut(BaseModel):
    text: str
    evidence: list[int] = []

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, v):
        """Accept a bare string, reading citations written inline ("... [2, 5]")."""
        if isinstance(v, str):
            refs = [int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", v)
                    for n in re.split(r"\s*,\s*", group)]
            text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", v).strip()
            return {"text": text, "evidence": refs}
        return v


class Addressed(BaseModel):
    issue: int
    change: str


class AnalysisModel(BaseModel):
    content: str
    claims: list[ClaimOut]
    addressed: list[Addressed] = []


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
        previous: Optional[AnalysisDraft] = None,
        issues: Optional[list[ReviewIssue]] = None,
        on_retry: Optional[OnRetry] = None,
    ) -> tuple[Subtask, AnalysisDraft]:
        """Return (updated subtask, analysis draft). ``revision`` tracks critic cycles; on a
        revision, ``previous`` is the draft that was reviewed and ``issues`` what to fix."""
        human = (
            f"Goal: {goal}\n\n"
            f"Analysis task: {subtask.description}\n\n"
            f"Evidence:\n{_render_evidence(evidence)}"
        )
        if memory_hits:
            recalled = "\n".join(f"- {h.content}" for h in memory_hits)
            human += f"\n\nRelevant prior findings recalled from this thread's long-term memory:\n{recalled}"
        if previous is not None:
            from .critic import render_claims  # the same numbering the critic used

            human += (f"\n\nYour previous draft (revise it; keep what was right):\n{previous.content}"
                      f"\n\nIts claims:\n{render_claims(previous)}")
        if issues:
            human += "\n\nOpen issues from the review — fix every one, and say what you changed for each in `addressed`:\n" + \
                "\n".join(f"#{i.id} [{i.severity}] {i.problem}" + (f" Fix: {i.fix}" if i.fix else "") for i in issues)
        elif feedback:
            human += f"\n\nAddress this critic feedback in your revision:\n{feedback}"

        out = self._structured(AnalysisModel, human, on_retry=on_retry)
        draft = AnalysisDraft(
            content=out.content,
            claims=[c.text for c in out.claims],
            refs=[sorted(set(c.evidence)) for c in out.claims],
            revision=revision,
            addressed=[f"#{a.issue}: {a.change}" for a in out.addressed],
        )
        done = subtask.model_copy(
            update={"status": SubtaskStatus.done, "result": f"analysis draft rev {revision}"}
        )
        self.log.info("analyst %s -> draft rev %d, %d claims", subtask.id, revision, len(draft.claims))
        return done, draft
