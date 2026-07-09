"""Writer subagent (§7).

Composes the final cited brief from the PASSED analysis draft and the evidence.
Its context is the goal, the analysis, and the evidence — not the researchers' or
critic's internals. When the critic loop hit its ceiling without a PASS, the brief
is still produced but flagged ``validated=False`` with a note (graceful degrade).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ..resilience import OnRetry
from ..state import AnalysisDraft, Answer, Evidence, Role, Subtask, SubtaskStatus
from .base import Subagent

WRITER_PROMPT = """You are a Writer subagent in a multi-agent research system.
Compose the FINAL cited brief from the analysis draft and the evidence.
- Use the analysis as the backbone; write clearly and in a logical structure.
- Cite sources by their identifiers where claims rest on them.
- Do not introduce claims that the analysis/evidence does not support.
Return the final brief text and the list of citations used."""


class WriterOutput(BaseModel):
    content: str
    citations: list[str]


def _render_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no evidence)"
    return "\n".join(f"[{i}] ({e.source}) {e.content}" for i, e in enumerate(evidence, start=1))


class Writer(Subagent):
    role = Role.writer
    name = "writer"
    system_prompt = WRITER_PROMPT

    def run(
        self,
        subtask: Subtask,
        goal: str,
        draft: AnalysisDraft,
        evidence: list[Evidence],
        *,
        validated: bool = True,
        on_retry: Optional[OnRetry] = None,
    ) -> tuple[Subtask, Answer]:
        claims = "\n".join(f"- {c}" for c in draft.claims) or "(none listed)"
        human = (
            f"Goal: {goal}\n\n"
            f"Analysis to write up:\n{draft.content}\n\n"
            f"Claims:\n{claims}\n\n"
            f"Evidence (cite by id/source):\n{_render_evidence(evidence)}"
        )
        out = self._structured(WriterOutput, human, on_retry=on_retry)
        notes = None if validated else (
            "Critic ceiling reached without a PASS; brief proceeds but is not fully validated."
        )
        answer = Answer(content=out.content, citations=list(out.citations), validated=validated, notes=notes)
        done = subtask.model_copy(update={"status": SubtaskStatus.done, "result": "final brief composed"})
        self.log.info("writer -> final brief (validated=%s, %d citations)", validated, len(answer.citations))
        return done, answer
