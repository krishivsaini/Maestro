"""Writer subagent.

Composes the final cited brief from the PASSED analysis draft and the evidence.
Its context is the goal, the analysis, and the evidence — not the researchers' or
critic's internals. When the critic loop hit its ceiling without a PASS, the brief
is still produced but flagged ``validated=False`` with a note (graceful degrade).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ..resilience import OnRetry
from ..state import AnalysisDraft, Answer, Citation, Evidence, Role, Subtask, SubtaskStatus
from .base import Subagent

WRITER_PROMPT = """You are a Writer subagent in a multi-agent research system.
Compose the FINAL cited brief from the analysis draft and the evidence.
- Use the analysis as the backbone, but WRITE THE BRIEF — do not restate or lightly
  reword the draft. The analysis is working material; the brief is the deliverable,
  and it should be materially fuller than the draft it came from.
- Structure it in markdown: a short opening that answers the question directly, then
  a `## ` section per decision dimension, then a `## Recommendation` that commits to
  an answer and names the tradeoff it accepts.
- Develop each section into a real paragraph or two: draw on the specifics in the
  evidence — figures, mechanisms, stated limitations — rather than asserting summary
  conclusions. Detail that sits in the evidence but not in the brief is detail lost.
- Cite sources by their identifiers where claims rest on them.
- Do not introduce claims that the analysis/evidence does not support. If the
  evidence is thin on a dimension, say so rather than padding.
Return the final brief text and the list of citations used."""


class WriterOutput(BaseModel):
    content: str
    citations: list[str]


def _render_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no evidence)"
    return "\n".join(f"[{i}] ({e.source}) {e.content}" for i, e in enumerate(evidence, start=1))


def _resolve_citations(raw: list[str], evidence: list[Evidence]) -> list[str]:
    """Turn the model's citations into real source identifiers.

    Evidence is shown to the writer as ``[1] (source) ...``, so the model tends to
    cite by the bracket index ("1", "[1]") rather than the source itself — which
    renders as meaningless "1, 2, 3" sources. Map those indices back to the actual
    evidence source; pass through anything that is already a known source (or any
    other string, as a last resort). De-duplicate, preserving order.
    """
    by_index = {str(i): e.source for i, e in enumerate(evidence, start=1)}
    known = {e.source for e in evidence}
    out: list[str] = []
    for c in raw:
        token = c.strip().strip("[]").strip()
        source = by_index.get(token) if token not in known else c
        if source is None:
            source = c  # already a source, or an unrecognized label — keep as-is
        if source and source not in out:
            out.append(source)
    return out


def _numbered_citations(raw: list[str], evidence: list[Evidence]) -> list[Citation]:
    """Resolve the model's citations to (evidence index, source) pairs.

    Same resolution as ``_resolve_citations``, but it keeps the index so the rendered
    source list can be numbered the way the prose cites it. De-duplicates by index and
    orders by it, so gaps (uncited evidence) stay gaps rather than silently shifting
    every later number down by one.
    """
    by_index = {str(i): e.source for i, e in enumerate(evidence, start=1)}
    first_index = {}
    for i, e in enumerate(evidence, start=1):
        first_index.setdefault(e.source, i)

    found: dict[int, str] = {}
    for c in raw:
        token = c.strip().strip("[]").strip()
        if token in by_index:
            found.setdefault(int(token), by_index[token])
        elif token in first_index:
            found.setdefault(first_index[token], token)
    return [Citation(n=n, source=found[n]) for n in sorted(found)]


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
        citations = _resolve_citations(list(out.citations), evidence)
        cited = _numbered_citations(list(out.citations), evidence)
        answer = Answer(content=out.content, citations=citations, cited=cited,
                        validated=validated, notes=notes)
        done = subtask.model_copy(update={"status": SubtaskStatus.done, "result": "final brief composed"})
        self.log.info("writer -> final brief (validated=%s, %d citations)", validated, len(answer.citations))
        return done, answer
