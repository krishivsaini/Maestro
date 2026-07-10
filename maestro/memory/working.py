"""Working (short-term) memory helpers.

Working memory itself lives in the graph state for the duration of a run
(``evidence``, ``subtasks``, ``analysis``, …). These helpers distill what is
worth persisting into long-term memory at the end of a run.
"""

from __future__ import annotations

from typing import Optional

from ..state import AnalysisDraft, Answer


def distill_findings(analysis: Optional[AnalysisDraft], answer: Optional[Answer]) -> list[str]:
    """Pick the durable, reusable findings from a completed run.

    The analysis claims are the grounded, reusable units; we store those.
    """
    if analysis and analysis.claims:
        return [c.strip() for c in analysis.claims if c.strip()]
    if answer and answer.content:
        return [answer.content.strip()]
    return []
