"""Researcher subagent (§7).

Gathers evidence for ONE assigned sub-question via the tools (web search by
default, or simple retrieval over a provided corpus for offline runs), then uses
its own prompt to distill grounded findings. Its context is only the sub-question
and its own tool results — never other agents' internals. Produces ``Evidence``
items tagged with their source and the originating subtask.

Tool failures / empty results are returned as a ``failed`` subtask (no raw raise);
the supervisor decides what the failure means and how to recover (§13, §15).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ..resilience import OnRetry
from ..state import Evidence, Role, Subtask, SubtaskStatus
from ..tools import DEFAULT_REGISTRY, ToolRegistry
from .base import Subagent

RESEARCHER_PROMPT = """You are a Researcher subagent in a multi-agent research system.
You are given ONE specific sub-question and raw search/retrieval results.
Your job:
- Extract the most relevant, factual findings that help answer the sub-question.
- Ground every finding in one of the provided sources; do NOT invent facts or sources.
- Keep each finding concise. Prefer 2-4 findings.
Return only the structured findings."""


class ResearchFinding(BaseModel):
    source: str
    content: str


class ResearchFindings(BaseModel):
    findings: list[ResearchFinding]


def _render_results(data: list[dict]) -> str:
    lines: list[str] = []
    for i, r in enumerate(data, start=1):
        source = r.get("url") or r.get("source") or f"result_{i}"
        title = r.get("title", "")
        text = r.get("snippet") or r.get("content") or ""
        lines.append(f"[{i}] source={source} {title}\n{text}".strip())
    return "\n\n".join(lines)


class Researcher(Subagent):
    role = Role.researcher
    name = "researcher"
    system_prompt = RESEARCHER_PROMPT

    def run(
        self,
        subtask: Subtask,
        *,
        tools: Optional[ToolRegistry] = None,
        corpus: Optional[list] = None,
        on_retry: Optional[OnRetry] = None,
    ) -> tuple[Subtask, list[Evidence]]:
        """Return (updated subtask, evidence). On failure, subtask.status == failed."""
        registry = tools or DEFAULT_REGISTRY
        query = subtask.description

        if corpus is not None:  # offline / deterministic path
            result = registry.run("retrieve", query=query, corpus=corpus)
        else:
            result = registry.run("web_search", query=query, on_retry=on_retry)

        if not result.ok:
            return self._fail(subtask, result.error or "tool failed"), []
        if not result.data:
            return self._fail(subtask, "no results for sub-question"), []

        findings = self._structured(
            ResearchFindings,
            f"Sub-question: {query}\n\nRaw results:\n{_render_results(result.data)}",
            on_retry=on_retry,
        )
        evidence = [
            Evidence(subtask_id=subtask.id, source=f.source, content=f.content)
            for f in findings.findings
        ]
        if not evidence:
            return self._fail(subtask, "no findings extracted from results"), []

        done = subtask.model_copy(
            update={
                "status": SubtaskStatus.done,
                "result": f"{len(evidence)} findings",
                "attempts": subtask.attempts + 1,
            }
        )
        self.log.info("researcher %s -> %d evidence items", subtask.id, len(evidence))
        return done, evidence

    def _fail(self, subtask: Subtask, error: str) -> Subtask:
        self.log.warning("researcher %s failed: %s", subtask.id, error)
        return subtask.model_copy(
            update={
                "status": SubtaskStatus.failed,
                "error": error,
                "attempts": subtask.attempts + 1,
            }
        )
