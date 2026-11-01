"""Researcher subagent.

Gathers evidence for ONE assigned sub-question via the tools (web search by
default, or simple retrieval over a provided corpus for offline runs), then uses
its own prompt to distill grounded findings. Its context is only the sub-question
and its own tool results — never other agents' internals. Produces ``Evidence``
items tagged with their source and the originating subtask.

Tool failures / empty results are returned as a ``failed`` subtask (no raw raise);
the supervisor decides what the failure means and how to recover.

Findings are checked in code before they become evidence: a finding whose source is
not one of the results the researcher was actually shown is dropped (it is invented),
a source given as a result number is mapped to that result, and near-duplicate
findings are removed. Everything downstream — the analysis, the critic's claim
checks, the brief's citations — rests on these items, so a fabricated or repeated
one costs more here than anywhere else.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
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
- Aim for 4-6 findings. Each one should be substantive: state the specific fact,
  figure, mechanism or tradeoff — not a topic label. A downstream Analyst sees only
  these findings, never your sources or reasoning, so a finding that omits the detail
  loses it for good.
- Prefer concrete specifics (numbers, named mechanisms, stated limitations) over
  general statements, and keep distinct points as separate findings.
- Some results include passages read from the page itself ("from the page"); they are
  fuller than search snippets, so draw specifics from them first.
- Keep the qualifiers a source uses ("up to", "in Germany", "a 2023 study") and the
  year of any figure when the source gives it.
- If a result is a vendor or advocacy source making a claim in its own favour, say so
  in the finding. Include limitations or counter-evidence when the results contain them.
- For `source`, copy the result's source exactly as shown after "source=". Never cite a
  source that is not in the results.
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
        text = f"(from the page)\n{r['page']}" if r.get("page") else (r.get("snippet") or r.get("content") or "")
        lines.append(f"[{i}] source={source} {title}\n{text}".strip())
    return "\n\n".join(lines)


_NORM = re.compile(r"[^a-z0-9 ]+")
_BY_NUMBER = re.compile(r"^\[?\s*(?:result|source)?[\s_#]*(\d+)\s*\]?$", re.I)


def _key(text: str) -> str:
    """How a source is written varies (scheme, www, case, trailing slash); what it points at doesn't."""
    t = text.strip().lower()
    t = re.sub(r"^https?://", "", t).removeprefix("www.").split("#")[0].rstrip("/")
    return t


def _ground(findings: list[ResearchFinding], data: list[dict]) -> list[ResearchFinding]:
    """Keep findings whose source is one of the shown results, however it is written:
    the URL in any form, the result's number ("[2]", "result 2"), or its title.
    Drops invented sources and near-duplicate findings."""
    shown = [r.get("url") or r.get("source") or f"result_{i}" for i, r in enumerate(data, start=1)]
    lookup = {_key(s): s for s in shown}
    for r, s in zip(data, shown):
        if r.get("title"):
            lookup.setdefault(_key(r["title"]), s)
    kept: list[ResearchFinding] = []
    seen: list[set[str]] = []
    for f in findings:
        raw = f.source.strip()
        url = re.search(r"https?://\S+", raw)
        key = _key(url.group(0) if url else raw)  # "URL  Page title - Site" -> the URL
        src = lookup.get(key)
        if src is None and len(key) >= 30:
            # a long URL the model shortened or truncated: one shown result it is a prefix of
            hits = {s for k, s in lookup.items() if k.startswith(key) or key.startswith(k) and len(k) >= 30}
            src = hits.pop() if len(hits) == 1 else None
        if src is None:
            m = _BY_NUMBER.match(f.source.strip())
            if m and 1 <= int(m.group(1)) <= len(shown):
                src = shown[int(m.group(1)) - 1]
        if src is None:
            continue  # not a source the researcher was shown
        words = set(_NORM.sub(" ", f.content.lower()).split())
        if not words or any(len(words & w) / len(words | w) >= 0.8 for w in seen):
            continue  # empty, or a near-copy of a finding already kept
        seen.append(words)
        kept.append(ResearchFinding(source=src, content=f.content.strip()))
    return kept


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
            if not result.ok:
                return self._fail(subtask, result.error or "tool failed"), []
            data = result.data
        else:
            data, error = self._search(subtask, registry, on_retry)
            if error:
                return self._fail(subtask, error), []
        if not data:
            return self._fail(subtask, "no results for sub-question"), []

        findings = self._structured(
            ResearchFindings,
            f"Sub-question: {query}\n\nRaw results:\n{_render_results(data)}",
            on_retry=on_retry,
        )
        result_data = data
        grounded = _ground(findings.findings, result_data)
        if len(grounded) < len(findings.findings):
            kept = {(g.content) for g in grounded}
            odd = sorted({f.source for f in findings.findings if f.content.strip() not in kept})
            self.log.info("researcher %s dropped %d ungrounded or duplicate findings (sources: %s; shown: %s)",
                          subtask.id, len(findings.findings) - len(grounded), odd[:4],
                          [r.get("url") or r.get("source") for r in result_data][:5])
        evidence = [Evidence(subtask_id=subtask.id, source=f.source, content=f.content) for f in grounded]
        if not evidence:
            return self._fail(subtask, "no grounded findings extracted from results"), []

        done = subtask.model_copy(
            update={
                "status": SubtaskStatus.done,
                "result": f"{len(evidence)} findings",
                "attempts": subtask.attempts + 1,
            }
        )
        self.log.info("researcher %s -> %d evidence items", subtask.id, len(evidence))
        return done, evidence

    def _search(self, subtask: Subtask, registry: ToolRegistry, on_retry: Optional[OnRetry]):
        """Run the planned queries (or the description), merge distinct results, and read
        the top pages. Returns (results, error); an error only if every search failed."""
        cfg = self.settings
        queries = (subtask.queries or [subtask.description])[: max(1, cfg.search_queries)]
        merged: list[dict] = []
        seen: set[str] = set()
        errors: list[str] = []
        for q in queries:
            r = registry.run("web_search", query=q, on_retry=on_retry)
            if not r.ok:
                errors.append(r.error or "search failed")
                continue
            for item in r.data or []:
                k = _key(item.get("url") or "")
                if k and k not in seen:
                    seen.add(k)
                    merged.append(dict(item))
        if not merged and errors:
            return None, errors[0]
        merged = merged[: cfg.search_results_kept]
        self._read_pages(merged, " ".join([subtask.description, *queries]))
        return merged, None

    def _read_pages(self, results: list[dict], question: str) -> None:
        """Attach the most relevant passages of the top pages, read in parallel. A page
        that can't be read keeps its snippet."""
        from ..tools import fetch_page  # looked up at call time: recordings replace fetch_text

        top = [r for r in results if r.get("url")][: self.settings.fetch_pages]
        if not top:
            return
        timeout = self.settings.fetch_timeout_seconds
        with ThreadPoolExecutor(max_workers=len(top)) as pool:
            texts = list(pool.map(lambda r: fetch_page.fetch_text(r["url"], timeout), top))
        read = 0
        for r, text in zip(top, texts):
            excerpt = fetch_page.passages(text, question, self.settings.page_chars) if text else ""
            if excerpt:
                r["page"] = excerpt
                read += 1
        self.log.info("researcher read %d of %d pages", read, len(top))

    def _fail(self, subtask: Subtask, error: str) -> Subtask:
        self.log.warning("researcher %s failed: %s", subtask.id, error)
        return subtask.model_copy(
            update={
                "status": SubtaskStatus.failed,
                "error": error,
                "attempts": subtask.attempts + 1,
            }
        )
