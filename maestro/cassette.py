"""Record a live run, replay it offline — a regression test made from real model output.

Live runs are slow, spend free-tier quota and never repeat exactly, so a change can
quietly make Maestro worse. A *cassette* captures one live run at its two outside
boundaries — every model response and every raw web search result — and replays it
offline in about a second. The hooks sit outside Maestro's own code (the model
factory and the raw search call), so everything in between still runs for real on
replay: planning validation, the scheduler, grounding and code checks, the critic's
decision, routing, recovery, citation handling. The cassette also stores what that
run produced (status, verdicts, event order, the brief), and a replay must produce
the same.

Model responses are keyed by output type and order (a researcher's by its
sub-question, since researchers run in parallel), not by prompt text, so editing a
prompt does not invalidate a cassette — the replay then checks that the code around
the model still behaves the same on real output. Adding or removing a model call
does invalidate it, loudly (``ReplayMiss``): re-record.

    python scripts/record_run.py "Compare solar and wind ..." --name solar-wind
    pytest tests/test_replays.py
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

FORMAT = 2  # 2: page text recorded alongside searches

# Settings that change behaviour. Secrets, paths and server options are not recorded.
_NOT_RECORDED = re.compile(r"key|path|dir|host|port|cors|provider_base|base_url")


class ReplayMiss(BaseException):
    """The code asked for a model response or search the cassette does not have.

    Deliberately not an ``Exception``: Maestro's tools and recovery turn errors into
    structured failures and carry on, which is right for a real outage but would let a
    replay quietly take a different path. Like KeyboardInterrupt, this must stop the
    run instead of being absorbed by ``except Exception``.
    """


class Book:
    """The calls of one run: model outputs by key, raw search results by query."""

    def __init__(self, calls: Optional[dict] = None, searches: Optional[dict] = None,
                 pages: Optional[dict] = None) -> None:
        self.calls: dict[str, dict] = dict(calls or {})
        self.searches: dict[str, list] = dict(searches or {})
        self.pages: dict[str, Optional[str]] = dict(pages or {})  # extracted page text, None = unreadable
        self._seen: Counter = Counter()
        self._lock = threading.Lock()

    def key(self, schema: str, messages: Any) -> str:
        """``Plan#0``, ``CriticOutput#1``; a researcher's ``ResearchFindings|<sub-question>#0``."""
        human = messages[-1][1] if messages and isinstance(messages[-1], tuple) else str(messages)
        m = re.match(r"Sub-question: (.*)", human)
        base = f"{schema}|{m.group(1).strip()}" if schema == "ResearchFindings" and m else schema
        with self._lock:
            n = self._seen[base]
            self._seen[base] += 1
        return f"{base}#{n}"


# --- recording -----------------------------------------------------------------------
class _Recorder:
    def __init__(self, real: Any, book: Book) -> None:
        self.real, self.book = real, book

    def with_structured_output(self, schema):
        inner = self.real.with_structured_output(schema)
        book = self.book

        class _Structured:
            def invoke(self, messages):
                out = inner.invoke(messages)  # only a success is recorded; retries are not calls
                book.calls[book.key(schema.__name__, messages)] = {
                    "schema": schema.__name__, "output": out.model_dump(mode="json")}
                return out

        return _Structured()

    def invoke(self, messages):
        resp = self.real.invoke(messages)
        self.book.calls[self.book.key("text", messages)] = {"schema": "text", "output": getattr(resp, "content", str(resp))}
        return resp


# --- replaying -----------------------------------------------------------------------
class _Replayer:
    def __init__(self, book: Book) -> None:
        self.book = book

    def _take(self, schema: str, messages: Any) -> Any:
        key = self.book.key(schema, messages)
        if key not in self.book.calls:
            raise ReplayMiss(f"the cassette has no recorded {key!r}: the run now makes a model call it didn't "
                             f"when recorded. Re-record it (scripts/record_run.py).")
        return self.book.calls[key]["output"]

    def with_structured_output(self, schema):
        take = self._take

        class _Structured:
            def invoke(self, messages):
                return schema.model_validate(take(schema.__name__, messages))

        return _Structured()

    def invoke(self, messages):
        return SimpleNamespace(content=self._take("text", messages))


@contextmanager
def _patched(model_factory, search, page):
    """Swap the model factory, the raw search call and the page fetch; everything
    between them stays real."""
    import maestro.llm as llm
    import maestro.tools.fetch_page as fp
    import maestro.tools.web_search as ws

    real_model, real_search, real_page = llm.get_chat_model, ws._ddgs_text, fp.fetch_text
    llm.get_chat_model = model_factory(real_model)
    ws._ddgs_text = search(real_search)
    fp.fetch_text = page(real_page)
    try:
        yield
    finally:
        llm.get_chat_model, ws._ddgs_text, fp.fetch_text = real_model, real_search, real_page


@contextmanager
def recording(book: Book):
    def models(real):
        return lambda *a, **k: _Recorder(real(*a, **k), book)

    def search(real):
        def run(query, n):
            raw = real(query, n)
            book.searches[query] = raw
            return raw
        return run

    def page(real):
        def run(url, timeout=6.0):
            text = real(url, timeout)
            book.pages[url] = text
            return text
        return run

    with _patched(models, search, page):
        yield book


@contextmanager
def replaying(book: Book):
    def models(_real):
        return lambda *a, **k: _Replayer(book)

    def search(_real):
        def run(query, n):
            if query not in book.searches:
                raise ReplayMiss(f"the cassette has no search for {query!r}. Re-record it (scripts/record_run.py).")
            return book.searches[query]
        return run

    def page(_real):
        def run(url, timeout=6.0):
            if url not in book.pages:
                raise ReplayMiss(f"the cassette has no page {url!r}. Re-record it (scripts/record_run.py).")
            return book.pages[url]
        return run

    with _patched(models, search, page):
        yield book


# --- cassettes -----------------------------------------------------------------------
def settings_snapshot(settings) -> dict:
    return {k: v for k, v in settings.model_dump().items() if not _NOT_RECORDED.search(k)}


def summarize(final: dict) -> dict:
    """What a run produced, in a form two runs of the same calls must agree on.

    Researchers run in parallel, so their events can interleave differently; they are
    compared as counts, everything else in order. Backoff retries only happen live.
    """
    events = [e for e in final.get("trace", []) if e.event_type.value != "rate_limit_backoff"]
    ans = final.get("final_output")
    return {
        "status": final.get("status"),
        "sequence": [f"{e.event_type.value}:{e.agent}" for e in events if e.agent != "researcher"],
        "researcher_events": dict(sorted(Counter(e.event_type.value for e in events if e.agent == "researcher").items())),
        "critic": [v.decision.value for v in final.get("critic_verdicts", [])],
        "open_issues": [[f"{i.severity}:{i.origin}" for i in v.open_issues()] for v in final.get("critic_verdicts", [])],
        "replans": final.get("replans", 0),
        "recoveries": final.get("recovery_attempts", 0),
        "subtasks": sorted(f"{s.id}:{s.status.value}" for s in final.get("subtasks", [])),
        "evidence": len(final.get("evidence", [])),
        "brief_sha256": hashlib.sha256(ans.content.encode()).hexdigest()[:16] if ans else None,
        "cited": [c.n for c in ans.cited] if ans else [],
        "validated": ans.validated if ans else None,
    }


def save(path: str, goal: str, settings, book: Book, final: dict, note: str = "") -> dict:
    data = {
        "format": FORMAT, "goal": goal, "note": note,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings_snapshot(settings),
        "calls": dict(sorted(book.calls.items())), "searches": book.searches, "pages": book.pages,
        "expected": summarize(final),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    return data


def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: cassette format {data.get('format')}, expected {FORMAT}")
    return data


def replay(data: dict) -> dict:
    """Run the cassette's goal offline against its recorded calls; return the final state."""
    from .config import Settings
    from .graph import build_default_agents, run_goal

    settings = Settings(**data["settings"])
    with replaying(Book(data["calls"], data["searches"], data.get("pages"))):
        return run_goal(data["goal"], agents=build_default_agents(settings), settings=settings)
