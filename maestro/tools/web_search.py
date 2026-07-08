"""Web search tool (keyless DuckDuckGo default) with a fault-injection switch (§13).

The fault-injection switch (``MAESTRO_FAULT_INJECTION=true`` +
``MAESTRO_FAULT_INJECTION_TOOL=web_search``) makes this tool return a structured
failure on demand, which drives the visible-recovery demo (§15). A genuine
no-results response is returned as ``ok=True`` but empty, so the supervisor — not
the tool — decides whether an empty result warrants recovery.
"""

from __future__ import annotations

from typing import Optional

from ..config import get_settings
from ..logging_config import get_logger
from ..resilience import OnRetry, resilient_call
from .registry import ToolResult

log = get_logger("tools.web_search")

TOOL_NAME = "web_search"


def _ddgs_text(query: str, max_results: int) -> list[dict]:
    """Raw DuckDuckGo call. Imported lazily so the module loads without the dep."""
    try:
        from ddgs import DDGS  # maintained package name (2026)
    except ImportError:  # pragma: no cover - fallback for older installs
        from duckduckgo_search import DDGS  # type: ignore
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def web_search(
    query: str,
    max_results: Optional[int] = None,
    on_retry: Optional[OnRetry] = None,
) -> ToolResult:
    settings = get_settings()

    # --- fault injection for the recovery demo ---
    if settings.fault_injection and settings.fault_injection_tool == TOOL_NAME:
        log.warning("web_search fault injection ACTIVE -> structured failure")
        return ToolResult.failure(TOOL_NAME, "injected_fault: web_search unavailable", injected=True)

    n = max_results or settings.search_max_results
    try:
        raw = resilient_call(_ddgs_text, query, n, on_retry=on_retry)
    except Exception as exc:  # backoff exhausted / provider down -> structured failure
        return ToolResult.failure(TOOL_NAME, f"{type(exc).__name__}: {exc}", query=query)

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href") or r.get("url", ""),
            "snippet": r.get("body") or r.get("snippet", ""),
        }
        for r in (raw or [])
    ]
    if not results:
        return ToolResult.success(TOOL_NAME, [], empty=True, query=query)
    return ToolResult.success(TOOL_NAME, results, query=query, count=len(results))
