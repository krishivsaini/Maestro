"""Simple retrieval tool — DELIBERATELY BASIC (§1.3, §22.7).

This is bag-of-words cosine over a provided corpus. It is *intentionally* not
production RAG: no embeddings, no hybrid retrieval, no reranking. The signal in
this project is orchestration, not retrieval quality. Long-term memory (Day 10)
uses real embeddings; this tool stays simple on purpose.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Optional, Sequence

from ..config import get_settings
from ..logging_config import get_logger
from .registry import ToolResult

log = get_logger("tools.retrieve")

TOOL_NAME = "retrieve"
_WORD = re.compile(r"[a-z0-9]+")


def _bag(text: str) -> Counter:
    return Counter(_WORD.findall(text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def retrieve(
    query: str,
    corpus: Sequence[Any],
    k: int = 3,
    on_retry: Any = None,  # accepted for a uniform tool signature; unused (no network)
) -> ToolResult:
    settings = get_settings()

    if settings.fault_injection and settings.fault_injection_tool == TOOL_NAME:
        log.warning("retrieve fault injection ACTIVE -> structured failure")
        return ToolResult.failure(TOOL_NAME, "injected_fault: retrieve unavailable", injected=True)

    if not corpus:
        return ToolResult.success(TOOL_NAME, [], empty=True, query=query)

    qv = _bag(query)
    scored: list[tuple[float, str, str]] = []
    for doc in corpus:
        if isinstance(doc, str):
            content, source = doc, "corpus"
        else:
            content = doc.get("content", "")
            source = doc.get("source", "corpus")
        scored.append((_cosine(qv, _bag(content)), source, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [
        {"source": s, "content": c, "score": round(sc, 4)}
        for sc, s, c in scored[:k]
        if sc > 0
    ]
    if not top:
        return ToolResult.success(TOOL_NAME, [], empty=True, query=query)
    return ToolResult.success(TOOL_NAME, top, query=query, count=len(top))
