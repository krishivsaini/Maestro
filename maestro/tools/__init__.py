"""Tool package — exposes the registry and a default-populated registry."""

from __future__ import annotations

from . import retrieve as _retrieve
from . import web_search as _web_search
from .registry import Tool, ToolRegistry, ToolResult

__all__ = ["Tool", "ToolRegistry", "ToolResult", "build_default_registry", "DEFAULT_REGISTRY"]


def build_default_registry() -> ToolRegistry:
    """Construct a registry with the standard Maestro tools."""
    reg = ToolRegistry()
    reg.register(
        Tool(
            name=_web_search.TOOL_NAME,
            description=(
                "Search the web for evidence on a query. Returns a list of "
                "{title, url, snippet} results, an empty result, or a structured failure."
            ),
            func=_web_search.web_search,
            schema={"query": "str", "max_results": "int?"},
        )
    )
    reg.register(
        Tool(
            name=_retrieve.TOOL_NAME,
            description=(
                "Simple bag-of-words cosine retrieval over a provided corpus "
                "(deliberately basic, not production RAG)."
            ),
            func=_retrieve.retrieve,
            schema={"query": "str", "corpus": "list", "k": "int?"},
        )
    )
    return reg


DEFAULT_REGISTRY = build_default_registry()
