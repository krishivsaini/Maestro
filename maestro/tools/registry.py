"""Tool registry + the structured result type (§13).

Tools return a ``ToolResult`` (ok / structured failure) and **never raise raw**
into the graph — the *supervisor* decides what a failure means (§15, §22.9).
``ToolRegistry.run`` is the safety net: even an unexpected exception is converted
into a structured failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..logging_config import get_logger

log = get_logger("tools.registry")


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: Any = None
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def success(cls, tool: str, data: Any, **meta: Any) -> "ToolResult":
        return cls(ok=True, tool=tool, data=data, meta=meta)

    @classmethod
    def failure(cls, tool: str, error: str, **meta: Any) -> "ToolResult":
        return cls(ok=False, tool=tool, error=error, meta=meta)

    @property
    def is_empty(self) -> bool:
        return bool(self.ok and self.meta.get("empty"))


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., ToolResult]
    schema: dict = field(default_factory=dict)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "schema": t.schema} for t in self._tools.values()]

    def run(self, name: str, **kwargs: Any) -> ToolResult:
        """Run a tool by name, guaranteeing a structured result (never a raw raise)."""
        if name not in self._tools:
            return ToolResult.failure(name, f"unknown tool: {name}")
        try:
            return self._tools[name].func(**kwargs)
        except Exception as exc:  # tools should self-handle; this is the last-resort net
            log.exception("tool %s raised unexpectedly", name)
            return ToolResult.failure(name, f"{type(exc).__name__}: {exc}")
