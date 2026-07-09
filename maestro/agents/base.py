"""Shared subagent scaffolding (§7, §18.1).

The single most important property here is **separate contexts**: a subagent is
built from *only* its own system prompt plus a scoped human input that the caller
assembles from structured state fields. There is no shared scratchpad and no way
for one subagent to read another's raw reasoning — the only hand-off between
specialists is structured data (Evidence, AnalysisDraft, Verdict) routed via the
supervisor. ``_messages`` is the chokepoint that guarantees this.

Every model call goes through ``resilient_call`` (backoff + jitter). A stub model
can be injected for offline/deterministic tests.
"""

from __future__ import annotations

from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from ..config import Settings, get_settings
from ..logging_config import get_logger
from ..resilience import OnRetry, resilient_call
from ..state import Role

T = TypeVar("T", bound=BaseModel)


class Subagent:
    """Base class for a role-specialized subagent with its own prompt/context."""

    role: Role
    name: str
    system_prompt: str = ""

    def __init__(self, *, model: Any = None, settings: Optional[Settings] = None) -> None:
        self._chat = model  # injectable; None -> lazily build the real Gemini model
        self.settings = settings or get_settings()
        self.log = get_logger(f"agent.{getattr(self, 'name', 'subagent')}")

    def _get_chat(self) -> Any:
        if self._chat is None:
            from ..llm import get_chat_model  # lazy

            self._chat = get_chat_model(self.settings)
        return self._chat

    def _messages(self, human: str) -> list[tuple[str, str]]:
        """A subagent sees ONLY its own system prompt + its scoped human input.

        This is the separate-context guarantee (§18.1): no shared scratchpad, no
        other agents' internals — only what the caller explicitly scopes in.
        """
        return [("system", self.system_prompt), ("human", human)]

    def _structured(self, schema: Type[T], human: str, *, on_retry: Optional[OnRetry] = None) -> T:
        structured = self._get_chat().with_structured_output(schema)
        return resilient_call(structured.invoke, self._messages(human), on_retry=on_retry)

    def _generate(self, human: str, *, on_retry: Optional[OnRetry] = None) -> str:
        resp = resilient_call(self._get_chat().invoke, self._messages(human), on_retry=on_retry)
        return getattr(resp, "content", str(resp))
