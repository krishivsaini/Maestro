"""What a run costs: model calls, tokens and time, per agent.

Every model client is created by ``llm.get_chat_model``; given a run's ``UsageMeter``,
it returns the client wrapped so each call records its agent, model, input and output
tokens, latency, and whether it succeeded. Failed attempts are recorded too: a call
the backoff retried three times cost three calls. Memory's embedding calls are counted
alongside.

Tokens come from the model's raw response (structured output is requested with
``include_raw``); a client that can't return one — a test stub, a cassette replay — is
still counted, with its time, and no tokens.

The free tier costs nothing, so no price is assumed: set ``price_input_per_million``
and ``price_output_per_million`` to see a dollar estimate.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class Call:
    agent: str
    model: str
    kind: str            # "chat" | "embedding"
    input_tokens: int
    output_tokens: int
    seconds: float
    ok: bool


class UsageMeter:
    def __init__(self) -> None:
        self.calls: list[Call] = []
        self._lock = threading.Lock()

    def record(self, agent: str, model: str, *, kind: str = "chat", input_tokens: int = 0,
               output_tokens: int = 0, seconds: float = 0.0, ok: bool = True) -> None:
        with self._lock:
            self.calls.append(Call(agent, model, kind, int(input_tokens or 0), int(output_tokens or 0),
                                   round(seconds, 3), ok))

    def summary(self, settings=None) -> dict:
        with self._lock:
            calls = list(self.calls)
        chat = [c for c in calls if c.kind == "chat"]
        by_agent: dict[str, dict] = {}
        for c in chat:
            a = by_agent.setdefault(c.agent, {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
                                              "seconds": 0.0})
            a["calls"] += 1
            a["failed"] += 0 if c.ok else 1
            a["input_tokens"] += c.input_tokens
            a["output_tokens"] += c.output_tokens
            a["seconds"] = round(a["seconds"] + c.seconds, 3)
        out = {
            "calls": len(chat),
            "failed_calls": sum(not c.ok for c in chat),
            "input_tokens": sum(c.input_tokens for c in chat),
            "output_tokens": sum(c.output_tokens for c in chat),
            "model_seconds": round(sum(c.seconds for c in chat), 2),
            "embedding_calls": sum(c.kind == "embedding" for c in calls),
            "models": sorted({c.model for c in chat}),
            "by_agent": by_agent,
        }
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
        if settings is not None and (settings.price_input_per_million or settings.price_output_per_million):
            out["estimated_cost_usd"] = round(out["input_tokens"] / 1e6 * settings.price_input_per_million
                                              + out["output_tokens"] / 1e6 * settings.price_output_per_million, 6)
        return out

    def raw(self) -> list[dict]:
        with self._lock:
            return [asdict(c) for c in self.calls]


class _MeteredStructured:
    def __init__(self, inner: Any, raw_mode: bool, meter: UsageMeter, agent: str, model: str) -> None:
        self.inner, self.raw_mode, self.meter, self.agent, self.model = inner, raw_mode, meter, agent, model

    def invoke(self, messages, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = self.inner.invoke(messages, *args, **kwargs)
        except Exception:
            self.meter.record(self.agent, self.model, seconds=time.perf_counter() - t0, ok=False)
            raise
        seconds = time.perf_counter() - t0
        if not self.raw_mode:
            self.meter.record(self.agent, self.model, seconds=seconds)
            return out
        parsed, error = out.get("parsed"), out.get("parsing_error")
        usage = getattr(out.get("raw"), "usage_metadata", None) or {}
        ok = error is None and parsed is not None
        self.meter.record(self.agent, self.model, input_tokens=usage.get("input_tokens", 0),
                          output_tokens=usage.get("output_tokens", 0), seconds=seconds, ok=ok)
        if error is not None:
            raise error
        if parsed is None:
            raise ValueError("the model returned no structured output")
        return parsed


class _Metered:
    """A chat model whose calls are recorded on a meter; otherwise the model itself."""

    def __init__(self, real: Any, meter: UsageMeter, agent: str, model: str) -> None:
        self.real, self.meter, self.agent, self.model = real, meter, agent, model

    def with_structured_output(self, schema, **kwargs):
        try:
            inner, raw_mode = self.real.with_structured_output(schema, include_raw=True, **kwargs), True
        except TypeError:  # a client without raw responses: count calls, no tokens
            inner, raw_mode = self.real.with_structured_output(schema, **kwargs), False
        return _MeteredStructured(inner, raw_mode, self.meter, self.agent, self.model)

    def invoke(self, messages, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            resp = self.real.invoke(messages, *args, **kwargs)
        except Exception:
            self.meter.record(self.agent, self.model, seconds=time.perf_counter() - t0, ok=False)
            raise
        usage = getattr(resp, "usage_metadata", None) or {}
        self.meter.record(self.agent, self.model, input_tokens=usage.get("input_tokens", 0),
                          output_tokens=usage.get("output_tokens", 0), seconds=time.perf_counter() - t0)
        return resp

    def __getattr__(self, name):
        return getattr(self.real, name)


def metered(model: Any, meter: Optional[UsageMeter], agent: str, model_id: str) -> Any:
    return _Metered(model, meter, agent, model_id) if meter is not None else model
