"""Rate-limit resilience — backoff + jitter around every LLM/tool call (§5, §15).

Built on Day 2, *before* anything parallelizes, because 429s appear immediately
under parallel load on the Gemini free tier. Nothing that calls the model or a
tool should do so without going through here.

Usage
-----
Functional::

    result = resilient_call(model.invoke, messages, on_retry=record_backoff)

Decorator::

    @resilient()
    def call_tool(...): ...

The optional ``on_retry(attempt, exc, is_rate_limit)`` callback lets a node append
a ``rate_limit_backoff`` TraceEvent so retries are *visible*, not silent.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Protocol

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import Settings, get_settings
from .logging_config import get_logger

log = get_logger("resilience")

# Substrings that identify a provider rate-limit / quota error across SDKs.
_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "resource exhausted",
    "resourceexhausted",
    "quota",
    "too many requests",
)

# Phrases / class-name fragments for transient server errors worth retrying.
_TRANSIENT_MARKERS = (
    "unavailable",
    "overloaded",
    "try again",
    "internal error",
    "internal server error",
    "deadline exceeded",
    "temporarily",
)
_TRANSIENT_CODES = {500, 502, 503, 504}

# Class-name fragments for transient server + network transport errors (httpx/httpcore,
# google/grpc), matched by name so resilience.py needn't import those libraries.
_TRANSIENT_NAME_FRAGMENTS = (
    "servererror",
    "serviceunavailable",
    "internalservererror",
    "deadlineexceeded",
    "connecterror",
    "connecttimeout",
    "connectionerror",
    "readtimeout",
    "readerror",
    "writeerror",
    "pooltimeout",
    "remoteprotocol",
    "proxyerror",
    "transporterror",
    "timeout",
)

# Transient network errors we also retry (in addition to rate limits).
DEFAULT_RETRY_ON: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)


class OnRetry(Protocol):
    def __call__(self, attempt: int, exc: BaseException, is_rate_limit: bool) -> None: ...


def is_rate_limit_error(exc: BaseException) -> bool:
    """True if the exception looks like a provider rate-limit / quota error."""
    name = type(exc).__name__.lower()
    if "resourceexhausted" in name or "ratelimit" in name:
        return True
    # google-genai attaches an HTTP-ish status code on some error types
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def is_transient_error(exc: BaseException) -> bool:
    """True for errors worth retrying: rate limits + transient 5xx server errors.

    Providers return 503 UNAVAILABLE ("high demand"), 500, 502, 504, and
    deadline-exceeded errors that are temporary and clear on retry. These are
    distinct from rate limits but handled by the same backoff.
    """
    if is_rate_limit_error(exc):
        return True
    # builtin network errors
    if isinstance(exc, OSError):
        return True
    name = type(exc).__name__.lower()
    if any(frag in name for frag in _TRANSIENT_NAME_FRAGMENTS):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in _TRANSIENT_CODES:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def _retry_predicate(retry_on: tuple[type[BaseException], ...]) -> Callable[[BaseException], bool]:
    def predicate(exc: BaseException) -> bool:
        return is_transient_error(exc) or isinstance(exc, retry_on)

    return predicate


def resilient_call(
    func: Callable[..., Any],
    *args: Any,
    settings: Optional[Settings] = None,
    on_retry: Optional[OnRetry] = None,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRY_ON,
    **kwargs: Any,
) -> Any:
    """Invoke ``func(*args, **kwargs)`` with exponential backoff + jitter.

    Retries rate-limit errors (always) and ``retry_on`` transient errors, up to
    ``settings.backoff_max_attempts``. Re-raises the final exception if all
    attempts are exhausted (``reraise=True``) so callers/tools can convert it into
    a structured failure.
    """
    cfg = settings or get_settings()
    fname = getattr(func, "__name__", repr(func))

    def _before_sleep(rcs: RetryCallState) -> None:
        exc = rcs.outcome.exception() if rcs.outcome else None
        attempt = rcs.attempt_number
        rate_limited = bool(exc) and is_rate_limit_error(exc)
        if rate_limited:
            log.warning("rate-limit backoff (attempt %d) on %s", attempt, fname)
        else:
            log.warning("transient backoff (attempt %d) on %s: %s", attempt, fname, exc)
        if on_retry and exc is not None:
            on_retry(attempt, exc, rate_limited)

    retryer = Retrying(
        retry=retry_if_exception(_retry_predicate(retry_on)),
        wait=wait_exponential_jitter(
            initial=cfg.backoff_base_seconds,
            max=cfg.backoff_max_seconds,
            jitter=cfg.backoff_jitter_seconds,
        ),
        stop=stop_after_attempt(cfg.backoff_max_attempts),
        before_sleep=_before_sleep,
        reraise=True,
    )
    return retryer(func, *args, **kwargs)


def resilient(
    *,
    settings: Optional[Settings] = None,
    on_retry: Optional[OnRetry] = None,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRY_ON,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of :func:`resilient_call`."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return resilient_call(
                func, *args, settings=settings, on_retry=on_retry, retry_on=retry_on, **kwargs
            )

        return wrapper

    return decorator
