"""Resilience — rate-limit + transient error classification and retry behavior.

Backs the Day 2 backoff layer and the transient-retry improvement (503/network)
found while running the live demo.
"""

import pytest

from maestro.config import Settings
from maestro.resilience import is_rate_limit_error, is_transient_error, resilient_call

FAST = Settings(
    backoff_base_seconds=0.001,
    backoff_max_seconds=0.01,
    backoff_jitter_seconds=0.0,
    backoff_max_attempts=5,
)


class _FakeServerError(Exception):
    def __init__(self) -> None:
        self.code = 503
        super().__init__("503 UNAVAILABLE: model experiencing high demand, try again later")


class ConnectError(Exception):  # mimics httpx.ConnectError by name
    pass


def test_rate_limit_classification():
    assert is_rate_limit_error(Exception("Error 429: Too Many Requests"))
    assert is_rate_limit_error(Exception("RESOURCE_EXHAUSTED: quota"))
    assert not is_rate_limit_error(ValueError("bad input"))


def test_transient_classification_covers_5xx_and_network():
    assert is_transient_error(_FakeServerError())  # code 503
    assert is_transient_error(Exception("Service Unavailable"))
    assert is_transient_error(ConnectError("connection reset"))  # by class name
    assert is_transient_error(OSError("network down"))
    assert is_transient_error(Exception("429 rate limit"))  # rate limits are transient too
    assert not is_transient_error(ValueError("permanent bad input"))


def test_resilient_call_retries_transient_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeServerError()
        return "ok"

    assert resilient_call(flaky, settings=FAST) == "ok"
    assert calls["n"] == 3


def test_resilient_call_does_not_retry_permanent_error():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        resilient_call(boom, settings=FAST)
    assert calls["n"] == 1
