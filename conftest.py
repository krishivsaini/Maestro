"""Pytest fixtures + a stub chat model for offline agent tests.

Ensures the repo root is importable so `import maestro` works without installing
the package, provides a `StubModel` that mimics the langchain chat-model surface
(`with_structured_output(...).invoke(...)` and `.invoke(...)`), and resets the
cached settings around every test so env-based fault injection can't leak.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubStructured:
    def __init__(self, schema, parent: "StubModel") -> None:
        self.schema = schema
        self.parent = parent

    def invoke(self, messages):
        self.parent.calls.append(messages)
        return self.parent._responder(self.schema, messages)


class StubModel:
    """Deterministic stand-in for a langchain chat model.

    `responder(schema, messages)` returns either a `schema` instance (for
    structured calls) or a string (for text calls). `.calls` records every
    message list seen, so tests can assert on the scoped context an agent built.
    """

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list = []

    def with_structured_output(self, schema):
        return _StubStructured(schema, self)

    def invoke(self, messages):
        self.calls.append(messages)
        return _Msg(self._responder(str, messages))


@pytest.fixture
def make_stub():
    """Return a factory: make_stub(responder) -> StubModel."""

    def factory(responder) -> StubModel:
        return StubModel(responder)

    return factory


@pytest.fixture(autouse=True)
def _reset_settings():
    """Clear the cached Settings before and after each test (fault-injection hygiene)."""
    from maestro.config import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()
