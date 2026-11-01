"""What a run costs: every model call metered, per agent, and reported with the result."""

import json
import os
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from maestro.config import Settings
from maestro.usage import UsageMeter, metered

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Answer(BaseModel):
    text: str


class RawModel:
    """A client that, like Gemini's, returns the raw message (with usage) alongside the parse."""

    def __init__(self, outcome):
        self.outcome = outcome

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw, "the meter asks for raw responses, where token counts live"
        outcome = self.outcome

        class S:
            def invoke(self, messages):
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        return S()


def raw(parsed=None, error=None, inp=120, out=30):
    return {"raw": SimpleNamespace(usage_metadata={"input_tokens": inp, "output_tokens": out}),
            "parsed": parsed, "parsing_error": error}


def test_a_call_records_its_tokens_and_returns_the_parse_unchanged():
    m = UsageMeter()
    out = metered(RawModel(raw(Answer(text="hi"))), m, "analyst", "gemini-x").with_structured_output(Answer).invoke([])
    assert out == Answer(text="hi")
    [c] = m.raw()
    assert (c["agent"], c["model"], c["input_tokens"], c["output_tokens"], c["ok"]) == ("analyst", "gemini-x", 120, 30, True)


def test_a_failed_call_still_counts_and_still_raises():
    m = UsageMeter()
    with pytest.raises(RuntimeError):
        metered(RawModel(RuntimeError("429")), m, "critic", "g").with_structured_output(Answer).invoke([])
    bad = ValueError("could not parse")
    with pytest.raises(ValueError):
        metered(RawModel(raw(error=bad)), m, "critic", "g").with_structured_output(Answer).invoke([])
    s = m.summary()
    assert s["calls"] == 2 and s["failed_calls"] == 2 and s["input_tokens"] == 120  # the parse failure was billed


def test_a_client_without_raw_responses_is_counted_without_tokens():
    class Plain:
        def with_structured_output(self, schema):
            return SimpleNamespace(invoke=lambda messages: Answer(text="x"))

    m = UsageMeter()
    assert metered(Plain(), m, "writer", "g").with_structured_output(Answer).invoke([]).text == "x"
    assert m.summary()["calls"] == 1 and m.summary()["total_tokens"] == 0


def test_no_meter_means_the_model_is_returned_untouched():
    model = RawModel(None)
    assert metered(model, None, "x", "y") is model


def test_the_summary_breaks_cost_down_by_agent_and_estimates_dollars_only_when_priced():
    m = UsageMeter()
    m.record("researcher", "g", input_tokens=1000, output_tokens=200, seconds=1.5)
    m.record("researcher", "g", input_tokens=500, output_tokens=100, seconds=1.0, ok=False)
    m.record("writer", "g", input_tokens=2000, output_tokens=800, seconds=3.0)
    m.record("memory", "e", kind="embedding", seconds=0.2)
    s = m.summary()
    assert (s["calls"], s["failed_calls"], s["total_tokens"], s["embedding_calls"]) == (3, 1, 4600, 1)
    assert s["by_agent"]["researcher"] == {"calls": 2, "failed": 1, "input_tokens": 1500, "output_tokens": 300,
                                           "seconds": 2.5}
    assert "estimated_cost_usd" not in s  # the free tier costs nothing: no price is assumed
    priced = m.summary(Settings(price_input_per_million=0.1, price_output_per_million=0.4))
    assert priced["estimated_cost_usd"] == pytest.approx(3500 / 1e6 * 0.1 + 1100 / 1e6 * 0.4)


def test_the_server_reports_each_agents_calls_with_the_result(monkeypatch):
    # A recorded run replayed through the server's real per-request path: real agents,
    # real graph, a fresh meter per request; only the model and the web are played back.
    from fastapi.testclient import TestClient

    import maestro.serve as serve
    from maestro import cassette
    from maestro.trace import TraceStore

    data = cassette.load(os.path.join(ROOT, "data", "cassettes", "solar-wind.json"))
    cfg = Settings(**data["settings"], memory_embedder="hashing", google_api_key="replay")
    monkeypatch.setattr(serve, "get_settings", lambda: cfg)
    with cassette.replaying(cassette.Book(data["calls"], data["searches"], data["pages"])):
        store = TraceStore(":memory:")
        client = TestClient(serve.create_app(trace_store=store))
        text = client.post("/run", json={"goal": data["goal"]}).text
    final = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")][-1]
    usage = final["usage"]
    recorded = {"supervisor": 1, "researcher": 2, "analyst": 1, "critic": 1, "writer": 1}
    assert {a: x["calls"] for a, x in usage["by_agent"].items()} == recorded
    assert usage["calls"] == len(data["calls"]) == 6
    # and it is kept with the run, so a past run still shows what it cost
    assert client.get(f"/runs/{final['run_id']}").json()["usage"]["calls"] == 6
