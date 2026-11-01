"""Usage limits and input validation for the public demo.

One shared free-tier key serves every visitor, so runs are limited per visitor and
per day before any model call, and requests are validated with messages a visitor
can act on.
"""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from maestro.config import Settings
from maestro.limits import DailyCap, RunLimits, SlidingWindow
from maestro.serve import create_app
from maestro.state import CriticDecision
from maestro.trace import TraceStore
from tests.test_serve import build_stub_agents


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_sliding_window_allows_the_limit_then_says_how_long_to_wait():
    clock = Clock()
    w = SlidingWindow(limit=2, window=60, clock=clock)
    assert w.take("ip") is None and w.take("ip") is None
    assert w.take("ip") == 61
    assert w.take("other") is None          # per key
    clock.t += 30
    assert w.take("ip") == 31
    clock.t += 31
    assert w.take("ip") is None             # the oldest event has aged out


def test_daily_cap_resets_at_utc_midnight():
    now = [datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc)]
    cap = DailyCap(limit=1, today=lambda: now[0])
    assert cap.take() is None
    assert cap.take() == 3600               # an hour to midnight
    now[0] += timedelta(hours=1, seconds=1)
    assert cap.take() is None


def test_own_key_runs_get_more_and_skip_the_shared_daily_cap():
    s = Settings(run_limit_per_ip=1, byok_limit_multiplier=3, shared_daily_runs=1)
    limits = RunLimits(s, clock=Clock())
    assert limits.check("a", own_key=False) is None
    r = limits.check("a", own_key=False)
    assert r.reason == "visitor" and "add your own free Gemini API key" in r.message
    assert all(limits.check("a", own_key=True) is None for _ in range(3))   # 3x, on their own quota
    assert limits.check("a", own_key=True).reason == "visitor"
    r = limits.check("b", own_key=False)                                   # someone else, but the day is used up
    assert r.reason == "daily" and "today's runs" in r.message


def client(**settings):
    import maestro.serve as serve

    cfg = Settings(**settings)
    original = serve.get_settings
    serve.get_settings = lambda: cfg
    try:
        app = create_app(agents=build_stub_agents(_stub, [CriticDecision.passed]), trace_store=TraceStore(":memory:"))
    finally:
        serve.get_settings = original
    return TestClient(app)


def _stub(responder):
    from conftest import StubModel
    return StubModel(responder)


def test_a_visitor_over_the_limit_is_refused_before_any_model_call():
    c = client(run_limit_per_ip=2, run_limit_window_minutes=60)
    for _ in range(2):
        assert c.post("/run", json={"goal": "Compare solar and wind"}).status_code == 200
    r = c.post("/run", json={"goal": "Compare solar and wind"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 3500
    assert r.json()["reason"] == "visitor" and "limit of 2 runs per 60 minutes" in r.json()["error"]
    # their own key still works
    assert c.post("/run", json={"goal": "Compare solar and wind", "api_key": "AIza-own"}).status_code == 200


def test_the_shared_daily_cap_covers_all_visitors():
    c = client(run_limit_per_ip=0, shared_daily_runs=1, trust_proxy=True)
    assert c.post("/run", json={"goal": "Compare solar and wind"}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    r = c.post("/run", json={"goal": "Compare solar and wind"}, headers={"X-Forwarded-For": "2.2.2.2"})
    assert r.status_code == 429 and r.json()["reason"] == "daily"


def test_forwarded_ips_are_trusted_only_behind_a_proxy():
    trusting = client(run_limit_per_ip=1, trust_proxy=True)
    assert trusting.post("/run", json={"goal": "Compare A and B"}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert trusting.post("/run", json={"goal": "Compare A and B"}, headers={"X-Forwarded-For": "2.2.2.2, 10.0.0.1"}).status_code == 200
    # not behind a proxy: a forged header doesn't buy a fresh limit
    direct = client(run_limit_per_ip=1, trust_proxy=False)
    assert direct.post("/run", json={"goal": "Compare A and B"}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert direct.post("/run", json={"goal": "Compare A and B"}, headers={"X-Forwarded-For": "9.9.9.9"}).status_code == 429


def test_invalid_requests_get_a_message_a_visitor_can_act_on():
    c = client()
    r = c.post("/run", json={"goal": "x" * 501})
    assert r.status_code == 422 and r.json() == {
        "error": "Questions can be up to 500 characters. Shorten it and try again.", "field": "goal"}
    r = c.post("/run", json={"goal": "Compare A and B", "thread_id": "../../etc"})
    assert r.status_code == 422 and r.json()["field"] == "thread_id"
    # a follow-up carries its opening question; only the asked part counts toward the limit
    long_root = "Compare " + "a " * 240 + "and b"
    r = c.post("/run", json={"goal": f'Follow-up to "{long_root}": what changed?'})
    assert r.status_code == 200
