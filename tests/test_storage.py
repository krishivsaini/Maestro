"""Durable storage: runs and thread memory survive a restart.

Every test runs on SQLite, and on Postgres when MAESTRO_TEST_DATABASE_URL points at
one (GitHub Actions runs a Postgres service for this). "A restart" is a second app
built over the same database, as after a deploy.
"""

import os

import pytest
from fastapi.testclient import TestClient

from maestro.agents.analyst import AnalysisModel
from maestro.config import Settings
from maestro.db import Database
from maestro.memory import SqlMemory
from maestro.serve import create_app
from maestro.state import CriticDecision
from tests.test_serve import _parse_sse, build_stub_agents

PG = os.environ.get("MAESTRO_TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def target(request, tmp_path):
    if request.param == "sqlite":
        return str(tmp_path / "maestro.db")
    if not PG:
        pytest.skip("set MAESTRO_TEST_DATABASE_URL to test against Postgres")
    db = Database(PG)
    db.script("DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS events; DROP TABLE IF EXISTS memory")
    db.close()
    return PG


def app_over(target, make_stub):
    """A fresh app (a 'restart') whose agents write topical findings to memory."""
    import maestro.serve as serve

    def topical(responder):
        def wrapped(schema, messages):
            if getattr(schema, "__name__", "") == "AnalysisModel":
                return AnalysisModel(content="We recommend wind.", claims=[
                    "wind has the higher capacity factor for the grid [1]",
                    "solar output peaks midday and needs storage [1]"])
            return responder(schema, messages)
        return make_stub(wrapped)

    cfg = Settings(database_url=target)
    original = serve.get_settings
    serve.get_settings = lambda: cfg
    try:
        return TestClient(create_app(agents=build_stub_agents(topical, [CriticDecision.passed])))
    finally:
        serve.get_settings = original


def test_runs_and_memory_survive_a_restart(target, make_stub):
    first = app_over(target, make_stub)
    frames = _parse_sse(first.post("/run", json={"goal": "Compare solar and wind", "thread_id": "t1"}).text)
    run_id = frames[0]["run_id"]

    restarted = app_over(target, make_stub)
    runs = restarted.get("/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == [run_id]
    record = restarted.get(f"/runs/{run_id}").json()
    assert record["status"] == "completed" and record["final_output"]["content"] == "FINAL BRIEF"
    assert restarted.get("/threads/t1").json()["memory_findings"] == 2

    # a follow-up after the restart recalls the thread's findings from the database
    goal = 'Follow-up to "Compare solar and wind": how much storage does solar need?'
    frames = _parse_sse(restarted.post("/run", json={"goal": goal, "thread_id": "t1"}).text)
    recalled = next(f for f in frames if f["type"] == "memory_recall")["recalled"]
    assert any("solar" in r for r in recalled)
    assert len(restarted.get("/threads/t1").json()["runs"]) == 2


def test_memory_is_thread_scoped_and_ranked(target):
    mem = SqlMemory(Database(target))
    mem.add("a", "wind capacity factor is about 35 percent")
    mem.add("a", "battery storage costs fell sharply")
    mem.add("b", "wind turbines in thread b")
    hits = mem.query("a", "wind capacity factor", k=3)
    assert [h.content for h in hits][0] == "wind capacity factor is about 35 percent"
    assert all(h.thread_id == "a" for h in hits)
    assert mem.count("a") == 2 and mem.count("b") == 1 and mem.count("nobody") == 0
    assert mem.findings("a") == ["wind capacity factor is about 35 percent", "battery storage costs fell sharply"]
    assert mem.query("nobody", "wind") == []


def test_a_failed_write_leaves_nothing_half_saved(target):
    from maestro.trace import TraceStore

    store = TraceStore(target)
    with pytest.raises(RuntimeError):
        with store.db.transaction():
            store.db.upsert("runs", "run_id", {"run_id": "r1", "thread_id": "t", "goal": "g", "status": "s",
                                               "created_at": "c", "event_count": 0, "payload": "{}"})
            raise RuntimeError("crash mid-save")
    assert store.list_runs() == []


def test_upsert_replaces_a_run_rather_than_duplicating_it(target):
    db = Database(target)
    db.script("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
    db.execute("DELETE FROM kv")
    db.upsert("kv", "k", {"k": "a", "v": "1"})
    db.upsert("kv", "k", {"k": "a", "v": "2"})
    assert db.query("SELECT k, v FROM kv") == [{"k": "a", "v": "2"}]


@pytest.mark.skipif(not PG, reason="Postgres only: serverless hosts drop idle connections")
def test_a_dropped_postgres_connection_reconnects():
    db = Database(PG)
    db.conn.close()  # as a host closing an idle connection would
    assert db.query("SELECT 1 AS one") == [{"one": 1}]


def test_a_database_url_never_reaches_a_cassette():
    from maestro.cassette import settings_snapshot

    snap = settings_snapshot(Settings(database_url="postgresql://user:hunter2@db.example/x"))
    assert "database_url" not in snap and "hunter2" not in str(snap)


def test_health_says_which_storage_is_live(target, make_stub):
    app = app_over(target, make_stub)
    expected = "postgres" if target.startswith("postgres") else "sqlite"
    assert app.get("/healthz").json() == {"status": "ok", "storage": expected}


def test_an_unreachable_database_falls_back_instead_of_taking_the_demo_down(make_stub):
    app = app_over("postgresql://nobody:x@127.0.0.1:1/nothing", make_stub)  # nothing listens on port 1
    assert app.get("/healthz").json() == {"status": "ok", "storage": "sqlite (database unreachable)"}
    assert app.post("/run", json={"goal": "Compare solar and wind"}).status_code == 200
