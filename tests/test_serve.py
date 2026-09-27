"""The API service (FR-31..FR-33).

Uses FastAPI's TestClient with stub-modeled agents (offline). Proves /healthz is
live, /run streams the coordination (plan -> critic reject -> critic pass -> final)
as SSE, the run is persisted, and it's reconstructable via /runs and /runs/{id}.
"""

import json

from fastapi.testclient import TestClient

from maestro.agents import Analyst, Critic, Researcher, Writer
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import CriticOutput
from maestro.agents.researcher import ResearchFinding, ResearchFindings
from maestro.agents.writer import WriterOutput
from maestro.config import Settings
from maestro.graph import MaestroAgents
from maestro.serve import create_app
from maestro.state import CriticDecision
from maestro.supervisor import heuristic_planner
from maestro.tools import DEFAULT_REGISTRY
from maestro.trace import TraceStore

CORPUS = [{"source": "src_solar", "content": "Solar capacity factor ~25%, peaks midday."},
          {"source": "src_wind", "content": "Wind capacity factor ~35%, stronger at night."}]


def build_stub_agents(make_stub, critic_decisions):
    counter = {"i": 0}

    def responder(schema, messages):
        name = getattr(schema, "__name__", "")
        if name == "ResearchFindings":
            return ResearchFindings(findings=[ResearchFinding(source="src_solar", content="solar peaks midday")])
        if name == "AnalysisModel":
            return AnalysisModel(content="analysis", claims=["c1", "c2"])
        if name == "CriticOutput":
            d = critic_decisions[min(counter["i"], len(critic_decisions) - 1)]
            counter["i"] += 1
            return CriticOutput(decision=d, feedback="" if d == CriticDecision.passed else "fix c2")
        if name == "WriterOutput":
            return WriterOutput(content="FINAL BRIEF", citations=["src_solar"])
        raise AssertionError(name)

    model = make_stub(responder)
    cfg = Settings(max_critic_iters=3, max_parallel=2)
    return MaestroAgents(
        researcher=Researcher(model=model, settings=cfg),
        analyst=Analyst(model=model, settings=cfg),
        critic=Critic(model=model, settings=cfg),
        writer=Writer(model=model, settings=cfg),
        settings=cfg,
        planner=heuristic_planner,
        tools=DEFAULT_REGISTRY,
        corpus=CORPUS,
    )


def make_client(make_stub, critic_decisions, store):
    app = create_app(agents=build_stub_agents(make_stub, critic_decisions), trace_store=store)
    return TestClient(app)


def _parse_sse(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def test_healthz(make_stub):
    client = make_client(make_stub, [CriticDecision.passed], TraceStore(":memory:"))
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_viewer_served_at_root(make_stub):
    client = make_client(make_stub, [CriticDecision.passed], TraceStore(":memory:"))
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # the real viewer is served (not the fallback), and drives the SSE endpoints
    assert "<title>Maestro" in r.text
    assert 'id="score"' in r.text and "/run" in r.text


def test_run_streams_coordination(make_stub):
    store = TraceStore(":memory:")
    client = make_client(make_stub, [CriticDecision.rejected, CriticDecision.passed], store)
    r = client.post("/run", json={"goal": "Compare solar and wind", "thread_id": "t1"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    types = [e["type"] for e in events]

    assert types[0] == "run_started"
    assert "plan_produced" in types
    assert "critic_reject" in types and "critic_pass" in types  # the loop streamed
    final = events[-1]
    assert final["type"] == "final"
    assert final["status"] == "completed"
    assert final["answer"]["content"] == "FINAL BRIEF"
    assert final["critic_iterations"] == 2


def test_plan_streams_with_descriptions_and_dependencies(make_stub):
    # The viewer shows what each subtask is for and what it waits on while the run is
    # still going, so the plan has to arrive with the plan_produced frame.
    client = make_client(make_stub, [CriticDecision.passed], TraceStore(":memory:"))
    events = _parse_sse(client.post("/run", json={"goal": "Compare solar and wind"}).text)
    plan = next(e for e in events if e["type"] == "plan_produced")["plan"]
    assert plan and all(p["id"] and p["role"] and p["description"] for p in plan)
    ids = {p["id"] for p in plan}
    assert all(set(p["depends_on"]) <= ids for p in plan)
    assert any(p["depends_on"] for p in plan)  # later stages wait on earlier ones
    # other frames carry no plan, and the final frame repeats it with statuses
    assert all(e.get("plan") is None for e in events if e["type"] not in ("plan_produced", "final"))
    final = events[-1]
    assert {s["id"] for s in final["subtasks"]} == ids
    assert all("description" in s and "depends_on" in s for s in final["subtasks"])


def test_follow_up_recalls_memory_after_a_restart(make_stub, tmp_path):
    # A follow-up draws on its thread's findings. Those must survive a server restart,
    # or every follow-up on an earlier run silently starts from nothing.
    def topical(responder):
        def wrapped(schema, messages):
            if getattr(schema, "__name__", "") == "AnalysisModel":
                return AnalysisModel(content="analysis", claims=[
                    "wind has the higher capacity factor for the grid",
                    "solar output peaks midday and needs storage"])
            return responder(schema, messages)
        return make_stub(wrapped)

    store, mem_dir = TraceStore(":memory:"), str(tmp_path / "memory")

    def client():
        agents = build_stub_agents(topical, [CriticDecision.passed])
        return TestClient(create_app(agents=agents, trace_store=store, memory_dir=mem_dir))

    first = _parse_sse(client().post("/run", json={"goal": "Compare solar and wind", "thread_id": "t1"}).text)
    wrote = next(e for e in first if e["type"] == "memory_write")
    assert len(wrote["stored"]) == 2  # the stream says what was saved, not just how many

    restarted = client()  # a fresh app over the same memory directory
    thread = restarted.get("/threads/t1").json()
    assert thread["memory_findings"] == 2
    assert [r["goal"] for r in thread["runs"]] == ["Compare solar and wind"]

    goal = 'Follow-up to "Compare solar and wind": how much storage does solar need?'
    second = _parse_sse(restarted.post("/run", json={"goal": goal, "thread_id": "t1"}).text)
    recalled = next(e for e in second if e["type"] == "memory_recall")["recalled"]
    assert any("solar" in r for r in recalled)
    runs = restarted.get("/threads/t1").json()["runs"]
    assert [r["goal"] for r in runs] == ["Compare solar and wind", goal]  # conversation order
    assert restarted.get("/threads/nobody").json() == {"thread_id": "nobody", "runs": [], "memory_findings": 0}


def test_models_endpoint_and_stub_ignores_switch(make_stub):
    client = make_client(make_stub, [CriticDecision.passed], TraceStore(":memory:"))
    m = client.get("/models").json()
    assert "default" in m and "fallback" in m
    assert m["switchable"] is False  # injected stub agents are not model-switchable
    # a model override is accepted but harmless with stub agents (uses the default graph)
    r = client.post("/run", json={"goal": "Compare solar and wind", "model": m["fallback"]})
    assert r.status_code == 200
    final = _parse_sse(r.text)[-1]
    assert final["status"] == "completed" and final["model"] == m["default"]


def test_api_key_is_never_echoed(make_stub):
    # bring-your-own-key must be used transiently and never reflected back to the client.
    client = make_client(make_stub, [CriticDecision.passed], TraceStore(":memory:"))
    m = client.get("/models").json()
    assert "has_server_key" in m
    secret = "AIzaSY-TOTALLY-FAKE-SECRET-KEY-123"
    r = client.post("/run", json={"goal": "Compare solar and wind", "api_key": secret})
    assert r.status_code == 200
    assert secret not in r.text  # the key never appears in the stream


def test_run_degrades_gracefully_on_terminal_failure(make_stub):
    # A terminal planner/LLM failure (e.g. free-tier quota exhausted) must produce a
    # clean error frame, not abort the chunked stream (what caused ERR_INCOMPLETE_CHUNKED_ENCODING).
    store = TraceStore(":memory:")
    agents = build_stub_agents(make_stub, [CriticDecision.passed])

    def boom(goal, roles, max_subtasks):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    agents.planner = boom
    client = TestClient(create_app(agents=agents, trace_store=store))

    r = client.post("/run", json={"goal": "Compare A and B"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    types = [e["type"] for e in events]
    assert "error" in types  # a clean error event was streamed
    final = events[-1]
    assert final["type"] == "final" and final["status"] == "error"
    assert final["error"] and "quota" in final["error"].lower()
    assert client.get("/runs").json()["runs"] == []  # a failed run is not persisted as completed


def test_run_is_persisted_and_replayable(make_stub):
    store = TraceStore(":memory:")
    client = make_client(make_stub, [CriticDecision.passed], store)
    r = client.post("/run", json={"goal": "Compare A and B"})
    final = _parse_sse(r.text)[-1]
    run_id = final["run_id"]

    runs = client.get("/runs").json()["runs"]
    assert any(x["run_id"] == run_id for x in runs)

    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["run_id"] == run_id and len(body["events"]) > 0

    assert client.get("/runs/does-not-exist").status_code == 404
