"""Day 12 — the API service (§16, FR-31..FR-33).

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
