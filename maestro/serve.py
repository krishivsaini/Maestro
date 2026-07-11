"""The API service (§16) — FastAPI streaming supervisor delegations + subagent progress.

``POST /run`` streams (SSE) the coordination as it happens: plan produced ->
subtasks dispatched -> each subagent's result -> critic verdicts -> recovery ->
final output. ``thread_id`` groups turns and long-term memory. Every completed run
is persisted to the trace store and reconstructable via ``GET /runs/{run_id}``.

Streaming is what makes a 90-second demo legible instead of opaque — watching the
supervisor delegate and the critic reject is the whole pitch (§1.2.9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import get_settings
from .graph import MaestroAgents, build_default_agents, build_graph
from .logging_config import get_logger
from .memory import HashingEmbedder, LongTermMemory
from .resilience import is_rate_limit_error
from .state import _utcnow, new_state
from .trace import TraceStore

log = get_logger("serve")

VIEWER_PATH = Path(__file__).resolve().parent.parent / "viewer" / "index.html"


class RunRequest(BaseModel):
    goal: str
    thread_id: Optional[str] = None
    model: Optional[str] = None  # None -> server default; else the primary/fallback model id
    api_key: Optional[str] = None  # bring-your-own-key: used transiently, never stored/logged


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _friendly_error(exc: BaseException) -> str:
    """A human-readable one-liner for a terminal failure surfaced to the viewer."""
    if is_rate_limit_error(exc):
        return (
            "Gemini free-tier quota/rate limit hit and retries were exhausted. "
            "Switch to the fallback model or paste your own API key to run on your quota, "
            "or wait for the daily quota to reset."
        )
    s = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if "api key" in s or "api_key_invalid" in s or "unauthenticated" in s or code in (401, 403):
        return "The API key was rejected — paste a valid Gemini API key (aistudio.google.com/apikey), or clear it to use the server's key."
    return f"{type(exc).__name__}: {exc}"


def create_app(
    *,
    agents: Optional[MaestroAgents] = None,
    memory: object = None,
    trace_store: Optional[TraceStore] = None,
) -> FastAPI:
    """Build the FastAPI app. Inject stub ``agents`` / an in-memory store for tests."""
    settings = get_settings()
    injected = agents is not None  # stub agents (tests) are not model-switchable
    if agents is None:
        agents = build_default_agents(settings)
    # attach thread-scoped long-term memory (HashingEmbedder default -> no torch)
    if memory is not None:
        agents.memory = memory
    elif agents.memory is None:
        agents.memory = LongTermMemory(embedder=HashingEmbedder())
    store = trace_store if trace_store is not None else TraceStore(settings.trace_db_path)
    default_graph = build_graph(agents=agents, settings=settings)

    # A run can pick the primary or the higher-throughput fallback model from the UI
    # (e.g. when the primary hits its free-tier quota). Graphs are built per model on
    # first use and cached; every model shares the one long-term memory.
    allowed_models = {settings.model_id, settings.fallback_model_id}
    graph_cache = {settings.model_id: default_graph}

    def _alt_graph(cfg):
        alt = build_default_agents(cfg)
        alt.memory = agents.memory  # every graph shares the one long-term memory
        return build_graph(agents=alt, settings=cfg)

    def graph_for(model_id: Optional[str], api_key: Optional[str] = None):
        if injected:  # stub agents (tests) don't use a real model/key
            return default_graph, settings.model_id
        target = model_id if model_id in allowed_models else settings.model_id
        if api_key:
            # bring-your-own-key: build an ephemeral graph, never cached (the key is a
            # secret) and never logged. It vanishes when this request's graph is GC'd.
            cfg = settings.model_copy(update={"model_id": target, "google_api_key": api_key})
            return _alt_graph(cfg), target
        if target == settings.model_id:
            return default_graph, settings.model_id
        if target not in graph_cache:
            graph_cache[target] = _alt_graph(settings.model_copy(update={"model_id": target}))
            log.info("built graph for model %s", target)
        return graph_cache[target], target

    app = FastAPI(title="Maestro", version="0.1.0")

    @app.get("/models")
    def models() -> dict:
        return {"default": settings.model_id, "fallback": settings.fallback_model_id,
                "switchable": not injected, "has_server_key": bool(settings.google_api_key)}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Serve the minimal coordination viewer (same-origin, so no CORS needed)."""
        if VIEWER_PATH.exists():
            return HTMLResponse(VIEWER_PATH.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<h1>Maestro</h1><p>Viewer not found. API: POST /run, GET /runs, GET /healthz.</p>"
        )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/run")
    def run(req: RunRequest) -> StreamingResponse:
        state = new_state(req.goal, thread_id=req.thread_id)
        run_id, thread_id = state["run_id"], state["thread_id"]
        graph, model_id = graph_for(req.model, req.api_key)
        # NB: never log req.api_key — only the goal, thread and model id are logged.
        log.info("run %s START | thread=%s | model=%s | byok=%s | goal=%r",
                 run_id, thread_id, model_id, bool(req.api_key), req.goal)

        def stream():
            yield _sse({"type": "run_started", "agent": "supervisor", "model": model_id,
                        "run_id": run_id, "thread_id": thread_id, "goal": req.goal})
            emitted = 0
            last = state
            error: Optional[str] = None
            try:
                for snap in graph.stream(
                    state, config={"recursion_limit": settings.max_steps + 10}, stream_mode="values"
                ):
                    last = snap
                    trace = snap.get("trace", [])
                    for e in trace[emitted:]:
                        yield _sse(
                            {
                                "type": e.event_type.value,
                                "agent": e.agent,
                                "summary": e.summary,
                                "subtask_id": e.subtask_id,
                                "critic_verdict": e.critic_verdict,
                                "recovery_decision": e.recovery_decision,
                                "timestamp": e.timestamp,
                            }
                        )
                    emitted = len(trace)
            except Exception as exc:  # noqa: BLE001 — a terminal planner/LLM failure must
                # degrade to a clean SSE frame, not abort the chunked stream (the graph's
                # recover/degrade ladder only covers researcher subagents, not planning).
                error = _friendly_error(exc)
                log.warning("run %s FAILED mid-stream: %s", run_id, exc)
                yield _sse({
                    "type": "error", "agent": "supervisor", "summary": error,
                    "subtask_id": None, "critic_verdict": None,
                    "recovery_decision": None, "timestamp": _utcnow(),
                })

            ans = last.get("final_output")
            yield _sse(
                {
                    "type": "final",
                    "run_id": run_id,
                    "model": model_id,
                    "status": "error" if error else last.get("status"),
                    "answer": ans.model_dump() if ans else None,
                    "error": error,
                    "subtasks": [
                        {"id": s.id, "role": s.role.value, "status": s.status.value}
                        for s in last.get("subtasks", [])
                    ],
                    "critic_iterations": last.get("critic_iterations", 0),
                    "recovery_attempts": last.get("recovery_attempts", 0),
                }
            )
            if error is None:  # only persist genuinely completed runs
                try:
                    store.save_run(last)
                except Exception as exc:  # persistence failure must not break the response
                    log.warning("failed to persist run %s: %s", run_id, exc)
            log.info(
                "run %s DONE | status=%s | steps=%s | critic_iters=%s | recoveries=%s",
                run_id, "error" if error else last.get("status"), last.get("step_count"),
                last.get("critic_iterations"), last.get("recovery_attempts"),
            )

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/runs")
    def list_runs() -> dict:
        return {"runs": store.list_runs()}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        data = store.export_json(run_id)
        if data is None:
            return JSONResponse({"error": "run not found"}, status_code=404)
        return data

    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    log.info("starting Maestro on %s:%s (model=%s)", settings.host, settings.port, settings.model_id)
    uvicorn.run(create_app(), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
