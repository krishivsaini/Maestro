"""The API service — FastAPI streaming supervisor delegations + subagent progress.

``POST /run`` streams (SSE) the coordination as it happens: plan produced ->
subtasks dispatched -> each subagent's result -> critic verdicts -> recovery ->
final output. ``thread_id`` groups turns and long-term memory. Every completed run
is persisted to the trace store and reconstructable via ``GET /runs/{run_id}``.

Streaming is what makes a 90-second demo legible instead of opaque — watching the
supervisor delegate and the critic reject is the whole pitch.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
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

    # A run can pick the primary or the higher-throughput alternative from the UI
    # (e.g. when the primary hits its free-tier daily quota). Every graph shares the
    # one long-term memory.
    allowed_models = {settings.model_id, settings.fallback_model_id}

    def graph_for(model_id: Optional[str], api_key: Optional[str] = None, *, event_sink=None):
        """Build this request's graph, wired to its own live event sink.

        Built per request rather than cached: the sink is per-run, and compiling the
        graph measures ~5ms, which is far below the cost of a single model call. A
        bring-your-own key is used transiently here and never stored or logged.
        """
        if injected:  # stub agents (tests) don't use a real model/key
            return build_graph(agents=agents, settings=settings, event_sink=event_sink), settings.model_id
        target = model_id if model_id in allowed_models else settings.model_id
        update = {"model_id": target}
        if api_key:
            update["google_api_key"] = api_key
        cfg = settings.model_copy(update=update)
        alt = build_default_agents(cfg)
        alt.memory = agents.memory  # every graph shares the one long-term memory
        return build_graph(agents=alt, settings=cfg, event_sink=event_sink), target

    app = FastAPI(title="Maestro", version="0.1.0")

    # Only enabled when the viewer is hosted on a separate origin (see
    # Settings.cors_origins). Same-origin — FastAPI serving the viewer at "/" —
    # needs no CORS at all, so the default adds no headers and no OPTIONS handling.
    origins = settings.cors_origin_list()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )
        log.info("CORS enabled for %s", ", ".join(origins))

    @app.get("/config.js")
    def config_js() -> Response:
        """Empty stub so the viewer's ``<script src="config.js">`` resolves same-origin.

        A static deployment ships its own ``config.js`` setting ``window.MAESTRO_API``
        to this service's origin; served from here the viewer is already same-origin,
        so the value stays empty.
        """
        return Response("window.MAESTRO_API = window.MAESTRO_API || '';",
                        media_type="application/javascript")

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
        # NB: never log req.api_key — only the goal, thread and model id are logged.
        log.info("run %s START | thread=%s | model=%s | byok=%s | goal=%r",
                 run_id, thread_id, req.model or settings.model_id, bool(req.api_key), req.goal)

        def stream():
            # The graph only yields between nodes, so a slow node (planning, or a call
            # climbing the backoff ladder) would leave the viewer silent for a minute
            # with no way to tell waiting from hung. Run the graph on its own thread
            # and forward the events it publishes as they happen.
            live: "queue.Queue" = queue.Queue()
            graph, model_id = graph_for(req.model, req.api_key, event_sink=live.put)

            yield _sse({"type": "run_started", "agent": "supervisor", "model": model_id,
                        "run_id": run_id, "thread_id": thread_id, "goal": req.goal})

            done = object()
            outcome: dict = {"last": state, "exc": None}

            def work() -> None:
                try:
                    for snap in graph.stream(
                        state, config={"recursion_limit": settings.max_steps + 10},
                        stream_mode="values",
                    ):
                        outcome["last"] = snap
                except Exception as exc:  # noqa: BLE001 — surfaced as an SSE frame below
                    outcome["exc"] = exc
                finally:
                    live.put(done)

            worker = threading.Thread(target=work, name=f"run-{run_id}", daemon=True)
            worker.start()

            seen: set[str] = set()

            def frame(e) -> str:
                return _sse({
                    "type": e.event_type.value,
                    "agent": e.agent,
                    "summary": e.summary,
                    "subtask_id": e.subtask_id,
                    "critic_verdict": e.critic_verdict,
                    "recovery_decision": e.recovery_decision,
                    "timestamp": e.timestamp,
                })

            while True:
                item = live.get()
                if item is done:
                    break
                uid = item.data.get("uid")
                if uid:
                    seen.add(uid)
                yield frame(item)

            worker.join()
            last = outcome["last"]

            error: Optional[str] = None
            if outcome["exc"] is not None:
                # A terminal planner/LLM failure degrades to a clean SSE frame rather
                # than aborting the chunked stream (the graph's recover/degrade ladder
                # only covers researcher subagents, not planning).
                exc = outcome["exc"]
                error = _friendly_error(exc)
                log.warning("run %s FAILED mid-stream: %s", run_id, exc)
                yield _sse({
                    "type": "error", "agent": "supervisor", "summary": error,
                    "subtask_id": None, "critic_verdict": None,
                    "recovery_decision": None, "timestamp": _utcnow(),
                })

            # Anything recorded in the trace but never published live (or dropped
            # because a node errored before returning) still belongs in the stream.
            for e in last.get("trace", []):
                if e.data.get("uid") not in seen:
                    yield frame(e)

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
