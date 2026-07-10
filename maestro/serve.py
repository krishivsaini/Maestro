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
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import get_settings
from .graph import MaestroAgents, build_default_agents, build_graph
from .logging_config import get_logger
from .memory import HashingEmbedder, LongTermMemory
from .state import new_state
from .trace import TraceStore

log = get_logger("serve")


class RunRequest(BaseModel):
    goal: str
    thread_id: Optional[str] = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def create_app(
    *,
    agents: Optional[MaestroAgents] = None,
    memory: object = None,
    trace_store: Optional[TraceStore] = None,
) -> FastAPI:
    """Build the FastAPI app. Inject stub ``agents`` / an in-memory store for tests."""
    settings = get_settings()
    if agents is None:
        agents = build_default_agents(settings)
    # attach thread-scoped long-term memory (HashingEmbedder default -> no torch)
    if memory is not None:
        agents.memory = memory
    elif agents.memory is None:
        agents.memory = LongTermMemory(embedder=HashingEmbedder())
    store = trace_store if trace_store is not None else TraceStore(settings.trace_db_path)
    graph = build_graph(agents=agents, settings=settings)

    app = FastAPI(title="Maestro", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/run")
    def run(req: RunRequest) -> StreamingResponse:
        state = new_state(req.goal, thread_id=req.thread_id)
        run_id, thread_id = state["run_id"], state["thread_id"]
        log.info("run %s START | thread=%s | goal=%r", run_id, thread_id, req.goal)

        def stream():
            yield _sse({"type": "run_started", "run_id": run_id, "thread_id": thread_id, "goal": req.goal})
            emitted = 0
            last = state
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

            ans = last.get("final_output")
            yield _sse(
                {
                    "type": "final",
                    "run_id": run_id,
                    "status": last.get("status"),
                    "answer": ans.model_dump() if ans else None,
                    "subtasks": [
                        {"id": s.id, "role": s.role.value, "status": s.status.value}
                        for s in last.get("subtasks", [])
                    ],
                    "critic_iterations": last.get("critic_iterations", 0),
                    "recovery_attempts": last.get("recovery_attempts", 0),
                }
            )
            try:
                store.save_run(last)
            except Exception as exc:  # persistence failure must not break the response
                log.warning("failed to persist run %s: %s", run_id, exc)
            log.info(
                "run %s DONE | status=%s | steps=%s | critic_iters=%s | recoveries=%s",
                run_id, last.get("status"), last.get("step_count"),
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
