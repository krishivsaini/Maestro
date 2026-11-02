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
import re
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .config import get_settings
from .graph import MaestroAgents, apply_plan_edits, build_default_agents, build_graph
from .guard import FollowUpGuard, ThreadContext, split_follow_up
from .limits import RunLimits
from .usage import UsageMeter
from .logging_config import get_logger
from .db import Database
from .memory import SqlMemory
from .resilience import is_rate_limit_error
from .state import _utcnow, new_state
from .trace import TraceStore

log = get_logger("serve")

VIEWER_PATH = Path(__file__).resolve().parent.parent / "viewer" / "index.html"


MAX_GOAL_CHARS = 500


class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=MAX_GOAL_CHARS + 200)  # a follow-up carries its opening question
    thread_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_.:-]{1,64}$")
    model: Optional[str] = Field(None, max_length=80)  # None -> server default; else the primary/fallback model id
    api_key: Optional[str] = Field(None, max_length=200)  # bring-your-own-key: used transiently, never stored/logged
    review_plan: bool = False  # pause after planning for the visitor to approve or edit the plan

    @field_validator("goal")
    @classmethod
    def _goal(cls, v: str) -> str:
        v = " ".join(v.split())
        from .guard import split_follow_up

        asked = split_follow_up(v)[1]
        if len(asked) > MAX_GOAL_CHARS:
            raise ValueError(f"a question can be at most {MAX_GOAL_CHARS} characters")
        if len(asked) < 3:
            raise ValueError("ask a question of at least a few words")
        return v


RUN_ID = re.compile(r"^run_[0-9a-f]{8}$")


class PlanEdit(BaseModel):
    id: str = Field(max_length=64)
    description: Optional[str] = Field(None, min_length=3, max_length=300)
    queries: Optional[list[str]] = Field(None, max_length=3)
    remove: bool = False


class PlanAdd(BaseModel):
    description: str = Field(min_length=3, max_length=300)
    queries: list[str] = Field(default_factory=list, max_length=3)


class PlanEdits(BaseModel):
    subtasks: list[PlanEdit] = Field(default_factory=list, max_length=8)
    add: list[PlanAdd] = Field(default_factory=list, max_length=2)


class ResumeRequest(BaseModel):
    model: Optional[str] = Field(None, max_length=80)
    api_key: Optional[str] = Field(None, max_length=200)
    plan: Optional[PlanEdits] = None  # edits to a plan paused for review


# What the visitor sees when a request is invalid, by field.
FIELD_ERRORS = {
    "goal": f"Questions can be up to {MAX_GOAL_CHARS} characters. Shorten it and try again.",
    "thread_id": "That thread id isn't valid: use letters, numbers, - and _, up to 64 characters.",
    "model": "That model isn't available.",
    "api_key": "That doesn't look like a Gemini API key.",
}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _frame(e) -> str:
    return _sse({
        "type": e.event_type.value,
        "agent": e.agent,
        "summary": e.summary,
        "subtask_id": e.subtask_id,
        "critic_verdict": e.critic_verdict,
        "recovery_decision": e.recovery_decision,
        "rate_limit_retry": e.rate_limit_retry,
        "timestamp": e.timestamp,
        "plan": e.data.get("plan"),
        # what memory actually held, so a follow-up can show what it drew on
        "recalled": e.data.get("recalled"),
        "stored": e.data.get("stored"),
        "review": e.data.get("review"),  # what a critic review found, and why
        "edits": e.data.get("edits"),    # a reviewer's changes to the plan
    })


def _plan(subtasks) -> list[dict]:
    return [{"id": s.id, "role": s.role.value, "status": s.status.value, "description": s.description,
             "depends_on": list(s.depends_on), "queries": list(s.queries)} for s in subtasks]


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



def make_checkpointer(store: TraceStore, settings, *, injected: bool):
    """Where a run's state is saved after every node, so it can pause for plan review or
    resume after a failure: the runs database (Postgres or SQLite file), or memory for an
    injected in-memory store (tests)."""
    db = store.db
    if db.dialect == "postgres":
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row

        saver = PostgresSaver(psycopg.connect(db.target, autocommit=True, row_factory=dict_row,
                                              prepare_threshold=None, connect_timeout=10))
        saver.setup()
        return saver
    if injected or db.target == ":memory:":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(sqlite3.connect(db.target, check_same_thread=False))


def memory_embedder(settings):
    """The embedder memory recalls with: by meaning via Gemini when configured and a key
    exists, else None (words only)."""
    if settings.memory_embedder == "gemini" and settings.google_api_key:
        from .memory.embedders import GeminiEmbedder

        return GeminiEmbedder(settings)
    return None


def import_file_memory(directory: Optional[str], memory) -> None:
    """Bring findings saved by the earlier file-based memory into the database, once."""
    import os

    meta = os.path.join(directory or "", "meta.json")
    if not directory or not os.path.exists(meta) or memory.db.query("SELECT 1 AS x FROM memory LIMIT 1"):
        return
    try:
        rows = json.load(open(meta)).get("meta", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", meta, exc)
        return
    by_thread: dict[str, list[str]] = {}
    for r in rows:
        by_thread.setdefault(r["thread_id"], []).append(r["content"])
    for thread_id, contents in by_thread.items():
        memory.add_many(thread_id, contents)
    log.info("imported %d findings from %s into the database", len(rows), directory)


def create_app(
    *,
    agents: Optional[MaestroAgents] = None,
    memory: object = None,
    trace_store: Optional[TraceStore] = None,
    followup_classifier=None,
) -> FastAPI:
    """Build the FastAPI app. Inject stub ``agents`` / an in-memory store for tests.

    Runs and long-term memory live in one database: SQLite locally, Postgres when
    ``DATABASE_URL`` is set (so both survive a restart on a host that wipes its disk).
    Memory recalls by meaning (Gemini embeddings) and falls back to words; injected
    stub agents use words only, so tests make no API calls.

    A follow-up (a run in a thread that already has runs) is checked against that
    thread's research first; see maestro/guard.py. Stub agents have no model to check
    with, so their guard is off unless a ``followup_classifier`` is injected.
    """
    settings = get_settings()
    injected = agents is not None  # stub agents (tests) are not model-switchable
    if agents is None:
        agents = build_default_agents(settings)
    db, storage = None, "sqlite"
    if settings.database_url and trace_store is None:
        try:
            db = Database(settings.database_url)
            storage = db.dialect
            log.info("storing runs and memory in %s", db.dialect)
        except Exception as exc:  # a bad URL or unreachable host must not take the demo down
            storage = "sqlite (database unreachable)"
            log.error("DATABASE_URL is set but the database can't be reached (%s: %s); falling back to "
                      "local storage, which a restart wipes", type(exc).__name__, str(exc).split("\n")[0][:200])
    store = trace_store if trace_store is not None else TraceStore(db or settings.trace_db_path)
    checkpointer = make_checkpointer(store, settings, injected=trace_store is not None)
    if memory is not None:
        agents.memory = memory
    elif agents.memory is None:
        agents.memory = SqlMemory(store.db, None if injected else memory_embedder(settings))
        if not injected:
            import_file_memory(settings.memory_store_dir, agents.memory)

    # A run can pick the primary or the higher-throughput alternative from the UI
    # (e.g. when the primary hits its free-tier daily quota). Every graph shares the
    # one long-term memory.
    allowed_models = {settings.model_id, settings.fallback_model_id}

    def graph_for(model_id: Optional[str], api_key: Optional[str] = None, *, event_sink=None, meter=None,
                  review_plan: bool = False):
        """Build this request's graph, wired to its own live event sink.

        Built per request rather than cached: the sink is per-run, and compiling the
        graph measures ~5ms, which is far below the cost of a single model call. A
        bring-your-own key is used transiently here and never stored or logged.
        """
        if injected:  # stub agents (tests) don't use a real model/key
            return build_graph(agents=agents, settings=settings, event_sink=event_sink, meter=meter,
                               checkpointer=checkpointer, review_plan=review_plan), settings.model_id
        target = model_id if model_id in allowed_models else settings.model_id
        update = {"model_id": target}
        if api_key:
            update["google_api_key"] = api_key
        cfg = settings.model_copy(update=update)
        alt = build_default_agents(cfg)
        alt.memory = agents.memory  # every graph shares the one long-term memory
        return build_graph(agents=alt, settings=cfg, event_sink=event_sink, meter=meter,
                           checkpointer=checkpointer, review_plan=review_plan), target

    def thread_context(thread_id: Optional[str]) -> Optional[ThreadContext]:
        """The research a follow-up in this thread has to stay within; None for a new thread."""
        if not thread_id:
            return None
        runs = store.list_runs(thread_id=thread_id)
        if not runs:
            return None
        root, latest = store.get_run(runs[0]["run_id"]), store.get_run(runs[-1]["run_id"])
        question = split_follow_up(root.goal)[0] or root.goal if root else runs[0]["goal"]
        findings = getattr(agents.memory, "findings", None)
        brief = latest.final_output.content[:1500] if latest and latest.final_output else ""
        return ThreadContext(question=question, plan=[s.description for s in (root.subtasks if root else [])],
                             findings=findings(thread_id) if findings else [], brief=brief)

    def guard_for(api_key: Optional[str], meter=None) -> FollowUpGuard:
        if injected and followup_classifier is None:
            return FollowUpGuard(settings, enabled=False)
        cfg = settings.model_copy(update={"google_api_key": api_key}) if api_key else settings
        return FollowUpGuard(cfg, classifier=followup_classifier, enabled=settings.followup_guard, meter=meter)

    limits = RunLimits(settings)

    def client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if settings.trust_proxy and fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    app = FastAPI(title="Maestro", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [str(e["loc"][-1]) for e in exc.errors() if e.get("loc")]
        field = next((f for f in fields if f in FIELD_ERRORS), None)
        return JSONResponse({"error": FIELD_ERRORS.get(field, "That request isn't valid."), "field": field},
                            status_code=422)

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
        # which storage is live, so a deploy can be checked without reading its logs
        return {"status": "ok", "storage": storage}

    def run_config(run_id: str) -> dict:
        # A run's checkpoints are keyed by its run id (the memory thread is separate).
        return {"configurable": {"thread_id": run_id}, "recursion_limit": settings.max_steps + 10}

    def stream_run(graph, live, model_id, config, start, meter, *, run_id, thread_id, seen):
        """Drive a run (fresh, or resumed from its checkpoint) and stream it as SSE.

        The graph only yields between nodes, so a slow node (planning, or a call climbing
        the backoff ladder) would leave the viewer silent for a minute with no way to tell
        waiting from hung. Run the graph on its own thread and forward the events it
        publishes as they happen. When it stops, its saved state says why: finished,
        paused for plan review, or failed and resumable.
        """
        done = object()
        outcome: dict = {"exc": None}

        def work() -> None:
            try:
                for _ in graph.stream(start, config=config, stream_mode="values"):
                    pass
            except Exception as exc:  # noqa: BLE001 — surfaced as an SSE frame below
                outcome["exc"] = exc
            finally:
                live.put(done)

        worker = threading.Thread(target=work, name=f"run-{run_id}", daemon=True)
        worker.start()
        while True:
            item = live.get()
            if item is done:
                break
            uid = item.data.get("uid")
            if uid:
                seen.add(uid)
            yield _frame(item)
        worker.join()

        snap = graph.get_state(config)
        last = snap.values or {}
        # Anything recorded in the trace but never published live still belongs in the stream.
        for e in last.get("trace", []):
            if e.data.get("uid") not in seen:
                seen.add(e.data.get("uid"))
                yield _frame(e)

        if outcome["exc"] is None and snap.next:
            # paused before research: the plan waits for a person to approve or edit it
            log.info("run %s PAUSED for plan review", run_id)
            yield _sse({"type": "paused", "run_id": run_id, "thread_id": thread_id, "next": list(snap.next),
                        "plan": _plan(last.get("subtasks", [])), "usage": meter.summary(settings)})
            return

        error: Optional[str] = None
        if outcome["exc"] is not None:
            # A terminal planner/LLM failure degrades to a clean SSE frame rather than
            # aborting the chunked stream; the checkpoint keeps what was done, so the run
            # can resume from its last finished step.
            exc = outcome["exc"]
            error = _friendly_error(exc)
            log.warning("run %s FAILED mid-stream (resumable at %s): %s", run_id, list(snap.next), exc)
            yield _sse({
                "type": "error", "agent": "supervisor", "summary": error,
                "subtask_id": None, "critic_verdict": None,
                "recovery_decision": None, "timestamp": _utcnow(),
            })

        # Save before the final frame: a viewer reacts to it by fetching the thread,
        # and must find this run (and its findings) already there.
        if error is None:  # only persist genuinely completed runs
            try:
                store.save_run({**last, "usage": meter.summary(settings)})
            except Exception as exc:  # persistence failure must not break the response
                log.warning("failed to persist run %s: %s", run_id, exc)
            forget_checkpoints(run_id)
        ans = last.get("final_output")
        yield _sse(
            {
                "type": "final",
                "run_id": run_id,
                "model": model_id,
                "status": "error" if error else last.get("status"),
                "answer": ans.model_dump() if ans else None,
                "error": error,
                "resumable": bool(error and snap.next),
                "subtasks": _plan(last.get("subtasks", [])),
                "critic_iterations": last.get("critic_iterations", 0),
                "recovery_attempts": last.get("recovery_attempts", 0),
                "usage": meter.summary(settings),
            }
        )
        log.info(
            "run %s DONE | status=%s | steps=%s | critic_iters=%s | recoveries=%s",
            run_id, "error" if error else last.get("status"), last.get("step_count"),
            last.get("critic_iterations"), last.get("recovery_attempts"),
        )

    def forget_checkpoints(run_id: str) -> None:
        """A finished run is replayed from the trace store; its checkpoints are no longer needed."""
        try:
            checkpointer.delete_thread(run_id)
        except Exception as exc:  # noqa: BLE001 — leftover checkpoints are harmless
            log.warning("could not delete checkpoints for %s: %s", run_id, exc)

    def refused(request: Request, own_key: bool) -> Optional[JSONResponse]:
        refusal = limits.check(client_ip(request), own_key=own_key)
        if not refusal:
            return None
        log.info("run refused (%s limit) for %s", refusal.reason, client_ip(request))
        return JSONResponse({"error": refusal.message, "reason": refusal.reason,
                             "retry_after": refusal.retry_after},
                            status_code=429, headers={"Retry-After": str(refusal.retry_after)})

    @app.post("/run")
    def run(req: RunRequest, request: Request) -> Response:
        # Refused before any model call: one visitor can't spend the shared quota.
        no = refused(request, bool(req.api_key))
        if no:
            return no
        state = new_state(req.goal, thread_id=req.thread_id)
        run_id, thread_id = state["run_id"], state["thread_id"]
        # NB: never log req.api_key — only the goal, thread and model id are logged.
        log.info("run %s START | thread=%s | model=%s | byok=%s | review=%s | goal=%r",
                 run_id, thread_id, req.model or settings.model_id, bool(req.api_key), req.review_plan, req.goal)

        meter = UsageMeter()  # every model and embedding call this request makes

        def stream():
            # A follow-up must be about its thread's research. Checked before any agent
            # runs, so an off-topic question costs at most one small call, not a run.
            ctx = thread_context(req.thread_id)
            if ctx is not None:
                verdict = guard_for(req.api_key, meter).check(split_follow_up(req.goal)[1], ctx)
                log.info("run %s follow-up guard: %s (%s) %s", run_id,
                         "allowed" if verdict.allowed else "BLOCKED", verdict.method, verdict.reason)
                if not verdict.allowed:
                    yield _sse({"type": "blocked", "reason": verdict.reason, "usage": meter.summary(settings),
                                "question": split_follow_up(req.goal)[1], "thread_id": thread_id})
                    return

            live: "queue.Queue" = queue.Queue()
            graph, model_id = graph_for(req.model, req.api_key, event_sink=live.put, meter=meter,
                                        review_plan=req.review_plan)
            yield _sse({"type": "run_started", "agent": "supervisor", "model": model_id,
                        "run_id": run_id, "thread_id": thread_id, "goal": req.goal})
            yield from stream_run(graph, live, model_id, run_config(run_id), state, meter,
                                  run_id=run_id, thread_id=thread_id, seen=set())

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/runs/{run_id}/resume")
    def resume(run_id: str, req: ResumeRequest, request: Request) -> Response:
        """Continue a run from its checkpoint: after plan review (with any edits), or after
        a failure (from its last finished step)."""
        if not RUN_ID.match(run_id):
            return JSONResponse({"error": "There's no run with that id."}, status_code=404)
        no = refused(request, bool(req.api_key))
        if no:
            return no
        meter = UsageMeter()
        live: "queue.Queue" = queue.Queue()
        graph, model_id = graph_for(req.model, req.api_key, event_sink=live.put, meter=meter)
        config = run_config(run_id)
        snap = graph.get_state(config)
        if not snap.values:
            return JSONResponse({"error": "That run can't be resumed: nothing was saved for it."}, status_code=404)
        if not snap.next:
            return JSONResponse({"error": "That run has already finished."}, status_code=409)
        approval = None
        if snap.next == ("research",):
            try:
                changed, approval = apply_plan_edits(snap.values["subtasks"],
                                                     req.plan.model_dump() if req.plan else {}, settings.max_subtasks)
            except ValueError as exc:
                return JSONResponse({"error": f"That plan can't run: {exc}."}, status_code=422)
            graph.update_state(config, {"subtasks": changed, "trace": [approval]})
        elif req.plan is not None:
            return JSONResponse({"error": "That run is past planning, so its plan can't change now."}, status_code=409)
        thread_id = snap.values.get("thread_id")
        seen = {e.data.get("uid") for e in snap.values.get("trace", [])}
        log.info("run %s RESUME at %s | edits=%s", run_id, list(snap.next), approval.data["edits"] if approval else None)

        def stream():
            yield _sse({"type": "resumed", "agent": "supervisor", "run_id": run_id, "thread_id": thread_id,
                        "model": model_id, "from": snap.next[0], "goal": snap.values.get("goal")})
            if approval is not None:
                seen.add(approval.data["uid"])
                yield _frame(approval)
            yield from stream_run(graph, live, model_id, config, None, meter,
                                  run_id=run_id, thread_id=thread_id, seen=seen)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/runs")
    def list_runs() -> dict:
        return {"runs": store.list_runs()}

    @app.get("/threads/{thread_id}")
    def get_thread(thread_id: str) -> dict:
        """A thread's runs in conversation order, and how much memory a follow-up has."""
        count = getattr(agents.memory, "count", None)
        return {"thread_id": thread_id, "runs": store.list_runs(thread_id=thread_id),
                "memory_findings": count(thread_id) if count else 0}

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
