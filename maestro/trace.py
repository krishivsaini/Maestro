"""Observability — replayable runs (§17).

Every meaningful event is already appended to ``state["trace"]`` by the graph
nodes. This module persists a completed run (SQLite, keyed by run id) so it can be
**replayed node-by-node afterward** — which subtasks ran in parallel, where the
critic pushed back, where recovery happened — and exported as JSON. It is the
production-thinking signal: a full audit log you can reconstruct after the fact.

Kept decoupled from the graph: the service/CLI calls ``save_run(final_state)``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterator, Optional

from pydantic import BaseModel

from .logging_config import get_logger
from .state import Answer, Subtask, TraceEvent

log = get_logger("trace")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecord(BaseModel):
    """A persisted, replayable record of one run."""

    run_id: str
    thread_id: str
    goal: str
    status: str
    step_count: int = 0
    critic_iterations: int = 0
    recovery_attempts: int = 0
    created_at: str
    final_output: Optional[Answer] = None
    subtasks: list[Subtask] = []
    events: list[TraceEvent] = []


def record_from_state(state: dict) -> RunRecord:
    events = list(state.get("trace", []))
    created = events[0].timestamp if events else _utcnow()
    return RunRecord(
        run_id=state.get("run_id", ""),
        thread_id=state.get("thread_id", ""),
        goal=state.get("goal", ""),
        status=state.get("status", ""),
        step_count=state.get("step_count", 0),
        critic_iterations=state.get("critic_iterations", 0),
        recovery_attempts=state.get("recovery_attempts", 0),
        created_at=created,
        final_output=state.get("final_output"),
        subtasks=list(state.get("subtasks", [])),
        events=events,
    )


class TraceStore:
    """SQLite-backed store of runs + their events (both for full-fidelity replay
    and queryable audit)."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT,
                goal TEXT,
                status TEXT,
                created_at TEXT,
                event_count INTEGER,
                payload TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT,
                seq INTEGER,
                event_type TEXT,
                agent TEXT,
                timestamp TEXT,
                summary TEXT,
                subtask_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
            CREATE INDEX IF NOT EXISTS idx_runs_thread ON runs(thread_id);
            """
        )
        self.conn.commit()

    def save_run(self, state: dict) -> str:
        """Persist a completed run's full state + events. Returns the run id."""
        rec = record_from_state(state)
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, thread_id, goal, status, created_at, event_count, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rec.run_id, rec.thread_id, rec.goal, rec.status, rec.created_at, len(rec.events), rec.model_dump_json()),
        )
        self.conn.execute("DELETE FROM events WHERE run_id = ?", (rec.run_id,))
        self.conn.executemany(
            "INSERT INTO events (run_id, seq, event_type, agent, timestamp, summary, subtask_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (rec.run_id, i, e.event_type.value, e.agent, e.timestamp, e.summary, e.subtask_id)
                for i, e in enumerate(rec.events)
            ],
        )
        self.conn.commit()
        log.info("persisted run %s (%d events, status=%s)", rec.run_id, len(rec.events), rec.status)
        return rec.run_id

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        row = self.conn.execute("SELECT payload FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row["payload"]) if row else None

    def list_runs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT run_id, thread_id, goal, status, created_at, event_count FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def replay(self, run_id: str) -> Iterator[TraceEvent]:
        """Yield the run's events in order — the node-by-node replay."""
        rec = self.get_run(run_id)
        if rec:
            yield from rec.events

    def query_events(self, run_id: Optional[str] = None, event_type: Optional[str] = None) -> list[dict]:
        clauses, params = [], []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM events{where} ORDER BY run_id, seq", params).fetchall()
        return [dict(r) for r in rows]

    def export_json(self, run_id: str) -> Optional[dict]:
        rec = self.get_run(run_id)
        return rec.model_dump(mode="json") if rec else None

    def export_json_file(self, run_id: str, path: str) -> None:
        data = self.export_json(run_id)
        if data is None:
            raise KeyError(f"unknown run id: {run_id}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def close(self) -> None:
        self.conn.close()
