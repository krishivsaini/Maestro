"""One storage seam over SQLite or Postgres.

Locally, runs and memory live in SQLite and files. The deployed demo's disk is wiped
on every restart, which loses every past run and every thread's memory, so with
``DATABASE_URL`` set (a free hosted Postgres such as Neon or Supabase) the same code
stores them there instead. Statements are written once, with ``?`` placeholders;
this translates them for Postgres.

A hosted serverless Postgres closes idle connections, so a statement that finds its
connection gone reconnects and runs once more. Prepared statements are off: poolers
such as Supabase's transaction pooler (port 6543) reject them, and at this volume
they buy nothing.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Optional

from .logging_config import get_logger

log = get_logger("db")


def is_postgres(url: str) -> bool:
    return url.startswith(("postgres://", "postgresql://"))


class Database:
    def __init__(self, target: str = ":memory:") -> None:
        self.target = target
        self.dialect = "postgres" if is_postgres(target) else "sqlite"
        self._lock = threading.RLock()
        self._tx_depth = 0
        self.conn = self._connect()

    def _connect(self):
        if self.dialect == "postgres":
            import psycopg
            from psycopg.rows import dict_row

            return psycopg.connect(self.target, autocommit=True, row_factory=dict_row,
                                   prepare_threshold=None, connect_timeout=10)
        conn = sqlite3.connect(self.target, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.dialect == "postgres" else sql

    def _run(self, fn):
        """Run ``fn(conn)``; for Postgres, reconnect once if the connection was dropped."""
        with self._lock:
            try:
                return fn(self.conn)
            except Exception as exc:
                if self.dialect != "postgres" or self._tx_depth or not self._dropped(exc):
                    raise
                log.warning("database connection lost (%s); reconnecting", type(exc).__name__)
                self.conn = self._connect()
                return fn(self.conn)

    def _dropped(self, exc: Exception) -> bool:
        import psycopg

        return isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)) or bool(
            getattr(self.conn, "closed", False))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        def go(conn):
            cur = conn.execute(self._sql(sql), tuple(params))
            rows = cur.fetchall() if cur.description else []
            return [dict(r) for r in rows]
        return self._run(go)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        self._run(lambda conn: conn.execute(self._sql(sql), tuple(params)))
        if self.dialect == "sqlite" and not self._tx_depth:
            self.conn.commit()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        def go(conn):
            if self.dialect == "postgres":
                with conn.cursor() as cur:
                    cur.executemany(self._sql(sql), rows)
            else:
                conn.executemany(sql, rows)
        self._run(go)
        if self.dialect == "sqlite" and not self._tx_depth:
            self.conn.commit()

    def script(self, statements: str) -> None:
        """Several ``;``-separated statements (schema setup)."""
        for stmt in (s.strip() for s in statements.split(";")):
            if stmt:
                self.execute(stmt)

    @contextmanager
    def transaction(self):
        """All statements inside commit together, or not at all."""
        with self._lock:
            self._tx_depth += 1
            try:
                if self.dialect == "postgres":
                    with self.conn.transaction():
                        yield self
                else:
                    try:
                        yield self
                        self.conn.commit()
                    except BaseException:
                        self.conn.rollback()
                        raise
            finally:
                self._tx_depth -= 1

    def add_column(self, table: str, column: str, kind: str) -> None:
        """Add a column to an existing table if it isn't there yet (a schema upgrade)."""
        if self.dialect == "postgres":
            self.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {kind}")
        elif column not in {r["name"] for r in self.query(f"PRAGMA table_info({table})")}:
            self.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    def upsert(self, table: str, key: str, row: dict) -> None:
        cols = list(row)
        marks = ", ".join("?" for _ in cols)
        if self.dialect == "postgres":
            updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != key)
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({marks}) ON CONFLICT ({key}) DO UPDATE SET {updates}"
        else:
            sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({marks})"
        self.execute(sql, [row[c] for c in cols])

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def open_database(target: Optional[str]) -> Database:
    return Database(target or ":memory:")
