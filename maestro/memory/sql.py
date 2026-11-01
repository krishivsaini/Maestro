"""Long-term memory in the database, beside the runs, recalled by meaning.

Findings are rows in the same database as the runs (SQLite locally, Postgres when
``DATABASE_URL`` is set — see maestro/db.py), so a thread's memory survives a restart
wherever the runs do.

Each finding is stored with two vectors:

- a **meaning** vector from a semantic embedder (``GeminiEmbedder``), tagged with the
  model that made it, so vectors from different models are never compared;
- a **word** vector from ``HashingEmbedder``, which needs no API call.

Recall ranks a thread's findings by meaning when it can, and falls back to words when
the embedding API is unavailable or a finding predates meaning vectors — memory
degrades instead of failing the run. A thread holds a handful of findings, so ranking
happens in numpy: no vector extension, any Postgres will do.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..db import Database
from ..logging_config import get_logger
from ..state import MemoryItem
from .longterm import Embedder, HashingEmbedder

log = get_logger("memory.sql")


def _encode(vec: np.ndarray) -> str:
    """Sparse JSON ({index: value}): word vectors are mostly zeros."""
    return json.dumps({int(i): round(float(vec[i]), 6) for i in np.flatnonzero(vec)})


def _decode(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    for i, v in json.loads(text).items():
        vec[int(i)] = v
    return vec


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


class SqlMemory:
    def __init__(self, db: Database, embedder: Optional[Embedder] = None) -> None:
        self.db = db
        self.words = HashingEmbedder()
        # a semantic embedder, or None for word matching only
        self.meaning = embedder if embedder is not None and not isinstance(embedder, HashingEmbedder) else None
        self.tag = f"{getattr(self.meaning, 'name', 'semantic')}:{getattr(self.meaning, 'model', '')}:" \
                   f"{getattr(self.meaning, 'dim', '')}" if self.meaning else None
        self.db.script(
            """
            CREATE TABLE IF NOT EXISTS memory (
                thread_id TEXT,
                content TEXT,
                embedding TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_thread ON memory(thread_id)
            """
        )
        # added after the first deploy: upgrade existing tables in place
        self.db.add_column("memory", "semantic", "TEXT")
        self.db.add_column("memory", "semantic_tag", "TEXT")

    def _embed(self, fn, arg, meter):
        """Call the meaning embedder, recording the call on a run's meter."""
        t0 = time.perf_counter()
        try:
            out = fn(arg)
        except Exception:
            if meter is not None:
                meter.record("memory", getattr(self.meaning, "model", "embedder"), kind="embedding",
                             seconds=time.perf_counter() - t0, ok=False)
            raise
        if meter is not None:
            meter.record("memory", getattr(self.meaning, "model", "embedder"), kind="embedding",
                         seconds=time.perf_counter() - t0)
        return out

    def _meanings(self, texts: list[str], meter=None) -> list[Optional[np.ndarray]]:
        if not self.meaning or not texts:
            return [None] * len(texts)
        try:
            docs = getattr(self.meaning, "embed_documents", self.meaning.embed)
            return [_unit(v) for v in self._embed(docs, texts, meter)]
        except Exception as exc:  # noqa: BLE001 — recall falls back to words for these
            log.warning("could not embed %d findings by meaning (%s); storing word vectors only",
                        len(texts), type(exc).__name__)
            return [None] * len(texts)

    def add_many(self, thread_id: str, contents: list[str], meter=None) -> None:
        """Store a run's findings, embedding them by meaning in one call."""
        contents = [c for c in contents if c.strip()]
        meanings = self._meanings(contents, meter)
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for content, meaning in zip(contents, meanings):
            words = _unit(self.words.embed([content])[0])
            rows.append((thread_id, content, _encode(words), now,
                         json.dumps([round(float(x), 6) for x in meaning]) if meaning is not None else None,
                         self.tag if meaning is not None else None))
        if rows:
            self.db.executemany("INSERT INTO memory (thread_id, content, embedding, created_at, semantic, semantic_tag) "
                                "VALUES (?, ?, ?, ?, ?, ?)", rows)

    def add(self, thread_id: str, content: str) -> None:
        self.add_many(thread_id, [content])

    def query(self, thread_id: str, query: str, k: int = 3, meter=None) -> list[MemoryItem]:
        rows = self.db.query("SELECT content, embedding, semantic, semantic_tag FROM memory "
                             "WHERE thread_id = ? ORDER BY created_at", (thread_id,))
        if not rows:
            return []
        q_meaning = None
        if self.meaning and any(r["semantic_tag"] == self.tag for r in rows):
            try:
                ask = getattr(self.meaning, "embed_query", None)
                q_meaning = _unit(self._embed(ask, query, meter) if ask
                                  else self._embed(self.meaning.embed, [query], meter)[0])
            except Exception as exc:  # noqa: BLE001
                log.warning("could not embed the question by meaning (%s); recalling by words", type(exc).__name__)
        q_words = _unit(self.words.embed([query])[0])

        by_meaning, by_words = [], []
        for r in rows:
            if q_meaning is not None and r["semantic"] and r["semantic_tag"] == self.tag:
                by_meaning.append((float(np.dot(q_meaning, np.asarray(json.loads(r["semantic"]), "float32"))),
                                   r["content"]))
            else:
                score = float(np.dot(q_words, _decode(r["embedding"], q_words.shape[0])))
                if score > 0:  # no shared words: not a match
                    by_words.append((score, r["content"]))
        # meaning first; word matches only fill what meaning couldn't rank
        ranked = sorted(by_meaning, key=lambda s: -s[0]) + sorted(by_words, key=lambda s: -s[0])
        return [MemoryItem(thread_id=thread_id, content=c, score=s) for s, c in ranked[:k]]

    def count(self, thread_id: str) -> int:
        return int(self.db.query("SELECT COUNT(*) AS n FROM memory WHERE thread_id = ?", (thread_id,))[0]["n"])

    def findings(self, thread_id: str) -> list[str]:
        return [r["content"] for r in self.db.query(
            "SELECT content FROM memory WHERE thread_id = ? ORDER BY created_at", (thread_id,))]

    def persist(self, directory: str) -> None:
        """Nothing to do: every finding is written as it is added."""
