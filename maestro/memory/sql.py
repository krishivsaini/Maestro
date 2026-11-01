"""Long-term memory in the database, so a thread's findings outlive a restart.

The file-backed FAISS store (longterm.py) is right locally, but a host that wipes its
disk on every deploy forgets every thread. With ``DATABASE_URL`` set, findings are
stored as rows alongside the runs (maestro/db.py), each with its embedding. A thread
holds a handful of findings, so recall reads that thread's rows and ranks them by
cosine similarity in numpy: no vector extension needed, so any Postgres will do.
Same interface as ``LongTermMemory``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..db import Database
from ..state import MemoryItem
from .longterm import Embedder, HashingEmbedder


def _encode(vec: np.ndarray) -> str:
    """Sparse JSON ({index: value}): hashing embeddings are mostly zeros."""
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
        self.embedder: Embedder = embedder or HashingEmbedder()
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

    def add(self, thread_id: str, content: str) -> None:
        vec = _unit(self.embedder.embed([content])[0])
        self.db.execute("INSERT INTO memory (thread_id, content, embedding, created_at) VALUES (?, ?, ?, ?)",
                        (thread_id, content, _encode(vec), datetime.now(timezone.utc).isoformat()))

    def query(self, thread_id: str, query: str, k: int = 3) -> list[MemoryItem]:
        rows = self.db.query("SELECT content, embedding FROM memory WHERE thread_id = ? ORDER BY created_at",
                             (thread_id,))
        if not rows:
            return []
        q = _unit(self.embedder.embed([query])[0])
        scored = [(float(np.dot(q, _decode(r["embedding"], q.shape[0]))), r["content"]) for r in rows]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: -s[0])
        return [MemoryItem(thread_id=thread_id, content=c, score=s) for s, c in scored[:k]]

    def count(self, thread_id: str) -> int:
        return int(self.db.query("SELECT COUNT(*) AS n FROM memory WHERE thread_id = ?", (thread_id,))[0]["n"])

    def findings(self, thread_id: str) -> list[str]:
        return [r["content"] for r in self.db.query(
            "SELECT content FROM memory WHERE thread_id = ? ORDER BY created_at", (thread_id,))]

    def persist(self, directory: str) -> None:
        """Nothing to do: every finding is written as it is added."""
