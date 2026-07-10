"""Long-term memory (§14) — a small FAISS vector store over distilled findings.

Persisted across turns of a thread. On completion the graph writes distilled
findings tagged with the thread id; on a later turn the supervisor queries it
during planning and results land in ``memory_hits``, informing the analysis.

The **embedder is swappable**: production uses local ``sentence-transformers``
(free, no API cost); tests/offline use a deterministic hashing embedder so the
whole thing runs without torch. Retrieval here is deliberately simple (§1.3) —
the signal in this project is orchestration, not retrieval quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional, Protocol

import numpy as np

from ..logging_config import get_logger
from ..state import MemoryItem

log = get_logger("memory")

_WORD = re.compile(r"[a-z0-9]+")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder — no ML deps, no torch.

    Stable across processes (uses md5, not Python's salted hash), so a persisted
    store reloads correctly.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in _WORD.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        return vecs


class SentenceTransformerEmbedder:
    """Production embedder — local sentence-transformers model (lazy-loaded)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy (heavy)

            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(texts), dtype="float32")


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class LongTermMemory:
    """Thread-scoped vector memory backed by a FAISS inner-product index.

    Vectors are L2-normalized so inner product == cosine similarity. Thread
    filtering is done post-search (the store is small).
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.index = None  # created lazily once we know the dim
        self.dim: Optional[int] = None
        self.meta: list[dict] = []

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            import faiss

            self.dim = dim
            self.index = faiss.IndexFlatIP(dim)

    def add(self, thread_id: str, content: str) -> None:
        vec = _normalize(self.embedder.embed([content]))
        self._ensure_index(vec.shape[1])
        self.index.add(vec)
        self.meta.append({"thread_id": thread_id, "content": content, "created_at": _utcnow()})

    def query(self, thread_id: str, query: str, k: int = 3) -> list[MemoryItem]:
        if self.index is None or self.index.ntotal == 0:
            return []
        qv = _normalize(self.embedder.embed([query]))
        n = min(self.index.ntotal, max(k * 5, k))
        scores, idxs = self.index.search(qv, n)
        hits: list[MemoryItem] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            m = self.meta[idx]
            if m["thread_id"] != thread_id or score <= 0:
                continue
            hits.append(MemoryItem(thread_id=thread_id, content=m["content"], score=float(score)))
            if len(hits) >= k:
                break
        return hits

    # --- persistence ---
    def persist(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"dim": self.dim, "meta": self.meta}, f)
        if self.index is not None:
            import faiss

            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

    @classmethod
    def load(cls, directory: str, embedder: Optional[Embedder] = None) -> "LongTermMemory":
        inst = cls(embedder)
        meta_path = os.path.join(directory, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                data = json.load(f)
            inst.dim = data.get("dim")
            inst.meta = data.get("meta", [])
        index_path = os.path.join(directory, "index.faiss")
        if os.path.exists(index_path):
            import faiss

            inst.index = faiss.read_index(index_path)
        return inst
