"""Meaning-based embeddings for long-term memory.

The default ``HashingEmbedder`` (longterm.py) is a bag of words: a follow-up recalls a
finding only if it shares words with it, which is why follow-ups had to carry their
thread's opening question. ``GeminiEmbedder`` embeds by meaning through the Gemini
embedding API, on the key Maestro already uses — no model to download, nothing held in
the small free instance's memory. Findings and questions are embedded as what they are
(``RETRIEVAL_DOCUMENT`` vs ``RETRIEVAL_QUERY``), which ranks better than embedding both
the same way. scripts/memory_eval.py measures the difference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import Settings, get_settings
from ..resilience import resilient_call


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, settings: Optional[Settings] = None, *, model: Optional[str] = None,
                 dim: Optional[int] = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.gemini_embedding_model
        self.dim = dim or self.settings.embedding_dim
        self._client = None

    def _embeddings(self):
        if self._client is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # lazy

            self._client = GoogleGenerativeAIEmbeddings(model=f"models/{self.model}",
                                                        google_api_key=self.settings.google_api_key)
        return self._client

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = resilient_call(self._embeddings().embed_documents, texts, task_type="RETRIEVAL_DOCUMENT",
                              output_dimensionality=self.dim, settings=self.settings)
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = resilient_call(self._embeddings().embed_query, text, task_type="RETRIEVAL_QUERY",
                             output_dimensionality=self.dim, settings=self.settings)
        return np.asarray(vec, dtype="float32")

    def embed(self, texts: list[str]) -> np.ndarray:  # the Embedder protocol
        return self.embed_documents(texts)
