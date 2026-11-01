"""Long-term memory recalls by meaning, and degrades to words instead of failing.

A fake meaning embedder stands in for the Gemini API: it maps topics to fixed vectors
and counts its calls. scripts/memory_eval.py measures the real one (12/12 paraphrased
follow-ups recalled first, against 1/12 for word matching).
"""

import json

import numpy as np
import pytest

from maestro.config import Settings
from maestro.db import Database
from maestro.memory import SqlMemory

TOPICS = {"night": 0, "sunset": 0, "dark": 0, "battery": 1, "storage": 1, "affordable": 1, "cheaper": 1,
          "hurricane": 2, "typhoon": 2, "storm": 2}


class FakeMeaning:
    """Embeds by topic, so 'after sunset' and 'at night' land together without shared words."""

    name, model, dim = "fake", "m1", 3

    def __init__(self, fail_documents=False, fail_queries=False):
        self.calls = {"documents": 0, "query": 0}
        self.fail_documents, self.fail_queries = fail_documents, fail_queries

    def _vec(self, text):
        v = np.full(3, 0.05, dtype="float32")
        for word, i in TOPICS.items():
            if word in text.lower():
                v[i] += 1
        return v

    def embed_documents(self, texts):
        self.calls["documents"] += 1
        if self.fail_documents:
            raise RuntimeError("429 embedding quota")
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        self.calls["query"] += 1
        if self.fail_queries:
            raise RuntimeError("503 unavailable")
        return self._vec(text)

    def embed(self, texts):
        return self.embed_documents(texts)


FINDINGS = ["Solar output falls to zero at night, so evening demand needs other sources.",
            "Four-hour battery storage costs fell roughly 70% between 2015 and 2023.",
            "Typhoon-rated turbines feather their blades and stop producing in a storm."]


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "m.db"))


def test_recall_by_meaning_finds_a_match_that_shares_no_words(db):
    mem = SqlMemory(db, FakeMeaning())
    mem.add_many("t", FINDINGS)
    assert mem.query("t", "Which one keeps producing after sunset?", k=1)[0].content == FINDINGS[0]
    assert mem.query("t", "Have they become more affordable?", k=1)[0].content == FINDINGS[1]
    assert mem.query("t", "What happens in a hurricane?", k=1)[0].content == FINDINGS[2]


def test_a_runs_findings_are_embedded_in_one_call(db):
    emb = FakeMeaning()
    SqlMemory(db, emb).add_many("t", FINDINGS)
    assert emb.calls["documents"] == 1


def test_if_embedding_fails_on_write_findings_are_still_kept_and_recalled_by_words(db):
    mem = SqlMemory(db, FakeMeaning(fail_documents=True))
    mem.add_many("t", FINDINGS)
    assert mem.count("t") == 3
    hits = mem.query("t", "battery storage costs", k=1)
    assert hits[0].content == FINDINGS[1]


def test_if_embedding_the_question_fails_recall_falls_back_to_words(db):
    SqlMemory(db, FakeMeaning()).add_many("t", FINDINGS)
    mem = SqlMemory(db, FakeMeaning(fail_queries=True))
    assert mem.query("t", "battery storage costs", k=1)[0].content == FINDINGS[1]


def test_vectors_from_a_different_model_are_never_compared(db):
    SqlMemory(db, FakeMeaning()).add_many("t", FINDINGS)
    newer = FakeMeaning()
    newer.model = "m2"  # the embedding model changed since these were stored
    mem = SqlMemory(db, newer)
    mem.query("t", "Which one keeps producing after sunset?")
    # the stored vectors came from m1, so m2 isn't asked to embed the question at all:
    # recall falls back to words rather than comparing vectors from different models
    assert newer.calls["query"] == 0


def test_word_only_memory_still_works(db):
    mem = SqlMemory(db)  # no meaning embedder
    mem.add_many("t", FINDINGS)
    assert mem.query("t", "storage costs fell", k=1)[0].content == FINDINGS[1]


def test_an_existing_table_is_upgraded_in_place(db):
    db.script("CREATE TABLE memory (thread_id TEXT, content TEXT, embedding TEXT, created_at TEXT)")
    SqlMemory(db).add("t", "battery storage costs fell")  # an old, word-only row
    mem = SqlMemory(db, FakeMeaning())                      # the new code over the old table
    cols = {r["name"] for r in db.query("PRAGMA table_info(memory)")}
    assert {"semantic", "semantic_tag"} <= cols
    assert mem.query("t", "battery storage")[0].content == "battery storage costs fell"


def test_findings_from_the_old_file_memory_are_imported_once(db, tmp_path):
    from maestro.serve import import_file_memory

    old = tmp_path / "memory_store"
    old.mkdir()
    (old / "meta.json").write_text(json.dumps({"dim": 256, "meta": [
        {"thread_id": "t1", "content": "wind is steadier at night", "created_at": "x"},
        {"thread_id": "t1", "content": "storage got cheaper", "created_at": "x"}]}))
    mem = SqlMemory(db)
    import_file_memory(str(old), mem)
    import_file_memory(str(old), mem)  # a second start doesn't duplicate them
    assert mem.findings("t1") == ["wind is steadier at night", "storage got cheaper"]


def test_the_server_recalls_by_meaning_when_it_has_a_key():
    from maestro.memory.embedders import GeminiEmbedder
    from maestro.serve import memory_embedder

    assert isinstance(memory_embedder(Settings(google_api_key="AIza-test", memory_embedder="gemini")), GeminiEmbedder)
    assert memory_embedder(Settings(google_api_key="", memory_embedder="gemini")) is None
    assert memory_embedder(Settings(google_api_key="AIza-test", memory_embedder="hashing")) is None
