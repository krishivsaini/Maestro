"""Recorded runs, replayed offline: real model output through the current code.

Each cassette in data/cassettes holds every model response and search result of one
live run, and what that run produced. Replaying runs all of Maestro's own code —
planning, scheduling, grounding and code checks, the critic's decision, routing,
citations — on that real output, in about a second, with no key or quota. If the code
now decides differently, the replay says exactly what changed.

A deliberate change: ``python scripts/record_run.py --refresh <name>``.
A new or removed model call: re-record (see maestro/cassette.py).
"""

import copy
import glob
import json
import os

import pytest

from maestro import cassette

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASSETTES = sorted(glob.glob(os.path.join(ROOT, "data", "cassettes", "*.json")))


def test_there_are_cassettes():
    assert len(CASSETTES) >= 3


@pytest.mark.parametrize("path", CASSETTES, ids=lambda p: os.path.basename(p)[:-5])
def test_a_recorded_run_replays_the_same(path):
    data = cassette.load(path)
    got = cassette.summarize(cassette.replay(data))
    expected = data["expected"]
    changed = {k: {"recorded": expected.get(k), "now": got.get(k)} for k in expected if expected.get(k) != got.get(k)}
    name = os.path.basename(path)[:-5]
    assert not changed, (
        f"{name} now behaves differently on the same model output:\n{json.dumps(changed, indent=1)}\n"
        f"If that is intended: python scripts/record_run.py --refresh {name}")


def test_a_changed_decision_is_caught(monkeypatch):
    # The critic's recorded review now says a claim is contradicted: the run takes a
    # different path (a revision), needs a model call the cassette doesn't have, and
    # the replay fails loudly instead of quietly passing.
    data = copy.deepcopy(cassette.load(os.path.join(ROOT, "data", "cassettes", "solar-wind.json")))
    review = data["calls"]["CriticOutput#0"]["output"]
    review["claim_checks"] = [{"claim": 1, "verdict": "contradicted", "note": "planted"}]
    with pytest.raises(cassette.ReplayMiss, match="AnalysisModel#1"):
        cassette.replay(data)


def test_a_replay_never_reaches_the_network(monkeypatch):
    # every search the run makes must come from the cassette
    data = copy.deepcopy(cassette.load(CASSETTES[0]))
    data["searches"] = {}
    with pytest.raises(cassette.ReplayMiss, match="no search"):
        cassette.replay(data)


def test_a_replay_never_fetches_a_page_live():
    data = copy.deepcopy(cassette.load(CASSETTES[0]))
    assert data["pages"], "cassettes record the pages researchers read"
    data["pages"] = {}
    with pytest.raises(cassette.ReplayMiss, match="no page"):
        cassette.replay(data)


@pytest.mark.parametrize("path", CASSETTES, ids=lambda p: os.path.basename(p)[:-5])
def test_cassettes_hold_no_secrets(path):
    # cassettes hold page text, which may well mention "api_key"; check for key material itself
    text = open(path).read()
    assert "AIza" not in text  # the prefix of every Google API key
    for var in ("GOOGLE_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "SEARCH_API_KEY"):
        key = os.environ.get(var)
        if key:
            assert key not in text
