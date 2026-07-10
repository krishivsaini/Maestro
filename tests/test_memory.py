"""Day 10 — long-term memory (§14, FR-28..FR-30).

The non-negotiable: long-term memory is written AND read-back-and-used. Proven by
a two-turn thread where turn 2's analysis provably contains a finding stored in
turn 1. Plus thread scoping and a persistence round-trip.
"""

from maestro.agents import Analyst, Critic, Writer
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import CriticOutput
from maestro.agents.writer import WriterOutput
from maestro.config import Settings
from maestro.graph import MaestroAgents, run_goal
from maestro.memory import HashingEmbedder, LongTermMemory
from maestro.state import CriticDecision, EventType, Evidence, SubtaskStatus
from maestro.supervisor import heuristic_planner

UNIQUE = "photovoltaicfactoid"  # a token that could only come from turn 1's memory


class OkResearcher:
    def run(self, subtask, *, tools=None, corpus=None, on_retry=None):
        done = subtask.model_copy(update={"status": SubtaskStatus.done, "result": "ok"})
        return done, [Evidence(subtask_id=subtask.id, source="s", content="e")]


def make_agents(make_stub, memory, responder):
    model = make_stub(responder)
    cfg = Settings(max_critic_iters=3, max_parallel=2)
    return MaestroAgents(
        researcher=OkResearcher(),
        analyst=Analyst(model=model, settings=cfg),
        critic=Critic(model=model, settings=cfg),
        writer=Writer(model=model, settings=cfg),
        settings=cfg,
        planner=heuristic_planner,
        memory=memory,
    )


def turn1_responder(schema, messages):
    name = getattr(schema, "__name__", "")
    if name == "AnalysisModel":
        return AnalysisModel(
            content="solar analysis",
            claims=[f"solar power {UNIQUE} capacity factor around 25 percent", "storage matters"],
        )
    if name == "CriticOutput":
        return CriticOutput(decision=CriticDecision.passed, feedback="")
    if name == "WriterOutput":
        return WriterOutput(content="brief", citations=["s"])
    raise AssertionError(name)


def test_two_turn_memory_written_then_used(make_stub):
    mem = LongTermMemory(embedder=HashingEmbedder(dim=256))

    # turn 1: writes distilled findings to long-term memory
    final1 = run_goal(
        "Explain solar power reliability",
        agents=make_agents(make_stub, mem, turn1_responder),
        thread_id="thread-A",
    )
    assert any(e.event_type == EventType.memory_write for e in final1["trace"])

    # turn 2 on the same thread: must recall and USE the stored finding
    captured = {}

    def turn2_responder(schema, messages):
        name = getattr(schema, "__name__", "")
        if name == "AnalysisModel":
            captured["human"] = messages[-1][1]  # what the analyst actually saw
            return AnalysisModel(content="analysis2", claims=["c"])
        if name == "CriticOutput":
            return CriticOutput(decision=CriticDecision.passed, feedback="")
        if name == "WriterOutput":
            return WriterOutput(content="brief2", citations=["s"])
        raise AssertionError(name)

    final2 = run_goal(
        "How reliable is solar power for the grid?",
        agents=make_agents(make_stub, mem, turn2_responder),
        thread_id="thread-A",
    )

    assert final2["memory_hits"], "expected long-term memory recall on turn 2"
    assert any(UNIQUE in h.content for h in final2["memory_hits"])
    assert any(e.event_type == EventType.memory_recall for e in final2["trace"])
    # provably USED, not just written: the recalled finding is in the analyst's input
    assert UNIQUE in captured["human"]


def test_memory_is_thread_scoped(make_stub):
    mem = LongTermMemory(embedder=HashingEmbedder(dim=256))
    run_goal(
        "Explain solar power reliability",
        agents=make_agents(make_stub, mem, turn1_responder),
        thread_id="thread-A",
    )
    # a DIFFERENT thread must not recall thread-A's findings
    final = run_goal(
        "How reliable is solar power?",
        agents=make_agents(make_stub, mem, turn1_responder),
        thread_id="thread-B",
    )
    assert not final.get("memory_hits")


def test_memory_query_and_persist_roundtrip(tmp_path):
    mem = LongTermMemory(embedder=HashingEmbedder(dim=128))
    mem.add("t1", "solar capacity factor is about 25 percent")
    mem.add("t1", "wind capacity factor is about 35 percent")
    mem.add("t2", "banana bread recipe with walnuts")

    hits = mem.query("t1", "solar capacity", k=2)
    assert hits and "solar" in hits[0].content
    assert all(h.thread_id == "t1" for h in hits)  # thread scoping

    mem.persist(str(tmp_path))
    reloaded = LongTermMemory.load(str(tmp_path), embedder=HashingEmbedder(dim=128))
    hits2 = reloaded.query("t1", "wind capacity", k=1)
    assert hits2 and "wind" in hits2[0].content
