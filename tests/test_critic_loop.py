"""Day 6 — the critic loop (§12, FR-13..FR-16).

Proves the disagreement mechanism: a rejected draft is revised and re-reviewed;
a later PASS ends the loop; and if the ceiling is hit without a PASS, the loop
degrades gracefully (best draft, flagged not-validated) instead of looping forever.
The whole exchange is captured in the verdict history.
"""

import pytest

from maestro.agents import Analyst, Critic, run_critic_loop
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import CriticOutput
from maestro.state import AnalysisDraft, CriticDecision, Evidence, Role, Subtask, SubtaskStatus

EVIDENCE = [Evidence(subtask_id="r1", source="src", content="fact A"), Evidence(subtask_id="r2", source="src2", content="fact B")]
ANALYSIS_SUBTASK = Subtask(id="analyze", description="compare", role=Role.analyst, depends_on=["r1", "r2"])
CRITIC_SUBTASK = Subtask(id="critique", description="review", role=Role.critic, depends_on=["analyze"])
INITIAL = AnalysisDraft(content="initial analysis", claims=["c1", "c2"], revision=0)


def critic_scripted(decisions):
    """Critic responder that returns a scripted sequence of PASS/REJECT verdicts."""
    state = {"i": 0}

    def responder(schema, messages):
        assert schema is CriticOutput
        d = decisions[min(state["i"], len(decisions) - 1)]
        state["i"] += 1
        fb = "" if d == CriticDecision.passed else "Claim c2 is unsupported by the evidence; cite a source or drop it."
        return CriticOutput(decision=d, feedback=fb)

    return responder


def analyst_responder(schema, messages):
    assert schema is AnalysisModel
    human = messages[-1][1]
    tag = "revised" if "critic feedback" in human.lower() else "initial"
    return AnalysisModel(content=f"{tag} analysis addressing feedback", claims=["c1", "c2-with-source"])


# --- Critic agent parses both verdicts --------------------------------------
def test_critic_can_reject_and_pass(make_stub):
    c_reject = Critic(model=make_stub(critic_scripted([CriticDecision.rejected])))
    _, v = c_reject.run(CRITIC_SUBTASK, "goal", INITIAL, EVIDENCE, iteration=1)
    assert v.decision == CriticDecision.rejected and v.feedback

    c_pass = Critic(model=make_stub(critic_scripted([CriticDecision.passed])))
    _, v2 = c_pass.run(CRITIC_SUBTASK, "goal", INITIAL, EVIDENCE, iteration=1)
    assert v2.decision == CriticDecision.passed


# --- reject -> revise -> pass -----------------------------------------------
def test_reject_then_revise_then_pass(make_stub):
    analyst = Analyst(model=make_stub(analyst_responder))
    critic = Critic(model=make_stub(critic_scripted([CriticDecision.rejected, CriticDecision.passed])))
    result = run_critic_loop(
        analyst=analyst,
        critic=critic,
        analysis_subtask=ANALYSIS_SUBTASK,
        critic_subtask=CRITIC_SUBTASK,
        goal="Compare X and Y",
        evidence=EVIDENCE,
        draft=INITIAL,
        max_iters=3,
    )
    assert result.passed and not result.degraded
    assert result.iterations == 2  # rejected on review 1, passed on review 2
    assert [v.decision for v in result.verdicts] == [CriticDecision.rejected, CriticDecision.passed]
    assert result.draft.revision == 1  # a revision actually happened
    assert "revised" in result.draft.content


# --- ceiling degrades instead of looping forever ----------------------------
def test_ceiling_halts_and_degrades(make_stub):
    analyst = Analyst(model=make_stub(analyst_responder))
    critic = Critic(model=make_stub(critic_scripted([CriticDecision.rejected])))  # always rejects
    result = run_critic_loop(
        analyst=analyst,
        critic=critic,
        analysis_subtask=ANALYSIS_SUBTASK,
        critic_subtask=CRITIC_SUBTASK,
        goal="Compare X and Y",
        evidence=EVIDENCE,
        draft=INITIAL,
        max_iters=3,
    )
    assert not result.passed and result.degraded
    assert result.iterations == 3  # bounded at the ceiling
    assert len(result.verdicts) == 3
    assert all(v.decision == CriticDecision.rejected for v in result.verdicts)
    assert result.draft.revision == 2  # revised after reviews 1 and 2, not after the 3rd


# --- forced-reject demo affordance ------------------------------------------
def test_force_critic_reject_triggers_rejection(make_stub, monkeypatch):
    from maestro.config import reset_settings_cache

    monkeypatch.setenv("MAESTRO_FORCE_CRITIC_REJECT", "1")
    reset_settings_cache()
    # even a PASS-scripted model is overridden to REJECT on review 1
    critic = Critic(model=make_stub(critic_scripted([CriticDecision.passed])))
    _, v = critic.run(CRITIC_SUBTASK, "goal", INITIAL, EVIDENCE, iteration=1)
    assert v.decision == CriticDecision.rejected
    assert "forced-reject" in v.feedback
