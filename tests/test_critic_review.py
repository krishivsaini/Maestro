"""The structured critic and the agents around it.

The old critic returned a bare PASS/REJECT and passed 16 of 16 real reviews. These
prove the new one's mechanics, offline: what the code checks catch without a model,
that the verdict follows from the findings (not from the model's say-so), that a
re-review tracks issues instead of moving the goalposts, and that the analyst,
writer and researcher do their part.
"""

import pytest

from maestro.agents import Analyst, Critic, Researcher, Writer
from maestro.agents.analyst import AnalysisModel
from maestro.agents.critic import (
    ClaimCheckOut,
    CriticOutput,
    IssueOut,
    PriorStatus,
    decide,
    precheck,
)
from maestro.agents.researcher import ResearchFinding, _ground
from maestro.agents.writer import WriterOutput
from maestro.config import Settings
from maestro.state import (
    AnalysisDraft,
    ClaimCheck,
    CriticDecision,
    CriticVerdict,
    Evidence,
    ReviewIssue,
    Role,
    Subtask,
)

EV = [Evidence(subtask_id="r1", source=f"s{i}", content=f"fact {i}") for i in range(1, 6)]
CRIT = Subtask(id="critique", description="review", role=Role.critic)
GOAL = "Compare solar and wind for the grid, and recommend one."
CLEAN = AnalysisDraft(content="We recommend wind: steadier output.", claims=["wind is steadier", "solar peaks midday"],
                      refs=[[1, 2], [3]])


def issue(id, severity, origin="critic", problem=None):
    return ReviewIssue(id=id, severity=severity, criterion="grounding", problem=problem or f"problem {id}", origin=origin)


def critic_returning(out: CriticOutput, seen=None):
    def responder(schema, messages):
        if seen is not None:
            seen.append(messages[-1][1])
        return out
    return responder


# --- layer 1: checks in code -----------------------------------------------------------
def test_a_clean_draft_passes_the_checks():
    assert precheck(GOAL, CLEAN, EV) == []


def test_checks_catch_uncited_claims_bad_citations_and_no_recommendation():
    draft = AnalysisDraft(content="Both have merits.", claims=["a", "b"], refs=[[], [9]])
    found = [(i.severity, i.criterion, i.problem) for i in precheck(GOAL, draft, EV)]
    assert any(s == "major" and "C1 cites no evidence" in p for s, _, p in found)
    assert any(s == "blocker" and "[9]" in p and "[1]–[5]" in p for s, _, p in found)
    assert any(s == "major" and c == "coverage" and "doesn't commit" in p for s, c, p in found)


def test_checks_flag_a_draft_resting_on_one_source():
    draft = AnalysisDraft(content="We recommend wind.", claims=["a", "b", "c"], refs=[[1], [1], [1]])
    assert any("rests on 1 of 5" in i.problem for i in precheck(GOAL, draft, EV))


def test_no_recommendation_is_required_when_none_is_asked():
    draft = AnalysisDraft(content="Both have merits.", claims=["a"], refs=[[1]])
    assert precheck("Explain how solar power works", draft, EV) == []


def test_a_draft_with_no_claims_is_a_blocker():
    assert precheck(GOAL, AnalysisDraft(content="x"), EV)[0].severity == "blocker"


# --- layer 3: the decision is the code's, not the model's -----------------------------------
@pytest.mark.parametrize("severities,decision", [
    ([], CriticDecision.passed),
    (["minor", "minor", "minor"], CriticDecision.passed),
    (["major"], CriticDecision.passed),
    (["major", "major"], CriticDecision.rejected),
    (["blocker"], CriticDecision.rejected),
])
def test_decision_follows_the_findings(severities, decision):
    assert decide([issue(i, s) for i, s in enumerate(severities, 1)], major_limit=2) == decision


def test_resolved_issues_do_not_count():
    assert decide([issue(1, "blocker").model_copy(update={"status": "resolved"})], 2) == CriticDecision.passed


def test_claim_checks_become_issues(make_stub):
    out = CriticOutput(claim_checks=[ClaimCheckOut(claim=1, verdict="contradicted", note="[1] says the opposite"),
                                     ClaimCheckOut(claim=2, verdict="supported")])
    _, v = Critic(model=make_stub(critic_returning(out))).run(CRIT, GOAL, CLEAN, EV)
    assert v.decision == CriticDecision.rejected
    blocker = v.open_issues("blocker")[0]
    assert "C1 is contradicted" in blocker.problem and "wind is steadier" in blocker.problem
    assert [c.verdict for c in v.claim_checks] == ["contradicted", "supported"]


def test_one_unsupported_claim_alone_does_not_reject(make_stub):
    out = CriticOutput(claim_checks=[ClaimCheckOut(claim=1, verdict="unsupported"), ClaimCheckOut(claim=2, verdict="supported")])
    _, v = Critic(model=make_stub(critic_returning(out))).run(CRIT, GOAL, CLEAN, EV)
    assert v.decision == CriticDecision.passed and len(v.open_issues("major")) == 1


def test_the_model_cannot_pass_a_draft_the_checks_fail(make_stub):
    # the model finds nothing, but the draft cites evidence that doesn't exist
    draft = CLEAN.model_copy(update={"refs": [[1], [42]]})
    _, v = Critic(model=make_stub(critic_returning(CriticOutput()))).run(CRIT, GOAL, draft, EV)
    assert v.decision == CriticDecision.rejected and v.open_issues("blocker")[0].origin == "check"


# --- re-reviews ---------------------------------------------------------------------------
def test_a_re_review_confirms_fixes_and_cannot_move_the_goalposts(make_stub):
    seen = []
    out = CriticOutput(prior=[PriorStatus(id=1, resolved=True)],
                       issues=[IssueOut(severity="major", criterion="coverage", problem="new nitpick")])
    _, v = Critic(model=make_stub(critic_returning(out, seen))).run(
        CRIT, GOAL, CLEAN, EV, iteration=2, prior=[issue(1, "blocker")])
    assert "PREVIOUS ISSUES" in seen[0] and "#1 [blocker] problem 1" in seen[0]
    assert v.issues[0].id == 1 and v.issues[0].status == "resolved"
    nitpick = next(i for i in v.issues if i.problem == "new nitpick")
    assert nitpick.severity == "minor" and nitpick.id == 2  # a re-review may add blockers only
    assert v.decision == CriticDecision.passed and not v.stalled


def test_a_re_review_can_still_add_a_blocker(make_stub):
    out = CriticOutput(prior=[PriorStatus(id=1, resolved=True)],
                       issues=[IssueOut(severity="blocker", criterion="consistency", problem="now contradicts itself")])
    _, v = Critic(model=make_stub(critic_returning(out))).run(CRIT, GOAL, CLEAN, EV, iteration=2, prior=[issue(1, "major")])
    assert v.decision == CriticDecision.rejected and not v.stalled  # #1 was fixed: progress, not a stall


def test_a_revision_that_fixes_nothing_is_stalled(make_stub):
    out = CriticOutput(prior=[PriorStatus(id=1, resolved=False)])
    _, v = Critic(model=make_stub(critic_returning(out))).run(CRIT, GOAL, CLEAN, EV, iteration=2, prior=[issue(1, "blocker")])
    assert v.decision == CriticDecision.rejected and v.stalled


def test_an_unconfirmed_prior_issue_stays_open(make_stub):
    # the model says nothing about #1: it is not quietly dropped
    _, v = Critic(model=make_stub(critic_returning(CriticOutput()))).run(
        CRIT, GOAL, CLEAN, EV, iteration=2, prior=[issue(1, "blocker")])
    assert v.issues[0].status == "open" and v.decision == CriticDecision.rejected


def test_code_check_issues_are_confirmed_by_the_code_not_the_model(make_stub):
    seen = []
    still_failing = precheck(GOAL, AnalysisDraft(content="Both have merits.", claims=["a"], refs=[[1]]), EV)[0]
    prior = [still_failing.model_copy(update={"id": 1})]
    # the model claims it is fixed; the draft still makes no recommendation
    out = CriticOutput(prior=[PriorStatus(id=1, resolved=True)])
    draft = AnalysisDraft(content="Both have merits.", claims=["a"], refs=[[1]])
    _, v = Critic(model=make_stub(critic_returning(out, seen))).run(CRIT, GOAL, draft, EV, iteration=2, prior=prior)
    assert "PREVIOUS ISSUES" not in seen[0]                  # not the model's to judge
    assert [i.status for i in v.issues] == ["open"]           # carried, not resolved, not duplicated
    assert v.decision == CriticDecision.passed                # one open major alone does not block


# --- the critic's model ------------------------------------------------------------------
def test_critic_reviews_on_its_own_model_only_within_the_runs_provider():
    google = Critic(model=None, settings=Settings(model_id="gemini-3.5-flash-lite", critic_model_id="gemini-3.6-flash"))
    assert google._critic_model() == "gemini-3.6-flash"
    groq = Critic(model=None, settings=Settings(llm_provider="groq", model_id="llama-3.3-70b", critic_model_id="gemini-3.6-flash"))
    assert groq._critic_model() is None
    assert Critic(model=None, settings=Settings(critic_model_id=""))._critic_model() is None


def test_critic_falls_back_to_the_runs_model_when_its_own_fails(make_stub, monkeypatch):
    import maestro.llm

    built = []

    def fake_get_chat_model(settings, *, model_id=None, temperature=None):
        built.append((model_id, temperature))
        if model_id == "gemini-3.6-flash":
            def broken(schema, messages):
                raise RuntimeError("429 quota exhausted")
            return make_stub(broken)
        return make_stub(critic_returning(CriticOutput()))

    monkeypatch.setattr(maestro.llm, "get_chat_model", fake_get_chat_model)
    cfg = Settings(model_id="gemini-3.5-flash-lite", critic_model_id="gemini-3.6-flash", backoff_max_attempts=1)
    _, v = Critic(settings=cfg).run(CRIT, GOAL, CLEAN, EV)
    assert v.decision == CriticDecision.passed
    assert built == [("gemini-3.6-flash", 0.0), ("gemini-3.5-flash-lite", 0.0)]


# --- the analyst --------------------------------------------------------------------------
def test_a_revision_starts_from_the_previous_draft_and_the_open_issues(make_stub):
    seen = []

    def responder(schema, messages):
        seen.append(messages[-1][1])
        return AnalysisModel(content="revised", claims=["wind is steadier [1, 2]"],
                             addressed=[{"issue": 3, "change": "cited [2] for C1"}])

    _, draft = Analyst(model=make_stub(responder)).run(
        Subtask(id="analyze", description="compare", role=Role.analyst), GOAL, EV, revision=1,
        previous=CLEAN, issues=[issue(3, "blocker", problem="C1 isn't supported")])
    assert "Your previous draft" in seen[0] and "We recommend wind" in seen[0]
    assert "C1. wind is steadier  (cites [1], [2])" in seen[0]
    assert "#3 [blocker] C1 isn't supported" in seen[0]
    assert draft.claims == ["wind is steadier"] and draft.refs == [[1, 2]]
    assert draft.addressed == ["#3: cited [2] for C1"] and draft.revision == 1


# --- the writer ---------------------------------------------------------------------------
def test_the_writer_sees_the_review_and_states_open_issues_as_caveats(make_stub):
    seen = []

    def responder(schema, messages):
        seen.append(messages[-1][1])
        return WriterOutput(content="brief", citations=["1"])

    review = CriticVerdict(decision=CriticDecision.rejected, issues=[issue(1, "major", problem="cost dimension missing")],
                           claim_checks=[ClaimCheck(claim=2, verdict="unsupported")])
    Writer(model=make_stub(responder)).run(Subtask(id="write", description="w", role=Role.writer),
                                           GOAL, CLEAN, EV, validated=False, review=review)
    assert "C2. solar peaks midday  [reviewer: UNSUPPORTED]" in seen[0]
    assert "state these as caveats" in seen[0] and "cost dimension missing" in seen[0]


def test_a_validated_brief_gets_no_caveats(make_stub):
    seen = []
    Writer(model=make_stub(lambda s, m: seen.append(m[-1][1]) or WriterOutput(content="b", citations=[]))).run(
        Subtask(id="write", description="w", role=Role.writer), GOAL, CLEAN, EV, validated=True,
        review=CriticVerdict(decision=CriticDecision.passed, issues=[issue(1, "minor")]))
    assert "caveats" not in seen[0]


# --- the researcher -----------------------------------------------------------------------
RESULTS = [{"url": "https://a.org/x", "snippet": "..."}, {"url": "https://b.org/y", "snippet": "..."}]


def test_findings_must_cite_a_source_the_researcher_was_shown():
    kept = _ground([ResearchFinding(source="https://a.org/x", content="real"),
                    ResearchFinding(source="https://made-up.com", content="invented"),
                    ResearchFinding(source="[2]", content="by number")], RESULTS)
    assert [(f.source, f.content) for f in kept] == [("https://a.org/x", "real"), ("https://b.org/y", "by number")]


def test_near_duplicate_findings_are_dropped():
    kept = _ground([ResearchFinding(source="https://a.org/x", content="Wind capacity factor is about 35 percent."),
                    ResearchFinding(source="https://b.org/y", content="Wind capacity factor is about 35 percent!"),
                    ResearchFinding(source="https://b.org/y", content="Solar peaks at midday.")], RESULTS)
    assert [f.content for f in kept] == ["Wind capacity factor is about 35 percent.", "Solar peaks at midday."]


def test_a_researcher_whose_findings_are_all_invented_fails(make_stub):
    from maestro.agents.researcher import ResearchFindings
    from maestro.tools import DEFAULT_REGISTRY

    model = make_stub(lambda s, m: ResearchFindings(findings=[ResearchFinding(source="https://nowhere.test", content="x")]))
    st, ev = Researcher(model=model).run(Subtask(id="r1", description="solar capacity factor", role=Role.researcher),
                                         tools=DEFAULT_REGISTRY, corpus=[{"source": "src_solar", "content": "solar capacity factor 25%"}])
    assert st.status.value == "failed" and ev == [] and "grounded" in st.error


# --- figures must be in the evidence they cite ------------------------------------------------
FIG_EV = [Evidence(subtask_id="r", source="a", content="Wind averaged a 35% capacity factor in 2022."),
          Evidence(subtask_id="r", source="b", content="Battery costs fell 70% between 2015 and 2023."),
          Evidence(subtask_id="r", source="c", content="Solar peaks at midday.")]


def test_figures_normalise_percentages_and_skip_small_counts():
    from maestro.agents.critic import figures

    assert figures("about 35% in 2022 and 24 percent") == {"35%", "2022", "24%"}
    assert figures("2 options for 1,000 users at 3.5x") == {"1000", "3.5"}


def test_a_figure_from_another_item_is_a_miscitation():
    draft = AnalysisDraft(content="We recommend wind.", claims=["Battery costs fell 70% from 2015 to 2023"], refs=[[1]])
    found = precheck(GOAL, draft, FIG_EV)
    assert [(i.severity, i.problem) for i in found] == [
        ("blocker", "Claim C1 cites [1], but its figures (2015, 2023, 70%) are in [2].")]


def test_a_figure_in_no_evidence_is_flagged_as_invented():
    draft = AnalysisDraft(content="We recommend wind.", claims=["Wind delivers 99.9% availability"], refs=[[1]])
    assert precheck(GOAL, draft, FIG_EV)[0].severity == "major"


def test_figures_that_match_their_citation_pass():
    draft = AnalysisDraft(content="We recommend wind.", claims=["Wind's capacity factor was 35 percent in 2022"], refs=[[1]])
    assert precheck(GOAL, draft, FIG_EV) == []


def test_a_miscited_claim_is_a_blocker(make_stub):
    out = CriticOutput(claim_checks=[ClaimCheckOut(claim=1, verdict="miscited", note="[3] states it, not [1]"),
                                     ClaimCheckOut(claim=2, verdict="supported")])
    _, v = Critic(model=make_stub(critic_returning(out))).run(CRIT, GOAL, CLEAN, EV)
    assert v.decision == CriticDecision.rejected and "[3] states it" in v.open_issues("blocker")[0].problem


def test_a_source_counts_however_it_is_written():
    data = [{"url": "https://www.a.org/x/", "title": "Wind power facts", "snippet": "..."},
            {"url": "https://b.org/y", "snippet": "..."}]
    kept = _ground([ResearchFinding(source="a.org/x", content="no scheme or www"),
                    ResearchFinding(source="Result 2", content="by result number"),
                    ResearchFinding(source="Wind power facts", content="by title"),
                    ResearchFinding(source="https://c.org/z", content="never shown")], data)
    assert [(f.source, f.content) for f in kept] == [("https://www.a.org/x/", "no scheme or www"),
                                                    ("https://b.org/y", "by result number"),
                                                    ("https://www.a.org/x/", "by title")]
