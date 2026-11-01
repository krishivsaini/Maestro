"""Seeded-flaw evaluation of the critic (live: needs GOOGLE_API_KEY).

Runs a critic over data/critic_eval_cases.json — per topic, one clean draft that is
correct against its evidence and five copies with exactly one planted flaw
(unsupported statistic, wrong citation, no recommendation, contradiction,
overclaim) — and reports how many flaws it catches and how many clean drafts it
wrongly rejects.

Two critics can be measured on the same cases:
  new     the structured critic in maestro/agents/critic.py
  legacy  the critic before it, reproduced here exactly: its prompt and its bare
          PASS/REJECT output. It is shown the same claims with their citations, so
          the comparison measures the reviewing, not access to information.

Usage:
    python scripts/critic_eval.py new                      # on Settings.critic_model_id
    python scripts/critic_eval.py new --model gemini-3.5-flash-lite
    python scripts/critic_eval.py legacy --model gemini-3.5-flash-lite
    python scripts/critic_eval.py new --out data/critic_eval_new.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pydantic import BaseModel  # noqa: E402

from maestro.state import AnalysisDraft, CriticDecision, Evidence, Role, Subtask  # noqa: E402

CASES = os.path.join(ROOT, "data", "critic_eval_cases.json")
FLAWS = ["unsupported", "wrong_citation", "no_recommendation", "contradiction", "overclaim"]

# --- the critic as it was before the structured review (verbatim) --------------------------
LEGACY_PROMPT = """You are a Critic subagent in a multi-agent research system.
You review an Analyst's draft AGAINST the evidence it is supposed to rest on.
Decide one of:
- PASS: claims are supported by the evidence, the analysis addresses the goal, and there
  are no major gaps, contradictions, or unsupported assertions.
- REJECT: there are unsupported claims, missing key considerations, contradictions, or the
  goal is not addressed. When you REJECT, give specific, actionable feedback naming the
  problems so the Analyst can fix them.
Be a genuine reviewer: do not rubber-stamp. If the draft is weak, REJECT it."""


class LegacyOutput(BaseModel):
    decision: CriticDecision
    feedback: str = ""


def build_cases(topics: list[str] | None = None) -> list[dict]:
    data = json.load(open(CASES))
    cases = []
    for t in data["topics"]:
        if topics and t["id"] not in topics:
            continue
        evidence = [Evidence(subtask_id="r", source=s, content=c) for s, c in t["evidence"]]
        clean = AnalysisDraft(content=t["clean"]["content"], claims=[c for c, _ in t["clean"]["claims"]],
                              refs=[r for _, r in t["clean"]["claims"]])
        cases.append({"topic": t["id"], "flaw": "clean", "goal": t["goal"], "evidence": evidence, "draft": clean})
        for flaw in FLAWS:
            op = t["flaws"][flaw]
            content, claims, refs = clean.content, list(clean.claims), [list(r) for r in clean.refs]
            for old, new in op.get("replace", []):
                hits = sum(old in x for x in [content, *claims])
                assert hits, f"{t['id']}/{flaw}: {old!r} not found"
                content = content.replace(old, new)
                claims = [c.replace(old, new) for c in claims]
            if "content_append" in op:
                content += op["content_append"]
            if "add_claim" in op:
                claims.append(op["add_claim"][0]); refs.append(op["add_claim"][1])
            for i, r in op.get("claim_refs", {}).items():
                refs[int(i) - 1] = r
            draft = AnalysisDraft(content=content, claims=claims, refs=refs)
            cases.append({"topic": t["id"], "flaw": flaw, "goal": t["goal"], "evidence": evidence, "draft": draft})
    return cases


def run_new(case, settings):
    from maestro.agents import Critic

    _, v = Critic(settings=settings).run(Subtask(id="critique", description="review", role=Role.critic),
                                         case["goal"], case["draft"], case["evidence"])
    top = [f"[{i.severity}] {i.problem}" for i in v.open_issues("blocker", "major")]
    return v.decision, top


def run_legacy(case, settings):
    from maestro.llm import get_chat_model
    from maestro.resilience import resilient_call

    d, ev = case["draft"], case["evidence"]
    claims = "\n".join(f"- {c} [{', '.join(map(str, r))}]" for c, r in zip(d.claims, d.refs))
    rendered = "\n".join(f"[{i}] ({e.source}) {e.content}" for i, e in enumerate(ev, start=1))
    human = (f"Goal: {case['goal']}\n\nAnalysis draft:\n{d.content}\n\nClaims made:\n{claims}\n\n"
             f"Evidence the draft must rest on:\n{rendered}")
    chat = get_chat_model(settings).with_structured_output(LegacyOutput)  # the run's model, default temperature
    out = resilient_call(chat.invoke, [("system", LEGACY_PROMPT), ("human", human)], settings=settings)
    return out.decision, [out.feedback] if out.feedback else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("critic", choices=["new", "legacy"])
    ap.add_argument("--model", help="model to review on (new: overrides critic_model_id; legacy: the run model)")
    ap.add_argument("--out", help="write per-case results as JSON")
    ap.add_argument("--topic", action="append", help="only these topic ids (repeatable)")
    ap.add_argument("--pace", type=float, default=4.6,
                    help="minimum seconds between reviews (the free tier allows ~15 requests/minute per model)")
    args = ap.parse_args()

    from maestro.config import get_settings

    s = get_settings()
    if args.model:
        s = s.model_copy(update={"critic_model_id": args.model} if args.critic == "new" else {"model_id": args.model})
    model = (s.critic_model_id or s.model_id) if args.critic == "new" else s.model_id
    run = run_new if args.critic == "new" else run_legacy
    print(f"critic={args.critic} model={model}\n")

    rows = []
    last = 0.0
    for case in build_cases(args.topic):
        time.sleep(max(0.0, args.pace - (time.time() - last)))
        t0 = last = time.time()
        decision, notes = run(case, s)
        expected = CriticDecision.passed if case["flaw"] == "clean" else CriticDecision.rejected
        ok = decision == expected
        rows.append({"topic": case["topic"], "flaw": case["flaw"], "decision": decision.value, "correct": ok, "notes": notes})
        print(f"{'ok  ' if ok else 'MISS'} {case['topic']:15} {case['flaw']:18} -> {decision.value:6} ({time.time() - t0:.1f}s)")
        for n in notes[:3]:
            print(f"       {n[:150]}")

    print()
    for flaw in FLAWS:
        hit = [r for r in rows if r["flaw"] == flaw]
        print(f"  caught {flaw:18} {sum(r['correct'] for r in hit)}/{len(hit)}")
    flawed = [r for r in rows if r["flaw"] != "clean"]
    clean = [r for r in rows if r["flaw"] == "clean"]
    print(f"\n  flaws caught:          {sum(r['correct'] for r in flawed)}/{len(flawed)}")
    print(f"  clean drafts passed:   {sum(r['correct'] for r in clean)}/{len(clean)}")
    if args.out:
        json.dump({"critic": args.critic, "model": model, "rows": rows}, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
