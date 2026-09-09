"""Benchmark candidate models across providers on Maestro's real workload.

Two questions decide a provider swap, and neither is answerable from a vendor page:

1. **Does structured output work?** The planner calls ``with_structured_output(Plan)``.
   A model that cannot do it is unusable here however fast it is.
2. **What is the real latency under Maestro's prompt sizes?** The analyst prompt
   carries the accumulated evidence, so it is far larger than the planning prompt —
   and on tight free-tier TPM ceilings that large call is where throttling shows up.

So each model is measured twice: a small structured planning call and a large
free-text analysis call padded to a realistic evidence load. Every model runs
``--trials`` times because a single sample has already misled this project once.

Usage:
    uv run --extra providers python scripts/bench_models.py
    uv run --extra providers python scripts/bench_models.py --trials 5
    uv run --extra providers python scripts/bench_models.py --models groq:llama-3.3-70b-versatile
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from typing import Optional

from maestro.config import get_settings
from maestro.llm import get_chat_model, split_model_id
from maestro.state import Role
from maestro.supervisor import DECOMP_SYSTEM, ROLE_GUIDE, Plan

GOAL = "Compare REST and GraphQL for a public API and recommend one."

# Stand-in for the evidence the analyst actually receives; the point is the token
# volume, since that is what runs into a TPM ceiling.
_EVIDENCE_UNIT = (
    "Source {i}: REST APIs benefit from mature HTTP caching across proxies and CDNs, "
    "which keeps read-heavy public endpoints cheap to serve. GraphQL exposes a single "
    "endpoint, so standard HTTP caching does not apply and callers must adopt persisted "
    "queries or an application-level cache. Security review notes that deeply nested "
    "GraphQL queries require depth limiting and cost analysis to avoid resource "
    "exhaustion, whereas REST endpoints are rate-limited per route. "
)

# Ids verified against each provider's live model listing — guessing them wasted a
# benchmark round on NotFoundError.
DEFAULT_MODELS = [
    "google:gemini-3.5-flash-lite",
    "google:gemini-3.6-flash",
    "groq:openai/gpt-oss-20b",
    "groq:openai/gpt-oss-120b",
    "groq:qwen/qwen3.8-27b",
    "nvidia:openai/gpt-oss-20b",
    "nvidia:nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia:nvidia/nemotron-nano-3-30b-a3b",
]


def _evidence(units: int = 14) -> str:
    return "".join(_EVIDENCE_UNIT.format(i=i) for i in range(units))


def _plan_call(model) -> int:
    """Structured planning call. Returns subtask count, raises if unsupported."""
    roles = list(Role)
    role_lines = "\n".join(f"- {r.value}: {ROLE_GUIDE[r]}" for r in roles)
    structured = model.with_structured_output(Plan)
    plan = structured.invoke([
        ("system", DECOMP_SYSTEM.format(roles=role_lines, max_subtasks=6)),
        ("human", f"Goal: {GOAL}"),
    ])
    return len(plan.subtasks)


def _analyst_call(model, evidence: str) -> int:
    """Large free-text call approximating the analyst. Returns output char count."""
    out = model.invoke([
        ("system", "You are an analyst. Synthesize the evidence into a comparison "
                   "with a clear recommendation. Ground every claim in the evidence."),
        ("human", f"Goal: {GOAL}\n\nEvidence:\n{evidence}"),
    ])
    return len(out.content if isinstance(out.content, str) else str(out.content))


_ERROR_LABELS = (
    ("quota", "quota"), ("429", "429"), ("rate limit", "rate-limit"),
    ("503", "503"), ("tool", "no-tools"), ("function calling", "no-tools"),
)


def _label(exc: BaseException) -> str:
    """A short, comparable tag for why a call failed."""
    msg = str(exc).lower()
    for marker, label in _ERROR_LABELS:
        if marker in msg:
            return label
    return type(exc).__name__[:14]


def _fmt(times: list[float], err: Optional[str], trials: int) -> str:
    """Median latency, plus how many trials survived and why the rest did not."""
    if not times:
        return f"FAIL ({err or 'error'})"
    cell = f"{statistics.median(times):6.1f}s"
    if len(times) < trials:
        cell += f" {len(times)}/{trials}"
        if err:
            cell += f" {err}"
    return cell


def _time(fn, *a) -> tuple[Optional[float], Optional[str]]:
    """Run fn, returning (seconds, None) or (None, error-label)."""
    t0 = time.time()
    try:
        fn(*a)
        return time.time() - t0, None
    except Exception as exc:  # noqa: BLE001 - a benchmark reports failures, never raises
        return None, _label(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--evidence-units", type=int, default=14)
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    cfg = get_settings()
    evidence = _evidence(args.evidence_units)
    approx_tokens = len(evidence) // 4

    print(f"trials={args.trials}  analyst prompt ~{approx_tokens} tokens\n")
    print(f"  {'model':<38} {'plan (structured)':<22} {'analyst (large)':<22} verdict")
    print(f"  {'-'*38} {'-'*22} {'-'*22} {'-'*7}")

    for spec in args.models:
        provider, _ = split_model_id(spec, cfg.llm_provider)
        key = {"google": cfg.google_api_key, "groq": cfg.groq_api_key,
               "nvidia": cfg.nvidia_api_key}.get(provider, "")
        if not key:
            print(f"  {spec:<38} {'— no key —':<22} {'':<22} skipped")
            continue

        plan_t: list[float] = []
        ana_t: list[float] = []
        plan_err: Optional[str] = None
        ana_err: Optional[str] = None

        for _ in range(args.trials):
            try:
                model = get_chat_model(cfg, model_id=spec)
            except Exception as exc:  # noqa: BLE001 - report, never raise
                plan_err = _label(exc)
                break
            dt, err = _time(_plan_call, model)
            if dt is None:
                plan_err = err
            else:
                plan_t.append(dt)
            dt, err = _time(_analyst_call, model, evidence)
            if dt is None:
                ana_err = err
            else:
                ana_t.append(dt)
            time.sleep(0.5)

        ok = bool(plan_t) and bool(ana_t)
        verdict = "usable" if ok else ("no structured out" if not plan_t else "unusable")
        print(f"  {spec:<38} {_fmt(plan_t, plan_err, args.trials):<22} "
              f"{_fmt(ana_t, ana_err, args.trials):<22} {verdict}")

    print("\n  plan  = with_structured_output(Plan); a FAIL here rules the model out")
    print("  analyst = large-prompt call; where a tight TPM ceiling shows up")


if __name__ == "__main__":
    main()
