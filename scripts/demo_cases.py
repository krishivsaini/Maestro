"""Lightweight evaluation / demo runner over data/eval_cases.jsonl (§19).

Deliberately light: the point is to *demonstrate the system behaves* — including
under failure — not to produce a metrics framework. For each case it records
whether it completed, subtask count, critic iterations, recoveries, degradation,
and memory recall (hand-judged output quality is done separately).

Usage:
    python scripts/demo_cases.py list             # list all cases (no API calls)
    python scripts/demo_cases.py run <id>         # run one case live (needs GOOGLE_API_KEY)
    python scripts/demo_cases.py run all           # run every case live
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CASES_PATH = os.path.join(ROOT, "data", "eval_cases.jsonl")


def load_cases() -> list[dict]:
    with open(CASES_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_list() -> int:
    cases = load_cases()
    print(f"{len(cases)} eval cases:\n")
    for c in cases:
        cfg = f"  cfg={c['config']}" if c.get("config") else ""
        thr = f"  thread={c['thread_id']}(turn {c.get('turn')})" if c.get("thread_id") else ""
        print(f"  [{c['category']:16}] {c['id']:8} {c['goal'][:60]}{cfg}{thr}")
        print(f"                     -> {c['exercises']}")
    return 0


def _metrics(final: dict) -> dict:
    trace = final.get("trace", [])
    types = [e.event_type.value for e in trace]
    return {
        "status": final.get("status"),
        "subtasks": len(final.get("subtasks", [])),
        "evidence": len(final.get("evidence", [])),
        "critic_iterations": final.get("critic_iterations", 0),
        "recovery_attempts": final.get("recovery_attempts", 0),
        "critic_rejected": "critic_reject" in types,
        "recovered": "recovery" in types,
        "degraded_subtasks": sum(1 for s in final.get("subtasks", []) if s.status.value == "degraded"),
        "memory_written": "memory_write" in types,
        "memory_recalled": "memory_recall" in types,
    }


def cmd_run(which: str) -> int:
    from maestro.config import get_settings, reset_settings_cache
    from maestro.graph import build_default_agents, run_goal
    from maestro.memory import HashingEmbedder, LongTermMemory

    cases = load_cases()
    if which != "all":
        cases = [c for c in cases if c["id"] == which]
        if not cases:
            print(f"no case with id {which!r}")
            return 1

    shared_memory = LongTermMemory(embedder=HashingEmbedder())  # persists across cases/turns
    print(f"{'id':8} {'category':16} {'status':10} {'subtasks':>8} {'critic':>6} {'rec':>4} {'deg':>4} {'recall':>6}")
    print("-" * 74)

    for c in cases:
        # apply per-case config (MAESTRO_* env), rebuild settings
        applied = c.get("config", {}) or {}
        for k, v in applied.items():
            os.environ[k] = str(v)
        reset_settings_cache()
        agents = build_default_agents(get_settings())
        agents.memory = shared_memory
        try:
            final = run_goal(c["goal"], agents=agents, thread_id=c.get("thread_id"))
            m = _metrics(final)
            print(f"{c['id']:8} {c['category']:16} {m['status']:10} {m['subtasks']:>8} "
                  f"{m['critic_iterations']:>6} {m['recovery_attempts']:>4} "
                  f"{m['degraded_subtasks']:>4} {str(m['memory_recalled']):>6}")
        except Exception as exc:  # noqa: BLE001
            print(f"{c['id']:8} {c['category']:16} ERROR: {type(exc).__name__}: {exc}")
        finally:
            for k in applied:  # don't leak config into the next case
                os.environ.pop(k, None)
            reset_settings_cache()
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "run"):
        print(__doc__)
        return 1
    if sys.argv[1] == "list":
        return cmd_list()
    return cmd_run(sys.argv[2] if len(sys.argv) > 2 else "all")


if __name__ == "__main__":
    sys.exit(main())
