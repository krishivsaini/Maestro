"""Record a live run as a cassette, or refresh a cassette's expected result.

A cassette (data/cassettes/<name>.json) holds every model response and search result
of one live run plus what the run produced; tests/test_replays.py replays it offline
on every test run. See maestro/cassette.py.

Usage:
    python scripts/record_run.py "Compare solar and wind ..." --name solar-wind     # live
    python scripts/record_run.py "..." --name critic-loop --force-critic-reject 1   # live
    python scripts/record_run.py --refresh solar-wind     # offline: accept a deliberate change

``--refresh`` replays the recorded calls through the current code and stores what it
now produces as the expected result. Use it only when a behaviour change is intended;
the diff of the cassette then shows exactly what changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from maestro import cassette  # noqa: E402

CASSETTES = os.path.join(ROOT, "data", "cassettes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("goal", nargs="?")
    ap.add_argument("--name", help="cassette name (data/cassettes/<name>.json)")
    ap.add_argument("--note", default="", help="what this cassette exercises")
    ap.add_argument("--force-critic-reject", type=int, default=0, help="force the first N reviews to REJECT")
    ap.add_argument("--refresh", metavar="NAME", help="replay NAME offline and store its result as expected")
    args = ap.parse_args()
    os.makedirs(CASSETTES, exist_ok=True)

    if args.refresh:
        path = os.path.join(CASSETTES, f"{args.refresh}.json")
        data = cassette.load(path)
        new = cassette.summarize(cassette.replay(data))
        changed = {k: (data["expected"].get(k), v) for k, v in new.items() if data["expected"].get(k) != v}
        data["expected"] = new
        with open(path, "w") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f"refreshed {path}" + ("" if changed else " (no change)"))
        for k, (old, v) in changed.items():
            print(f"  {k}: {old} -> {v}")
        return 0

    if not (args.goal and args.name):
        ap.error("a goal and --name are required to record")
    from maestro.config import get_settings
    from maestro.graph import build_default_agents, run_goal

    settings = get_settings().model_copy(update={"force_critic_reject": args.force_critic_reject})
    book = cassette.Book()
    with cassette.recording(book):
        final = run_goal(args.goal, agents=build_default_agents(settings), settings=settings)
    path = os.path.join(CASSETTES, f"{args.name}.json")
    data = cassette.save(path, args.goal, settings, book, final, note=args.note)
    e = data["expected"]
    print(f"recorded {path}: {len(book.calls)} model calls, {len(book.searches)} searches")
    print(f"  status={e['status']} critic={e['critic']} replans={e['replans']} recoveries={e['recoveries']} "
          f"evidence={e['evidence']} cited={e['cited']}")

    # prove it replays before calling it a test
    again = cassette.summarize(cassette.replay(cassette.load(path)))
    diff = {k for k in e if e[k] != again[k]}
    print("  replays identically" if not diff else f"  WARNING: replay differs in {sorted(diff)}")
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
