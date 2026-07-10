"""Regenerate graph.png (and graph.mmd) from the compiled StateGraph.

Building the graph structure does NOT call the LLM (agents build their models
lazily), so this runs without a GOOGLE_API_KEY. PNG rendering uses the mermaid.ink
API; if that is unreachable, graph.mmd (Mermaid source) is the fallback.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from maestro.graph import build_graph  # noqa: E402


def main() -> int:
    graph = build_graph()
    g = graph.get_graph()

    mermaid = g.draw_mermaid()
    with open(os.path.join(ROOT, "graph.mmd"), "w") as f:
        f.write(mermaid)
    print("wrote graph.mmd\n")
    print(mermaid)

    try:
        png = g.draw_mermaid_png()
        with open(os.path.join(ROOT, "graph.png"), "wb") as f:
            f.write(png)
        print(f"\nwrote graph.png ({len(png)} bytes)")
    except Exception as exc:  # noqa: BLE001
        print(f"\nPNG render unavailable ({type(exc).__name__}: {exc}); use graph.mmd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
