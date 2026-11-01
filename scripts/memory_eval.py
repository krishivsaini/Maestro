"""Recall evaluation for long-term memory (live for Gemini embedders).

For each thread in data/memory_eval.json, every query is a follow-up phrased in
different words from the finding it should recall. Reports how often that finding
ranks first (top-1) and the mean reciprocal rank (MRR: 1 = always first, 0.5 = second).

    python scripts/memory_eval.py hashing
    python scripts/memory_eval.py gemini --model gemini-embedding-001
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("embedder", choices=["hashing", "gemini"])
    ap.add_argument("--model", help="Gemini embedding model id")
    args = ap.parse_args()

    from maestro.memory import HashingEmbedder
    from maestro.memory.embedders import GeminiEmbedder

    emb = HashingEmbedder() if args.embedder == "hashing" else GeminiEmbedder(model=args.model)
    docs = getattr(emb, "embed_documents", emb.embed)
    query = getattr(emb, "embed_query", lambda t: emb.embed([t])[0])
    data = json.load(open(os.path.join(ROOT, "data", "memory_eval.json")))

    ranks = []
    for t in data["threads"]:
        f = unit(docs(t["findings"]))
        for q, expected in t["queries"]:
            scores = f @ unit(query(q))
            order = list(np.argsort(-scores))
            rank = order.index(expected) + 1
            ranks.append(rank)
            print(f"{'ok  ' if rank == 1 else f'#{rank}  '} {q[:70]:70} -> {t['findings'][order[0]][:50]}")
    top1 = sum(r == 1 for r in ranks)
    mrr = float(np.mean([1 / r for r in ranks]))
    label = args.embedder + (f" ({emb.model})" if args.embedder == "gemini" else "")
    print(f"\n{label}: top-1 {top1}/{len(ranks)}, MRR {mrr:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
