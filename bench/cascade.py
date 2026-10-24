"""Confidence cascade: the small model answers when its max probability clears a
threshold, otherwise the case escalates to the large model. Thresholds are swept on
the test set here only to draw the curve; pick the operating point on a holdout.

    uv run --with pyarrow python bench/cascade.py small.jsonl large.jsonl
"""

from __future__ import annotations

import json
import math
import sys

import pyarrow.parquet as pq

GOLD = {r["id"]: json.loads(r["gold"]) for r in pq.read_table("data/all_test.parquet").to_pylist()}


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        g = GOLD[r["id"]][r["question"]]["probabilities"]
        pp = {str(k).lower(): float(v) for k, v in r["probabilities"].items()}
        p = [max(pp.get(k, 0.0), 1e-12) for k in g]
        s = sum(p)
        out[(r["id"], r["question"])] = ([x / s for x in p], [float(g[k]) for k in g])
    return out


small, large = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(small)
print(f"{'thresh':>6s} {'to-large':>8s} {'acc':>7s} {'kl':>7s} {'brier':>7s}")
for th in [0.0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]:
    hits = kl = br = esc = 0
    for k in keys:
        p, g = small[k]
        if max(p) < th:
            p, esc = large[k][0], esc + 1
        hits += p.index(max(p)) == g.index(max(g))
        kl += sum(a * math.log(a / max(b, 1e-9)) for a, b in zip(g, p) if a > 0)
        br += sum((a - b) ** 2 for a, b in zip(p, g))
    n = len(keys)
    print(f"{th:6.2f} {esc/n:8.1%} {hits/n:7.4f} {kl/n:7.4f} {br/n:7.4f}")
