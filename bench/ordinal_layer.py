"""Ordinal neighbour-smoothing output layer for `score` questions.

Gold is the mean of three teacher samples, so its mass spreads to adjacent levels.
After the soft temperature, each level keeps (1 - a) of its mass and passes a/2 to
each neighbour (edges reflect back). One scalar a, fit on the holdout only.

    uv run --with torch --with pyarrow python bench/ordinal_layer.py results/C/logits.pt
"""

from __future__ import annotations

import json
import math
import sys

import pyarrow.parquet as pq
import torch

d = torch.load(sys.argv[1])
H, T = d["holdout"], d["test"]
SCORE = d["types"].index("score")
GOLD = {r["id"]: json.loads(r["gold"]) for r in pq.read_table("data/all_test.parquet").to_pylist()}
GRID = torch.exp(torch.linspace(-1.5, 2.5, 161)).tolist()


def smooth(p, a):
    k = len(p)
    out = [0.0] * k
    for i, v in enumerate(p):
        out[i] += (1 - a) * v
        for j in (i - 1, i + 1):
            out[j if 0 <= j < k else i] += a / 2 * v
    return out


def kl(g, p):
    return sum(x * math.log(x / max(y, 1e-12)) for x, y in zip(g, p) if x > 0)


hidx = [i for i, q in enumerate(H["qtype"]) if q == SCORE]
hz, hg = [H["logits"][i] for i in hidx], [H["target"][i] for i in hidx]
t = min(GRID, key=lambda t: sum(kl(g, torch.softmax(z / t, -1).tolist()) for z, g in zip(hz, hg)))
hp = [torch.softmax(z / t, -1).tolist() for z in hz]
a = min([i / 100 for i in range(0, 61)], key=lambda a: sum(kl(g, smooth(p, a)) for p, g in zip(hp, hg)))
print(f"holdout: T_soft {t:.3f}, alpha {a:.2f}")

for name, alpha in (("softmax only", 0.0), ("+ ordinal layer", a)):
    n = hit = k_sum = br = mae = 0.0
    for z, m, qt in zip(T["logits"], T["meta"], T["qtype"]):
        if qt != SCORE:
            continue
        g = GOLD[m["id"]][m["question"]]["probabilities"]
        gv = [float(g[o]) for o in m["options"]]
        p = smooth(torch.softmax(z / t, -1).tolist(), alpha)
        n += 1
        hit += p.index(max(p)) == gv.index(max(gv))
        k_sum += kl(gv, p)
        br += sum((x - y) ** 2 for x, y in zip(p, gv))
        mae += abs(sum(i * x for i, x in enumerate(p)) - sum(i * x for i, x in enumerate(gv)))
    print(f"  {name:16s} score questions n={int(n)}: acc {hit/n:.4f} | KL {k_sum/n:.4f} | "
          f"Brier {br/n:.4f} | exp-MAE {mae/n:.4f}")
