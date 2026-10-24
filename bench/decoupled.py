"""Decoupled calibration: one set of logits, two temperatures, two outputs.

    probabilities  softmax(z / T_soft), T_soft fit to the soft gold     -> KL, Brier
    confidence     max softmax(z / T_hard), T_hard fit to the hard label -> ECE, deferral

Both temperatures are fit per question type on the holdout only. The board computes
ECE from max(probabilities), so a single distribution has to trade KL against ECE;
splitting the outputs removes the trade without retraining.

    uv run --with torch --with pyarrow python bench/decoupled.py results/C/logits.pt [calibration.json]

The optional second argument writes both temperature sets, which ship with the adapter
so anyone can reproduce `probabilities` and `confidence` from the raw logits.
"""

from __future__ import annotations

import json
import math
import sys

import pyarrow.parquet as pq
import torch

d = torch.load(sys.argv[1])
H, T = d["holdout"], d["test"]
GRID = torch.exp(torch.linspace(-2.5, 2.5, 201)).tolist()
GOLD = {r["id"]: json.loads(r["gold"]) for r in pq.read_table("data/all_test.parquet").to_pylist()}


def ce(zs, ts, t):
    return sum(float(-(torch.tensor(g) * torch.log_softmax(z / t, -1)).sum()) for z, g in zip(zs, ts))


def fit(hard):
    out = {}
    for qt in set(H["qtype"]):
        idx = [i for i, q in enumerate(H["qtype"]) if q == qt]
        zs = [H["logits"][i] for i in idx]
        ts = [H["target"][i] for i in idx]
        if hard:
            ts = [[float(j == max(range(len(g)), key=g.__getitem__)) for j in range(len(g))] for g in ts]
        out[qt] = min(GRID, key=lambda t: ce(zs, ts, t))
    return out


soft, hard = fit(False), fit(True)
print("T_soft", {d["types"][k]: round(v, 3) for k, v in soft.items()})
print("T_hard", {d["types"][k]: round(v, 3) for k, v in hard.items()})
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        json.dump({"fitted_on": "holdout split (train rows held out with seed 3407), never test",
                   "probabilities_temperature": {d["types"][k]: v for k, v in soft.items()},
                   "confidence_temperature": {d["types"][k]: v for k, v in hard.items()},
                   "usage": "probabilities = softmax(logits / probabilities_temperature[type]); "
                            "confidence = max(softmax(logits / confidence_temperature[type]))"},
                  f, indent=2)


def ece(rows):
    bins = {}
    for h, c in rows:
        bins.setdefault(min(int(c * 10), 9), []).append((h, c))
    return sum(len(b) / len(rows) * abs(sum(x for x, _ in b) / len(b) - sum(c for _, c in b) / len(b))
               for b in bins.values())


acc = kl = br = 0.0
same, split = [], []
for z, m, qt in zip(T["logits"], T["meta"], T["qtype"]):
    g = GOLD[m["id"]][m["question"]]["probabilities"]
    gv = [float(g[o]) for o in m["options"]]
    p = torch.softmax(z / soft[qt], -1).tolist()
    hit = p.index(max(p)) == gv.index(max(gv))
    acc += hit
    kl += sum(a * math.log(a / max(b, 1e-9)) for a, b in zip(gv, p) if a > 0)
    br += sum((a - b) ** 2 for a, b in zip(p, gv))
    same.append((hit, max(p)))
    split.append((hit, float(torch.softmax(z / hard[qt], -1).max())))
n = len(same)
print(f"accuracy {acc/n:.4f} | KL {kl/n:.4f} | Brier {br/n:.4f} | "
      f"ECE single-output {ece(same):.4f} -> decoupled confidence {ece(split):.4f}")
