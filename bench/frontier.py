"""Two checks a public claim needs.

1. Paired bootstrap over test cases: Pramana minus each reference, 95% CI per metric.
2. Temperature frontier: rescale every model's distribution (p^(1/T), renormalised,
   the same as dividing logits by T) and trace KL against top-label ECE. If one
   temperature cannot minimise both, the board's metrics pull against each other.

    uv run --with pyarrow python bench/frontier.py results/C/preds_temper.jsonl
"""

from __future__ import annotations

import collections
import json
import math
import random
import sys

import pyarrow.parquet as pq

GOLD = {r["id"]: json.loads(r["gold"]) for r in pq.read_table("data/all_test.parquet").to_pylist()}
MODELS = {"Pramana-2B": sys.argv[1], "Decider-4B": "results/norm/decider4b.jsonl",
          "sysone-0.4B": "results/norm/sysone04b.jsonl", "Arbiter-4B": "results/norm/arbiter4b.jsonl"}


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


def temper(p, t):
    w = [math.exp(math.log(x) / t) for x in p]
    s = sum(w)
    return [x / s for x in w]


def per_decision(p, g):
    am = p.index(max(p)) == g.index(max(g))
    kl = sum(a * math.log(a / max(b, 1e-9)) for a, b in zip(g, p) if a > 0)
    return am, kl, sum((a - b) ** 2 for a, b in zip(p, g)), max(p)


def ece(rows):
    bins = collections.defaultdict(list)
    for hit, conf in rows:
        bins[min(int(conf * 10), 9)].append((hit, conf))
    n = len(rows)
    return sum(len(b) / n * abs(sum(h for h, _ in b) / len(b) - sum(c for _, c in b) / len(b))
               for b in bins.values())


def metrics(dec):
    rows = [per_decision(p, g) for p, g in dec]
    n = len(rows)
    return {"acc": sum(r[0] for r in rows) / n, "kl": sum(r[1] for r in rows) / n,
            "brier": sum(r[2] for r in rows) / n, "ece": ece([(r[0], r[3]) for r in rows])}


data = {m: load(p) for m, p in MODELS.items()}
keys = sorted(data["Pramana-2B"])
cases = sorted({k[0] for k in keys})
by_case = collections.defaultdict(list)
for k in keys:
    by_case[k[0]].append(k)

print("== paired bootstrap, Pramana-2B minus reference, 2000 resamples of 400 cases, 95% CI")
rng = random.Random(0)
draws = [[rng.choice(cases) for _ in cases] for _ in range(2000)]
out = {"bootstrap": {}, "frontier": {}}
for ref in [m for m in MODELS if m != "Pramana-2B"]:
    diffs = collections.defaultdict(list)
    for d in draws:
        ks = [k for c in d for k in by_case[c]]
        a = metrics([data["Pramana-2B"][k] for k in ks])
        b = metrics([data[ref][k] for k in ks])
        for m in a:
            diffs[m].append(a[m] - b[m])
    row = {}
    for m, v in diffs.items():
        v.sort()
        row[m] = (round(v[50], 4), round(v[1949], 4))
    out["bootstrap"][ref] = row
    print(f"  vs {ref:12s}", "  ".join(f"{m} [{lo:+.4f},{hi:+.4f}]" for m, (lo, hi) in row.items()))

print("\n== temperature frontier (T applied post hoc to each model's probabilities)")
for m, dec in data.items():
    curve = []
    for t in [0.33, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]:
        r = metrics([(temper(p, t), g) for p, g in dec.values()])
        curve.append({"T": t, **{k: round(v, 4) for k, v in r.items()}})
    out["frontier"][m] = curve
    bk = min(curve, key=lambda r: r["kl"])
    be = min(curve, key=lambda r: r["ece"])
    print(f"  {m:12s} min-KL at T={bk['T']}: kl {bk['kl']:.4f} ece {be and bk['ece']:.4f}"
          f" | min-ECE at T={be['T']}: kl {be['kl']:.4f} ece {be['ece']:.4f} brier {be['brier']:.4f}")
json.dump(out, open("results/frontier.json", "w"), indent=1)
