"""Average several adapters of one base, then calibrate and cascade the result.

Each member's probabilities temperature is fitted on the shared holdout (seed 3407), the
members' probabilities are averaged, and the confidence temperature is fitted on the
averaged holdout distribution against the hard label. With --small, it also picks the
0.8B-to-ensemble cascade threshold on the holdout. Nothing is fitted on test.

    uv run --with torch --with pyarrow python bench/ensemble.py \
        results/final/large/logits.pt results/final/lowlr/logits.pt \
        --small results/final/small/logits.pt --out results/final/pramana2b
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import pyarrow.parquet as pq
import torch

GRID = torch.exp(torch.linspace(-2.5, 2.5, 201)).tolist()
GOLD = {r["id"]: json.loads(r["gold"]) for r in pq.read_table("data/all_test.parquet").to_pylist()}


def fit(rows, targets, qtypes):
    """Per question type temperature minimising cross-entropy of softmax(rows / T)."""
    out = {}
    for qt in set(qtypes):
        idx = [i for i, q in enumerate(qtypes) if q == qt]
        out[qt] = min(GRID, key=lambda t: sum(
            float(-(torch.tensor(targets[i]) * torch.log_softmax(rows[i] / t, -1)).sum())
            for i in idx))
    return out


def hard(targets):
    return [[float(j == max(range(len(g)), key=g.__getitem__)) for j in range(len(g))]
            for g in targets]


def member_probs(d, split, temps):
    return [torch.softmax(z / temps[q], -1) for z, q in zip(d[split]["logits"], d[split]["qtype"])]


def ece(rows):
    bins = {}
    for h, c in rows:
        bins.setdefault(min(int(c * 10), 9), []).append((h, c))
    return sum(len(b) / len(rows) * abs(sum(x for x, _ in b) / len(b) - sum(c for _, c in b) / len(b))
               for b in bins.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+")
    ap.add_argument("--small", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = [torch.load(p) for p in a.members]
    ref = ds[0]
    for d in ds[1:]:
        assert d["holdout"]["target"] == ref["holdout"]["target"], "members need the same holdout"
        assert [(m["id"], m["question"]) for m in d["test"]["meta"]] == \
               [(m["id"], m["question"]) for m in ref["test"]["meta"]]
    types = ref["types"]
    ht, hq = ref["holdout"]["target"], ref["holdout"]["qtype"]
    soft = [fit(d["holdout"]["logits"], ht, hq) for d in ds]

    def averaged(split):
        per = [member_probs(d, split, t) for d, t in zip(ds, soft)]
        return [sum(ps) / len(ps) for ps in zip(*per)]

    hp, tp = averaged("holdout"), averaged("test")
    # Confidence: temperature on the log of the averaged distribution, fitted to the hard label.
    conf_t = fit([torch.log(p.clamp_min(1e-12)) for p in hp], hard(ht), hq)
    tq = ref["test"]["qtype"]
    conf = [float(torch.softmax(torch.log(p.clamp_min(1e-12)) / conf_t[q], -1).max())
            for p, q in zip(tp, tq)]

    with open(out / "preds.jsonl", "w") as f:
        for p, c, m in zip(tp, conf, ref["test"]["meta"]):
            f.write(json.dumps({"id": m["id"], "question": m["question"],
                                "probabilities": dict(zip(m["options"], p.tolist())),
                                "confidence": c}) + "\n")
    json.dump({"fitted_on": "holdout split (train rows held out with seed 3407), never test",
               "members": [str(pathlib.Path(p).parent.name) for p in a.members],
               "member_probabilities_temperature": [{types[k]: v for k, v in t.items()} for t in soft],
               "confidence_temperature": {types[k]: v for k, v in conf_t.items()},
               "usage": "p_i = softmax(logits_i / member_probabilities_temperature[i][type]); "
                        "probabilities = mean_i p_i; "
                        "confidence = max(softmax(log(probabilities) / confidence_temperature[type]))"},
              open(out / "calibration.json", "w"), indent=2)

    rows = []
    for p, c, m in zip(tp, conf, ref["test"]["meta"]):
        g = GOLD[m["id"]][m["question"]]["probabilities"]
        gv = [float(g[o]) for o in m["options"]]
        rows.append((int(p.argmax()) == gv.index(max(gv)), c))
    print(f"confidence ECE on test: {ece(rows):.4f}")

    if a.small:
        s = torch.load(a.small)
        assert s["holdout"]["target"] == ht
        st = fit(s["holdout"]["logits"], ht, hq)
        sh, stp = member_probs(s, "holdout", st), member_probs(s, "test", st)
        am = lambda v: int(v.argmax())
        gold_h = [g.index(max(g)) for g in ht]
        acc = lambda ps: sum(am(p) == g for p, g in zip(ps, gold_h)) / len(ps)
        target = acc(hp)
        th = next(t / 100 for t in range(101)
                  if acc([e if float(x.max()) < t / 100 else x for x, e in zip(sh, hp)]) >= target)
        n = 0
        with open(out / "cascade.jsonl", "w") as f:
            for x, e, m in zip(stp, tp, ref["test"]["meta"]):
                use = e if float(x.max()) < th else x
                n += use is e
                f.write(json.dumps({"id": m["id"], "question": m["question"],
                                    "probabilities": dict(zip(m["options"], use.tolist()))}) + "\n")
        print(f"cascade threshold {th:.2f} (holdout), sends {n / len(stp):.1%} of test to the 2B")


if __name__ == "__main__":
    main()
