"""Score Typed Decisions predictions with the benchmark's own metric code.

Metrics are byte-for-byte the definitions in
LocalLLaMA/typed-decisions bench/score_with_jobs.py so numbers are comparable to
the published leaderboard:
    accuracy       argmax(pred) == argmax(gold)
    kl_from_gold   sum(g * log(g / p))          lower is better
    brier          sum((p - g) ** 2)            lower is better
    ece            10-bin |acc - conf|, conf = max(p)

Input: predictions.jsonl, one line per decision:
    {"id": <case id>, "question": <qid>, "probabilities": {opt: p, ...}}

Usage:
    uv run --with pyarrow python bench/score.py results/preds.jsonl
    uv run --with pyarrow python bench/score.py results/preds.jsonl --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_PARQUET = ROOT / "data" / "all_test.parquet"
N_BINS = 10


def load_gold() -> dict[str, dict]:
    """case id -> {workflow, n_questions, gold{qid: {...}}}"""
    import pyarrow.parquet as pq

    out = {}
    for row in pq.read_table(TEST_PARQUET).to_pylist():
        out[row["id"]] = {
            "workflow": row["workflow"],
            "gold": json.loads(row["gold"]),
        }
    return out


def norm_key(k) -> str:
    """Gold uses "true"/"false" strings; engines emit yes/no or bools."""
    s = str(k).strip().lower()
    # Only noul yes/no needs normalising. Ordered `score` questions use the
    # option keys "0".."3", so mapping "0"/"1" to true/false silently destroys
    # three quarters of every score question. bench/analyze.py never had this
    # mapping, which is why it reproduced Decider-4B's and sysone's own reports.
    return {"yes": "true", "no": "false"}.get(s, s)


def score_one(pred_probs: dict, gold: dict) -> tuple[bool, float, float, float, str]:
    opts = list(gold["probabilities"].keys())
    g = [float(gold["probabilities"][o]) for o in opts]
    pp = {norm_key(k): float(v) for k, v in pred_probs.items()}
    p = [pp.get(norm_key(o), 0.0) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
    am_p = p.index(max(p))
    am_g = g.index(max(g))
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return am_p == am_g, kl, brier, max(p), gold.get("type", "?")


class Acc:
    def __init__(self):
        self.hits: list[bool] = []
        self.kls: list[float] = []
        self.briers: list[float] = []
        self.confs: list[float] = []

    def add(self, hit, kl, brier, conf):
        self.hits.append(hit)
        self.kls.append(kl)
        self.briers.append(brier)
        self.confs.append(conf)

    @property
    def n(self):
        return len(self.hits)

    def summary(self) -> dict:
        n = self.n
        if not n:
            return {}
        bin_hits, bin_total, bin_conf = [0] * N_BINS, [0] * N_BINS, [0.0] * N_BINS
        for hit, conf in zip(self.hits, self.confs):
            b = min(int(conf * N_BINS), N_BINS - 1)
            bin_total[b] += 1
            bin_hits[b] += int(hit)
            bin_conf[b] += conf
        ece = 0.0
        for b in range(N_BINS):
            if bin_total[b]:
                ece += bin_total[b] / n * abs(bin_hits[b] / bin_total[b] - bin_conf[b] / bin_total[b])
        return {
            "n_decisions": n,
            "accuracy": sum(self.hits) / n,
            "kl_from_gold": sum(self.kls) / n,
            "brier": sum(self.briers) / n,
            "ece": ece,
            "mean_confidence": sum(self.confs) / n,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--json", default=None, help="write full breakdown here")
    args = ap.parse_args()

    gold = load_gold()
    overall = Acc()
    by_wf = collections.defaultdict(Acc)
    by_q = collections.defaultdict(Acc)
    seen = set()

    for line in pathlib.Path(args.preds).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid, qid = rec["id"], rec["question"]
        if cid not in gold:
            raise SystemExit(f"unknown case id {cid!r}")
        g = gold[cid]["gold"].get(qid)
        if g is None:
            raise SystemExit(f"unknown question {qid!r} for {cid!r}")
        hit, kl, brier, conf, qtype = score_one(rec["probabilities"], g)
        overall.add(hit, kl, brier, conf)
        by_wf[gold[cid]["workflow"]].add(hit, kl, brier, conf)
        by_q[f"{gold[cid]['workflow']}/{qid}"].add(hit, kl, brier, conf)
        seen.add((cid, qid))

    expected = sum(len(v["gold"]) for v in gold.values())
    missing = expected - len(seen)
    print(f"scored {overall.n}/{expected} decisions from {len(gold)} cases"
          + (f"  [WARN missing {missing}]" if missing else ""))

    o = overall.summary()
    print("\nALL (board-comparable)")
    print(f"  accuracy {o['accuracy']:.4f} | kl {o['kl_from_gold']:.4f} | "
          f"brier {o['brier']:.4f} | ece {o['ece']:.4f}")

    print("\nBY WORKFLOW")
    print(f"  {'workflow':32s} {'acc':>7s} {'kl':>7s} {'brier':>7s} {'ece':>7s}")
    for wf in sorted(by_wf):
        s = by_wf[wf].summary()
        print(f"  {wf:32s} {s['accuracy']:7.4f} {s['kl_from_gold']:7.4f} "
              f"{s['brier']:7.4f} {s['ece']:7.4f}")

    print("\nBY QUESTION")
    print(f"  {'workflow/question':48s} {'acc':>7s} {'kl':>7s} {'ece':>7s}")
    for q in sorted(by_q):
        s = by_q[q].summary()
        print(f"  {q:48s} {s['accuracy']:7.4f} {s['kl_from_gold']:7.4f} {s['ece']:7.4f}")

    if args.json:
        out = {
            "overall": o,
            "by_workflow": {k: v.summary() for k, v in sorted(by_wf.items())},
            "by_question": {k: v.summary() for k, v in sorted(by_q.items())},
            "coverage": {"scored": overall.n, "expected": expected, "missing": missing},
        }
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
