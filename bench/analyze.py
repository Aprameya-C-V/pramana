"""Evaluation suite for decision models, beyond accuracy.

The published Typed Decisions board reports accuracy, KL, Brier and top-label ECE.
That is enough to rank argmax accuracy and not much else. This script scores the
same 2,000 decisions on four additional axes, so models can be compared on what a
decision model is actually used for:

  selective     accuracy at a fixed coverage, when low-confidence decisions are
                deferred to a human. Nobody reports this on the board.
  conformal     distribution-free coverage guarantee: split-conformal sets at a
                target coverage, with the mean set size that buys it.
  ordinal       on `score` questions, the expected-score error and the level
                error. A model can win the argmax and still be a level off.
  decomposition Murphy reliability / resolution / uncertainty on the Brier score,
                which separates calibration from informativeness.

Core metrics (accuracy, KL, Brier, ECE) reproduce bench/score_with_jobs.py.

Usage:
    uv run --with pyarrow python bench/analyze.py \
        --model Decider-4B results/others/norm/decider4b.jsonl \
        --model sysone-0.4B results/others/norm/sysone04b.jsonl \
        --model ours results/ours/preds.jsonl --json results/analysis.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_PARQUET = ROOT / "data" / "all_test.parquet"
COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.9)
CONFORMAL_ALPHA = 0.10
CAL_FRACTION = 0.5
SEED = 3407


def norm_key(k) -> str:
    s = str(k).strip().lower()
    return {"yes": "true", "no": "false"}.get(s, s)


def load_gold() -> dict[str, dict]:
    import pyarrow.parquet as pq

    out = {}
    for row in pq.read_table(TEST_PARQUET).to_pylist():
        out[row["id"]] = {"workflow": row["workflow"], "gold": json.loads(row["gold"])}
    return out


def load_preds(path: pathlib.Path) -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["id"], r["question"])] = {norm_key(k): float(v) for k, v in r["probabilities"].items()}
    return out


class Decision:
    __slots__ = ("cid", "qid", "workflow", "qtype", "opts", "p", "g", "gold_idx", "pred_idx")

    def __init__(self, cid, qid, workflow, qtype, opts, p, g):
        self.cid, self.qid, self.workflow, self.qtype = cid, qid, workflow, qtype
        self.opts, self.p, self.g = opts, p, g
        self.gold_idx = g.index(max(g))
        self.pred_idx = p.index(max(p))

    @property
    def conf(self):
        return max(self.p)

    @property
    def hit(self):
        return self.pred_idx == self.gold_idx

    @property
    def kl(self):
        return sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(self.g, self.p) if gi > 0)

    @property
    def brier(self):
        return sum((pi - gi) ** 2 for pi, gi in zip(self.p, self.g))

    def expected(self, dist):
        return sum(i * v for i, v in enumerate(dist))


def build(models: dict[str, dict], gold: dict[str, dict]) -> dict[str, list[Decision]]:
    out = {}
    for name, preds in models.items():
        rows = []
        for cid, meta in gold.items():
            for qid, gq in meta["gold"].items():
                probs = preds.get((cid, qid))
                if probs is None:
                    continue
                opts = list(gq["probabilities"].keys())
                g = [float(gq["probabilities"][o]) for o in opts]
                p = [max(0.0, float(probs.get(norm_key(o), 0.0))) for o in opts]
                s = sum(p)
                p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
                rows.append(Decision(cid, qid, meta["workflow"], gq.get("type", "?"), opts, p, g))
        out[name] = rows
    return out


def ece(decisions: list[Decision], n_bins: int = 10) -> float:
    n = len(decisions)
    if not n:
        return float("nan")
    bh, bt, bc = [0] * n_bins, [0] * n_bins, [0.0] * n_bins
    for d in decisions:
        b = min(int(d.conf * n_bins), n_bins - 1)
        bt[b] += 1
        bh[b] += int(d.hit)
        bc[b] += d.conf
    return sum(bt[b] / n * abs(bh[b] / bt[b] - bc[b] / bt[b]) for b in range(n_bins) if bt[b])


def core(decisions: list[Decision]) -> dict:
    n = len(decisions)
    return {
        "n": n,
        "accuracy": sum(d.hit for d in decisions) / n,
        "kl_from_gold": sum(d.kl for d in decisions) / n,
        "brier": sum(d.brier for d in decisions) / n,
        "ece_10": ece(decisions, 10),
        "ece_15": ece(decisions, 15),
    }


def selective(decisions: list[Decision]) -> dict:
    """Accuracy when the least confident share of decisions is deferred."""
    ranked = sorted(decisions, key=lambda d: d.conf, reverse=True)
    n = len(ranked)
    out = {}
    for c in COVERAGES:
        k = max(1, int(round(c * n)))
        kept = ranked[:k]
        out[f"acc@{int(c*100)}"] = sum(d.hit for d in kept) / len(kept)
    # AUC of accuracy over the full coverage sweep (higher = better ranking of confidence)
    steps = 20
    auc = 0.0
    for i in range(1, steps + 1):
        k = max(1, int(round(i / steps * n)))
        auc += sum(d.hit for d in ranked[:k]) / k
    out["selective_auc"] = auc / steps
    return out


# Questions that literally ask whether a human should look at this case. Only two
# of the four workflows expose one; invoice_processing and security_incidents use
# their noul questions for factual properties instead, so the gate comparison is
# scoped to the workflows that have a real review question.
REVIEW_QIDS = {"needs_review", "needs_human"}


def gate_by_case(decisions: list[Decision]) -> dict[str, float]:
    """case id -> probability that this case needs a human."""
    out: dict[str, float] = {}
    for d in decisions:
        if d.qid in REVIEW_QIDS:
            if "true" in d.opts:
                out[d.cid] = d.p[d.opts.index("true")]
            else:
                out[d.cid] = max(d.p)
    return out


def selective_gate(decisions: list[Decision]) -> dict:
    """Answer the cases the model thinks need no human, ranked by its own gate.

    Compared with `selective`, which ranks by max probability. Same coverage
    levels, different signal.
    """
    gate = gate_by_case(decisions)
    if not gate:
        return {}
    scoped = [d for d in decisions if d.cid in gate]
    ranked = sorted(scoped, key=lambda d: gate[d.cid])  # low review prob = answer first
    n = len(ranked)
    out = {"n_decisions": n, "n_cases": len(gate)}
    for c in COVERAGES:
        k = max(1, int(round(c * n)))
        out[f"acc@{int(c*100)}"] = sum(d.hit for d in ranked[:k]) / k
    return out


def conformal_mondrian(decisions: list[Decision], alpha: float = CONFORMAL_ALPHA) -> dict:
    """Split-conformal coverage per question type.

    Pooled conformal hides a type that is badly under-covered inside a good
    average. Mondrian splits the guarantee by qtype.
    """
    out = {}
    for qtype in sorted({d.qtype for d in decisions}):
        sub = [d for d in decisions if d.qtype == qtype]
        r = conformal(sub, alpha)
        if r:
            out[qtype] = r
    return out


def cost_optimal(decisions: list[Decision], ratios: tuple[int, ...] = (5, 10, 20)) -> dict:
    """The coverage that minimises expected cost, for a given error:review cost ratio.

    Answering a case automatically and getting it wrong costs `ratio`; sending it
    to a human costs 1. Sweeping coverage with the confidence ranking gives the
    operating point, which is the decision a team actually has to make. Reporting
    accuracy at an arbitrary 80% coverage does not answer it.
    """
    ranked = sorted(decisions, key=lambda d: d.conf, reverse=True)
    n = len(ranked)
    out = {}
    for ratio in ratios:
        best = {"coverage": 0.0, "cost": float("inf"), "accuracy": 1.0}
        for k in range(1, n + 1):
            kept = ranked[:k]
            c = k / n
            acc = sum(d.hit for d in kept) / k
            cost = c * (1 - acc) * ratio + (1 - c) * 1.0
            if cost < best["cost"]:
                best = {"coverage": c, "cost": cost, "accuracy": acc}
        out[f"{ratio}:1"] = {k: round(v, 4) for k, v in best.items()}
    return out


def conformal(decisions: list[Decision], alpha: float = CONFORMAL_ALPHA) -> dict:
    """Split-conformal sets for the gold argmax, at coverage 1 - alpha."""
    keys = sorted({(d.cid, d.qid) for d in decisions})
    rng = random.Random(SEED)
    rng.shuffle(keys)
    cut = int(len(keys) * CAL_FRACTION)
    cal_keys, eval_keys = set(keys[:cut]), set(keys[cut:])

    by_key = {(d.cid, d.qid): d for d in decisions}
    scores = [1.0 - by_key[k].p[by_key[k].gold_idx] for k in cal_keys]
    scores.sort()
    if not scores:
        return {}
    rank = min(len(scores) - 1, math.ceil((len(scores) + 1) * (1 - alpha)) - 1)
    qhat = scores[rank]

    covered, sizes = 0, 0
    n = 0
    for k in eval_keys:
        d = by_key[k]
        # include options whose nonconformity score is <= qhat; always keep the argmax
        s = set(i for i, pi in enumerate(d.p) if 1.0 - pi <= qhat)
        s.add(d.pred_idx)
        covered += int(d.gold_idx in s)
        sizes += len(s) / len(d.opts)
        n += 1
    return {
        "target_coverage": 1 - alpha,
        "coverage": covered / n,
        "mean_set_fraction": sizes / n,
        "qhat": qhat,
        "n_eval": n,
    }


def ordinal(decisions: list[Decision]) -> dict:
    """`score` questions only: expected-level error and level error."""
    sc = [d for d in decisions if d.qtype == "score"]
    if not sc:
        return {}
    exp_err = [abs(d.expected(d.p) - d.expected(d.g)) for d in sc]
    lvl_err = [abs(d.pred_idx - d.gold_idx) for d in sc]
    # monotonicity: mass should lean toward the gold argmax level, not away from it
    inv = 0
    for d in sc:
        far = [i for i in range(len(d.p)) if abs(i - d.gold_idx) >= 2]
        if far and max(d.p[i] for i in far) > d.p[d.gold_idx]:
            inv += 1
    return {
        "n": len(sc),
        "expected_score_mae": sum(exp_err) / len(sc),
        "level_error": sum(lvl_err) / len(sc),
        "far_level_leak_rate": inv / len(sc),
    }


def noul(decisions: list[Decision]) -> dict:
    nl = [d for d in decisions if d.qtype == "noul"]
    if not nl:
        return {}
    b = []
    correct = 0
    for d in nl:
        true_p = d.p[d.opts.index("true")] if "true" in d.opts else d.p[1]
        gold_true = d.g[d.opts.index("true")] if "true" in d.opts else d.g[1]
        b.append((true_p - gold_true) ** 2)
        correct += int((true_p >= 0.5) == (gold_true >= 0.5))
    return {"n": len(nl), "brier": sum(b) / len(b), "acc@0.5": correct / len(nl)}


def decomposition(decisions: list[Decision], n_bins: int = 10) -> dict:
    """Murphy decomposition of the top-label Brier score."""
    n = len(decisions)
    if not n:
        return {}
    bh, bt, bc = [0] * n_bins, [0] * n_bins, [0.0] * n_bins
    for d in decisions:
        b = min(int(d.conf * n_bins), n_bins - 1)
        bt[b] += 1
        bh[b] += int(d.hit)
        bc[b] += d.conf
    acc = sum(d.hit for d in decisions) / n
    rel = res = 0.0
    for b in range(n_bins):
        if bt[b]:
            a_b = bh[b] / bt[b]
            c_b = bc[b] / bt[b]
            rel += bt[b] / n * (a_b - c_b) ** 2
            res += bt[b] / n * (a_b - acc) ** 2
    return {"reliability": rel, "resolution": res, "uncertainty": acc * (1 - acc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=2, action="append", metavar=("NAME", "PREDS"),
                    required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    gold = load_gold()
    models = {name: load_preds(pathlib.Path(path)) for name, path in args.model}
    built = build(models, gold)

    report = {}
    for name, decisions in built.items():
        r = {
            "coverage": len(decisions),
            "core": core(decisions),
            "selective": selective(decisions),
            "conformal": conformal(decisions),
            "ordinal": ordinal(decisions),
            "noul": noul(decisions),
            "decomposition": decomposition(decisions),
            "selective_gate": selective_gate(decisions),
            "conformal_mondrian": conformal_mondrian(decisions),
            "cost_optimal": cost_optimal(decisions),
            "by_workflow": {},
            "by_question": {},
        }
        for wf in sorted({d.workflow for d in decisions}):
            sub = [d for d in decisions if d.workflow == wf]
            r["by_workflow"][wf] = {**core(sub), "acc@80": selective(sub).get("acc@80")}
        for wf, qid in sorted({(d.workflow, d.qid) for d in decisions}):
            sub = [d for d in decisions if d.workflow == wf and d.qid == qid]
            r["by_question"][f"{wf}/{qid}"] = core(sub)
        report[name] = r

    exp = sum(len(v["gold"]) for v in gold.values())
    print(f"expected decisions per model: {exp}")
    for name, r in report.items():
        if r["coverage"] != exp:
            print(f"  WARN {name}: {r['coverage']} decisions")

    def table(title, cols):
        print(f"\n### {title}")
        print("| model | " + " | ".join(c[0] for c in cols) + " |")
        print("|" + "|".join("---" for _ in range(len(cols) + 1)) + "|")
        for name, r in report.items():
            row = []
            for _, fn in cols:
                v = fn(r)
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            print("| " + name + " | " + " | ".join(row) + " |")

    table("Core (board metrics)", [
        ("accuracy", lambda r: r["core"]["accuracy"]),
        ("KL", lambda r: r["core"]["kl_from_gold"]),
        ("Brier", lambda r: r["core"]["brier"]),
        ("ECE10", lambda r: r["core"]["ece_10"]),
        ("ECE15", lambda r: r["core"]["ece_15"]),
    ])
    table("Selective (accuracy at coverage)", [
        ("acc@50", lambda r: r["selective"]["acc@50"]),
        ("acc@70", lambda r: r["selective"]["acc@70"]),
        ("acc@80", lambda r: r["selective"]["acc@80"]),
        ("acc@90", lambda r: r["selective"]["acc@90"]),
        ("sel-AUC", lambda r: r["selective"]["selective_auc"]),
    ])
    table("Conformal (90% target)", [
        ("coverage", lambda r: r["conformal"]["coverage"]),
        ("mean set", lambda r: r["conformal"]["mean_set_fraction"]),
    ])
    table("Score questions", [
        ("exp-MAE", lambda r: r["ordinal"]["expected_score_mae"]),
        ("level err", lambda r: r["ordinal"]["level_error"]),
        ("far leak", lambda r: r["ordinal"]["far_level_leak_rate"]),
    ])
    table("Brier decomposition", [
        ("reliability", lambda r: r["decomposition"]["reliability"]),
        ("resolution", lambda r: r["decomposition"]["resolution"]),
        ("uncertainty", lambda r: r["decomposition"]["uncertainty"]),
    ])

    # --- novelty: learned gate versus confidence thresholding ---
    print("\n### Deferral policy (the model's own review question vs max probability)")
    print("| model | scope decisions | conf acc@70 | gate acc@70 | conf acc@80 | gate acc@80 |")
    print("|---|---|---|---|---|---|")
    for name, r in report.items():
        g = r.get("selective_gate") or {}
        if not g:
            print(f"| {name} | no review question | — | — | — | — |")
            continue
        s = r["selective"]
        print(f"| {name} | {g['n_decisions']} | {s['acc@70']:.4f} | {g['acc@70']:.4f} | "
              f"{s['acc@80']:.4f} | {g['acc@80']:.4f} |")

    # --- novelty: cost-optimal operating point ---
    print("\n### Cost-optimal deferral (error:review cost ratio -> best coverage)")
    print("| model | 5:1 coverage | 5:1 acc | 10:1 coverage | 10:1 acc | 20:1 coverage | 20:1 acc |")
    print("|---|---|---|---|---|---|---|")
    for name, r in report.items():
        co = r.get("cost_optimal") or {}
        if not co:
            continue
        cells = []
        for k in ("5:1", "10:1", "20:1"):
            v = co.get(k, {})
            cells += [f"{v.get('coverage', float('nan')):.3f}", f"{v.get('accuracy', float('nan')):.4f}"]
        print(f"| {name} | " + " | ".join(cells) + " |")

    # --- novelty: per-question-type conformal coverage ---
    print("\n### Mondrian conformal, per question type (target 0.90)")
    print("| model | type | coverage | mean set |")
    print("|---|---|---|---|")
    for name, r in report.items():
        for qtype, c in (r.get("conformal_mondrian") or {}).items():
            print(f"| {name} | {qtype} | {c['coverage']:.4f} | {c['mean_set_fraction']:.4f} |")

    print("\n### Per workflow accuracy / acc@80")
    wfs = sorted({wf for r in report.values() for wf in r["by_workflow"]})
    print("| model | " + " | ".join(wfs) + " |")
    print("|" + "|".join("---" for _ in range(len(wfs) + 1)) + "|")
    for name, r in report.items():
        cells = [f"{r['by_workflow'][wf]['accuracy']:.4f}/{r['by_workflow'][wf]['acc@80']:.4f}"
                 for wf in wfs]
        print(f"| {name} | " + " | ".join(cells) + " |")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
