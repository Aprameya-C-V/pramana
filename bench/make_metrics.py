"""Produce the single metrics.json that the publish step and the model card read.

    uv run --with pyarrow python bench/make_metrics.py results/C2/preds.jsonl \
        --repo <user>/pramana-2b --out results/metrics.json

Writes the four board metrics plus the breakdowns, so every number in the card
traces back to one file produced from the raw predictions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "score", pathlib.Path(__file__).resolve().parent / "score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default="results/metrics.json")
    args = ap.parse_args()

    gold = score.load_gold()
    overall = score.Acc()
    by_wf: dict = {}
    by_q: dict = {}

    for line in pathlib.Path(args.preds).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        g = gold[rec["id"]]["gold"].get(rec["question"])
        if g is None:
            continue
        hit, kl, brier, conf, _ = score.score_one(rec["probabilities"], g)
        overall.add(hit, kl, brier, conf)
        by_wf.setdefault(gold[rec["id"]]["workflow"], score.Acc()).add(hit, kl, brier, conf)
        by_q.setdefault(f"{gold[rec['id']]['workflow']}/{rec['question']}",
                        score.Acc()).add(hit, kl, brier, conf)

    o = overall.summary()
    o["repo"] = args.repo
    o["run"] = args.run
    o["by_workflow"] = {k: v.summary() for k, v in sorted(by_wf.items())}
    o["by_question"] = {k: v.summary() for k, v in sorted(by_q.items())}
    pathlib.Path(args.out).write_text(json.dumps(o, indent=2))
    print(f"accuracy {o['accuracy']:.4f} | kl {o['kl_from_gold']:.4f} | "
          f"brier {o['brier']:.4f} | ece {o['ece']:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
