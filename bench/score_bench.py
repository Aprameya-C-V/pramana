"""Score a benchmark run from bench/run_bench.py.

Computes top-1 accuracy, the accuracy of the model's most likely label, plus the
calibration of the chosen answer (ECE over the confidence it gave its own pick).
The second number is the point: a transfer test that only reports accuracy hides
whether the model knows when it is out of its depth on an unseen label space.

Usage:
    uv run python bench/score_bench.py results/bench/banking77.jsonl results/bench/banking77_gold.jsonl
"""

from __future__ import annotations

import json
import pathlib
import sys


def main() -> None:
    preds_path, gold_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in gold_path.read_text().splitlines() if l.strip()}

    hits, confs, missing = [], [], 0
    for line in preds_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        probs = r["probabilities"] or {}
        g = gold.get(r["id"])
        if g is None:
            continue
        if not probs:
            missing += 1
            hits.append(False)
            confs.append(0.0)
            continue
        best = max(probs, key=probs.get)
        hits.append(best == g)
        confs.append(float(probs[best]))

    n = len(hits)
    if not n:
        raise SystemExit("no scored items")
    acc = sum(hits) / n

    bins = 10
    bh, bt, bc = [0] * bins, [0] * bins, [0.0] * bins
    for h, c in zip(hits, confs):
        b = min(int(c * bins), bins - 1)
        bt[b] += 1
        bh[b] += int(h)
        bc[b] += c
    ece = sum(bt[b] / n * abs(bh[b] / bt[b] - bc[b] / bt[b]) for b in range(bins) if bt[b])

    print(f"{preds_path.name}: n={n} | accuracy {acc:.4f} | mean confidence {sum(confs)/n:.4f} | "
          f"ECE {ece:.4f}" + (f" | empty predictions {missing}" if missing else ""))


if __name__ == "__main__":
    main()
