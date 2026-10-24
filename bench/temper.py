"""Fit per-question-type temperature on the HOLDOUT against the soft gold, apply to test.

The library's calibrate() fits against the hard label, which is right for "confidence =
chance of being correct" (ECE) but wrong for the board's KL and Brier, which compare
against the soft teacher distribution. Nothing here looks at the test set while fitting.

    python bench/temper.py runs/C/logits.pt results/C/preds_temper.jsonl [fit_from_logits.pt]
"""

from __future__ import annotations

import json
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src)
# Optional third argument: fit on that run's holdout instead (for a --no-holdout run,
# whose own holdout was trained on). Test logits always come from src.
H, T = torch.load(sys.argv[3])["holdout"] if len(sys.argv) > 3 else d["holdout"], d["test"]
GRID = torch.exp(torch.linspace(-1.5, 2.5, 161))  # T from 0.22 to 12.2


def soft_ce(zs, targets, t):
    return sum(float(-(torch.tensor(g) * torch.log_softmax(z / t, -1)).sum())
               for z, g in zip(zs, targets)) / max(len(zs), 1)


def kl(zs, targets, temps):
    out = 0.0
    for z, g, t in zip(zs, targets, temps):
        g = torch.tensor(g)
        out += float((g * (torch.log(g.clamp_min(1e-12)) - torch.log_softmax(z / t, -1))).sum())
    return out / len(zs)


fitted = {}
for qt, name in enumerate(d["types"]):
    idx = [i for i, q in enumerate(H["qtype"]) if q == qt]
    zs, gs = [H["logits"][i] for i in idx], [H["target"][i] for i in idx]
    if not idx:
        continue
    best = min(GRID.tolist(), key=lambda t: soft_ce(zs, gs, t))
    fitted[qt] = best
    print(f"{name:7s} n={len(idx):4d}  served T={sum(H['served_T'][i] for i in idx)/len(idx):.3f}"
          f"  soft-fit T={best:.3f}")

served_kl = kl(H["logits"], H["target"], H["served_T"])
fit_kl = kl(H["logits"], H["target"], [fitted.get(q, 1.0) for q in H["qtype"]])
print(f"holdout KL: served {served_kl:.4f} -> soft-fit {fit_kl:.4f}")

with open(dst, "w") as f:
    for z, m, qt in zip(T["logits"], T["meta"], T["qtype"]):
        p = torch.softmax(z / fitted.get(qt, 1.0), -1).tolist()
        f.write(json.dumps({"id": m["id"], "question": m["question"],
                            "probabilities": dict(zip(m["options"], p))}) + "\n")
# Same logits at the served temperatures, unrounded: isolates the rounding effect.
with open(dst.replace(".jsonl", "_served.jsonl"), "w") as f:
    for z, m, t in zip(T["logits"], T["meta"], T["served_T"]):
        p = torch.softmax(z / t, -1).tolist()
        f.write(json.dumps({"id": m["id"], "question": m["question"],
                            "probabilities": dict(zip(m["options"], p))}) + "\n")
print(f"wrote {dst}")
