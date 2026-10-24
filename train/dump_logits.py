"""Dump raw per-question logits (no temperature, no rounding) for the holdout and test.

predict() rounds probabilities to 4 decimals (hard zeros, which KL punishes) and
serves temperatures that calibrate() fitted against hard labels. The board scores
KL and Brier against the soft gold, so we refit temperature ourselves on the
holdout (bench/temper.py) and never touch the test set while fitting.

    python train/dump_logits.py --adapter runs2/C/adapter --out runs2/C/logits.pt
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import load_dataset
from unsloth import FastDecisionModel
from unsloth.models import decision as D
from unsloth.models._decision_common import _served_temperatures
from unsloth.models.clef import question_options

DATASET = "LocalLLaMA/typed-decisions"
TYPES = ["choice", "score", "noul"]  # Laya numbering used by _clef_logits' qtype


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()

    model, tok = FastDecisionModel.from_pretrained(model_name=a.adapter, max_seq_length=a.seq,
                                                   load_in_4bit=True)
    FastDecisionModel.for_inference(model)
    print("[config]", {k: v for k, v in model.decision_config.items() if "temp" in k}, flush=True)

    train = load_dataset(DATASET, "all", split="train")
    items, report = FastDecisionModel.build_dataset(train, tok, model)
    _, hold = FastDecisionModel.split_holdout(items, seed=a.seed)
    hz, hq = D._decision_logits(model, tok, hold)
    print(f"[holdout] {len(hold)} cases, {len(hz)} decisions", flush=True)

    test = load_dataset(DATASET, "all", split="test")
    titems, treport = FastDecisionModel.build_dataset(test, tok, model)
    print(f"[test build] {treport}", flush=True)
    tz, tq = D._decision_logits(model, tok, titems)
    meta = []
    for item in titems:
        for name, q in item["source"]["questions"].items():
            meta.append({"id": test[item["row"]]["id"], "question": name,
                         "options": [k for k, _ in question_options(q)]})
    assert len(meta) == len(tz), (len(meta), len(tz))
    print(f"[test] {len(titems)} cases, {len(tz)} decisions", flush=True)

    torch.save({
        "holdout": {"logits": hz, "target": [q["target"] for q in hq],
                    "qtype": [q["qtype"] for q in hq],
                    "served_T": _served_temperatures(model.decision_config, hz, hq)},
        "test": {"logits": tz, "meta": meta, "qtype": [q["qtype"] for q in tq],
                 "served_T": _served_temperatures(model.decision_config, tz, tq)},
        "types": TYPES,
    }, a.out)
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
