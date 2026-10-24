"""Dump full probability distributions for the Typed Decisions test split.

Writes one JSONL line per decision so bench/score.py (board-identical metrics)
can score it:

    {"id": ..., "question": ..., "probabilities": {option: p, ...}}

Usage:
    python dump_preds.py --adapter /content/niti/run1/final --out /content/niti/preds_run1.jsonl
    python dump_preds.py --adapter none --out /content/niti/preds_base.jsonl   # base model, no LoRA
"""

from __future__ import annotations

import argparse
import json
import time

from datasets import load_dataset
from unsloth import FastDecisionModel

DATASET = "LocalLLaMA/typed-decisions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none", help="LoRA dir, or 'none' for the base model")
    ap.add_argument("--base", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    args = ap.parse_args()

    name = args.base if args.adapter == "none" else args.adapter
    print(f"[load] {name}", flush=True)
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=name, max_seq_length=2048, load_in_4bit=True)
    FastDecisionModel.for_inference(model)

    ds = load_dataset(DATASET, "all", split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[data] {len(ds)} cases", flush=True)

    n = 0
    t0 = time.time()
    with open(args.out, "w") as f:
        for i, row in enumerate(ds):
            state_raw = row["state"]
            try:
                state = json.loads(state_raw)
            except (json.JSONDecodeError, TypeError):
                state = state_raw
            questions = json.loads(row["questions"])
            gold = json.loads(row["gold"])
            answers = FastDecisionModel.predict(model, tokenizer, state, questions)
            for qid in gold:
                ans = answers.get(qid) or {}
                probs = ans.get("probabilities") or {}
                # normalise noul keys to the gold convention ("true"/"false")
                probs = {
                    ("true" if str(k).lower() in ("true", "yes") else
                     "false" if str(k).lower() in ("false", "no") else str(k)): float(v)
                    for k, v in probs.items()
                }
                f.write(json.dumps({"id": row["id"], "question": qid,
                                    "probabilities": probs}) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"[pred] {i+1}/{len(ds)} cases | {el:.0f}s | "
                      f"eta {el/(i+1)*(len(ds)-i-1)/60:.0f} min", flush=True)

    print(f"[done] {n} decisions -> {args.out} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
