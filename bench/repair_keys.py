"""Undo the writer-side key bug in train/run.py and train/dump_preds.py.

Those scripts mapped option keys "0"/"1" to "false"/"true" before writing, which
renames levels 0 and 1 of every `score` question. Mass is preserved, only the key
is wrong, so mapping them back for numeric-option questions is lossless.

    uv run --with pyarrow python bench/repair_keys.py in.jsonl out.jsonl
"""

from __future__ import annotations

import json
import sys

import pyarrow.parquet as pq

src, dst = sys.argv[1], sys.argv[2]
gold = {r["id"]: json.loads(r["gold"]) for r in pq.read_table("data/all_test.parquet").to_pylist()}
fixed = bad_sum = 0
with open(dst, "w") as f:
    for line in open(src):
        r = json.loads(line)
        opts = gold[r["id"]][r["question"]]["probabilities"]
        if "0" in opts and "true" not in opts:
            r["probabilities"] = {{"false": "0", "true": "1"}.get(k, k): v
                                  for k, v in r["probabilities"].items()}
            fixed += 1
        if abs(sum(r["probabilities"].values()) - 1) > 1e-3:
            bad_sum += 1
        f.write(json.dumps(r) + "\n")
print(f"{src}: remapped {fixed} numeric-option decisions, {bad_sum} with probs not summing to 1")
