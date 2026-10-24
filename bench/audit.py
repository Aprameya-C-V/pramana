"""Contamination audit: prove the model never saw test data.

The typed-decisions card forbids training on test rows, test labels, or anything
derived from predictions on them, and a previous leaderboard entry was demoted
for violating it. This prints the same checks so the claim is auditable.

Checks
  ids        exact case-id overlap
  states     exact serialised-state overlap
  shingles   any 60-character substring of a test state appearing in train
             (catches paraphrased or lightly edited duplicates)
  gold       any test gold distribution appearing verbatim in train

Usage:
    uv run --with pyarrow python bench/audit.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "all_train.parquet"
TEST = ROOT / "data" / "all_test.parquet"
SHINGLE = 60


def load(path: pathlib.Path) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    for r in rows:
        for k in ("state", "questions", "gold"):
            if isinstance(r[k], str):
                try:
                    r[k + "_parsed"] = json.loads(r[k])
                except json.JSONDecodeError:
                    r[k + "_parsed"] = None
    return rows


def state_text(row: dict) -> str:
    s = row["state_parsed"]
    return json.dumps(s, sort_keys=True) if s is not None else str(row["state"])


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    t = " ".join(text.split())
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))} if len(t) >= n else {t}


def main() -> int:
    train, test = load(TRAIN), load(TEST)
    print(f"train {len(train)} cases | test {len(test)} cases")
    failures = []

    train_ids, test_ids = {r["id"] for r in train}, {r["id"] for r in test}
    shared_ids = train_ids & test_ids
    print(f"[ids]      overlap: {len(shared_ids)} {sorted(shared_ids)[:5]}")
    if shared_ids:
        failures.append("id overlap")

    train_states = {state_text(r) for r in train}
    test_states = {state_text(r) for r in test}
    shared_states = train_states & test_states
    print(f"[states]   exact overlap: {len(shared_states)}")
    if shared_states:
        failures.append("exact state overlap")

    test_shingles: set[str] = set()
    for s in test_states:
        test_shingles |= shingles(s)
    hits = {}
    for r in train:
        overlap = test_shingles & shingles(state_text(r))
        if overlap:
            hits[r["id"]] = len(overlap)
    print(f"[shingles] train states sharing a {SHINGLE}-char substring with a test state: "
          f"{len(hits)} {list(hits.items())[:5]}")
    if hits:
        failures.append("shingle overlap")

    train_gold = {json.dumps(r["gold_parsed"], sort_keys=True) for r in train}
    shared_gold = {json.dumps(r["gold_parsed"], sort_keys=True) for r in test} & train_gold
    print(f"[gold]     verbatim shared distributions: {len(shared_gold)}")

    by_wf: dict[str, int] = {}
    for r in train:
        by_wf[r["workflow"]] = by_wf.get(r["workflow"], 0) + 1
    print(f"[coverage] train workflows: {by_wf}")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nCLEAN: no id, state, shingle or gold overlap between train and test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
