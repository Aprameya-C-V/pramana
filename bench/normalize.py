"""Normalise third-party prediction files into the canonical format.

Canonical line, one per (case, question):
    {"id": <case id>, "question": <qid>, "probabilities": {option: p, ...}}

Sources handled
    per-case with nested answers   Decider-4B, Arbiter-4B  (response.answers)
    per-case with flat answers     sysone-0.4B             (answers)
    per-decision                   Qwen3.5-4B-typed-decisions

Usage:
    uv run python bench/normalize.py results/others/decider4b.jsonl results/others/norm/decider4b.jsonl
"""

from __future__ import annotations

import json
import pathlib
import sys


def norm_prob_key(k) -> str:
    s = str(k).strip().lower()
    return {"yes": "true", "no": "false"}.get(s, s)


def as_probs(d: dict) -> dict[str, float]:
    return {norm_prob_key(k): float(v) for k, v in (d or {}).items()}


def emit(records: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for (cid, qid), probs in records.items():
            f.write(json.dumps({"id": cid, "question": qid, "probabilities": probs}) + "\n")


def main() -> None:
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    records: dict[tuple[str, str], dict] = {}
    per_decision = True

    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "question" in r and "probabilities" in r:  # already canonical
            records[(r["id"], r["question"])] = as_probs(r["probabilities"])
            continue
        per_decision = False
        answers = (r.get("response") or {}).get("answers") or r.get("answers") or {}
        for qid, ans in answers.items():
            if isinstance(ans, dict) and "probabilities" in ans:
                records[(r["id"], qid)] = as_probs(ans["probabilities"])

    emit(records, dst)
    n_cases = len({c for c, _ in records})
    print(f"{src.name}: {'per-decision' if per_decision else 'per-case'} "
          f"-> {len(records)} decisions / {n_cases} cases -> {dst}")


if __name__ == "__main__":
    main()
