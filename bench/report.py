"""Build the results tables from analysis JSON files, with explicit win detection.

Reads one or more analysis files produced by bench/analyze.py and writes a
markdown report:

  assets/results.md   tables + a "where Pramana stands" section

Win detection is mechanical: for each axis it prints who ranks first among the
models present, and whether ours is first, second, or neither. Nothing is
claimed that the numbers in the JSON do not support.

Usage:
    uv run python bench/report.py \
        --model "Decider-4B" results/analysis_decider.json \
        --model "Pramana-2B" results/analysis_niti.json \
        --ours Pramana-2B --out assets/results.md
"""

from __future__ import annotations

import argparse
import json
import pathlib

AXES = [
    # label, path, lower_is_better
    ("accuracy", ("core", "accuracy"), False),
    ("KL from gold", ("core", "kl_from_gold"), True),
    ("Brier", ("core", "brier"), True),
    ("ECE (10 bins)", ("core", "ece_10"), True),
    ("acc@80% coverage", ("selective", "acc@80"), False),
    ("selective AUC", ("selective", "selective_auc"), False),
    ("conformal coverage (target 0.90)", ("conformal", "coverage"), False),
    ("conformal mean set share", ("conformal", "mean_set_fraction"), True),
    ("expected-score MAE", ("ordinal", "expected_score_mae"), True),
    ("score level error", ("ordinal", "level_error"), True),
    ("far-level leak rate", ("ordinal", "far_level_leak_rate"), True),
    ("Brier reliability", ("decomposition", "reliability"), True),
    ("Brier resolution", ("decomposition", "resolution"), False),
    ("noul Brier", ("noul", "brier"), True),
]


def dig(rec: dict, path: tuple[str, ...]):
    cur = rec
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=2, action="append", required=True, metavar=("NAME", "JSON"))
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out", default="assets/results.md")
    args = ap.parse_args()

    data = {}
    for name, path in args.model:
        raw = json.loads(pathlib.Path(path).read_text())
        # analyze.py keys its output by model name; accept either shape
        data[name] = raw[name] if name in raw else next(iter(raw.values()))
    names = list(data)
    if args.ours not in data:
        raise SystemExit(f"--ours {args.ours!r} is not among the --model names {names}")

    lines: list[str] = []
    lines.append("# Results\n")

    lines.append("## Head to head, 2,000 decisions on the Typed Decisions test split\n")
    header = "| axis | " + " | ".join(names) + " | best | ours |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(names) + 3)) + "|")

    wins, losses = [], []
    for label, path, lower in AXES:
        vals = {n: dig(data[n], path) for n in names}
        present = {n: v for n, v in vals.items() if isinstance(v, (int, float))}
        if not present:
            continue
        best_name = min(present, key=present.get) if lower else max(present, key=present.get)
        ours = vals.get(args.ours)
        # rank of ours among present values
        ordered = sorted(present.items(), key=lambda kv: kv[1], reverse=not lower)
        rank = [n for n, _ in ordered].index(args.ours) + 1 if args.ours in present else None
        marker = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th" if rank else "—")
        if rank == 1:
            wins.append(label)
        elif rank and rank > 1:
            losses.append((label, marker, best_name, present[best_name]))
        row = [label] + [fmt(vals[n]) for n in names] + [best_name, marker]
        lines.append("| " + " | ".join(row) + " |")

    lines.append(f"\n## Where {args.ours} stands\n")
    if wins:
        lines.append("First on: " + ", ".join(f"**{w}**" for w in wins) + ".\n")
    if losses:
        lines.append("Not first on:\n")
        lines.append("| axis | ours | leader | leader value |")
        lines.append("|---|---|---|---|")
        for label, marker, best_name, best_v in losses:
            lines.append(f"| {label} | {marker} | {best_name} | {fmt(best_v)} |")
        lines.append("")

    lines.append("## Per workflow\n")
    wfs = sorted({wf for n in names for wf in data[n].get("by_workflow", {})})
    lines.append("| workflow | " + " | ".join(names) + " |")
    lines.append("|" + "|".join(["---"] * (len(names) + 1)) + "|")
    for wf in wfs:
        cells = [fmt(dig(data[n], ("by_workflow", wf, "accuracy"))) for n in names]
        lines.append(f"| {wf} | " + " | ".join(cells) + " |")

    lines.append("\n## Per question\n")
    qs = sorted({q for n in names for q in data[n].get("by_question", {})})
    lines.append("| question | " + " | ".join(names) + " |")
    lines.append("|" + "|".join(["---"] * (len(names) + 1)) + "|")
    for q in qs:
        cells = [fmt(dig(data[n], ("by_question", q, "accuracy"))) for n in names]
        lines.append(f"| {q} | " + " | ".join(cells) + " |")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
