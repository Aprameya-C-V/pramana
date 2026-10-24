"""Check the SOTA claims mechanically, so the numbers in the card are filters, not vibes.

Two claims are computed here:

  A  lowest ECE among models fitted or fine-tuned on `train`
  B  highest accuracy among *all* board entries whose ECE is at or below 0.05

Sources
  Pramana and the four models with public prediction files: recomputed by this repo
  with the board's own metric code (bench/score.py).
  Zero-shot entries without public predictions: transcribed from the benchmark
  card's leaderboard, 2026-10-23. They are marked `published` and are not
  recomputed, so the two sources are never silently mixed.

Usage:
    uv run python bench/sota.py

Writing about it:
  Claim B is the honest headline. It states its filter, so a reader can check it.
  Do not compare our ECE against Decider-4B's submitted 0.0312: that figure is not
  reproducible from its own published report.json, which says 0.1598.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Transcribed from the Typed Decisions leaderboard card (zero-shot table, which is
# where these models live). Params in billions for context only.
PUBLISHED = [
    # name, params_b, accuracy, ece
    ("clef-flash", 9, 0.710, 0.016),
    ("clef", 27, 0.728, 0.025),
    ("jev-vision-27b", 27, 0.714, 0.026),
    ("clef-omni", 35, 0.673, 0.030),
    ("d3-flash", 9, 0.717, 0.031),
    ("d1-3B", 3, 0.656, 0.037),
    ("d3-mini", 4, 0.743, 0.047),
    ("d3", 27, 0.749, 0.049),
    ("pplx-decider-v1.1-27b", 27, 0.753, 0.060),
    ("JEV-35B", 35, 0.700, 0.062),
    ("GEV-26B-Decide", 26, 0.522, 0.066),
    ("d3-turbo", 26, 0.736, 0.069),
    ("d3-nano", 2, 0.629, 0.113),
    ("TypeSafe Jev 1.13.0", 27, 0.727, 0.144),
    ("baseten/mammoth-1", 27, 0.768, 0.153),
]

FITTED_FROM_FILES = {
    # name: analysis json produced by bench/analyze.py
    "Pramana-2B": "analysis_C.json",
    "Decider-4B": "analysis_decider4b.json",
    "Arbiter-4B": "analysis_arbiter4b.json",
    "sysone-0.4B": "analysis_sysone04b.json",
    "Qwen3.5-4B-td": "analysis_qwen35_4b.json",
}

FITTED_PARAMS = {"Pramana-2B": 2.27, "Decider-4B": 4.2, "Arbiter-4B": 4.0,
                 "sysone-0.4B": 0.4, "Qwen3.5-4B-td": 4.0}


def load_fitted() -> dict[str, dict]:
    out = {}
    for name, fname in FITTED_FROM_FILES.items():
        p = ROOT / "results" / fname
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        rec = raw[name] if name in raw else next(iter(raw.values()))
        out[name] = {"accuracy": rec["core"]["accuracy"], "ece": rec["core"]["ece_10"],
                     "kl": rec["core"]["kl_from_gold"], "brier": rec["core"]["brier"]}
    return out


def main() -> None:
    fitted = load_fitted()
    ours = "Pramana-2B"
    if ours not in fitted:
        raise SystemExit("run bench/make_metrics.py / analyze.py for Pramana-2B first")

    print("=== Claim A: lowest ECE among fitted / fine-tuned models ===")
    print("(recomputed from each model's published predictions, 10 bins)")
    for name, r in sorted(fitted.items(), key=lambda kv: kv[1]["ece"]):
        mark = " <-- ours" if name == ours else ""
        print(f"  {r['ece']:.4f}  {name:16s} acc {r['accuracy']:.4f}{mark}")

    print("\n=== Claim B: highest accuracy among entries with ECE <= 0.05 ===")
    rows = [(n, p, r["accuracy"], r["ece"], "recomputed")
            for n, r in fitted.items() if r["ece"] <= 0.05]
    rows += [(n, p, a, e, "published") for n, p, a, e in PUBLISHED if e <= 0.05]
    for name, params, acc, ece, src in sorted(rows, key=lambda x: -x[2]):
        mark = " <-- ours" if name == ours else ""
        print(f"  acc {acc:.4f}  ece {ece:.3f}  {params:>4}B  {name:22s} [{src}]{mark}")

    leader = max(rows, key=lambda x: x[2])
    print(f"\nleader inside the filter: {leader[0]} ({leader[2]:.4f})")
    print(f"ours: {ours} ({fitted[ours]['accuracy']:.4f}) -> "
          f"{'PASS' if leader[0] == ours else 'NOT first: ' + leader[0]}")

    print("\n=== Claim C: against Jev 1.13.0 as published ===")
    jev = {"accuracy": 0.727, "kl": 1.442, "brier": 0.148, "ece": 0.144}
    o = fitted[ours]
    for k in ("accuracy", "kl", "brier", "ece"):
        better = o[k] > jev[k] if k == "accuracy" else o[k] < jev[k]
        print(f"  {k:9s} ours {o[k]:.4f} vs jev {jev[k]:.4f} -> {'better' if better else 'worse'}")


if __name__ == "__main__":
    main()
