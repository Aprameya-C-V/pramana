"""Zero-shot the Pramana adapter on public decision-shaped benchmarks.

These are transfer tests, not training runs: the adapter was fitted on Typed
Decisions only. They answer "does a model trained on typed decisions generalise to
an unseen label space", which is the claim the Unsloth decision-model docs make
for BANKING77 and CLINC150.

Runs on a GPU pod:
    python bench/run_bench.py --bench banking77 --adapter /root/niti/runs/A/adapter \
        --out /root/niti/bench/banking77.jsonl --limit 1000

Benchmarks
    banking77   77 banking intents, single text field
    clinc150    150 assistant intents plus out-of-scope
    when2call   NVIDIA When2Call "mcq", classified by kind of correct answer

Writes one line per item:
    {"id": ..., "question": "intent", "probabilities": {label: p}}

Notes on honesty
    when2call: this scores the *answer kind* (tool_call vs cannot_answer vs direct),
    which is not the full multiple-choice protocol that Artificial Analysis runs.
    Label it as a proxy, never as an Artificial Analysis score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from datasets import load_dataset
from unsloth import FastDecisionModel

BENCH_QID = "answer"


def _label_names(ds, col: str) -> list[str]:
    """Label strings for a column, whatever the hub exposes for it.

    mteb/banking77 ships the label as a plain Value, not a ClassLabel, so
    `.names` does not exist; other mirrors expose a ClassLabel. Handle both.
    """
    feat = ds.features[col]
    if hasattr(feat, "names") and feat.names:
        return list(feat.names)
    for candidate in ("label_text", "intent_text", "text_label"):
        if candidate in ds.column_names:
            return sorted({str(v) for v in ds[candidate]})
    values = sorted({str(ds[i][col]) for i in range(min(len(ds), 500))})
    return values


def readable(name: str) -> str:
    return name.replace("_", " ").replace("intent", "").strip()


def build_banking77(limit: int):
    ds = load_dataset("mteb/banking77", "default", split="test")
    labels = _label_names(ds, "label")
    if limit:
        ds = ds.shuffle(seed=3407).select(range(min(limit, len(ds))))
    for i, r in enumerate(ds):
        state = r["text"]
        gold = labels[r["label"]]
        criteria = {lbl: readable(lbl) for lbl in labels}
        yield i, state, criteria, gold


def build_clinc150(limit: int):
    ds = load_dataset("clinc/clinc_oos", "plus", split="test")
    labels = _label_names(ds, "intent")
    if limit:
        ds = ds.shuffle(seed=3407).select(range(min(limit, len(ds))))
    for i, r in enumerate(ds):
        state = r["text"]
        gold = labels[r["intent"]]
        criteria = {lbl: readable(lbl) for lbl in labels}
        yield i, state, criteria, gold


def build_when2call(limit: int):
    ds = load_dataset("nvidia/When2Call", split="mcq")
    if limit:
        ds = ds.shuffle(seed=3407).select(range(min(limit, len(ds))))
    kinds = ["tool_call", "cannot_answer", "direct"]
    for i, r in enumerate(ds):
        tools = r.get("tools") or []
        tool_text = "\n".join(str(t) for t in tools[:20])
        state = f"Request: {r['question']}\n\nAvailable tools:\n{tool_text}"
        gold = r["correct_answer"]
        criteria = {k: {"tool_call": "call one of the available tools",
                        "cannot_answer": "the request cannot be answered with these tools",
                        "direct": "answer directly without a tool"}[k] for k in kinds}
        yield i, state, criteria, gold


BUILDERS = {"banking77": build_banking77, "clinc150": build_clinc150,
            "when2call": build_when2call}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--base", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold-out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seq", type=int, default=2048)
    args = ap.parse_args()

    name = args.base if args.adapter == "none" else args.adapter
    print(f"[load] {name}", flush=True)
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=name, max_seq_length=args.seq, load_in_4bit=True)
    FastDecisionModel.for_inference(model)

    questions_tmpl = lambda criteria: {  # noqa: E731
        BENCH_QID: {"type": "choice",
                    "instructions": "Which label applies to this input?",
                    "criteria": criteria}
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gold_out = pathlib.Path(args.gold_out) if args.gold_out else out.with_name(out.stem + "_gold.jsonl")

    t0 = time.time()
    n = 0
    with out.open("w") as f, gold_out.open("w") as gf:
        for i, state, criteria, gold in BUILDERS[args.bench](args.limit):
            try:
                answers = FastDecisionModel.predict(model, tokenizer, state, questions_tmpl(criteria))
                probs = (answers.get(BENCH_QID) or {}).get("probabilities") or {}
            except Exception as e:  # noqa: BLE001
                print(f"[warn] item {i} failed: {str(e)[:120]}", flush=True)
                probs = {}
            f.write(json.dumps({"id": str(i), "question": BENCH_QID,
                                "probabilities": {str(k): float(v) for k, v in probs.items()}}) + "\n")
            gf.write(json.dumps({"id": str(i), "gold": gold}) + "\n")
            f.flush()
            gf.flush()
            n += 1
            if n % 100 == 0:
                el = time.time() - t0
                print(f"[pred] {n} | {el/60:.1f} min | eta {el/n*(args.limit-n)/60 if args.limit else 0:.1f} min",
                      flush=True)

    print(f"[done] {n} items -> {out} and {gold_out} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
