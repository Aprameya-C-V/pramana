"""End-to-end run: train -> calibrate -> predict test split -> save adapter.

Stages are controlled by flags so a mistake in the prediction path cannot cost a
full training run:

    python run.py --out /content/niti/runs/smoke --steps 30 --limit 20
    python run.py --out /content/niti/runs/p1   --steps 300

Writes to --out:
    adapter/            LoRA + decision head
    preds.jsonl         one line per (case, question) with full distributions
    run_config.json     args, timings, holdout metrics, calibration result
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported

DATASET = "LocalLLaMA/typed-decisions"


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen3.5-2B")
    p.add_argument("--out", required=True)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--ga", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--brier-weight", type=float, default=0.0)
    p.add_argument("--ordinal-weight", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--permute-fields", action="store_true")
    p.add_argument("--gc", default="false", choices=["false", "true", "unsloth"])
    p.add_argument("--workers", type=int, default=8, help="dataloader workers; 0 is synchronous")
    p.add_argument("--limit", type=int, default=0, help="test cases to predict; 0 = all")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-calibrate", action="store_true",
                   help="calibrate() fits hard labels on CPU; bench/temper.py refits against soft gold")
    p.add_argument("--skip-predict", action="store_true", help="use train/dump_logits.py instead")
    p.add_argument("--no-holdout", action="store_true",
                   help="train on every case; temperatures then come from a sibling run's holdout")
    p.add_argument("--adapter", default=None, help="load adapters from here (with --skip-train)")
    return p.parse_args()


def main() -> None:
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = {**vars(a)}
    print(f"[cfg] {json.dumps(cfg)}", flush=True)

    t0 = time.time()
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=a.adapter or a.model, max_seq_length=a.seq, load_in_4bit=True)
    if not a.skip_train:
        gc = {"false": False, "true": True, "unsloth": "unsloth"}[a.gc]
        model = FastDecisionModel.get_peft_model(
            model, r=a.r, lora_alpha=a.alpha, lora_dropout=0,
            use_gradient_checkpointing=gc, random_state=a.seed)
    else:
        FastDecisionModel.for_inference(model)
    print(f"[load] {time.time()-t0:.0f}s | reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GB",
          flush=True)

    holdout_metrics = None
    train_minutes = None
    calib = None

    if not a.skip_train:
        ds = load_dataset(DATASET, "all", split="train")
        items, report = FastDecisionModel.build_dataset(ds, tokenizer, model)
        print(f"[build] {report}", flush=True)
        train_items, eval_items = FastDecisionModel.split_holdout(items, seed=a.seed)
        if a.no_holdout:
            train_items, eval_items = items, eval_items[:8]  # eval_items only feeds the log line
        print(f"[split] {len(train_items)} train / {len(eval_items)} holdout", flush=True)
        cfg.update(build_report=report, n_train_items=len(train_items),
                   n_holdout=len(eval_items), n_train_cases=len(ds))

        targs = dict(
            per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.ga,
            warmup_steps=a.warmup, learning_rate=a.lr, lr_scheduler_type="cosine",
            weight_decay=0.01, bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(), logging_steps=5,
            output_dir=os.path.join(a.out, "checkpoints"), report_to="none",
            seed=a.seed, save_strategy="steps", save_steps=a.save_steps,
            save_total_limit=2, max_steps=a.steps,
            dataloader_num_workers=a.workers, dataloader_pin_memory=True)
        tkw = dict(model=model, processing_class=tokenizer, train_dataset=train_items,
                   eval_dataset=eval_items, args=TrainingArguments(**targs),
                   brier_weight=a.brier_weight, ordinal_weight=a.ordinal_weight,
                   label_smoothing=a.label_smoothing, permute_fields=a.permute_fields)
        if a.head_lr is not None:
            tkw["head_learning_rate"] = a.head_lr
        trainer = DecisionTrainer(**tkw)

        t1 = time.time()
        stats = trainer.train()
        train_minutes = (time.time() - t1) / 60
        print(f"[train] {train_minutes:.1f} min | peak {torch.cuda.max_memory_reserved()/2**30:.2f} GB"
              f" | {stats.metrics}", flush=True)
        cfg["train_metrics"] = {k: v for k, v in stats.metrics.items()
                                if isinstance(v, (int, float))}

        holdout_metrics = FastDecisionModel.evaluate(model, tokenizer, eval_items)
        print(f"[holdout] {holdout_metrics}", flush=True)
        if not a.skip_calibrate:
            calib = FastDecisionModel.calibrate(model, tokenizer, eval_items)
            print(f"[calibrate] {calib}", flush=True)

        adapter_dir = os.path.join(a.out, "adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[save] {adapter_dir}", flush=True)

    cfg.update(train_minutes=train_minutes, holdout=holdout_metrics, calibrate=calib,
               peak_vram_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
               bf16=is_bfloat16_supported(), torch=torch.__version__)

    if a.skip_predict:
        with open(os.path.join(a.out, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        print(f"[done] {a.out}", flush=True)
        return

    # ---- predict test split ----
    FastDecisionModel.for_inference(model)
    test = load_dataset(DATASET, "all", split="test")
    if a.limit:
        test = test.select(range(min(a.limit, len(test))))
    preds_path = os.path.join(a.out, "preds.jsonl")
    n, t2 = 0, time.time()
    debugged = False
    with open(preds_path, "w") as f:
        for i, row in enumerate(test):
            try:
                state = json.loads(row["state"])
            except (json.JSONDecodeError, TypeError):
                state = row["state"]
            questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
            try:
                answers = FastDecisionModel.predict(model, tokenizer, state, questions)
            except Exception:
                print(f"[predict] FAILED on case {row['id']}", flush=True)
                traceback.print_exc()
                raise
            if not debugged:
                print(f"[predict] first case {row['id']} q={list(answers)[0]} -> "
                      f"{json.dumps(answers[list(answers)[0]])[:300]}", flush=True)
                debugged = True
            for qid in gold:
                probs = (answers.get(qid) or {}).get("probabilities") or {}
                probs = {("true" if str(k).lower() in ("true", "yes") else
                          "false" if str(k).lower() in ("false", "no") else str(k)): float(v)
                         for k, v in probs.items()}
                f.write(json.dumps({"id": row["id"], "question": qid,
                                    "probabilities": probs}) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t2
                print(f"[predict] {i+1}/{len(test)} cases | {el/60:.1f} min | "
                      f"eta {el/(i+1)*(len(test)-i-1)/60:.1f} min", flush=True)
    cfg["predict_minutes"] = round((time.time() - t2) / 60, 2)
    cfg["n_decisions"] = n
    print(f"[predict] {n} decisions -> {preds_path} in {cfg['predict_minutes']} min", flush=True)

    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
