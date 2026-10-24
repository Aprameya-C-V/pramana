#!/usr/bin/env bash
# Final release run on one A100: three Pramana-2B variants and Pramana-0.8B, in parallel.
#
#   large  Qwen3.5-2B,   seq 1024, lr 2e-4, the recipe behind the published numbers
#   lowlr  same, lr 1e-4                       (overfitting was the limit at rank 64)
#   perm   same, field-order permutation       (regulariser against the same overfitting)
#   small  Qwen3.5-0.8B, seq 2048, lr 3e-4
#
# Selection, fixed before any test number is seen: release the 2B variant with the
# lowest holdout KL after temperature fitting (all share seed 3407, so one holdout).
# Variants use batch 4 x 2 (same effective batch) so four runs fit in 80 GB.
#
# Every run skips calibrate() (hard-label, CPU-bound, and it inflates KL) and the
# rounded predict(); dump_logits.py + temper.py produce the scored predictions.
# Each run writes final/<name>/DONE when its adapter, logits and predictions exist.
set -uo pipefail
export UNSLOTH_SKIP_TORCHVISION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/niti
mkdir -p final
COMMON="--seed 3407 --gc false --workers 4 --steps 400 --save-steps 400
        --brier-weight 1.0 --ordinal-weight 0.1 --skip-calibrate --skip-predict"

run() {  # name, then extra run.py args
    local name=$1; shift
    ( python train/run.py --out final/$name $COMMON "$@" > final/$name.log 2>&1 \
      && python train/dump_logits.py --adapter final/$name/adapter --out final/$name/logits.pt \
           >> final/$name.log 2>&1 \
      && python bench/temper.py final/$name/logits.pt final/$name/preds_temper.jsonl \
           >> final/$name.log 2>&1 \
      && touch final/$name/DONE ) &
}

run large --model unsloth/Qwen3.5-2B   --seq 1024 --lr 2e-4 --bs 8 --ga 1
run lowlr --model unsloth/Qwen3.5-2B   --seq 1024 --lr 1e-4 --bs 4 --ga 2
run perm  --model unsloth/Qwen3.5-2B   --seq 1024 --lr 2e-4 --bs 4 --ga 2 --permute-fields
run small --model unsloth/Qwen3.5-0.8B --seq 2048 --lr 3e-4 --bs 8 --ga 1
wait
echo ALLDONE
