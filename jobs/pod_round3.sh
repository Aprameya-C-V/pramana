#!/usr/bin/env bash
# Round 3, 30-minute budget.
#
#   1. C2  ordinal fix. Round 2 showed brier_weight=1.0 is the right scale and
#          that ordinal_weight=0.1 does nothing; the entire accuracy deficit sits
#          in ordered `score` questions (exp-score MAE 0.59 vs 0.19 for a 4B).
#          So: ordinal_weight 1.0 and light label smoothing on top of C's recipe.
#   2. Two clean transfer benchmarks on the C2 adapter, concurrently, shuffled
#          sampling this time (the earlier 300-item slice covered 10 of 150
#          intents, which is not a benchmark number).
set -uo pipefail
export UNSLOTH_SKIP_TORCHVISION_CHECK=1
export PIP_BREAK_SYSTEM_PACKAGES=1
cd /root/niti

echo "=== C2 ordinal fix $(date -u +%H:%M:%S) ==="
python train/run.py --out /root/niti/runs3/C2 \
    --seq 1024 --bs 8 --ga 1 --lr 2e-4 --gc false --workers 16 \
    --steps 400 --save-steps 200 \
    --brier-weight 1.0 --ordinal-weight 1.0 --label-smoothing 0.05
echo "=== C2 done $(date -u +%H:%M:%S) ==="

mkdir -p /root/niti/benchout2
echo "=== transfer benchmarks $(date -u +%H:%M:%S) ==="
python bench/run_bench.py --bench banking77 --adapter /root/niti/runs3/C2/adapter \
    --out /root/niti/benchout2/banking77.jsonl --limit 600 --seq 2048 \
    > /root/niti/b2_banking77.log 2>&1 &
B1=$!
python bench/run_bench.py --bench clinc150 --adapter /root/niti/runs3/C2/adapter \
    --out /root/niti/benchout2/clinc150.jsonl --limit 600 --seq 2048 \
    > /root/niti/b2_clinc150.log 2>&1 &
B2=$!
wait $B1 $B2
echo "=== round3 done $(date -u +%H:%M:%S) ==="
python - <<'PY'
import json
c = json.load(open("/root/niti/runs3/C2/run_config.json"))
print("C2 train", round(c["train_minutes"], 1), "min | holdout",
      c["holdout"], "| calib", c["calibrate"])
PY
