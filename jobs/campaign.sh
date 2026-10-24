#!/usr/bin/env bash
# Two runs in one job, so the container and model download are paid once.
#
#   A  baseline   vanilla DecisionTrainer recipe, matches what the docs do
#   B  recipe     same data and steps, plus brier and ordinal losses and
#                 option-order permutation, which is the delta being tested
#
# Writes /runs/A and /runs/B, each with adapter/, preds.jsonl, run_config.json.
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q --upgrade unsloth unsloth_zoo bitsandbytes trl datasets
python -c "import unsloth, torch, transformers; print('unsloth', unsloth.__version__, 'torch', torch.__version__, 'tf', transformers.__version__)"

STEPS="${STEPS:-300}"
COMMON="--seq 1024 --bs 2 --ga 4 --lr 2e-4 --gc false --steps ${STEPS} --save-steps 150"

echo "== run A: baseline =="
python /repo/train/run.py --out /runs/A ${COMMON} 2>&1 | tail -40

echo "== run B: recipe =="
python /repo/train/run.py --out /runs/B ${COMMON} \
    --brier-weight 0.1 --ordinal-weight 0.1 --permute-fields 2>&1 | tail -40

echo "== summary =="
for r in A B; do
    python - "$r" <<'PY'
import json, sys
r = sys.argv[1]
c = json.load(open(f"/runs/{r}/run_config.json"))
print(r, "| train", round(c.get("train_minutes") or 0, 1), "min |",
      "holdout", c.get("holdout"), "| calib", c.get("calibrate"))
PY
done
