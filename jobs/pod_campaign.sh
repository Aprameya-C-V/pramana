#!/usr/bin/env bash
# A then B on a rented GPU. One container, one model download.
#
#   A  baseline  vanilla DecisionTrainer recipe
#   B  recipe    + brier loss, + ordinal loss, + option-order permutation
#
# bs 8 / ga 1 keeps the effective batch at 8 (same optimisation as bs 2 / ga 4)
# while cutting micro-step overhead, which is the real cost on a 2B with float32
# activations.
#
#   RUNS_DIR=/root/niti/runs bash pod_campaign.sh
set -euo pipefail

RUNS_DIR="${RUNS_DIR:-/root/niti/runs}"
STEPS="${STEPS:-300}"
mkdir -p "$RUNS_DIR"

COMMON="--seq 1024 --bs 8 --ga 1 --lr 2e-4 --gc false --workers 16 --steps ${STEPS} --save-steps $((STEPS / 2))"

echo "=== A baseline $(date -u +%H:%M:%S) ==="
python /root/niti/train/run.py --out "$RUNS_DIR/A" ${COMMON}
echo "=== A done $(date -u +%H:%M:%S) ==="

echo "=== B recipe $(date -u +%H:%M:%S) ==="
python /root/niti/train/run.py --out "$RUNS_DIR/B" ${COMMON} \
    --brier-weight 0.1 --ordinal-weight 0.1 --permute-fields
echo "=== B done $(date -u +%H:%M:%S) ==="

echo "=== summary ==="
for r in A B; do
    python - "$r" "$RUNS_DIR" <<'PY'
import json, sys
name, base = sys.argv[1], sys.argv[2]
c = json.load(open(f"{base}/{name}/run_config.json"))
print(f"{name}: train {c.get('train_minutes', 0):.1f} min | peak {c.get('peak_vram_gb')} GB")
print(f"    holdout   {c.get('holdout')}")
print(f"    calibrated {c.get('calibrate')}")
PY
done
