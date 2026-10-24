#!/usr/bin/env bash
# Round 2: isolate the two open questions.
#
#   C  big Brier weight    can distributional fidelity (KL/Brier) be fixed by
#                          weighting the Brier term properly, rather than at 0.1
#   D  longer training     does accuracy still climb past 250 steps at bs 8
#
# Same data, same base, same calibration. Only one variable moves per run.
set -euo pipefail
export UNSLOTH_SKIP_TORCHVISION_CHECK=1
RUNS_DIR="${RUNS_DIR:-/root/niti/runs2}"
STEPS="${STEPS:-400}"
mkdir -p "$RUNS_DIR"
BASE="--seq 1024 --bs 8 --ga 1 --lr 2e-4 --gc false --workers 16"

echo "=== C brier=1.0 steps=${STEPS} $(date -u +%H:%M:%S) ==="
python /root/niti/train/run.py --out "$RUNS_DIR/C" ${BASE} --steps "${STEPS}" \
    --brier-weight 1.0 --ordinal-weight 0.1

echo "=== D long steps=600 $(date -u +%H:%M:%S) ==="
python /root/niti/train/run.py --out "$RUNS_DIR/D" ${BASE} --steps 600

echo "=== round2 done $(date -u +%H:%M:%S) ==="
for r in C D; do
    python - "$r" "$RUNS_DIR" <<'PY'
import json, sys
name, base = sys.argv[1], sys.argv[2]
c = json.load(open(f"{base}/{name}/run_config.json"))
print(f"{name}: train {c.get('train_minutes', 0):.1f} min | holdout "
      f"{(c.get('holdout') or {}).get('accuracy')} | calib ece "
      f"{round((c.get('calibrate') or {}).get('ece', 0), 4)}")
PY
done
