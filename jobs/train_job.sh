#!/usr/bin/env bash
# Runs inside an HF Jobs container.
#
# Mounted by the job:
#   /repo           this project, read-only   (hf://datasets/<user>/niti-code)
#   /runs           writable output           (hf://datasets/<user>/niti-runs)
#
# Env: RUN_NAME, plus any extra args in EXTRA_ARGS.
set -euo pipefail

echo "== gpu =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== install =="
pip install -q --upgrade unsloth unsloth_zoo bitsandbytes trl datasets

echo "== versions =="
python -c "import unsloth, torch, transformers, trl; \
print('unsloth', unsloth.__version__, '| torch', torch.__version__, \
'| tf', transformers.__version__, '| trl', trl.__version__)"

RUN_NAME="${RUN_NAME:-run}"
mkdir -p "/runs/${RUN_NAME}"

echo "== train ${RUN_NAME} =="
python /repo/train/run.py \
    --out "/runs/${RUN_NAME}" \
    --seq 1024 --bs 2 --ga 4 --lr 2e-4 --gc false \
    ${EXTRA_ARGS:-}

echo "== done =="
ls -la "/runs/${RUN_NAME}"
