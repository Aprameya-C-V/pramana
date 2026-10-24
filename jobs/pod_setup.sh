#!/usr/bin/env bash
# First-time setup on a fresh RunPod pod (runpod/pytorch:2.8.0-cuda12.8.4-cudnn9-devel).
#
# Compiles the two kernels Unsloth wants and that are usually missing:
#   xformers      matching torch 2.8 (the official decision notebook pins this)
#   causal_conv1d Qwen3.5's hybrid attention falls back to a slow reference path
#                 without it; Unsloth flags this explicitly on first step
#
# Both are best-effort: a failure degrades speed, never correctness.
set -uo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1

echo "=== host ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"

echo "=== core deps ==="
pip install -q --upgrade unsloth unsloth_zoo bitsandbytes trl datasets 2>&1 | tail -3

echo "=== xformers (torch 2.8 pin) ==="
pip install -q "xformers==0.0.32.post2" 2>&1 | tail -3 || echo "xformers: install failed, continuing"

echo "=== build backend ==="
pip install -q ninja packaging 2>&1 | tail -2

echo "=== causal_conv1d (source build, slow) ==="
pip uninstall -y causal-conv1d >/dev/null 2>&1 || true
timeout 900 pip install -q --no-build-isolation causal-conv1d 2>&1 | tail -5 \
  || echo "causal_conv1d: build failed, will use the slow reference path"

echo "=== verify ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
try:
    import xformers; print("xformers", xformers.__version__)
except Exception as e:
    print("xformers: missing", str(e)[:80])
try:
    import causal_conv1d; print("causal_conv1d", getattr(causal_conv1d, "__version__", "?"))
except Exception as e:
    print("causal_conv1d: missing", str(e)[:80])
import unsloth; print("unsloth", unsloth.__version__)
from unsloth import FastDecisionModel, DecisionTrainer
print("decision api ok")
PY
