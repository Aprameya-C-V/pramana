#!/usr/bin/env bash
# Pinned setup, for reproducibility. Use this instead of pod_setup.sh when the
# image's torch version must not drift.
#
# Why the pins exist:
#   `pip install unsloth` pulls a newer torch (2.14.1 in this image) and leaves
#   torchvision/torchaudio compiled against the old one. Unsloth then fails with
#   "operator torchvision::nms does not exist". Pinning all three back to the
#   image's torch, and skipping Unsloth's vision check, keeps the stack coherent.
#   The decision head does not use torchvision.
set -uo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export UNSLOTH_SKIP_TORCHVISION_CHECK=1

TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"

echo "=== restore coherent torch trio ==="
pip install -q --force-reinstall --no-cache-dir \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" 2>&1 | tail -3

echo "=== unsloth stack ==="
pip install -q "unsloth==${UNSLOTH_VERSION:-2026.10.7}" unsloth_zoo bitsandbytes trl datasets 2>&1 | tail -3

echo "=== re-pin torch after unsloth (it upgrades) ==="
pip install -q --force-reinstall --no-deps --no-cache-dir "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" 2>&1 | tail -2

echo "=== drop torchao (unsloth pulls one built for newer torch; it is optional) ==="
pip uninstall -y -q torchao 2>&1 | tail -1

echo "=== kernels ==="
pip install -q "xformers==0.0.32.post2" 2>&1 | tail -2 || echo "xformers failed, continuing"
pip install -q ninja packaging 2>&1 | tail -1
timeout 600 pip install -q --no-build-isolation causal-conv1d 2>&1 | tail -3 \
  || echo "causal_conv1d failed, slow reference path"

echo "=== verify ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
for mod in ("torchvision", "xformers", "causal_conv1d"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print(mod, "missing:", str(e)[:70])
import unsloth
from unsloth import FastDecisionModel, DecisionTrainer
print("unsloth", unsloth.__version__, "| decision api ok")
PY
