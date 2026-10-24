"""Probe which GPU types this Modal workspace can actually launch.

Each probe starts a container on that GPU, runs one CUDA call, and returns the
device name and VRAM. Sequential on purpose: it is the cheapest way to answer
"what do I have" and it shows per-type failures instead of one aggregate error.

    modal run gpu_probe.py
"""

from __future__ import annotations

import modal

app = modal.App("niti-probe")

image = modal.Image.from_registry(
    "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel", add_python=None
)

# Modal's GPU type strings. Anything not offered to this workspace fails on launch.
TYPES = {
    "T4": "T4",
    "L4": "L4",
    "A10": "A10",
    "L40S": "L40S",
    "A100-40GB": "A100-40GB",
    "A100-80GB": "A100-80GB",
    "H100": "H100",
    "H200": "H200",
}


def body() -> dict:
    import torch

    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(p.total_memory / 2 ** 30, 1),
        "torch": torch.__version__,
    }


fns = {
    slug: app.function(image=image, gpu=gpu, timeout=300, name=f"probe_{slug.lower().replace('-', '_')}")(
        body
    )
    for slug, gpu in TYPES.items()
}


@app.local_entrypoint()
def main() -> None:
    for slug, fn in fns.items():
        try:
            r = fn.remote()
            print(f"{slug:<12} OK    {r['gpu']} | {r['vram_gb']} GB | torch {r['torch']}", flush=True)
        except Exception as e:  # noqa: BLE001 - we want the reason, whatever it is
            print(f"{slug:<12} FAIL  {type(e).__name__}: {str(e)[:180]}", flush=True)
