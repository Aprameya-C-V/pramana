"""Run Niti training on Modal.

Two runs in one container, so the image, the model download and the dataset are
paid once:

    A  baseline   vanilla DecisionTrainer recipe (what the Unsloth docs do)
    B  recipe     same data and steps, plus brier and ordinal losses and
                  option-order permutation

Artifacts land in the Modal Volume "niti-runs": /runs/<name>/adapter,
preds.jsonl, run_config.json.

Usage
    modal run modal_app.py::check                     # verify image + unsloth import
    modal run --detach modal_app.py --steps 300       # launch the campaign
    modal volume ls niti-runs /runs                   # inspect
    modal volume get niti-runs /runs/B ./results/B    # download
"""

from __future__ import annotations

import json
import subprocess

import modal

APP_NAME = "niti"
GPU = "L40S"
VOLUME = "niti-runs"

app = modal.App(APP_NAME)

image = (
    # CUDA 12.8 / torch 2.8 image, the same pairing the official Unsloth decision
    # notebook pins, so xformers and bitsandbytes resolve without version games.
    modal.Image.from_registry(
        "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel", add_python=None
    )
    .pip_install("unsloth", "unsloth_zoo", "bitsandbytes", "trl", "datasets", "transformers")
    .add_local_dir(".", "/repo", copy=True, ignore=["results/**", "data/**", "hf/**", ".git/**"])
)

run_volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


def _common(steps: int, batch: int, ga: int) -> list[str]:
    return ["--seq", "1024", "--bs", str(batch), "--ga", str(ga), "--lr", "2e-4",
            "--gc", "false", "--steps", str(steps), "--save-steps", str(max(100, steps // 2))]


def _run(name: str, args: list[str]) -> dict:
    print(f"===== {name} =====", flush=True)
    subprocess.run(["python", "/repo/train/run.py", "--out", f"/runs/{name}", *args], check=True)
    run_volume.commit()
    with open(f"/runs/{name}/run_config.json") as f:
        return json.load(f)


@app.function(image=image, gpu=GPU, timeout=4 * 3600, volumes={"/runs": run_volume})
def campaign(steps: int = 300, batch: int = 2, ga: int = 4, modes: str = "AB") -> list[dict]:
    import torch
    import unsloth
    import transformers

    print(f"gpu={torch.cuda.get_device_name(0)} | vram="
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)
    print(f"unsloth={unsloth.__version__} torch={torch.__version__} "
          f"tf={transformers.__version__}", flush=True)

    common = _common(steps, batch, ga)
    results = []
    if "A" in modes:
        results.append({"name": "A", **_run("A", common)})
    if "B" in modes:
        results.append({"name": "B", **_run("B", common + [
            "--brier-weight", "0.1", "--ordinal-weight", "0.1", "--permute-fields"])})
    return results


@app.function(image=image, gpu=GPU, timeout=20 * 60)
def check() -> dict:
    import bitsandbytes
    import torch
    import unsloth
    import transformers
    from unsloth import DecisionTrainer, FastDecisionModel

    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "torch": torch.__version__,
        "unsloth": unsloth.__version__,
        "transformers": transformers.__version__,
        "decision_api": bool(FastDecisionModel and DecisionTrainer),
        "cuda_available": torch.cuda.is_available(),
    }


@app.local_entrypoint()
def main(steps: int = 300, batch: int = 2, ga: int = 4, modes: str = "AB") -> None:
    for r in campaign.remote(steps=steps, batch=batch, ga=ga, modes=modes):
        print(f"\n=== {r['name']} ===")
        for k in ("train_minutes", "peak_vram_gb", "holdout", "calibrate", "predict_minutes"):
            if k in r:
                print(f"  {k}: {r[k]}")
