"""Inspect the Unsloth decision-model API so the training script uses it correctly.

Prints signatures + source of the pieces we need to hook into:
  FastDecisionModel.predict / evaluate / build_dataset / calibrate / split_holdout
  DecisionTrainer.__init__ / compute_loss / prediction_step
"""

import inspect

import unsloth
from unsloth import FastDecisionModel, DecisionTrainer

print("unsloth", unsloth.__version__)

for name in ["from_pretrained", "get_peft_model", "build_dataset", "split_holdout",
             "evaluate", "calibrate", "predict", "for_inference"]:
    fn = getattr(FastDecisionModel, name, None)
    print(f"\n=== FastDecisionModel.{name} ===")
    if fn is None:
        print("  MISSING")
        continue
    try:
        print("sig:", inspect.signature(fn))
    except Exception as e:
        print("sig: <unavailable>", e)
    try:
        src = inspect.getsource(fn)
        print(src[:2500])
    except Exception as e:
        print("src unavailable:", e)

for name in ["__init__", "compute_loss", "prediction_step", "evaluate", "train"]:
    fn = getattr(DecisionTrainer, name, None)
    print(f"\n=== DecisionTrainer.{name} ===")
    if fn is None:
        print("  MISSING")
        continue
    try:
        print("sig:", inspect.signature(fn))
    except Exception as e:
        print("sig: <unavailable>", e)
    try:
        print(inspect.getsource(fn)[:2500])
    except Exception as e:
        print("src unavailable:", e)

print("\n=== decision module file ===")
import unsloth.models as um
mod = getattr(FastDecisionModel, "__module__", None)
print("module:", mod)
try:
    import importlib
    m = importlib.import_module(mod)
    print("file:", m.__file__)
    print("names:", [n for n in dir(m) if "Decision" in n or "decision" in n][:40])
except Exception as e:
    print("import failed:", e)
