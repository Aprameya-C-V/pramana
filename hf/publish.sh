#!/usr/bin/env bash
# Publish one Pramana model to the Hub, with the artifacts that make its numbers auditable.
#
#   export HF_TOKEN=hf_...
#   bash hf/publish.sh <hf-username> 2b      # Pramana-2B: adapters large + lowlr, averaged
#   bash hf/publish.sh <hf-username> 0.8b    # Pramana-0.8B: adapter small
#
# Uploads:
#   adapters/<name>/ or adapter/        LoRA + decision head (two for 2B, one for 0.8B)
#   calibration.json                    temperatures for probabilities and confidence
#   preds.jsonl                         test predictions, so anyone can re-score
#   results.json                        the four board metrics plus breakdowns
#   .eval_results/typed-decisions.yaml  what the Hub leaderboard reads
#   README.md                           model card (hf/model_card_<size>.md)
#
# Requires the huggingface CLI: pip install -U "huggingface_hub[cli]" or `brew install hf`.
set -euo pipefail

USER_NAME="${1:?usage: publish.sh <hf-username> <2b|0.8b>}"
SIZE="${2:?usage: publish.sh <hf-username> <2b|0.8b>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FINAL="$ROOT/results/final"
case "$SIZE" in
    2b)   NAME="Pramana-2B";   PREDS="$FINAL/pramana2b/preds.jsonl"; CALIB="$FINAL/pramana2b/calibration.json" ;;
    0.8b) NAME="Pramana-0.8B"; PREDS="$FINAL/small/preds_temper.jsonl"; CALIB="$FINAL/small/calibration.json" ;;
    *) echo "size must be 2b or 0.8b" >&2; exit 2 ;;
esac
REPO="${USER_NAME}/pramana-${SIZE}"

: "${HF_TOKEN:?HF_TOKEN is not set}"

echo "== staging =="
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/.eval_results"
if [ "$SIZE" = 2b ]; then
    mkdir -p "$STAGE/adapters"
    cp -r "$FINAL/large/adapter" "$STAGE/adapters/large"
    cp -r "$FINAL/lowlr/adapter" "$STAGE/adapters/lowlr"
else
    cp -r "$FINAL/small/adapter" "$STAGE/adapter"
fi
cp "$PREDS" "$STAGE/preds.jsonl"
cp "$CALIB" "$STAGE/calibration.json"
uv run -q --with pyarrow python "$ROOT/bench/make_metrics.py" "$PREDS" \
    --repo "$REPO" --run "$NAME" --out "$STAGE/results.json"

python3 - "$ROOT" "$STAGE" "$NAME" "$REPO" "$SIZE" <<'PY'
import json, sys, pathlib
root, stage = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
name, repo, size = sys.argv[3:6]
m = json.loads((stage / "results.json").read_text())
card = (root / "hf" / f"model_card_{size}.md").read_text().replace("{{USER}}", repo.split("/")[0])
assert "{{" not in card, "unfilled placeholder in model card"
(stage / "README.md").write_text(card)
lines = []
for task_id in ("accuracy", "kl_from_gold", "brier", "ece"):
    lines.append(f"""- dataset:
    id: LocalLLaMA/typed-decisions
    task_id: {task_id}
  value: {m[task_id]:.6f}
  date: "2026-10-23"
  source:
    url: https://huggingface.co/{repo}/blob/main/results.json
    name: "Self-reported; raw predictions in the same repo"
  notes: "Fine-tuned on the train split only. ECE is 10 equal-width top-label bins over `probabilities`."
""")
(stage / ".eval_results" / "typed-decisions.yaml").write_text("".join(lines))
print(f"staged {name}: acc {m['accuracy']:.4f} kl {m['kl_from_gold']:.4f} "
      f"brier {m['brier']:.4f} ece {m['ece']:.4f}")
PY

echo "== upload =="
hf upload "$REPO" "$STAGE" . --repo-type model --commit-message "$NAME: adapter, calibration, predictions, eval results"
echo "== done =="
echo "model card: https://huggingface.co/$REPO"
echo "leaderboard: open a discussion on https://huggingface.co/datasets/LocalLLaMA/typed-decisions with the four numbers"
rm -rf "$STAGE"
