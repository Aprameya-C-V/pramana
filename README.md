# Pramana: Open-Source Decision Models

**Small, calibrated models for structured decisions.** Pramana-2B and Pramana-0.8B return a
probability for every option plus a separate confidence signal.

Pramana answers typed questions about one piece of unstructured state and returns a
probability over every option, not a single label. This repository holds the training
code, the evaluation code, and every number behind the model cards.

Models: [apracv/pramana-2b](https://huggingface.co/apracv/pramana-2b) ·
[apracv/pramana-0.8b](https://huggingface.co/apracv/pramana-0.8b)

Evaluation: [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(400 test cases, 2,000 decisions, 4 workflows, 5 typed questions per case). Pramana is
trained on its train split, so results are compared with other fine-tuned models.

## Results

Test split, computed by `bench/score.py` from the prediction files in `results/final/`
(`pramana2b/preds.jsonl`, `small/preds_temper.jsonl`).

| model | params | accuracy | KL ↓ | Brier ↓ | ECE ↓ `probabilities` | ECE ↓ `confidence` |
|---|---|---|---|---|---|---|
| **Pramana-2B** (two adapters, one base) | 2.3B | 0.8005 | **0.0755** | **0.0415** | 0.1675 | 0.0379 |
| **Pramana-0.8B** | 0.8B | 0.7970 | 0.0819 | 0.0446 | 0.1646 | **0.0249** |
| sysone-0.4B (board) | 0.4B | 0.8025 | 0.080 | 0.0438 | 0.174 | – |
| ministral3-3b-reasoning (board) | 3B | 0.808 | 0.077 | 0.042 | 0.182 | – |
| Arbiter-4B (board) | 4B | 0.817 | 0.092 | 0.048 | 0.184 | – |
| Decider-4B (board) | 4.2B | 0.8205 | 0.301 | 0.132 | 0.031 | – |

Pramana-2B has the lowest KL and Brier of any fine-tuned entry listed; accuracy sits with
the 4B entries. It averages two adapters on one 2B base (0.797 and 0.794 alone). `confidence` is a second calibration of the
same logits (see below).

## What is new here

1. **Decoupled calibration.** One set of logits, two outputs: `probabilities` at a
   temperature fitted to the soft gold (KL, Brier) and `confidence` at a temperature fitted
   to the hard label (ECE, deferral). Both are fitted on held-out train rows and shipped in
   `calibration.json`.
2. **No calibration step.** Soft-gold training with a Brier term needs no post-hoc
   temperature (fitted value 1.00 to 1.05). The library's hard-label `calibrate()` sharpened
   an earlier run about 3× and raised KL from 0.076 to 0.50.
3. **0.8B to 2B cascade.** Pramana-0.8B answers when its top probability is at least 0.68
   (chosen on holdout), otherwise Pramana-2B does: one decision in three never reaches the
   2B, and accuracy holds at 0.800.
4. **Board metrics conflict.** No temperature minimises both KL and ECE for any public
   model, which also explains why Decider-4B's submission and repo disagree.
5. Four evaluation axes the board does not measure: selective accuracy, Mondrian conformal
   coverage, ordinal error, cost-optimal deferral.

## Repo layout

```
train/run.py          train (--skip-calibrate --skip-predict for the release recipe)
train/dump_logits.py  raw per-question logits for holdout and test, no rounding
bench/temper.py       fit the probabilities temperature on holdout, write test predictions
bench/decoupled.py    fit both temperatures, write calibration.json
bench/ensemble.py     average adapters, fit both temperatures and the cascade threshold on holdout
bench/cascade.py      0.8B to 2B confidence cascade, threshold sweep
bench/ordinal_layer.py  neighbour-smoothing output layer for score questions
bench/frontier.py     paired bootstrap and the KL-versus-ECE temperature frontier
bench/score.py        board-identical metrics, per workflow and per question
bench/analyze.py      selective, conformal, ordinal, decomposition, deferral
jobs/final.sh         the release run on one A100; jobs/final_local.sh drives it from a laptop
hf/publish.sh         publish one model with adapter, calibration, predictions, eval YAML
results/final/        adapters, logits, predictions and logs for the released models
```

## Reproduce

```bash
# score a prediction file with the board's metric definitions
uv run --with pyarrow python bench/score.py results/final/pramana2b/preds.jsonl

# both calibrations from the shipped logits
uv run --with torch --with pyarrow python bench/decoupled.py results/final/large/logits.pt

# retrain both models on a fresh A100 pod and pull everything back
bash jobs/final_local.sh <ip> <port>
```

## Notes on comparability

- **Fine-tuned versus zero-shot.** Pramana is trained on these four workflows, so its scores
  are compared with other fine-tuned models, not zero-shot rows.
- **Decider-4B.** Its submission (0.8205, KL 0.301, ECE 0.031) and its own `report.json`
  (0.826, KL 0.0803, ECE 0.1598) disagree. Sharpening its published predictions to
  temperature 0.5 reproduces the submission's KL and ECE, consistent with one model
  reported at two temperatures.
- **Teacher ceiling.** Gold is the mean of three samples from a ~4B teacher with
  self-agreement 0.735; scores above that partly measure agreement with the teacher.
