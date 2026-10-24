---
license: apache-2.0
base_model: unsloth/Qwen3.5-0.8B
datasets:
  - LocalLLaMA/typed-decisions
tags:
  - decision-model
  - calibration
  - typed-decisions
---

<div align="center">

# Pramana-0.8B

**Small, calibrated models for structured decisions.**

[![Evaluation](https://img.shields.io/badge/Evaluated%20on-Typed%20Decisions-blue)](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
[![Pramana-2B](https://img.shields.io/badge/Sibling-Pramana--2B-green)](https://huggingface.co/{{USER}}/pramana-2b)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

## 1. Introduction

Pramana-0.8B is a decision model for structured workflows. Given the state of a case (an agent
trace, a customer thread, an invoice, a security alert) and a set of typed questions (`noul`
yes/no, `choice` one-of-N, `score` ordered rubric), it returns a probability distribution over
every option for every question in one forward pass, plus a separately calibrated `confidence`.

It is trained and evaluated on [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
and covers its four workflows: agent-trace review, customer service, invoice processing and
security incidents. It is the smaller member of the Pramana pair: accuracy 0.7970, KL 0.0819, Brier 0.0446,
and a `confidence` ECE of 0.025.

## 2. Method

**Decoupled calibration.** Against this benchmark's soft gold, KL and top-label ECE have
different optimal temperatures, so one distribution cannot minimise both. Pramana emits two
calibrations of the same logits:

- `probabilities = softmax(z / T_p)`, with `T_p` fitted to the soft gold. This is the scored output.
- `confidence = max softmax(log probabilities / T_c)`, with `T_c` fitted to the hard label.
  ECE 0.025, against 0.165 for `max(probabilities)`.

**Soft-gold training without post-hoc calibration.** Training uses cross-entropy against the
soft gold plus a Brier term (weight 1.0). Holdout KL is 0.0747 before and after fitting `T_p`,
so no correction is needed.

**Cascade with Pramana-2B.** Pramana-0.8B answers when its top probability is at least 0.68
and defers the rest to Pramana-2B: 33.2% of decisions are settled by the 0.8B alone, at an
overall accuracy of 0.8000.

`T_p`, `T_c` and the cascade threshold are fitted on 80 training cases held out from training,
never on test, and are shipped in `calibration.json`.

## 3. Model Downloads

| Model | Base | Adapters | Accuracy | KL | Brier | Download |
|---|---|---|---|---|---|---|
| Pramana-2B | Qwen3.5-2B | 2 | 0.8005 | 0.0755 | 0.0415 | [🤗](https://huggingface.co/{{USER}}/pramana-2b) |
| Pramana-0.8B | Qwen3.5-0.8B | 1 | 0.7970 | 0.0819 | 0.0446 | [🤗](https://huggingface.co/{{USER}}/pramana-0.8b) |

## 4. Evaluation

Test split: 400 cases, 2,000 decisions. Metrics follow the benchmark's definitions: accuracy
(argmax agreement with gold), KL(gold ‖ prediction), multi-class Brier, and 10-bin top-label
ECE. Pramana numbers are computed from `preds.jsonl` in this repository. Best value per column
in **bold**.

### 4.1 Fine-tuned models (like-for-like)

All models below were trained on the train split. Rows other than Pramana are as reported on
the benchmark leaderboard.

| Model | Params | Accuracy ↑ | KL ↓ | Brier ↓ | ECE ↓ |
|---|---|---|---|---|---|
| **Pramana-0.8B** | 0.8B | 0.7970 | 0.0819 | 0.0446 | 0.165 (**0.025** `confidence`) |
| Pramana-2B | 2B | 0.8005 | **0.0755** | **0.0415** | 0.168 (0.038 `confidence`) |
| Arbiter-4B | 4B | **0.817** | 0.092 | 0.048 | 0.184 |
| sysone-0.4B | 0.4B | 0.8025 | 0.080 | 0.0438 | 0.174 |
| od1-typed-decisions | 4B | 0.797 | 0.082 | 0.045 | 0.156 |
| OpenDecider-nano | 0.4B | 0.796 | 0.079 | 0.043 | – |
| Phocinae-Largha-150M | 0.15B | 0.791 | – | – | – |
| soft-decider-421m | 0.4B | 0.774 | – | – | 0.141 |
| embeddinggemma-2-ft-typed-decision | 0.3B | 0.770 | 0.119 | 0.065 | 0.155 |
| gemma3-270m-typed-decisions | 0.27B | 0.761 | 0.132 | 0.072 | 0.148 |

Paired bootstrap, Pramana-0.8B minus reference, 2,000 resamples of the 400 test cases, 95%
intervals, references computed from their public prediction files:

| vs | Accuracy | KL | Brier | ECE |
|---|---|---|---|---|
| Arbiter-4B | −0.034 to −0.002 | **−0.018 to −0.002** | −0.008 to +0.001 | −0.034 to +0.001 |
| sysone-0.4B | −0.022 to +0.010 | −0.004 to +0.008 | −0.003 to +0.004 | −0.026 to +0.006 |

Pramana-0.8B is significantly better than Arbiter-4B on KL and behind it on accuracy, and
statistically tied with sysone-0.4B on all four metrics. Encoder models near 0.4B
(sysone-0.4B, OpenDecider-nano) match it on accuracy with slightly lower listed KL and Brier.

### 4.2 Zero-shot systems (reference only)

These general systems are scored without training on these workflows and are not comparable
with fine-tuned models; they are listed for scale.

| Model | Access | Accuracy | KL | Brier | ECE |
|---|---|---|---|---|---|
| Quyet-1.0-Large (31B) | open weights | 0.791 | 0.281 | 0.114 | 0.093 |
| meraGPT Decider 1 | API | 0.768 | 0.096 | 0.052 | 0.180 |
| Baseten Mammoth 1 (27B) | API | 0.768 | 0.122 | 0.063 | 0.153 |
| Liquid AI d1 | API | 0.742 | 0.475 | 0.155 | 0.124 |
| TypeSafe Jev 1.13.0 | API | 0.727 | 1.442 | 0.148 | 0.144 |

Pramana is not a zero-shot model and is not expected to transfer outside these workflows.

### 4.3 Notes

- Gold is the mean of three samples from a ~4B teacher with self-agreement 0.735; accuracy
  above that partly measures agreement with the teacher.

## 5. How to Run Locally

```python
import json, math
from huggingface_hub import snapshot_download
from unsloth import FastDecisionModel

root = snapshot_download("{{USER}}/pramana-0.8b")
model, tok = FastDecisionModel.from_pretrained(f"{root}/adapter", load_in_4bit=True)
FastDecisionModel.for_inference(model)
temps = json.load(open(f"{root}/calibration.json"))["confidence_temperature"]

def decide(state, questions):
    answers = FastDecisionModel.predict(model, tok, state, questions)
    for q, spec in questions.items():
        probs = answers[q]["probabilities"]
        z = [math.log(max(p, 1e-12)) / temps[spec["type"]] for p in probs.values()]
        top = max(z)
        answers[q]["confidence"] = 1 / sum(math.exp(v - top) for v in z)
    return answers
```

`state` and `questions` follow the Typed Decisions request format. `predict()` serves this
adapter at temperature 1.0 and rounds to four decimals; exact scored values are reproduced from
raw logits by the source repository.

**Training configuration.** Train split only (1,200 cases; 80 held out). LoRA rank 16, alpha 16,
4-bit base, 400 steps (~2.9 epochs), batch 8, learning rate 3e-4, cosine schedule, warmup 10,
weight decay 0.01, sequence length 2,048. Loss: soft-gold cross-entropy + Brier (1.0) + ordinal
(0.1). One A100 80GB.

## 6. Limitations

- Answers only the four workflows it was trained on.
- Scores measure agreement with a teacher model, not ground truth.
- 80 held-out cases (400 decisions) is a small calibration set; validate `confidence`
  thresholds and the cascade on production traffic.

## 7. License

Adapter: Apache 2.0. Base model Qwen3.5-0.8B: Apache 2.0.

## 8. Citation

```bibtex
@misc{pramana2026,
  title  = {Pramana: Open-Source Decision Models},
  author = {{{USER}}},
  year   = {2026},
  url    = {https://huggingface.co/{{USER}}/pramana-0.8b}
}
```

## 9. Acknowledgements

- Base model: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Qwen team), via
  [unsloth/Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B).
- Training framework and decision head: [Unsloth](https://github.com/unslothai/unsloth)
  `DecisionTrainer` / `FastDecisionModel`, version 2026.10.7.
- Data, metric definitions and comparison figures:
  [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (LocalLLaMA).
