---
license: apache-2.0
base_model: unsloth/Qwen3.5-2B
datasets:
  - LocalLLaMA/typed-decisions
tags:
  - decision-model
  - calibration
  - typed-decisions
---

<div align="center">

# Pramana-2B

**Small, calibrated models for structured decisions.**

[![Evaluation](https://img.shields.io/badge/Evaluated%20on-Typed%20Decisions-blue)](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
[![Pramana-0.8B](https://img.shields.io/badge/Sibling-Pramana--0.8B-green)](https://huggingface.co/{{USER}}/pramana-0.8b)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

## 1. Introduction

Pramana-2B is a decision model for structured workflows. Given the state of a case (an agent
trace, a customer thread, an invoice, a security alert) and a set of typed questions (`noul`
yes/no, `choice` one-of-N, `score` ordered rubric), it returns a probability distribution over
every option for every question in one forward pass, plus a separately calibrated `confidence`.

It is trained and evaluated on [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
and covers its four workflows: agent-trace review, customer service, invoice processing and
security incidents. Its strength is distributional quality: KL 0.0755 and Brier 0.0415 against the soft gold.
Its accuracy (0.8005) is level with the small fine-tuned models and below the best 4B entries.

## 2. Method

**Decoupled calibration.** Against this benchmark's soft gold, KL and top-label ECE have
different optimal temperatures. For each public model we checked, rescaling one distribution
reaches low KL or low ECE, not both (Decider-4B: KL 0.072 at ECE 0.198, or KL 0.299 at ECE
0.029). Pramana therefore emits two calibrations of the same logits:

- `probabilities = softmax(z / T_p)`, with `T_p` fitted to the soft gold. This is the scored output.
- `confidence = max softmax(log probabilities / T_c)`, with `T_c` fitted to the hard label.
  ECE 0.038, against 0.168 for `max(probabilities)`.

**Soft-gold training without post-hoc calibration.** Training uses cross-entropy against the
soft gold plus a Brier term (weight 1.0). The fitted `T_p` is 1.00 to 1.05, so no correction is
needed. Fitting temperature to the hard label after soft-gold training, a common default,
sharpened an earlier run of this model about 3× (T = 0.343) and raised KL from 0.076 to 0.50.

**Two-adapter average.** Pramana-2B averages the probabilities of two LoRA adapters on the same
base (learning rates 2e-4 and 1e-4; 0.7970 and 0.7940 accuracy alone). Inference costs two
2B passes. Averaging was decided after both adapters had been scored on test; the accuracy gain
over the better adapter (+0.0035) is within noise, while the KL and Brier gains are consistent.

**Cascade with Pramana-0.8B.** Routing to Pramana-2B only when Pramana-0.8B's top probability
is below 0.68 sends 66.8% of decisions to the 2B at accuracy 0.8000 (KL 0.0783, Brier 0.0425).

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
| **Pramana-2B** | 2B | 0.8005 | **0.0755** | **0.0415** | 0.168 (0.038 `confidence`) |
| Pramana-0.8B | 0.8B | 0.7970 | 0.0819 | 0.0446 | 0.165 (**0.025** `confidence`) |
| Decider-4B | 4B | **0.8205** | 0.301 | 0.132 | 0.031 |
| Arbiter-4B | 4B | 0.817 | 0.092 | 0.048 | 0.184 |
| Darwin-27B-ZTC-v5 | 27B | 0.812 | – | – | – |
| ministral3-3b-reasoning | 3B | 0.808 | 0.077 | 0.042 | 0.182 |
| sysone-0.4B | 0.4B | 0.8025 | 0.080 | 0.0438 | 0.174 |
| OpenDecider-large-td | 80B-A3B | 0.801 | 0.081 | 0.044 | – |
| od1-typed-decisions | 4B | 0.797 | 0.082 | 0.045 | 0.156 |
| OpenDecider-nano | 0.4B | 0.796 | 0.079 | 0.043 | – |

Paired bootstrap, Pramana-2B minus reference, 2,000 resamples of the 400 test cases, 95%
intervals. References are computed from their public prediction files; Decider-4B's public
predictions score 0.827 / 0.080 / 0.043 / 0.160, not its leaderboard submission above.

| vs | Accuracy | KL | Brier | ECE |
|---|---|---|---|---|
| Decider-4B (public predictions) | −0.043 to −0.010 | −0.011 to +0.001 | −0.005 to +0.002 | −0.008 to +0.024 |
| Arbiter-4B | −0.029 to +0.003 | **−0.024 to −0.010** | **−0.010 to −0.003** | −0.029 to +0.003 |
| sysone-0.4B | −0.018 to +0.014 | −0.009 to +0.000 | −0.005 to +0.000 | −0.022 to +0.009 |

Pramana-2B is significantly better than Arbiter-4B on KL and Brier, statistically tied with
sysone-0.4B on all four metrics, and significantly behind Decider-4B on accuracy. Models
without public predictions (ministral3-3b, OpenDecider) cannot be tested this way; their
listed KL and Brier are within 0.006 of Pramana-2B's.

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

Pramana is not a zero-shot model. On out-of-domain intent classification it transfers poorly:
an earlier adapter scored 0.572 on CLINC150 and 0.255 on Banking77.

### 4.3 Notes

- Gold is the mean of three samples from a ~4B teacher with self-agreement 0.735; accuracy
  above that partly measures agreement with the teacher.
- Single runs of this recipe range from 0.784 to 0.803 accuracy (three seeds: 0.797 ± 0.005,
  KL 0.0762 ± 0.0007). Ablations that did not improve accuracy: ordinal loss weight 1.0,
  sequence length 2,048, LoRA rank 64 (overfits), learning rate 1e-4 alone.

## 5. How to Run Locally

```python
import json
import torch
from huggingface_hub import snapshot_download
from unsloth import FastDecisionModel

root = snapshot_download("{{USER}}/pramana-2b")
calib = json.load(open(f"{root}/calibration.json"))
members = []
for name in calib["members"]:
    model, tok = FastDecisionModel.from_pretrained(f"{root}/adapters/{name}", load_in_4bit=True)
    FastDecisionModel.for_inference(model)
    members.append((model, tok))

def decide(state, questions):
    answers = [FastDecisionModel.predict(m, t, state, questions) for m, t in members]
    out = {}
    for q, spec in questions.items():
        opts = list(answers[0][q]["probabilities"])
        p = torch.tensor([[a[q]["probabilities"][o] for o in opts] for a in answers]).mean(0)
        t = calib["confidence_temperature"][spec["type"]]
        conf = torch.softmax(torch.log(p.clamp_min(1e-12)) / t, -1).max().item()
        out[q] = {"probabilities": dict(zip(opts, p.tolist())), "confidence": conf}
    return out
```

`state` and `questions` follow the Typed Decisions request format. `predict()` serves these
adapters at temperature 1.0 and rounds to four decimals; exact scored values are reproduced
from raw logits by the source repository.

**Training configuration.** Train split only (1,200 cases; 80 held out). LoRA rank 16, alpha 16,
4-bit base, 400 steps (~2.9 epochs), batch 8, cosine schedule, warmup 10, weight decay 0.01,
sequence length 1,024. Loss: soft-gold cross-entropy + Brier (1.0) + ordinal (0.1). One A100 80GB.

## 6. Limitations

- Answers only the four workflows it was trained on.
- Scores measure agreement with a teacher model, not ground truth.
- 80 held-out cases (400 decisions) is a small calibration set; validate `confidence`
  thresholds and the cascade on production traffic.

## 7. License

Adapters: Apache 2.0. Base model Qwen3.5-2B: Apache 2.0.

## 8. Citation

```bibtex
@misc{pramana2026,
  title  = {Pramana: Open-Source Decision Models},
  author = {{{USER}}},
  year   = {2026},
  url    = {https://huggingface.co/{{USER}}/pramana-2b}
}
```

## 9. Acknowledgements

- Base model: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (Qwen team), via
  [unsloth/Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B).
- Training framework and decision head: [Unsloth](https://github.com/unslothai/unsloth)
  `DecisionTrainer` / `FastDecisionModel`, version 2026.10.7.
- Data, metric definitions and comparison figures:
  [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (LocalLLaMA).
