# Findings

Every Typed Decisions number here is recomputed from raw prediction files with the board's
metric code (`bench/score.py`). Numbers for other models are labelled as board-reported or
recomputed from their public predictions. Pramana's released numbers come from
`results/final/`.

## 0. Two bugs that hid the real numbers

**Writer-side key bug.** `train/run.py` and `train/dump_preds.py` mapped option keys `"0"`
and `"1"` to `"false"`/`"true"` before writing predictions. That renamed levels 0 and 1 of
all 800 `score` decisions, which the scorer then read as zero probability. An earlier fix
corrected the same mapping in the scorer but not in the writers, and `bench/analyze.py` read
the same broken files, so two independent scorers agreed on a wrong 0.723. Repairing the
files (lossless, `bench/repair_keys.py`) moved accuracy to 0.80.

Lesson: scorer agreement is not validation when both scorers read the same file.

**Calibration aimed at the wrong target.** Unsloth's `calibrate()` fits temperature against
the hard label and folded T = 0.343 into the head, sharpening it about 3×. `predict()` also
rounds to 4 decimals, leaving exact zeros that KL punishes. Refitting on the holdout against
the soft gold from raw logits took KL from 0.59 to 0.076 with no retraining.

Everything measured before these fixes (including "ordinality is the bottleneck",
"confidence is inverted" and "not usable for deferral") is withdrawn.

## 1. Where Pramana stands

| model | accuracy | KL ↓ | Brier ↓ | ECE ↓ |
|---|---|---|---|---|
| Pramana-2B (two adapters averaged) | 0.8005 | 0.0755 | 0.0415 | 0.1675 |
| Pramana-2B, adapter `large` alone | 0.7970 | 0.0765 | 0.0422 | 0.1618 |
| Pramana-2B, adapter `lowlr` alone | 0.7940 | 0.0771 | 0.0423 | 0.1618 |
| Pramana-0.8B | 0.7970 | 0.0819 | 0.0446 | 0.1646 |
| sysone-0.4B (board) | 0.8025 | 0.080 | 0.0438 | 0.174 |
| ministral3-3b-reasoning (board) | 0.808 | 0.077 | 0.042 | 0.182 |
| Arbiter-4B (board) | 0.817 | 0.092 | 0.048 | 0.184 |
| Decider-4B (board) | 0.8205 | 0.301 | 0.132 | 0.031 |
| TypeSafe Jev 1.13.0 (board, zero-shot) | 0.727 | 1.442 | 0.148 | 0.144 |

Paired bootstrap, Pramana-2B (two adapters) minus reference, 2,000 resamples of the 400 test
cases, 95% CI:

| vs | accuracy | KL | Brier | ECE |
|---|---|---|---|---|
| Decider-4B | −0.043 to −0.010 | −0.011 to +0.001 | −0.005 to +0.002 | −0.008 to +0.024 |
| sysone-0.4B | −0.018 to +0.014 | −0.009 to +0.000 | −0.005 to +0.000 | −0.022 to +0.009 |
| Arbiter-4B | −0.029 to +0.003 | −0.024 to −0.010 | −0.010 to −0.003 | −0.029 to +0.003 |

Pramana-2B ties sysone on all four, is significantly better than Arbiter-4B on KL and Brier
(tied on accuracy and ECE), and is behind Decider-4B on accuracy. Pramana-0.8B ties sysone
on all four and is better than Arbiter-4B on KL but behind it on accuracy.

The two adapters were trained as candidates for a pre-registered holdout selection, which
picked `large`; averaging both was decided afterwards, so the averaged model's test numbers
were seen before that choice. Its gain over `large` alone (+0.0035 accuracy) is within noise. Three seeds of the same recipe give accuracy
0.797 ± 0.005, KL 0.0762 ± 0.0007, Brier 0.0418 ± 0.0006; every seed is below sysone on KL
and Brier, but no single seed is significantly so. An earlier seed reached 0.8025 accuracy;
that was the top of the seed range, not the recipe's level.

## 2. Decoupled calibration

| model | KL | ECE of `probabilities` | ECE of `confidence` |
|---|---|---|---|
| Pramana-2B (two adapters) | 0.0755 | 0.168 | 0.0379 |
| Pramana-2B, `large` alone | 0.0765 | 0.162 | 0.0385 |
| Pramana-0.8B | 0.0819 | 0.165 | 0.0249 |

Both temperature sets are fitted per question type on the holdout. `probabilities` keeps the
soft-gold fit (T ≈ 1.0); `confidence` uses the hard-label fit (T ≈ 0.31 to 0.42). No model
on the board reports both low KL and low ECE from one output.

## 3. The board's metrics pull against each other

Rescaling each model's distribution by temperature traces its KL-versus-ECE curve:

| model | min-KL point | min-ECE point |
|---|---|---|
| Pramana-2B (earlier seed) | KL 0.076, ECE 0.169 | KL 0.497, ECE 0.032 |
| Decider-4B | KL 0.072, ECE 0.198 | KL 0.299, ECE 0.029 |
| sysone-0.4B | KL 0.080, ECE 0.174 | KL 0.489, ECE 0.028 |
| Arbiter-4B | KL 0.092, ECE 0.180 | KL 0.534, ECE 0.027 |

The soft gold is less confident than the teacher is accurate, so a single distribution
trades one metric for the other, and ranking partly depends on a temperature its authors
picked. Decider-4B at T = 0.5 gives KL 0.299 and ECE 0.029, close to its submission (0.301,
0.031); Brier (0.098 vs 0.132) and accuracy do not match exactly, so this is consistent with,
not proof of, one model reported at two temperatures.

## 4. Training without a calibration step

Both released models were trained with `--skip-calibrate`. Their fitted `probabilities`
temperatures are 1.00 to 1.05, and Pramana-0.8B's holdout KL is 0.0747 before and after the
fit. Soft-gold training with a Brier term (weight 1.0) already yields soft-calibrated output;
the hard-label step is what broke it.

## 5. 0.8B to 2B cascade

Pramana-0.8B answers when `max(probabilities) ≥ 0.68`, otherwise Pramana-2B (two adapters)
answers. The threshold is the lowest whose holdout accuracy matches Pramana-2B's
(`bench/ensemble.py`). Board scorer:

| | accuracy | KL | Brier | share sent to 2B |
|---|---|---|---|---|
| Pramana-0.8B alone | 0.7970 | 0.0819 | 0.0446 | 0% |
| Pramana-2B alone | 0.8005 | 0.0755 | 0.0415 | 100% |
| cascade | 0.8000 | 0.0783 | 0.0425 | 66.8% |

## 6. Ordinal neighbour-smoothing layer

On `score` questions, each level keeps (1 − α) of its mass and passes α/2 to each neighbour,
with α fitted on the holdout (0.02 to 0.03). On an earlier seed: KL 0.0844 → 0.0833, Brier
0.0421 → 0.0416, accuracy unchanged. Small and consistent; not part of the released outputs.

## 7. Ablations

| change | test accuracy | verdict |
|---|---|---|
| Brier 0.1 + field shuffling, 250 steps vs Brier 1.0, 400 steps | 0.784 vs 0.801 | second better on every metric; several changes at once, so confounded |
| ordinal weight 1.0 + label smoothing 0.05 | 0.7965 | no accuracy gain |
| sequence length 2,048 (2B) | 0.7950 | no gain on long-log workflows; truncation is not the cause |
| LoRA rank 64, LR 3e-4 | 0.7855 | overfits: best holdout (0.8325), worst test |
| LR 1e-4 | 0.7940 | worse holdout KL (0.0726 vs 0.0699), rejected by the pre-registered rule; test agreed |

Across every run, single models land between 0.784 and 0.803. The entries above us on accuracy
(Decider-4B, Arbiter-4B) report training on "train and other decision data"; more data, not
more tuning, is the remaining lever.

## 8. Axes the board does not measure

Pramana-2B numbers are from an earlier seed of the same recipe, after both fixes; the
other models' numbers come from their public prediction files (`bench/analyze.py`):

- **Selective accuracy.** Pramana-2B: 0.943 at 50% coverage, 0.868 at 80%, selective AUC
  0.927 (Decider-4B 0.932).
- **Review head is not uncertainty.** Ranking by the learned `needs_review` probability loses
  8 to 10 points against plain confidence at matched coverage (Decider-4B, sysone-0.4B).
- **Mondrian conformal.** Pooled 90% coverage hides per-type under-coverage (Decider-4B
  minimum 0.880 on `noul`; sysone-0.4B 0.863 on `choice`).
- **Cost-optimal deferral.** At an error-to-review cost of 5:1, Pramana-2B covers 49% of
  decisions at 94.8% accuracy (earlier seed).

## Withdrawn claims

- 0.723 accuracy, KL 3.10 and everything derived from them (scorer-key bug).
- "Beats Decider-4B on KL and Brier": a statistical tie, and Decider-4B wins at its own best
  temperature.
- 0.83 accuracy: a holdout number from the rank-64 run, which scored 0.7855 on test.
