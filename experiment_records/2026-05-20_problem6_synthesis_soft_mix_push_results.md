# Problem 6 Synthesis Soft Mix Push Results

Date: 2026-05-20

## Purpose

`synthesis_core_push` showed that hard `predicted_disc_mix` makes synthetic candidates much more target-sensitive, but does not improve final target accuracy over the current main configuration. This experiment tested whether softer discriminative concept injection can keep the semantic loop while reducing prediction noise.

## Main Metrics

| Method | Overall | Macro-F1 | Balanced Acc | Worst Class | Target Acc | Base Acc | Target->Base |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current main `target_score_w015` | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 |
| `old_spatial_pred_disc_mix_w025` | 65.90 | 65.90 | 65.95 | 6.67 | 42.86 | 76.07 | 20.71 |
| `old_spatial_pred_disc_mix_w050` | 65.81 | 65.79 | 65.87 | 6.67 | 41.43 | 76.92 | 22.14 |
| `aligned_pred_disc_mix_w025` | 65.77 | 65.77 | 65.82 | 6.67 | 42.86 | 76.92 | 21.43 |
| `old_spatial_step2_soft_w025` | 65.76 | 65.72 | 65.83 | 6.67 | 43.57 | 76.07 | 21.43 |

## Pair-Level Diagnosis

| Method | Pair0 | Pair1 | Pair2 | Pair3 | Pair4 |
|---|---:|---:|---:|---:|---:|
| Current main | 15.00 | 26.67 | 23.33 | 80.00 | 66.67 |
| `old_spatial_pred_disc_mix_w025` | 10.00 | 26.67 | 20.00 | 80.00 | 66.67 |
| `old_spatial_pred_disc_mix_w050` | 20.00 | 26.67 | 20.00 | 76.67 | 56.67 |
| `aligned_pred_disc_mix_w025` | 15.00 | 26.67 | 20.00 | 76.67 | 66.67 |
| `old_spatial_step2_soft_w025` | 20.00 | 26.67 | 16.67 | 83.33 | 63.33 |

## Interpretation

Soft injection did not beat the current main method. The reason is now clearer:

1. `w=0.25/0.50` suppresses the hard semantic-loop target evidence; max target probability falls back to about `0.008-0.009`.
2. Pair2 remains the most problematic target-base pair, with high target-to-base confusion around `43-47%` in several variants.
3. Pair0 and pair2 are not fixed by fusion/task-concept mixing, which suggests that the discriminative concept set D itself is too weak or too sparse.

## Follow-Up

The next experiment should keep the same target/base pairs but refine `discriminative_concepts` using model-aware concept prototypes:

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group model_aware_d_push
```

This directly tests whether the current bottleneck is the quality of D rather than the scalar fusion or loss-weight settings.
