# Problem 6 Synthesis Core Push Results

Date: 2026-05-20

## Purpose

This experiment directly tested the suspected implementation bottleneck in D-CGFS synthesis:

1. task logits in synthesis should depend on the synthetic feature-derived concept probabilities, not only on base concept probabilities;
2. target discriminative concept features should be injected into the base structure more explicitly;
3. saved synthetic samples should carry mixed target/base concept embeddings on discriminative concepts.

## Compared Methods

| Method | Fusion | Synthesis task concept source | Training task concept source |
|---|---|---|---|
| `dcgfs_target_score_w015` | previous main | base | base |
| `dcgfs_aligned_pred_disc_mix` | aligned pooled target feature | predicted discriminative mix | predicted discriminative mix |
| `dcgfs_aligned_step2_only` | aligned pooled target feature | predicted discriminative mix | base |
| `dcgfs_old_spatial_pred_disc_mix` | old spatial feature mix | predicted discriminative mix | predicted discriminative mix |

## Main Metrics

| Method | Overall | Macro-F1 | Balanced Acc | Worst Class | Target Acc | Base Acc | Target->Base |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline SSCBM | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 |
| Current main `target_score_w015` | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 |
| `aligned_pred_disc_mix` | 65.79 | 65.79 | 65.83 | 10.00 | 40.71 | 76.92 | 21.43 |
| `aligned_step2_only` | 65.72 | 65.74 | 65.78 | 6.67 | 42.14 | 77.78 | 22.14 |
| `old_spatial_pred_disc_mix` | 66.14 | 66.22 | 66.20 | 6.67 | 42.86 | 77.78 | 20.71 |

## Synthesis Diagnostics

The new semantic-closed-loop synthesis clearly changed candidate scoring:

| Method | Max target probability in synthetic candidates |
|---|---:|
| Previous `target_score_w015` family | about 0.0181 |
| `aligned_pred_disc_mix` / `aligned_step2_only` | 0.1931 |
| `old_spatial_pred_disc_mix` | 0.1132 |

This confirms the code-level diagnosis: the previous synthesis scoring path was too weakly coupled to the synthetic feature. After making task logits depend on synthetic feature-derived concepts, the synthetic candidates can produce much stronger target-class evidence.

## Interpretation

The code fix is directionally correct, but the hard `predicted_disc_mix` setting is not yet the best final method. It improves the synthetic target-confidence signal, yet end-to-end target accuracy is lower than the current main configuration.

Most likely cause:

1. using predicted concept probabilities on discriminative concepts with weight 1.0 injects noisy concept estimates too aggressively;
2. `aligned_pool` may erase spatial details that are useful for some target-base pairs;
3. the stronger target signal in top-k selection does not automatically mean better training samples if concept prediction noise and repeated source pairs remain.

## Decision

Do not replace the current main method yet. Keep `dcgfs_target_score_w015` as the official main configuration.

Next experiment: test soft discriminative concept injection with `synthesis_disc_mix_weight` / `synthetic_disc_mix_weight` below 1.0. This keeps the corrected semantic loop while reducing noisy hard replacement.
