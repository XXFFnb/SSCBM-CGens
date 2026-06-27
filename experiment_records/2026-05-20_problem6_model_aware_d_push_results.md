# Problem 6 Model-Aware D Push Results

Date: 2026-05-20

## Purpose

This experiment tested whether D-CGFS is bottlenecked by the discriminative concept set `D`.

The target/base pairs were kept fixed. Only the discriminative concepts and target/base concept prototypes were recomputed using:

```text
hybrid_proto = 0.5 * CUB_label_proto + 0.5 * SSCBM_predicted_concept_proto
```

## D Changes

| Pair | Old D | New D |
|---:|---|---|
| 0 | `13;111` | `111;13;81;79;38;62;88;61;20;107` |
| 1 | `33;34;108` | `108;33;34;14;29;35;68;94;58;62` |
| 2 | `0;5;67;79;100` | `5;79;0;100;67;74;11;62;68;94` |
| 3 | `2;22;86;100;108` | `2;100;86;108;22;61;83;42;28;19` |
| 4 | `6;12;13;27;31;33;34;56;60;80` | `33;60;89;27;80;56;31;110;12;105` |

## Main Metrics

| Method | Overall | Macro-F1 | Balanced Acc | Worst Class | Target Acc | Base Acc | Target->Base |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current main `target_score_w015` | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 |
| `model_aware_d_w050` | 65.62 | 65.63 | 65.69 | 6.67 | 40.71 | 76.07 | 22.14 |

## Synthesis Diagnostics

Compared with the current main configuration, model-aware D expanded the concept set but reduced final quality:

1. candidate target probability max was only `0.0029`;
2. concept delta mean became negative for pair1, pair2, and pair4;
3. target accuracy dropped from `44.29%` to `40.71%`.

## Interpretation

Simple D expansion is not the solution. The model-aware D selected more concepts, but the resulting masks and concept supervision became noisier. This supports a sharper conclusion:

The current bottleneck is not only which concept indices are selected. It is the synthesis mechanism itself: masked feature mixing does not reliably create target-class-consistent features even when D is changed.

## State Restoration

The experiment temporarily overwrote:

```text
data/D-CGFS_Auxiliary/target_base_pairs.csv
```

The original file was backed up as:

```text
data/D-CGFS_Auxiliary/target_base_pairs.csv.before_model_aware
```

After recording this result, the original `target_base_pairs.csv` was restored to avoid contaminating later experiments.

## Decision

Keep `dcgfs_target_score_w015` as the current main method.

Stop blind sweeps over:

1. class-balanced or target-weighted losses;
2. pair top-k/adaptive top-k;
3. hard/soft predicted concept task mixing;
4. simple model-aware D expansion.

The next useful direction is a synthesis-module rewrite, especially retrieval-based or prototype-aligned feature transplant rather than direct masked feature addition.
