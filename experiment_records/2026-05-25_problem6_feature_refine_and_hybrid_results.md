# 2026-05-25 Problem 6 Feature Refinement / Hybrid Results

## Main Baseline To Beat

`dcgfs_target_score_w015`

- overall_a_acc: 65.9993%
- macro_f1: 65.9593%
- balanced_accuracy: 66.0369%
- worst_class_acc: 10.00%
- selected_target_acc: 44.2857%
- selected_base_acc: 78.6325%
- target_to_base_rate: 19.2857%

## Feature Refinement

`dcgfs_feature_refine_pred_disc`

- overall_a_acc: 65.8440%
- macro_f1: 65.8542%
- balanced_accuracy: 65.9388%
- selected_target_acc: 42.8571%
- selected_base_acc: 76.0684%
- target_to_base_rate: 22.1429%

Conclusion: feature refinement makes synthetic candidates much stronger in the frozen model
space, but hurts final target/base tradeoff. It behaves like an overly model-specific feature
optimization rather than a robust augmentation.

`dcgfs_feature_refine_window`

- overall_a_acc: 65.9303%
- macro_f1: 65.9247%
- balanced_accuracy: 66.0165%
- selected_target_acc: 40.7143%
- selected_base_acc: 76.9231%
- target_to_base_rate: 21.4286%

Conclusion: filtering out saturated refined samples does not recover target performance.

## Pair-Wise Hybrid

`dcgfs_hybrid_pair0_refine_base_train`

- overall_a_acc: 65.8785%
- macro_f1: 65.8288%
- balanced_accuracy: 65.9507%
- selected_target_acc: 41.4286%
- selected_base_acc: 76.9231%
- target_to_base_rate: 20.0000%

`dcgfs_hybrid_pair0_refine_pred_disc_train`

- overall_a_acc: 66.0338%
- macro_f1: 66.0336%
- balanced_accuracy: 66.1059%
- selected_target_acc: 43.5714%
- selected_base_acc: 77.7778%
- target_to_base_rate: 20.0000%

Hybrid pred-disc slightly improves overall/macro/balanced accuracy over the current main line,
but still loses on the primary weak-target metric.

Per-target comparison:

| run | cls59 | cls65 | cls74 | cls144 | cls176 | target correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| target_score_w015 | 8/30 | 3/20 | 24/30 | 7/30 | 20/30 | 62/140 |
| feature_refine_pred_disc | 8/30 | 5/20 | 24/30 | 6/30 | 17/30 | 60/140 |
| feature_refine_window | 6/30 | 3/20 | 24/30 | 7/30 | 17/30 | 57/140 |
| hybrid_pair0_refine_pred_disc_train | 8/30 | 4/20 | 23/30 | 7/30 | 19/30 | 61/140 |

## Decision

Do not promote feature refinement or hybrid refinement as the main method.

Current main remains:

`dcgfs_target_score_w015`

Feature refinement can be described as a negative/diagnostic experiment: it confirms that
frozen-model target evidence can be injected into synthetic features, but those features do not
transfer cleanly to real test images and can introduce distribution-shift noise.
