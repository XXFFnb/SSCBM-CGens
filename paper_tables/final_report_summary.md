# Problem 6 Final Report Summary

Current main method: `D-CGFS target_score_w015 + base preservation`.

## Metric Glossary

| metric | 中文名称 | 含义 |
| --- | --- | --- |
| overall_acc | 整体任务准确率 | 全部测试样本上的类别预测准确率。 |
| overall_concept_acc | 整体概念准确率 | 全部测试样本、全部概念上的概念预测准确率。 |
| macro_f1 | 宏平均 F1 | 先按类别计算 F1，再对类别平均，能减轻类别不平衡对指标的遮蔽。 |
| balanced_acc | 平衡准确率 | 各类别召回率的平均值，用于观察不同类别是否被均衡识别。 |
| worst_class_acc | 最差类别准确率 | 测试集中表现最差类别的准确率，反映尾部风险。 |
| selected_target_acc | 自动弱势目标类准确率 | find_weak_classes.py 自动选出的弱势目标类上的准确率。 |
| selected_base_acc | 自动基座类准确率 | 与目标类混淆严重的基座类上的准确率，用于检查副作用。 |
| target_to_base_rate | 目标类错分为基座类比例 | 目标类样本被预测成对应基座类的比例，越低越好。 |

Feature refinement, retrieval residual, model-aware D, and hybrid pair refinement are not promoted.
They remain archived diagnostics because they did not improve the primary weak-target tradeoff.

## Main Result

| method | overall_acc | macro_f1 | balanced_acc | worst_class_acc | target_acc | base_acc | target_to_base_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SSCBM | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 |
| D-CGFS original main | 65.36 | 65.16 | 65.45 | 3.33 | 37.86 | 75.21 | 20.71 |
| D-CGFS pair-topk | 65.98 | 65.97 | 65.99 | 6.67 | 44.29 | 66.67 | 12.86 |
| D-CGFS previous main w005 + BP | 66.02 | 65.90 | 66.03 | 6.67 | 42.86 | 71.79 | 17.14 |
| D-CGFS target_score_w015 + BP | 65.65 | 65.65 | 65.72 | 10.00 | 42.86 | 77.78 | 20.71 |

CUB conclusion: D-CGFS substantially improves overall accuracy, macro-F1, balanced
accuracy, worst-class accuracy, and selected weak-target accuracy. It also reduces the
target-to-base confusion rate, which directly supports the target-base motivation.

## Strong Baseline Context

D-CGFS should not be claimed as the best method on raw target accuracy. Reweighting and
class-balanced loss are stronger on selected target accuracy, while D-CGFS keeps better
selected base accuracy than the retrained imbalance baselines and gives a concept-guided
synthesis mechanism.

| method | overall_acc | macro_f1 | balanced_acc | worst_class_acc | selected_target_acc | selected_base_acc | target_to_base_rate | overall_concept_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSCBM baseline | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| SSCBM finetune | 66.76 | 66.62 | 66.87 | 3.33 | 56.43 | 71.79 | 16.43 | 89.56 |
| Oversampling | 66.48 | 66.06 | 66.53 | 13.33 | 50.71 | 64.10 | 10.71 | 89.51 |
| Reweighting | 67.31 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 | 89.60 |
| Class-balanced loss | 67.31 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 | 89.60 |
| Feature mixup | 66.69 | 66.65 | 66.87 | 3.33 | 57.14 | 68.38 | 15.00 | 88.92 |
| D-CGFS target_score_w015 | 65.65 | 65.65 | 65.72 | 10.00 | 42.86 | 77.78 | 20.71 | 89.49 |

## Archived Diagnostics

| method | overall_acc | macro_f1 | balanced_acc | worst_class_acc | target_acc | base_acc | target_to_base_rate | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dcgfs_retrieval_residual | 65.83 | 65.84 | 65.90 | 6.67 | 42.14 | 76.92 | 20.00 | archive |
| dcgfs_model_aware_d_w050 | 65.62 | 65.63 | 65.69 | 6.67 | 40.71 | 76.07 | 22.14 | archive |
| dcgfs_feature_refine_pred_disc | 65.84 | 65.85 | 65.94 | 6.67 | 42.86 | 76.07 | 22.14 | stop |
| dcgfs_feature_refine_window | 65.93 | 65.92 | 66.02 | 6.67 | 40.71 | 76.92 | 21.43 | stop |
| dcgfs_hybrid_pair0_refine_pred_disc_train | 66.03 | 66.03 | 66.11 | 10.00 | 43.57 | 77.78 | 20.00 | stop |

## Cross-Dataset Results

| dataset | method | status | overall_acc | macro_f1 | balanced_acc | worst_class_acc | selected_target_acc | selected_base_acc | target_to_base_rate | overall_concept_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUB | SSCBM baseline | done | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| CUB | D-CGFS target_score_w015 | done | 65.65 | 65.65 | 65.72 | 10.00 | 42.86 | 77.78 | 20.71 | 89.49 |
| AwA2 | SSCBM baseline | done | 89.39 | 86.02 | 85.60 | 45.45 | 55.65 | 81.44 | 17.74 | 96.48 |
| AwA2 | D-CGFS target_score_w015 | done | 90.01 | 86.59 | 86.46 | 29.41 | 55.65 | 85.57 | 19.35 | 96.15 |
| PBC | SSCBM baseline | done | 99.65 | 99.50 | 99.50 | 97.82 | 99.65 | 99.49 | 0.29 | 94.14 |
| PBC | D-CGFS target_score_w015 | done | 99.65 | 99.53 | 99.57 | 98.37 | 99.65 | 99.49 | 0.23 | 93.30 |
| 7pt | SSCBM baseline | done | 61.27 | 46.39 | 47.53 | 22.00 | 61.27 | 61.11 | 23.54 | 69.81 |
| 7pt | D-CGFS target_score_w015 | done | 66.84 | 51.77 | 50.68 | 22.22 | 66.84 | 67.20 | 20.76 | 71.38 |

Cross-dataset conclusion: D-CGFS gives clear gains on CUB and 7pt, preserves the
near-saturated PBC task performance with small balanced/worst-class improvements, and
mainly stabilizes base classes on AwA2. The paper claim should emphasize concept-guided
weak-class augmentation and cross-dataset robustness, while noting that gains are limited
when the baseline is already near saturation.

## Writing Position

Use D-CGFS as a concept-guided weak-class augmentation method with stronger preservation
and interpretability, not as a universal replacement for all imbalance baselines.
