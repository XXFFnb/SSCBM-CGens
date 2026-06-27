# 2026-05-20 Problem6 Target-Push Results

## 运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group target_push
```

## 实验目的

strong baseline 结果显示，当前 D-CGFS 主方法虽然显著优于原始 SSCBM，但 target accuracy 低于 `reweighting` 和 `class_balanced_loss`。因此本次只做诊断性改动，验证两个假设：

1. 合成阶段 pair-topk 的 target confidence 权重偏低，导致进入训练的合成样本对目标类分类边界推动不足。
2. 训练阶段合成样本损失权重偏低，导致模型没有充分学习合成目标类样本。

## 实验设置

| 实验 | 合成数据 | pair-score target weight | synthetic loss weight | base preservation |
| --- | --- | ---: | ---: | --- |
| D-CGFS main w015 | `generated_data/dcgfs_pair_topk` | 0.05 | 0.50 | 0.15 |
| target_score_w015 | `generated_data/problem6_target_score_w015` | 0.15 | 0.50 | 0.15 |
| syn_loss_w075 | `generated_data/dcgfs_pair_topk` | 0.05 | 0.75 | 0.15 |
| target_score_w015_syn_loss_w075 | `generated_data/problem6_target_score_w015` | 0.15 | 0.75 | 0.15 |

## 总体结果

结果汇总另存为：

- `paper_tables/problem6_target_push_comparison.csv`

| Method | Overall Acc | Macro-F1 | Balanced Acc | Worst Class Acc | Target Acc | Base Acc | Target-to-Base | Concept Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| D-CGFS main w015 | 65.74 | 65.72 | 65.80 | 6.67 | 44.29 | 77.78 | 17.86 | 89.46 |
| target_score_w015 | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 | 89.46 |
| syn_loss_w075 | 65.65 | 65.64 | 65.68 | 6.67 | 40.71 | 76.07 | 19.29 | 89.41 |
| target_score_w015_syn_loss_w075 | 66.02 | 66.06 | 66.09 | 10.00 | 43.57 | 76.92 | 20.00 | 89.48 |
| Reweighting | 67.31 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 | 89.60 |
| Class-balanced loss | 67.31 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 | 89.60 |

## Pair 级变化

### target_score_w015

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 26.67 | 23.33 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 40.00 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 80.00 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 66.67 | 20.00 |

### syn_loss_w075

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 16.67 | 23.33 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 36.67 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 80.00 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 60.00 | 23.33 |

### target_score_w015_syn_loss_w075

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 23.33 | 23.33 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 26.67 | 40.00 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 80.00 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 63.33 | 23.33 |

## 判断

`target_score_w015` 是本轮三个候选里最稳的：

- Overall Acc: `65.74 -> 66.00`
- Macro-F1: `65.72 -> 65.96`
- Balanced Acc: `65.80 -> 66.04`
- Worst Class Acc: `6.67 -> 10.00`
- Base Acc: `77.78 -> 78.63`
- Target Acc: 保持 `44.29`

这说明提高 pair-score 中的 target confidence 权重有轻微正向作用，尤其改善了 worst-class 和 base preservation，但它没有解决 target accuracy 明显落后 strong baseline 的核心问题。

`syn_loss_w075` 不建议继续作为主线：

- Target Acc 从 `44.29` 降到 `40.71`
- Base Acc 从 `77.78` 降到 `76.07`
- Overall/Macro 也低于主方法

这说明单纯提高合成样本损失权重会放大合成样本噪声，并不会带来更强 target 修复。

组合实验 `target_score_w015_syn_loss_w075` 的 Overall/Macro 略高于 `target_score_w015`，但 Target Acc 和 Base Acc 都更差，因此不适合作为当前主方法。

## 结论

本轮实验基本否定了“只要提高合成损失权重就能追回 strong baseline”的假设。当前瓶颈更可能在合成样本本身的语义质量，而不是训练时给合成样本的权重不够。

下一步更合理的方向不是继续加大 synthetic loss，而是改合成样本选择策略：

1. 保留 `target_score_w015` 作为一个可选增强候选，但暂时不替换正式主方法。
2. 继续诊断 target probability 极低的问题，考虑加入更强的 target-confidence rerank 或 per-pair adaptive top-k。
3. 针对 pair 0、1、4 单独分析合成候选的 score 分布，确认是否 top-500 中仍混入大量低 target-confidence 样本。
