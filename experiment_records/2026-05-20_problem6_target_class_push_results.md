# 2026-05-20 Problem6 Target-Class-Push Results

## 运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group target_class_push
```

## 实验目的

`optimization_push` 已经说明，全局 class-balanced task loss 会轻微提高 overall/macro/balanced，但会降低自动目标类准确率。因此本轮不再对所有少样本类别加权，而是只对 `target_base_pairs.csv` 中自动选择出的 5 个目标弱势类加权：

```text
target labels, 0-based = [58, 64, 73, 143, 175]
target labels, 1-based = [59, 65, 74, 144, 176]
```

本轮测试权重为 `1.5 / 2.0 / 2.5`。

## 结果汇总

结果表格另存为：

- `paper_tables/problem6_target_class_push_comparison.csv`

| Method | Overall Acc | Macro-F1 | Balanced Acc | Worst Class Acc | Target Acc | Base Acc | Target-to-Base | Concept Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| D-CGFS target_score_w015 | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 | 89.46 |
| dcgfs_target_class_w15 | 66.12 | 66.13 | 66.20 | 10.00 | 41.43 | 76.92 | 21.43 | 89.47 |
| dcgfs_target_class_w20 | 66.09 | 66.04 | 66.11 | 6.67 | 43.57 | 76.92 | 17.86 | 89.39 |
| dcgfs_target_class_w25 | 65.86 | 65.84 | 65.92 | 3.33 | 43.57 | 77.78 | 19.29 | 89.44 |

## Pair 级观察

当前主方法 `D-CGFS target_score_w015` 的 pair 级目标准确率为：

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 26.67 | 23.33 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 40.00 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 80.00 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 66.67 | 20.00 |

`dcgfs_target_class_w20` 的 pair 级结果为：

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 10.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 20.00 | 26.67 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 36.67 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 80.00 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 73.33 | 13.33 |

## 判断

本轮结果不能替代当前主方法。

`dcgfs_target_class_w15` 的 overall/macro/balanced 略高，但自动目标类准确率从 `44.29` 降到 `41.43`，base accuracy 从 `78.63` 降到 `76.92`，target-to-base 也从 `19.29` 升到 `21.43`。这说明轻度定向加权并没有真正修复弱势目标类。

`dcgfs_target_class_w20` 是本轮最接近主方法的版本：overall/macro/balanced 小幅高于主方法，target-to-base 降到 `17.86`。但它的 target accuracy 仍低于主方法 `0.72` 个百分点，base accuracy 也低 `1.71` 个百分点，因此不能替换主方法。

`dcgfs_target_class_w25` 权重更大后，overall/macro/worst-class 继续下降，说明继续提高目标类权重会破坏整体分类边界。

## 结论

当前正式主方法仍保持：

```text
D-CGFS target_score_w015
```

这轮实验进一步说明，当前瓶颈不是简单的损失权重不足。无论是全局 class-balanced，还是只对自动目标类加权，都没有稳定超过主方法的 target accuracy。

下一步优化方向应从“训练损失加权”转向“合成样本本身和 pair 级策略”：

1. 对 pair 0、1、2 单独分析合成样本质量，因为这些 pair 仍然是目标类修复不足的主要来源。
2. 考虑 pair-adaptive top-k 或 pair-adaptive score 权重，而不是所有 pair 统一 `topk=500`、统一 `target_weight=0.15`。
3. 分析 target candidate 的真实 top-score 分布，判断是否某些 pair 应少选样本，避免低质量合成样本进入训练。
