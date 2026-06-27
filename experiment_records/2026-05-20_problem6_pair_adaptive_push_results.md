# 2026-05-20 Problem6 Pair-Adaptive-Push Results

## 运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group pair_adaptive_push
```

## 实验目的

前几轮训练侧实验已经说明，继续调 class-balanced、target-class weighted loss 或 synthetic loss weight 都不能稳定超过当前主方法。因此本轮回到合成阶段，验证一个更直接的假设：

> 某些 pair 的合成样本质量较低，如果仍然硬保留 top-500，可能会把低质量样本送入训练；减少这些 pair 的样本数，可能提升最终性能。

本轮测试三种策略：

1. `target_conf`: 按 `target_prob >= 1e-6` 的候选数量自适应决定每个 pair 的 top-k。
2. `source_diversity`: 按唯一 target/base 源图组合数量决定每个 pair 的 top-k。
3. `target_conf + diverse20`: 在 `target_conf` 基础上限制同一 target/base 源图组合最多保留 20 个样本。

## 合成样本数量

| Method | Pair 0 | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D-CGFS target_score_w015 | 500 | 500 | 500 | 500 | 500 | 2500 |
| dcgfs_pair_adaptive_target_conf | 500 | 500 | 200 | 416 | 200 | 1816 |
| dcgfs_pair_adaptive_source_diversity | 500 | 500 | 500 | 500 | 500 | 2500 |
| dcgfs_pair_adaptive_conf_diverse20 | 500 | 500 | 200 | 416 | 200 | 1816 |

## 结果汇总

结果表格另存为：

- `paper_tables/problem6_pair_adaptive_push_comparison.csv`

| Method | Synthetic Samples | Overall Acc | Macro-F1 | Balanced Acc | Worst Class Acc | Target Acc | Base Acc | Target-to-Base | Concept Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 0 | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| D-CGFS target_score_w015 | 2500 | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 | 89.46 |
| dcgfs_pair_adaptive_target_conf | 1816 | 65.72 | 65.67 | 65.78 | 10.00 | 43.57 | 78.63 | 19.29 | 89.44 |
| dcgfs_pair_adaptive_source_diversity | 2500 | 65.72 | 65.68 | 65.77 | 10.00 | 42.14 | 78.63 | 18.57 | 89.44 |
| dcgfs_pair_adaptive_conf_diverse20 | 1816 | 65.69 | 65.58 | 65.74 | 6.67 | 42.14 | 78.63 | 19.29 | 89.46 |

## Pair 级变化

当前主方法 `D-CGFS target_score_w015`：

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 26.67 | 23.33 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 40.00 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 80.00 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 66.67 | 20.00 |

`dcgfs_pair_adaptive_target_conf`：

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 26.67 | 23.33 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 20.00 | 40.00 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 80.00 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 66.67 | 20.00 |

`dcgfs_pair_adaptive_conf_diverse20`：

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 5.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 26.67 | 23.33 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 33.33 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 83.33 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 53.33 | 30.00 |

## 判断

本轮结果不能替代当前主方法。

`target_conf` 策略把 pair 2 和 pair 4 从 500 个样本压到 200 个，总样本从 2500 降到 1816。它保持了 base accuracy，但 overall/macro/balanced 和 target accuracy 都低于主方法，说明简单少选低置信 pair 的样本并不能提升泛化。

`source_diversity` 在当前参数下实际仍保留每个 pair 500 个样本，因此它没有形成有效的自适应变化，结果也低于主方法。

`conf_diverse20` 改善了 pair 0、2 的 target-to-base 错分，但明显损伤 pair 4 的目标类准确率：`66.67 -> 53.33`。这说明简单去重或少选样本会伤害某些依赖重复高分候选的 pair。

## 结论

当前正式主方法仍保持：

```text
D-CGFS target_score_w015
```

这轮实验排除了“少选低质量 pair 样本就能提升结果”的假设。当前 D-CGFS 更像是需要足够的合成样本覆盖来推动目标类，而不是越少越干净越好。

下一步不建议继续沿着 `adaptive top-k 减样本` 方向调参。更合理的方向是：

1. 保持主方法的 2500 合成样本规模。
2. 在训练侧让合成样本的任务监督更直接作用于分类头，而不是减少样本。
3. 或者做 pair-specific 策略：pair 4 不应减少样本，pair 0/2 可重点降低 target-to-base 错分。
