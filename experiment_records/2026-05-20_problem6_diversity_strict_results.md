# 2026-05-20 Problem6 Diversity-Strict Results

## 运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group diversity_strict
```

## 实验目的

`diversity_push` 中的 `max_per_source_pair=20` 仍然留下 95% 左右重复率，因此本轮进一步使用 `max_per_source_pair=10`，验证更严格的多样性约束是否能改善 target accuracy。

## 总体结果

汇总表另存为：

- `paper_tables/problem6_diversity_strict_comparison.csv`

| Method | Overall Acc | Macro-F1 | Balanced Acc | Worst Class Acc | Target Acc | Base Acc | Target-to-Base | Concept Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| D-CGFS main | 65.74 | 65.72 | 65.80 | 6.67 | 44.29 | 77.78 | 17.86 | 89.46 |
| target_score_w015 | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 | 89.46 |
| pair_topk_diverse20 | 65.79 | 65.79 | 65.86 | 10.00 | 42.14 | 76.92 | 20.71 | 89.48 |
| target_score_w015_diverse20 | 66.09 | 66.09 | 66.17 | 6.67 | 42.86 | 77.78 | 17.86 | 89.46 |
| pair_topk_diverse10 | 65.79 | 65.73 | 65.83 | 6.67 | 43.57 | 76.07 | 19.29 | 89.47 |
| target_score_w015_diverse10 | 65.81 | 65.77 | 65.84 | 10.00 | 42.86 | 76.07 | 20.00 | 89.48 |
| Reweighting | 67.31 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 | 89.60 |

## Pair 级结果

### pair_topk_diverse10

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 26.67 | 20.00 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 36.67 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 83.33 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 60.00 | 26.67 |

### target_score_w015_diverse10

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 10.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 26.67 | 20.00 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 20.00 | 46.67 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 76.67 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 66.67 | 20.00 |

## 实际多样性变化

| Method | Pair | Kept | Unique Source Pairs | Duplicate Rate | Target Prob Median | Target Prob Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 1 | 500 | 15 | 97.00% | 2.853e-08 | 1.814e-02 |
| main | 4 | 500 | 15 | 97.00% | 5.550e-11 | 1.875e-06 |
| diverse20 | 1 | 500 | 25 | 95.00% | 8.266e-08 | 1.814e-02 |
| diverse20 | 4 | 500 | 25 | 95.00% | 2.393e-11 | 1.875e-06 |
| diverse10 | 1 | 500 | 50 | 90.00% | 1.938e-08 | 1.814e-02 |
| diverse10 | 4 | 380 | 38 | 90.00% | 7.373e-12 | 1.875e-06 |
| target_score_w015_diverse10 | 1 | 500 | 50 | 90.00% | 1.938e-08 | 1.814e-02 |
| target_score_w015_diverse10 | 4 | 380 | 38 | 90.00% | 7.373e-12 | 1.875e-06 |

## 判断

`diverse10` 证明了“只靠减少重复”不是当前瓶颈的主要解法：

- 多样性确实提高了：pair 1 从 15 个唯一源图组合增至 50，pair 4 从 15 增至 38。
- 但 target accuracy 没有提升，反而低于当前主方法。
- base accuracy 也下降到 `76.07`，弱于主方法的 `77.78`。

因此，当前问题更可能是合成样本的 target 语义强度不够，而不是 top-k 重复本身。

## 结论

当前可保留的最好候选仍是：

1. 当前正式主方法 `D-CGFS main`
   - Target Acc 最高：`44.29`
   - Target-to-Base 最低：`17.86`

2. `target_score_w015`
   - Overall/Macro/Base/Worst 更好
   - Target Acc 与主方法相同
   - 可以作为后续替换主方法的候选，但需要先做可解释性评估

下一步建议转向：

- 对 `target_score_w015` 跑 step5 可解释性评估；
- 如果可解释性不差于主方法，可以考虑把正式主方法从 `main w015` 换成 `target_score_w015`；
- 如果可解释性变差，则保持当前主方法，并把 `target_score_w015` 写成增强排序消融。
