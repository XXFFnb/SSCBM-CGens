# 2026-05-20 Problem6 Diversity-Push Results

## 运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group diversity_push
```

## 实验目的

前一轮合成质量诊断发现，部分 pair 的 top-500 合成样本高度重复：

- pair 1: 500 个样本只来自 15 个唯一 target/base 源图组合，重复率 97%。
- pair 4: 500 个样本只来自 15 个唯一 target/base 源图组合，重复率 97%。

本轮实验加入 `--pair-topk-max-per-source-pair 20`，限制同一 target/base 源图组合最多保留 20 次，验证提高有效多样性是否能改善 target accuracy。

## 总体结果

汇总表另存为：

- `paper_tables/problem6_diversity_push_comparison.csv`

| Method | Overall Acc | Macro-F1 | Balanced Acc | Worst Class Acc | Target Acc | Base Acc | Target-to-Base | Concept Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| D-CGFS main | 65.74 | 65.72 | 65.80 | 6.67 | 44.29 | 77.78 | 17.86 | 89.46 |
| target_score_w015 | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 | 89.46 |
| pair_topk_diverse20 | 65.79 | 65.79 | 65.86 | 10.00 | 42.14 | 76.92 | 20.71 | 89.48 |
| target_score_w015_diverse20 | 66.09 | 66.09 | 66.17 | 6.67 | 42.86 | 77.78 | 17.86 | 89.46 |
| Reweighting | 67.31 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 | 89.60 |

## Pair 级结果

### pair_topk_diverse20

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 5.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 23.33 | 26.67 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 36.67 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 83.33 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 56.67 | 30.00 |

### target_score_w015_diverse20

| Pair | Target | Base | Target Acc | Target-to-Base |
| ---: | --- | --- | ---: | ---: |
| 0 | 65 Slaty_backed_Gull | 62 Herring_Gull | 15.00 | 5.00 |
| 1 | 59 California_Gull | 62 Herring_Gull | 23.33 | 23.33 |
| 2 | 144 Common_Tern | 141 Artic_Tern | 23.33 | 36.67 |
| 3 | 74 Florida_Jay | 15 Lazuli_Bunting | 76.67 | 0.00 |
| 4 | 176 Prairie_Warbler | 163 Cape_May_Warbler | 66.67 | 20.00 |

## 实际多样性变化

| Method | Pair | Kept | Unique Source Pairs | Duplicate Rate | Target Prob Median | Target Prob Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 0 | 500 | 75 | 85.00% | 7.659e-12 | 9.978e-05 |
| main | 1 | 500 | 15 | 97.00% | 2.853e-08 | 1.814e-02 |
| main | 4 | 500 | 15 | 97.00% | 5.550e-11 | 1.875e-06 |
| diverse20 | 0 | 500 | 75 | 85.00% | 7.659e-12 | 9.978e-05 |
| diverse20 | 1 | 500 | 25 | 95.00% | 8.266e-08 | 1.814e-02 |
| diverse20 | 4 | 500 | 25 | 95.00% | 2.393e-11 | 1.875e-06 |
| target_score_w015_diverse20 | 0 | 500 | 75 | 85.00% | 2.310e-07 | 9.978e-05 |
| target_score_w015_diverse20 | 1 | 500 | 25 | 95.00% | 9.956e-07 | 1.814e-02 |
| target_score_w015_diverse20 | 4 | 500 | 25 | 95.00% | 8.248e-11 | 1.875e-06 |

## 判断

`diverse20` 没有解决核心问题：

- `pair_topk_diverse20` 的 Target Acc 从主方法 `44.29` 降到 `42.14`。
- `target_score_w015_diverse20` 的 Target Acc 为 `42.86`，也低于主方法。
- 虽然 `target_score_w015_diverse20` 的 Overall/Macro/Balanced 略高，但 target 修复变弱。

原因是 `max_per_source_pair=20` 约束太弱。pair 1 和 pair 4 的唯一源图组合只从 15 增到 25，重复率仍然高达 95%。

## 后续动作

已经新增更严格的实验组：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group diversity_strict
```

该组使用 `--pair-topk-max-per-source-pair 10`，允许部分 pair 少于 500 个样本，用来验证“少而多样”是否优于“多但高度重复”。
