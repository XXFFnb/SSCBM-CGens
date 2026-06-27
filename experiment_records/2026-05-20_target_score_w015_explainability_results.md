# 2026-05-20 Target-Score w015 Explainability Results

## 运行命令

```bash
.venv/bin/python step5_explainability_evaluation.py --candidate-checkpoint checkpoints/problem6_dcgfs_target_score_w015.pt --candidate-name dcgfs_target_score_w015 --save-dir explainability_results/problem6_dcgfs_target_score_w015
```

## 输出目录

- `explainability_results/problem6_dcgfs_target_score_w015/`

## 与当前主方法对比

汇总表另存为：

- `paper_tables/problem6_target_score_explainability_comparison.csv`

| Method | Overall C Acc | Target C Acc | Target Disc C Acc | Heatmap Entropy | Mask Compactness | BBox Energy | Original Acc | Intervention Acc | Intervention Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 89.72 | 90.54 | 21.64 | 93.08 | 11.43 | 62.28 | 19.29 | 77.86 | 58.57 |
| D-CGFS main | 89.46 | 91.18 | 32.19 | 94.26 | 11.50 | 59.66 | 44.29 | 79.29 | 35.00 |
| target_score_w015 | 89.46 | 91.11 | 31.78 | 94.10 | 11.25 | 59.79 | 44.29 | 82.14 | 37.86 |

## 结合分类指标

| Method | Overall Acc | Macro-F1 | Balanced Acc | Worst Class | Target Acc | Base Acc | Target-to-Base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-CGFS main | 65.74 | 65.72 | 65.80 | 6.67 | 44.29 | 77.78 | 17.86 |
| target_score_w015 | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 |

## 判断

`target_score_w015` 相比当前主方法有以下优点：

- Overall Acc 更高：`65.74 -> 66.00`
- Macro-F1 更高：`65.72 -> 65.96`
- Balanced Acc 更高：`65.80 -> 66.04`
- Worst Class Acc 更高：`6.67 -> 10.00`
- Base Acc 更高：`77.78 -> 78.63`
- Intervention Acc 更高：`79.29 -> 82.14`
- Intervention Gain 更高：`35.00 -> 37.86`

它的代价是：

- Target-to-Base Rate 略差：`17.86 -> 19.29`
- Target Disc Concept Acc 略低：`32.19 -> 31.78`
- Heatmap Entropy 与 Mask Compactness 略低，差距很小。

## 结论

`target_score_w015` 可以作为新的正式主方法候选。它没有提升 target accuracy，但在整体分类、base preservation、worst-class 和概念干预效果上都优于当前主方法，可解释性没有明显崩坏。

如果论文主叙事强调“弱势类修复 + 基座类保护 + 可解释干预”，`target_score_w015` 比当前主方法更适合作为主配置。

如果论文主叙事更强调“降低 target-to-base 混淆率”，则当前主方法仍略占优势。
