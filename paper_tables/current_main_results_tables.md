# D-CGFS 当前论文表格草稿

本文档把当前已完成实验整理成论文表格草稿。当前主实验配置已更新为：

`D-CGFS target-score pair-topk + base preservation, pair_score_target_weight=0.15, base_preservation_weight=0.15`

主结果目录：

- 分类评估：`final_evaluation_results/problem6_dcgfs_target_score_w015/`
- 可解释性评估：`explainability_results/problem6_dcgfs_target_score_w015/`

## Table 1. Main Results

| Method | Overall Acc | Macro-F1 | Balanced Acc | Worst-Class Acc | Target Acc | Base Acc | Target-to-Base Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 |
| D-CGFS original main | 65.36 | 65.16 | 65.45 | 3.33 | 37.86 | 75.21 | 20.71 |
| D-CGFS no conf filter | 65.62 | 65.58 | 65.67 | 6.67 | 43.57 | 66.67 | 16.43 |
| D-CGFS pair-topk | 65.98 | 65.97 | 65.99 | 6.67 | 44.29 | 66.67 | 12.86 |
| D-CGFS previous main: pair-score w=0.05 + BP, w=0.15 | 65.74 | 65.72 | 65.80 | 6.67 | 44.29 | 77.78 | 17.86 |
| D-CGFS target-score w=0.15 + BP, w=0.15 | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 |

建议论文表述：当前主配置在不降低 target acc 的前提下，提高 overall、macro、balanced、worst-class、base acc 和 intervention acc；代价是 target-to-base rate 比旧主配置略高 1.43 个百分点。

## Table 2. Base Preservation Weight Ablation

| Weight | Overall Acc | Macro-F1 | Balanced Acc | Worst-Class Acc | Target Acc | Base Acc | Target-to-Base Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.050 | 66.02 | 65.90 | 66.03 | 6.67 | 42.86 | 71.79 | 17.14 |
| 0.100 | 65.98 | 65.97 | 66.04 | 10.00 | 42.14 | 75.21 | 18.57 |
| 0.125 | 66.02 | 66.04 | 66.09 | 10.00 | 42.86 | 76.07 | 17.86 |
| 0.135 | 66.12 | 66.10 | 66.17 | 6.67 | 41.43 | 77.78 | 22.14 |
| 0.145 | 66.03 | 66.04 | 66.12 | 10.00 | 40.71 | 76.92 | 20.71 |
| 0.150 | 65.74 | 65.72 | 65.80 | 6.67 | 44.29 | 77.78 | 17.86 |
| 0.155 | 66.00 | 65.99 | 66.07 | 10.00 | 42.86 | 77.78 | 18.57 |
| 0.165 | 65.77 | 65.77 | 65.87 | 10.00 | 40.71 | 77.78 | 22.86 |
| 0.175 | 65.64 | 65.64 | 65.68 | 6.67 | 43.57 | 74.36 | 18.57 |

建议论文表述：`w=0.15` 不是整体平均指标最高的权重，但在 target acc、base acc 和 target-to-base rate 三个核心目标之间最平衡，因此作为主实验配置。

## Table 3. Explainability Results

| Method | Target Concept Acc | Target Discriminative Concept Acc | Intervention Acc |
| --- | ---: | ---: | ---: |
| SSCBM | 90.54 | 21.64 | 77.86 |
| D-CGFS pair-topk | 90.82 | 31.10 | 80.71 |
| D-CGFS previous main: pair-score w=0.05 + BP, w=0.15 | 91.18 | 32.19 | 79.29 |
| D-CGFS target-score w=0.15 + BP, w=0.15 | 91.11 | 31.78 | 82.14 |

建议论文表述：当前主配置保持目标类概念预测和目标判别概念预测的提升，同时取得更高的 intervention acc；不要写成所有可解释性指标全面最优，因为 target discriminative concept acc 略低于旧主配置。

## Table 4. LaTeX Draft: Main Results

```latex
\begin{table}[t]
\centering
\caption{Main results on CUB. Target Acc and Target-to-Base Rate evaluate the automatically selected weak target classes and their confused base classes.}
\label{tab:main_results}
\begin{tabular}{lrrrrrrr}
\toprule
Method & Overall & Macro-F1 & Balanced & Worst & Target & Base & T$\rightarrow$B \\
\midrule
SSCBM & 57.80 & 57.23 & 57.99 & 0.00 & 19.29 & 82.91 & 48.57 \\
D-CGFS original & 65.36 & 65.16 & 65.45 & 3.33 & 37.86 & 75.21 & 20.71 \\
D-CGFS no conf filter & 65.62 & 65.58 & 65.67 & 6.67 & 43.57 & 66.67 & 16.43 \\
D-CGFS pair-topk & 65.98 & 65.97 & 65.99 & 6.67 & 44.29 & 66.67 & 12.86 \\
D-CGFS previous main & 65.74 & 65.72 & 65.80 & 6.67 & 44.29 & 77.78 & 17.86 \\
D-CGFS target-score + BP & 66.00 & 65.96 & 66.04 & 10.00 & 44.29 & 78.63 & 19.29 \\
\bottomrule
\end{tabular}
\end{table}
```

## Table 5. LaTeX Draft: Weight Ablation

```latex
\begin{table}[t]
\centering
\caption{Ablation study of the base preservation weight. The main configuration uses $w=0.15$.}
\label{tab:base_preservation_weight}
\begin{tabular}{crrrrrrr}
\toprule
$w$ & Overall & Macro-F1 & Balanced & Worst & Target & Base & T$\rightarrow$B \\
\midrule
0.050 & 66.02 & 65.90 & 66.03 & 6.67 & 42.86 & 71.79 & 17.14 \\
0.100 & 65.98 & 65.97 & 66.04 & 10.00 & 42.14 & 75.21 & 18.57 \\
0.125 & 66.02 & 66.04 & 66.09 & 10.00 & 42.86 & 76.07 & 17.86 \\
0.135 & 66.12 & 66.10 & 66.17 & 6.67 & 41.43 & 77.78 & 22.14 \\
0.145 & 66.03 & 66.04 & 66.12 & 10.00 & 40.71 & 76.92 & 20.71 \\
0.150 & 65.74 & 65.72 & 65.80 & 6.67 & 44.29 & 77.78 & 17.86 \\
0.155 & 66.00 & 65.99 & 66.07 & 10.00 & 42.86 & 77.78 & 18.57 \\
0.165 & 65.77 & 65.77 & 65.87 & 10.00 & 40.71 & 77.78 & 22.86 \\
0.175 & 65.64 & 65.64 & 65.68 & 6.67 & 43.57 & 74.36 & 18.57 \\
\bottomrule
\end{tabular}
\end{table}
```
