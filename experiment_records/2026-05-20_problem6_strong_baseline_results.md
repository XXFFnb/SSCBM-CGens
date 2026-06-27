# 2026-05-20 Problem6 Strong Baseline Results

## 运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group strong_baseline
```

## 输出目录

本次 strong baseline 结果已保存到以下目录：

- `final_evaluation_results/problem6_sscbm_finetune/`
- `final_evaluation_results/problem6_oversampling/`
- `final_evaluation_results/problem6_reweighting/`
- `final_evaluation_results/problem6_class_balanced_loss/`
- `final_evaluation_results/problem6_feature_mixup/`

D-CGFS 主方法对照目录：

- `final_evaluation_results/problem6_dcgfs_pair_topk_base_w015/`

汇总表另存为：

- `paper_tables/problem6_strong_baseline_comparison.csv`

## 核心指标对比

| Method | Overall Acc | Macro-F1 | Balanced Acc | Worst Class Acc | Target Acc | Base Acc | Target-to-Base Rate | Concept Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| SSCBM finetune | 66.76 | 66.62 | 66.87 | 3.33 | 56.43 | 71.79 | 16.43 | 89.56 |
| Oversampling | 66.48 | 66.06 | 66.53 | 13.33 | 50.71 | 64.10 | 10.71 | 89.51 |
| Reweighting | 67.31 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 | 89.60 |
| Class-balanced loss | 67.31 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 | 89.60 |
| Feature mixup | 66.69 | 66.65 | 66.87 | 3.33 | 57.14 | 68.38 | 15.00 | 88.92 |
| D-CGFS pair_topk_base_w015 | 65.74 | 65.72 | 65.80 | 6.67 | 44.29 | 77.78 | 17.86 | 89.46 |

## 结果判断

本次结果说明，D-CGFS 当前主方法已经显著优于原始 SSCBM，但尚未在所有分类指标上超过 strong baseline。

相对原始 SSCBM，D-CGFS 的提升非常明确：

- Overall Acc: `57.80 -> 65.74`
- Macro-F1: `57.23 -> 65.72`
- Balanced Acc: `57.99 -> 65.80`
- Target Acc: `19.29 -> 44.29`
- Target-to-Base Rate: `48.57 -> 17.86`

相对 strong baseline，当前 D-CGFS 的不足也很明确：

- `Reweighting` 和 `Class-balanced loss` 的 Overall Acc、Macro-F1、Balanced Acc、Target Acc 都高于 D-CGFS。
- `SSCBM finetune`、`Feature mixup` 也在整体分类指标上略高于 D-CGFS。
- D-CGFS 当前最明显的相对优势是 base 类保护更好：`77.78`，高于所有 strong baseline。

因此，当前不能把论文主结论写成“D-CGFS 全面超过所有强基线”。更合理的结论是：

1. D-CGFS 显著修复 SSCBM 的弱势目标类错误。
2. D-CGFS 在目标类修复和基座类保护之间取得了更平衡的 trade-off。
3. D-CGFS 具备概念级、pair 级、可解释的修复机制，而 strong baseline 主要是黑盒式再训练策略。
4. 若要冲击顶会或顶刊，需要继续增强 D-CGFS，使它至少在 Macro-F1、Balanced Acc、Target Acc 中接近或超过 strong baseline。

## 后续重点

下一步不建议直接进入完整 protocol 大规模运行。当前应先解决一个核心问题：

> 为什么普通 reweighting / class-balanced loss 的目标类准确率高于 D-CGFS？

优先排查方向：

1. D-CGFS 合成样本 target probability 过低，可能导致合成特征对目标类边界推动不够强。
2. `base preservation weight=0.15` 保护了 base 类，但可能压制了目标类进一步提升。
3. 当前 pair-topk 只保留每个 pair 的 top-500，可能样本数量不足，或质量分数没有充分反映最终分类收益。
4. 训练损失只看整体 loss，不知道目标类、基座类、合成样本三部分各自的优化状态，后续需要更细粒度日志。

建议下一步先做诊断，而不是盲目继续跑全量实验：

- 对 strong baseline 和 D-CGFS 做同一组目标类/基座类的 classwise 对比。
- 查看 D-CGFS 失败的目标类，确认是哪些 target pair 没有修复好。
- 评估是否需要提高合成样本质量门槛、增加 pair-topk 数量，或引入目标类置信度约束。
