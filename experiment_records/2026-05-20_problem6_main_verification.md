# 2026-05-20 Problem6 Main Verification

## 运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group main
```

## 当前主方法配置

- 方法名称：`D-CGFS pair_topk_base_w015`
- 合成过滤：`pair_topk_filter`
- 每个 target-base pair 保留数量：`top-500`
- 合成数据目录：`generated_data/dcgfs_pair_topk`
- 训练 checkpoint：`checkpoints/problem6_dcgfs_pair_topk_base_w015.pt`
- base preservation：启用
- base preservation 权重：`0.15`
- 训练轮数：`20 epochs`
- 计算设备：`cuda`

## 合成数据诊断

本次先生成 10000 个候选合成样本，然后在每个 target-base pair 内按照质量分数选择 top-k：

| pair_id | 候选数量 | 保留数量 |
| --- | ---: | ---: |
| 0 | 1617 | 500 |
| 1 | 1966 | 500 |
| 2 | 954 | 500 |
| 3 | 751 | 500 |
| 4 | 1266 | 500 |

最终保留合成样本：`2500 / 10000`。

过滤来源统计：

- `strict`: 0
- `quality_fallback`: 0
- `pair_topk`: 2500

合成样本质量诊断：

- `target_prob`: 最大值仅为 `0.0181`，说明原始 SSCBM 对合成样本仍然很少直接预测为目标类。
- `sim_target - sim_base`: median 为 `0.0051`，说明按概念相似度看，保留下来的样本相对更偏向目标类，但优势并不极强。
- `check_synthetic_data.py` 抽查前 5 个样本时，保存标签均为目标类 64，但原始模型均预测为 197。这个结果不直接否定合成数据，因为 D-CGFS 的合成样本本来就是用于纠正原始模型弱势类识别的反事实样本；不过它提示当前合成样本在原始分类器语义空间中的 target confidence 仍偏低，后续论文中需要谨慎解释质量过滤策略。

## 训练损失曲线

| Epoch | Avg Loss |
| ---: | ---: |
| 1 | 6.6118 |
| 2 | 2.6165 |
| 3 | 2.2385 |
| 4 | 2.0396 |
| 5 | 1.9643 |
| 6 | 1.9248 |
| 7 | 1.9031 |
| 8 | 1.7729 |
| 9 | 1.7699 |
| 10 | 1.7144 |
| 11 | 1.6824 |
| 12 | 1.7581 |
| 13 | 1.6617 |
| 14 | 1.6454 |
| 15 | 1.6341 |
| 16 | 1.6353 |
| 17 | 1.6106 |
| 18 | 1.6012 |
| 19 | 1.6130 |
| 20 | 1.5999 |

判断：loss 在前 10 个 epoch 快速下降，17 到 20 epoch 已经基本进入平台区间，没有出现第 20 epoch 仍大幅下降的现象。因此当前阶段继续保留 `20 epochs` 是合理的，可以保证与已有结果和已有消融实验一致。是否增加 epoch 应放到后续统一实验协议中单独验证。

## 最终分类结果

结果目录：`final_evaluation_results/problem6_dcgfs_pair_topk_base_w015/`

| 指标 | SSCBM baseline | D-CGFS pair_topk_base_w015 | 变化 |
| --- | ---: | ---: | ---: |
| Overall Acc | 57.80% | 65.74% | +7.94% |
| Overall Concept Acc | 89.72% | 89.46% | -0.25% |
| Macro-F1 | 57.23% | 65.72% | +8.49% |
| Balanced Acc | 57.99% | 65.80% | +7.80% |
| Worst Class Acc | 0.00% | 6.67% | +6.67% |
| Selected Target Acc | 19.29% | 44.29% | +25.00% |
| Selected Base Acc | 82.91% | 77.78% | -5.13% |
| Target-to-Base Rate | 48.57% | 17.86% | -30.71% |

该结果与当前已经固定的主结果一致，说明 `run_problem6_experiments.py --only-group main` 已经正确对齐到 D-CGFS 当前正式主方法。

## 可解释性结果

结果目录：`explainability_results/problem6_dcgfs_pair_topk_base_w015/`

### Concept Accuracy

| 指标 | SSCBM baseline | D-CGFS pair_topk_base_w015 | 变化 |
| --- | ---: | ---: | ---: |
| Overall Concept Acc | 0.8972 | 0.8946 | -0.0025 |
| Target Concept Acc | 0.9054 | 0.9118 | +0.0064 |
| Target Discriminative Concept Acc | 0.2164 | 0.3219 | +0.1055 |

### Heatmap Quality

| 指标 | SSCBM baseline | D-CGFS pair_topk_base_w015 | 变化 |
| --- | ---: | ---: | ---: |
| Heatmap Entropy | 0.9308 | 0.9426 | +0.0118 |
| Mask Compactness | 0.1143 | 0.1150 | +0.0007 |
| BBox Energy Ratio | 0.6228 | 0.5966 | -0.0262 |

### Concept Intervention

| 指标 | SSCBM baseline | D-CGFS pair_topk_base_w015 | 变化 |
| --- | ---: | ---: | ---: |
| Original Acc | 0.1929 | 0.4429 | +0.2500 |
| Intervention Acc | 0.7786 | 0.7929 | +0.0143 |
| Intervention Gain | 0.5857 | 0.3500 | -0.2357 |

解释：D-CGFS 在目标类原始准确率上已经明显提升，因此干预后的额外提升空间变小，导致 `intervention_gain` 下降。更重要的是，干预后准确率本身仍从 `0.7786` 提升到 `0.7929`，同时目标判别概念准确率明显提升，说明方法确实增强了目标类相关判别概念的利用。

## 当前结论

本次运行确认了三点：

1. 当前正式主方法 `D-CGFS pair_topk_base_w015` 能稳定复现已有主结果。
2. `20 epochs` 暂时不需要改，因为训练后期 loss 已基本平台化。
3. 方法的主要优势集中在弱势目标类修复、target-to-base 混淆下降、macro/balanced 指标提升，以及目标判别概念准确率提升。

当前主要不足：

1. 合成样本的原始模型 target probability 偏低，说明合成样本在原始分类器空间中仍不够像目标类。
2. base 类准确率仍下降 `5.13%`，虽然比未加 base preservation 的版本更稳，但仍需要在论文中报告 trade-off。
3. 整体概念准确率轻微下降 `0.25%`，需要强调 D-CGFS 的改进目标是弱势类与判别概念，而不是无条件提高所有概念预测。

## 下一步建议

下一步应运行强基线组，检查 D-CGFS 是否不仅优于原始 SSCBM，也优于常规增强、重采样、重加权等 baseline：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group strong_baseline
```

如果强基线结果明显弱于当前主方法，则可以进入 `problem6` 的完整表格整理；如果某个强基线接近或超过 D-CGFS，则需要进一步分析它提升的是整体准确率、目标类准确率，还是牺牲了基座类与概念可解释性。
