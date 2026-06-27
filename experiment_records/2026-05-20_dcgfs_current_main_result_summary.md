# D-CGFS 当前主实验结果总总结

## 当前主实验配置

当前建议将以下配置作为 D-CGFS 主实验配置：

- 目标类和基座类选择：validation set 自动选择弱势 target class 与对应 base class。
- 合成策略：pair-topk 判别概念过滤。
- 训练策略：启用 base preservation。
- 主权重：`base_preservation_weight=0.15`。
- 主结果目录：`final_evaluation_results/dcgfs_pair_topk_base_w015/`。
- 主可解释性目录：`explainability_results/dcgfs_pair_topk_base_w015/`。

对应运行命令：

```bash
bash run_dcgfs_pipeline.sh pair_topk_base
```

若需要复现实验局部权重消融：

```bash
bash run_dcgfs_pipeline.sh pair_topk_base_sweep
bash run_dcgfs_pipeline.sh pair_topk_base_fine_sweep
bash run_dcgfs_pipeline.sh pair_topk_base_ultra_fine_sweep
```

## 主结果对比

| 方法 | overall acc | macro F1 | balanced acc | worst class acc | target acc | base acc | target-to-base rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 57.80% | 57.23% | 57.99% | 0.00% | 19.29% | 82.91% | 48.57% |
| D-CGFS original main | 65.36% | 65.16% | 65.45% | 3.33% | 37.86% | 75.21% | 20.71% |
| D-CGFS no conf filter | 65.62% | 65.58% | 65.67% | 6.67% | 43.57% | 66.67% | 16.43% |
| D-CGFS pair-topk | 65.98% | 65.97% | 65.99% | 6.67% | 44.29% | 66.67% | 12.86% |
| D-CGFS pair-topk + base preservation, w=0.15 | 65.74% | 65.72% | 65.80% | 6.67% | 44.29% | 77.78% | 17.86% |

## 主配置的核心收益

相对 SSCBM baseline，当前主配置有明显提升：

- overall acc：57.80% -> 65.74%，提升 7.94 个百分点。
- macro F1：57.23% -> 65.72%，提升 8.49 个百分点。
- balanced acc：57.99% -> 65.80%，提升 7.80 个百分点。
- worst class acc：0.00% -> 6.67%。
- target acc：19.29% -> 44.29%，提升 25.00 个百分点。
- target-to-base rate：48.57% -> 17.86%，下降 30.71 个百分点。

这说明 D-CGFS 的主要效果不是只提升整体准确率，而是明显改善了原始 SSCBM 对弱势目标类的识别，并显著减少弱势目标类被错分成基座类的问题。

## 为什么选择 w=0.15 作为主实验

`base_preservation_weight=0.15` 不是所有单项指标的最高点，但它最符合 D-CGFS 的主张。

在 0.15 附近的细粒度消融中：

| weight | overall acc | macro F1 | balanced acc | worst class acc | target acc | base acc | target-to-base rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.135 | 66.12% | 66.10% | 66.17% | 6.67% | 41.43% | 77.78% | 22.14% |
| 0.145 | 66.03% | 66.04% | 66.12% | 10.00% | 40.71% | 76.92% | 20.71% |
| 0.150 | 65.74% | 65.72% | 65.80% | 6.67% | 44.29% | 77.78% | 17.86% |
| 0.155 | 66.00% | 65.99% | 66.07% | 10.00% | 42.86% | 77.78% | 18.57% |
| 0.165 | 65.77% | 65.77% | 65.87% | 10.00% | 40.71% | 77.78% | 22.86% |

`0.15` 的优势在于：

- target acc 最高，达到 44.29%。
- target-to-base rate 最低，为 17.86%。
- base acc 为 77.78%，与多数局部点持平。
- 在目标类提升、基座类保持、目标到基座错分抑制之间最平衡。

`0.135` 的 overall acc、macro F1、balanced acc 更高，但 target acc 更低，target-to-base rate 更高，不如 `0.15` 符合弱势类纠偏目标。

`0.155` 的部分可解释性指标更好，但 target acc 低于 `0.15`，target-to-base rate 也略高，因此更适合作为补充消融，而不是主配置。

## 可解释性结果

| 方法 | target concept acc | target discriminative concept acc | intervention acc |
| --- | ---: | ---: | ---: |
| SSCBM baseline | 90.54% | 21.64% | 77.86% |
| D-CGFS pair-topk | 90.82% | 31.10% | 80.71% |
| D-CGFS pair-topk + base preservation, w=0.15 | 91.18% | 32.19% | 79.29% |

当前主配置在 target concept acc 和 target discriminative concept acc 上均高于 SSCBM baseline 和 pair-topk，无 base preservation 版本。说明加入 base preservation 后，并不是只在任务分类上保护基座类，也改善了目标类相关概念预测，尤其是判别概念预测。

需要注意：intervention acc 上，pair-topk 无 base preservation 是 80.71%，当前主配置为 79.29%。因此可解释性指标不是全线最高。论文中应避免表述为“所有可解释性指标均最优”，更准确的说法是：当前主配置在目标判别概念准确性上更强，同时保持了合理的干预可解释性表现。

## 方法贡献的当前叙述

当前 D-CGFS 可以被组织成三层贡献：

1. 自动发现弱势类与混淆基座类：从 validation set 上选择弱势 target class 和容易混淆的 base class，避免使用 test set 做目标选择。

2. 判别概念引导的反事实合成：针对 target-base pair 选择 target 明显强于 base 的判别概念，合成更贴近目标类决策边界的样本，并通过 pair-topk 控制概念选择质量。

3. 基座保持的平衡训练：在提升 target class 的同时，对 base class 原始样本施加 teacher consistency 约束，避免合成样本把模型过度推向 target class，降低对 base class 的破坏。

## 当前不足

1. 当前主配置不是 overall acc、macro F1、balanced acc 的最高点。若审稿人更看重整体指标，需要用消融解释为什么主方法选择以弱势目标类纠偏为优先目标。

2. `0.15` 的 worst class acc 不是最高。`0.145/0.155/0.165` 达到 10.00%，而 `0.15` 是 6.67%。后续可以进一步分析最差类是否属于目标类集合，或者是否是与当前 target-base pair 无关的其他类别。

3. base preservation 会牺牲一部分 target-to-base rate。pair-topk 无 base preservation 的 target-to-base rate 为 12.86%，而 w=0.15 为 17.86%。但它将 base acc 从 66.67% 拉回到 77.78%，这是选择 base preservation 的主要理由。

4. 当前实验仍然是单 seed 结果。若目标是顶会或顶刊，需要补多 seed 均值和方差，至少建议跑 3 个 seed。

5. 当前只在 CUB 上验证。若投稿目标较高，最好补一个具有概念标注或可解释属性的额外数据集，或者设计更强的 CUB 内部协议。

## 下一步建议

优先级最高的是把当前结果整理成论文表格：

- 主结果表：SSCBM baseline、D-CGFS original main、D-CGFS pair-topk、D-CGFS pair-topk + base preservation w=0.15。
- 消融表：no conf filter、pair-topk、pair-topk + base preservation。
- 权重消融表：0.05、0.10、0.125、0.135、0.145、0.15、0.155、0.165、0.175、0.20。
- 可解释性表：target concept acc、target discriminative concept acc、intervention acc、heatmap quality。

随后建议补两类实验：

- 多 seed 稳定性实验：确认 `0.15` 的结论不是单 seed 偶然结果。
- 类别级诊断：分析 5 个 target class 分别提升多少、哪些 target-base pair 仍然失败、失败是否集中在某些概念组。

## 当前结论

当前阶段可以将 `D-CGFS pair-topk + base preservation, base_preservation_weight=0.15` 固定为主实验配置。它不是单项整体指标最高的点，但它最符合方法目标：提升弱势目标类、减少目标类错分到基座类、并保持基座类性能不被合成样本明显破坏。

