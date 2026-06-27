# D-CGFS pair_topk_filter 实验记录与评价

## 记录信息

- 日期：2026-05-19
- 实验目的：验证 pair 内 top-k 过滤是否优于原始 `target_prob` 硬阈值和完全去掉置信度过滤。
- 运行命令：`bash run_dcgfs_pipeline.sh pair_topk`
- 合成数据目录：`generated_data/dcgfs_pair_topk/`
- 训练 checkpoint：`checkpoints/best_sscbm_dcgfs_pair_topk.pt`
- 分类评估目录：`final_evaluation_results/dcgfs_pair_topk/`
- 可解释性评估目录：`explainability_results/dcgfs_pair_topk/`
- 运行日志目录：`run_logs/`

## 方法设置

本次实验使用新加入的 `pair_topk_filter`：

```text
score = concept_delta + 0.05 * log(target_prob) - 0.05 * log(base_prob)
每个 target-base pair 保留 top-500 个样本
关闭 quality_fallback
```

这个策略的目的不是继续使用绝对目标类置信度阈值，而是在每个 pair 内挑相对更可靠的合成样本。

## 合成数据结果

来自 `generated_data/dcgfs_pair_topk/synthesized_metadata.csv`：

```text
总样本数: 2500
filter_mode: pair_topk 2500
```

目标类分布：

| target_class | 合成样本数 |
|---:|---:|
| 59 | 500 |
| 65 | 500 |
| 74 | 500 |
| 144 | 500 |
| 176 | 500 |

候选与保留情况：

| pair_id | 概念一致候选数 | 保留数 |
|---:|---:|---:|
| 0 | 1617 | 500 |
| 1 | 1966 | 500 |
| 2 | 954 | 500 |
| 3 | 751 | 500 |
| 4 | 1266 | 500 |

评价：这解决了 `no_conf_filter` 中各类合成样本数量不均衡的问题，也避免了原始 main 中严重依赖 `quality_fallback` 的问题。

## 分类性能对比

### pair_topk 相对 Baseline

| 指标 | SSCBM Baseline | D-CGFS pair_topk | 变化 |
|---|---:|---:|---:|
| overall_a_acc | 57.80% | 65.98% | +8.18% |
| overall_c_acc | 89.72% | 89.36% | -0.35% |
| macro_f1 | 57.23% | 65.97% | +8.74% |
| balanced_accuracy | 57.99% | 65.99% | +8.00% |
| worst_class_acc | 0.00% | 6.67% | +6.67% |
| selected_target_acc | 19.29% | 44.29% | +25.00% |
| selected_base_acc | 82.91% | 66.67% | -16.24% |
| target_to_base_rate | 48.57% | 12.86% | -35.71% |

### 三版方法对比

| 指标 | D-CGFS main | no_conf_filter | pair_topk |
|---|---:|---:|---:|
| overall_a_acc | 65.36% | 65.62% | 65.98% |
| overall_c_acc | 89.92% | 89.38% | 89.36% |
| macro_f1 | 65.16% | 65.58% | 65.97% |
| balanced_accuracy | 65.45% | 65.67% | 65.99% |
| worst_class_acc | 3.33% | 6.67% | 6.67% |
| selected_target_acc | 37.86% | 43.57% | 44.29% |
| selected_base_acc | 75.21% | 66.67% | 66.67% |
| target_to_base_rate | 20.71% | 16.43% | 12.86% |

评价：`pair_topk` 是当前三版中分类性能最强的版本。它在整体准确率、Macro-F1、Balanced Acc、目标类准确率和目标类错分为基座类比例上均优于 main 和 no_conf_filter。

## 目标类结果

| target_class | Baseline | D-CGFS main | no_conf_filter | pair_topk |
|---:|---:|---:|---:|---:|
| 59 | 6.67% | 13.33% | 23.33% | 20.00% |
| 65 | 0.00% | 10.00% | 25.00% | 15.00% |
| 74 | 66.67% | 80.00% | 83.33% | 80.00% |
| 144 | 13.33% | 26.67% | 16.67% | 20.00% |
| 176 | 3.33% | 50.00% | 63.33% | 76.67% |

评价：`pair_topk` 的总目标类准确率最高，但并不是每个目标类都优于 `no_conf_filter`。它最明显的优势来自 `Prairie_Warbler`，从 baseline 的 `3.33%` 提升到 `76.67%`。

## Target-Base 混淆

| target -> base | Baseline | D-CGFS main | no_conf_filter | pair_topk |
|---|---:|---:|---:|---:|
| Slaty_backed_Gull -> Herring_Gull | 55.00% | 0.00% | 0.00% | 0.00% |
| California_Gull -> Herring_Gull | 66.67% | 13.33% | 6.67% | 6.67% |
| Common_Tern -> Artic_Tern | 36.67% | 43.33% | 46.67% | 40.00% |
| Florida_Jay -> Lazuli_Bunting | 0.00% | 0.00% | 0.00% | 0.00% |
| Prairie_Warbler -> Cape_May_Warbler | 86.67% | 40.00% | 23.33% | 13.33% |

评价：`pair_topk` 对 target-to-base confusion 的改善最强，总体降到 `12.86%`。不过 `Common_Tern -> Artic_Tern` 仍然没有被解决，只是比 no_conf_filter 略好，仍高于 baseline。

## 可解释性结果

### 概念准确率

| 指标 | SSCBM | D-CGFS main | no_conf_filter | pair_topk |
|---|---:|---:|---:|---:|
| overall_c_acc | 89.72% | 89.92% | 89.38% | 89.36% |
| target_c_acc | 90.54% | 90.94% | 90.96% | 90.82% |
| target_disc_c_acc | 21.64% | 26.16% | 30.55% | 31.10% |

评价：`pair_topk` 在判别概念准确率上是当前最强版本，但整体概念准确率略低于 baseline 和 main。

### Heatmap 质量

| 指标 | SSCBM | D-CGFS main | no_conf_filter | pair_topk | 评价 |
|---|---:|---:|---:|---:|---|
| heatmap_entropy | 0.9308 | 0.9369 | 0.9439 | 0.9423 | 比 main 更分散，略好于 no_conf |
| mask_compactness | 0.1143 | 0.1074 | 0.1263 | 0.1225 | 比 main 更紧凑，略低于 no_conf |
| bbox_energy_ratio | 0.6228 | 0.6162 | 0.6002 | 0.6011 | 比 main 更差，略好于 no_conf |

评价：Heatmap 结果仍然混合。`pair_topk` 没有明显改善空间定位质量，主要收益来自分类边界和判别概念预测。

### 概念干预

| 指标 | SSCBM | D-CGFS main | no_conf_filter | pair_topk |
|---|---:|---:|---:|---:|
| original_acc | 19.29% | 37.86% | 43.57% | 44.29% |
| intervention_acc | 78.57% | 87.14% | 82.14% | 80.71% |
| intervention_gain | 59.29% | 49.29% | 38.57% | 36.43% |

评价：`pair_topk` 原始目标类准确率最高，但干预后准确率不如 main。这说明 pair_topk 更强地改变了分类边界，但没有让概念干预上限更高。

## 结论

`pair_topk_filter` 是目前最值得作为下一版主方法的候选。

理由：

- 不依赖 `quality_fallback`。
- 合成样本数量均衡，每个目标类 500 个。
- `overall_a_acc`、`macro_f1`、`balanced_accuracy` 均为三版最高。
- `selected_target_acc` 达到 `44.29%`，三版最高。
- `target_to_base_rate` 降到 `12.86%`，三版最低。
- `target_disc_c_acc` 达到 `31.10%`，三版最高。

但仍有两个重要问题：

- `selected_base_acc` 仍只有 `66.67%`，和 no_conf_filter 一样低，明显低于 main 的 `75.21%`。
- `Common_Tern -> Artic_Tern` 的混淆仍未解决。

因此，下一步不建议再回到 target_prob 硬阈值，而是应在 `pair_topk_filter` 基础上继续做基座类保护。

建议下一步实验：

```text
pair_topk_filter + base preservation
```

可选实现方向：

1. 降低每个 pair 的 top-k，例如从 500 改成 300，观察 base accuracy 是否恢复。
2. 在训练阶段增加 base preservation loss，让原始基座类样本不要被目标类过度侵蚀。
3. 对共享同一个基座类的多个 target pair 限制总合成量，例如 Herring_Gull 同时作为 class 65 和 59 的 base，可能受影响更大。
4. 单独分析 Common_Tern -> Artic_Tern，必要时调整该 pair 的判别概念或降低其合成权重。
