# D-CGFS pair_topk_base 实验记录与评价

## 记录信息

- 日期：2026-05-19
- 实验目的：验证在 `pair_topk_filter` 基础上加入基座类保持损失后，是否能缓解基座类准确率下降问题。
- 运行命令：`bash run_dcgfs_pipeline.sh pair_topk_base`
- 复用合成数据目录：`generated_data/dcgfs_pair_topk/`
- 训练 checkpoint：`checkpoints/best_sscbm_dcgfs_pair_topk_base.pt`
- 分类评估目录：`final_evaluation_results/dcgfs_pair_topk_base/`
- 可解释性评估目录：`explainability_results/dcgfs_pair_topk_base/`
- 运行日志目录：`run_logs/`

## 方法设置

本次实验不重新生成合成数据，而是复用 `pair_topk_filter` 生成的 2500 个合成样本：

```text
每个 target-base pair 保留 top-500 个合成样本
关闭 quality_fallback
训练阶段开启 base preservation loss
base_preservation_weight = 0.2
```

`base preservation loss` 的目的，是让模型在学习目标类判别概念时，不要过度破坏原始基座类样本的预测边界。它只作用在原始训练 batch 中属于基座类的样本上，用 teacher-student KL 约束保持基座类行为。

本次运行日志显示训练设备为：

```text
当前训练设备: cuda
```

## 分类性能对比

### pair_topk_base 相对 Baseline

| 指标 | SSCBM Baseline | D-CGFS pair_topk_base | 变化 |
|---|---:|---:|---:|
| overall_a_acc | 57.80% | 66.09% | +8.28% |
| overall_c_acc | 89.72% | 89.47% | -0.25% |
| macro_f1 | 57.23% | 66.11% | +8.88% |
| balanced_accuracy | 57.99% | 66.14% | +8.15% |
| worst_class_acc | 0.00% | 10.00% | +10.00% |
| selected_target_acc | 19.29% | 40.71% | +21.43% |
| selected_base_acc | 82.91% | 77.78% | -5.13% |
| target_to_base_rate | 48.57% | 22.86% | -25.71% |

### 四版方法对比

| 指标 | D-CGFS main | no_conf_filter | pair_topk | pair_topk_base |
|---|---:|---:|---:|---:|
| overall_a_acc | 65.36% | 65.62% | 65.98% | 66.09% |
| overall_c_acc | 89.92% | 89.38% | 89.36% | 89.47% |
| macro_f1 | 65.16% | 65.58% | 65.97% | 66.11% |
| balanced_accuracy | 65.45% | 65.67% | 65.99% | 66.14% |
| worst_class_acc | 3.33% | 6.67% | 6.67% | 10.00% |
| selected_target_acc | 37.86% | 43.57% | 44.29% | 40.71% |
| selected_base_acc | 75.21% | 66.67% | 66.67% | 77.78% |
| target_to_base_rate | 20.71% | 16.43% | 12.86% | 22.86% |

评价：`pair_topk_base` 在整体准确率、Macro-F1、Balanced Accuracy、Worst-class Accuracy 上是当前最强版本；同时它把 `pair_topk` 的基座类准确率从 `66.67%` 拉回到 `77.78%`。但是它牺牲了一部分目标类修复效果，`selected_target_acc` 从 `44.29%` 降到 `40.71%`，`target_to_base_rate` 从 `12.86%` 回升到 `22.86%`。

## 目标类结果

| target_class | Baseline | pair_topk | pair_topk_base |
|---:|---:|---:|---:|
| 59 | 6.67% | 20.00% | 20.00% |
| 65 | 0.00% | 15.00% | 15.00% |
| 74 | 66.67% | 80.00% | 80.00% |
| 144 | 13.33% | 20.00% | 20.00% |
| 176 | 3.33% | 76.67% | 60.00% |

评价：前四个目标类与 `pair_topk` 持平，主要回退来自 `Prairie_Warbler`，从 `76.67%` 降到 `60.00%`。这说明基座类保持损失确实改变了目标类与基座类之间的边界强度，尤其影响原本提升最明显的 pair。

## Target-Base 混淆

| target -> base | Baseline | pair_topk | pair_topk_base |
|---|---:|---:|---:|
| Slaty_backed_Gull -> Herring_Gull | 55.00% | 0.00% | 15.00% |
| California_Gull -> Herring_Gull | 66.67% | 6.67% | 26.67% |
| Common_Tern -> Artic_Tern | 36.67% | 40.00% | 43.33% |
| Florida_Jay -> Lazuli_Bunting | 0.00% | 0.00% | 0.00% |
| Prairie_Warbler -> Cape_May_Warbler | 86.67% | 13.33% | 26.67% |

评价：`pair_topk_base` 仍然显著优于 baseline，但不如 `pair_topk`。这说明 `base_preservation_weight=0.2` 可能偏强，保护基座类的同时削弱了目标类向正确类别移动的力度。

## 可解释性结果

### 概念准确率

| 指标 | SSCBM | pair_topk | pair_topk_base |
|---|---:|---:|---:|
| overall_c_acc | 89.72% | 89.36% | 89.47% |
| target_c_acc | 90.54% | 90.82% | 91.08% |
| target_disc_c_acc | 21.64% | 31.10% | 32.60% |

评价：`pair_topk_base` 的判别概念准确率进一步提高到 `32.60%`，是当前最好结果。这支持一个判断：基座类保持并没有破坏判别概念学习，反而让判别概念预测更稳定。

### Heatmap 质量

| 指标 | SSCBM | pair_topk | pair_topk_base | 评价 |
|---|---:|---:|---:|---|
| heatmap_entropy | 0.9308 | 0.9423 | 0.9428 | 更分散，不是理想变化 |
| mask_compactness | 0.1143 | 0.1225 | 0.1169 | 比 baseline 略好，但低于 pair_topk |
| bbox_energy_ratio | 0.6228 | 0.6011 | 0.5939 | 低于 baseline 和 pair_topk |

评价：Heatmap 仍然不是当前方法的主要优势。分类和判别概念准确率在提升，但空间定位质量没有同步改善。

### 概念干预

| 指标 | SSCBM | pair_topk | pair_topk_base |
|---|---:|---:|---:|
| original_acc | 19.29% | 44.29% | 40.71% |
| intervention_acc | 78.57% | 80.71% | 79.29% |
| intervention_gain | 58.57% | 36.43% | 38.57% |

评价：`pair_topk_base` 的原始目标类准确率低于 `pair_topk`，干预后准确率也略低于 `pair_topk`。说明这次改动主要改善整体稳健性和基座类保持，而不是提高目标类的可干预上限。

## 结论

`pair_topk_base` 是目前最均衡的版本，但不是目标类修复最强的版本。

它的优点：

- `overall_a_acc = 66.09%`，当前最高。
- `macro_f1 = 66.11%`，当前最高。
- `balanced_accuracy = 66.14%`，当前最高。
- `worst_class_acc = 10.00%`，当前最高。
- `selected_base_acc = 77.78%`，明显优于 `pair_topk` 的 `66.67%`。
- `target_disc_c_acc = 32.60%`，当前最高。

它的问题：

- `selected_target_acc = 40.71%`，低于 `pair_topk` 的 `44.29%`。
- `target_to_base_rate = 22.86%`，明显差于 `pair_topk` 的 `12.86%`。
- `Common_Tern -> Artic_Tern` 的混淆仍未解决，甚至略差。
- Heatmap 空间定位指标仍然没有形成稳定优势。

## 下一步建议

下一步不建议放弃 `base preservation`，因为它确实恢复了基座类准确率，并提升了整体指标。更合理的方向是调小基座类保持强度，寻找目标类修复和基座类保持之间的折中点。

建议优先运行：

```text
pair_topk_filter + base_preservation_weight = 0.05
pair_topk_filter + base_preservation_weight = 0.10
pair_topk_filter + base_preservation_weight = 0.15
```

判断标准：

```text
selected_target_acc 尽量接近或超过 44.29%
selected_base_acc 保持高于 75%
target_to_base_rate 尽量低于 16%
overall_a_acc / macro_f1 / balanced_accuracy 不低于当前 pair_topk_base
```

如果 `0.10` 左右能同时保持 `selected_base_acc >= 75%` 和 `target_to_base_rate <= 16%`，那么它会比当前 `0.2` 更适合作为论文主结果。
