# D-CGFS no_conf_filter 对照实验记录与评价

## 记录信息

- 日期：2026-05-19
- 实验目的：验证 `target_prob > 0.01` 目标类置信度过滤是否过严。
- 合成数据目录：`generated_data/dcgfs_no_conf_filter/`
- 训练 checkpoint：`checkpoints/best_sscbm_dcgfs_no_conf_filter.pt`
- 分类评估目录：`final_evaluation_results/dcgfs_no_conf_filter/`
- 可解释性评估目录：`explainability_results/dcgfs_no_conf_filter/`
- 对比对象：第一次主实验 `final_evaluation_results/dcgfs_main/`

## 运行命令

```bash
.venv/bin/python step2_synthesize_data.py \
  --ablation-mode no_conf_filter \
  --output-dir generated_data/dcgfs_no_conf_filter \
  --no-enable-quality-fallback

.venv/bin/python step3_balance_training.py \
  --method dcgfs \
  --gen-data-dir generated_data/dcgfs_no_conf_filter \
  --output-checkpoint checkpoints/best_sscbm_dcgfs_no_conf_filter.pt

.venv/bin/python step4_final_evaluation.py \
  --candidate-checkpoint checkpoints/best_sscbm_dcgfs_no_conf_filter.pt \
  --candidate-name "SSCBM + D-CGFS no_conf_filter" \
  --save-dir final_evaluation_results/dcgfs_no_conf_filter

.venv/bin/python step5_explainability_evaluation.py \
  --candidate-checkpoint checkpoints/best_sscbm_dcgfs_no_conf_filter.pt \
  --candidate-name "SSCBM + D-CGFS no_conf_filter" \
  --save-dir explainability_results/dcgfs_no_conf_filter
```

本次训练在非沙箱环境下确认使用 RTX 3090，训练进程显存占用最高观察到约 `17.8GB`。

## 合成数据结果

`no_conf_filter` 只去掉目标类置信度过滤，仍保留概念一致性过滤，并关闭 `quality_fallback`。

合成结果：

```text
保留合成样本数量: 6553/10000
strict: 6553
quality_fallback: 0
```

目标类分布：

| target_class | 合成样本数 |
|---:|---:|
| 59 | 1966 |
| 65 | 1617 |
| 74 | 752 |
| 144 | 952 |
| 176 | 1266 |

这说明之前主实验的瓶颈确实主要来自目标类置信度过滤，而不是概念一致性过滤。

## 分类性能对比

### no_conf_filter 相对 Baseline

| 指标 | SSCBM Baseline | D-CGFS no_conf_filter | 变化 |
|---|---:|---:|---:|
| overall_a_acc | 57.80% | 65.62% | +7.82% |
| overall_c_acc | 89.72% | 89.38% | -0.34% |
| macro_f1 | 57.23% | 65.58% | +8.35% |
| balanced_accuracy | 57.99% | 65.67% | +7.68% |
| worst_class_acc | 0.00% | 6.67% | +6.67% |
| selected_target_acc | 19.29% | 43.57% | +24.29% |
| selected_base_acc | 82.91% | 66.67% | -16.24% |
| target_to_base_rate | 48.57% | 16.43% | -32.14% |

### no_conf_filter 相对第一次主实验 D-CGFS

| 指标 | D-CGFS main | D-CGFS no_conf_filter | 变化 |
|---|---:|---:|---:|
| overall_a_acc | 65.36% | 65.62% | +0.26% |
| overall_c_acc | 89.92% | 89.38% | -0.54% |
| macro_f1 | 65.16% | 65.58% | +0.42% |
| balanced_accuracy | 65.45% | 65.67% | +0.22% |
| worst_class_acc | 3.33% | 6.67% | +3.33% |
| selected_target_acc | 37.86% | 43.57% | +5.71% |
| selected_base_acc | 75.21% | 66.67% | -8.55% |
| target_to_base_rate | 20.71% | 16.43% | -4.29% |

## 目标类结果

| target_class | Baseline | D-CGFS main | no_conf_filter |
|---:|---:|---:|---:|
| 59 | 6.67% | 13.33% | 23.33% |
| 65 | 0.00% | 10.00% | 25.00% |
| 74 | 66.67% | 80.00% | 83.33% |
| 144 | 13.33% | 26.67% | 16.67% |
| 176 | 3.33% | 50.00% | 63.33% |

no_conf_filter 在 4 个目标类上优于主实验，只有 `Common_Tern` 从主实验的 `26.67%` 降到 `16.67%`。

## Target-Base 混淆

| target -> base | Baseline target-to-base | D-CGFS main | no_conf_filter |
|---|---:|---:|---:|
| Slaty_backed_Gull -> Herring_Gull | 55.00% | 0.00% | 0.00% |
| California_Gull -> Herring_Gull | 66.67% | 13.33% | 6.67% |
| Common_Tern -> Artic_Tern | 36.67% | 43.33% | 46.67% |
| Florida_Jay -> Lazuli_Bunting | 0.00% | 0.00% | 0.00% |
| Prairie_Warbler -> Cape_May_Warbler | 86.67% | 40.00% | 23.33% |

no_conf_filter 进一步降低了多数 target-to-base 混淆，但 `Common_Tern -> Artic_Tern` 仍然是失败案例，并且比主实验更差。

## 可解释性结果

### 概念准确率

| 指标 | SSCBM | D-CGFS main | no_conf_filter |
|---|---:|---:|---:|
| overall_c_acc | 89.72% | 89.92% | 89.38% |
| target_c_acc | 90.54% | 90.94% | 90.96% |
| target_disc_c_acc | 21.64% | 26.16% | 30.55% |

no_conf_filter 显著提升目标类判别概念准确率，比主实验更强。

### Heatmap 质量

| 指标 | SSCBM | D-CGFS main | no_conf_filter | 评价 |
|---|---:|---:|---:|---|
| heatmap_entropy | 0.9308 | 0.9369 | 0.9439 | 更分散，较差 |
| mask_compactness | 0.1143 | 0.1074 | 0.1263 | 更紧凑，较好 |
| bbox_energy_ratio | 0.6228 | 0.6162 | 0.6002 | 框内能量下降，较差 |

Heatmap 指标混合。no_conf_filter 的 mask compactness 更好，但 entropy 和 bbox energy ratio 变差。

### 概念干预

| 指标 | SSCBM | D-CGFS main | no_conf_filter |
|---|---:|---:|---:|
| original_acc | 19.29% | 37.86% | 43.57% |
| intervention_acc | 78.57% | 87.14% | 82.14% |
| intervention_gain | 59.29% | 49.29% | 38.57% |

no_conf_filter 的原始目标类准确率更高，但干预后准确率低于主实验。这说明它更强地改变了分类边界，但人工概念干预的上限没有主实验高。

## 评价

本次实验支持一个重要修改方向：固定的目标类置信度过滤 `target_prob > 0.01` 不适合作为弱势类合成样本的硬过滤条件。

理由：

- 关闭该过滤后，严格概念一致样本从 `33` 个增加到 `6553` 个。
- 不需要 `quality_fallback`，样本全部为 `strict`。
- `selected_target_acc` 从主实验的 `37.86%` 进一步提升到 `43.57%`。
- `target_to_base_rate` 从主实验的 `20.71%` 进一步降到 `16.43%`。
- `target_disc_c_acc` 从主实验的 `26.16%` 进一步提升到 `30.55%`。

但它也带来明显副作用：

- `selected_base_acc` 从主实验的 `75.21%` 降到 `66.67%`。
- `overall_c_acc` 从主实验的 `89.92%` 降到 `89.38%`。
- `Common_Tern -> Artic_Tern` 混淆继续恶化。
- Heatmap 的 entropy 和 bbox energy ratio 变差。

因此不建议简单把主方法改成完全 `no_conf_filter`。更合理的下一步是把目标类置信度过滤从“绝对阈值”改成“相对证据过滤”或“pair 内 top-k/top-ratio 过滤”：

1. 保留概念一致性过滤作为主约束。
2. 不再要求 `target_prob > 0.01`。
3. 对每个 target-base pair 按质量分数选择固定比例或固定数量样本。
4. 质量分数可以包含 `concept_delta`、`target_prob`、`target_prob - base_prob`，但不使用绝对目标类概率硬阈值。
5. 对基座类加入保护项或控制合成比例，缓解 base accuracy 下降。

## 下一步建议

优先做一个新的主方法候选：`pair_topk_filter`。

目标：

- 每个 target-base pair 保留数量更均衡。
- 不依赖 `quality_fallback`。
- 不使用过严的绝对目标类置信度阈值。
- 尽量保留 no_conf_filter 的 target acc 提升，同时减少 base acc 损失。

建议第一版策略：

```text
score = concept_delta + 0.05 * log(target_prob + eps) - 0.05 * log(base_prob + eps)
每个 pair 保留 top 300 或 top 500 个候选样本。
```

然后训练并比较：

- `dcgfs_main`
- `dcgfs_no_conf_filter`
- `dcgfs_pair_topk`

如果 `pair_topk` 能保持 `selected_target_acc >= 40%`，同时把 `selected_base_acc` 拉回到 `70%+`，它会比当前两个版本都更适合作为论文主方法。
