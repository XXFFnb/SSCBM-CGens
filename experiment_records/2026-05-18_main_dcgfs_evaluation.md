# D-CGFS 主流程实验记录与评价

## 记录信息

- 日期：2026-05-18
- 运行命令：`bash run_dcgfs_pipeline.sh main`
- 主方法：`D-CGFS`
- 评估目录：`final_evaluation_results/dcgfs_main/`
- 可解释性评估目录：`explainability_results/dcgfs_main/`
- 训练 checkpoint：`checkpoints/problem6_dcgfs.pt`
- 运行日志目录：`run_logs/`

## 读取的结果文件

本次评价主要参考以下文件：

- `final_evaluation_results/dcgfs_main/model_summary.csv`
- `final_evaluation_results/dcgfs_main/target_class_accuracy.csv`
- `final_evaluation_results/dcgfs_main/target_base_confusion.csv`
- `final_evaluation_results/dcgfs_main/classwise_metrics.csv`
- `explainability_results/dcgfs_main/concept_accuracy.csv`
- `explainability_results/dcgfs_main/heatmap_quality.csv`
- `explainability_results/dcgfs_main/concept_intervention.csv`
- `run_logs/03_synthesize_data_dcgfs.log`
- `run_logs/05_train_dcgfs.log`
- `run_logs/06_final_evaluation.log`
- `run_logs/07_explainability_evaluation.log`

## 总体结论

本次主实验结果总体明显正向。

D-CGFS 相比 baseline SSCBM 在整体分类性能、macro-F1、balanced accuracy、自动弱势目标类准确率和 target-to-base confusion 上都有清晰改善。

尤其重要的是：

- `overall_a_acc` 提升 `+7.58%`
- `macro_f1` 提升 `+7.94%`
- `balanced_accuracy` 提升 `+7.48%`
- `selected_target_acc` 提升 `+18.57%`
- `target_to_base_rate` 下降 `-27.86%`

这说明 D-CGFS 不只是提升单个 case，而是对自动发现的弱势目标类集合有明显帮助，并且确实缓解了“弱势目标类被相似基座类吸走”的问题。

不过，本次结果还不能直接作为最终论文结果。主要风险是合成阶段严格过滤通过的样本较少，大部分合成样本来自 `quality_fallback`，后续需要做过滤策略和 fallback 相关消融。

## 分类性能结果

来自 `model_summary.csv`：

| 指标 | SSCBM Baseline | D-CGFS | 变化 |
|---|---:|---:|---:|
| overall_a_acc | 57.78% | 65.36% | +7.58% |
| overall_c_acc | 89.72% | 89.92% | +0.20% |
| macro_f1 | 57.22% | 65.16% | +7.94% |
| balanced_accuracy | 57.98% | 65.45% | +7.48% |
| worst_class_acc | 0.00% | 3.33% | +3.33% |
| selected_target_acc | 19.29% | 37.86% | +18.57% |
| selected_base_acc | 82.91% | 75.21% | -7.69% |
| target_to_base_rate | 48.57% | 20.71% | -27.86% |

### 评价

分类性能结果是本次实验最强的部分。

整体准确率和 macro 指标同步提升，说明 D-CGFS 并不是只优化少数目标类而牺牲整体性能。

`selected_target_acc` 从 `19.29%` 到 `37.86%`，说明目标弱势类集合的识别能力显著增强。

`target_to_base_rate` 从 `48.57%` 降到 `20.71%`，非常符合 D-CGFS 的核心动机：减少目标类被语义相似的强势基座类错误吸收。

需要注意的是，`selected_base_acc` 从 `82.91%` 降到 `75.21%`。这说明模型确实把一部分决策边界从基座类向目标类移动。这个 trade-off 可以接受，但论文里需要解释，并通过整体 accuracy、macro-F1 和 target-to-base confusion 的改善证明收益大于损失。

## 自动弱势目标类结果

来自 `target_class_accuracy.csv`：

| 目标类 | Baseline | D-CGFS | 变化 |
|---|---:|---:|---:|
| California_Gull | 6.67% | 13.33% | +6.66% |
| Slaty_backed_Gull | 0.00% | 10.00% | +10.00% |
| Florida_Jay | 66.67% | 80.00% | +13.33% |
| Common_Tern | 13.33% | 26.67% | +13.34% |
| Prairie_Warbler | 3.33% | 50.00% | +46.67% |

### 评价

5 个自动选择的弱势目标类全部提升。

其中 `Prairie_Warbler` 从 `3.33%` 提升到 `50.00%`，是最明显的成功案例。

`Florida_Jay` 原本已经有一定准确率，仍从 `66.67%` 提升到 `80.00%`，说明方法不只对极低准确率类别有效。

这组结果支持“多弱势类增强”的论文叙事，比只报告单一 class 65 更有说服力。

## Target-Base 混淆结果

来自 `target_base_confusion.csv`：

| target -> base | Baseline target acc | D-CGFS target acc | Baseline target-to-base | D-CGFS target-to-base |
|---|---:|---:|---:|---:|
| Slaty_backed_Gull -> Herring_Gull | 0.00% | 10.00% | 55.00% | 0.00% |
| California_Gull -> Herring_Gull | 6.67% | 13.33% | 66.67% | 13.33% |
| Common_Tern -> Artic_Tern | 13.33% | 26.67% | 36.67% | 43.33% |
| Florida_Jay -> Lazuli_Bunting | 66.67% | 80.00% | 0.00% | 0.00% |
| Prairie_Warbler -> Cape_May_Warbler | 3.33% | 50.00% | 86.67% | 40.00% |

### 评价

大部分 target-base pair 的混淆明显下降。

最强结果：

- `Slaty_backed_Gull -> Herring_Gull`: 55.00% 降到 0.00%
- `California_Gull -> Herring_Gull`: 66.67% 降到 13.33%
- `Prairie_Warbler -> Cape_May_Warbler`: 86.67% 降到 40.00%

这强烈支持 D-CGFS 的动机：目标类的判别概念注入后，模型不再那么容易把目标类预测成基座类。

但有一个局部失败案例：

- `Common_Tern -> Artic_Tern`: 36.67% 升到 43.33%

虽然 `Common_Tern` 的目标类准确率从 13.33% 到 26.67%，但它被基座类吸走的比例没有改善，反而略升。这说明不是每个 pair 的 base selection 或概念融合都同样有效，后续需要单独分析这个 pair 的判别概念、mask 和合成样本质量。

## 类别整体影响

来自 `classwise_metrics.csv` 的统计：

- 提升类别数：129
- 持平类别数：20
- 下降类别数：51

最大下降类别包括：

| class_id | accuracy 变化 |
|---:|---:|
| 62 | -30.00% |
| 146 | -26.67% |
| 72 | -23.33% |
| 119 | -20.69% |
| 64 | -20.00% |

最大提升类别包括：

| class_id | accuracy 变化 |
|---:|---:|
| 66 | +63.33% |
| 31 | +53.33% |
| 165 | +53.33% |
| 3 | +50.00% |
| 182 | +50.00% |
| 176 | +46.67% |

### 评价

从类别分布看，D-CGFS 对多数类别是正向的：129 个类别提升，51 个类别下降。

这说明整体提升不是少数类别极端提升造成的，而是比较广泛的性能改善。

不过 class 62 下降 30%，而 class 62 正好是多个 gull 目标类的基座类 `Herring_Gull`。这与 `selected_base_acc` 下降一致，说明 D-CGFS 在纠正目标类被基座类吸收时，对部分基座类产生了明显影响。后续可以通过更强的 teacher consistency、base-class preservation loss 或更均衡的合成比例来缓解。

## 合成阶段质量

来自 `run_logs/03_synthesize_data_dcgfs.log`：

```text
过滤后保留合成样本数量: 230/10000
strict: 33
quality_fallback: 197
target_prob:
  min=0.0000, p25=0.0000, median=0.0000, p75=0.0000, max=0.0182
concept_delta = sim_target - sim_base:
  min=-0.1655, p25=-0.0144, median=0.0051, p75=0.0207, max=0.1947
```

### 评价

这是本次实验最大的风险点。

严格过滤通过的样本只有 33 个，大部分样本是 `quality_fallback` 保留的。这说明 baseline 对合成样本目标类概率普遍非常低，`target_prob` 最大也只有 0.0182。

考虑到 CUB 是 200 类任务，随机概率约为 0.005，所以 `0.0182` 并不是完全无意义，但这也说明当前合成样本在 baseline 任务空间中仍然不够强。

因此，虽然最终分类结果很正向，但论文中不能忽略这个问题。必须补充：

1. 不同 `target_conf_threshold` 的敏感性实验。
2. `--no-enable-quality-fallback` 消融。
3. `strict` 样本和 `quality_fallback` 样本分开训练或分开统计。
4. 每个 target-base pair 的保留率分析。

否则审稿人可能质疑结果依赖 fallback 策略，而不是严格的 concept-consistent synthesis。

## 训练过程

来自 `run_logs/05_train_dcgfs.log`：

```text
Epoch 1/20  Avg Loss: 10.2985
Epoch 5/20  Avg Loss: 3.0553
Epoch 10/20 Avg Loss: 2.1218
Epoch 15/20 Avg Loss: 1.9155
Epoch 20/20 Avg Loss: 1.7512
```

### 评价

训练损失整体下降明显，说明微调过程是稳定的。

第 15 到 20 轮之间有轻微波动，但没有发散。考虑到训练混合了原始数据、合成特征、teacher consistency 和 distribution regularization，这个表现可以接受。

后续建议记录每个 loss 分量，而不是只记录总 loss。这样可以判断提升主要来自 task loss、concept loss、prototype loss、teacher consistency 还是 distribution regularization。

## 可解释性结果

### 概念准确率

来自 `concept_accuracy.csv`：

| 指标 | SSCBM | D-CGFS | 变化 |
|---|---:|---:|---:|
| overall_c_acc | 89.72% | 89.92% | +0.20% |
| target_c_acc | 90.54% | 90.94% | +0.40% |
| target_disc_c_acc | 21.64% | 26.16% | +4.52% |

评价：

概念准确率是正向的。尤其目标类判别概念准确率提升 4.52 个百分点，这对 D-CGFS 很重要，说明方法确实改善了目标类判别概念预测，而不是只改变了分类头。

### Heatmap 质量

来自 `heatmap_quality.csv`：

| 指标 | SSCBM | D-CGFS | 变化 |
|---|---:|---:|---:|
| heatmap_entropy | 0.9308 | 0.9369 | +0.0061 |
| mask_compactness | 0.1144 | 0.1074 | -0.0070 |
| bbox_energy_ratio | 0.6227 | 0.6162 | -0.0065 |

评价：

heatmap 质量不是全面提升。

`mask_compactness` 略好，说明高响应区域更紧凑；但 `heatmap_entropy` 略升，`bbox_energy_ratio` 略降，说明热图集中性和落在鸟体 bbox 内的能量没有改善。

因此，当前不能说 D-CGFS 让 heatmap 定位质量显著变好。更准确的表述是：

> D-CGFS 在保持总体概念准确率的同时提升了目标类判别概念预测，但 heatmap localization 仍基本持平，甚至部分指标略有下降。

### 概念干预

来自 `concept_intervention.csv`：

| 指标 | SSCBM | D-CGFS | 变化 |
|---|---:|---:|---:|
| original_acc | 19.29% | 37.86% | +18.57% |
| intervention_acc | 78.57% | 87.14% | +8.57% |
| intervention_gain | 59.29% | 49.29% | -10.00% |

评价：

D-CGFS 的原始目标类准确率和干预后准确率都更高，这是正向结果。

但 `intervention_gain` 下降，是因为 D-CGFS 原始准确率已经更高，留给干预提升的空间变小。这个指标不能简单解读为坏事。论文中应同时报告 `intervention_acc` 和 `intervention_gain`，并说明 D-CGFS 后模型无需干预时已经更强。

## 当前结论

本次主实验可以作为 D-CGFS 的一个强正向初步结果。

支持点：

1. 整体 accuracy、macro-F1、balanced accuracy 全部明显提升。
2. 5 个自动弱势目标类全部提升。
3. target-to-base confusion 大幅下降。
4. 总体概念准确率没有下降。
5. 目标类判别概念准确率提升。
6. 概念干预后的绝对准确率提升。

主要风险：

1. 合成样本严格过滤通过率太低。
2. 大部分合成样本来自 `quality_fallback`。
3. selected base class accuracy 下降。
4. Common_Tern 的 target-to-base confusion 没有改善。
5. heatmap localization 指标没有明显提升。

## 下一步建议

优先级最高的后续实验：

1. 跑 `--no-enable-quality-fallback`，确认没有 fallback 时结果如何。
2. 跑不同过滤阈值：
   - `target_conf_threshold = 0.005`
   - `target_conf_threshold = 0.01`
   - `target_conf_threshold = 0.02`
   - `concept_margin = 0.0 / 0.01 / 0.03`
3. 跑关键消融：
   - `random_mask`
   - `all_concepts_mask`
   - `fixed_alpha`
   - `without_concept_loss`
   - `without_proto_loss`
   - `without_teacher_consistency`
   - `without_distribution_reg`
4. 对 `Common_Tern -> Artic_Tern` 做单独分析，检查判别概念、mask 和合成样本是否合理。
5. 对 class 62 `Herring_Gull` 的下降做分析，因为它是多个 gull 类的基座类。
6. 在训练日志中增加 loss 分量记录，便于判断各损失项贡献。

## 总体评价

当前结果已经超过“只是能跑通”的水平，具备论文方法继续推进的价值。

但当前还不是最终可投稿结论。它更像是：

> D-CGFS 主方法在 CUB 上取得了显著初步收益，但需要通过消融实验、过滤策略分析和合成样本质量分析来证明收益确实来自判别概念引导特征合成，而不是来自 fallback、训练扰动或偶然正则化。

如果后续消融能支持主方法，D-CGFS 就具备较强的顶会/顶刊候选潜力。
