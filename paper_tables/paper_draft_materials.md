# D-CGFS 论文初稿生成材料总表

本文档用于交给 AI 生成论文或报告初稿。它汇总了本项目的研究动机、方法思路、模型结构、算法流程、实验设置、结果表、结论边界、代码文件位置和写作注意事项。生成论文初稿时，应以本文档为主，结合 `paper_tables/` 下的结果表，不要再把历史探索分支误写成主方法。

当前正式方法名称：

```text
D-CGFS: Discriminative Concept-Guided Feature Synthesis
```

中文可译为：

```text
判别概念引导的特征合成方法
```

当前主方法配置：

```text
method = D-CGFS
baseline = SSCBM
synthesis_mode = pair_topk_filter
pair_topk = 500
pair_score_target_weight = 0.15
pair_score_base_weight = 0.05
fusion_mode = old_spatial
synthesis_task_concept_source = base
base_preservation_weight = 0.15
seed = 42
```

主结果表路径：

```text
paper_tables/multi_dataset_results.csv
paper_tables/final_report_summary.md
paper_tables/final_main_results.csv
paper_tables/final_strong_baseline_context.csv
paper_tables/final_archived_diagnostics.csv
paper_tables/metric_glossary_zh.csv
paper_tables/cub_seed_sweep_summary.csv
paper_tables/7pt_seed_sweep_summary.csv
paper_tables/7pt_labeled_ratio_sweep.csv
paper_tables/cub_main_explainability.csv
```

## 1. 项目整体目标

本项目基于 SSCBM（Semi-Supervised Concept Bottleneck Models）提出 D-CGFS，用于改善半监督概念瓶颈模型在弱势类别、难分类类别或不均衡类别上的识别能力。

SSCBM 的核心特点是：模型不是直接从图像预测类别，而是先预测一组人类可理解的概念，再通过概念表示完成分类。因此，SSCBM 同时具备：

1. 任务分类能力。
2. 概念预测能力。
3. 概念空间中的可解释表示。
4. 概念响应热图或局部区域响应。
5. 概念干预接口。

本项目的核心问题是：

```text
如何把 SSCBM 已经学习到的概念空间和概念区域信息，从“解释模型”进一步转化为“增强模型”的能力？
```

D-CGFS 的回答是：

```text
自动发现弱势目标类及其容易混淆的基座类，提取目标类相对于基座类更强的判别概念区域，
在 SSCBM 中间特征空间中将目标类判别概念证据注入基座类结构特征，
再用合成特征对模型后半部分进行平衡微调。
```

## 2. 背景与动机

### 2.1 SSCBM 的优势与不足

SSCBM 通过少量有概念标注数据和大量弱标注数据学习概念瓶颈表示。它比普通黑盒分类器更可解释，也比传统 CBM 更适合半监督场景。

但是，在细粒度视觉分类或类别分布不均衡的数据中，SSCBM 仍存在以下问题：

1. 某些类别的训练样本或有效概念证据不足。
2. 弱势类与强势类共享相似外观，容易被强势类吸收。
3. 目标类关键概念区域定位不稳定。
4. 普通重采样和重加权只改变样本出现频率或损失权重，不能生成新的判别概念组合。
5. 图像空间复制粘贴可能产生边缘伪影、尺度错位和背景割裂，不适合细粒度语义增强。

因此，项目尝试利用 SSCBM 自身的概念结构进行更有语义指向的数据增强。

### 2.2 D-CGFS 的核心动机

D-CGFS 的核心动机可以写成：

```text
概念瓶颈模型已经学到了概念空间和概念响应区域；
这些概念不应只用于解释预测，也可以反向用于指导弱势类特征合成。
```

与普通增强方法相比，D-CGFS 不从零生成整图，也不随机混合样本，而是：

1. 找到一个模型表现差的目标类。
2. 找到一个和目标类相似、且常与目标类混淆的基座类。
3. 找到目标类相对于基座类更强的判别概念集合。
4. 使用 SSCBM 的概念热图定位这些判别概念区域。
5. 在中间特征空间中合成新目标类特征。
6. 使用合成特征训练模型，同时保护基座类决策边界。

## 3. 关键术语

| 术语 | 英文 | 含义 |
| --- | --- | --- |
| 目标类 | target class | baseline 中表现较差、需要增强的弱势类。 |
| 基座类 | base class | 与目标类相似且容易混淆的类别，用于提供结构和非判别上下文。 |
| 判别概念集合 | discriminative concept set D | 目标类相对于基座类更强的概念集合。 |
| 概念原型 | concept prototype | 某类别在概念标签或概念预测空间中的平均向量。 |
| 视觉原型 | visual prototype | 某类别在 backbone 特征空间中的平均向量。 |
| 概念区域 mask | concept region mask | 由判别概念热图融合并阈值化得到的空间区域。 |
| pair-topk | pair-wise top-k filtering | 每个 target-base pair 内独立保留质量最高的合成样本。 |
| base preservation | base preservation | 训练时保护基座类输出，避免增强目标类时破坏相近基座类。 |

## 4. 方法总览

D-CGFS 的完整流程可以分为七个步骤。

### Step 1: 弱势目标类发现

在验证集上评估 baseline SSCBM，统计每个类别的准确率、混淆矩阵、视觉原型和概念原型。选择表现较差或容易混淆的类别作为目标类。

代码入口：

```text
find_weak_classes.py
```

输出：

```text
data/D-CGFS_Auxiliary*/target_base_pairs.csv
```

### Step 2: 基座类选择

对于每个目标类 `y_t`，选择一个语义兼容的基座类 `y_b`。基座类选择综合三类信息：

```text
S(y_t, y_b) =
    alpha * S_confusion(y_t, y_b)
  + beta  * S_visual(y_t, y_b)
  + gamma * S_concept(y_t, y_b)
```

其中：

1. `S_confusion`：baseline 将目标类错分为候选基座类的比例。
2. `S_visual`：目标类与候选基座类的视觉原型余弦相似度。
3. `S_concept`：目标类与候选基座类的概念原型余弦相似度。

当前代码中的默认权重：

```text
alpha = 0.5
beta = 0.3
gamma = 0.2
```

### Step 3: 判别概念集合构造

对于一个 target-base pair，计算概念原型差：

```text
gap_k = proto_target[k] - proto_base[k]
```

只保留目标类比基座类更强的概念：

```text
D = TopK({k | gap_k > 0})
```

默认 `DISC_TOP_K = 10`。这一步保证方法关注真正区分目标类与基座类的概念，而不是所有概念。

### Step 4: 判别概念区域定位

利用 SSCBM 的 `plot_heatmap` 得到每个概念的空间响应。对于判别概念集合 `D`，融合热图：

```text
M_D(i, j) = max_{k in D} H_k(i, j)
```

阈值化得到概念区域：

```text
R_D(i, j) = 1[M_D(i, j) > tau]
```

代码入口：

```text
step1_generate_mapping.py
```

输出：

```text
concept_region_mapping.csv
base_sample_regions.csv
masks/
```

### Step 5: 判别概念引导的特征合成

D-CGFS 在 SSCBM 中间特征空间进行合成。设：

```text
F_t = 目标样本特征图
F_b = 基座样本特征图
M_t = 目标样本判别概念区域 mask
M_b = 基座样本判别概念区域 mask
```

目标判别区域特征：

```text
F_t^D = F_t * M_t
```

基座非判别结构：

```text
F_b^keep = F_b * (1 - M_b)
```

合成思想：

```text
保留基座类结构 + 注入目标类判别概念区域特征
```

当前主配置使用 `fusion_mode = old_spatial`，即保留当前主实验中验证过的空间 masked feature 融合逻辑。代码中也实现了更复杂的 pooled prototype 和概念门控模块，但当前主结果以已验证配置为准。

核心模块：

```text
models/dcgfs_modules.py
```

合成入口：

```text
step2_synthesize_data.py
```

输出：

```text
generated_data/*_target_score_w015/
  metadata.csv
  feat_*.pt
  synthesized_images/
  candidate_filter_diagnostics.csv
  pair_filter_summary.csv
```

### Step 6: pair-topk 质量筛选

由于弱势目标类本来就难以被 baseline 正确识别，如果使用过高的绝对目标类置信度阈值，会把大多数合成样本过滤掉。因此主方法采用 pair 内排序策略。

每个 target-base pair 内保留质量分数最高的 `top-k` 个合成样本，当前：

```text
pair_topk = 500
```

质量分数：

```text
score = concept_delta
      + lambda_t * log P(y_target | F_syn)
      - lambda_b * log P(y_base | F_syn)
```

其中：

```text
concept_delta = sim(c_syn, proto_target) - sim(c_syn, proto_base)
lambda_t = 0.15
lambda_b = 0.05
```

含义：

1. 合成样本的概念预测应更接近目标类。
2. 合成样本应具有目标类分类证据。
3. 合成样本不应仍然过度停留在基座类决策区域。

### Step 7: 平衡训练与 base preservation

使用合成特征对 SSCBM 后半部分进行微调。训练时冻结 `pre_concept_model`，避免少量合成特征破坏底层视觉表示。

代码入口：

```text
step3_balance_training.py
```

主要训练目标包括：

```text
L = L_original
  + L_synthetic
  + concept/prototype constraints
  + lambda_preserve * L_base_preserve
```

其中 `base preservation` 使用 teacher model 保护基座类样本上的输出稳定性。当前主配置：

```text
base_preservation_weight = 0.15
```

## 5. 与其他方法的区别

### 5.1 与重采样不同

重采样只是重复已有样本，无法提供新的判别概念组合。D-CGFS 构造的是带有目标类判别概念证据的新特征样本。

### 5.2 与重加权不同

重加权改变损失权重，但不改变数据覆盖。D-CGFS 直接扩展弱势目标类在特征空间中的覆盖。

### 5.3 与 MixUp/CutMix 不同

普通 MixUp/CutMix 不知道哪些区域或概念真正区分目标类和基座类。D-CGFS 显式使用判别概念集合 `D` 和概念区域 mask。

### 5.4 与生成模型不同

D-CGFS 不依赖 GAN 或 diffusion 生成整张图像，而是在 SSCBM 已学习到的概念特征空间中进行局部语义合成，因此更轻量，也更贴合 SSCBM 的内部决策机制。

### 5.5 与 feature refinement 等历史分支不同

历史探索中尝试过 feature refinement、retrieval residual、model-aware D、hybrid pair refinement 等更激进策略。它们可作为诊断实验记录，但不作为当前论文主方法。论文主线只写：

```text
D-CGFS target_score_w015 + base preservation
```

历史分支可以在讨论或附录中一笔带过，不应在方法章节中作为主算法组成部分。

## 6. 实验设置

### 6.1 数据集

本项目最终覆盖四个数据集，与 SSCBM 原论文使用的数据集范围对齐：

| 数据集 | 任务类型 | 类别数 | 概念数 | 当前用途 |
| --- | --- | ---: | ---: | --- |
| CUB-200-2011 | 鸟类细粒度分类 | 200 | 112 | 主实验数据集，提供最完整的分析结果。 |
| AwA2 | 动物属性分类 | 50 | 85 | 跨数据集验证。 |
| PBC / WBCatt | 白细胞分类 | 5 | 31 | 高性能近饱和场景验证。 |
| 7-point | 皮肤病变分类 | 5 | 19 | 困难低基线场景验证。 |

数据集位置：

```text
data/CUB_200_2011
data/AwA2/Animals_with_Attributes2
data/PBC
data/7-point/release_v0
```

配置文件：

```text
configs/CUB-200-2011.yaml
configs/AwA2.yaml
configs/PBC.yaml
configs/7pt.yaml
```

### 6.2 Baseline

基础模型是 SSCBM。各数据集的 baseline checkpoint：

```text
checkpoints/CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt
checkpoints/AwA2_16-15/SemiSupervisedConceptEmbeddingModel.pt
checkpoints/PBC_17-01/SemiSupervisedConceptEmbeddingModel.pt
checkpoints/7pt_17-44/SemiSupervisedConceptEmbeddingModel.pt
```

D-CGFS 训练后的 checkpoint：

```text
checkpoints/problem6_dcgfs_target_score_w015.pt
checkpoints/awa2_dcgfs_target_score_w015.pt
checkpoints/pbc_dcgfs_target_score_w015.pt
checkpoints/7pt_dcgfs_target_score_w015.pt
```

### 6.3 随机种子

当前实验统一使用：

```text
seed = 42
```

主表仍以 seed=42 作为可复现实验结果。同时，为验证结论稳定性，已额外完成 CUB 与 7-point 的 3-seed 汇总：

```text
paper_tables/cub_seed_sweep_detail.csv
paper_tables/cub_seed_sweep_summary.csv
paper_tables/7pt_seed_sweep_detail.csv
paper_tables/7pt_seed_sweep_summary.csv
```

写作时建议同时报告主表和多 seed 补充表。多 seed 结果说明 D-CGFS 的整体趋势存在，但不同 seed 下弱势类选择、baseline 强度和尾部类别表现会有波动。

### 6.4 评价指标

| 指标 | 中文名称 | 含义 |
| --- | --- | --- |
| overall_acc / overall_a_acc | 整体任务准确率 | 全部测试样本上的类别预测准确率。 |
| overall_concept_acc / overall_c_acc | 整体概念准确率 | 全部测试样本、全部概念上的概念预测准确率。 |
| macro_f1 | 宏平均 F1 | 先按类别计算 F1，再对类别平均，减轻类别不平衡影响。 |
| balanced_accuracy | 平衡准确率 | 各类别召回率平均值，观察类别是否被均衡识别。 |
| worst_class_acc | 最差类别准确率 | 测试集中表现最差类别准确率，反映尾部风险。 |
| selected_target_acc | 自动弱势目标类准确率 | `find_weak_classes.py` 自动选出的目标类上的准确率。 |
| selected_base_acc | 自动基座类准确率 | 被选为基座类的相似类别上的准确率，用于检查副作用。 |
| target_to_base_rate | 目标类错分为基座类比例 | 目标类样本被预测成对应基座类的比例，越低越好。 |

论文中不要只报告 overall accuracy。D-CGFS 的核心主张与弱势类、基座类、副作用和 target-base confusion 有关，因此必须报告 selected target/base 和 target-to-base rate。

## 7. 主实验结果

最新跨数据集结果来自：

```text
paper_tables/multi_dataset_results.csv
```

### 7.1 四数据集主表

| 数据集 | 方法 | 整体任务准确率 | Macro-F1 | Balanced Acc | 最差类别准确率 | 自动弱势目标类准确率 | 自动基座类准确率 | 目标错分为基座比例 | 整体概念准确率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUB | SSCBM | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| CUB | D-CGFS | 65.65 | 65.65 | 65.72 | 10.00 | 42.86 | 77.78 | 20.71 | 89.49 |
| AwA2 | SSCBM | 89.39 | 86.02 | 85.60 | 45.45 | 55.65 | 81.44 | 17.74 | 96.48 |
| AwA2 | D-CGFS | 90.01 | 86.59 | 86.46 | 29.41 | 55.65 | 85.57 | 19.35 | 96.15 |
| PBC | SSCBM | 99.65 | 99.50 | 99.50 | 97.82 | 99.65 | 99.49 | 0.29 | 94.14 |
| PBC | D-CGFS | 99.65 | 99.53 | 99.57 | 98.37 | 99.65 | 99.49 | 0.23 | 93.30 |
| 7pt | SSCBM | 61.27 | 46.39 | 47.53 | 22.00 | 61.27 | 61.11 | 23.54 | 69.81 |
| 7pt | D-CGFS | 66.84 | 51.77 | 50.68 | 22.22 | 66.84 | 67.20 | 20.76 | 71.38 |

### 7.2 CUB 结果解读

CUB 是主实验数据集。D-CGFS 在 CUB 上效果最明显：

1. 整体任务准确率从 57.80% 提升到 65.65%，提升 7.85 个百分点。
2. Macro-F1 从 57.23% 提升到 65.65%，提升 8.42 个百分点。
3. Balanced Acc 从 57.99% 提升到 65.72%，提升 7.72 个百分点。
4. 最差类别准确率从 0.00% 提升到 10.00%。
5. 自动弱势目标类准确率从 19.29% 提升到 42.86%，提升 23.57 个百分点。
6. 目标类错分为基座类比例从 48.57% 降到 20.71%，下降 27.86 个百分点。
7. 整体概念准确率略降，从 89.72% 到 89.49%，说明任务收益并非来自概念预测整体提升，而主要来自判别概念引导的决策边界调整。

可写结论：

```text
On CUB, D-CGFS substantially improves weak-target recognition and reduces target-to-base confusion,
validating the motivation of discriminative concept-guided synthesis.
```

### 7.2.1 CUB 多 seed 稳定性

CUB 已完成 seed 0、seed 1、seed 42 的 3-seed 汇总，结果在：

```text
paper_tables/cub_seed_sweep_summary.csv
```

3-seed 平均结果：

| 指标 | SSCBM mean | D-CGFS mean | 平均变化 |
| --- | ---: | ---: | ---: |
| 整体任务准确率 | 65.17 | 68.59 | +3.42 |
| 整体概念准确率 | 91.05 | 90.19 | -0.86 |
| Macro-F1 | 64.80 | 68.50 | +3.70 |
| Balanced Acc | 65.27 | 68.66 | +3.39 |
| 最差类别准确率 | 4.44 | 5.56 | +1.11 |
| 自动弱势目标类准确率 | 36.72 | 47.24 | +10.53 |
| 自动基座类准确率 | 65.66 | 63.42 | -2.24 |
| 目标类错分为基座类比例 | 22.74 | 13.45 | -9.29 |

该结果支持 CUB 上的主结论：D-CGFS 平均提升整体任务表现、Macro-F1、Balanced Acc 和弱势目标类准确率，并降低 target-to-base confusion。需要注意的是，seed 间波动较大，尤其 seed42 的 baseline 明显更低，因此论文中应同时报告 mean/std，不应只展示单次最强结果。

### 7.3 AwA2 结果解读

AwA2 上 D-CGFS 的表现更温和：

1. 整体任务准确率从 89.39% 提升到 90.01%。
2. Macro-F1 从 86.02% 提升到 86.59%。
3. Balanced Acc 从 85.60% 提升到 86.46%。
4. 自动弱势目标类准确率保持 55.65% 不变。
5. 自动基座类准确率从 81.44% 提升到 85.57%。
6. target-to-base rate 从 17.74% 上升到 19.35%。
7. 整体概念准确率略降。

可写结论：

```text
AwA2 shows overall generalization and base-class stabilization, but weak-target improvement is dataset-dependent.
```

不要把 AwA2 写成“弱势目标类显著提升”。正确写法是整体鲁棒性提升、基座类更稳、目标类收益有限。

### 7.4 PBC 结果解读

PBC baseline 已经非常强，任务准确率 99.65%，几乎接近饱和。因此 D-CGFS 没有明显提升空间：

1. 整体任务准确率保持 99.65%。
2. Macro-F1 从 99.50% 微升到 99.53%。
3. Balanced Acc 从 99.50% 微升到 99.57%。
4. 最差类别准确率从 97.82% 提升到 98.37%。
5. target-to-base rate 从 0.29% 降到 0.23%。
6. 整体概念准确率从 94.14% 降到 93.30%。

可写结论：

```text
On the near-saturated PBC dataset, D-CGFS preserves task performance and slightly improves balanced and worst-class metrics,
indicating that the method does not destabilize an already strong baseline.
```

不要夸大 PBC。它主要证明方法不会明显破坏高性能模型。

### 7.5 7-point 结果解读

7-point 是困难数据集，baseline 明显较弱，因此更能体现 D-CGFS 的增益：

1. 整体任务准确率从 61.27% 提升到 66.84%，提升 5.57 个百分点。
2. 整体概念准确率从 69.81% 提升到 71.38%，提升 1.57 个百分点。
3. Macro-F1 从 46.39% 提升到 51.77%，提升 5.38 个百分点。
4. Balanced Acc 从 47.53% 提升到 50.68%，提升 3.15 个百分点。
5. 自动弱势目标类准确率从 61.27% 提升到 66.84%。
6. 自动基座类准确率从 61.11% 提升到 67.20%。
7. target-to-base rate 从 23.54% 降到 20.76%。

可写结论：

```text
On 7-point, D-CGFS improves both task and concept metrics, suggesting that concept-guided synthesis is especially useful when the baseline has substantial room for improvement.
```

### 7.5.1 7-point 多 seed 稳定性

7-point 已完成 seed 0、seed 1、seed 2 的 3-seed 汇总，结果在：

```text
paper_tables/7pt_seed_sweep_summary.csv
```

3-seed 平均结果：

| 指标 | SSCBM mean | D-CGFS mean | 平均变化 |
| --- | ---: | ---: | ---: |
| 整体任务准确率 | 62.28 | 64.98 | +2.70 |
| 整体概念准确率 | 69.03 | 68.69 | -0.33 |
| Macro-F1 | 42.37 | 44.03 | +1.66 |
| Balanced Acc | 43.06 | 44.01 | +0.95 |
| 最差类别准确率 | 9.26 | 7.41 | -1.85 |
| 自动弱势目标类准确率 | 62.28 | 64.98 | +2.70 |
| 自动基座类准确率 | 68.85 | 72.11 | +3.26 |
| 目标类错分为基座类比例 | 23.88 | 23.38 | -0.51 |

该结果说明 7-point 上 D-CGFS 平均提升整体任务准确率、弱势目标类准确率和基座类准确率，但 Macro-F1、Balanced Acc、worst-class 等尾部指标存在明显 seed 波动，不能写成“所有尾部指标稳定提升”。更稳妥的论文表述是：D-CGFS 在 7-point 多 seed 上显示出平均整体收益，但尾部类别改善仍不稳定。

### 7.5.2 7-point 标注比例敏感性

为补充半监督设置下不同概念标注比例的影响，已在 7-point 上完成 `labeled_ratio=0.05` 和 `labeled_ratio=0.20` 的主方法实验。结果表在：

```text
paper_tables/7pt_labeled_ratio_sweep.csv
```

核心结果：

| 标注比例 | 指标 | SSCBM | D-CGFS | 变化 |
| --- | --- | ---: | ---: | ---: |
| 0.05 | 整体任务准确率 | 67.09 | 70.38 | +3.29 |
| 0.05 | 整体概念准确率 | 68.78 | 68.33 | -0.45 |
| 0.05 | Macro-F1 | 49.02 | 55.36 | +6.34 |
| 0.05 | Balanced Acc | 48.03 | 52.22 | +4.19 |
| 0.05 | 最差类别准确率 | 21.95 | 27.78 | +5.83 |
| 0.05 | 目标类错分为基座类比例 | 25.06 | 22.28 | -2.78 |
| 0.20 | 整体任务准确率 | 57.47 | 65.82 | +8.35 |
| 0.20 | 整体概念准确率 | 71.41 | 72.83 | +1.43 |
| 0.20 | Macro-F1 | 39.20 | 44.67 | +5.47 |
| 0.20 | Balanced Acc | 37.80 | 46.26 | +8.46 |
| 0.20 | 最差类别准确率 | 9.76 | 11.11 | +1.36 |
| 0.20 | 目标类错分为基座类比例 | 33.92 | 22.78 | -11.14 |

该结果说明，在 7-point 上 D-CGFS 的收益不是只出现在默认 `labeled_ratio=0.10`。在更低标注比例 0.05 和更高标注比例 0.20 下，D-CGFS 均提升整体任务准确率、Macro-F1、Balanced Acc 和弱势目标类准确率，并降低 target-to-base confusion。需要注意，0.20 下 baseline 任务准确率低于 0.10/0.05，说明该数据集的划分和训练波动较明显；论文中应把该实验写作敏感性补充，而不是绝对单调的标注比例规律。

## 8. 强 baseline 与主方法定位

CUB 上的强 baseline 结果在：

```text
paper_tables/final_strong_baseline_context.csv
```

核心事实：

1. Reweighting 和 class-balanced loss 在 selected target accuracy 上高于 D-CGFS。
2. 但它们对 selected base accuracy 的损伤更大。
3. D-CGFS 的优势不是单纯追求 selected target accuracy 最高，而是提供概念引导、target-base 显式建模、可解释合成与基座类保持之间的折中。

写作建议：

```text
D-CGFS should be positioned as a concept-guided weak-class augmentation method,
not as a universal replacement for all imbalance baselines.
```

中文写法：

```text
D-CGFS 并不试图替代所有不平衡学习方法，而是利用概念瓶颈模型内部概念结构，
提供一种可解释、可追踪、面向 target-base confusion 的弱势类增强机制。
```

## 9. 可解释性结果

CUB 上已经运行过可解释性评估，原始结果路径：

```text
explainability_results/problem6_dcgfs_target_score_w015/
```

论文表格汇总：

```text
paper_tables/cub_main_explainability.csv
```

关键结果：

### 9.1 概念准确率

```text
overall_c_acc: 0.8972 -> 0.8949
target_c_acc: 0.9054 -> 0.9104
target_disc_c_acc: 0.2164 -> 0.3110
```

解读：

1. 整体概念准确率略降。
2. 目标类概念准确率略升。
3. 目标类判别概念准确率明显提升，从 21.64% 到 31.10%。

这支持 D-CGFS 的核心动机：方法不是提升所有概念，而是强化弱势目标类相关判别概念。

### 9.2 Heatmap 质量

```text
heatmap_entropy: 0.9308 -> 0.9399
mask_compactness: 0.1143 -> 0.1071
bbox_energy_ratio: 0.6228 -> 0.5983
```

解读应谨慎。heatmap 质量没有全面改善，部分指标略有下降。论文中不要声称 heatmap localization 全面提升。可以写成：

```text
D-CGFS mainly improves discriminative concept prediction and task-level behavior;
spatial heatmap compactness is not consistently improved.
```

### 9.3 概念干预

```text
original_acc: 0.1929 -> 0.4286
intervention_acc: 0.7786 -> 0.8143
intervention_gain: 0.5857 -> 0.3857
```

解读：

1. 未干预准确率大幅提升。
2. 干预后准确率也提升。
3. intervention gain 下降，是因为原始准确率变高，留给干预的提升空间变小。

可写：

```text
After D-CGFS, the model relies less on external intervention to reach higher weak-target accuracy,
suggesting that discriminative concept evidence has been partially internalized.
```

## 10. 消融和历史探索

历史探索结果不作为主方法，但可以用于讨论方法选择。

路径：

```text
paper_tables/main_component_ablation.csv
paper_tables/final_archived_diagnostics.csv
experiment_records/
final_evaluation_results/problem6_dcgfs_*
```

### 10.1 主组件消融

当前已整理主组件消融表：

```text
paper_tables/main_component_ablation.csv
```

核心结果：

| 方法 | 整体任务准确率 | Macro-F1 | Balanced Acc | 最差类别准确率 | 弱势目标类准确率 | 基座类准确率 | 目标错分为基座比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 |
| no confidence/pair quality filtering | 65.62 | 65.58 | 65.67 | 6.67 | 43.57 | 66.67 | 16.43 |
| pair-topk without base preservation | 65.98 | 65.97 | 65.99 | 6.67 | 44.29 | 66.67 | 12.86 |
| pair-topk + weak base preservation w005 | 66.02 | 65.90 | 66.03 | 6.67 | 42.86 | 71.79 | 17.14 |
| pair-topk + base preservation w015 | 65.74 | 65.72 | 65.80 | 6.67 | 44.29 | 77.78 | 17.86 |
| main target_score_w015 + base preservation | 65.65 | 65.65 | 65.72 | 10.00 | 42.86 | 77.78 | 20.71 |

消融结论应这样写：

1. 所有 D-CGFS 变体相对 SSCBM baseline 都提升整体任务准确率、Macro-F1、Balanced Acc 和弱势目标类准确率。
2. 不加 base preservation 的 pair-topk 在 target-to-base rate 上更低，但基座类准确率下降明显。
3. 加入 base preservation 后，基座类准确率从 66.67% 恢复到 77.78%，说明该约束有效限制了对相似基座类的副作用。
4. 当前主方法牺牲了一部分 target-to-base rate 最优性，换取更好的 worst-class accuracy 和基座类保持，因此是更均衡的论文主配置。

### 10.2 历史探索分支

需要说明的分支：

| 分支 | 结论 |
| --- | --- |
| retrieval residual | 作为诊断保留，没有成为主方法。 |
| model-aware D | 判别概念更激进，但收益不稳定。 |
| feature refinement | 特征空间优化更强，但没有带来稳定主收益。 |
| feature refine window | 加入置信窗口和源样本多样性后仍不足以成为主线。 |
| hybrid pair refinement | 局部 pair 替换略有收益，但复杂度高，不作为主方案。 |

写论文时：

1. 主方法章节不要展开这些分支。
2. 如果需要讨论，可在 ablation 或 limitation 中写“更激进的特征优化没有稳定超过主方案”。
3. 不要让读者以为这些分支是 D-CGFS 的必要组件。

## 11. 项目结构与文件说明

### 11.1 核心方法文件

| 文件 | 作用 |
| --- | --- |
| `dataset_specs.py` | 统一管理 CUB、AwA2、PBC、7pt 的配置、checkpoint、概念数、类别数和 loader。 |
| `dcgfs_config.py` | D-CGFS 主方法名称、权重等统一配置。 |
| `find_weak_classes.py` | 自动发现弱势目标类、基座类和判别概念集合。 |
| `step1_generate_mapping.py` | 生成判别概念区域 mask 和 target/base 映射 CSV。 |
| `step2_synthesize_data.py` | 在特征空间合成 D-CGFS 样本，并进行 pair-topk 质量筛选。 |
| `step3_balance_training.py` | 使用原始数据和合成特征训练 D-CGFS 模型。 |
| `step4_final_evaluation.py` | 输出分类指标、目标类/基座类指标和混淆指标。 |
| `step5_explainability_evaluation.py` | CUB 上的概念准确率、heatmap、概念干预评估。 |
| `models/dcgfs_modules.py` | ConceptLocator 和 ConceptGatedFuser 模块定义。 |
| `models/sscbm.py` | SSCBM 模型主体。 |
| `run_problem6_experiments.py` | 当前主实验调度入口。 |

### 11.2 数据加载文件

| 文件 | 作用 |
| --- | --- |
| `data/cub_loader.py` | CUB 数据加载。 |
| `data/awa2_loader.py` | AwA2 数据加载。 |
| `data/pbc_loader.py` | PBC/WBCatt 数据加载。 |
| `data/pt_loader.py` | 7-point 数据加载。 |

### 11.3 配置文件

| 文件 | 作用 |
| --- | --- |
| `configs/CUB-200-2011.yaml` | CUB baseline 配置。 |
| `configs/AwA2.yaml` | AwA2 baseline 配置。 |
| `configs/PBC.yaml` | PBC baseline 配置。 |
| `configs/7pt.yaml` | 7-point baseline 配置。 |

### 11.4 结果目录

| 目录 | 内容 |
| --- | --- |
| `final_evaluation_results/` | 最终分类评估结果。 |
| `explainability_results/` | CUB 可解释性评估结果。 |
| `paper_tables/` | 论文表格、汇总结果、本文档。 |
| `experiment_records/` | 实验过程记录和历史探索总结。 |
| `generated_data/` | D-CGFS 合成特征与可视化图片。 |
| `data/D-CGFS_Auxiliary*` | target-base pair、mask、mapping 等辅助文件。 |
| `checkpoints/` | baseline 和 D-CGFS checkpoint。 |
| `problem6_experiment_protocol/` | 自动生成的实验命令 CSV。 |

### 11.5 baseline 资料

| 文件 | 作用 |
| --- | --- |
| `baseline/Semi-supervised Concept Bottleneck Models.pdf` | SSCBM 原论文。 |
| `baseline/current_idea_D-CGFS.md` | D-CGFS 主方案 idea 文档。 |
| `baseline/README.md` | baseline 目录说明。 |

## 12. 复现实验命令

### 12.1 CUB 主实验

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group main
```

### 12.2 AwA2 主实验

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group awa2_main
```

### 12.3 PBC 主实验

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group pbc_main
```

### 12.4 7-point 主实验

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group 7pt_main
```

### 12.5 报告表格生成

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group report
```

### 12.6 多 seed 和标注比例补充实验

CUB 多 seed：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group cub_seed_sweep_extra
```

7-point 多 seed：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group 7pt_seed_sweep
```

7-point 标注比例敏感性：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group 7pt_labeled_ratio_sweep
```

### 12.7 baseline 训练命令

PBC baseline：

```bash
.venv/bin/python main.py --dataset PBC --device cuda --image_encoder resnet34
```

7-point baseline：

```bash
.venv/bin/python main.py --dataset 7pt --device cuda --image_encoder resnet34
```

AwA2 baseline：

```bash
.venv/bin/python main.py --dataset AwA2 --device cuda --image_encoder resnet34
```

## 13. 推荐论文结构

建议论文或报告按以下结构写。

### Abstract

应包含：

1. SSCBM 在半监督概念瓶颈学习中有解释性优势。
2. 弱势类和相似类别混淆仍然存在。
3. 提出 D-CGFS，将概念解释能力转化为概念引导的数据增强能力。
4. 方法自动选择 target-base pair，定位判别概念区域，在特征空间合成目标类样本。
5. 四个数据集上验证，CUB 和 7pt 的主要指标提升较明显，AwA2/PBC 保持或小幅改善整体鲁棒性。

### Introduction

建议逻辑：

1. 概念瓶颈模型提高可解释性。
2. 半监督概念瓶颈模型降低概念标注需求。
3. 但弱势类、类别不平衡和细粒度混淆仍然限制模型。
4. 现有重采样/重加权/普通混合增强缺少概念级语义指向。
5. 本文提出 D-CGFS，用 SSCBM 内部概念响应指导特征合成。
6. 总结贡献。

### Related Work

可分为：

1. Concept Bottleneck Models。
2. Semi-supervised Concept Bottleneck Models。
3. Imbalanced learning。
4. Feature synthesis and data augmentation。
5. Explainable concept-guided learning。

### Method

建议小节：

1. Problem formulation。
2. SSCBM baseline。
3. Weak target and base class discovery。
4. Discriminative concept selection。
5. Concept region localization。
6. Discriminative concept-guided feature synthesis。
7. Pair-wise quality selection。
8. Balanced training with base preservation。

### Experiments

建议小节：

1. Datasets。
2. Implementation details。
3. Evaluation metrics。
4. Main results across datasets。
5. CUB detailed analysis。
6. Strong baseline comparison。
7. Explainability analysis。
8. Ablation/diagnostic discussion。

### Discussion

建议强调：

1. D-CGFS 在 CUB 和 7pt 上主要指标提升较明显。
2. AwA2 的收益主要体现在整体和基座类稳定。
3. PBC baseline 近饱和，提升空间有限，但 D-CGFS 没有破坏性能。
4. 方法不是所有不平衡学习 baseline 的替代品，而是概念引导的弱势类增强机制。
5. 未来可以做四数据集完整多 seed、更大范围 labeled-ratio sweep、更多合成质量控制和更强 concept localization。

### Conclusion

建议写法：

```text
本文提出 D-CGFS，将 SSCBM 的概念解释能力转化为可解释的数据增强能力。
通过自动发现 weak target-base pair、选择判别概念、定位概念区域并进行特征空间合成，
D-CGFS 能够改善弱势类和困难分类场景中的任务表现。
四个数据集实验表明，该方法在 CUB 和 7-point 上提升多项主要指标，
在 AwA2 和 PBC 上保持或小幅改善整体鲁棒性。
```

## 14. 建议贡献点写法

可以写成三点贡献：

1. 提出一种面向半监督概念瓶颈模型的弱势类增强框架 D-CGFS，将概念解释能力用于指导特征合成。
2. 设计 target-base pair discovery 与 discriminative concept selection 机制，显式建模弱势类与相似基座类之间的混淆关系。
3. 提出 pair-wise quality filtering 与 base preservation 训练策略，在提升弱势类表现的同时限制对相似基座类的副作用，并在四个数据集上验证方法有效性与边界。

## 15. 推荐摘要草稿

下面是一段可供 AI 初稿使用的中文摘要草稿：

```text
概念瓶颈模型通过显式预测人类可理解概念提升了深度模型的可解释性，半监督概念瓶颈模型进一步降低了对大规模概念标注的依赖。然而，在细粒度和类别不均衡场景中，弱势类别仍容易缺乏稳定的判别概念证据，并被语义相似的强势类别混淆。为此，本文提出 D-CGFS，一种判别概念引导的特征合成方法。该方法首先基于验证集错误模式、视觉原型和概念原型自动发现弱势目标类及其语义兼容基座类；随后选择目标类相对于基座类更强的判别概念集合，并利用 SSCBM 的概念响应定位判别区域；最后在中间特征空间中将目标类判别概念证据注入基座类结构特征，生成用于平衡训练的合成目标类特征。为控制合成样本质量，D-CGFS 采用 pair 内 top-k 筛选，并结合 base preservation 约束减轻对相似基座类的副作用。在 CUB-200-2011、AwA2、PBC/WBCatt 和 7-point 四个数据集上的实验表明，D-CGFS 在 CUB 和 7-point 上提升任务准确率、Macro-F1 和弱势类相关指标，在 AwA2 和 PBC 上保持或小幅改善整体鲁棒性。结果表明，将概念解释结构转化为概念引导的数据增强机制，是提升半监督概念瓶颈模型弱势类表现的一条有效路径。
```

## 16. 推荐英文标题

可选标题：

```text
Discriminative Concept-Guided Feature Synthesis for Imbalanced Semi-Supervised Concept Bottleneck Models
```

中文标题：

```text
面向不平衡半监督概念瓶颈模型的判别概念引导特征合成方法
```

## 17. 写作边界和注意事项

论文初稿生成时必须注意：

1. 不要声称 D-CGFS 在所有指标上都超过重加权或 class-balanced loss。
2. 不要声称 AwA2 的弱势目标类准确率提升，因为它保持不变。
3. 不要声称 PBC 有显著提升，因为 baseline 已近饱和。
4. 不要把 feature refinement、retrieval residual、model-aware D、hybrid pair refinement 写成主方法。
5. 不要把 heatmap localization 写成全面提升；CUB heatmap 指标并不全面改善。
6. 可以强调 CUB 和 7pt 的主要指标提升，以及 PBC/AwA2 上的方法稳定性和跨数据集鲁棒性。
7. 可以强调 target-to-base confusion 在 CUB 和 7pt 上下降，这与方法动机直接相关。
8. 可以强调 selected target/base 指标体现了方法不是只看整体准确率，而是关注弱势类和副作用。
9. 当前已经补充 CUB 与 7-point 的 3-seed 结果，以及 7-point 的 labeled-ratio 0.05/0.20 敏感性实验；AwA2/PBC 仍主要报告 seed=42，若需要更严格统计显著性，可以继续扩展到四数据集全 seed。
10. 论文应把 D-CGFS 定位为概念引导的增强方法，而不是通用不平衡学习最优解。

## 18. 一句话核心结论

```text
D-CGFS 将 SSCBM 的概念解释能力转化为弱势类增强能力：通过自动发现 target-base confusion、选择判别概念、定位概念区域并在特征空间合成目标类证据，它在 CUB 和 7-point 等困难场景中改善分类表现，并在高性能数据集上保持整体稳定。
```
