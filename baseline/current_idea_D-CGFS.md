# D-CGFS 主方案与 Idea 阐述

**D-CGFS: Discriminative Concept-Guided Feature Synthesis for Imbalanced Semi-Supervised Concept Bottleneck Models**

## 1. 核心 Idea

本项目基于 SSCBM（Semi-Supervised Concept Bottleneck Models）提出 D-CGFS，用于缓解半监督概念瓶颈模型中的类别不平衡问题。

SSCBM 的优势在于，它不是直接从图像黑盒预测类别，而是先预测一组人类可理解的概念，再通过概念表示完成分类。因此，SSCBM 不仅能输出类别预测，还能提供概念概率、概念 embedding、概念区域响应和概念级干预接口。

D-CGFS 的核心思想是：

> 既然 SSCBM 已经学习到了概念空间和概念区域，那么这些概念不应该只用于解释模型预测，也可以反过来指导弱势类的数据增强。

更具体地说，对于一个弱势目标类，D-CGFS 不尝试从零生成整张图像，也不做普通的随机 MixUp/CutMix，而是寻找一个语义相近、结构可复用的基座类，将目标类相对于基座类更有判别力的概念区域迁移到基座类特征中，从而生成目标类一致、结构合理、概念可追踪的合成特征。

## 2. 研究问题

在细粒度视觉数据集中，不同类别往往共享高度相似的姿态、轮廓、背景和局部结构。对于少数类或弱势类，SSCBM 容易出现以下问题：

1. 目标类判别概念学习不足。
2. 目标类的关键概念区域定位不稳定。
3. 目标类容易被语义相似的强势类吸走。
4. 普通重采样或重加权只改变训练分布，不直接补充目标类缺失的判别概念证据。

因此，本项目关注的问题是：

> 如何利用概念瓶颈模型自身学到的概念结构，为弱势目标类构造具有判别概念证据的合成特征，并在提升弱势类识别能力的同时尽量保持基座类和整体概念预测稳定？

## 3. 方法假设

D-CGFS 建立在三个假设上。

### 3.1 相似类别共享非判别结构

细粒度类别之间虽然标签不同，但通常共享大量非判别性结构，例如鸟的身体轮廓、姿态、背景、拍摄角度和基础纹理。

因此，对于弱势目标类，不一定需要生成完整新样本。更合理的做法是：

```text
保留基座类结构上下文 + 注入目标类判别概念证据
```

### 3.2 判别概念比全部概念更关键

目标类和基座类之间真正决定分类边界的不是全部概念，而是目标类相对于基座类更强的那部分概念。

对于一个 target-base pair，定义概念原型差：

```text
gap_k = proto_target[k] - proto_base[k]
```

判别概念集合定义为：

```text
D(target, base) = Top-K concepts where gap_k > 0
```

这个集合表示目标类相对于基座类更具有区分力的概念。

### 3.3 特征空间合成比图像空间合成更适合 SSCBM

图像空间 copy-paste 容易引入边缘伪影、尺度错位和背景割裂。D-CGFS 直接在 SSCBM 的中间特征空间进行合成，使增强过程更贴近模型真正使用的概念表示。

合成目标可以概括为：

```text
target discriminative concept feature
        +
base structural feature
        ->
synthetic target feature
```

## 4. 方法流程

D-CGFS 的完整流程包括六个阶段。

### 4.1 弱势目标类发现

首先在验证集上评估 SSCBM 的类别表现，自动选择识别较差或容易混淆的类别作为弱势目标类。

这一步的目的不是人工指定少数类，而是让模型根据自身错误模式暴露出需要增强的类别。

输出为目标类集合：

```text
Y_target = {y_t^1, y_t^2, ..., y_t^m}
```

### 4.2 语义兼容基座类选择

对于每个目标类 `y_t`，D-CGFS 自动选择一个基座类 `y_b`。基座类需要满足两个条件：

1. 与目标类在视觉或概念空间中足够相似，能够提供可复用结构。
2. 与目标类存在混淆关系，说明它位于目标类附近的决策边界上。

综合得分可以写成：

```text
S(y_t, y_b) =
    alpha * S_confusion(y_t, y_b)
  + beta  * S_visual(y_t, y_b)
  + gamma * S_concept(y_t, y_b)
```

其中：

- `S_confusion` 表示 baseline 将目标类误分为基座类的倾向。
- `S_visual` 表示视觉特征原型相似度。
- `S_concept` 表示概念原型相似度。

### 4.3 判别概念集合构造

对每个 target-base pair，计算目标类和基座类的概念原型差，只保留目标类更强的概念作为判别概念集合。

```text
D = {k | proto_target[k] - proto_base[k] > 0}
```

实际实现中取分数最高的 Top-K 个概念，避免过多弱相关概念引入噪声。

这一步使方法从普通概念增强变为判别概念增强。

### 4.4 判别概念区域定位

对于判别概念集合 `D`，利用 SSCBM 的概念响应生成空间 mask：

```text
M_D(i, j) = max_{k in D} H_k(i, j)
```

其中 `H_k` 是概念 `k` 的空间响应。阈值化后得到判别概念区域：

```text
R_D(i, j) = 1[M_D(i, j) > tau]
```

该区域表示目标类相对于基座类更关键的局部语义区域。

### 4.5 判别概念门控特征合成

D-CGFS 在 SSCBM 的中间特征图上进行合成。设目标类特征图为 `F_t`，基座类特征图为 `F_b`，基座类判别区域 mask 为 `M_b`。

目标类判别区域特征：

```text
F_t^D = F_t * M_t
```

根据目标类和基座类的判别概念差异生成门控强度：

```text
G_D = sigmoid(temperature * mean(proto_target[D] - proto_base[D]))
```

合成特征：

```text
F_syn = F_b * (1 - M_b)
      + Fuse(F_b * M_b, F_t^D, G_D)
```

直观上，合成特征保留了基座类的整体结构和背景，只在判别概念区域注入目标类证据。

### 4.6 Pair 内质量选择

由于弱势目标类本身难以被原模型识别，不能简单使用绝对 target confidence 阈值过滤样本。D-CGFS 使用 pair 内排序策略：每个 target-base pair 内独立选择质量最高的合成样本。

质量分数：

```text
score = concept_delta
      + lambda_t * log P(y_target | F_syn)
      - lambda_b * log P(y_base | F_syn)
```

其中：

```text
concept_delta =
    sim(c_syn, proto_target) - sim(c_syn, proto_base)
```

当前主方案使用：

```text
lambda_t = 0.15
lambda_b = 0.05
```

这个设计同时考虑三件事：

1. 合成样本的概念预测应更接近目标类。
2. 合成样本应保留一定目标类分类证据。
3. 合成样本不应过度停留在基座类决策边界内。

## 5. 训练目标

合成样本用于训练 SSCBM 的后半部分。特征提取器保持冻结，避免少量合成特征破坏底层视觉表示。

训练损失包括：

```text
L = L_original
  + L_synthetic
  + lambda_concept * L_concept
  + lambda_proto * L_proto
  + lambda_preserve * L_base_preserve
```

其中：

- `L_original` 保持原始训练数据上的任务学习。
- `L_synthetic` 使用合成特征的目标类标签进行监督。
- `L_concept` 约束判别概念预测。
- `L_proto` 约束合成概念表示靠近目标类概念原型。
- `L_base_preserve` 保护基座类样本上的 teacher 输出，避免目标类增强破坏相近基座类。

当前主方案使用：

```text
lambda_preserve = 0.15
```

## 6. 方法特点

D-CGFS 与常见不平衡学习方法的区别如下。

### 6.1 不同于重采样

重采样只是重复已有少数类样本，不能补充新的判别区域组合。D-CGFS 构造的是带有目标类判别概念证据的新特征样本。

### 6.2 不同于重加权

重加权只改变损失权重，不能改变弱势类样本覆盖不足的问题。D-CGFS 同时改变训练样本的特征分布和概念监督。

### 6.3 不同于普通 MixUp/CutMix

普通 MixUp/CutMix 不知道哪些区域或概念真正区分目标类与基座类。D-CGFS 只围绕判别概念区域合成。

### 6.4 不同于黑盒生成模型

D-CGFS 不依赖 GAN 或 diffusion 生成整图，而是在 SSCBM 的概念特征空间中构造可追踪的增强样本。

## 7. 论文贡献收束

D-CGFS 可以收束为三个主要贡献。

1. **Weak target-base pair discovery**

   基于验证集混淆、视觉原型和概念原型，自动发现需要增强的弱势目标类及其语义兼容基座类。

2. **Discriminative concept-guided feature synthesis**

   利用目标类相对于基座类更强的判别概念集合，在 SSCBM 中间特征空间进行局部语义合成。

3. **Pair-wise quality selection with base preservation**

   在每个 target-base pair 内选择概念一致且带有目标类证据的合成样本，并通过 base preservation 约束保护相近基座类决策边界。

## 8. 当前主方案配置

当前主方案可用如下配置表示：

```text
method = D-CGFS
synthesis_mode = pair_topk_filter
pair_topk = 500
pair_score_target_weight = 0.15
pair_score_base_weight = 0.05
fusion_mode = old_spatial
synthesis_task_concept_source = base
base_preservation_weight = 0.15
```

该配置体现的核心取舍是：

1. 使用 pair 内 top-k，避免弱势类被绝对置信度阈值过度过滤。
2. 在质量分数中加入目标类概率和基座类概率项，使合成样本更靠近目标类决策方向。
3. 使用 base preservation，在增强目标类的同时保持相近基座类稳定。

## 9. 写作定位

论文中应把 D-CGFS 定位为：

> 一种利用概念瓶颈模型内部概念结构进行弱势类增强的可解释特征合成方法。

重点不是宣称它替代所有不平衡学习 baseline，而是强调：

1. 它利用了 SSCBM 独有的概念空间和概念区域信息。
2. 它能够针对 target-base confusion 构造有语义指向的合成特征。
3. 它在弱势类增强、基座类保持和概念可解释性之间提供了明确机制。

## 10. 当前实验结论

### 10.1 CUB-200-2011

CUB 是当前主实验数据集。结果显示，D-CGFS 对弱势目标类和整体性能均有明显提升。

| 指标 | SSCBM | D-CGFS |
| --- | ---: | ---: |
| 整体任务准确率 | 57.80% | 65.65% |
| Macro-F1 | 57.23% | 65.65% |
| Balanced Acc | 57.99% | 65.72% |
| 最差类别准确率 | 0.00% | 10.00% |
| 自动弱势目标类准确率 | 19.29% | 42.86% |
| 自动基座类准确率 | 82.91% | 77.78% |
| 目标类错分为基座类比例 | 48.57% | 20.71% |

这说明 D-CGFS 在 CUB 上确实缓解了 target-base confusion，并显著提升了弱势目标类识别能力。基座类准确率有所下降，但通过 base preservation 后仍维持在相对可接受范围内。

### 10.2 AwA2

AwA2 作为第二数据集，用于验证 D-CGFS 是否只在 CUB 鸟类细粒度分类上有效。

| 指标 | SSCBM | D-CGFS |
| --- | ---: | ---: |
| 整体任务准确率 | 89.39% | 90.01% |
| Macro-F1 | 86.02% | 86.59% |
| Balanced Acc | 85.60% | 86.46% |
| 最差类别准确率 | 45.45% | 29.41% |
| 自动弱势目标类准确率 | 55.65% | 55.65% |
| 自动基座类准确率 | 81.44% | 85.57% |
| 目标类错分为基座类比例 | 17.74% | 19.35% |
| 整体概念准确率 | 96.48% | 96.15% |

AwA2 的结果更复杂：D-CGFS 提升了整体任务准确率、Macro-F1、Balanced Acc 和基座类准确率，但没有提升自动弱势目标类准确率，目标类错分为基座类比例略有上升。

因此，跨数据集结论应谨慎表述为：

> D-CGFS 在第二数据集 AwA2 上表现出整体泛化收益和基座类稳定性收益，但弱势目标类定向改善存在数据集依赖。

### 10.3 与强 baseline 的关系

在 CUB 上，reweighting 和 class-balanced loss 的自动弱势目标类准确率高于 D-CGFS，但它们对基座类准确率的损伤也更明显。D-CGFS 的写作重点不应是“所有指标最好”，而应是：

1. 它提供了概念引导的弱势类增强机制。
2. 它显式建模 target-base confusion。
3. 它能在改善弱势类和保持基座类之间形成可解释的折中。
4. 它的合成样本和训练目标可以追踪到具体判别概念。

## 11. 一句话总结

D-CGFS 将 SSCBM 的概念解释能力转化为数据增强能力：通过自动发现弱势目标类和相似基座类，在判别概念区域进行特征空间合成，从而为不平衡半监督概念瓶颈模型提供可解释、可追踪的弱势类增强机制。
