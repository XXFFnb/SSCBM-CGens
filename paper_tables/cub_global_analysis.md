# CUB 全局分析：D-CGFS 主方法

本文档整理 CUB-200-2011 上已有评估结果，重点补充整体指标和全类别分布分析。

## 指标解释

| metric | 中文名称 | 解释 |
| --- | --- | --- |
| overall_a_acc | 整体任务准确率 | 所有测试样本上的类别预测准确率。 |
| overall_c_acc | 整体概念准确率 | 所有概念预测的平均准确率。 |
| macro_f1 | 宏平均 F1 | 先计算每个类别 F1，再对类别取平均；更关注类别均衡性。 |
| balanced_accuracy | 平衡准确率 | 每个类别召回率的平均值，能减少类别样本数差异的影响。 |
| worst_class_acc | 最差类别准确率 | 所有类别中准确率最低的类别表现。 |
| selected_target_acc | 自动弱势目标类准确率 | 自动选出的弱势目标类集合上的准确率。 |
| selected_base_acc | 自动基座类准确率 | 与弱势目标类配对的基座类集合上的准确率。 |
| target_to_base_rate | 目标类错分为基座类比例 | 弱势目标类样本被预测成对应基座类的比例，越低越好。 |

## 整体指标对比

表中数值均为百分比。`目标类错分为基座类比例` 越低越好，其余指标越高越好。

| method | overall_a_acc | overall_c_acc | macro_f1 | balanced_accuracy | worst_class_acc | selected_target_acc | selected_base_acc | target_to_base_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSCBM 原始基线 | 57.80 | 89.72 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 |
| SSCBM 微调 | 66.76 | 89.56 | 66.62 | 66.87 | 3.33 | 56.43 | 71.79 | 16.43 |
| 过采样 Oversampling | 66.48 | 89.51 | 66.06 | 66.53 | 13.33 | 50.71 | 64.10 | 10.71 |
| 重加权 Reweighting | 67.31 | 89.60 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 |
| Class-balanced loss | 67.31 | 89.60 | 67.23 | 67.42 | 3.33 | 58.57 | 70.94 | 15.71 |
| Feature Mixup | 66.69 | 88.92 | 66.65 | 66.87 | 3.33 | 57.14 | 68.38 | 15.00 |
| D-CGFS 主方法 | 65.65 | 89.49 | 65.65 | 65.72 | 10.00 | 42.86 | 77.78 | 20.71 |

## 相对 SSCBM 原始基线的变化

表中数值为百分点变化。正数表示提升，负数表示下降。

| method | overall_a_acc_变化 | overall_c_acc_变化 | macro_f1_变化 | balanced_accuracy_变化 | worst_class_acc_变化 | selected_target_acc_变化 | selected_base_acc_变化 | target_to_base_rate_变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSCBM 微调 | 8.96 | -0.16 | 9.39 | 8.88 | 3.33 | 37.14 | -11.12 | -32.14 |
| 过采样 Oversampling | 8.68 | -0.21 | 8.83 | 8.54 | 13.33 | 31.42 | -18.81 | -37.86 |
| 重加权 Reweighting | 9.51 | -0.12 | 10.00 | 9.43 | 3.33 | 39.28 | -11.97 | -32.86 |
| Class-balanced loss | 9.51 | -0.12 | 10.00 | 9.43 | 3.33 | 39.28 | -11.97 | -32.86 |
| Feature Mixup | 8.89 | -0.80 | 9.42 | 8.88 | 3.33 | 37.85 | -14.53 | -33.57 |
| D-CGFS 主方法 | 7.85 | -0.23 | 8.42 | 7.73 | 10.00 | 23.57 | -5.13 | -27.86 |

## 全类别分组分析

这里把 200 个类别分成自动弱势目标类、自动基座类、非目标非基座类，并统计逐类准确率变化。

| 类别组 | 类别数 | 样本数 | Baseline 平均类准确率 | D-CGFS 平均类准确率 | 平均变化 | 中位数变化 | 提升类别数 | 下降类别数 | 不变类别数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 全部类别 | 200 | 5794 | 57.99 | 65.72 | 7.72 | 6.67 | 134 | 43 | 23 |
| 自动弱势目标类 | 5 | 140 | 18.00 | 41.33 | 23.33 | 20.00 | 5 | 0 | 0 |
| 自动基座类 | 4 | 117 | 82.92 | 78.10 | -4.82 | -5.00 | 1 | 2 | 1 |
| 非目标非基座类 | 191 | 5537 | 58.52 | 66.10 | 7.58 | 6.67 | 128 | 41 | 22 |

## 自动 target-base pair 细节

| model | pair_id | target_class_id | target_class_name | base_class_id | base_class_name | target_samples | target_correct | target_accuracy | target_pred_as_base | target_to_base_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSCBM (Baseline) | 0 | 65 | Slaty backed Gull | 62 | Herring Gull | 20 | 0 | 0.00 | 11 | 55.00 |
| SSCBM (Baseline) | 1 | 59 | California Gull | 62 | Herring Gull | 30 | 2 | 6.67 | 20 | 66.67 |
| SSCBM (Baseline) | 2 | 144 | Common Tern | 141 | Artic Tern | 30 | 4 | 13.33 | 11 | 36.67 |
| SSCBM (Baseline) | 3 | 74 | Florida Jay | 15 | Lazuli Bunting | 30 | 20 | 66.67 | 0 | 0.00 |
| SSCBM (Baseline) | 4 | 176 | Prairie Warbler | 163 | Cape May Warbler | 30 | 1 | 3.33 | 26 | 86.67 |
| D-CGFS target_score_w015 | 0 | 65 | Slaty backed Gull | 62 | Herring Gull | 20 | 4 | 20.00 | 1 | 5.00 |
| D-CGFS target_score_w015 | 1 | 59 | California Gull | 62 | Herring Gull | 30 | 9 | 30.00 | 6 | 20.00 |
| D-CGFS target_score_w015 | 2 | 144 | Common Tern | 141 | Artic Tern | 30 | 6 | 20.00 | 12 | 40.00 |
| D-CGFS target_score_w015 | 3 | 74 | Florida Jay | 15 | Lazuli Bunting | 30 | 25 | 83.33 | 0 | 0.00 |
| D-CGFS target_score_w015 | 4 | 176 | Prairie Warbler | 163 | Cape May Warbler | 30 | 16 | 53.33 | 10 | 33.33 |

## 准确率提升最多的类别

| class_id | class_name | group | samples | accuracy_baseline | accuracy_dcgfs | accuracy_delta | f1_baseline | f1_dcgfs | f1_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 182 | Yellow Warbler | 普通非配对类 | 30 | 16.67 | 76.67 | 60.00 | 28.57 | 83.64 | 55.06 |
| 55 | Evening Grosbeak | 普通非配对类 | 30 | 23.33 | 76.67 | 53.33 | 37.84 | 85.19 | 47.35 |
| 176 | Prairie Warbler | 自动弱势目标类 | 30 | 3.33 | 53.33 | 50.00 | 6.25 | 58.18 | 51.93 |
| 96 | Hooded Oriole | 普通非配对类 | 30 | 30.00 | 73.33 | 43.33 | 42.86 | 72.13 | 29.27 |
| 140 | Summer Tanager | 普通非配对类 | 30 | 33.33 | 73.33 | 40.00 | 50.00 | 83.02 | 33.02 |
| 165 | Chestnut sided Warbler | 普通非配对类 | 30 | 20.00 | 60.00 | 40.00 | 32.43 | 73.47 | 41.04 |
| 39 | Least Flycatcher | 普通非配对类 | 29 | 3.45 | 41.38 | 37.93 | 6.06 | 24.74 | 18.68 |
| 66 | Western Gull | 普通非配对类 | 30 | 3.33 | 40.00 | 36.67 | 6.25 | 43.64 | 37.39 |
| 51 | Horned Grebe | 普通非配对类 | 30 | 33.33 | 70.00 | 36.67 | 46.51 | 64.62 | 18.10 |
| 116 | Chipping Sparrow | 普通非配对类 | 30 | 30.00 | 66.67 | 36.67 | 41.86 | 68.97 | 27.11 |
| 3 | Sooty Albatross | 普通非配对类 | 28 | 17.86 | 53.57 | 35.71 | 29.41 | 56.60 | 27.19 |
| 156 | White eyed Vireo | 普通非配对类 | 30 | 30.00 | 63.33 | 33.33 | 46.15 | 73.08 | 26.92 |
| 31 | Black billed Cuckoo | 普通非配对类 | 30 | 13.33 | 46.67 | 33.33 | 22.86 | 58.33 | 35.48 |
| 188 | Pileated Woodpecker | 普通非配对类 | 30 | 50.00 | 83.33 | 33.33 | 66.67 | 89.29 | 22.62 |
| 153 | Philadelphia Vireo | 普通非配对类 | 29 | 10.34 | 41.38 | 31.03 | 16.67 | 42.86 | 26.19 |

## 准确率下降最多的类别

| class_id | class_name | group | samples | accuracy_baseline | accuracy_dcgfs | accuracy_delta | f1_baseline | f1_dcgfs | f1_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 164 | Cerulean Warbler | 普通非配对类 | 30 | 80.00 | 40.00 | -40.00 | 78.69 | 57.14 | -21.55 |
| 130 | Tree Sparrow | 普通非配对类 | 30 | 50.00 | 20.00 | -30.00 | 44.78 | 31.58 | -13.20 |
| 43 | Yellow bellied Flycatcher | 普通非配对类 | 29 | 58.62 | 31.03 | -27.59 | 26.77 | 30.00 | 3.23 |
| 146 | Forsters Tern | 普通非配对类 | 30 | 60.00 | 36.67 | -23.33 | 48.65 | 39.29 | -9.36 |
| 64 | Ring billed Gull | 普通非配对类 | 30 | 46.67 | 26.67 | -20.00 | 44.44 | 30.19 | -14.26 |
| 62 | Herring Gull | 自动基座类 | 30 | 73.33 | 53.33 | -20.00 | 28.95 | 45.07 | 16.12 |
| 120 | Fox Sparrow | 普通非配对类 | 30 | 73.33 | 53.33 | -20.00 | 55.70 | 65.31 | 9.61 |
| 72 | Pomarine Jaeger | 普通非配对类 | 30 | 53.33 | 33.33 | -20.00 | 35.96 | 41.67 | 5.71 |
| 155 | Warbling Vireo | 普通非配对类 | 30 | 56.67 | 36.67 | -20.00 | 36.96 | 37.93 | 0.97 |
| 50 | Eared Grebe | 普通非配对类 | 30 | 73.33 | 56.67 | -16.67 | 63.77 | 56.67 | -7.10 |
| 1 | Black footed Albatross | 普通非配对类 | 30 | 63.33 | 46.67 | -16.67 | 46.34 | 49.12 | 2.78 |
| 113 | Baird Sparrow | 普通非配对类 | 20 | 55.00 | 40.00 | -15.00 | 57.89 | 55.17 | -2.72 |
| 27 | Shiny Cowbird | 普通非配对类 | 30 | 36.67 | 23.33 | -13.33 | 32.84 | 29.79 | -3.05 |
| 52 | Pied billed Grebe | 普通非配对类 | 30 | 86.67 | 73.33 | -13.33 | 81.25 | 80.00 | -1.25 |
| 160 | Black throated Blue Warbler | 普通非配对类 | 29 | 82.76 | 72.41 | -10.34 | 78.69 | 76.36 | -2.32 |

## 结论摘要

1. D-CGFS 主方法不仅提升自动弱势目标类，也提升 overall、macro-F1、balanced accuracy 和 worst-class accuracy。
2. 自动基座类准确率相对 SSCBM 原始基线下降，这是弱势目标类增强带来的主要 trade-off；base preservation 的作用是减轻这种下降。
3. 非目标非基座类的平均变化用于判断方法是否只服务少数 target pair，还是对整体类别分布也保持稳定。
4. 后续论文表述应同时报告整体指标、目标/基座指标和全类别分布，而不是只报告 selected target accuracy。
