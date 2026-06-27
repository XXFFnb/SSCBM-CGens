# 第二数据集实验总结

当前 CUB 主实验和 AwA2 第二数据集验证均已完成。AwA2 具有类别属性/概念标注，因此用于检验 D-CGFS 是否只在 CUB 鸟类细粒度分类上有效。

## 当前状态

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| AwA2 配置文件 | 已有 | `configs/AwA2.yaml` 已整理为相对路径 `data/AwA2`，类别数 50，概念数 85。 |
| AwA2 loader | 已有 | `data/awa2_loader.py` 可以读取 AwA2 的类别、属性矩阵和图像。 |
| AwA2 数据 | 已有 | 数据位于 `data/AwA2/Animals_with_Attributes2`，共 50 类、37322 张图。 |
| AwA2 预训练 SSCBM | 已完成 | 主 checkpoint 为 `checkpoints/AwA2_16-15/SemiSupervisedConceptEmbeddingModel.pt`。 |
| D-CGFS 脚本泛化 | 已完成主流程 | `find_weak_classes.py`、`step1`、`step2`、`check_synthetic_data.py`、`step3`、`step4` 已支持 `--dataset AwA2`。 |
| AwA2 D-CGFS 冒烟测试 | 已通过 | 已用少量 batch 验证弱势类选择、映射、合成、训练和最终评估代码路径。 |
| AwA2 正式 D-CGFS 主实验 | 已完成 | 结果已写入 `final_evaluation_results/awa2_dcgfs_target_score_w015/` 和 `paper_tables/second_dataset_results.csv`。 |

## AwA2 Baseline 结果

本轮 AwA2 使用 `seed=42`、`resnet34`、`max_epochs=50` 训练 SSCBM。

| 指标 | 中文含义 | 数值 |
| --- | --- | --- |
| test_c_acc | 测试集概念准确率 | 96.48% |
| test_y_acc | 测试集任务/类别准确率 | 89.39% |
| test_c_auc | 测试集概念 AUC | 89.20% |
| test_y_auc | 测试集任务/类别 AUC | 89.39% |
| test_y_f1 | 测试集类别 F1 | 81.71% |

该 checkpoint 可作为 AwA2 上 D-CGFS 的第二数据集基座模型。

## AwA2 D-CGFS 结果

| 指标 | 中文含义 | SSCBM | D-CGFS | 变化 |
| --- | --- | ---: | ---: | ---: |
| overall_acc | 整体任务准确率 | 89.39% | 90.01% | +0.62% |
| macro_f1 | 宏平均 F1 | 86.02% | 86.59% | +0.57% |
| balanced_acc | 平衡准确率 | 85.60% | 86.46% | +0.86% |
| worst_class_acc | 最差类别准确率 | 45.45% | 29.41% | -16.04% |
| selected_target_acc | 自动弱势目标类准确率 | 55.65% | 55.65% | +0.00% |
| selected_base_acc | 自动基座类准确率 | 81.44% | 85.57% | +4.13% |
| target_to_base_rate | 目标类错分为基座类比例 | 17.74% | 19.35% | +1.61% |
| overall_concept_acc | 整体概念准确率 | 96.48% | 96.15% | -0.33% |

## 结论

AwA2 上的结果支持一个更谨慎的跨数据集结论：

1. D-CGFS 在 AwA2 上提升了整体任务准确率、Macro-F1 和 Balanced Acc。
2. D-CGFS 提升了自动基座类准确率，说明 base preservation 在第二数据集上仍有稳定边界的作用。
3. D-CGFS 没有提升 AwA2 自动弱势目标类准确率，并且目标类错分为基座类比例略升。
4. 因此，论文中不能写成“第二数据集同样显著提升弱势目标类”；更合适的表述是“第二数据集验证了整体泛化收益和基座类稳定性，但弱势类定向改善存在数据集依赖”。

## 复现实验命令

检查数据和 checkpoint 是否就绪：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group second_dataset_check
```

重新运行 AwA2 主实验：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group awa2_main
```

重新生成报告表格：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group report
```

AwA2 主实验命令会依次执行：

1. AwA2 弱势目标类和基座类自动选择。
2. AwA2 判别概念映射生成。
3. AwA2 pair-topk 合成数据生成。
4. AwA2 合成特征检查。
5. AwA2 D-CGFS 平衡训练。
6. AwA2 最终分类评估。
