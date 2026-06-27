# 2026-05-20 Problem6 Optimization-Push Results

## 运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group optimization_push
```

## 实验目的

本轮实验针对前一轮发现的瓶颈：`D-CGFS target_score_w015` 已经比原始 SSCBM 明显更强，但自动目标类准确率仍低于 strong baseline。因此本轮验证两个实现层面的假设：

1. 将 strong baseline 中有效的 class-balanced task loss 融合进 D-CGFS，是否能提升 target accuracy。
2. 合成样本任务 logits 如果继续使用 base concept probabilities，可能没有充分把目标概念语义注入分类头；因此测试 `target_disc_mix` 和 `target_proto` 两种更目标导向的 synthetic task concept source。

## 结果汇总

结果表格另存为：

- `paper_tables/problem6_optimization_push_comparison.csv`

| Method | Overall Acc | Macro-F1 | Balanced Acc | Worst Class Acc | Target Acc | Base Acc | Target-to-Base | Concept Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSCBM baseline | 57.80 | 57.23 | 57.99 | 0.00 | 19.29 | 82.91 | 48.57 | 89.72 |
| D-CGFS target_score_w015 | 66.00 | 65.96 | 66.04 | 10.00 | 44.29 | 78.63 | 19.29 | 89.46 |
| dcgfs_cb_task_loss | 66.12 | 66.11 | 66.15 | 10.00 | 41.43 | 76.07 | 19.29 | 89.45 |
| dcgfs_syn_task_target_disc_mix | 65.95 | 66.00 | 66.04 | 6.67 | 40.71 | 76.07 | 20.71 | 89.46 |
| dcgfs_cb_task_target_disc_mix | 66.24 | 66.22 | 66.31 | 6.67 | 40.00 | 77.78 | 20.71 | 89.48 |
| dcgfs_cb_task_target_proto | 66.22 | 66.26 | 66.30 | 6.67 | 40.00 | 77.78 | 20.71 | 89.47 |

## 判断

本轮结果不能替代当前主方法。

`dcgfs_cb_task_target_proto` 的 Macro-F1 最高，为 `66.26`，比当前主方法 `65.96` 高 `+0.30`；`dcgfs_cb_task_target_disc_mix` 的 Balanced Acc 最高，为 `66.31`，比当前主方法 `66.04` 高 `+0.27`。但它们的自动目标类准确率都只有 `40.00`，低于当前主方法 `44.29`，且 target-to-base 错分率也从 `19.29` 升到 `20.71`。

`dcgfs_cb_task_loss` 说明全局 class-balanced loss 确实能轻微提高 overall/macro/balanced，但自动目标类准确率从 `44.29` 降到 `41.43`，base accuracy 从 `78.63` 降到 `76.07`。这表明普通全局重加权并没有对准 D-CGFS 的核心问题。

`target_disc_mix` 和 `target_proto` 没有带来 target accuracy 增益，说明直接替换 synthetic task logits 的概念来源会改变训练信号，但并不一定改善真实测试图像上的目标类判别边界。

## 结论

当前正式主方法仍应保持为：

```text
D-CGFS target_score_w015
pair_topk_filter + pair_score_target_weight=0.15 + base_preservation_weight=0.15
```

本轮实验的价值是排除了两条不够精确的优化方向：

1. 不应把 D-CGFS 直接改成全局 class-balanced loss，因为这会削弱自动目标类修复。
2. 不应直接把合成任务 logits 改成 target prototype 或 target discriminative mix，因为目标概念注入过强时会牺牲真实样本上的目标类准确率。

下一步更合理的方向是做“定向目标类加权”：只对 `target_base_pairs.csv` 中自动选出的目标弱势类增加任务损失权重，而不是对所有少样本类别做全局重加权。对应代码入口已加入 `step3_balance_training.py`，实验组为 `target_class_push`。

推荐下一条命令：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group target_class_push
```
