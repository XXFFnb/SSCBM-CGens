# D-CGFS base preservation 权重扫描实验记录

## 记录信息

- 日期：2026-05-19
- 实验目的：在 `pair_topk_filter` 基础上扫描 `base_preservation_weight`，寻找目标类修复与基座类保持之间的折中点。
- 运行命令：`bash run_dcgfs_pipeline.sh pair_topk_base_sweep`
- 复用合成数据目录：`generated_data/dcgfs_pair_topk/`
- 扫描权重：`0.05`、`0.10`、`0.15`
- 对照权重：此前已运行的 `0.20`
- 分类评估目录：
  - `final_evaluation_results/dcgfs_pair_topk_base_w005/`
  - `final_evaluation_results/dcgfs_pair_topk_base_w010/`
  - `final_evaluation_results/dcgfs_pair_topk_base_w015/`
  - `final_evaluation_results/dcgfs_pair_topk_base/`
- 可解释性评估目录：
  - `explainability_results/dcgfs_pair_topk_base_w005/`
  - `explainability_results/dcgfs_pair_topk_base_w010/`
  - `explainability_results/dcgfs_pair_topk_base_w015/`
  - `explainability_results/dcgfs_pair_topk_base/`

## 代码更新

本轮为 `run_dcgfs_pipeline.sh` 新增了 `pair_topk_base_sweep` 模式。

该模式会复用 `generated_data/dcgfs_pair_topk`，依次训练和评估：

```text
base_preservation_weight = 0.05
base_preservation_weight = 0.10
base_preservation_weight = 0.15
```

每个权重都有独立的 checkpoint、分类评估目录、可解释性评估目录和日志文件，不会覆盖已有结果。

## 分类性能对比

| 方法 | overall_a_acc | macro_f1 | balanced_acc | worst_class_acc | selected_target_acc | selected_base_acc | target_to_base_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| pair_topk | 65.98% | 65.97% | 65.99% | 6.67% | 44.29% | 66.67% | 12.86% |
| base_w005 | 66.02% | 65.90% | 66.03% | 6.67% | 42.86% | 71.79% | 17.14% |
| base_w010 | 65.98% | 65.97% | 66.04% | 10.00% | 42.14% | 75.21% | 18.57% |
| base_w015 | 65.74% | 65.72% | 65.80% | 6.67% | 44.29% | 77.78% | 17.86% |
| base_w020 | 66.09% | 66.11% | 66.14% | 10.00% | 40.71% | 77.78% | 22.86% |

评价：

- `base_w005` 的整体准确率略高于 `pair_topk`，但基座类准确率只恢复到 `71.79%`，没有达到预期。
- `base_w010` 把基座类准确率恢复到 `75.21%`，与 main 版本相当，但目标类准确率降到 `42.14%`。
- `base_w015` 是当前最平衡的权重：目标类准确率 `44.29%` 追平 `pair_topk`，基座类准确率 `77.78%` 追平 `base_w020`，同时 `target_to_base_rate=17.86%` 明显好于 `base_w020`。
- `base_w020` 的整体指标最高，但目标类修复和 target-to-base confusion 回退明显，说明权重偏强。

## base_w015 目标类结果

| target_class | Baseline | base_w015 |
|---:|---:|---:|
| 59 | 6.67% | 30.00% |
| 65 | 0.00% | 15.00% |
| 74 | 66.67% | 76.67% |
| 144 | 13.33% | 23.33% |
| 176 | 3.33% | 66.67% |

评价：`base_w015` 不是依赖单个类别拉高平均值，而是 5 个目标类全部相对 baseline 有提升。其中 `California_Gull`、`Common_Tern`、`Prairie_Warbler` 的提升比 `base_w020` 更合理。

## base_w015 Target-Base 混淆

| target -> base | Baseline | base_w015 |
|---|---:|---:|
| Slaty_backed_Gull -> Herring_Gull | 55.00% | 5.00% |
| California_Gull -> Herring_Gull | 66.67% | 23.33% |
| Common_Tern -> Artic_Tern | 36.67% | 36.67% |
| Florida_Jay -> Lazuli_Bunting | 0.00% | 0.00% |
| Prairie_Warbler -> Cape_May_Warbler | 86.67% | 20.00% |

评价：`base_w015` 显著缓解了大多数 target-to-base confusion，但 `Common_Tern -> Artic_Tern` 仍未解决，只是回到 baseline 水平。这仍然是后续需要单独分析的 pair。

## 可解释性对比

### 概念准确率

| 方法 | overall_c_acc | target_c_acc | target_disc_c_acc |
|---|---:|---:|---:|
| pair_topk | 89.36% | 90.82% | 31.10% |
| base_w005 | 89.41% | 91.06% | 33.56% |
| base_w010 | 89.40% | 91.13% | 32.47% |
| base_w015 | 89.46% | 91.18% | 32.19% |
| base_w020 | 89.47% | 91.08% | 32.60% |

评价：所有 base preservation 权重都提升了判别概念准确率，说明这个约束没有破坏判别概念学习。`base_w005` 的判别概念准确率最高，但分类层面的基座类保持不足。

### Heatmap 质量

| 方法 | heatmap_entropy | mask_compactness | bbox_energy_ratio |
|---|---:|---:|---:|
| pair_topk | 0.9423 | 0.1225 | 0.6011 |
| base_w005 | 0.9429 | 0.1124 | 0.5973 |
| base_w010 | 0.9439 | 0.1136 | 0.5949 |
| base_w015 | 0.9426 | 0.1150 | 0.5966 |
| base_w020 | 0.9428 | 0.1169 | 0.5939 |

评价：Heatmap 指标仍然不是主要优势。权重变化对空间定位质量没有形成稳定改善，后续如果要强化可解释性贡献，需要单独设计 heatmap 或概念定位相关约束。

### 概念干预

| 方法 | original_acc | intervention_acc | intervention_gain |
|---|---:|---:|---:|
| pair_topk | 44.29% | 80.71% | 36.43% |
| base_w005 | 42.86% | 78.57% | 35.71% |
| base_w010 | 42.14% | 80.00% | 37.86% |
| base_w015 | 44.29% | 79.29% | 35.00% |
| base_w020 | 40.71% | 79.29% | 38.57% |

评价：干预结果没有随着 base preservation 权重单调变化。`base_w010` 的 intervention_acc 达到 `80.00%`，但整体分类折中不如 `base_w015`。

## 结论

当前最推荐把 `base_preservation_weight=0.15` 作为下一版主候选。

理由：

- 目标类准确率 `44.29%`，追平目前目标类修复最强的 `pair_topk`。
- 基座类准确率 `77.78%`，明显高于 `pair_topk` 的 `66.67%`，也高于 main 的 `75.21%`。
- target-to-base rate 为 `17.86%`，虽然不如 `pair_topk` 的 `12.86%`，但明显优于 `base_w020` 的 `22.86%`。
- 5 个目标类均相对 baseline 提升，没有只依赖单个类别。
- 判别概念准确率 `32.19%`，高于 `pair_topk` 的 `31.10%`。

因此，`base_w015` 比 `pair_topk` 更适合作为论文主方法：它保留了目标类修复能力，同时解决了 `pair_topk` 基座类准确率明显下降的问题。

## 下一步建议

下一步不建议继续大范围扫权重。更值得做的是围绕 `0.15` 做更细的局部验证：

```text
base_preservation_weight = 0.125
base_preservation_weight = 0.175
```

如果资源有限，可以先不跑这两个权重，直接把 `0.15` 暂定为主方法，然后转向两个更关键的问题：

1. 单独分析 `Common_Tern -> Artic_Tern`，因为它是目前唯一没有明显解决的 target-base pair。
2. 准备论文结果表，把 `pair_topk` 作为无基座保护消融，把 `base_w015` 作为完整 D-CGFS。
