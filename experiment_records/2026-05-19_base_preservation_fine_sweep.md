# D-CGFS base preservation 局部细扫实验记录

## 记录信息

- 日期：2026-05-19
- 实验目的：围绕当前较优的 `base_preservation_weight=0.15` 做局部验证，判断是否存在更合适权重。
- 运行命令：`bash run_dcgfs_pipeline.sh pair_topk_base_fine_sweep`
- 新增扫描权重：`0.125`、`0.175`
- 对照权重：`0.15`
- 复用合成数据目录：`generated_data/dcgfs_pair_topk/`
- 分类评估目录：
  - `final_evaluation_results/dcgfs_pair_topk_base_w0125/`
  - `final_evaluation_results/dcgfs_pair_topk_base_w0175/`
  - `final_evaluation_results/dcgfs_pair_topk_base_w015/`
- 可解释性评估目录：
  - `explainability_results/dcgfs_pair_topk_base_w0125/`
  - `explainability_results/dcgfs_pair_topk_base_w0175/`
  - `explainability_results/dcgfs_pair_topk_base_w015/`

## 代码更新

本轮为 `run_dcgfs_pipeline.sh` 新增了 `pair_topk_base_fine_sweep` 模式。

该模式只补跑 `0.125` 和 `0.175`，不会重复训练已完成的 `0.15`。

## 分类性能对比

| 权重 | overall_a_acc | macro_f1 | balanced_acc | worst_class_acc | selected_target_acc | selected_base_acc | target_to_base_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 66.02% | 66.04% | 66.09% | 10.00% | 42.86% | 76.07% | 17.86% |
| 0.150 | 65.74% | 65.72% | 65.80% | 6.67% | 44.29% | 77.78% | 17.86% |
| 0.175 | 65.64% | 65.64% | 65.68% | 6.67% | 43.57% | 74.36% | 18.57% |

评价：

- `0.125` 的整体指标和最差类别准确率较好，但目标类准确率低于 `0.15`，基座类准确率也低于 `0.15`。
- `0.175` 的目标类准确率高于 `0.125`，但整体指标、基座类准确率和 target-to-base rate 都弱于 `0.15`。
- `0.15` 仍然是分类主指标上最均衡的权重：目标类准确率最高，基座类准确率最高，target-to-base rate 与 `0.125` 持平。

## 可解释性对比

### 概念准确率

| 权重 | overall_c_acc | target_c_acc | target_disc_c_acc |
|---:|---:|---:|---:|
| 0.125 | 89.46% | 91.07% | 32.74% |
| 0.150 | 89.46% | 91.18% | 32.19% |
| 0.175 | 89.42% | 90.92% | 31.10% |

评价：`0.125` 的判别概念准确率略高于 `0.15`，但分类目标类修复和基座类保持都不如 `0.15`。`0.175` 的判别概念准确率下降较明显。

### 概念干预

| 权重 | original_acc | intervention_acc | intervention_gain |
|---:|---:|---:|---:|
| 0.125 | 42.86% | 80.00% | 37.14% |
| 0.150 | 44.29% | 79.29% | 35.00% |
| 0.175 | 43.57% | 81.43% | 37.86% |

评价：`0.175` 的干预后准确率最高，说明它在概念干预上有一定优势。但本项目当前主目标仍是弱势目标类修复、基座类保持和 target-to-base confusion 的平衡，因此不能仅凭干预指标替代 `0.15`。

## 结论

局部细扫后，`base_preservation_weight=0.15` 仍然是当前最合适的主方法权重。

理由：

- `selected_target_acc=44.29%`，高于 `0.125` 和 `0.175`。
- `selected_base_acc=77.78%`，高于 `0.125` 和 `0.175`。
- `target_to_base_rate=17.86%`，与 `0.125` 持平，优于 `0.175`。
- 虽然整体准确率略低于 `0.125`，但差距只有 `0.28%`，不足以抵消目标类和基座类指标的优势。

因此，建议后续将 `0.15` 固定为完整 D-CGFS 的默认主结果。

## 下一步建议

不建议继续在 base preservation 权重上投入太多时间。当前更值得做的是：

1. 把 `0.15` 写入默认配置或默认运行脚本，作为完整 D-CGFS 主方法。
2. 把 `pair_topk` 作为无 base preservation 消融。
3. 单独分析 `Common_Tern -> Artic_Tern` 这一组仍未明显解决的 target-base pair。
