# D-CGFS base preservation 权重细粒度局部验证记录

## 实验目的

在前一轮局部验证中，`base_preservation_weight=0.15` 在目标类准确率、基座类保持、目标类错分基座比例之间表现较均衡，但并不是所有指标都最优。因此本次围绕 `0.15` 做更细粒度验证，补充运行：

- `0.135`
- `0.145`
- `0.155`
- `0.165`

本次实验复用 `generated_data/dcgfs_pair_topk/` 中同一组合成样本，只改变 `step3_balance_training.py` 中的 `base_preservation_weight`，用于观察 base preservation 强度的局部趋势。

## 运行命令

```bash
bash run_dcgfs_pipeline.sh pair_topk_base_ultra_fine_sweep
```

运行过程中确认训练设备为 `cuda`。

## 输出位置

分类评估结果：

- `final_evaluation_results/dcgfs_pair_topk_base_w0135/`
- `final_evaluation_results/dcgfs_pair_topk_base_w0145/`
- `final_evaluation_results/dcgfs_pair_topk_base_w015/`
- `final_evaluation_results/dcgfs_pair_topk_base_w0155/`
- `final_evaluation_results/dcgfs_pair_topk_base_w0165/`

可解释性评估结果：

- `explainability_results/dcgfs_pair_topk_base_w0135/`
- `explainability_results/dcgfs_pair_topk_base_w0145/`
- `explainability_results/dcgfs_pair_topk_base_w015/`
- `explainability_results/dcgfs_pair_topk_base_w0155/`
- `explainability_results/dcgfs_pair_topk_base_w0165/`

## 分类指标对比

| base preservation weight | overall acc | macro F1 | balanced acc | worst class acc | target acc | base acc | target-to-base rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.135 | 66.12% | 66.10% | 66.17% | 6.67% | 41.43% | 77.78% | 22.14% |
| 0.145 | 66.03% | 66.04% | 66.12% | 10.00% | 40.71% | 76.92% | 20.71% |
| 0.150 | 65.74% | 65.72% | 65.80% | 6.67% | 44.29% | 77.78% | 17.86% |
| 0.155 | 66.00% | 65.99% | 66.07% | 10.00% | 42.86% | 77.78% | 18.57% |
| 0.165 | 65.77% | 65.77% | 65.87% | 10.00% | 40.71% | 77.78% | 22.86% |

## 可解释性指标对比

| base preservation weight | target concept acc | target discriminative concept acc | intervention acc |
| --- | ---: | ---: | ---: |
| 0.135 | 90.98% | 32.60% | 79.29% |
| 0.145 | 90.91% | 32.60% | 80.00% |
| 0.150 | 91.18% | 32.19% | 79.29% |
| 0.155 | 91.01% | 33.01% | 80.71% |
| 0.165 | 90.97% | 31.23% | 80.71% |

## 结果评价

`0.135` 的 overall acc、macro F1、balanced acc 最高，但它的目标类准确率只有 41.43%，目标类错分为基座类比例达到 22.14%。这说明较低的 base preservation 权重更有利于整体平均指标，但没有很好解决 D-CGFS 最核心的弱势目标类纠偏问题。

`0.145` 的 worst class acc 达到 10.00%，target-to-base rate 也低于 `0.135`，但 target acc 只有 40.71%，base acc 也下降到 76.92%。因此它不适合作为主结果权重。

`0.150` 仍然是当前最适合作为主结果的权重。它的 target acc 最高，为 44.29%；base acc 与多数局部点持平，为 77.78%；target-to-base rate 最低，为 17.86%。虽然 overall acc、macro F1、balanced acc 不是最高，但它最符合 D-CGFS 的核心目标：提升弱势目标类，同时尽量压低目标类向基座类的错误迁移。

`0.155` 是一个值得保留的对照点。它的 worst class acc 为 10.00%，target discriminative concept acc 最高，为 33.01%，intervention acc 也达到 80.71%。但它的 target acc 为 42.86%，低于 `0.150`，target-to-base rate 也略高。因此它更适合作为“可解释性指标更优但主分类纠偏略弱”的补充实验，而不是替代 `0.150`。

`0.165` 的 intervention acc 与 `0.155` 持平，但 target acc 降到 40.71%，target-to-base rate 升到 22.86%，说明继续加大 base preservation 后主纠偏目标开始明显变差。

## 当前结论

本轮细粒度局部验证后，建议继续把 `base_preservation_weight=0.15` 作为 D-CGFS 主实验配置。

论文或报告中可以这样解释：`0.15` 不是单项整体指标最高的点，但它在弱势目标类提升、基座类保持、目标类错分基座抑制三者之间取得了最符合方法目标的平衡。`0.155` 可以作为补充消融，说明稍微增强 base preservation 会提升部分可解释性指标，但会牺牲目标类纠偏强度。

