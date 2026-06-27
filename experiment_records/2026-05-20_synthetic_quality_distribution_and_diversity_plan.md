# 2026-05-20 Synthetic Quality Distribution and Diversity Plan

## 诊断命令

```bash
.venv/bin/python analyze_synthetic_quality_distribution.py
```

## 输出文件

- `diagnostic_results/synthetic_quality_distribution/pair_quality_summary.csv`
- `diagnostic_results/synthetic_quality_distribution/target_score_vs_main_diff.csv`
- `diagnostic_results/synthetic_quality_distribution/synthetic_quality_report.md`

## 关键发现

当前 D-CGFS 的瓶颈不只是 target probability 偏低，还包括 top-k 合成样本有效多样性不足。

主合成策略 `main_pair_topk_w005` 的 pair 内重复情况：

| Pair | Target | Kept | Unique Source Pairs | Duplicate Rate | Target Prob Median | Target Prob Max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 65 | 500 | 75 | 85.00% | 7.659e-12 | 9.978e-05 |
| 1 | 59 | 500 | 15 | 97.00% | 2.853e-08 | 1.814e-02 |
| 2 | 144 | 500 | 434 | 13.20% | 9.308e-18 | 5.044e-08 |
| 3 | 74 | 500 | 434 | 13.20% | 6.881e-08 | 1.111e-05 |
| 4 | 176 | 500 | 15 | 97.00% | 5.550e-11 | 1.875e-06 |

`target_score_w015` 虽然提高了部分 pair 的 target probability 中位数，但没有明显增加高置信样本数量：

| Pair | Target | Target Prob Median Diff | >=1e-4 Diff | >=1e-3 Diff | Concept Delta Mean Diff | Duplicate Rate Diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 65 | 2.310e-07 | +0 | +0 | +0.0017 | 0.00% |
| 1 | 59 | 1.134e-05 | +0 | +0 | +0.0085 | -0.20% |
| 2 | 144 | 7.010e-16 | +0 | +0 | +0.0004 | -0.40% |
| 3 | 74 | 5.111e-09 | +0 | +0 | +0.0003 | +0.20% |
| 4 | 176 | 1.912e-09 | +0 | +0 | -0.0094 | 0.00% |

## 解释

pair 1 和 pair 4 的 top-500 实际只来自 15 个不同的 target/base 源图组合，重复率达到 97%。这意味着训练阶段看到的 500 个样本并不是真正的 500 个有效变化样本，而是大量重复的高分组合。

这可以解释为什么：

1. 提高 target-score 权重只能带来很小的 overall/macro 改善。
2. 单纯提高 synthetic loss 会放大重复样本和噪声，导致 target/base 指标下降。
3. D-CGFS 的 base preservation 表现不错，但 target accuracy 追不上 strong baseline。

## 已完成代码修改

### `step2_synthesize_data.py`

新增参数：

```bash
--pair-topk-max-per-source-pair
```

含义：

- 默认值 `0`：不限制重复，完全保持当前主实验配置。
- 大于 `0`：同一个 target/base 源图组合最多保留 N 次。

该改动只影响显式传参的新实验，不会改变当前主方法 `D-CGFS pair_topk_base_w015`。

### `run_problem6_experiments.py`

新增实验组：

```bash
diversity_push
```

包含两个候选：

1. `dcgfs_pair_topk_diverse20`
   - 使用原 pair-score 权重。
   - 每个 target/base 源图组合最多保留 20 次。

2. `dcgfs_target_score_w015_diverse20`
   - 使用 target-score 权重 0.15。
   - 每个 target/base 源图组合最多保留 20 次。

## 下一步运行命令

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group diversity_push
```

运行后重点观察：

1. 每个 pair 最终保留样本数是否从 500 降低，尤其 pair 1 和 pair 4。
2. target accuracy 是否超过当前主方法的 `44.29`。
3. base accuracy 是否仍能保持接近或高于 `77.78`。
4. overall/macro 是否能进一步接近 strong baseline。

## 当前判断

如果 `diverse20` 有效，说明 D-CGFS 的问题主要是合成样本多样性不足，而不是方法框架本身不行。

如果 `diverse20` 无效，则下一步需要更强的策略，例如：

1. per-pair adaptive top-k，而不是固定每个 pair 保留 500。
2. target-confidence rerank 与 concept-delta rerank 分阶段选择。
3. 对 pair 0、1、4 使用不同的合成/过滤阈值。
