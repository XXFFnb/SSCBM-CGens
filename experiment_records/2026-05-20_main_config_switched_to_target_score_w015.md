# 2026-05-20 Main Config Switched to target_score_w015

## 决策

从本记录开始，D-CGFS 的正式主配置从旧的：

```text
pair_topk_filter + pair_score_target_weight=0.05 + base_preservation_weight=0.15
```

切换为：

```text
pair_topk_filter + pair_score_target_weight=0.15 + pair_score_base_weight=0.05 + base_preservation_weight=0.15
```

主方法名称统一写为：

```text
D-CGFS target_score_w015
```

## 理由

`target_score_w015` 在不降低 Target Acc 的前提下，相比旧主配置提升了更多论文主指标：

| Metric | Previous Main | target_score_w015 |
| --- | ---: | ---: |
| Overall Acc | 65.74 | 66.00 |
| Macro-F1 | 65.72 | 65.96 |
| Balanced Acc | 65.80 | 66.04 |
| Worst Class Acc | 6.67 | 10.00 |
| Target Acc | 44.29 | 44.29 |
| Base Acc | 77.78 | 78.63 |
| Target-to-Base Rate | 17.86 | 19.29 |
| Intervention Acc | 79.29 | 82.14 |
| Intervention Gain | 35.00 | 37.86 |

代价是 Target-to-Base Rate 略高 `1.43` 个百分点，Target Disc Concept Acc 也略低：

| Metric | Previous Main | target_score_w015 |
| --- | ---: | ---: |
| Target Disc Concept Acc | 32.19 | 31.78 |
| Target-to-Base Rate | 17.86 | 19.29 |

综合判断：如果论文叙事强调“弱势类修复 + 基座类保护 + 可解释干预”，`target_score_w015` 更适合作为正式主配置。

## 已修改代码

- `dcgfs_config.py`
  - 新增 `PAIR_SCORE_TARGET_WEIGHT = 0.15`
  - 新增 `PAIR_SCORE_BASE_WEIGHT = 0.05`

- `run_problem6_experiments.py`
  - `main` 组改为 `dcgfs_target_score_w015`
  - 主合成数据目录改为 `generated_data/problem6_target_score_w015`
  - 主 checkpoint 改为 `checkpoints/problem6_dcgfs_target_score_w015.pt`
  - 主分类结果目录改为 `final_evaluation_results/problem6_dcgfs_target_score_w015`
  - 主可解释性目录改为 `explainability_results/problem6_dcgfs_target_score_w015`

- `run_dcgfs_pipeline.sh`
  - `main` / `resume` / `pair_topk_base` 默认改为 target-score 主配置
  - 主合成命令显式加入：

```bash
--pair-score-target-weight 0.15 --pair-score-base-weight 0.05
```

- `paper_tables/current_main_results_tables.md`
  - 当前主方法表述更新为 `target_score_w015`

- `paper_tables/main_results.csv`
  - 新增当前主方法行
  - 保留旧主配置作为 previous main

## 后续运行

后续默认主实验使用：

```bash
bash run_dcgfs_pipeline.sh main
```

或只跑 problem6 主实验：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group main
```

如果已有 `generated_data/problem6_target_score_w015`、`checkpoints/problem6_dcgfs_target_score_w015.pt`、`final_evaluation_results/problem6_dcgfs_target_score_w015` 和 `explainability_results/problem6_dcgfs_target_score_w015`，则不需要立刻重跑；这些结果已经是当前主配置结果。
