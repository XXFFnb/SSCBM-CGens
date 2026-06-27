# SSCBM-D-CGFS

本项目基于 SSCBM（Semi-Supervised Concept Bottleneck Models）实现并评估 D-CGFS：

```text
Discriminative Concept-Guided Feature Synthesis
```

中文名称：

```text
判别概念引导的特征合成方法
```

当前正式主方法：

```text
D-CGFS target_score_w015 + base preservation
```

## 当前主配置

```text
baseline = SSCBM
ablation_mode = pair_topk_filter
pair_topk = 500
pair_score_target_weight = 0.15
pair_score_base_weight = 0.05
base_preservation_weight = 0.15
fusion_mode = old_spatial
synthesis_task_concept_source = base
```

feature refinement、retrieval residual、model-aware D、hybrid pair refinement 等分支只作为历史诊断记录，不作为论文主方法。

## 必读文档

| 文件 | 用途 |
| --- | --- |
| `paper_tables/paper_draft_materials.md` | 论文初稿生成材料总表，包含动机、方法、实验、结论边界。 |
| `docs/PROJECT_STRUCTURE.md` | 项目结构说明，说明每个目录和关键文件的用途。 |
| `docs/FILE_AUDIT.md` | 文件审计与清理建议，说明哪些可清理、哪些应保留。 |
| `docs/GITHUB_REPOSITORY_GUIDE.md` | GitHub 仓库维护说明，说明应提交和忽略哪些内容。 |
| `baseline/current_idea_D-CGFS.md` | D-CGFS 主方案 idea 文档。 |
| `paper_tables/final_report_summary.md` | 最终报告汇总。 |

## 当前实验状态

四个数据集主结果已经完成：

| 数据集 | Baseline 整体任务准确率 | D-CGFS 整体任务准确率 | 说明 |
| --- | ---: | ---: | --- |
| CUB-200-2011 | 57.80% | 65.65% | 主实验数据集，提升最明确。 |
| AwA2 | 89.39% | 90.01% | 整体和基座类稳定性略有提升，弱势目标类不变。 |
| PBC / WBCatt | 99.65% | 99.65% | baseline 近饱和，主要证明不破坏性能。 |
| 7-point | 61.27% | 66.84% | 困难低基线数据集，任务和概念指标均有提升。 |

补充实验已经完成：

| 实验 | 结果文件 |
| --- | --- |
| CUB 多 seed | `paper_tables/cub_seed_sweep_summary.csv` |
| 7-point 多 seed | `paper_tables/7pt_seed_sweep_summary.csv` |
| 7-point 标注比例敏感性 | `paper_tables/7pt_labeled_ratio_sweep.csv` |
| CUB 主组件消融 | `paper_tables/main_component_ablation.csv` |
| CUB 可解释性分析 | `paper_tables/cub_main_explainability.csv` |
| 强 baseline 对比 | `paper_tables/final_strong_baseline_context.csv` |

## 常用命令

只生成实验协议：

```bash
.venv/bin/python run_problem6_experiments.py --only-group main
```

CUB 主实验：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group main
```

AwA2 主实验：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group awa2_main
```

PBC 主实验：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group pbc_main
```

7-point 主实验：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group 7pt_main
```

生成论文/报告结果汇总：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group report
```

补充实验：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group cub_seed_sweep_extra
.venv/bin/python run_problem6_experiments.py --run --only-group 7pt_seed_sweep
.venv/bin/python run_problem6_experiments.py --run --only-group 7pt_labeled_ratio_sweep
```

## 主要脚本

| 文件 | 作用 |
| --- | --- |
| `find_weak_classes.py` | 自动发现弱势目标类、基座类和判别概念集合。 |
| `step1_generate_mapping.py` | 根据判别概念生成目标/基座区域映射。 |
| `step2_synthesize_data.py` | 生成 D-CGFS 合成特征并进行质量筛选。 |
| `check_synthetic_data.py` | 检查合成特征标签和 baseline 预测。 |
| `step3_balance_training.py` | 使用原始数据和合成特征做平衡训练。 |
| `step4_final_evaluation.py` | 输出分类、弱势类、基座类和错分指标。 |
| `step5_explainability_evaluation.py` | 输出 CUB 可解释性指标。 |
| `run_problem6_experiments.py` | 实验调度入口。 |
| `summarize_problem6_results.py` | 生成论文/报告结果表。 |

## 目录说明

| 目录 | 用途 |
| --- | --- |
| `baseline/` | SSCBM 原论文和 D-CGFS idea 文档。 |
| `configs/` | 数据集训练配置。 |
| `data/` | 数据 loader、本地数据集和 D-CGFS 辅助文件。 |
| `models/` | SSCBM 与 D-CGFS 模型代码。 |
| `train/` | 训练逻辑。 |
| `eval/` | 评估逻辑。 |
| `checkpoints/` | baseline 与 D-CGFS checkpoint，不应提交。 |
| `generated_data/` | D-CGFS 合成特征，不应提交。 |
| `final_evaluation_results/` | step4 分类评估结果。 |
| `explainability_results/` | step5 可解释性评估结果。 |
| `paper_tables/` | 论文表格和论文初稿输入材料。 |
| `experiment_records/` | 历史实验记录。 |
| `docs/` | 项目结构和文件清理说明。 |

更详细的文件分类见：

```text
docs/PROJECT_STRUCTURE.md
docs/FILE_AUDIT.md
docs/GITHUB_REPOSITORY_GUIDE.md
```

## 数据集位置

当前代码默认数据集位于：

```text
data/CUB_200_2011
data/AwA2/Animals_with_Attributes2
data/PBC
data/7-point/release_v0
```

## 注意事项

1. 不要提交 `.venv/`、`data/`、`checkpoints/`、`generated_data/` 等大目录。
2. 不要把历史探索分支写成主方法。
3. 写论文时以 `paper_tables/paper_draft_materials.md` 为主。
4. `problem6_experiment_protocol/problem6_commands.csv` 会被调度脚本按当前 group 重写。
5. 清理文件前先看 `docs/FILE_AUDIT.md`。

## 协议

本项目基于 MIT 协议，见 `LICENSE`。
