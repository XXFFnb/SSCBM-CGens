# GitHub 仓库维护说明

本文档说明本项目上传到 GitHub 时应保留哪些文件、忽略哪些文件，以及仓库首页应该如何被读者理解。

## 推荐提交到 GitHub 的内容

### 代码

```text
main.py
dataset_specs.py
dcgfs_config.py
find_weak_classes.py
step1_generate_mapping.py
step2_synthesize_data.py
check_synthetic_data.py
step3_balance_training.py
step4_final_evaluation.py
step5_explainability_evaluation.py
run_problem6_experiments.py
summarize_problem6_results.py
models/
train/
eval/
interventions/
visualization/
configs/
```

### 文档

```text
README.md
LICENSE
baseline/README.md
baseline/current_idea_D-CGFS.md
docs/
paper_tables/paper_draft_materials.md
paper_tables/README.md
experiment_records/
```

### 论文汇总结果

建议提交 `paper_tables/` 中的论文级汇总表，例如：

```text
paper_tables/multi_dataset_results.csv
paper_tables/final_report_summary.md
paper_tables/cub_seed_sweep_summary.csv
paper_tables/7pt_seed_sweep_summary.csv
paper_tables/7pt_labeled_ratio_sweep.csv
paper_tables/main_component_ablation.csv
paper_tables/cub_main_explainability.csv
```

这些文件体积小，且足以支持论文初稿和结果说明。

## 不建议提交到 GitHub 的内容

这些内容通常很大，或属于本地可再生成产物：

```text
.venv/
data/
checkpoints/
generated_data/
final_evaluation_results/
explainability_results/
diagnostic_results/
run_logs/
problem6_experiment_protocol/problem6_commands.csv
```

对应目录中的 `README.md` 可以保留，用于说明本地运行时这些目录放什么。

## 当前仓库首页应该表达什么

GitHub 访问者打开仓库后，应先看到：

1. 本项目提出 D-CGFS：Discriminative Concept-Guided Feature Synthesis。
2. 当前主方法是 `target_score_w015 + base preservation`。
3. 四个数据集主实验已完成：CUB、AwA2、PBC/WBCatt、7-point。
4. CUB 和 7-point 上主要指标提升较明显，AwA2/PBC 用于验证跨数据集稳定性和边界。
5. 历史探索分支不是主方法。
6. 论文初稿材料入口是 `paper_tables/paper_draft_materials.md`。

## 推荐提交前检查

提交前运行：

```bash
git status --short
```

确认没有误加入以下大目录：

```text
data/
checkpoints/
generated_data/
final_evaluation_results/
explainability_results/
diagnostic_results/
.venv/
```

如果需要确认忽略规则是否生效：

```bash
git check-ignore -v generated_data/problem6_target_score_w015/feat_0.pt
git check-ignore -v checkpoints/problem6_dcgfs_target_score_w015.pt
git check-ignore -v data/CUB_200_2011
```

## 推荐 commit 组织

如果要把当前项目整理提交到 GitHub，建议分成几类 commit：

1. `docs: update project overview and repository structure`
2. `chore: update gitignore for local artifacts`
3. `feat: add D-CGFS experiment pipeline`
4. `docs: add paper tables and draft materials`

如果只是备份当前工作，可以合并成一个 commit，但不建议把数据和 checkpoint 一起提交。
