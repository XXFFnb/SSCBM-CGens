# paper_tables

存放论文和报告使用的结果表、汇总 CSV 和 Markdown 草稿。该目录不直接参与训练，但论文初稿应优先参考这里的最终汇总。

## 论文初稿入口

```text
paper_draft_materials.md
```

该文件汇总了研究动机、方法、实验设置、主结果、补充实验、消融、可解释性分析和写作边界。

## 当前主结果表

```text
multi_dataset_results.csv
final_report_summary.md
final_main_results.csv
metric_glossary_zh.csv
```

## 补充实验表

```text
cub_seed_sweep_detail.csv
cub_seed_sweep_summary.csv
7pt_seed_sweep_detail.csv
7pt_seed_sweep_summary.csv
7pt_labeled_ratio_sweep.csv
main_component_ablation.csv
cub_main_explainability.csv
final_strong_baseline_context.csv
```

## CUB 全局分析

```text
cub_global_analysis.md
cub_global_metrics.csv
cub_global_metric_deltas.csv
cub_class_group_summary.csv
cub_classwise_global_detail.csv
cub_target_base_detail.csv
cub_top_improved_classes.csv
cub_top_dropped_classes.csv
cub_metric_glossary.csv
```

`cub_global_analysis.md` 是 CUB 全局分析入口，包含中文指标解释、整体指标、全类别分组、target-base pair 细节，以及提升/下降最多的类别。

## 历史诊断和探索表

这些文件保留用于解释方法选择，但不作为论文主方法结果：

```text
final_archived_diagnostics.csv
problem6_*_comparison.csv
second_dataset_plan.md
second_dataset_results.csv
explainability_results.csv
problem6_target_score_explainability_comparison.csv
```

写论文时，以 `paper_draft_materials.md` 中明确引用的表为准。
