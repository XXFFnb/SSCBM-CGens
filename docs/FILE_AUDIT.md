# 文件审计与整理建议

本文档回答三个问题：

1. 哪些文件/目录目前不需要作为主线使用？
2. 哪些内容还没有很好归纳？
3. 哪些内容适合后续建立文件夹归档？

本次整理原则：除非确认是缓存或 smoke/debug 产物，否则不删除。

## 一、当前不属于论文主线的内容

这些内容不是 D-CGFS 当前主方法的一部分，但可能仍有诊断价值，因此暂不删除。

### 历史探索结果

路径示例：

```text
final_evaluation_results/problem6_dcgfs_feature_refine_pred_disc/
final_evaluation_results/problem6_dcgfs_feature_refine_window/
final_evaluation_results/problem6_dcgfs_retrieval_residual/
final_evaluation_results/problem6_dcgfs_model_aware_d_w050/
final_evaluation_results/problem6_dcgfs_hybrid_pair0_refine_base_train/
final_evaluation_results/problem6_dcgfs_hybrid_pair0_refine_pred_disc_train/
generated_data/problem6_feature_refine_pred_disc/
generated_data/problem6_feature_refine_window/
generated_data/problem6_retrieval_residual/
generated_data/problem6_model_aware_d_w050/
generated_data/problem6_hybrid_pair0_refine/
```

处理建议：

```text
保留，但只作为历史诊断，不写入主方法。
```

论文材料中已经明确：feature refinement、retrieval residual、model-aware D、hybrid pair refinement 不作为主方案。

### smoke/debug 产物

路径示例：

```text
generated_data/_smoke_feature_refine/
generated_data/_smoke_feature_refine_freeze/
generated_data/_smoke_feature_refine_strong/
generated_data/_smoke_hybrid_pair0_refine/
generated_data/_smoke_retrieval_residual/
generated_data/debug_synthesis_core_smoke/
generated_data/awa2_smoke_synthesis/
final_evaluation_results/awa2_smoke_step4/
checkpoints/awa2_smoke/
checkpoints/awa2_smoke_step3.pt
```

处理建议：

```text
这些通常可以删除或归档，但建议等论文初稿完成后再处理。
```

### Python 缓存

路径示例：

```text
__pycache__/
configs/__pycache__/
data/__pycache__/
models/__pycache__/
train/__pycache__/
interventions/__pycache__/
```

处理建议：

```text
可删除；不会影响代码和实验结果。
```

本次未删除，因为用户要求除非确认不需要，否则不要删除。缓存文件可以在后续单独清理。

## 二、还没有很好归纳的内容

### 1. 大量历史结果目录仍在原位置

现状：

```text
final_evaluation_results/
generated_data/
checkpoints/
```

这些目录里同时包含：

1. 当前主方法结果。
2. 多 seed 结果。
3. 标注比例敏感性结果。
4. 强 baseline 结果。
5. 消融结果。
6. 历史探索分支。
7. smoke/debug 产物。

当前整理方式：

```text
不移动目录，通过 README 和 docs/PROJECT_STRUCTURE.md 建立索引。
```

后续如果要物理归档，可建立：

```text
archive/historical_results/
archive/smoke_tests/
archive/old_logs/
```

但这会改变路径，必须同步更新文档和可能的脚本。

### 2. 根目录脚本数量较多

根目录脚本中主流程、分析脚本和兼容脚本混在一起。当前没有移动，原因是这些脚本多以项目根目录为运行位置，移动后可能破坏相对路径。

当前整理方式：

```text
在 docs/PROJECT_STRUCTURE.md 中按用途归类。
```

后续若要物理整理，可考虑：

```text
scripts/pipeline/
scripts/analysis/
scripts/maintenance/
```

但需要逐个检查 import 和相对路径。

### 3. paper_tables 中同时有最终表和历史对比表

当前重要主表：

```text
paper_tables/paper_draft_materials.md
paper_tables/multi_dataset_results.csv
paper_tables/final_report_summary.md
paper_tables/cub_seed_sweep_summary.csv
paper_tables/7pt_seed_sweep_summary.csv
paper_tables/7pt_labeled_ratio_sweep.csv
paper_tables/main_component_ablation.csv
paper_tables/cub_main_explainability.csv
```

历史/诊断表：

```text
paper_tables/problem6_*_comparison.csv
paper_tables/final_archived_diagnostics.csv
paper_tables/second_dataset_plan.md
paper_tables/second_dataset_results.csv
paper_tables/explainability_results.csv
```

处理建议：

```text
保留。写论文时以 paper_draft_materials.md 中列出的主表为准。
```

## 三、建议建立的文件夹

本次已经新增：

```text
docs/
```

用途：

```text
存放项目结构说明、文件审计、清理建议和后续维护说明。
```

当前文件：

```text
docs/PROJECT_STRUCTURE.md
docs/FILE_AUDIT.md
```

后续可选新增：

```text
archive/
archive/smoke_tests/
archive/historical_results/
archive/old_logs/
scripts/
scripts/analysis/
scripts/pipeline/
```

但这些都涉及移动文件，不建议在论文初稿完成前做大范围重排。

## 四、可以立即删除但本次未删除的内容

如果后续确认要清理磁盘，可优先删除：

```text
__pycache__/
configs/__pycache__/
data/__pycache__/
models/__pycache__/
train/__pycache__/
interventions/__pycache__/
```

这些是 Python 缓存。

如果论文初稿和结果核验都完成，可以再考虑删除或离线备份：

```text
generated_data/_smoke_*
generated_data/debug_synthesis_core_smoke
generated_data/awa2_smoke_synthesis
final_evaluation_results/awa2_smoke_step4
checkpoints/awa2_smoke
checkpoints/awa2_smoke_step3.pt
```

## 五、不要删除的内容

至少在论文完成前，不要删除：

```text
paper_tables/
experiment_records/
final_evaluation_results/
explainability_results/
diagnostic_results/
data/D-CGFS_Auxiliary*
checkpoints/*target_score_w015*
checkpoints/CUB-200-2011_22-56/
checkpoints/AwA2_16-15/
checkpoints/PBC_17-01/
checkpoints/7pt_17-44/
generated_data/problem6_target_score_w015/
generated_data/awa2_target_score_w015/
generated_data/pbc_target_score_w015/
generated_data/7pt_target_score_w015/
```

这些文件或目录与主结果、复现实验或论文表格直接相关。
