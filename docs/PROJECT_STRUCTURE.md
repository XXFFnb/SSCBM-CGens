# 项目结构说明

本文档用于快速判断项目中每个目录的用途，以及哪些文件属于当前论文主线、历史实验或可清理产物。

## 当前主线

当前论文主方法是：

```text
D-CGFS target_score_w015 + base preservation
```

主方法配置：

```text
pair_topk = 500
pair_score_target_weight = 0.15
pair_score_base_weight = 0.05
base_preservation_weight = 0.15
fusion_mode = old_spatial
synthesis_task_concept_source = base
```

论文材料入口：

```text
paper_tables/paper_draft_materials.md
```

主结果入口：

```text
paper_tables/multi_dataset_results.csv
paper_tables/final_report_summary.md
paper_tables/cub_seed_sweep_summary.csv
paper_tables/7pt_seed_sweep_summary.csv
paper_tables/7pt_labeled_ratio_sweep.csv
paper_tables/main_component_ablation.csv
paper_tables/cub_main_explainability.csv
```

## 根目录脚本分类

### 主流程脚本

这些脚本是 D-CGFS 当前主线的一部分，应保留在根目录，方便按步骤运行。

| 文件 | 用途 |
| --- | --- |
| `find_weak_classes.py` | 自动发现 weak target class、base class 和判别概念。 |
| `step1_generate_mapping.py` | 生成 target/base 的判别概念区域映射和 mask。 |
| `step2_synthesize_data.py` | 生成 D-CGFS 合成特征，并执行 pair-topk 质量筛选。 |
| `check_synthetic_data.py` | 快速检查合成特征标签和 baseline 预测。 |
| `step3_balance_training.py` | 使用原始数据和合成特征训练 D-CGFS 模型。 |
| `step4_final_evaluation.py` | 输出整体、弱势类、基座类和 target-to-base 指标。 |
| `step5_explainability_evaluation.py` | CUB 可解释性评估。 |
| `run_problem6_experiments.py` | 实验调度入口，包含主实验、补充实验、报告生成等 group。 |
| `summarize_problem6_results.py` | 汇总论文/报告表格。 |

### 配置与公共模块

| 文件 | 用途 |
| --- | --- |
| `dataset_specs.py` | 统一管理 CUB、AwA2、PBC、7pt 的数据集配置和 loader。 |
| `dcgfs_config.py` | D-CGFS 方法名、路径和权重等配置。 |
| `main.py` | SSCBM baseline 训练入口。 |
| `utils.py` | 公共工具函数。 |
| `configs/` | baseline 训练配置。 |
| `models/` | SSCBM 和 D-CGFS 模块。 |
| `data/` | 数据集 loader 和本地数据。 |
| `train/` | 训练逻辑。 |
| `eval/` | 评估相关代码。 |

### 分析和诊断脚本

这些脚本不是主流程必需，但用于生成论文分析或诊断结果，建议保留。

| 文件 | 用途 |
| --- | --- |
| `analyze_cub_global_results.py` | CUB 全局类别分析。 |
| `analyze_problem6_pair_failures.py` | target-base pair 失败诊断。 |
| `analyze_synthetic_quality_distribution.py` | 合成样本质量分布诊断。 |
| `check_second_dataset_readiness.py` | 第二数据集准备检查。 |
| `prepare_imbalance_splits.py` | 不平衡划分准备。 |

### 兼容或辅助脚本

| 文件 | 用途 |
| --- | --- |
| `run_dcgfs_pipeline.sh` | 早期 shell pipeline 入口，当前推荐优先使用 `run_problem6_experiments.py`。 |
| `experiments.sh` | 早期实验命令记录，可保留为历史参考。 |
| `prs_hook.py` | 项目辅助脚本；若后续确认没有外部依赖可再归档。 |

## 目录分类

### 应保留的代码和配置目录

| 目录 | 用途 |
| --- | --- |
| `configs/` | 数据集训练配置。 |
| `data/` | 数据 loader、本地数据集、D-CGFS 辅助 CSV/mask。 |
| `models/` | SSCBM 与 D-CGFS 模型代码。 |
| `train/` | 训练逻辑。 |
| `eval/` | 评估逻辑。 |
| `interventions/` | 概念干预相关代码。 |
| `visualization/` | 可视化辅助代码。 |
| `cem/` | 原 CEM 相关依赖代码，当前项目仍可能间接依赖，暂不删除。 |

### 应保留的论文和实验材料目录

| 目录 | 用途 |
| --- | --- |
| `baseline/` | SSCBM 原论文、D-CGFS idea 文档和 baseline 材料。 |
| `paper_tables/` | 论文表格、最终结果、论文初稿输入材料。 |
| `experiment_records/` | 按日期保存的实验过程和历史探索记录。 |
| `diagnostic_results/` | 诊断分析输出。 |
| `problem6_experiment_protocol/` | 自动生成的实验命令协议。 |
| `run_logs/` | 旧实验日志。 |

### 大型产物目录

这些目录很大，不应提交到远程仓库。当前不要直接删除，因为论文结果、复现实验和检查可能仍会引用。

| 目录 | 用途 |
| --- | --- |
| `checkpoints/` | baseline 和 D-CGFS 模型 checkpoint。 |
| `generated_data/` | D-CGFS 合成特征和可视化样本。 |
| `final_evaluation_results/` | step4 分类评估输出。 |
| `explainability_results/` | step5 可解释性评估输出。 |

## 当前最重要的结果目录

### 四数据集主实验

```text
final_evaluation_results/problem6_dcgfs_target_score_w015/
final_evaluation_results/awa2_dcgfs_target_score_w015/
final_evaluation_results/pbc_dcgfs_target_score_w015/
final_evaluation_results/7pt_dcgfs_target_score_w015/
```

### 多 seed 和标注比例补充实验

```text
final_evaluation_results/problem6_seed0_dcgfs_target_score_w015/
final_evaluation_results/problem6_seed1_dcgfs_target_score_w015/
final_evaluation_results/7pt_seed0_dcgfs_target_score_w015/
final_evaluation_results/7pt_seed1_dcgfs_target_score_w015/
final_evaluation_results/7pt_seed2_dcgfs_target_score_w015/
final_evaluation_results/7pt_r005_dcgfs_target_score_w015/
final_evaluation_results/7pt_r020_dcgfs_target_score_w015/
```

### CUB 消融和强 baseline

```text
final_evaluation_results/problem6_dcgfs_target_score_w015/
final_evaluation_results/problem6_dcgfs_pair_topk_base_w015/
final_evaluation_results/problem6_reweighting/
final_evaluation_results/problem6_class_balanced_loss/
final_evaluation_results/problem6_oversampling/
final_evaluation_results/problem6_feature_mixup/
final_evaluation_results/problem6_sscbm_finetune/
```

## 不建议删除的内容

1. `paper_tables/`：论文初稿和最终结果依赖这里。
2. `final_evaluation_results/`：很多结果表可以由这里重新汇总。
3. `experiment_records/`：解释方法选择和历史探索边界。
4. `checkpoints/` 中主实验、多 seed、标注比例相关 checkpoint。
5. `generated_data/` 中主实验、多 seed、标注比例相关合成数据。
6. `data/D-CGFS_Auxiliary*`：保存 target-base pair 和 mask，复现实验会用到。

## 可以后续清理或归档的内容

这些内容目前不直接用于论文主线，但建议在确认不需要复查后再删除或移动到离线备份。

| 路径 | 建议 |
| --- | --- |
| `__pycache__/`、`*/__pycache__/` | Python 缓存，可删除。 |
| `generated_data/_smoke_*` | smoke test 产物，可归档或删除。 |
| `generated_data/debug_synthesis_core_smoke` | debug 产物，可归档或删除。 |
| `generated_data/awa2_smoke_synthesis` | AwA2 smoke 产物，可归档或删除。 |
| `final_evaluation_results/awa2_smoke_step4` | smoke 评估结果，可归档或删除。 |
| `checkpoints/awa2_smoke`、`checkpoints/awa2_smoke_step3.pt` | smoke checkpoint，可归档或删除。 |
| `run_logs/` | 旧日志，结果已汇总后可压缩归档。 |

## 建议的后续物理整理

当前没有移动结果目录，原因是路径已被论文材料和脚本引用。若后续要进一步整理，建议按以下方式做，并同步修改文档和脚本路径：

```text
archive/smoke_tests/
archive/historical_branches/
archive/old_logs/
```

只有在确认不再复现实验、不再读取旧路径后，才移动大型结果目录。
