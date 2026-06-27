# final_evaluation_results

存放 `step4_final_evaluation.py` 生成的最终分类评估结果。该目录是论文表格的重要来源，不建议删除。

## 四数据集主实验

```text
problem6_dcgfs_target_score_w015/
awa2_dcgfs_target_score_w015/
pbc_dcgfs_target_score_w015/
7pt_dcgfs_target_score_w015/
```

## 多 seed 和标注比例补充实验

```text
problem6_seed0_dcgfs_target_score_w015/
problem6_seed1_dcgfs_target_score_w015/
7pt_seed0_dcgfs_target_score_w015/
7pt_seed1_dcgfs_target_score_w015/
7pt_seed2_dcgfs_target_score_w015/
7pt_r005_dcgfs_target_score_w015/
7pt_r020_dcgfs_target_score_w015/
```

## 强 baseline 与消融

```text
problem6_sscbm_finetune/
problem6_oversampling/
problem6_reweighting/
problem6_class_balanced_loss/
problem6_feature_mixup/
problem6_dcgfs_pair_topk_base_w015/
problem6_dcgfs_target_score_w015/
```

## 历史探索结果

这些目录保留为诊断记录，不作为当前主方法：

```text
problem6_dcgfs_feature_refine_pred_disc/
problem6_dcgfs_feature_refine_window/
problem6_dcgfs_retrieval_residual/
problem6_dcgfs_model_aware_d_w050/
problem6_dcgfs_hybrid_pair0_refine_base_train/
problem6_dcgfs_hybrid_pair0_refine_pred_disc_train/
problem6_dcgfs_old_spatial_*
problem6_dcgfs_pair_adaptive_*
```

## smoke/debug 结果

```text
awa2_smoke_step4/
```

论文完成后如需清理，可优先归档或删除 smoke/debug 结果。

## 常见文件

```text
model_summary.csv
classwise_metrics.csv
target_base_confusion.csv
metric_glossary.csv
```

论文写作时优先使用 `paper_tables/` 中已经汇总好的表格。
