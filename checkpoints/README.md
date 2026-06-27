# checkpoints

存放训练得到的模型权重文件，包括原始 SSCBM checkpoint、D-CGFS 主方法 checkpoint、强 baseline checkpoint 和历史诊断实验 checkpoint。

该目录通常包含大文件，不应提交到远程仓库。

## Baseline checkpoint

```text
CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt
AwA2_16-15/SemiSupervisedConceptEmbeddingModel.pt
PBC_17-01/SemiSupervisedConceptEmbeddingModel.pt
7pt_17-44/SemiSupervisedConceptEmbeddingModel.pt
```

## 四数据集主方法 checkpoint

```text
problem6_dcgfs_target_score_w015.pt
awa2_dcgfs_target_score_w015.pt
pbc_dcgfs_target_score_w015.pt
7pt_dcgfs_target_score_w015.pt
```

## 多 seed 和标注比例补充 checkpoint

```text
problem6_seed0_dcgfs_target_score_w015.pt
problem6_seed1_dcgfs_target_score_w015.pt
7pt_seed0_dcgfs_target_score_w015.pt
7pt_seed1_dcgfs_target_score_w015.pt
7pt_seed2_dcgfs_target_score_w015.pt
7pt_r005_dcgfs_target_score_w015.pt
7pt_r020_dcgfs_target_score_w015.pt
```

## 历史探索 checkpoint

这些 checkpoint 不属于当前论文主方法，但保留为诊断记录：

```text
problem6_dcgfs_feature_refine_*.pt
problem6_dcgfs_retrieval_residual.pt
problem6_dcgfs_model_aware_d_w050.pt
problem6_dcgfs_hybrid_pair0_*.pt
problem6_dcgfs_old_spatial_*.pt
problem6_dcgfs_pair_adaptive_*.pt
```

## smoke/debug checkpoint

```text
awa2_smoke/
awa2_smoke_step3.pt
```

这些可在论文初稿完成并确认不再复查后归档或删除。
