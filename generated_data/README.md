# generated_data

存放 D-CGFS 合成得到的特征样本、合成元数据和可视化图片。该目录体积较大，不应提交到远程仓库。

## 当前主线目录

四数据集主实验：

```text
problem6_target_score_w015/
awa2_target_score_w015/
pbc_target_score_w015/
7pt_target_score_w015/
```

多 seed 和标注比例补充实验：

```text
problem6_seed0_target_score_w015/
problem6_seed1_target_score_w015/
7pt_seed0_target_score_w015/
7pt_seed1_target_score_w015/
7pt_seed2_target_score_w015/
7pt_r005_target_score_w015/
7pt_r020_target_score_w015/
```

## 历史探索目录

这些目录对应 feature refinement、retrieval residual、model-aware D、hybrid pair refinement 等历史分支，不作为论文主方法：

```text
problem6_feature_refine_pred_disc/
problem6_feature_refine_window/
problem6_retrieval_residual/
problem6_model_aware_d_w050/
problem6_hybrid_pair0_refine/
problem6_old_spatial_*
problem6_pair_adaptive_*
problem6_target_score_w015_diverse*
```

## smoke/debug 目录

这些通常是开发测试产物，论文初稿完成并确认不再复查后可以归档或删除：

```text
_smoke_feature_refine/
_smoke_feature_refine_freeze/
_smoke_feature_refine_strong/
_smoke_hybrid_pair0_refine/
_smoke_retrieval_residual/
debug_synthesis_core_smoke/
awa2_smoke_synthesis/
```

## 常见文件

```text
feat_*.pt
metadata.csv
synthesized_metadata.csv
candidate_filter_diagnostics.csv
pair_filter_summary.csv
synthesized_images/
```

清理建议见：

```text
docs/FILE_AUDIT.md
```
