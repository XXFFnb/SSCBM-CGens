# 2026-05-20 Code Review and Optimization-Push Plan

## 已修缮文档

- `问题解决方案/current_idea_D-CGFS.md`

主要更新：

1. 当前正式主配置改为 `D-CGFS target_score_w015`。
2. 明确 pair-score 公式：

```text
score = concept_delta + 0.15 * log(target_prob) - 0.05 * log(base_prob)
```

3. 补充当前实验短板：D-CGFS 已明显优于 SSCBM，但 target accuracy 仍低于 reweighting/class-balanced loss。
4. 补充下一步算法优化方向：把 class-balanced loss 融合进 D-CGFS，并让 synthetic task logits 更目标语义导向。

## 代码审查发现

### 1. 合成任务 logits 过度依赖基座概念概率

在旧实现中，合成样本训练调用：

```python
c_probs_s, t_logits_s = model.predict_from_features(
    feat_s,
    pos_emb_s,
    neg_emb_s,
    concept_probs_base_s,
)
```

这意味着 synthetic task logits 使用的是 `concept_probs_base_s` 组合正/负概念 embedding。结果是：

1. 合成样本虽然标签是目标类，但任务 logits 的概念概率来源仍偏基座类。
2. 合成特征 `feat_s` 对概念预测有影响，但对分类 logits 的目标语义推动不够直接。
3. 这可以解释为什么 D-CGFS 能提升 target acc，但仍追不上 reweighting/class-balanced loss。

### 2. D-CGFS 没有吸收 strong baseline 的有效成分

实验已经显示 `reweighting/class-balanced loss` 的 target accuracy 明显更高。旧 D-CGFS 只是和它们对比，没有把类别均衡任务损失整合进 D-CGFS。

因此下一步最合理的优化不是继续只改合成样本，而是测试：

```text
D-CGFS + class-balanced task loss
```

### 3. SSCBM 中存在调试打印

`models/sscbm.py` 在 `output_embeddings=True` 时会打印 `output_embedding`，导致 step2 合成阶段控制台刷屏，影响运行效率和日志可读性。

## 已完成代码修改

### `step3_balance_training.py`

新增参数：

```bash
--enable-class-balanced-task-loss
```

作用：

- 在 D-CGFS 中启用类别均衡任务损失。
- 同时影响原始样本任务损失和合成样本任务损失。

新增参数：

```bash
--synthetic-task-concept-source {base,predicted,target_proto,target_disc_mix}
```

作用：

- `base`: 复现旧实现。
- `predicted`: 使用合成特征预测出的概念概率构造 synthetic task logits。
- `target_proto`: 使用目标类概念原型构造 synthetic task logits。
- `target_disc_mix`: 只在判别概念上使用目标类概念原型，其他概念仍使用基座概念概率。

### `run_problem6_experiments.py`

新增实验组：

```bash
optimization_push
```

包含四个候选：

1. `dcgfs_cb_task_loss`
2. `dcgfs_syn_task_target_disc_mix`
3. `dcgfs_cb_task_target_disc_mix`
4. `dcgfs_cb_task_target_proto`

这些实验都复用当前主配置合成数据：

```text
generated_data/problem6_target_score_w015
```

### `models/sscbm.py`

删除 `output_embedding` / `output_latent` / `output_intervention` 调试打印，减少 step2 合成阶段无意义刷屏。

## 验证

已通过：

```bash
.venv/bin/python -m py_compile models/sscbm.py step3_balance_training.py run_problem6_experiments.py dcgfs_config.py step2_synthesize_data.py
bash -n run_dcgfs_pipeline.sh
.venv/bin/python run_problem6_experiments.py --only-group optimization_push
.venv/bin/python run_problem6_experiments.py
```

## 下一步命令

建议优先运行：

```bash
.venv/bin/python run_problem6_experiments.py --run --only-group optimization_push
```

如果这组中任一方法能把 Target Acc 从当前主方法的 `44.29` 显著推高，同时 Base Acc 不低于 strong baseline，则 D-CGFS 的结果会更接近顶会/顶刊要求。
