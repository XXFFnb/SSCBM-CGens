# ==========================================================================================
# D-CGFS 统一配置
#
# D-CGFS:
# Discriminative Concept-Guided Feature Synthesis for Imbalanced Semi-Supervised
# Concept Bottleneck Models.
#
# 这个文件用于把论文主贡献统一收束为 D-CGFS。
# 历史命令行参数 cgens 仍可被识别，但会立即映射为 dcgfs。
# ==========================================================================================


METHOD_NAME = "Discriminative Concept-Guided Feature Synthesis"
METHOD_ACRONYM = "D-CGFS"

METHOD_KEY = "dcgfs"
LEGACY_METHOD_ALIAS = "cgens"

# 合成样本损失权重：
# L_syn = L_task + lambda_concept * L_concept + lambda_proto * L_proto
SYN_CONCEPT_LOSS_WEIGHT = 0.2
SYN_PROTO_LOSS_WEIGHT = 0.1

# 问题八补强项：
# 1. teacher-student consistency：保持微调模型在原始样本上的输出不要偏离 baseline teacher 太多。
# 2. distribution regularization：约束合成 batch 的平均预测分布接近目标标签分布，避免合成监督坍缩。
TEACHER_CONSISTENCY_WEIGHT = 0.05
DISTRIBUTION_REG_WEIGHT = 0.05
TEACHER_TEMPERATURE = 2.0

# 当前主实验配置：
# pair_topk_filter 负责生成每个 target-base pair 内质量更稳定的合成样本。
# 经过 target-score 诊断后，主方法将 pair 内排序的 target confidence 权重从 0.05
# 提高到 0.15：这样不会降低 target acc，同时提升 overall/macro/base/intervention 指标。
# base preservation 在训练阶段对自动选择的基座类样本施加额外 teacher 约束，
# 让 student 不要为了提升弱势目标类而过度破坏原始基座类决策边界。
# 经过 0.135/0.145/0.150/0.155/0.165 局部验证后，0.15 在目标类准确率、
# 基座类保持和 target-to-base 错分抑制之间最符合 D-CGFS 的主目标。
PAIR_SCORE_TARGET_WEIGHT = 0.15
PAIR_SCORE_BASE_WEIGHT = 0.05
BASE_PRESERVATION_WEIGHT = 0.15


def normalize_method_name(method):
    """
    将历史命令行名称统一映射到 D-CGFS 的内部主方法键。

    返回值使用 'dcgfs'，避免代码内部继续混用旧名。
    """
    if method in {"dcgfs", "d-cgfs", LEGACY_METHOD_ALIAS}:
        return METHOD_KEY
    return method
