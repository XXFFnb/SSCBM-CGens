import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptLocator(nn.Module):
    """
    新增定位模块：从目标样本中提取纯概念特征 [cite: 78]
    """

    def __init__(self):
        super(ConceptLocator, self).__init__()

    def forward(self, feature_map, concept_mask):
        """
        输入:
            feature_map: SSCBM Backbone 输出的特征图 [batch, 512, 7, 7] [cite: 53]
            concept_mask: 对应 7x7 尺寸的二值掩码 [batch, 1, 7, 7]
        输出:
            纯概念特征张量 [cite: 81]
        """
        # 将掩码应用到特征图上，保留掩码为 1 的区域，其余置为 0
        localized_features = feature_map * concept_mask
        return localized_features


class ConceptGatedFuser(nn.Module):
    """
    概念门控融合模块：用判别概念差异控制目标/基座特征的融合强度。

    问题五的核心是避免 D-CGFS 被看作简单的 masked feature mixup。
    因此这里不再使用固定 alpha 做线性融合：

        fused = alpha * F_target + (1 - alpha) * F_base

    而是使用由判别概念集合 D 生成的门控：

        F = G_D * F_target + (1 - G_D) * F_base

    其中 G_D 来自目标类和基座类在判别概念上的原型差异。
    差异越大，说明该区域越应该注入目标类特征；差异越小，则更多保留基座类结构。
    """

    def __init__(self, gate_temperature=5.0, min_gate=0.5, max_gate=0.95):
        super(ConceptGatedFuser, self).__init__()
        self.gate_temperature = gate_temperature
        self.min_gate = min_gate
        self.max_gate = max_gate

    def forward(
            self,
            base_feature_map,
            base_mask,
            target_concept_features,
            target_proto,
            base_proto,
            discriminative_mask,
    ):
        """
        输入:
            base_feature_map: 基座样本特征图 [batch, C, H, W]
            base_mask: 基座样本判别区域 mask [batch, 1, H, W]
            target_concept_features: 目标样本判别区域特征 [batch, C, H, W]
            target_proto: 目标类概念原型 [batch, n_concepts]
            base_proto: 基座类概念原型 [batch, n_concepts]
            discriminative_mask: 判别概念集合 D 的 mask [batch, n_concepts]

        输出:
            out_feature_map: 概念门控融合后的特征图 [batch, C, H, W]
            gate: 每个样本的融合门控强度 [batch, 1, 1, 1]
        """
        # 1. 提取基座图像中将被替换/融合的判别区域。
        base_region_features = base_feature_map * base_mask

        # 2. 根据判别概念差异生成 gate。
        # gap = p_target - p_base，只在 D 中计算平均差异。
        # gate 越大，融合区域越偏向目标类特征。
        concept_gap = (target_proto - base_proto) * discriminative_mask
        denom = discriminative_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        gate_score = concept_gap.sum(dim=1, keepdim=True) / denom
        gate = torch.sigmoid(self.gate_temperature * gate_score)
        gate = self.min_gate + (self.max_gate - self.min_gate) * gate
        gate = gate.view(-1, 1, 1, 1)

        # 3. 概念门控融合。
        fused_region = gate * target_concept_features + (1 - gate) * base_region_features

        # 4. L2 归一化，保持特征分布稳定。
        fused_region = F.normalize(fused_region, p=2, dim=1)

        # 5. 将融合后的判别区域放回基座特征图。
        out_feature_map = (base_feature_map * (1 - base_mask)) + fused_region

        return out_feature_map, gate
