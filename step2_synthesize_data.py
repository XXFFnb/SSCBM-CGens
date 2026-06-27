# ==========================================================================================
# SSCBM + D-CGFS 研究项目 - 步骤 2: 在特征空间合成新样本并可视化
#
# 核心目标：
# 利用步骤 1 生成的区域掩码，在特征空间中执行“概念引导的语义合成”。
# 同时，在图像空间中模拟这一过程，生成可供直观观察的合成图片。
#
# 工作流程：
# 1. 加载预训练的 SSCBM 模型，用于提取图像的深层特征图。
# 2. 初始化 D-CGFS 模块：ConceptLocator 和 ConceptGatedFuser。
# 3. 创建一个 DataLoader，每次迭代都提供一个目标样本和一个基座样本。
#    如果 step1 中存在多个自动目标类-基座类 pair，则只在同一个 pair_id 内做配对，
#    避免把 A 目标类错误融合到 B 基座类上。
# 4. 在训练循环中，执行以下操作：
#    a. (特征空间) 提取特征图，通过 D-CGFS 模块融合，生成“合成特征图”。
#    b. (特征空间) 使用目标类先验标签，并用目标类置信度与概念一致性过滤合成样本。
#    c. (图像空间) 调用 `save_visual_synthesis` 函数，将被替换区域的像素进行融合，
#       生成并保存一张可视化的合成图像。
#    d. 保存“合成特征图”、目标类先验标签和判别概念监督到 .pt 文件，供步骤 3 使用。
# 5. 记录元数据到 CSV 文件。
#
# 作者：[肖凡]
# 日期：[2026年4月20日]
# ==========================================================================================

import torch
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from models.sscbm import SSCBM
from models.dcgfs_modules import ConceptLocator, ConceptGatedFuser
from torchvision import transforms
from torchvision.models import resnet34
from train.utils import wrap_pretrained_model
from dcgfs_config import METHOD_ACRONYM, normalize_method_name
from dataset_specs import build_sscbm, load_dataset_config, load_dataset_spec

# --- 1. 全局配置 ---
AUX_DIR = "data/D-CGFS_Auxiliary"
DATA_ROOT = "data/CUB_200_2011"
IMAGE_SUBDIR = "images"
CLASS_ID_OFFSET = 1
GEN_DATA_DIR = "generated_data"
# 新增一个目录，专门用于存放合成的图片
SYN_IMG_DIR = os.path.join(GEN_DATA_DIR, "synthesized_images")
os.makedirs(GEN_DATA_DIR, exist_ok=True)
os.makedirs(SYN_IMG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_RES = (10, 10)

# 问题三解决方案的两个过滤超参数：
# 1. 目标类置信度过滤：只有 P(y_target | x_syn) 高于随机猜测水平的合成样本才保留。
# 2. 概念一致性过滤：合成样本的概念预测必须比基座类更接近目标类概念原型。
#
# 注意：CUB 是 200 类任务，随机水平约为 1/200=0.005。
# 当前目标类是 baseline 在验证集上自动发现的弱势类，如果阈值设为 0.3，
# 等价于要求“增强前模型已经非常相信目标类”，这会导致所有合成样本被过滤。
# 因此默认值设为 0.01：仍高于随机水平，但适合 weak-class augmentation 场景。
TARGET_CONF_THRESHOLD = 0.01
CONCEPT_MARGIN = 0.0


def parse_args():
    """
    解析问题六实验所需的合成阶段参数。

    默认 ablation_mode='dcgfs' 表示运行完整 D-CGFS。
    仍然接受历史别名 cgens，但内部会自动映射为 dcgfs。
    其他模式用于消融实验，目的是回答“到底是哪一部分带来了提升”：
        - random_mask: 用随机区域替代概念定位区域，检验概念 mask 是否必要。
        - all_concepts_mask: 用全部概念参与门控和监督，检验判别概念 D 是否必要。
        - no_conf_filter: 去掉目标类置信度过滤，检验伪标签过滤是否必要。
        - no_concept_filter: 去掉概念一致性过滤，检验概念语义约束是否必要。
        - fixed_alpha: 用固定 alpha 融合，检验动态概念门控是否必要。
        - pair_topk_filter: 不使用绝对 target_prob 硬阈值，而是在每个 target-base pair 内
          按质量分数选择 top-k 样本，作为新的主方法候选。
    """
    parser = argparse.ArgumentParser(description=f"{METHOD_ACRONYM} 合成数据生成与消融实验")
    parser.add_argument("--dataset", default="CUB-200-2011", choices=["CUB-200-2011", "AwA2", "PBC", "7pt"])
    parser.add_argument("--checkpoint", default=None, help="Baseline SSCBM checkpoint；不填使用数据集默认路径。")
    parser.add_argument("--aux-dir", default=None, help="step1 生成的辅助映射目录；不填使用数据集默认目录。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子；控制 DataLoader shuffle 和随机 mask 消融。")
    parser.add_argument("--ablation-mode", default="dcgfs", choices=[
        "dcgfs",
        "cgens",
        "random_mask",
        "all_concepts_mask",
        "no_conf_filter",
        "no_concept_filter",
        "fixed_alpha",
        "pair_topk_filter",
    ])
    parser.add_argument("--samples-per-pair", type=int, default=2000)
    parser.add_argument("--output-dir", default=GEN_DATA_DIR)
    parser.add_argument("--fixed-alpha", type=float, default=0.7)
    parser.add_argument(
        "--fusion-mode",
        choices=["aligned_pool", "old_spatial"],
        default="old_spatial",
        help=(
            "D-CGFS 特征融合方式。aligned_pool 先池化目标判别区域，再注入基座 mask 区域；"
            "old_spatial 保留当前主方法使用的同位置 masked feature 相加逻辑。"
        ),
    )
    parser.add_argument(
        "--synthesis-task-concept-source",
        choices=["base", "predicted", "target_proto", "target_disc_mix", "predicted_disc_mix"],
        default="base",
        help=(
            "step2 合成质量评分使用的任务概念概率来源。predicted_disc_mix 表示判别概念用"
            "合成特征预测概念，非判别概念保留基座概率。默认 base 复现当前主配置。"
        ),
    )
    parser.add_argument(
        "--synthesis-disc-mix-weight",
        type=float,
        default=1.0,
        help=(
            "判别概念任务概率的软注入权重，仅作用于 target_disc_mix/predicted_disc_mix。"
            "1.0 表示完全使用目标/合成判别概念，0.0 退化为基座概念概率。"
        ),
    )
    parser.add_argument("--target-conf-threshold", type=float, default=TARGET_CONF_THRESHOLD)
    parser.add_argument("--concept-margin", type=float, default=CONCEPT_MARGIN)
    parser.add_argument(
        "--pair-topk",
        type=int,
        default=500,
        help="pair_topk_filter 模式下，每个 target-base pair 最多保留的合成样本数量。",
    )
    parser.add_argument(
        "--pair-topk-max-per-source-pair",
        type=int,
        default=0,
        help=(
            "pair_topk_filter 模式下，同一 target/base 源图组合最多保留多少次；"
            "0 表示不限制，用于保持当前主实验配置。"
        ),
    )
    parser.add_argument(
        "--pair-adaptive-topk-mode",
        choices=["none", "target_conf", "source_diversity"],
        default="none",
        help=(
            "pair_topk_filter 的自适应保留策略。none 保持每个 pair 固定 top-k；"
            "target_conf 根据目标类置信度达到阈值的候选数量决定 top-k；"
            "source_diversity 根据唯一 target/base 源图组合数量决定 top-k。"
        ),
    )
    parser.add_argument(
        "--pair-adaptive-min-topk",
        type=int,
        default=200,
        help="自适应 top-k 模式下，每个 pair 至少保留的样本数，防止弱 pair 完全没有训练信号。",
    )
    parser.add_argument(
        "--pair-adaptive-conf-threshold",
        type=float,
        default=1e-6,
        help="target_conf 自适应模式中，认为候选样本仍有目标类证据的 target_prob 阈值。",
    )
    parser.add_argument(
        "--pair-adaptive-conf-multiplier",
        type=float,
        default=4.0,
        help="target_conf 自适应模式中，有效目标置信候选数到保留样本数的放大倍数。",
    )
    parser.add_argument(
        "--pair-adaptive-source-multiplier",
        type=float,
        default=20.0,
        help="source_diversity 自适应模式中，唯一源图组合数到保留样本数的放大倍数。",
    )
    parser.add_argument(
        "--pair-score-target-weight",
        type=float,
        default=0.05,
        help="pair_topk_filter 质量分数中 target_prob 对数项的权重。",
    )
    parser.add_argument(
        "--pair-score-base-weight",
        type=float,
        default=0.05,
        help="pair_topk_filter 质量分数中 base_prob 对数惩罚项的权重。",
    )
    parser.add_argument("--min-keep-samples", type=int, default=200)
    parser.add_argument(
        "--enable-quality-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="严格过滤保留过少时，按质量分数补充少量样本，并在 metadata 中标记。",
    )
    return parser.parse_args()


class SynthesisDataset(Dataset):
    """
    自定义数据集，用于配对加载“目标样本”和“基座样本”。

    输入 CSV:
        concept_region_mapping.csv:
            step1 生成的目标类图像与目标 mask 映射。
        base_sample_regions.csv:
            step1 生成的基座类图像与基座 mask 映射。

    核心逻辑:
        - 每条样本由一个 target 图像和一个 base 图像组成。
        - 二者必须来自同一个 pair_id。
        - pair_id 必须由 step1_generate_mapping.py 根据自动配对结果写入。
        - discriminative_concepts 记录当前 pair 使用了哪些判别概念生成 mask。
        - target_concept_proto/base_concept_proto 用于判断合成样本概念预测是否更像目标类。
    """
    def __init__(self, target_csv, base_csv, transform, samples_per_pair=2000):
        self.targets = pd.read_csv(target_csv)
        self.bases = pd.read_csv(base_csv)
        self.transform = transform

        # 每个 target-base pair 生成多少条合成样本。
        # 总合成样本数 = samples_per_pair * pair 数量。
        self.samples_per_pair = samples_per_pair

        required_cols = {
            "pair_id",
            "img_path",
            "mask_path",
            "class",
            "target_class",
            "base_class",
            "discriminative_concepts",
            "target_concept_proto",
            "base_concept_proto",
        }
        target_missing = required_cols - set(self.targets.columns)
        base_missing = required_cols - set(self.bases.columns)
        if target_missing:
            raise ValueError(f"{target_csv} 缺少必要列: {sorted(target_missing)}。请重新运行 step1_generate_mapping.py。")
        if base_missing:
            raise ValueError(f"{base_csv} 缺少必要列: {sorted(base_missing)}。请重新运行 step1_generate_mapping.py。")

        # 只保留 target 和 base 两边都存在的 pair。
        # 如果某个 pair 只生成了 target 或只生成了 base，不能用于融合。
        target_pairs = set(self.targets["pair_id"].unique())
        base_pairs = set(self.bases["pair_id"].unique())
        self.pair_ids = sorted(target_pairs & base_pairs)
        if not self.pair_ids:
            raise ValueError("目标样本和基座样本没有可匹配的 pair_id，请先运行 step1_generate_mapping.py。")

    def __len__(self):
        return self.samples_per_pair * len(self.pair_ids)

    def __getitem__(self, idx):
        # 先确定当前样本属于哪个 pair，再在该 pair 内循环取 target/base。
        # 这样多个 pair 会被均匀采样，而不会发生跨 pair 配对。
        pair_id = self.pair_ids[idx % len(self.pair_ids)]
        pair_idx = idx // len(self.pair_ids)
        target_rows = self.targets[self.targets["pair_id"] == pair_id]
        base_rows = self.bases[self.bases["pair_id"] == pair_id]

        t_row = target_rows.iloc[pair_idx % len(target_rows)]
        b_row = base_rows.iloc[pair_idx % len(base_rows)]

        # 返回原始图像路径，以便在可视化函数中加载未经 transform 的原图
        t_img_path = os.path.join(DATA_ROOT, IMAGE_SUBDIR, t_row['img_path'])
        b_img_path = os.path.join(DATA_ROOT, IMAGE_SUBDIR, b_row['img_path'])
        
        t_img = self.transform(Image.open(t_img_path).convert('RGB'))
        b_img = self.transform(Image.open(b_img_path).convert('RGB'))

        t_mask_path = os.path.join(AUX_DIR, t_row['mask_path'])
        b_mask_path = os.path.join(AUX_DIR, b_row['mask_path'])

        # step1 保存的 mask 是原图尺寸。
        # 特征融合发生在 SSCBM 的空间特征图上，所以这里缩放到 MASK_RES。
        t_mask = Image.open(t_mask_path).convert('L').resize(MASK_RES)
        b_mask = Image.open(b_mask_path).convert('L').resize(MASK_RES)

        # mask 转成 [1, H, W]，后续可以和 [C, H, W] 的特征图广播相乘。
        t_mask = torch.from_numpy(np.array(t_mask) / 255.0).float().unsqueeze(0)
        b_mask = torch.from_numpy(np.array(b_mask) / 255.0).float().unsqueeze(0)

        # 将原始路径、掩码路径和 pair 元信息也一并返回：
        # - 路径用于保存可视化合成图；
        # - pair/class 元信息用于写入 generated_data/synthesized_metadata.csv。
        target_class = int(t_row.get("target_class", t_row["class"]))
        base_class = int(b_row.get("base_class", b_row["class"]))
        discriminative_concepts = str(t_row["discriminative_concepts"])
        target_concept_proto = str(t_row["target_concept_proto"])
        base_concept_proto = str(t_row["base_concept_proto"])

        return (
            t_img, t_mask, b_img, b_mask,
            t_img_path, b_img_path, t_mask_path, b_mask_path,
            int(pair_id), target_class, base_class, discriminative_concepts,
            target_concept_proto, base_concept_proto,
        )


def parse_float_vector(vector_str):
    """解析分号分隔的浮点向量，例如 target/base 概念原型。"""
    return [float(item) for item in str(vector_str).split(";") if item.strip() != ""]


def parse_index_vector(index_str):
    """解析分号分隔的整数索引，例如判别概念集合 D。"""
    return [int(item) for item in str(index_str).split(";") if item.strip() != ""]


def build_batch_vectors(vector_strings, device):
    """将 batch 中的字符串形式概念原型转换成 [batch, n_concepts] 的 tensor。"""
    vectors = [parse_float_vector(item) for item in vector_strings]
    return torch.tensor(vectors, dtype=torch.float32, device=device)


def build_discriminative_supervision(discriminative_strings, target_proto, device):
    """
    构造合成样本的判别概念监督。

    返回:
        concept_targets:
            [batch, n_concepts]，只在判别概念 D 上填入目标类概念原型值。
        concept_mask:
            [batch, n_concepts]，D 中概念为 1，其余为 0。

    训练时只在 mask=1 的判别概念上计算概念损失：
        L_concept(c_syn, c_target^D)
    """
    concept_targets = torch.zeros_like(target_proto, device=device)
    concept_mask = torch.zeros_like(target_proto, device=device)
    for row_idx, concept_str in enumerate(discriminative_strings):
        concept_indices = parse_index_vector(concept_str)
        if not concept_indices:
            raise ValueError("discriminative_concepts 为空，请重新运行 find_weak_classes.py 和 step1_generate_mapping.py。")
        concept_targets[row_idx, concept_indices] = target_proto[row_idx, concept_indices]
        concept_mask[row_idx, concept_indices] = 1.0
    return concept_targets, concept_mask


def mix_discriminative_embeddings(base_embeddings, target_embeddings, concept_mask):
    """
    在判别概念维度上混合 target/base 的正负概念 embedding。

    SSCBM 的任务头不是直接吃空间特征，而是吃每个概念的 embedding：
        e_k = e_k^+ * p_k + e_k^- * (1 - p_k)

    如果合成样本仍然完全使用基座样本的 pos/neg embedding，即使空间特征中注入了目标
    区域，分类头看到的概念上下文仍会偏基座类。这里仅在判别概念集合 D 上替换为
    目标样本的概念 embedding，非判别概念继续保留基座上下文，符合“只迁移目标类判别
    概念，保留基座非判别结构”的 D-CGFS 设定。
    """
    concept_mask_3d = concept_mask.unsqueeze(-1)
    return base_embeddings * (1.0 - concept_mask_3d) + target_embeddings * concept_mask_3d


def build_synthesis_task_concepts(
        source,
        base_concept_probs,
        predicted_concept_probs,
        target_proto,
        concept_mask,
        disc_mix_weight=1.0,
):
    """
    构造 step2 质量评分所用的任务概念概率。

    旧实现使用 base_concept_probs 计算 task logits，导致 target_prob 基本不反映合成特征。
    新的默认值 predicted_disc_mix 更贴合理论：
        - 判别概念 D：使用合成特征预测出的概念概率；
        - 非判别概念：保留基座样本原始概念概率。

    这样 target_prob 才会受合成特征影响，同时又不会把整张图的概念分布硬改成目标类原型。
    """
    if source == "base":
        return base_concept_probs
    if source == "predicted":
        return predicted_concept_probs
    if source == "target_proto":
        return target_proto
    mix_weight = float(max(0.0, min(1.0, disc_mix_weight)))
    if source == "target_disc_mix":
        mixed_disc_probs = (
            base_concept_probs * (1.0 - mix_weight)
            + target_proto * mix_weight
        )
        return base_concept_probs * (1.0 - concept_mask) + mixed_disc_probs * concept_mask
    if source == "predicted_disc_mix":
        mixed_disc_probs = (
            base_concept_probs * (1.0 - mix_weight)
            + predicted_concept_probs * mix_weight
        )
        return base_concept_probs * (1.0 - concept_mask) + mixed_disc_probs * concept_mask
    raise ValueError(f"未知 synthesis_task_concept_source: {source}")


def randomize_mask_like(mask):
    """
    构造和原 mask 面积相近的随机 mask。

    这是问题六中的 random mask 消融：
    如果随机区域也能取得同样效果，说明 D-CGFS 的概念定位贡献不充分；
    如果随机区域明显变差，才能证明概念 mask 本身是有效设计。
    """
    random_mask = torch.zeros_like(mask)
    flat_random = random_mask.view(random_mask.size(0), -1)
    flat_source = mask.view(mask.size(0), -1)
    for row_idx in range(mask.size(0)):
        active_count = int((flat_source[row_idx] > 0.5).sum().item())
        active_count = max(1, min(active_count, flat_random.size(1)))
        perm = torch.randperm(flat_random.size(1), device=mask.device)[:active_count]
        flat_random[row_idx, perm] = 1.0
    return random_mask


def fixed_alpha_fusion(base_feature_map, base_mask, target_concept_features, alpha):
    """
    固定 alpha 融合消融。

    完整 D-CGFS 使用 ConceptGatedFuser，根据判别概念原型差异动态决定融合强度。
    这里故意退化为固定 alpha，用来验证“动态门控”是否比普通 feature mixup 更有效。
    """
    base_region_features = base_feature_map * base_mask
    fused_region = alpha * target_concept_features + (1 - alpha) * base_region_features
    fused_region = torch.nn.functional.normalize(fused_region, p=2, dim=1)
    out_feature_map = (base_feature_map * (1 - base_mask)) + fused_region
    gate = torch.full(
        (base_feature_map.size(0), 1, 1, 1),
        float(alpha),
        dtype=base_feature_map.dtype,
        device=base_feature_map.device,
    )
    return out_feature_map, gate


def save_visual_synthesis(t_path, b_path, b_mask_path, alpha, save_path):
    """
    在图像空间模拟特征融合过程，生成一张可视化的合成图片。
    Args:
        t_path (str): 原始目标图像的路径。
        b_path (str): 原始基座图像的路径。
        b_mask_path (str): 基座图像对应的、在步骤 1 中生成的 *原始尺寸* 掩码路径。
        alpha (float): 图像空间可视化用的融合权重，对应 ConceptGatedFuser 产生的 gate。
        save_path (str): 合成图像的保存路径。
    """
    t_img_orig = Image.open(t_path).convert('RGB')
    b_img = Image.open(b_path).convert('RGB')

    # 关键修复：将目标图像缩放到与基座图像相同的尺寸
    t_img = t_img_orig.resize(b_img.size)
    
    # 加载为基座图像生成的原始尺寸掩码
    b_mask = Image.open(b_mask_path).convert('L').resize(b_img.size)
    
    # 将 PIL Image 转换为 numpy 数组进行像素操作
    t_arr = np.array(t_img)
    b_arr = np.array(b_img)
    mask_arr = np.array(b_mask) / 255.0
    # 将掩码扩展到 3 个颜色通道
    mask_3d = np.expand_dims(mask_arr, axis=2)

    # --- 图像空间融合 ---
    # 1. 提取基座图像中将被替换的区域
    base_region = b_arr * mask_3d
    # 2. 提取目标图像中用于“粘贴”的区域
    target_region = t_arr * mask_3d
    # 3. 线性融合
    fused_region = (alpha * target_region) + ((1 - alpha) * base_region)
    # 4. 将基座图像的非掩码部分与融合后的区域合并
    synthesized_arr = (b_arr * (1 - mask_3d)) + fused_region

    # 将 numpy 数组转回 PIL Image 并保存
    Image.fromarray(synthesized_arr.astype(np.uint8)).save(save_path)


def summarize_values(name, values):
    """打印过滤诊断统计，帮助判断阈值是否过严或过松。"""
    if not values:
        print(f"{name}: 无记录")
        return
    arr = np.array(values, dtype=np.float64)
    print(
        f"{name}: min={arr.min():.4f}, p25={np.percentile(arr, 25):.4f}, "
        f"median={np.median(arr):.4f}, p75={np.percentile(arr, 75):.4f}, max={arr.max():.4f}"
    )


def summarize_candidate_diagnostics(candidate_records, gen_data_dir):
    """
    保存合成候选样本的过滤诊断。

    synthesized_metadata.csv 只记录最终被保留的样本，无法解释“为什么大部分样本被拒绝”。
    因此这里额外保存两个诊断文件：
        1. candidate_filter_diagnostics.csv:
           每个候选样本的 target_prob、concept_delta、是否通过各层过滤。
        2. pair_filter_summary.csv:
           按 target-base pair 汇总通过率和主要失败原因。

    这两个文件只用于实验分析，不参与 step3 训练，避免改变原有训练流程。
    """
    diagnostics_path = os.path.join(gen_data_dir, "candidate_filter_diagnostics.csv")
    summary_path = os.path.join(gen_data_dir, "pair_filter_summary.csv")

    candidate_df = pd.DataFrame(candidate_records)
    candidate_df.to_csv(diagnostics_path, index=False)

    if candidate_df.empty:
        pd.DataFrame().to_csv(summary_path, index=False)
        print(f"候选样本诊断已保存: {diagnostics_path}")
        print(f"pair 过滤汇总已保存: {summary_path}")
        return

    # 按 pair 汇总严格通过率、置信度过滤通过率、概念一致性通过率。
    # 同时统计“只卡在置信度”“只卡在概念一致性”“两者都失败”的数量。
    summary_rows = []
    for (pair_id, target_class, base_class), group in candidate_df.groupby(
            ["pair_id", "target_class", "base_class"],
            sort=True,
    ):
        total = len(group)
        pass_conf = group["pass_conf_filter"].astype(bool)
        pass_concept = group["pass_concept_filter"].astype(bool)
        pass_strict = group["pass_strict_filter"].astype(bool)
        concept_delta = group["concept_delta"]

        summary_rows.append({
            "pair_id": pair_id,
            "target_class": target_class,
            "base_class": base_class,
            "candidate_count": total,
            "strict_pass_count": int(pass_strict.sum()),
            "strict_pass_rate": float(pass_strict.mean()),
            "conf_pass_count": int(pass_conf.sum()),
            "conf_pass_rate": float(pass_conf.mean()),
            "concept_pass_count": int(pass_concept.sum()),
            "concept_pass_rate": float(pass_concept.mean()),
            "fail_conf_only_count": int((~pass_conf & pass_concept).sum()),
            "fail_concept_only_count": int((pass_conf & ~pass_concept).sum()),
            "fail_both_count": int((~pass_conf & ~pass_concept).sum()),
            "target_prob_mean": float(group["target_prob"].mean()),
            "target_prob_max": float(group["target_prob"].max()),
            "base_prob_mean": float(group["base_prob"].mean()) if "base_prob" in group else np.nan,
            "base_prob_max": float(group["base_prob"].max()) if "base_prob" in group else np.nan,
            "concept_delta_mean": float(concept_delta.mean()),
            "concept_delta_max": float(concept_delta.max()),
            "pair_quality_score_mean": float(group["pair_quality_score"].mean())
            if "pair_quality_score" in group else np.nan,
            "pair_quality_score_max": float(group["pair_quality_score"].max())
            if "pair_quality_score" in group else np.nan,
        })

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"候选样本诊断已保存: {diagnostics_path}")
    print(f"pair 过滤汇总已保存: {summary_path}")


def save_synthetic_sample(
        record_id,
        gen_data_dir,
        syn_img_dir,
        fused_feat_in,
        b_pos_emb,
        b_neg_emb,
        c_sem_b,
        synthetic_labels,
        concept_targets,
        concept_mask,
        target_proto,
        pair_ids,
        target_classes,
        base_classes,
        discriminative_concepts,
        target_probs,
        sim_target,
        sim_base,
        fusion_gate,
        t_paths,
        b_paths,
        b_mask_paths,
        sample_idx,
        filter_mode,
):
    """
    保存单个合成样本。

    filter_mode 用于区分：
        - strict: 同时通过目标类置信度和概念一致性过滤。
        - quality_fallback: 严格过滤样本过少时，按质量分数补充保留。

    这样后续论文和调试时可以清楚知道样本是如何进入训练集的。
    """
    feat_save_path = os.path.join(gen_data_dir, f"feat_{record_id}.pt")
    torch.save({
        'feature': fused_feat_in[sample_idx].cpu(),
        'pos_embedding': b_pos_emb[sample_idx].cpu(),
        'neg_embedding': b_neg_emb[sample_idx].cpu(),
        'concept_probs_base': c_sem_b[sample_idx].cpu(),
        'label': synthetic_labels[sample_idx].cpu(),
        'concepts': concept_targets[sample_idx].cpu(),
        'concept_mask': concept_mask[sample_idx].cpu(),
        'target_concept_proto': target_proto[sample_idx].cpu(),
        'pair_id': int(pair_ids[sample_idx].item()),
        'target_class': int(target_classes[sample_idx].item()),
        'base_class': int(base_classes[sample_idx].item()),
        'discriminative_concepts': discriminative_concepts[sample_idx],
        'target_prob': target_probs[sample_idx].cpu(),
        'concept_sim_target': sim_target[sample_idx].cpu(),
        'concept_sim_base': sim_base[sample_idx].cpu(),
        'fusion_gate': fusion_gate[sample_idx].reshape(-1).cpu(),
        'filter_mode': filter_mode,
    }, feat_save_path)

    img_save_path = os.path.join(syn_img_dir, f"syn_{record_id}.jpg")
    visual_alpha = float(fusion_gate[sample_idx].reshape(-1).item())
    save_visual_synthesis(
        t_paths[sample_idx],
        b_paths[sample_idx],
        b_mask_paths[sample_idx],
        visual_alpha,
        img_save_path,
    )

    return {
        "feat_path": f"feat_{record_id}.pt",
        "img_path": os.path.join("synthesized_images", f"syn_{record_id}.jpg"),
        "pair_id": int(pair_ids[sample_idx].item()),
        "target_class": int(target_classes[sample_idx].item()),
        "base_class": int(base_classes[sample_idx].item()),
        "discriminative_concepts": discriminative_concepts[sample_idx],
        "target_prob": target_probs[sample_idx].item(),
        "concept_sim_target": sim_target[sample_idx].item(),
        "concept_sim_base": sim_base[sample_idx].item(),
        "fusion_gate": float(fusion_gate[sample_idx].reshape(-1).item()),
        "target_src": os.path.basename(t_paths[sample_idx]),
        "base_src": os.path.basename(b_paths[sample_idx]),
        "label": synthetic_labels[sample_idx].item(),
        "filter_mode": filter_mode,
    }


def build_pair_topk_payload(
        fused_feat_in,
        b_pos_emb,
        b_neg_emb,
        c_sem_b,
        synthetic_labels,
        concept_targets,
        concept_mask,
        target_proto,
        pair_ids,
        target_classes,
        base_classes,
        discriminative_concepts,
        target_probs,
        base_probs,
        sim_target,
        sim_base,
        concept_delta,
        pair_quality_scores,
        fusion_gate,
        t_paths,
        b_paths,
        b_mask_paths,
        sample_idx,
):
    """
    构造 pair_topk_filter 候选样本的轻量保存包。

    pair_topk_filter 必须先看完同一个 target-base pair 的所有候选样本，才能决定 top-k。
    因此它不能像普通 strict 过滤那样边生成边写文件，而是先把有机会进入 top-k 的样本
    放到内存中，最后统一排序、截断并保存。

    这里把张量立即搬到 CPU，避免长时间占用 GPU 显存；路径和标量元信息用于最后保存
    .pt 特征文件、可视化图片和 metadata。
    """
    pair_id = int(pair_ids[sample_idx].item())
    target_class = int(target_classes[sample_idx].item())
    base_class = int(base_classes[sample_idx].item())
    return {
        "pair_id": pair_id,
        "target_class": target_class,
        "base_class": base_class,
        "score": float(pair_quality_scores[sample_idx].item()),
        "feature": fused_feat_in[sample_idx].cpu(),
        "pos_embedding": b_pos_emb[sample_idx].cpu(),
        "neg_embedding": b_neg_emb[sample_idx].cpu(),
        "concept_probs_base": c_sem_b[sample_idx].cpu(),
        "label": synthetic_labels[sample_idx].cpu(),
        "concepts": concept_targets[sample_idx].cpu(),
        "concept_mask": concept_mask[sample_idx].cpu(),
        "target_concept_proto": target_proto[sample_idx].cpu(),
        "discriminative_concepts": discriminative_concepts[sample_idx],
        "target_prob": float(target_probs[sample_idx].item()),
        "base_prob": float(base_probs[sample_idx].item()),
        "concept_sim_target": float(sim_target[sample_idx].item()),
        "concept_sim_base": float(sim_base[sample_idx].item()),
        "concept_delta": float(concept_delta[sample_idx].item()),
        "fusion_gate": fusion_gate[sample_idx].reshape(-1).cpu(),
        "target_src_path": t_paths[sample_idx],
        "base_src_path": b_paths[sample_idx],
        "source_pair_key": (
            os.path.basename(t_paths[sample_idx]),
            os.path.basename(b_paths[sample_idx]),
        ),
        "base_mask_path": b_mask_paths[sample_idx],
    }


def compute_adaptive_pair_topk(candidates, args):
    """
    根据单个 pair 的候选质量计算自适应 top-k。

    默认主方法对每个 pair 都保留固定 top-k=500。诊断发现这会带来两个问题：
    1. 某些 pair 的 target_prob 极低，硬塞满 500 个样本可能放大低质量合成噪声。
    2. 某些 pair 的 top-k 来自极少数 target/base 源图组合，样本数量大但有效多样性低。

    因此这里提供两种论文可解释的自适应策略：
    - target_conf: 根据 target_prob >= 阈值的候选数量估计该 pair 的可靠合成容量。
    - source_diversity: 根据唯一 target/base 源图组合数量估计该 pair 的有效多样性容量。

    该函数只在显式开启 adaptive 模式时生效；默认 none 完全复现当前主配置。
    """
    if args.pair_adaptive_topk_mode == "none":
        return int(args.pair_topk)

    min_topk = max(1, int(args.pair_adaptive_min_topk))
    max_topk = int(args.pair_topk)
    if args.pair_adaptive_topk_mode == "target_conf":
        valid_count = sum(
            candidate["target_prob"] >= args.pair_adaptive_conf_threshold
            for candidate in candidates
        )
        adaptive_topk = int(round(valid_count * args.pair_adaptive_conf_multiplier))
    elif args.pair_adaptive_topk_mode == "source_diversity":
        unique_source_pairs = {candidate["source_pair_key"] for candidate in candidates}
        adaptive_topk = int(round(len(unique_source_pairs) * args.pair_adaptive_source_multiplier))
    else:
        raise ValueError(f"未知 pair_adaptive_topk_mode: {args.pair_adaptive_topk_mode}")

    return max(min_topk, min(max_topk, adaptive_topk))


def select_pair_topk_candidates(
        candidates,
        pair_topk,
        max_per_source_pair=0,
):
    """
    在单个 target-base pair 内选择最终保留的 top-k 合成样本。

    默认 max_per_source_pair=0 时完全复现原始 pair-topk：只按质量分数排序取前 k。
    当 max_per_source_pair>0 时，限制同一 target/base 源图组合的重复次数。
    这个参数用于解决诊断中发现的高重复问题：某些 pair 的 top-500 实际只来自十几个
    图像组合，表面样本量很大，但有效多样性很低。
    """
    ordered_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)

    if max_per_source_pair <= 0:
        return ordered_candidates[:pair_topk]

    selected = []
    source_pair_counts = {}
    for candidate in ordered_candidates:
        source_key = candidate["source_pair_key"]
        if source_pair_counts.get(source_key, 0) >= max_per_source_pair:
            continue
        selected.append(candidate)
        source_pair_counts[source_key] = source_pair_counts.get(source_key, 0) + 1
        if len(selected) >= pair_topk:
            break
    return selected


def save_pair_topk_payload(record_id, gen_data_dir, syn_img_dir, payload):
    """
    保存 pair_topk_filter 最终选中的样本。

    该函数与 save_synthetic_sample 写出的 .pt 字段和 metadata 字段保持一致，
    额外增加 pair_quality_score/base_prob/concept_delta，方便后续解释 top-k 选择依据。
    """
    feat_save_path = os.path.join(gen_data_dir, f"feat_{record_id}.pt")
    torch.save({
        'feature': payload["feature"],
        'pos_embedding': payload["pos_embedding"],
        'neg_embedding': payload["neg_embedding"],
        'concept_probs_base': payload["concept_probs_base"],
        'label': payload["label"],
        'concepts': payload["concepts"],
        'concept_mask': payload["concept_mask"],
        'target_concept_proto': payload["target_concept_proto"],
        'pair_id': payload["pair_id"],
        'target_class': payload["target_class"],
        'base_class': payload["base_class"],
        'discriminative_concepts': payload["discriminative_concepts"],
        'target_prob': torch.tensor(payload["target_prob"]),
        'base_prob': torch.tensor(payload["base_prob"]),
        'concept_sim_target': torch.tensor(payload["concept_sim_target"]),
        'concept_sim_base': torch.tensor(payload["concept_sim_base"]),
        'fusion_gate': payload["fusion_gate"],
        'pair_quality_score': torch.tensor(payload["score"]),
        'filter_mode': "pair_topk",
    }, feat_save_path)

    img_save_path = os.path.join(syn_img_dir, f"syn_{record_id}.jpg")
    visual_alpha = float(payload["fusion_gate"].reshape(-1).item())
    save_visual_synthesis(
        payload["target_src_path"],
        payload["base_src_path"],
        payload["base_mask_path"],
        visual_alpha,
        img_save_path,
    )

    return {
        "feat_path": f"feat_{record_id}.pt",
        "img_path": os.path.join("synthesized_images", f"syn_{record_id}.jpg"),
        "pair_id": payload["pair_id"],
        "target_class": payload["target_class"],
        "base_class": payload["base_class"],
        "discriminative_concepts": payload["discriminative_concepts"],
        "target_prob": payload["target_prob"],
        "base_prob": payload["base_prob"],
        "concept_sim_target": payload["concept_sim_target"],
        "concept_sim_base": payload["concept_sim_base"],
        "concept_delta": payload["concept_delta"],
        "pair_quality_score": payload["score"],
        "fusion_gate": float(payload["fusion_gate"].reshape(-1).item()),
        "target_src": os.path.basename(payload["target_src_path"]),
        "base_src": os.path.basename(payload["base_src_path"]),
        "label": int(payload["label"].item()),
        "filter_mode": "pair_topk",
    }


@torch.no_grad()
def run_synthesis(args=None):
    """主函数：执行整个数据合成流程。"""
    global AUX_DIR, DATA_ROOT, IMAGE_SUBDIR, CLASS_ID_OFFSET

    if args is None:
        args = parse_args()
    args.ablation_mode = normalize_method_name(args.ablation_mode)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = load_dataset_spec(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        aux_dir=args.aux_dir,
    )
    config = load_dataset_config(spec)
    AUX_DIR = spec.aux_dir
    DATA_ROOT = config["dataset_config"]["root_dir"]
    if spec.name == "CUB-200-2011":
        IMAGE_SUBDIR = "images"
    elif spec.name == "AwA2":
        IMAGE_SUBDIR = "JPEGImages"
    elif spec.name == "7pt":
        IMAGE_SUBDIR = "images"
    elif spec.name == "PBC":
        IMAGE_SUBDIR = ""
    else:
        raise ValueError(f"不支持的数据集: {spec.name}")
    CLASS_ID_OFFSET = spec.class_id_base

    gen_data_dir = args.output_dir
    syn_img_dir = os.path.join(gen_data_dir, "synthesized_images")
    os.makedirs(gen_data_dir, exist_ok=True)
    os.makedirs(syn_img_dir, exist_ok=True)

    print(f"当前合成实验模式: {args.ablation_mode}")
    print(f"当前数据集: {spec.name}")
    print(f"当前辅助目录: {AUX_DIR}")
    print(f"当前 checkpoint: {spec.checkpoint_path}")
    print(f"合成数据输出目录: {gen_data_dir}")
    print(f"当前计算设备: {DEVICE}")
    if args.ablation_mode == "pair_topk_filter":
        print(
            "pair_topk_filter 设置: "
            f"每个 pair 保留 top-{args.pair_topk}，"
            f"score = concept_delta + {args.pair_score_target_weight} * log(target_prob) "
            f"- {args.pair_score_base_weight} * log(base_prob)"
        )
        print(
            "D-CGFS 合成核心设置: "
            f"fusion_mode={args.fusion_mode}, "
            f"synthesis_task_concept_source={args.synthesis_task_concept_source}"
        )
        if args.pair_topk_max_per_source_pair > 0:
            print(
                "pair_topk_filter 多样性限制: "
                f"同一 target/base 源图组合最多保留 {args.pair_topk_max_per_source_pair} 次"
            )
        if args.pair_adaptive_topk_mode != "none":
            print(
                "pair_topk_filter 自适应保留策略: "
                f"mode={args.pair_adaptive_topk_mode}, "
                f"min_topk={args.pair_adaptive_min_topk}"
            )
            if args.pair_adaptive_topk_mode == "target_conf":
                print(
                    "  - target_conf 参数: "
                    f"threshold={args.pair_adaptive_conf_threshold}, "
                    f"multiplier={args.pair_adaptive_conf_multiplier}"
                )
            if args.pair_adaptive_topk_mode == "source_diversity":
                print(
                    "  - source_diversity 参数: "
                    f"multiplier={args.pair_adaptive_source_multiplier}"
                )

    print("步骤 1: 正在加载预训练的 SSCBM 模型...")
    model = build_sscbm(spec, DEVICE)
    for param in model.parameters():
        param.requires_grad_(False)
    print("模型加载成功。")

    print("\n步骤 2: 正在初始化 ConceptLocator 和 ConceptGatedFuser 模块...")
    locator = ConceptLocator().to(DEVICE)
    fuser = ConceptGatedFuser().to(DEVICE)

    print("\n步骤 3: 正在准备用于合成的数据加载器...")
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = SynthesisDataset(
        os.path.join(AUX_DIR, "concept_region_mapping.csv"),
        os.path.join(AUX_DIR, "base_sample_regions.csv"),
        transform,
        samples_per_pair=args.samples_per_pair,
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    synthesized_records = []
    candidate_records = []
    # pair_topk_candidates 只在 pair_topk_filter 模式使用。
    # key 是 pair_id，value 是该 pair 内通过概念一致性过滤的候选样本。
    pair_topk_candidates = {}
    strict_keep_count = 0
    fallback_keep_count = 0
    pair_topk_keep_count = 0
    target_prob_values = []
    concept_delta_values = []
    print(f"\n步骤 4: 开始批量合成新样本，目标数量: {len(dataset)}...")

    for i, batch in enumerate(dataloader):
        # batch 中前 4 项是训练实际需要的张量；
        # 后面的路径和类别信息只用于可视化保存与 metadata 记录。
        (
            t_img, t_mask, b_img, b_mask,
            t_paths, b_paths, _, b_mask_paths,
            pair_ids, target_classes, base_classes, discriminative_concepts,
            target_proto_strings, base_proto_strings,
        ) = batch
        t_img, t_mask = t_img.to(DEVICE), t_mask.to(DEVICE)
        b_img, b_mask = b_img.to(DEVICE), b_mask.to(DEVICE)
        target_classes = target_classes.to(DEVICE)
        target_labels = target_classes.long() - CLASS_ID_OFFSET
        target_proto = build_batch_vectors(target_proto_strings, DEVICE)
        base_proto = build_batch_vectors(base_proto_strings, DEVICE)
        concept_targets, concept_mask = build_discriminative_supervision(
            discriminative_concepts,
            target_proto,
            DEVICE,
        )
        if args.ablation_mode == "all_concepts_mask":
            # all concepts mask 消融：
            # 不再只使用目标类强于基座类的判别概念 D，而是让全部概念都参与门控和监督。
            # 如果该模式不如完整 D-CGFS，说明“判别概念选择”确实有价值。
            concept_targets = target_proto.clone()
            concept_mask = torch.ones_like(target_proto, device=DEVICE)

        if args.ablation_mode == "random_mask":
            # random mask 消融：
            # 保持 mask 面积近似不变，但打乱空间位置，检验空间概念定位是否真的有效。
            t_mask = randomize_mask_like(t_mask)
            b_mask = randomize_mask_like(b_mask)

        res_t = model._forward(t_img, output_embeddings=True, output_feature_map=True)
        res_b = model._forward(b_img, output_embeddings=True, output_feature_map=True)

        # _forward(..., output_feature_map=True) 的最后一项是空间特征图:
        #   t_feat / b_feat: [batch, H, W, D]
        # res_t/res_b 中还包含正/负概念 embedding 和概念概率，用于构造 c_embedding。
        t_feat, b_feat = res_t[-1], res_b[-1]
        t_pos_emb, t_neg_emb, c_sem_t = res_t[-3], res_t[-2], res_t[0]
        b_pos_emb, b_neg_emb, c_sem_b = res_b[-3], res_b[-2], res_b[0]

        # 只在判别概念集合 D 上使用目标样本的概念 embedding。
        # 非判别概念继续保留基座样本 embedding，从而保持基座结构和背景语义。
        synthesis_pos_emb = mix_discriminative_embeddings(b_pos_emb, t_pos_emb, concept_mask)
        synthesis_neg_emb = mix_discriminative_embeddings(b_neg_emb, t_neg_emb, concept_mask)

        # D-CGFS 模块按 PyTorch 卷积习惯使用 [batch, channels, H, W]，
        # 因此需要从 [batch, H, W, D] 转成 [batch, D, H, W]。
        t_feat_permuted = t_feat.permute(0, 3, 1, 2)
        b_feat_permuted = b_feat.permute(0, 3, 1, 2)
        base_labels = base_classes.long() - CLASS_ID_OFFSET

        def evaluate_fused_features(candidate_fused_feat):
            candidate_fused_feat_in = candidate_fused_feat.permute(0, 2, 3, 1)
            candidate_concept_probs = model.predict_from_features(
                candidate_fused_feat_in,
                synthesis_pos_emb,
                synthesis_neg_emb,
                c_sem_b,
            )[0]
            candidate_task_concept_probs = build_synthesis_task_concepts(
                source=args.synthesis_task_concept_source,
                base_concept_probs=c_sem_b,
                predicted_concept_probs=candidate_concept_probs,
                target_proto=target_proto,
                concept_mask=concept_mask,
                disc_mix_weight=args.synthesis_disc_mix_weight,
            )
            _, candidate_task_logits = model.predict_from_features(
                candidate_fused_feat_in,
                synthesis_pos_emb,
                synthesis_neg_emb,
                c_sem_b,
                task_concept_probs=candidate_task_concept_probs,
            )
            candidate_task_probs = torch.softmax(candidate_task_logits, dim=1)
            candidate_target_probs = candidate_task_probs[
                torch.arange(t_img.size(0), device=DEVICE),
                target_labels,
            ]
            candidate_base_probs = candidate_task_probs[
                torch.arange(t_img.size(0), device=DEVICE),
                base_labels,
            ]
            candidate_sim_target = torch.nn.functional.cosine_similarity(
                candidate_concept_probs,
                target_proto,
                dim=1,
            )
            candidate_sim_base = torch.nn.functional.cosine_similarity(
                candidate_concept_probs,
                base_proto,
                dim=1,
            )
            candidate_concept_delta = candidate_sim_target - candidate_sim_base
            prob_eps = 1e-12
            candidate_quality_scores = (
                    candidate_concept_delta
                    + args.pair_score_target_weight * torch.log(candidate_target_probs.clamp_min(prob_eps))
                    - args.pair_score_base_weight * torch.log(candidate_base_probs.clamp_min(prob_eps))
            )
            return {
                "fused_feat_in": candidate_fused_feat_in,
                "concept_probs": candidate_concept_probs,
                "target_probs": candidate_target_probs,
                "base_probs": candidate_base_probs,
                "sim_target": candidate_sim_target,
                "sim_base": candidate_sim_base,
                "concept_delta": candidate_concept_delta,
                "pair_quality_scores": candidate_quality_scores,
            }

        # 1. ConceptLocator: 从目标类样本中提取 mask 覆盖区域的概念特征。
        # old_spatial 是当前论文主方法；aligned_pool 仅保留为轻量消融选项。
        target_localized_feat, target_pooled_feat = locator(t_feat_permuted, t_mask)
        target_concept_feat = (
            target_localized_feat
            if args.fusion_mode == "old_spatial"
            else target_pooled_feat
        )

        # 2. ConceptGatedFuser: 用判别概念差异生成门控 G_D，再融合目标/基座特征。
        if args.ablation_mode == "fixed_alpha":
            fused_feat, fusion_gate = fixed_alpha_fusion(
                b_feat_permuted,
                b_mask,
                target_localized_feat,
                args.fixed_alpha,
            )
        else:
            fused_feat, fusion_gate = fuser(
                b_feat_permuted,
                b_mask,
                target_concept_feat,
                target_proto,
                base_proto,
                concept_mask,
            )
        evaluated = evaluate_fused_features(fused_feat)
        fused_feat_in = evaluated["fused_feat_in"]
        concept_probs = evaluated["concept_probs"]
        target_probs = evaluated["target_probs"]
        base_probs = evaluated["base_probs"]
        sim_target = evaluated["sim_target"]
        sim_base = evaluated["sim_base"]
        concept_delta = evaluated["concept_delta"]
        pair_quality_scores = evaluated["pair_quality_scores"]

        # 第一层：目标类先验标签。
        # 合成过程是把目标类判别概念注入基座类结构中，因此保存标签固定为目标类，而不是 argmax。
        synthetic_labels = target_labels

        target_prob_values.extend(target_probs.detach().cpu().tolist())
        concept_delta_values.extend(concept_delta.detach().cpu().tolist())

        # 分开记录两层过滤是否通过，方便定位样本被拒绝的主因。
        # 对应消融模式关闭某一层过滤时，该层默认视为通过。
        pass_conf_mask = torch.ones_like(target_probs, dtype=torch.bool)
        if args.ablation_mode != "no_conf_filter":
            pass_conf_mask = target_probs > args.target_conf_threshold
        pass_concept_mask = torch.ones_like(target_probs, dtype=torch.bool)
        if args.ablation_mode != "no_concept_filter":
            pass_concept_mask = sim_target > sim_base + args.concept_margin
        keep_mask = pass_conf_mask & pass_concept_mask

        # 保存所有候选样本的诊断信息。这里不保存大张量，只保存足够解释过滤行为的标量。
        for row_idx in range(t_img.size(0)):
            candidate_records.append({
                "candidate_id": len(candidate_records),
                "pair_id": int(pair_ids[row_idx].item()),
                "target_class": int(target_classes[row_idx].item()),
                "base_class": int(base_classes[row_idx].item()),
                "target_prob": float(target_probs[row_idx].item()),
                "base_prob": float(base_probs[row_idx].item()),
                "concept_sim_target": float(sim_target[row_idx].item()),
                "concept_sim_base": float(sim_base[row_idx].item()),
                "concept_delta": float(concept_delta[row_idx].item()),
                "pair_quality_score": float(pair_quality_scores[row_idx].item()),
                "fusion_gate": float(fusion_gate[row_idx].reshape(-1).item()),
                "pass_conf_filter": bool(pass_conf_mask[row_idx].item()),
                "pass_concept_filter": bool(pass_concept_mask[row_idx].item()),
                "pass_strict_filter": bool(keep_mask[row_idx].item()),
                "target_src": os.path.basename(t_paths[row_idx]),
                "base_src": os.path.basename(b_paths[row_idx]),
            })

        if args.ablation_mode == "pair_topk_filter":
            # pair_topk_filter 不在 batch 内立即保存样本。
            # 它先保留所有“概念一致”的候选，再在每个 pair 内按质量分数选 top-k。
            # 这样可以避免 no_conf_filter 完全放开导致某些 pair 样本过多，也避免绝对
            # target_prob 阈值导致弱势类样本几乎全被过滤。
            selected_indices = pass_concept_mask.nonzero(as_tuple=False).reshape(-1).tolist()
            for j in selected_indices:
                pair_id = int(pair_ids[j].item())
                pair_topk_candidates.setdefault(pair_id, []).append(build_pair_topk_payload(
                    fused_feat_in=fused_feat_in,
                    b_pos_emb=synthesis_pos_emb,
                    b_neg_emb=synthesis_neg_emb,
                    c_sem_b=c_sem_b,
                    synthetic_labels=synthetic_labels,
                    concept_targets=concept_targets,
                    concept_mask=concept_mask,
                    target_proto=target_proto,
                    pair_ids=pair_ids,
                    target_classes=target_classes,
                    base_classes=base_classes,
                    discriminative_concepts=discriminative_concepts,
                    target_probs=target_probs,
                    base_probs=base_probs,
                    sim_target=sim_target,
                    sim_base=sim_base,
                    concept_delta=concept_delta,
                    pair_quality_scores=pair_quality_scores,
                    fusion_gate=fusion_gate,
                    t_paths=t_paths,
                    b_paths=b_paths,
                    b_mask_paths=b_mask_paths,
                    sample_idx=j,
                ))
            continue

        selected_indices = keep_mask.nonzero(as_tuple=False).reshape(-1).tolist()
        selected_modes = ["strict"] * len(selected_indices)

        # 如果严格过滤暂时没有样本通过，并且总保留数量还没有达到最低训练需求，
        # 就按质量分数补充当前 batch 中最好的一个样本。
        # 质量分数同时考虑目标类概率和目标/基座概念相似度差。
        if (
                args.enable_quality_fallback
                and not selected_indices
                and len(synthesized_records) < args.min_keep_samples
        ):
            quality_score = target_probs + 0.1 * concept_delta
            best_idx = int(torch.argmax(quality_score).item())
            selected_indices = [best_idx]
            selected_modes = ["quality_fallback"]

        for j, filter_mode in zip(selected_indices, selected_modes):
            record_id = len(synthesized_records)
            synthesized_records.append(save_synthetic_sample(
                record_id=record_id,
                gen_data_dir=gen_data_dir,
                syn_img_dir=syn_img_dir,
                fused_feat_in=fused_feat_in,
                b_pos_emb=synthesis_pos_emb,
                b_neg_emb=synthesis_neg_emb,
                c_sem_b=c_sem_b,
                synthetic_labels=synthetic_labels,
                concept_targets=concept_targets,
                concept_mask=concept_mask,
                target_proto=target_proto,
                pair_ids=pair_ids,
                target_classes=target_classes,
                base_classes=base_classes,
                discriminative_concepts=discriminative_concepts,
                target_probs=target_probs,
                sim_target=sim_target,
                sim_base=sim_base,
                fusion_gate=fusion_gate,
                t_paths=t_paths,
                b_paths=b_paths,
                b_mask_paths=b_mask_paths,
                sample_idx=j,
                filter_mode=filter_mode,
            ))
            if filter_mode == "strict":
                strict_keep_count += 1
            else:
                fallback_keep_count += 1

    if args.ablation_mode == "pair_topk_filter":
        print("\n步骤 5: 正在按 pair 内质量分数选择 top-k 合成样本...")
        for pair_id in sorted(pair_topk_candidates):
            candidates = pair_topk_candidates[pair_id]
            pair_topk = compute_adaptive_pair_topk(candidates, args)
            selected = select_pair_topk_candidates(
                candidates,
                pair_topk,
                args.pair_topk_max_per_source_pair,
            )
            print(
                f"  - pair_id={pair_id}: 候选 {len(candidates)} 个，"
                f"adaptive_topk={pair_topk}，保留 {len(selected)} 个"
            )
            for payload in selected:
                record_id = len(synthesized_records)
                synthesized_records.append(save_pair_topk_payload(
                    record_id=record_id,
                    gen_data_dir=gen_data_dir,
                    syn_img_dir=syn_img_dir,
                    payload=payload,
                ))
                pair_topk_keep_count += 1

    metadata_columns = [
        "feat_path",
        "img_path",
        "pair_id",
        "target_class",
        "base_class",
        "discriminative_concepts",
        "target_prob",
        "base_prob",
        "concept_sim_target",
        "concept_sim_base",
        "concept_delta",
        "pair_quality_score",
        "fusion_gate",
        "target_src",
        "base_src",
        "label",
        "filter_mode",
    ]
    pd.DataFrame(synthesized_records, columns=metadata_columns).to_csv(
        os.path.join(gen_data_dir, "synthesized_metadata.csv"),
        index=False,
    )
    summarize_candidate_diagnostics(candidate_records, gen_data_dir)
    print(f"\n过滤后保留合成样本数量: {len(synthesized_records)}/{len(dataset)}")
    print(f"  - strict: {strict_keep_count}")
    print(f"  - quality_fallback: {fallback_keep_count}")
    print(f"  - pair_topk: {pair_topk_keep_count}")
    summarize_values("目标类概率 target_prob", target_prob_values)
    summarize_values("概念相似度差 sim_target - sim_base", concept_delta_values)
    if not synthesized_records:
        print("警告：没有合成样本通过过滤。建议降低 --target-conf-threshold 或 --concept-margin 后重新运行。")
    print(f"\n成功！步骤 2 完成。合成数据已保存至: {gen_data_dir}")
    print(f"可视化的合成图片已保存至: {syn_img_dir}")


if __name__ == "__main__":
    run_synthesis()
