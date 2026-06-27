# ==========================================================================================
# SSCBM + D-CGFS 研究项目 - 步骤 3: 混合数据平衡微调
#
# 核心目标：
# 利用步骤 2 生成的合成数据，对预训练的 SSCBM 模型进行平衡微调 (Fine-tuning)，
# 以增强模型对少数类的识别能力。
#
# 工作流程：
# 1. 加载在步骤 2 中生成的合成特征数据集。
# 2. 同时加载原始的训练数据集。
# 3. 采用“混合训练 (Hybrid Training)”策略：在每个训练批次中，一部分数据来自原始
#    数据集（通过完整的模型骨干网），另一部分数据来自合成数据集（直接输入特征）。
# 4. 冻结 SSCBM 模型的特征提取器 (Backbone)，只对后续的分类层和概念层进行训练。
#    - 目的：保留模型强大的通用特征提取能力，避免其在微调过程中被破坏，同时
#      加快训练速度，降低过拟合风险。
# 5. 计算来自原始数据和合成数据的损失，并将它们加权合并，共同用于模型优化。
#    - 目的：确保模型在学习新知识（识别合成样本）的同时，不会忘记旧知识（识别
#      原始样本），达到“温故知新”的效果。
#    - 对合成样本，使用问题三和问题五方案：
#      L_syn = L_task(y_syn, y_target) + lambda_c * L_concept(c_syn, c_target^D)
#      其中概念损失只在判别概念集合 D 上计算。
#      同时加入原型约束 L_proto = ||c_syn - p_target||_2，使合成样本概念预测接近目标类概念原型。
# 6. 按问题八补强论文级损失：
#    - teacher-student consistency：防止微调后模型遗忘原始 SSCBM 的稳定决策。
#    - distribution regularization：约束合成 batch 的预测分布接近目标标签分布，避免合成监督坍缩。
# 7. 训练完成后，保存微调后的新模型权重，供最终评估使用。
#
# 作者：[肖凡]
# 日期：[2026年4月20日]
# ==========================================================================================

import torch
import os
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dataset_specs import build_sscbm, load_dataset_config, load_dataset_spec
from dcgfs_config import (
    BASE_PRESERVATION_WEIGHT,
    METHOD_ACRONYM,
    SYN_CONCEPT_LOSS_WEIGHT,
    SYN_PROTO_LOSS_WEIGHT,
    TEACHER_CONSISTENCY_WEIGHT,
    DISTRIBUTION_REG_WEIGHT,
    TEACHER_TEMPERATURE,
    normalize_method_name,
)

# --- 1. 全局配置 ---
# 步骤 2 生成的合成数据目录
GEN_DATA_DIR = "generated_data"
# 预训练 SSCBM 模型权重路径
CHECKPOINT_PATH = "checkpoints/CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt"
# 运行设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIR_CSV_PATH = "data/D-CGFS_Auxiliary/target_base_pairs.csv"
N_CLASSES = 200
CLASS_ID_OFFSET = 1


def parse_args():
    """
    解析问题六实验所需的训练阶段参数。

    method 用于强 baseline 对比：
        - dcgfs: 完整 D-CGFS，使用原始数据 + 合成数据混合训练。
          历史别名 cgens 仍可识别，但内部会自动映射为 dcgfs。
        - sscbm_finetune: 只用原始数据微调，作为公平的 fine-tuning baseline。
        - oversampling: 对训练集按类别重采样，作为经典不平衡学习 baseline。
        - reweighting/class_balanced_loss: 对分类损失加类别权重。
        - feature_mixup: 在 SSCBM 的空间特征和概念上下文上做 mixup。

    disable_* 参数用于 D-CGFS 的训练损失消融：
        - disable_syn_concept_loss: 去掉判别概念一致性损失。
        - disable_syn_proto_loss: 去掉目标类概念原型约束。
        - disable_teacher_consistency: 去掉 teacher-student consistency。
        - disable_distribution_reg: 去掉合成预测分布正则。

    base preservation 参数用于 pair_topk_filter 后续补强：
        - enable_base_preservation: 对自动基座类原始样本加额外 teacher 约束。
        - base_preservation_weight: 该约束的权重。

    synthetic_loss_weight 用于诊断当前 D-CGFS target accuracy 低于 strong baseline 的原因：
        - 默认仍为 0.5，保持当前论文主配置不变。
        - 若调高该权重，模型会更重视合成样本的目标类任务损失和概念约束，
          用于验证“合成样本对目标类分类边界推动不足”这一假设。

    enable_class_balanced_task_loss 用于把 strong baseline 中有效的类别重加权思想
    融合进 D-CGFS，而不是只作为外部对照。它会同时影响原始样本任务损失和合成样本
    任务损失，是当前最值得验证的 target accuracy 补强方向。

    enable-target-class-weighted-loss 是比 class-balanced 更贴合 D-CGFS 的补强：
    它只给 find_weak_classes.py 自动选出的目标类加权，而不抬高所有少样本类别。
    这样可以验证“目标类边界推动不足”是否能被定向任务损失缓解，同时避免普通
    class-balanced loss 把优化目标带偏到全局少数类重加权。

    synthetic_task_concept_source 控制合成样本任务 logits 使用哪一组概念概率：
        - base: 复现旧实现，使用基座样本原始概念概率。
        - predicted: 使用合成特征预测出的概念概率。
        - target_proto: 使用目标类概念原型，直接强化目标语义。
        - target_disc_mix: 只在判别概念上使用目标原型，其余保留基座概念概率。
        - predicted_disc_mix: 只在判别概念上使用合成特征预测概念，其余保留基座概念概率。
    """
    parser = argparse.ArgumentParser(description=f"SSCBM + {METHOD_ACRONYM} 平衡训练和顶会级实验")
    parser.add_argument("--dataset", default="CUB-200-2011", choices=["CUB-200-2011", "AwA2", "PBC", "7pt"])
    parser.add_argument("--checkpoint", default=None, help="Baseline SSCBM checkpoint；不填使用数据集默认路径。")
    parser.add_argument("--pair-csv", default=None, help="target_base_pairs.csv 路径；不填使用数据集默认路径。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子；控制训练数据划分、采样和 mixup。")
    parser.add_argument("--method", default="dcgfs", choices=[
        "dcgfs",
        "cgens",
        "sscbm_finetune",
        "oversampling",
        "reweighting",
        "class_balanced_loss",
        "feature_mixup",
    ])
    parser.add_argument("--gen-data-dir", default=GEN_DATA_DIR)
    parser.add_argument("--metadata-name", default="synthesized_metadata.csv")
    parser.add_argument("--output-checkpoint", default="checkpoints/best_sscbm_dcgfs_hybrid.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="仅用于快速冒烟测试；>0 时每个 epoch 最多训练指定 batch 数，正式实验保持 0。",
    )
    parser.add_argument("--batch-size-syn", type=int, default=16)
    parser.add_argument("--labeled-ratio", type=float, default=0.1)
    parser.add_argument("--class-attr-data-dir", default="class_attr_data_10")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disable-syn-concept-loss", action="store_true")
    parser.add_argument("--disable-syn-proto-loss", action="store_true")
    parser.add_argument("--disable-teacher-consistency", action="store_true")
    parser.add_argument("--disable-distribution-reg", action="store_true")
    parser.add_argument("--teacher-consistency-weight", type=float, default=TEACHER_CONSISTENCY_WEIGHT)
    parser.add_argument("--distribution-reg-weight", type=float, default=DISTRIBUTION_REG_WEIGHT)
    parser.add_argument("--teacher-temperature", type=float, default=TEACHER_TEMPERATURE)
    parser.add_argument(
        "--enable-base-preservation",
        action="store_true",
        help="对 target-base pair 中的基座类原始样本施加额外 teacher 约束，缓解 base accuracy 下降。",
    )
    parser.add_argument("--base-preservation-weight", type=float, default=BASE_PRESERVATION_WEIGHT)
    parser.add_argument(
        "--synthetic-loss-weight",
        type=float,
        default=0.5,
        help="D-CGFS 中合成样本总损失 loss_s 的权重；默认 0.5 保持当前主实验配置。",
    )
    parser.add_argument(
        "--enable-class-balanced-task-loss",
        action="store_true",
        help="在 D-CGFS 中启用类别均衡任务损失，用于吸收 reweighting/class-balanced baseline 的有效成分。",
    )
    parser.add_argument(
        "--enable-target-class-weighted-loss",
        action="store_true",
        help="只对自动选择的目标弱势类加任务损失权重，避免退化成普通全局 class-balanced loss。",
    )
    parser.add_argument(
        "--target-class-loss-weight",
        type=float,
        default=2.0,
        help="启用 target-class weighted loss 时，自动目标类的任务损失权重。",
    )
    parser.add_argument(
        "--synthetic-task-concept-source",
        choices=["base", "predicted", "target_proto", "target_disc_mix", "predicted_disc_mix"],
        default="base",
        help="合成样本任务 logits 的概念概率来源；默认 base 复现旧实现。",
    )
    parser.add_argument(
        "--synthetic-disc-mix-weight",
        type=float,
        default=1.0,
        help=(
            "判别概念任务概率的软注入权重，仅作用于 target_disc_mix/predicted_disc_mix。"
            "1.0 表示完全使用目标/合成判别概念，0.0 退化为基座概念概率。"
        ),
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    return parser.parse_args()


class SyntheticFeatureDataset(Dataset):
    """
    自定义数据集，专门用于加载步骤 2 生成的 .pt 特征文件。
    它直接加载特征，跳过了图像预处理和模型骨干网的计算，效率更高。
    """
    def __init__(self, metadata_csv, gen_data_dir=GEN_DATA_DIR):
        self.metadata = pd.read_csv(metadata_csv)
        self.gen_data_dir = gen_data_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.gen_data_dir, self.metadata.iloc[idx]['feat_path']))
        # 返回 D-CGFS 流程所需的所有关键信息
        return (
            data['feature'],
            data['label'],
            data['concepts'],
            data['concept_mask'],
            data['pos_embedding'],
            data['neg_embedding'],
            data['concept_probs_base'],
            data['target_concept_proto'],
        )


def get_class_weights_from_dataset(dataset, n_classes=200):
    """
    根据原始训练集类别频次构造类别权重。

    少数类样本越少，权重越大；权重最后归一到均值约为 1，避免整体 loss 尺度剧烈变化。
    这对应问题六中的 reweighting / class-balanced loss baseline。
    """
    if hasattr(dataset, "data"):
        labels = [int(item["class_label"]) for item in dataset.data]
    elif hasattr(dataset, "ds") and hasattr(dataset.ds, "labels"):
        labels = [int(label) for label in dataset.ds.labels]
    else:
        raise ValueError("当前数据集不支持 class-balanced loss：无法读取训练标签。")
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_oversampling_loader(train_loader):
    """
    构造类别均衡重采样 DataLoader。

    oversampling baseline 不改变模型和损失，只改变原始训练样本出现频率，
    用来证明 D-CGFS 的提升不是普通重复采样就能替代的。
    """
    dataset = train_loader.dataset
    if hasattr(dataset, "data"):
        labels = [int(item["class_label"]) for item in dataset.data]
    elif hasattr(dataset, "ds") and hasattr(dataset.ds, "labels"):
        labels = [int(label) for label in dataset.ds.labels]
    else:
        raise ValueError("当前数据集不支持 oversampling：无法读取训练标签。")
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    sample_weights = [1.0 / max(counts[label], 1.0) for label in labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=train_loader.num_workers,
    )


def soft_cross_entropy(logits, soft_targets):
    """支持 mixup 软标签的交叉熵。"""
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def load_base_labels_from_pairs(pair_csv_path=PAIR_CSV_PATH):
    """
    读取自动 target-base pair 中的基座类标签。

    CUB 的 pair 文件使用 1-based 类别 id，AwA2 使用 0-based 类别 id；
    CLASS_ID_OFFSET 会把它们统一转换成模型训练标签。base preservation 只作用于这些
    基座类的原始训练样本，避免目标类增强把基座类决策边界整体推坏。
    """
    if not os.path.exists(pair_csv_path):
        raise FileNotFoundError(f"未找到 {pair_csv_path}，无法启用 base preservation。")
    pair_df = pd.read_csv(pair_csv_path)
    if "base_class_id" not in pair_df.columns:
        raise ValueError(f"{pair_csv_path} 缺少 base_class_id 列，无法启用 base preservation。")
    base_labels = sorted({int(class_id) - CLASS_ID_OFFSET for class_id in pair_df["base_class_id"].dropna().tolist()})
    if not base_labels:
        raise ValueError(f"{pair_csv_path} 中没有可用基座类，无法启用 base preservation。")
    return torch.tensor(base_labels, dtype=torch.long, device=DEVICE)


def load_target_labels_from_pairs(pair_csv_path=PAIR_CSV_PATH):
    """
    读取自动 target-base pair 中的目标类标签。

    与 base preservation 一样，该函数会根据当前数据集的 CLASS_ID_OFFSET 把
    pair 文件中的类别 id 转成训练标签。它服务于 target-class weighted loss：
    只给这些自动识别出的弱势目标类加权，保持 D-CGFS 的优化目标聚焦。
    """
    if not os.path.exists(pair_csv_path):
        raise FileNotFoundError(f"未找到 {pair_csv_path}，无法启用 target-class weighted loss。")
    pair_df = pd.read_csv(pair_csv_path)
    if "target_class_id" not in pair_df.columns:
        raise ValueError(f"{pair_csv_path} 缺少 target_class_id 列，无法启用 target-class weighted loss。")
    target_labels = sorted({int(class_id) - CLASS_ID_OFFSET for class_id in pair_df["target_class_id"].dropna().tolist()})
    if not target_labels:
        raise ValueError(f"{pair_csv_path} 中没有可用目标类，无法启用 target-class weighted loss。")
    return torch.tensor(target_labels, dtype=torch.long, device=DEVICE)


def build_target_class_weights(target_labels, target_weight, n_classes=None):
    """
    构造只强调自动目标弱势类的任务损失权重。

    所有非目标类权重保持 1.0，目标类权重设为 target_weight。
    这比全局 class-balanced loss 更温和，因为它不会同时改变所有类别的损失比例；
    对当前问题来说，它更直接地检验 D-CGFS 是否只是“目标类边界推动不够”。
    """
    n_classes = n_classes or N_CLASSES
    weights = torch.ones(n_classes, dtype=torch.float32, device=DEVICE)
    weights[target_labels] = float(target_weight)
    return weights


def compute_base_preservation_loss(student_logits, teacher_logits, labels, base_labels, temperature):
    """
    计算基座类保护损失。

    只挑出当前原始 batch 中属于自动基座类集合的样本，然后让 student 的输出分布
    贴近冻结 baseline teacher。它比全局 teacher consistency 更聚焦，因为 pair_topk
    的主要副作用集中在 base class accuracy 下降。
    """
    if base_labels is None or base_labels.numel() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    is_base_sample = (labels[:, None] == base_labels[None, :]).any(dim=1)
    if not bool(is_base_sample.any().item()):
        return torch.tensor(0.0, device=student_logits.device)

    teacher_prob = torch.softmax(teacher_logits[is_base_sample] / temperature, dim=1)
    student_log_prob = torch.log_softmax(student_logits[is_base_sample] / temperature, dim=1)
    return (
        torch.nn.functional.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        * (temperature ** 2)
    )


def run_feature_mixup_loss(model, imgs, labels, alpha):
    """
    在 SSCBM 特征空间执行 mixup baseline。

    为了和 D-CGFS 的“特征空间合成”公平对比，这里不是简单图像级 MixUp，
    而是在 SSCBM 已提取出的空间特征图、正/负概念 embedding、概念概率上做线性混合。
    它不使用概念 mask、target-base selection 或概念门控，因此是一个强但不具备
    D-CGFS 语义约束的 baseline。
    """
    if imgs.size(0) < 2:
        outputs = model(imgs)
        return torch.nn.functional.cross_entropy(outputs[3], labels)

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(imgs.size(0), device=imgs.device)

    res = model._forward(imgs, output_embeddings=True, output_feature_map=True)
    c_sem = res[0]
    pos_emb = res[-3]
    neg_emb = res[-2]
    feat = res[-1]

    mixed_feat = lam * feat + (1 - lam) * feat[perm]
    mixed_pos = lam * pos_emb + (1 - lam) * pos_emb[perm]
    mixed_neg = lam * neg_emb + (1 - lam) * neg_emb[perm]
    mixed_c_sem = lam * c_sem + (1 - lam) * c_sem[perm]

    _, logits = model.predict_from_features(mixed_feat, mixed_pos, mixed_neg, mixed_c_sem)
    hard_onehot = torch.nn.functional.one_hot(labels, num_classes=N_CLASSES).float()
    soft_targets = lam * hard_onehot + (1 - lam) * hard_onehot[perm]
    return soft_cross_entropy(logits, soft_targets)


def predict_synthetic_with_task_concepts(
        model,
        feat_s,
        pos_emb_s,
        neg_emb_s,
        concept_probs_base_s,
        target_proto_s,
        concept_mask_s,
        task_concept_source,
        disc_mix_weight=1.0,
):
    """
    预测合成特征的概念概率和任务 logits。

    旧实现中，合成样本的任务 logits 使用 base concept probabilities 组合正/负概念
    embedding。这会让合成样本任务监督更像是在把“基座概念表示”强行映射到目标标签，
    合成特征本身对分类边界的推动偏弱。

    这里保留旧实现作为 base，同时新增更目标导向的选项：
        - predicted: 用合成空间特征预测出的概念概率构造任务 logits。
        - target_proto: 用目标类概念原型构造任务 logits。
        - target_disc_mix: 只在判别概念 D 上使用目标原型，非判别概念仍保留基座概率。
        - predicted_disc_mix: 只在判别概念 D 上使用合成预测概念，非判别概念仍保留基座概率。

    这几个选项用于验证“合成监督是否真正把目标概念语义注入分类头”。
    """
    c_probs_s, _ = model.predict_from_features(
        feat_s,
        pos_emb_s,
        neg_emb_s,
        concept_probs_base_s,
    )

    mix_weight = float(max(0.0, min(1.0, disc_mix_weight)))
    if task_concept_source == "base":
        task_concept_probs = concept_probs_base_s
    elif task_concept_source == "predicted":
        task_concept_probs = c_probs_s.detach()
    elif task_concept_source == "target_proto":
        task_concept_probs = target_proto_s
    elif task_concept_source == "target_disc_mix":
        mixed_disc_probs = (
                concept_probs_base_s * (1.0 - mix_weight)
                + target_proto_s * mix_weight
        )
        task_concept_probs = (
                concept_probs_base_s * (1.0 - concept_mask_s)
                + mixed_disc_probs * concept_mask_s
        )
    elif task_concept_source == "predicted_disc_mix":
        mixed_disc_probs = (
                concept_probs_base_s * (1.0 - mix_weight)
                + c_probs_s.detach() * mix_weight
        )
        task_concept_probs = (
                concept_probs_base_s * (1.0 - concept_mask_s)
                + mixed_disc_probs * concept_mask_s
        )
    else:
        raise ValueError(f"未知 synthetic_task_concept_source: {task_concept_source}")

    synthetic_embedding = model.compute_concept_embedding(
        pos_emb_s,
        neg_emb_s,
        task_concept_probs,
    )
    _, t_logits_s = model.predict_task_from_concept_embedding(synthetic_embedding)
    return c_probs_s, t_logits_s


def hybrid_balance_train(args=None):
    """主函数：执行混合数据平衡微调流程。"""
    global CHECKPOINT_PATH, PAIR_CSV_PATH, N_CLASSES, CLASS_ID_OFFSET

    if args is None:
        args = parse_args()
    args.method = normalize_method_name(args.method)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = load_dataset_spec(dataset=args.dataset, checkpoint_path=args.checkpoint)
    CHECKPOINT_PATH = spec.checkpoint_path
    PAIR_CSV_PATH = args.pair_csv or os.path.join(spec.aux_dir, "target_base_pairs.csv")
    N_CLASSES = spec.n_classes
    CLASS_ID_OFFSET = spec.class_id_base
    print(f"当前训练设备: {DEVICE}")
    print(f"当前数据集: {spec.name}")
    print(f"当前 baseline checkpoint: {CHECKPOINT_PATH}")
    print(f"当前 target-base pair 文件: {PAIR_CSV_PATH}")

    # 1. 加载预训练的 SSCBM 模型
    print("步骤 1: 正在加载预训练的 SSCBM 模型...")
    model = build_sscbm(spec, DEVICE)
    print("模型加载成功。")

    # 冻结的 teacher model 用于问题八提出的 teacher-student consistency。
    # teacher 保持原始 SSCBM 权重，student 是正在微调的模型。
    # 该损失只约束原始训练样本上的任务输出，目的是防止 D-CGFS 只追求弱势类提升而破坏整体决策边界。
    teacher_model = None
    need_teacher_model = (
            args.method == "dcgfs"
            and (not args.disable_teacher_consistency or args.enable_base_preservation)
    )
    if need_teacher_model:
        teacher_model = build_sscbm(spec, DEVICE)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print(f"已加载冻结 teacher model，用于 {METHOD_ACRONYM} 一致性正则。")

    # base preservation 只在完整 D-CGFS 训练中生效。
    # 它读取 find_weak_classes.py 自动选择出的基座类，而不是硬编码某个类别。
    base_labels_for_preservation = None
    if args.method == "dcgfs" and args.enable_base_preservation:
        base_labels_for_preservation = load_base_labels_from_pairs(PAIR_CSV_PATH)
        print(
            "已启用 base preservation，保护基座类标签(0-based): "
            f"{base_labels_for_preservation.detach().cpu().tolist()}，"
            f"权重={args.base_preservation_weight}"
        )

    # 2. 冻结骨干网络 (Backbone)
    # 这是微调的关键策略：我们相信模型的特征提取能力已经足够好，只需要调整后续的
    # 决策层来适应新的数据分布。
    print("\n步骤 2: 正在冻结模型的特征提取器 (pre_concept_model)...")
    for param in model.pre_concept_model.parameters():
        param.requires_grad = False
    print("特征提取器已冻结，只训练后续层。")

    # 3. 准备原始数据和合成数据的加载器
    print("\n步骤 3: 正在准备原始训练集和合成特征集...")
    config = load_dataset_config(spec)
    if spec.name == "CUB-200-2011":
        config["dataset_config"]["class_attr_data_dir"] = args.class_attr_data_dir
    config["dataset_config"]["num_workers"] = 0
    # 复用项目原有的数据加载逻辑，获取原始训练集
    train_loader_orig, _, _, _, _ = spec.data_module.generate_data(
        config=config['dataset_config'], seed=args.seed, labeled_ratio=args.labeled_ratio  # 模拟半监督场景
    )

    if args.method == "oversampling":
        train_loader_orig = build_oversampling_loader(train_loader_orig)

    # 只有完整 D-CGFS 才需要加载步骤 2 生成的合成特征数据集。
    train_loader_syn = None
    if args.method == "dcgfs":
        metadata_csv = os.path.join(args.gen_data_dir, args.metadata_name)
        syn_dataset = SyntheticFeatureDataset(metadata_csv, gen_data_dir=args.gen_data_dir)
        train_loader_syn = DataLoader(syn_dataset, batch_size=args.batch_size_syn, shuffle=True)
    print("数据加载器准备完毕。")

    # 4. 定义优化器和损失函数
    # 只将需要更新的参数（即未被冻结的参数）传入优化器。
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    use_class_balanced_task_loss = (
            args.method in {"reweighting", "class_balanced_loss"}
            or (args.method == "dcgfs" and args.enable_class_balanced_task_loss)
    )
    use_target_class_weighted_loss = (
            args.method == "dcgfs"
            and args.enable_target_class_weighted_loss
    )
    if use_target_class_weighted_loss and use_class_balanced_task_loss:
        raise ValueError("target-class weighted loss 和 class-balanced task loss 不能同时启用，避免损失权重含义混乱。")

    if use_class_balanced_task_loss:
        class_weights = get_class_weights_from_dataset(train_loader_orig.dataset).to(DEVICE)
        criterion_task = torch.nn.CrossEntropyLoss(weight=class_weights)
        if args.method == "dcgfs":
            print("已启用 D-CGFS class-balanced task loss：原始样本和合成样本任务损失都会使用类别权重。")
    elif use_target_class_weighted_loss:
        target_labels_for_weighting = load_target_labels_from_pairs(PAIR_CSV_PATH)
        target_class_weights = build_target_class_weights(
            target_labels=target_labels_for_weighting,
            target_weight=args.target_class_loss_weight,
        )
        criterion_task = torch.nn.CrossEntropyLoss(weight=target_class_weights)
        print(
            "已启用 D-CGFS target-class weighted loss："
            f"目标类标签(0-based)={target_labels_for_weighting.detach().cpu().tolist()}，"
            f"权重={args.target_class_loss_weight}"
        )
    else:
        criterion_task = torch.nn.CrossEntropyLoss()
    # predict_from_features 返回的是已经 sigmoid 后的概念概率，因此这里使用 BCELoss。
    # reduction='none' 是为了只在判别概念 mask=1 的位置计算损失。
    criterion_concept = torch.nn.BCELoss(reduction='none')

    print("\n步骤 4: 开始混合平衡训练...")
    model.train()  # 确保模型处于训练模式

    for epoch in range(args.epochs):  # 微调通常不需要太多轮次
        total_loss = 0
        num_batches = 0
        # 完整 D-CGFS 使用原始数据和合成数据混合训练；其他 baseline 只用原始数据。
        iterator = (
            zip(train_loader_orig, train_loader_syn)
            if args.method == "dcgfs"
            else ((batch_orig, None) for batch_orig in train_loader_orig)
        )
        for batch_idx, (batch_orig, batch_syn) in enumerate(iterator):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            optimizer.zero_grad()

            # --- a. 处理原始数据 (经过完整的模型路径) ---
            # batch_orig 的格式: (img, label, concept, is_labeled, nbr_concept, nbr_weight)
            imgs_o, labels_o = batch_orig[0].to(DEVICE), batch_orig[1].to(DEVICE)
            if args.method == "feature_mixup":
                loss_o = run_feature_mixup_loss(model, imgs_o, labels_o, args.mixup_alpha)
            else:
                outputs_o = model(imgs_o)
                # outputs_o[3] 是任务预测的 logits
                loss_o = criterion_task(outputs_o[3], labels_o)

            loss_teacher_consistency = torch.tensor(0.0, device=DEVICE)
            loss_base_preservation = torch.tensor(0.0, device=DEVICE)
            if args.method == "dcgfs" and teacher_model is not None:
                # teacher-student consistency:
                # student 在原始样本上的输出分布应接近冻结 baseline teacher。
                # 使用温度缩放 KL，比直接约束 hard label 更平滑。
                with torch.no_grad():
                    teacher_logits = teacher_model(imgs_o)[3]
                student_logits = outputs_o[3]
                temperature = args.teacher_temperature
                if not args.disable_teacher_consistency:
                    teacher_prob = torch.softmax(teacher_logits / temperature, dim=1)
                    student_log_prob = torch.log_softmax(student_logits / temperature, dim=1)
                    loss_teacher_consistency = (
                        torch.nn.functional.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
                        * (temperature ** 2)
                    )
                if args.enable_base_preservation:
                    # 额外聚焦基座类样本。全局 teacher consistency 会平均到所有类别，
                    # 对 Herring_Gull 等受影响基座类的保护力度不够，因此这里单独加权。
                    loss_base_preservation = compute_base_preservation_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        labels=labels_o,
                        base_labels=base_labels_for_preservation,
                        temperature=temperature,
                    )

            # --- b. 处理合成数据 (直接从特征开始) ---
            if args.method != "dcgfs":
                loss = loss_o
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                continue

            (
                feat_s,
                labels_s,
                concepts_s,
                concept_mask_s,
                pos_emb_s,
                neg_emb_s,
                concept_probs_base_s,
                target_proto_s,
            ) = batch_syn
            feat_s = feat_s.to(DEVICE)
            labels_s = labels_s.to(DEVICE)
            concepts_s = concepts_s.to(DEVICE).float()
            concept_mask_s = concept_mask_s.to(DEVICE).float()
            pos_emb_s = pos_emb_s.to(DEVICE)
            neg_emb_s = neg_emb_s.to(DEVICE)
            concept_probs_base_s = concept_probs_base_s.to(DEVICE)
            target_proto_s = target_proto_s.to(DEVICE).float()
            # 调用合成特征预测函数。
            # 默认 task_concept_source=base 时复现旧实现；
            # 其他选项用于让合成任务监督更直接地注入目标概念语义。
            c_probs_s, t_logits_s = predict_synthetic_with_task_concepts(
                model=model,
                feat_s=feat_s,
                pos_emb_s=pos_emb_s,
                neg_emb_s=neg_emb_s,
                concept_probs_base_s=concept_probs_base_s,
                target_proto_s=target_proto_s,
                concept_mask_s=concept_mask_s,
                task_concept_source=args.synthetic_task_concept_source,
                disc_mix_weight=args.synthetic_disc_mix_weight,
            )

            # 合成样本任务损失：标签来自目标类先验，而不是 baseline argmax。
            loss_s_task = criterion_task(t_logits_s, labels_s)

            # 合成样本概念损失：只监督目标类相对基座类更强的判别概念 D。
            # 数值保护：BCELoss 要求输入在 [0, 1]，这里避免极端概率导致 log(0)。
            c_probs_s = c_probs_s.clamp(1e-6, 1 - 1e-6)
            concept_loss_matrix = criterion_concept(c_probs_s, concepts_s)
            mask_sum = concept_mask_s.sum().clamp_min(1.0)
            loss_s_concept = (concept_loss_matrix * concept_mask_s).sum() / mask_sum

            # 原型约束融合：让合成样本的整体概念预测靠近目标类概念原型。
            # 这对应问题五中的 L_proto = ||c_syn - p_target||_2，进一步区别于普通 feature mixup。
            loss_s_proto = torch.nn.functional.mse_loss(c_probs_s, target_proto_s)

            # Distribution regularization:
            # 合成 batch 的平均预测分布应接近该 batch 的目标标签分布。
            # 这可以抑制合成样本全部被 student 压到少数几个类别的坍缩现象，
            # 也对应问题八中“distribution regularization”的论文叙事。
            task_probs_s = torch.softmax(t_logits_s, dim=1)
            pred_distribution = task_probs_s.mean(dim=0).clamp_min(1e-8)
            target_distribution = torch.nn.functional.one_hot(labels_s, num_classes=N_CLASSES).float().mean(dim=0)
            target_distribution = target_distribution.to(DEVICE).clamp_min(1e-8)
            pred_distribution = pred_distribution / pred_distribution.sum()
            target_distribution = target_distribution / target_distribution.sum()
            loss_distribution = torch.nn.functional.kl_div(
                pred_distribution.log(),
                target_distribution,
                reduction="sum",
            )

            loss_s = (
                    loss_s_task
                    + (0.0 if args.disable_syn_concept_loss else SYN_CONCEPT_LOSS_WEIGHT) * loss_s_concept
                    + (0.0 if args.disable_syn_proto_loss else SYN_PROTO_LOSS_WEIGHT) * loss_s_proto
                    + (0.0 if args.disable_distribution_reg else args.distribution_reg_weight) * loss_distribution
            )

            # --- c. 合并损失并反向传播 ---
            # 将原始损失和合成损失加权相加。
            # synthetic_loss_weight 默认保持 0.5，对应当前固定主结果；
            # 后续诊断实验可以适度调高它，观察目标类准确率能否追近 strong baseline。
            loss = (
                    loss_o
                    + args.synthetic_loss_weight * loss_s
                    + args.teacher_consistency_weight * loss_teacher_consistency
                    + args.base_preservation_weight * loss_base_preservation
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch + 1}/{args.epochs}, 平均损失 (Avg Loss): {total_loss / max(num_batches, 1):.4f}")

    # 5. 保存微调后的模型
    checkpoint_dir = os.path.dirname(args.output_checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output_checkpoint)
    print(f"\n成功！步骤 3 完成。模型已保存为 '{args.output_checkpoint}'")


if __name__ == "__main__":
    hybrid_balance_train()
