import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from dataset_specs import build_sscbm, load_dataset_config, load_dataset_spec


# ==========================================================================================
# 自动目标类/基座类选择
#
# 自动化选择目标和基座类：
#   1. 先用 Baseline SSCBM 在验证集上统计每个类别的表现，找出弱势类别 yt；
#   2. 再为每个弱势类别自动选择一个或多个适合做语义融合的基座类别 yb；
#   3. 对每个目标类-基座类 pair 计算判别概念集合 D = TopK({k | p_yt,k - p_yb,k > 0})；
#   4. 将选择结果保存为 target_base_pairs.csv，供 step1_generate_mapping.py 和
#      step2_synthesize_data.py 后续使用。
#
# 基座类选择分数：
#   S(yt, yb) = alpha * Sconf(yt, yb)
#             + beta  * Svis(yt, yb)
#             + gamma * Sconcept(yt, yb)
#
# 其中：
#   Sconf    : Baseline 混淆矩阵中，目标类 yt 被错分为候选基座类 yb 的比例。
#              如果模型经常把 yt 预测成 yb，说明二者在当前模型看来容易混淆。
#   Svis     : 两个类别的视觉原型相似度。这里使用 SSCBM backbone 输出的图像特征均值。
#   Sconcept : 两个类别的概念原型相似度。这里使用 CUB 概念标签的类别均值。
#
# 判别概念集合：
#   p_yt 表示目标类的概念原型，p_yb 表示基座类的概念原型。
#   D = TopK({k | p_yt,k - p_yb,k > 0})
#   也就是先筛掉基座类更强或二者相同的概念，再选择“目标类明显强于基座类”的概念。
#   step1 后续只会用这些概念的 heatmap 生成 mask，而不是再对所有概念取 max。
#
# 输出文件:
#   data/D-CGFS_Auxiliary/target_base_pairs.csv
# 关键字段:
#   pair_id, target_class_id, base_class_id, score, score_conf, score_visual, score_concept,
#   discriminative_concepts, target_concept_proto, base_concept_proto

# 作者：[肖凡]
# 日期：[2026年4月20日]
# ==========================================================================================


CHECKPOINT_BASE = "checkpoints/CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt"
CONFIG_PATH = "configs/CUB-200-2011.yaml"
SAVE_DIR = "data/D-CGFS_Auxiliary"
PAIR_CSV_PATH = os.path.join(SAVE_DIR, "target_base_pairs.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CONCEPTS = 112
N_CLASSES = 200
EMB_SIZE = 32
CURRENT_SPEC = None
CURRENT_SEED = 42

# 联合打分的三个权重。
# 如果后续实验发现某一项更重要，可以在这里调参做消融实验。
ALPHA_CONF = 0.5
BETA_VIS = 0.3
GAMMA_CONCEPT = 0.2

# TOP_TARGETS: 自动选择多少个表现最弱的目标类。
# BASES_PER_TARGET: 每个目标类选择多少个基座类；当前每个目标类选 1 个，方便保持原流程简单。
TOP_TARGETS = 5
BASES_PER_TARGET = 1

# 每个 target-base pair 选择多少个判别概念。
# 这些概念来自 TopK({k | p_target,k - p_base,k > 0})，用于 step1 生成判别性概念区域 mask。
DISC_TOP_K = 10

# 类别至少要有多少样本才参与统计。CUB 验证集中每类通常都有样本，这里主要防止除零。
MIN_CLASS_SAMPLES = 1


def _safe_normalize(x, eps=1e-12):
    """对向量做 L2 归一化；eps 用来避免零向量导致除零。"""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _cosine_similarity_matrix(a, b):
    """
    计算两个原型矩阵之间的余弦相似度。
    输入:
        a: [num_classes, dim]
        b: [num_classes, dim]
    输出:
        sim: [num_classes, num_classes]，sim[i, j] 表示第 i 类和第 j 类的相似度。
    """
    a = _safe_normalize(a)
    b = _safe_normalize(b)
    return np.matmul(a, b.T)


def _format_concept_indices(indices):
    """将 0-based 概念索引列表保存成分号分隔字符串，便于写入 CSV 并被 step1 读取。"""
    return ";".join(str(int(idx)) for idx in indices)


def _format_float_values(values):
    """将浮点列表保存成分号分隔字符串，用于记录判别概念的原型差值。"""
    return ";".join(f"{float(value):.6f}" for value in values)


def _select_discriminative_concepts(concept_proto, target_idx, base_idx, top_k=DISC_TOP_K):
    """
    计算目标类相对于基座类的判别概念集合。

    输入:
        concept_proto:
            [num_classes, n_concepts]，每一行是一个类别的平均概念向量。
        target_idx:
            目标类索引，0-based。
        base_idx:
            基座类索引，0-based。
        top_k:
            最多选择多少个目标类更强的概念。

    输出:
        selected:
            0-based 概念索引。该索引可以直接用于 heatmaps_np[selected]。
        concept_gap:
            p_target - p_base，记录每个概念在目标类相对基座类上的强弱差异。

    注意:
        这里不会从全部概念中无条件取 TopK。
        我们先筛选 concept_gap > 0 的概念，只保留目标类比基座类更强的概念。
        这样可以避免把“基座类更强”的概念误作为目标类判别区域。
    """
    concept_gap = concept_proto[target_idx] - concept_proto[base_idx]
    positive_indices = np.where(concept_gap > 0)[0]
    if len(positive_indices) == 0:
        return np.array([], dtype=int), concept_gap

    k = min(top_k, len(positive_indices))
    selected_order = np.argsort(-concept_gap[positive_indices])[:k]
    selected = positive_indices[selected_order]
    return selected, concept_gap


def _load_model():
    """加载已经训练好的 Baseline SSCBM，用于统计弱势类和类别相似性。"""
    if CURRENT_SPEC is not None:
        return build_sscbm(CURRENT_SPEC, DEVICE)
    raise RuntimeError("CURRENT_SPEC 未初始化。")


def _load_validation_loader():
    """
    复用 CUB loader 生成验证集。

    这里刻意使用 val_dl，而不是 test_dl：
        - 验证集可以用于选择弱势类、选择基座类和调节超参数；
        - 测试集必须留到 step4_final_evaluation.py 中做最终一次性评估；
        - 这样可以避免 test leakage，即避免方法设计阶段提前“看见”测试集错误模式。

    labeled_ratio=1.0 表示在验证集统计概念原型时使用全部概念标签。
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["dataset_config"]["num_workers"] = 0
    data_module = CURRENT_SPEC.data_module
    _, val_dl, _, _, _ = data_module.generate_data(
        config=config["dataset_config"],
        seed=CURRENT_SEED,
        labeled_ratio=1.0,
    )
    return val_dl


@torch.no_grad()
def collect_baseline_statistics(model, loader, max_batches=0):
    """
    在验证集上收集自动配对所需的统计量。

    返回:
        class_acc:
            每个类别的 baseline 分类准确率，用于找弱势目标类。
        class_total:
            每个类别的样本数。
        confusion:
            混淆矩阵，confusion[i, j] 表示真实类别 i 被预测为类别 j 的次数。
        visual_proto:
            视觉原型，每类所有样本的 backbone 特征均值。
        concept_proto:
            概念原型，每类所有样本的 CUB 概念标签均值。

    注意:
        代码内部 labels 是 0-based；保存到 CSV 时会转成 CUB 官方习惯的 1-based 类别 ID。
    """
    class_correct = np.zeros(N_CLASSES)
    class_total = np.zeros(N_CLASSES)
    confusion = np.zeros((N_CLASSES, N_CLASSES))
    feature_dim = getattr(model, "resnet_out_features", 512)
    visual_sum = np.zeros((N_CLASSES, feature_dim), dtype=np.float64)
    concept_sum = np.zeros((N_CLASSES, N_CONCEPTS), dtype=np.float64)

    for batch_idx, data_tuple in enumerate(tqdm(loader, desc="collect baseline stats")):
        if max_batches and batch_idx >= max_batches:
            break
        imgs = data_tuple[0].to(DEVICE)
        labels = data_tuple[1].to(DEVICE)
        concept_labels = data_tuple[2].to(DEVICE)

        outputs = model(imgs)
        preds = torch.argmax(outputs[3], dim=1)

        # 使用 SSCBM 的分类 backbone 输出作为视觉特征。
        # 每个类别所有样本的特征均值就是该类别的视觉原型。
        visual_features = model.pre_concept_model(imgs)
        visual_features = F.normalize(visual_features, p=2, dim=1)

        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = preds[i].item()

            # 统计分类表现和混淆关系。
            class_total[label] += 1
            confusion[label, pred] += 1

            # 累加视觉特征和概念标签，循环结束后除以样本数得到类别原型。
            visual_sum[label] += visual_features[i].detach().cpu().numpy()
            concept_sum[label] += concept_labels[i].detach().cpu().numpy()
            if pred == label:
                class_correct[label] += 1

    class_acc = class_correct / np.maximum(class_total, 1)
    visual_proto = visual_sum / np.maximum(class_total[:, None], 1)
    concept_proto = concept_sum / np.maximum(class_total[:, None], 1)
    return class_acc, class_total, confusion, visual_proto, concept_proto


def select_target_base_pairs(
        class_acc,
        class_total,
        confusion,
        visual_proto,
        concept_proto,
        top_targets=TOP_TARGETS,
        bases_per_target=BASES_PER_TARGET,
):
    """
    根据弱势类准确率和联合相似度分数，自动选择目标类-基座类 pair。

    目标类 yt:
        从 baseline 准确率最低的类别中选择。

    基座类 yb:
        对每个目标类，在所有其他类别中计算联合分数，选择分数最高者。
        目标类自身会被排除，防止出现 target == base。
    """
    # 先筛出有足够样本的类别，再按准确率从低到高排序，取前 top_targets 个弱势类。
    valid_targets = np.where(class_total >= MIN_CLASS_SAMPLES)[0]
    sorted_targets = valid_targets[np.argsort(class_acc[valid_targets])]
    target_indices = sorted_targets[:top_targets]

    # 余弦相似度原本范围是 [-1, 1]，这里线性映射到 [0, 1]，便于和混淆分数加权相加。
    visual_sim = (_cosine_similarity_matrix(visual_proto, visual_proto) + 1.0) / 2.0
    concept_sim = (_cosine_similarity_matrix(concept_proto, concept_proto) + 1.0) / 2.0

    rows = []
    for target_idx in target_indices:
        # Sconf: 只统计“错分到某个类别”的比例。
        # 如果某个弱势类没有错分样本，则混淆分数全部置零，此时主要由视觉/概念相似度决定。
        error_count = class_total[target_idx] - confusion[target_idx, target_idx]
        if error_count > 0:
            conf_score = confusion[target_idx] / error_count
        else:
            conf_score = np.zeros(N_CLASSES)
        conf_score[target_idx] = 0.0

        # 三项分数加权得到最终基座类选择分数。
        joint_score = (
            ALPHA_CONF * conf_score
            + BETA_VIS * visual_sim[target_idx]
            + GAMMA_CONCEPT * concept_sim[target_idx]
        )

        # 排除目标类自身，以及样本数不足的类别。
        joint_score[target_idx] = -np.inf
        joint_score[class_total < MIN_CLASS_SAMPLES] = -np.inf

        # 不能只看前 bases_per_target 个候选，因为某些候选虽然联合分数高，
        # 但可能没有任何 p_target - p_base > 0 的判别概念。
        # 因此这里按分数从高到低遍历，直到为当前目标类收集到足够的有效基座类。
        base_rank = 0
        for base_idx in np.argsort(-joint_score):
            if not np.isfinite(joint_score[base_idx]):
                continue

            # 问题二解决方案的关键：为当前 target-base pair 计算判别概念集合。
            # 后续 step1 不再使用所有概念 heatmap，而只使用这些概念生成 mask。
            disc_concepts, concept_gap = _select_discriminative_concepts(
                concept_proto=concept_proto,
                target_idx=target_idx,
                base_idx=base_idx,
            )
            if len(disc_concepts) == 0:
                print(
                    f"警告：目标类 {target_idx + 1} 相对于基座类 {base_idx + 1} "
                    "没有 concept_gap > 0 的判别概念，跳过该 pair。"
                )
                continue

            base_rank += 1

            target_class_id = int(target_idx + CURRENT_SPEC.class_id_base)
            base_class_id = int(base_idx + CURRENT_SPEC.class_id_base)
            rows.append({
                "pair_id": len(rows),
                "target_class_id": target_class_id,
                "target_class_name": CURRENT_SPEC.class_names[target_idx],
                "target_acc": float(class_acc[target_idx]),
                "target_samples": int(class_total[target_idx]),
                "base_rank": base_rank,
                "base_class_id": base_class_id,
                "base_class_name": CURRENT_SPEC.class_names[base_idx],
                "score": float(joint_score[base_idx]),
                "score_conf": float(conf_score[base_idx]),
                "score_visual": float(visual_sim[target_idx, base_idx]),
                "score_concept": float(concept_sim[target_idx, base_idx]),
                "discriminative_concepts": _format_concept_indices(disc_concepts),
                "discriminative_concept_gaps": _format_float_values(concept_gap[disc_concepts]),
                # 保存 target/base 的完整概念原型，供 step2 做概念一致性过滤：
                # sim(c_syn, p_target) > sim(c_syn, p_base) + margin。
                "target_concept_proto": _format_float_values(concept_proto[target_idx]),
                "base_concept_proto": _format_float_values(concept_proto[base_idx]),
                "disc_top_k": DISC_TOP_K,
                "alpha_conf": ALPHA_CONF,
                "beta_visual": BETA_VIS,
                "gamma_concept": GAMMA_CONCEPT,
            })
            if base_rank >= bases_per_target:
                break
    return pd.DataFrame(rows)


@torch.no_grad()
def find_weakest_classes(args=None):
    """
    主入口：
    1. 加载验证集和 baseline；
    2. 收集类别准确率、混淆矩阵、视觉原型、概念原型；
    3. 自动选择目标类-基座类；
    4. 保存 CSV，作为后续 step1/step2 的输入。
    """
    global CURRENT_SPEC, CHECKPOINT_BASE, CONFIG_PATH, SAVE_DIR, PAIR_CSV_PATH, CURRENT_SEED
    global N_CONCEPTS, N_CLASSES, EMB_SIZE

    args = args or parse_args()
    CURRENT_SEED = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    CURRENT_SPEC = load_dataset_spec(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        aux_dir=args.save_dir,
    )
    CHECKPOINT_BASE = CURRENT_SPEC.checkpoint_path
    CONFIG_PATH = CURRENT_SPEC.config_path
    SAVE_DIR = CURRENT_SPEC.aux_dir
    PAIR_CSV_PATH = os.path.join(SAVE_DIR, "target_base_pairs.csv")
    N_CONCEPTS = CURRENT_SPEC.n_concepts
    N_CLASSES = CURRENT_SPEC.n_classes
    EMB_SIZE = CURRENT_SPEC.emb_size

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"当前数据集: {CURRENT_SPEC.name}")
    print(f"当前 checkpoint: {CURRENT_SPEC.checkpoint_path}")
    print(f"输出目录: {SAVE_DIR}")
    print("步骤 1: 正在加载验证集和 Baseline SSCBM...")
    loader = _load_validation_loader()
    model = _load_model()

    print("步骤 2: 正在统计每类准确率、混淆矩阵、视觉原型和概念原型...")
    stats = collect_baseline_statistics(model, loader, max_batches=args.max_batches)
    class_acc, class_total, confusion, visual_proto, concept_proto = stats

    print("步骤 3: 正在自动选择目标类-基座类配对...")
    pair_df = select_target_base_pairs(
        class_acc=class_acc,
        class_total=class_total,
        confusion=confusion,
        visual_proto=visual_proto,
        concept_proto=concept_proto,
    )
    pair_df.to_csv(PAIR_CSV_PATH, index=False)

    print("\n--- Baseline 表现最差的类别及自动选择的基座类 ---")
    for _, row in pair_df.iterrows():
        print(
            f"Pair {int(row['pair_id']):02d} | "
            f"目标类 {int(row['target_class_id']):3d} {row['target_class_name']:<28s} "
            f"Acc={row['target_acc']:.2%} | "
            f"基座类 {int(row['base_class_id']):3d} {row['base_class_name']:<28s} "
            f"Score={row['score']:.4f} "
            f"(conf={row['score_conf']:.4f}, vis={row['score_visual']:.4f}, concept={row['score_concept']:.4f}) | "
            f"判别概念 D={row['discriminative_concepts']}"
        )
    print(f"\n已保存自动配对结果: {PAIR_CSV_PATH}")
    return pair_df


def parse_args():
    parser = argparse.ArgumentParser(description="自动选择 D-CGFS 的弱势目标类和基座类。")
    parser.add_argument("--dataset", default="CUB-200-2011", choices=["CUB-200-2011", "AwA2", "PBC", "7pt"])
    parser.add_argument("--checkpoint", default=None, help="Baseline SSCBM checkpoint；不填则使用数据集默认路径。")
    parser.add_argument("--save-dir", default=None, help="target_base_pairs.csv 输出目录；不填则使用数据集默认目录。")
    parser.add_argument("--max-batches", type=int, default=0, help="调试用：最多统计多少个 batch；0 表示完整统计。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子；用于数据加载和可复现实验。")
    return parser.parse_args()


if __name__ == "__main__":
    find_weakest_classes()
