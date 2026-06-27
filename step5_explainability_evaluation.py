# ==========================================================================================
# SSCBM + D-CGFS 研究项目 - 步骤 5: 可解释性评估
#
# 核心目标：
# 问题七指出：如果只报告分类准确率，D-CGFS 容易被认为只是普通增强方法。
# 因此本脚本专门评估 D-CGFS 是否保持或增强了概念瓶颈模型的可解释性。
#
# 本脚本包含三类实验：
# 1. Concept Accuracy:
#    - overall_c_acc: 所有样本、所有概念的概念准确率。
#    - target_c_acc: 自动发现的弱势目标类上的概念准确率。
#    - target_disc_c_acc: 目标类判别概念 D 上的概念准确率。
#
# 2. Concept Heatmap Quality:
#    - heatmap_entropy: heatmap 越集中，熵越低。
#    - mask_compactness: 高响应区域占比，越低通常越紧凑。
#    - bbox_energy_ratio: heatmap 能量落在 CUB bounding box 内的比例，越高越少激活背景。
#
# 3. Concept Intervention:
#    - original_acc: 不干预概念时的目标类分类准确率。
#    - intervention_acc: 将判别概念 D 替换为真实概念标签后的目标类分类准确率。
#    - intervention_gain = intervention_acc - original_acc。
#
# 论文叙事：
# 如果 D-CGFS 后 C_acc 不下降、heatmap 更集中/更少背景、intervention gain 更明显，
# 就能说明方法不是普通 long-tail augmentation，而是仍然保留了 CBM 的可解释优势。
#
# 作者：[肖凡]
# 日期：[2026年5月18日]
# ==========================================================================================

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from models.sscbm import SSCBM
from torchvision.models import resnet34
from train.utils import wrap_pretrained_model
import data.cub_loader as cub_data_module
from dcgfs_config import METHOD_ACRONYM


DATA_ROOT = "data/CUB_200_2011"
PAIR_CSV_PATH = "data/D-CGFS_Auxiliary/target_base_pairs.csv"
CHECKPOINT_BASE = "checkpoints/CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt"
CHECKPOINT_CGENS = "checkpoints/best_sscbm_dcgfs_hybrid.pt"
SAVE_DIR = "explainability_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CONCEPTS = 112
N_CLASSES = 200
EMB_SIZE = 32
HEATMAP_THRESHOLD = 0.6


def parse_args():
    """
    解析可解释性评估参数。

    主流程训练出的 D-CGFS checkpoint 可能不是历史默认文件名，因此这里允许显式传入。
    """
    parser = argparse.ArgumentParser(description=f"SSCBM + {METHOD_ACRONYM} 可解释性评估")
    parser.add_argument("--baseline-checkpoint", default=CHECKPOINT_BASE)
    parser.add_argument("--candidate-checkpoint", default=CHECKPOINT_CGENS)
    parser.add_argument("--candidate-name", default=f"SSCBM+{METHOD_ACRONYM}")
    parser.add_argument("--save-dir", default=SAVE_DIR)
    return parser.parse_args()


def parse_index_vector(index_str):
    """解析形如 '3;17;42' 的判别概念索引字符串。"""
    return [int(item) for item in str(index_str).split(";") if item.strip() != ""]


def load_target_pairs():
    """
    读取自动 target-base pair，并构造 target_class -> discriminative concepts 的映射。

    返回:
        pair_df:
            target_base_pairs.csv 的完整内容。
        target_to_disc:
            key 是 0-based 目标类标签，value 是该目标类对应的判别概念集合。
    """
    if not os.path.exists(PAIR_CSV_PATH):
        raise FileNotFoundError(f"未找到 {PAIR_CSV_PATH}，请先运行 find_weak_classes.py。")

    pair_df = pd.read_csv(PAIR_CSV_PATH)
    required_cols = {"target_class_id", "discriminative_concepts"}
    missing = required_cols - set(pair_df.columns)
    if missing:
        raise ValueError(f"{PAIR_CSV_PATH} 缺少必要列: {sorted(missing)}")

    target_to_disc = {}
    for _, row in pair_df.iterrows():
        target_label = int(row["target_class_id"]) - 1
        concept_indices = parse_index_vector(row["discriminative_concepts"])
        if target_label not in target_to_disc:
            target_to_disc[target_label] = set()
        target_to_disc[target_label].update(concept_indices)

    target_to_disc = {label: sorted(indices) for label, indices in target_to_disc.items()}
    return pair_df, target_to_disc


def get_test_loader():
    """加载标准测试集。可解释性最终报告也必须使用测试集，和 step4 的最终评估保持一致。"""
    with open("configs/CUB-200-2011.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    _, _, test_dl, _, _ = cub_data_module.generate_data(
        config=config["dataset_config"],
        seed=42,
        labeled_ratio=1.0,
    )
    return test_dl


def load_model(model_path):
    """初始化 SSCBM 并加载指定 checkpoint。"""
    model = SSCBM(
        n_concepts=N_CONCEPTS,
        n_tasks=N_CLASSES,
        emb_size=EMB_SIZE,
        c_extractor_arch=wrap_pretrained_model(resnet34),
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.eval()
    return model


def normalize_rel_path(img_path):
    """把 pkl 中可能出现的绝对路径或半相对路径统一成 CUB images 下的相对路径。"""
    if "CUB_200_2011/images/" in img_path:
        return img_path.split("CUB_200_2011/images/")[-1]
    if "/images/" in img_path:
        return img_path.split("/images/")[-1]
    return img_path


def load_cub_metadata():
    """
    读取 CUB 官方索引和 bounding box。

    返回:
        path_to_id:
            图像相对路径 -> image_id。
        bbox_by_id:
            image_id -> (x, y, w, h)，坐标基于原始图像尺寸。
    """
    images = pd.read_csv(
        os.path.join(DATA_ROOT, "images.txt"),
        sep=" ",
        names=["image_id", "path"],
    )
    bboxes = pd.read_csv(
        os.path.join(DATA_ROOT, "bounding_boxes.txt"),
        sep=" ",
        names=["image_id", "x", "y", "w", "h"],
    )
    path_to_id = dict(zip(images["path"], images["image_id"]))
    bbox_by_id = {
        int(row["image_id"]): (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
        for _, row in bboxes.iterrows()
    }
    return path_to_id, bbox_by_id


def heatmap_entropy(heatmap):
    """
    计算归一化 heatmap entropy。
    heatmap 越集中，entropy 越低；如果均匀铺满图像，entropy 越高。
    """
    h = heatmap.astype(np.float64)
    h = h - h.min()
    if h.max() > 0:
        h = h / h.max()
    prob = h.reshape(-1)
    prob = prob / (prob.sum() + 1e-12)
    entropy = -(prob * np.log(prob + 1e-12)).sum()
    return float(entropy / np.log(len(prob)))


def mask_compactness(heatmap, threshold=HEATMAP_THRESHOLD):
    """高响应区域占整张 heatmap 的比例，越小说明响应越紧凑。"""
    h = heatmap.astype(np.float64)
    h = h - h.min()
    if h.max() > 0:
        h = h / h.max()
    return float((h > threshold).mean())


def bbox_energy_ratio(heatmap, rel_path, path_to_id, bbox_by_id):
    """
    计算 heatmap 能量落在 CUB bounding box 内的比例。

    这是一个弱定位指标：值越高，说明概念响应越集中在鸟本体区域，而不是背景。
    """
    if rel_path not in path_to_id:
        return np.nan

    image_id = path_to_id[rel_path]
    if image_id not in bbox_by_id:
        return np.nan

    img_full_path = os.path.join(DATA_ROOT, "images", rel_path)
    with Image.open(img_full_path) as img:
        img_w, img_h = img.size

    x, y, w, h = bbox_by_id[image_id]
    hm_h, hm_w = heatmap.shape
    x1 = int(np.floor(x / img_w * hm_w))
    y1 = int(np.floor(y / img_h * hm_h))
    x2 = int(np.ceil((x + w) / img_w * hm_w))
    y2 = int(np.ceil((y + h) / img_h * hm_h))
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, hm_w), min(y2, hm_h)

    hmap = heatmap.astype(np.float64)
    hmap = hmap - hmap.min()
    total_energy = hmap.sum() + 1e-12
    inside_energy = hmap[y1:y2, x1:x2].sum()
    return float(inside_energy / total_energy)


@torch.no_grad()
def evaluate_concept_accuracy(model, loader, target_to_disc, model_name):
    """
    评估概念准确率是否保持或提升。

    除 overall C_acc 外，还单独看自动弱势目标类和判别概念 D 上的 C_acc。
    这对应问题七中的第一类可解释性实验。
    """
    target_labels = set(target_to_disc.keys())
    total_correct = 0
    total_count = 0
    target_correct = 0
    target_count = 0
    disc_correct = 0
    disc_count = 0

    for batch in loader:
        imgs = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)
        concept_labels = batch[2].to(DEVICE)

        outputs = model(imgs)
        concept_probs = outputs[0]
        concept_preds = (concept_probs > 0.5).int()

        total_correct += (concept_preds == concept_labels).sum().item()
        total_count += concept_labels.numel()

        for row_idx in range(labels.size(0)):
            label = int(labels[row_idx].item())
            if label not in target_labels:
                continue
            target_correct += (concept_preds[row_idx] == concept_labels[row_idx]).sum().item()
            target_count += concept_labels.size(1)

            disc_indices = target_to_disc[label]
            disc_correct += (
                concept_preds[row_idx, disc_indices] == concept_labels[row_idx, disc_indices]
            ).sum().item()
            disc_count += len(disc_indices)

    return {
        "model": model_name,
        "overall_c_acc": total_correct / total_count if total_count else 0,
        "target_c_acc": target_correct / target_count if target_count else 0,
        "target_disc_c_acc": disc_correct / disc_count if disc_count else 0,
    }


@torch.no_grad()
def evaluate_heatmap_quality(model, loader, target_to_disc, model_name, max_batches=None):
    """
    评估判别概念 heatmap 是否更集中、更少背景激活。

    指标：
        heatmap_entropy: 越低越集中。
        mask_compactness: 高响应区域占比，越低通常越紧凑。
        bbox_energy_ratio: 越高说明越落在鸟本体 bbox 内。
    """
    path_to_id, bbox_by_id = load_cub_metadata()
    dataset_data = loader.dataset.data
    sample_offset = 0
    entropies = []
    compactness_values = []
    bbox_ratios = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        imgs = batch[0].to(DEVICE)
        labels = batch[1]
        heatmaps = model.plot_heatmap(imgs).detach().cpu().numpy()

        for row_idx in range(labels.size(0)):
            label = int(labels[row_idx].item())
            if label not in target_to_disc:
                continue

            disc_indices = target_to_disc[label]
            combined_heatmap = np.max(heatmaps[row_idx, disc_indices], axis=0)
            rel_path = normalize_rel_path(dataset_data[sample_offset + row_idx]["img_path"])

            entropies.append(heatmap_entropy(combined_heatmap))
            compactness_values.append(mask_compactness(combined_heatmap))
            bbox_ratios.append(bbox_energy_ratio(combined_heatmap, rel_path, path_to_id, bbox_by_id))

        sample_offset += labels.size(0)

    return {
        "model": model_name,
        "heatmap_entropy": float(np.nanmean(entropies)) if entropies else np.nan,
        "mask_compactness": float(np.nanmean(compactness_values)) if compactness_values else np.nan,
        "bbox_energy_ratio": float(np.nanmean(bbox_ratios)) if bbox_ratios else np.nan,
        "num_heatmap_samples": len(entropies),
    }


@torch.no_grad()
def evaluate_concept_intervention(model, loader, target_to_disc, model_name):
    """
    评估概念干预收益。

    对自动弱势目标类样本，只干预对应判别概念集合 D：
        - 原始预测：model(imgs)
        - 干预预测：model(imgs, c=true_concepts, intervention_idxs=D_mask)

    如果干预后准确率提升明显，说明模型仍然保留了 CBM 的“概念可修正”优势。
    """
    target_labels = set(target_to_disc.keys())
    original_correct = 0
    intervention_correct = 0
    total = 0

    for batch in loader:
        imgs = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)
        concept_labels = batch[2].to(DEVICE)

        target_rows = [idx for idx in range(labels.size(0)) if int(labels[idx].item()) in target_labels]
        if not target_rows:
            continue

        imgs_t = imgs[target_rows]
        labels_t = labels[target_rows]
        concepts_t = concept_labels[target_rows]
        intervention_idxs = torch.zeros_like(concepts_t)
        for local_idx, batch_idx in enumerate(target_rows):
            label = int(labels[batch_idx].item())
            intervention_idxs[local_idx, target_to_disc[label]] = 1

        original_logits = model(imgs_t)[3]
        intervened_logits = model(
            imgs_t,
            c=concepts_t,
            intervention_idxs=intervention_idxs,
        )[3]

        original_preds = torch.argmax(original_logits, dim=1)
        intervened_preds = torch.argmax(intervened_logits, dim=1)
        original_correct += (original_preds == labels_t).sum().item()
        intervention_correct += (intervened_preds == labels_t).sum().item()
        total += labels_t.size(0)

    original_acc = original_correct / total if total else 0
    intervention_acc = intervention_correct / total if total else 0
    return {
        "model": model_name,
        "original_acc": original_acc,
        "intervention_acc": intervention_acc,
        "intervention_gain": intervention_acc - original_acc,
        "num_intervention_samples": total,
    }


def print_comparison(title, base_result, candidate_result, keys):
    """打印 Baseline 和 D-CGFS 的指标对比。"""
    print("\n" + "=" * 70)
    print(title)
    print("-" * 70)
    for key in keys:
        base_value = base_result[key]
        candidate_value = candidate_result[key]
        if isinstance(base_value, float):
            print(f"{key}: {base_value:.4f} -> {candidate_value:.4f} (diff={candidate_value - base_value:+.4f})")
        else:
            print(f"{key}: {base_value} -> {candidate_value}")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("步骤 1: 正在读取自动 target-base pair 和判别概念集合 D...")
    pair_df, target_to_disc = load_target_pairs()
    print(f"目标类数量: {len(target_to_disc)}")
    print(f"目标类 ID (1-based): {[label + 1 for label in target_to_disc.keys()]}")

    print("\n步骤 2: 正在加载测试集...")
    test_loader = get_test_loader()

    print(f"\n步骤 3: 正在加载 Baseline 与 {METHOD_ACRONYM} 模型...")
    baseline_model = load_model(args.baseline_checkpoint)
    candidate_model = load_model(args.candidate_checkpoint)

    print("\n步骤 4: 正在评估概念准确率...")
    base_c_acc = evaluate_concept_accuracy(baseline_model, test_loader, target_to_disc, "SSCBM")
    candidate_c_acc = evaluate_concept_accuracy(candidate_model, test_loader, target_to_disc, args.candidate_name)
    print_comparison(
        "Concept Accuracy",
        base_c_acc,
        candidate_c_acc,
        ["overall_c_acc", "target_c_acc", "target_disc_c_acc"],
    )

    print("\n步骤 5: 正在评估判别概念 heatmap 质量...")
    base_heatmap = evaluate_heatmap_quality(baseline_model, test_loader, target_to_disc, "SSCBM")
    candidate_heatmap = evaluate_heatmap_quality(candidate_model, test_loader, target_to_disc, args.candidate_name)
    print_comparison(
        "Concept Heatmap Quality",
        base_heatmap,
        candidate_heatmap,
        ["heatmap_entropy", "mask_compactness", "bbox_energy_ratio", "num_heatmap_samples"],
    )

    print("\n步骤 6: 正在评估判别概念干预收益...")
    base_intervention = evaluate_concept_intervention(baseline_model, test_loader, target_to_disc, "SSCBM")
    candidate_intervention = evaluate_concept_intervention(candidate_model, test_loader, target_to_disc, args.candidate_name)
    print_comparison(
        "Concept Intervention",
        base_intervention,
        candidate_intervention,
        ["original_acc", "intervention_acc", "intervention_gain", "num_intervention_samples"],
    )

    pd.DataFrame([base_c_acc, candidate_c_acc]).to_csv(
        os.path.join(args.save_dir, "concept_accuracy.csv"),
        index=False,
    )
    pd.DataFrame([base_heatmap, candidate_heatmap]).to_csv(
        os.path.join(args.save_dir, "heatmap_quality.csv"),
        index=False,
    )
    pd.DataFrame([base_intervention, candidate_intervention]).to_csv(
        os.path.join(args.save_dir, "concept_intervention.csv"),
        index=False,
    )
    pair_df.to_csv(os.path.join(args.save_dir, "used_target_base_pairs.csv"), index=False)

    print(f"\n可解释性评估结果已保存到: {args.save_dir}")
