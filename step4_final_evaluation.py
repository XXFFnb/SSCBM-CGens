# ==========================================================================================
# SSCBM + D-CGFS 研究项目 - 步骤 4: 最终效果评估
#
# 核心目标：
# 对比分析“原始 SSCBM 模型 (Baseline)”和“经过 D-CGFS 平衡微调后的模型”在标准
# 测试集上的分类性能，并且重点评估 find_weak_classes.py 自动选出的弱势目标类。
#
# 重要原则：
# 1. 不再使用固定的 TARGET_CLASS_ID=65。
# 2. 必须读取 data/D-CGFS_Auxiliary/target_base_pairs.csv。
# 3. step4 只在测试集上做最终评估；目标类和基座类已经在验证集上由
#    find_weak_classes.py 选择，避免 test leakage。
#
# 输出指标：
# - overall_a_acc: 所有类别的总体分类准确率。
# - overall_c_acc: 所有概念的总体概念准确率。
# - selected_target_acc: 自动弱势目标类集合上的分类准确率。
# - selected_base_acc: 自动基座类集合上的分类准确率。
# - target_to_base_rate: 目标类样本被错分成对应基座类的比例。
# - macro_f1 / balanced_accuracy / worst_class_acc: 问题六要求的顶会级不平衡评估指标。
#
# 输出文件：
# final_evaluation_results/model_summary.csv
# final_evaluation_results/target_class_accuracy.csv
# final_evaluation_results/target_base_confusion.csv
# final_evaluation_results/used_target_base_pairs.csv
#
# 作者：[肖凡]
# 日期：[2026年5月18日]
# ==========================================================================================

import os
import argparse
import torch
import numpy as np
import pandas as pd
from dataset_specs import build_sscbm, load_dataset_config, load_dataset_spec
from dcgfs_config import METHOD_ACRONYM


# --- 1. 全局配置 ---
PAIR_CSV_PATH = "data/D-CGFS_Auxiliary/target_base_pairs.csv"
SAVE_DIR = "final_evaluation_results"

# 原始 Baseline 模型权重路径
CHECKPOINT_BASE = "checkpoints/CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt"
# 经过 D-CGFS 混合微调后的模型权重路径。
CHECKPOINT_CGENS = "checkpoints/best_sscbm_dcgfs_hybrid.pt"

N_CONCEPTS = 112
N_CLASSES = 200
EMB_SIZE = 32
CLASS_ID_OFFSET = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """
    解析最终评估参数。

    问题六需要评估多个 baseline 和 ablation checkpoint。
    因此 step4 不再只绑定一个 D-CGFS checkpoint，而是允许通过命令行指定待评估模型。
    默认值仍然对应完整 D-CGFS 主实验。
    """
    parser = argparse.ArgumentParser(description=f"SSCBM + {METHOD_ACRONYM} 最终分类性能评估")
    parser.add_argument("--dataset", default="CUB-200-2011", choices=["CUB-200-2011", "AwA2", "PBC", "7pt"])
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--candidate-checkpoint", default=CHECKPOINT_CGENS)
    parser.add_argument("--candidate-name", default=f"SSCBM + {METHOD_ACRONYM}")
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--pair-csv", default=None, help="target_base_pairs.csv 路径；不填使用数据集默认路径。")
    parser.add_argument("--class-attr-data-dir", default="class_attr_data_10")
    parser.add_argument("--seed", type=int, default=42, help="随机种子；用于测试集 loader 构建。")
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="仅用于快速冒烟测试；>0 时最多评估指定 batch 数，正式实验保持 0。",
    )
    return parser.parse_args()


def load_target_base_pairs(pair_csv_path=PAIR_CSV_PATH):
    """
    读取自动选择得到的 target-base pair。

    这里故意不提供旧版 65/64 的回退逻辑：
    如果没有 target_base_pairs.csv，说明前面的自动选择流程没有完成，继续评估会导致
    实验叙事和实际代码不一致，因此直接报错。
    """
    if not os.path.exists(pair_csv_path):
        raise FileNotFoundError(f"未找到 {pair_csv_path}，请先运行 find_weak_classes.py。")

    pair_df = pd.read_csv(pair_csv_path)
    required_cols = {"target_class_id", "base_class_id"}
    missing = required_cols - set(pair_df.columns)
    if missing:
        raise ValueError(f"{pair_csv_path} 缺少必要列: {sorted(missing)}")
    if pair_df.empty:
        raise ValueError(f"{pair_csv_path} 为空，无法确定最终评估的目标类。")

    # CUB 的 pair 文件使用 1-based 类别 id，AwA2 使用 0-based 类别 id；
    # CLASS_ID_OFFSET 会统一转换到模型训练使用的 0-based 标签。
    pair_df = pair_df.copy()
    pair_df["target_label"] = pair_df["target_class_id"].astype(int) - CLASS_ID_OFFSET
    pair_df["base_label"] = pair_df["base_class_id"].astype(int) - CLASS_ID_OFFSET

    if "pair_id" not in pair_df.columns:
        # 兼容 find_weak_classes.py 未来可能调整列顺序的情况，但仍然不回退到固定类别。
        pair_df["pair_id"] = [f"pair_{idx:03d}" for idx in range(len(pair_df))]

    return pair_df


def get_test_loader(spec, class_attr_data_dir="class_attr_data_10", seed=42):
    """
    构建标准测试集加载器。

    step4 是最终分类性能评估，必须使用测试集；目标类选择已经在验证集完成，
    因此这里使用测试集不会造成选择阶段的信息泄露。
    """
    config = load_dataset_config(spec)
    if spec.name == "CUB-200-2011":
        config["dataset_config"]["class_attr_data_dir"] = class_attr_data_dir
    config["dataset_config"]["num_workers"] = 0

    _, _, test_dl, _, _ = spec.data_module.generate_data(
        config=config["dataset_config"],
        seed=seed,
        labeled_ratio=1.0,
    )
    return test_dl


def load_model(spec, model_path):
    """初始化 SSCBM 并加载指定 checkpoint。"""
    return build_sscbm(spec, DEVICE, checkpoint_path=model_path)


@torch.no_grad()
def collect_predictions(spec, model_path, loader, max_batches=0):
    """
    在测试集上收集预测结果。

    返回值包含：
    - labels: 真实类别标签，0-based。
    - preds: 模型预测类别，0-based。
    - concept_correct: 每个概念预测是否正确。
    """
    model = load_model(spec, model_path)
    all_labels = []
    all_preds = []
    concept_correct = []

    for batch_idx, data_tuple in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        imgs = data_tuple[0].to(DEVICE)
        labels = data_tuple[1].to(DEVICE)
        concept_labels = data_tuple[2].to(DEVICE)

        # outputs 格式: (c_sem, c_pred, c_pred_unlabeled, task_logits, ...)
        outputs = model(imgs)
        c_sem = outputs[0]
        task_logits = outputs[3]

        preds = torch.argmax(task_logits, dim=1)
        c_preds = (c_sem > 0.5).int()

        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())
        concept_correct.append((c_preds == concept_labels).detach().cpu())

    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()
    concept_correct = torch.cat(concept_correct).numpy()
    return labels, preds, concept_correct


def compute_classwise_metrics(labels, preds, n_classes=None):
    """
    计算类别级指标。

    问题六指出，只看 overall accuracy 容易掩盖少数类失败。
    因此这里补充：
        - macro_f1: 每个类别 F1 的简单平均，类别不平衡时比 overall accuracy 更公平。
        - balanced_accuracy: 每个类别 recall 的简单平均。
        - worst_class_acc: 表现最差类别的 accuracy，用来观察尾部风险。
    """
    rows = []
    f1_values = []
    recall_values = []
    acc_values = []
    n_classes = n_classes or N_CLASSES
    for cls in range(n_classes):
        true_mask = labels == cls
        pred_mask = preds == cls
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        samples = int(true_mask.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = tp / samples if samples > 0 else np.nan

        if samples > 0:
            f1_values.append(f1)
            recall_values.append(recall)
            acc_values.append(acc)

        rows.append(
            {
                "class_id": cls + CLASS_ID_OFFSET,
                "samples": samples,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": acc,
            }
        )

    return {
        "macro_f1": float(np.mean(f1_values)) if f1_values else np.nan,
        "balanced_accuracy": float(np.mean(recall_values)) if recall_values else np.nan,
        "worst_class_acc": float(np.nanmin(acc_values)) if acc_values else np.nan,
        "class_rows": rows,
    }


def summarize_model(labels, preds, concept_correct, pair_df, model_name):
    """
    汇总单个模型的整体指标、目标类指标和 target-base 混淆指标。

    target-base 混淆率是本项目很关键的指标：
    D-CGFS 的设计初衷就是减少弱势目标类被强势基座类吸走的情况。
    """
    target_labels = sorted(pair_df["target_label"].unique().tolist())
    base_labels = sorted(pair_df["base_label"].unique().tolist())

    overall_a_acc = float((preds == labels).mean())
    overall_c_acc = float(concept_correct.mean())
    class_metrics = compute_classwise_metrics(labels, preds, n_classes=N_CLASSES)

    target_mask = np.isin(labels, target_labels)
    base_mask = np.isin(labels, base_labels)
    selected_target_acc = float((preds[target_mask] == labels[target_mask]).mean()) if target_mask.any() else np.nan
    selected_base_acc = float((preds[base_mask] == labels[base_mask]).mean()) if base_mask.any() else np.nan

    target_rows = []
    for target_label in target_labels:
        mask = labels == target_label
        samples = int(mask.sum())
        correct = int((preds[mask] == labels[mask]).sum()) if samples > 0 else 0
        target_rows.append(
            {
                "model": model_name,
                "target_class_id": target_label + CLASS_ID_OFFSET,
                "samples": samples,
                "correct": correct,
                "accuracy": correct / samples if samples > 0 else np.nan,
            }
        )

    pair_rows = []
    target_to_base_wrong = 0
    target_pair_samples = 0
    for _, row in pair_df.iterrows():
        target_label = int(row["target_label"])
        base_label = int(row["base_label"])
        mask = labels == target_label
        samples = int(mask.sum())
        correct = int((preds[mask] == target_label).sum()) if samples > 0 else 0
        wrong_to_base = int((preds[mask] == base_label).sum()) if samples > 0 else 0

        target_to_base_wrong += wrong_to_base
        target_pair_samples += samples

        pair_rows.append(
            {
                "model": model_name,
                "pair_id": row["pair_id"],
                "target_class_id": int(row["target_class_id"]),
                "target_class_name": row.get("target_class_name", ""),
                "base_class_id": int(row["base_class_id"]),
                "base_class_name": row.get("base_class_name", ""),
                "target_samples": samples,
                "target_correct": correct,
                "target_accuracy": correct / samples if samples > 0 else np.nan,
                "target_pred_as_base": wrong_to_base,
                "target_to_base_rate": wrong_to_base / samples if samples > 0 else np.nan,
            }
        )

    target_to_base_rate = (
        target_to_base_wrong / target_pair_samples if target_pair_samples > 0 else np.nan
    )

    summary = {
        "model": model_name,
        "overall_a_acc": overall_a_acc,
        "overall_c_acc": overall_c_acc,
        "macro_f1": class_metrics["macro_f1"],
        "balanced_accuracy": class_metrics["balanced_accuracy"],
        "worst_class_acc": class_metrics["worst_class_acc"],
        "selected_target_acc": selected_target_acc,
        "selected_base_acc": selected_base_acc,
        "target_to_base_rate": target_to_base_rate,
        "target_samples": int(target_mask.sum()),
        "base_samples": int(base_mask.sum()),
    }
    class_rows = []
    for row in class_metrics["class_rows"]:
        row = row.copy()
        row["model"] = model_name
        class_rows.append(row)

    return summary, target_rows, pair_rows, class_rows


def format_delta(new_value, old_value):
    """格式化 D-CGFS 相对 Baseline 的变化量。"""
    if np.isnan(new_value) or np.isnan(old_value):
        return "N/A"
    return f"{new_value - old_value:+.2%}"


if __name__ == "__main__":
    args = parse_args()
    spec = load_dataset_spec(dataset=args.dataset, checkpoint_path=args.baseline_checkpoint)
    PAIR_CSV_PATH = args.pair_csv or os.path.join(spec.aux_dir, "target_base_pairs.csv")
    CHECKPOINT_BASE = spec.checkpoint_path
    N_CONCEPTS = spec.n_concepts
    N_CLASSES = spec.n_classes
    EMB_SIZE = spec.emb_size
    CLASS_ID_OFFSET = spec.class_id_base

    os.makedirs(args.save_dir, exist_ok=True)

    print("步骤 1: 正在读取自动选择的 target-base pair...")
    print(f"当前数据集: {spec.name}")
    print(f"当前 baseline checkpoint: {CHECKPOINT_BASE}")
    print(f"当前 target-base pair 文件: {PAIR_CSV_PATH}")
    pair_df = load_target_base_pairs(PAIR_CSV_PATH)
    print(f"已读取 {len(pair_df)} 个 target-base pair，覆盖 {pair_df['target_label'].nunique()} 个目标类。")

    print("\n步骤 2: 正在准备标准测试数据集...")
    test_loader = get_test_loader(spec, args.class_attr_data_dir, args.seed)
    print("测试数据准备完毕。")

    print("\n步骤 3: 正在评估原始 Baseline 模型...")
    base_labels, base_preds, base_concept_correct = collect_predictions(
        spec,
        CHECKPOINT_BASE,
        test_loader,
        max_batches=args.max_eval_batches,
    )
    res_base, base_target_rows, base_pair_rows, base_class_rows = summarize_model(
        base_labels,
        base_preds,
        base_concept_correct,
        pair_df,
        "SSCBM (Baseline)",
    )

    print(f"\n步骤 4: 正在评估待比较模型 {args.candidate_name}...")
    candidate_labels, candidate_preds, candidate_concept_correct = collect_predictions(
        spec,
        args.candidate_checkpoint,
        test_loader,
        max_batches=args.max_eval_batches,
    )
    res_candidate, candidate_target_rows, candidate_pair_rows, candidate_class_rows = summarize_model(
        candidate_labels,
        candidate_preds,
        candidate_concept_correct,
        pair_df,
        args.candidate_name,
    )

    # 将结果保存为 CSV，方便后续画表、画图和写论文。
    summary_df = pd.DataFrame([res_base, res_candidate])
    target_df = pd.DataFrame(base_target_rows + candidate_target_rows)
    pair_eval_df = pd.DataFrame(base_pair_rows + candidate_pair_rows)
    classwise_df = pd.DataFrame(base_class_rows + candidate_class_rows)

    summary_df.to_csv(os.path.join(args.save_dir, "model_summary.csv"), index=False)
    target_df.to_csv(os.path.join(args.save_dir, "target_class_accuracy.csv"), index=False)
    pair_eval_df.to_csv(os.path.join(args.save_dir, "target_base_confusion.csv"), index=False)
    classwise_df.to_csv(os.path.join(args.save_dir, "classwise_metrics.csv"), index=False)
    pair_df.to_csv(os.path.join(args.save_dir, "used_target_base_pairs.csv"), index=False)

    print("\n" + "=" * 70)
    print(f"实验结论：核心指标对比 (Baseline -> {args.candidate_name})")
    print("-" * 70)
    print(
        f"整体任务准确率 overall_a_acc       : "
        f"{res_base['overall_a_acc']:.2%} -> {res_candidate['overall_a_acc']:.2%} "
        f"(变化: {format_delta(res_candidate['overall_a_acc'], res_base['overall_a_acc'])})"
    )
    print(
        f"整体概念准确率 overall_c_acc       : "
        f"{res_base['overall_c_acc']:.2%} -> {res_candidate['overall_c_acc']:.2%} "
        f"(变化: {format_delta(res_candidate['overall_c_acc'], res_base['overall_c_acc'])})"
    )
    print(
        f"Macro-F1 macro_f1                 : "
        f"{res_base['macro_f1']:.2%} -> {res_candidate['macro_f1']:.2%} "
        f"(变化: {format_delta(res_candidate['macro_f1'], res_base['macro_f1'])})"
    )
    print(
        f"Balanced Acc balanced_accuracy    : "
        f"{res_base['balanced_accuracy']:.2%} -> {res_candidate['balanced_accuracy']:.2%} "
        f"(变化: {format_delta(res_candidate['balanced_accuracy'], res_base['balanced_accuracy'])})"
    )
    print(
        f"最差类别准确率 worst_class_acc      : "
        f"{res_base['worst_class_acc']:.2%} -> {res_candidate['worst_class_acc']:.2%} "
        f"(变化: {format_delta(res_candidate['worst_class_acc'], res_base['worst_class_acc'])})"
    )
    print(
        f"自动目标类准确率 selected_target_acc: "
        f"{res_base['selected_target_acc']:.2%} -> {res_candidate['selected_target_acc']:.2%} "
        f"(变化: {format_delta(res_candidate['selected_target_acc'], res_base['selected_target_acc'])})"
    )
    print(
        f"自动基座类准确率 selected_base_acc  : "
        f"{res_base['selected_base_acc']:.2%} -> {res_candidate['selected_base_acc']:.2%} "
        f"(变化: {format_delta(res_candidate['selected_base_acc'], res_base['selected_base_acc'])})"
    )
    print(
        f"目标类错分为基座类比例 target_to_base_rate: "
        f"{res_base['target_to_base_rate']:.2%} -> {res_candidate['target_to_base_rate']:.2%} "
        f"(变化: {format_delta(res_candidate['target_to_base_rate'], res_base['target_to_base_rate'])})"
    )
    print("-" * 70)
    print(f"详细结果已保存到: {args.save_dir}/")

    if res_candidate["selected_target_acc"] > res_base["selected_target_acc"]:
        print(f">>> 结论：{args.candidate_name} 提升了自动弱势目标类集合上的识别能力。")
    else:
        print(">>> 结论：自动弱势目标类准确率提升不明显，需要结合 step2/step3 的合成质量和训练日志继续分析。")
    print("=" * 70)
