# ==========================================================================================
# SSCBM + D-CGFS 研究项目 - 问题六：不同不平衡比例数据构造
#
# 核心目标：
# 生成 imbalance ratio = 10/50/100 等 long-tail 训练集，支持顶会级不平衡实验。
#
# 输入：
# data/CUB_200_2011/class_attr_data_10/train.pkl
# data/CUB_200_2011/class_attr_data_10/val.pkl
# data/CUB_200_2011/class_attr_data_10/test.pkl
#
# 输出示例：
# data/CUB_200_2011/class_attr_data_ir10/train.pkl
# data/CUB_200_2011/class_attr_data_ir10/val.pkl
# data/CUB_200_2011/class_attr_data_ir10/test.pkl
#
# 设计说明：
# - 只重采样 train.pkl，val/test 原样复制，保证最终评估集一致。
# - 使用指数 long-tail 规则：
#       n_i = n_max * (1 / IR) ** (rank_i / (C - 1))
#   其中 IR 是最大类和最小类的目标样本数比例。
# - 类别顺序默认按 class id 排列；这不是为了模拟真实 CUB 分布，而是为了构造可控实验。
#
# 作者：[肖凡]
# 日期：[2026年5月18日]
# ==========================================================================================

import argparse
import os
import pickle
import random
import shutil
from collections import defaultdict


DATA_ROOT = "data/CUB_200_2011"
SOURCE_DIR = "class_attr_data_10"
N_CLASSES = 200


def load_pickle(path):
    """读取 CUB 原始 pkl 数据。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, path):
    """保存新的 pkl 数据。"""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def build_long_tail_train_split(train_data, imbalance_ratio, seed):
    """
    构造指数 long-tail 训练集。

    每个类别至少保留 1 个样本，避免某些类别完全消失导致分类头无法学习。
    """
    rng = random.Random(seed)
    class_to_items = defaultdict(list)
    for item in train_data:
        class_to_items[int(item["class_label"])].append(item)

    max_count = max(len(items) for items in class_to_items.values())
    new_train = []
    class_count_rows = []

    for class_id in range(N_CLASSES):
        items = list(class_to_items[class_id])
        rng.shuffle(items)

        # rank 越靠后，保留样本越少；最后一个类别约为 max_count / imbalance_ratio。
        keep_ratio = (1.0 / float(imbalance_ratio)) ** (class_id / (N_CLASSES - 1))
        desired_count = int(round(max_count * keep_ratio))
        keep_count = max(1, min(len(items), desired_count))

        new_train.extend(items[:keep_count])
        class_count_rows.append(
            {
                "class_id": class_id + 1,
                "original_count": len(items),
                "kept_count": keep_count,
            }
        )

    rng.shuffle(new_train)
    return new_train, class_count_rows


def prepare_split(imbalance_ratio, seed):
    """生成单个 imbalance ratio 对应的数据目录。"""
    source_dir = os.path.join(DATA_ROOT, SOURCE_DIR)
    target_dir_name = f"class_attr_data_ir{imbalance_ratio}"
    target_dir = os.path.join(DATA_ROOT, target_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    train_data = load_pickle(os.path.join(source_dir, "train.pkl"))
    new_train, class_count_rows = build_long_tail_train_split(train_data, imbalance_ratio, seed)
    save_pickle(new_train, os.path.join(target_dir, "train.pkl"))

    # val/test 不参与重采样，保持所有实验在同一验证/测试分布上比较。
    for split in ["val.pkl", "test.pkl"]:
        shutil.copyfile(os.path.join(source_dir, split), os.path.join(target_dir, split))

    # 保存类别样本数，方便论文中报告实际 imbalance ratio。
    csv_path = os.path.join(target_dir, "class_counts.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class_id,original_count,kept_count\n")
        for row in class_count_rows:
            f.write(f"{row['class_id']},{row['original_count']},{row['kept_count']}\n")

    print(f"已生成 IR={imbalance_ratio} 数据目录: {target_dir}")
    print(f"训练样本数: {len(train_data)} -> {len(new_train)}")
    return target_dir_name


def main():
    parser = argparse.ArgumentParser(description="生成 CUB long-tail imbalance split")
    parser.add_argument("--ratios", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for ratio in args.ratios:
        prepare_split(ratio, args.seed)


if __name__ == "__main__":
    main()
