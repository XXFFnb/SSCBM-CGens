# ==========================================================================================
# SSCBM + D-CGFS 研究项目 - 步骤 1: 生成概念区域映射
#
# 核心目标：
# 利用一个预训练好的 SSCBM 模型，为数据集中特定类别的图像生成“概念激活热图” (Concept Heatmap)，
# 并将这些热图转化为二值化的“语义区域掩码” (Semantic Region Mask)。
#
# 工作流程：
# 1. 加载一个在目标数据集上预训练好的 SSCBM 模型。
# 2. 读取一个或多个“目标类 (Target Class)”和“基座类 (Base Class)”配对。
#    - 必须先运行 find_weak_classes.py 自动生成 target_base_pairs.csv。
#    - 目标类：通常是我们希望增强的、表现不佳的弱势类。
#    - 基座类：与目标类视觉/概念上相似的类别，我们将从它身上“借用”背景和非关键区域。
# 3. 遍历目标类和基座类的所有图像。
# 4. 对每张图像，使用 SSCBM 的 `plot_heatmap` 方法生成概念激活热图。
# 5. 只选择 find_weak_classes.py 计算出的判别概念集合 D，并融合这些概念的热图。
#    即 M(i,j)=max_{k in D} H_k(i,j)，不再对所有概念取 max。
# 6. 将图像路径、掩码路径和 pair_id 保存到 CSV 文件中，作为步骤 2 的输入。
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
from torchvision import transforms
from dataset_specs import build_sscbm, load_dataset_config, load_dataset_spec

# --- 1. 全局配置 ---
# CUB 数据集根目录
DATA_ROOT = "data/CUB_200_2011"
# 预训练 SSCBM 模型权重路径
CHECKPOINT_PATH = "checkpoints/CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt"
# 生成的辅助文件 (掩码和 CSV) 的保存目录
SAVE_DIR = "data/D-CGFS_Auxiliary"
# 自动目标类-基座类选择结果，由 find_weak_classes.py 生成
PAIR_CSV_PATH = os.path.join(SAVE_DIR, "target_base_pairs.csv")
# 确保掩码目录存在
os.makedirs(os.path.join(SAVE_DIR, "masks"), exist_ok=True)

# 运行设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_SPEC = None
IMAGE_SUBDIR = "images"

# 图像预处理：必须与训练 SSCBM 时所用的预处理保持完全一致
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_cub_images_by_class(class_id):
    """
    辅助函数：根据 CUB 数据集的官方索引文件，筛选出指定类别的所有图像路径。
    Args:
        class_id (int): 类别 ID (1-based)。
    Returns:
        list: 该类别所有图像的相对路径列表。
    """
    images_txt = os.path.join(DATA_ROOT, "images.txt")
    labels_txt = os.path.join(DATA_ROOT, "image_class_labels.txt")

    df_img = pd.read_csv(images_txt, sep=' ', names=['id', 'path'])
    df_lab = pd.read_csv(labels_txt, sep=' ', names=['id', 'class'])

    merged = pd.merge(df_img, df_lab, on='id')
    class_images = merged[merged['class'] == class_id]['path'].tolist()
    return class_images


def get_awa2_images_by_class(class_id):
    class_idx = class_id - CURRENT_SPEC.class_id_base
    class_name = CURRENT_SPEC.class_names[class_idx]
    class_dir = os.path.join(DATA_ROOT, "JPEGImages", class_name)
    if not os.path.isdir(class_dir):
        return []
    return [
        os.path.join(class_name, filename)
        for filename in sorted(os.listdir(class_dir))
        if filename.lower().endswith(".jpg")
    ]


def get_pbc_images_by_class(class_id):
    class_idx = class_id - CURRENT_SPEC.class_id_base
    class_name = CURRENT_SPEC.class_names[class_idx]
    csv_paths = [
        os.path.join(DATA_ROOT, "PBC_dataset_normal_DIB", "pbc_attr_v1_train.csv"),
        os.path.join(DATA_ROOT, "PBC_dataset_normal_DIB", "pbc_attr_v1_val.csv"),
        os.path.join(DATA_ROOT, "PBC_dataset_normal_DIB", "pbc_attr_v1_test.csv"),
    ]
    frames = [pd.read_csv(path) for path in csv_paths if os.path.exists(path)]
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    return sorted(df[df["label"] == class_name]["path"].astype(str).tolist())


def _build_7pt_file_mapping(image_dir):
    mapping = {}
    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, image_dir)
                mapping[rel_path.lower()] = rel_path
    return mapping


def get_7pt_images_by_class(class_id):
    diagnosis_groups = {
        "basal cell carcinoma": 0,
        "blue nevus": 1, "clark nevus": 1, "combined nevus": 1,
        "congenital nevus": 1, "dermal nevus": 1, "recurrent nevus": 1,
        "reed or spitz nevus": 1,
        "melanoma": 2, "melanoma (in situ)": 2, "melanoma (less than 0.76 mm)": 2,
        "melanoma (0.76 to 1.5 mm)": 2, "melanoma (more than 1.5 mm)": 2,
        "melanoma metastasis": 2,
        "dermatofibroma": 3, "lentigo": 3, "melanosis": 3,
        "miscellaneous": 3, "vascular lesion": 3,
        "seborrheic keratosis": 4,
    }
    meta_path = os.path.join(DATA_ROOT, "meta", "meta.csv")
    image_dir = os.path.join(DATA_ROOT, "images")
    if not os.path.exists(meta_path) or not os.path.isdir(image_dir):
        return []
    file_mapping = _build_7pt_file_mapping(image_dir)
    df = pd.read_csv(meta_path)
    rows = df[df["diagnosis"].map(diagnosis_groups) == (class_id - CURRENT_SPEC.class_id_base)]
    image_paths = []
    for image_name in rows["derm"].astype(str):
        image_paths.append(file_mapping.get(image_name.lower(), image_name))
    return sorted(image_paths)


def get_images_by_class(class_id):
    if CURRENT_SPEC.name == "CUB-200-2011":
        return get_cub_images_by_class(class_id)
    if CURRENT_SPEC.name == "AwA2":
        return get_awa2_images_by_class(class_id)
    if CURRENT_SPEC.name == "PBC":
        return get_pbc_images_by_class(class_id)
    if CURRENT_SPEC.name == "7pt":
        return get_7pt_images_by_class(class_id)
    raise ValueError(f"不支持的数据集: {CURRENT_SPEC.name}")


def get_image_subdir(spec_name):
    if spec_name == "CUB-200-2011":
        return "images"
    if spec_name == "AwA2":
        return "JPEGImages"
    if spec_name == "7pt":
        return "images"
    if spec_name == "PBC":
        return ""
    raise ValueError(f"不支持的数据集: {spec_name}")


def load_target_base_pairs():
    """
    读取自动目标类-基座类配对。
    本步骤强制依赖 find_weak_classes.py 的输出，不再提供手动类别回退。

    返回:
        DataFrame，每行是一组 pair，至少包含：
            pair_id: 该目标类-基座类配对的唯一编号。
            target_class_id: 目标类 ID，使用 CUB 官方 1-based 编号。
            base_class_id: 基座类 ID，使用 CUB 官方 1-based 编号。
            discriminative_concepts: 判别概念集合，0-based 概念索引用分号分隔。
            target_concept_proto: 目标类概念原型，用于 step2 概念一致性过滤。
            base_concept_proto: 基座类概念原型，用于 step2 概念一致性过滤。
    """
    if not os.path.exists(PAIR_CSV_PATH):
        raise FileNotFoundError(
            f"未找到自动目标类-基座类配对文件: {PAIR_CSV_PATH}。"
            "请先运行 find_weak_classes.py 生成该文件。"
        )

    pair_df = pd.read_csv(PAIR_CSV_PATH)
    required_cols = {
        "pair_id",
        "target_class_id",
        "base_class_id",
        "discriminative_concepts",
        "target_concept_proto",
        "base_concept_proto",
    }
    missing = required_cols - set(pair_df.columns)
    if missing:
        raise ValueError(f"{PAIR_CSV_PATH} 缺少必要列: {sorted(missing)}")
    if pair_df.empty:
        raise ValueError(f"{PAIR_CSV_PATH} 为空，请重新运行 find_weak_classes.py。")

    print(f"已读取自动目标类-基座类配对: {PAIR_CSV_PATH}")
    return pair_df


def parse_discriminative_concepts(concept_str, n_concepts=None):
    """
    解析 target_base_pairs.csv 中保存的判别概念集合。

    输入:
        concept_str:
            形如 "3;17;42" 的字符串，表示 0-based 概念索引。
        n_concepts:
            SSCBM 的概念数量，用于检查索引范围。

    返回:
        list[int]，可以直接用于 heatmaps_np[concept_indices]。
    """
    if pd.isna(concept_str) or str(concept_str).strip() == "":
        raise ValueError("discriminative_concepts 为空，请重新运行 find_weak_classes.py。")

    concept_indices = [int(item) for item in str(concept_str).split(";") if item.strip() != ""]
    if not concept_indices:
        raise ValueError("未解析到任何判别概念，请检查 target_base_pairs.csv。")

    n_concepts = n_concepts or CURRENT_SPEC.n_concepts
    invalid = [idx for idx in concept_indices if idx < 0 or idx >= n_concepts]
    if invalid:
        raise ValueError(f"判别概念索引越界: {invalid}，合法范围是 [0, {n_concepts - 1}]。")
    return concept_indices


@torch.no_grad()
def process_images(
        model,
        image_paths,
        class_id,
        mapping_list,
        prefix,
        pair_id,
        target_class_id,
        base_class_id,
        discriminative_concepts,
        target_concept_proto,
        base_concept_proto,
):
    """
    核心处理函数：为一批图像生成并保存其对应的概念区域掩码。
    Args:
        model (SSCBM): 预训练的 SSCBM 模型。
        image_paths (list): 待处理的图像路径列表。
        class_id (int): 当前处理的类别 ID。
        mapping_list (list): 用于收集图像-掩码映射关系的列表。
        prefix (str): 保存掩码文件时的前缀 (例如 'target' 或 'base')。
        pair_id (int): 该图像所属的目标类-基座类配对编号。
        target_class_id (int): 当前 pair 的目标类 ID。
        base_class_id (int): 当前 pair 的基座类 ID。
        discriminative_concepts (list[int]): 当前 pair 的判别概念集合，0-based。
        target_concept_proto (str): 当前 pair 的目标类概念原型。
        base_concept_proto (str): 当前 pair 的基座类概念原型。
    """
    print(f"  正在处理类别 ID: {class_id} ({prefix}), 共 {len(image_paths)} 张图片...")
    for img_rel_path in image_paths:
        img_full_path = os.path.join(DATA_ROOT, IMAGE_SUBDIR, img_rel_path)
        raw_img = Image.open(img_full_path).convert('RGB')
        input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

        # --- D-CGFS 关键步骤：判别概念定位 ---
        # 1. 调用 SSCBM 的 plot_heatmap 方法，获取模型认为的、这张图上所有概念的激活区域。
        # 返回的 heatmaps 形状为 [1, n_concepts, H, W]，例如 [1, 112, 10, 10]。
        heatmaps = model.plot_heatmap(input_tensor)
        
        # 将 PyTorch tensor 转换为 numpy 数组，并去掉 batch 维度
        heatmaps_np = heatmaps.squeeze(0).cpu().numpy()

        # 2. 只融合判别概念集合 D 中的热图，生成“判别性概念区域图”。
        # 旧做法是 np.max(heatmaps_np, axis=0)，会得到任意概念强响应区域；
        # 新做法是 max_{k in D} H_k(i,j)，强调目标类相对于基座类更强的概念区域。
        selected_heatmaps = heatmaps_np[discriminative_concepts]
        combined_heatmap = np.max(selected_heatmaps, axis=0)
        
        # 3. 将低分辨率的热图上采样到原始图像尺寸，以便后续可能的可视化或精确操作。
        combined_heatmap_img = Image.fromarray((combined_heatmap * 255).astype(np.uint8), mode='L')
        combined_heatmap_img = combined_heatmap_img.resize(raw_img.size, Image.Resampling.BILINEAR)
        combined_heatmap_resized = np.array(combined_heatmap_img) / 255.0

        # 4. 二值化处理，生成最终的区域掩码。
        # 阈值 (如 0.6) 是一个超参数，用于判断多大的激活强度才算“显著”。
        mask = (combined_heatmap_resized > 0.6).astype(np.uint8) * 255

        # 5. 保存掩码图片，并记录映射关系。
        # 文件名加入 pair_id，避免多个 pair 中不同类别图片同名时互相覆盖。
        mask_name = f"pair{pair_id}_{prefix}_{os.path.basename(img_rel_path).replace('.jpg', '.png')}"
        mask_path = os.path.join(SAVE_DIR, "masks", mask_name)
        Image.fromarray(mask).save(mask_path)

        # target_class/base_class 会写入 CSV。
        # step2 根据 pair_id 只把同一个 pair 内的 target 和 base 配对，避免跨 pair 错误融合。
        mapping_list.append({
            "pair_id": pair_id,
            "img_path": img_rel_path,
            "mask_path": os.path.join("masks", mask_name),
            "class": class_id,
            "target_class": target_class_id,
            "base_class": base_class_id,
            "discriminative_concepts": ";".join(str(idx) for idx in discriminative_concepts),
            "target_concept_proto": target_concept_proto,
            "base_concept_proto": base_concept_proto,
        })


@torch.no_grad()
def generate_metadata():
    """主函数：执行整个映射文件生成流程。"""
    global CURRENT_SPEC, DATA_ROOT, CHECKPOINT_PATH, SAVE_DIR, PAIR_CSV_PATH, IMAGE_SUBDIR

    args = parse_args()
    CURRENT_SPEC = load_dataset_spec(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        aux_dir=args.save_dir,
    )
    config = load_dataset_config(CURRENT_SPEC)
    DATA_ROOT = config["dataset_config"]["root_dir"]
    CHECKPOINT_PATH = CURRENT_SPEC.checkpoint_path
    SAVE_DIR = CURRENT_SPEC.aux_dir
    PAIR_CSV_PATH = args.pair_csv or os.path.join(SAVE_DIR, "target_base_pairs.csv")
    IMAGE_SUBDIR = get_image_subdir(CURRENT_SPEC.name)
    os.makedirs(os.path.join(SAVE_DIR, "masks"), exist_ok=True)

    # 1. 初始化并加载预训练的 SSCBM 模型
    # 这里的模型参数 (n_concepts, emb_size, c_extractor_arch) 必须与训练时完全一致。
    print("步骤 1: 正在加载预训练的 SSCBM 模型...")
    print(f"当前数据集: {CURRENT_SPEC.name}")
    print(f"当前 checkpoint: {CHECKPOINT_PATH}")
    print(f"输出目录: {SAVE_DIR}")
    model = build_sscbm(CURRENT_SPEC, DEVICE)
    print("SSCBM 模型加载成功。")

    # 用于存储最终 CSV 数据的列表
    concept_mapping = []
    base_mapping = []

    # 2. 获取目标类-基座类配对
    print("\n步骤 2: 正在读取目标类-基座类配对...")
    pair_df = load_target_base_pairs()

    # 3. 为每组目标类和基座类分别生成并保存掩码。
    # 多个 pair 会被追加到同一份 concept_region_mapping.csv / base_sample_regions.csv 中，
    # 后续通过 pair_id 区分它们。
    print("\n步骤 3: 正在为目标类和基座类生成概念区域掩码...")
    for _, pair in pair_df.iterrows():
        pair_id = int(pair["pair_id"])
        target_class_id = int(pair["target_class_id"])
        base_class_id = int(pair["base_class_id"])
        discriminative_concepts = parse_discriminative_concepts(pair["discriminative_concepts"])
        target_concept_proto = str(pair["target_concept_proto"])
        base_concept_proto = str(pair["base_concept_proto"])
        print(f"\nPair {pair_id}: target={target_class_id}, base={base_class_id}")
        print(f"  判别概念集合 D: {discriminative_concepts}")

        target_images = get_images_by_class(target_class_id)
        base_images = get_images_by_class(base_class_id)
        if args.max_images_per_class > 0:
            target_images = target_images[:args.max_images_per_class]
            base_images = base_images[:args.max_images_per_class]
        if not target_images or not base_images:
            print(f"警告：未能找到目标类 {target_class_id} 或基座类 {base_class_id} 的图像，跳过该 pair。")
            continue

        process_images(
            model, target_images, target_class_id, concept_mapping, "target",
            pair_id, target_class_id, base_class_id, discriminative_concepts,
            target_concept_proto, base_concept_proto,
        )
        process_images(
            model, base_images, base_class_id, base_mapping, "base",
            pair_id, target_class_id, base_class_id, discriminative_concepts,
            target_concept_proto, base_concept_proto,
        )

    # 4. 将记录的映射关系导出为 CSV 文件
    print("\n步骤 4: 正在将映射关系导出为 CSV 文件...")
    pd.DataFrame(concept_mapping).to_csv(os.path.join(SAVE_DIR, "concept_region_mapping.csv"), index=False)
    pd.DataFrame(base_mapping).to_csv(os.path.join(SAVE_DIR, "base_sample_regions.csv"), index=False)
    print(f"\n成功！步骤 1 完成。辅助映射文件已生成在: {SAVE_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="生成 D-CGFS 概念区域映射。")
    parser.add_argument("--dataset", default="CUB-200-2011", choices=["CUB-200-2011", "AwA2", "PBC", "7pt"])
    parser.add_argument("--checkpoint", default=None, help="Baseline SSCBM checkpoint；不填使用数据集默认路径。")
    parser.add_argument("--save-dir", default=None, help="辅助映射输出目录；不填使用数据集默认目录。")
    parser.add_argument("--pair-csv", default=None, help="target_base_pairs.csv 路径；不填使用 save-dir 下默认文件。")
    parser.add_argument("--max-images-per-class", type=int, default=0, help="调试用：每个类别最多处理多少张图；0 表示全部。")
    return parser.parse_args()


if __name__ == "__main__":
    generate_metadata()
