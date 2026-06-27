import argparse
import os
from pathlib import Path

import yaml


EXPECTED_AWA2_FILES = [
    "classes.txt",
    "predicate-matrix-binary.txt",
]

EXPECTED_PBC_FILES = [
    "PBC_dataset_normal_DIB/pbc_attr_v1_train.csv",
    "PBC_dataset_normal_DIB/pbc_attr_v1_val.csv",
    "PBC_dataset_normal_DIB/pbc_attr_v1_test.csv",
]

EXPECTED_7PT_FILES = [
    "meta/meta.csv",
    "meta/train_indexes.csv",
    "meta/valid_indexes.csv",
    "meta/test_indexes.csv",
]


def check_exists(path):
    return Path(path).exists()


def status_line(ok, label, detail):
    mark = "OK" if ok else "MISSING"
    return f"[{mark}] {label}: {detail}"


def load_dataset_config(dataset):
    config_path = Path("configs") / f"{dataset}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config_path, config


def check_awa2():
    config_path, config = load_dataset_config("AwA2")
    dataset_config = config["dataset_config"]
    root_dir = Path(dataset_config["root_dir"])

    checks = []
    checks.append((config_path.exists(), "AwA2 配置文件", str(config_path)))
    checks.append((root_dir.exists(), "AwA2 数据目录", str(root_dir)))

    for rel_path in EXPECTED_AWA2_FILES:
        checks.append((check_exists(root_dir / rel_path), f"AwA2 元数据文件 {rel_path}", str(root_dir / rel_path)))

    image_dir = root_dir / "JPEGImages"
    checks.append((image_dir.exists(), "AwA2 图像目录 JPEGImages", str(image_dir)))
    if image_dir.exists():
        class_dirs = [p for p in image_dir.iterdir() if p.is_dir()]
        jpg_count = sum(1 for _ in image_dir.rglob("*.jpg"))
        checks.append((len(class_dirs) >= 50, "AwA2 类别图像子目录数量", f"{len(class_dirs)} / 50"))
        checks.append((jpg_count > 0, "AwA2 jpg 图像数量", str(jpg_count)))

    checkpoint_candidates = list(Path("checkpoints").glob("AwA2*/*.pt"))
    checks.append((bool(checkpoint_candidates), "AwA2 预训练 SSCBM checkpoint", "、".join(str(p) for p in checkpoint_candidates) or "未找到"))

    print("第二数据集 AwA2 准备状态检查")
    print("=" * 70)
    for ok, label, detail in checks:
        print(status_line(ok, label, detail))

    print("\n当前结论")
    if all(ok for ok, _, _ in checks):
        print("AwA2 数据和预训练 checkpoint 已就绪，可以进入 AwA2 baseline 评估和 D-CGFS 迁移。")
    else:
        print("AwA2 尚未就绪。需要先补齐数据目录和 AwA2 预训练 SSCBM checkpoint。")


def _checkpoint_candidates(dataset):
    return list(Path("checkpoints").glob(f"{dataset}*/*.pt"))


def _print_checks(title, checks, ready_message, missing_message):
    print(title)
    print("=" * 70)
    for ok, label, detail in checks:
        print(status_line(ok, label, detail))

    print("\n当前结论")
    if all(ok for ok, _, _ in checks):
        print(ready_message)
    else:
        print(missing_message)


def check_pbc():
    config_path, config = load_dataset_config("PBC")
    dataset_config = config["dataset_config"]
    root_dir = Path(dataset_config["root_dir"])

    checks = [
        (config_path.exists(), "PBC/WBCatt 配置文件", str(config_path)),
        (root_dir.exists(), "PBC/WBCatt 数据根目录", str(root_dir)),
    ]
    for rel_path in EXPECTED_PBC_FILES:
        checks.append((check_exists(root_dir / rel_path), f"PBC/WBCatt 元数据文件 {rel_path}", str(root_dir / rel_path)))

    if root_dir.exists():
        image_count = sum(1 for path in root_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        checks.append((image_count > 0, "PBC/WBCatt 图像文件数量", str(image_count)))

    checkpoint_candidates = _checkpoint_candidates("PBC")
    checks.append((bool(checkpoint_candidates), "PBC/WBCatt 预训练 SSCBM checkpoint", "、".join(str(p) for p in checkpoint_candidates) or "未找到"))

    _print_checks(
        "第三数据集 PBC/WBCatt 准备状态检查",
        checks,
        "PBC/WBCatt 数据和预训练 checkpoint 已就绪，可以进入 D-CGFS 适配和正式实验。",
        "PBC/WBCatt 尚未就绪。需要先补齐数据目录和预训练 SSCBM checkpoint。",
    )


def check_7pt():
    config_path, config = load_dataset_config("7pt")
    dataset_config = config["dataset_config"]
    root_dir = Path(dataset_config["root_dir"])

    checks = [
        (config_path.exists(), "7-point 配置文件", str(config_path)),
        (root_dir.exists(), "7-point 数据根目录", str(root_dir)),
    ]
    for rel_path in EXPECTED_7PT_FILES:
        checks.append((check_exists(root_dir / rel_path), f"7-point 元数据文件 {rel_path}", str(root_dir / rel_path)))

    image_dir = root_dir / "images"
    checks.append((image_dir.exists(), "7-point 图像目录 images", str(image_dir)))
    if image_dir.exists():
        image_count = sum(1 for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        checks.append((image_count > 0, "7-point 图像文件数量", str(image_count)))

    checkpoint_candidates = _checkpoint_candidates("7pt")
    checks.append((bool(checkpoint_candidates), "7-point 预训练 SSCBM checkpoint", "、".join(str(p) for p in checkpoint_candidates) or "未找到"))

    _print_checks(
        "第四数据集 7-point 准备状态检查",
        checks,
        "7-point 数据和预训练 checkpoint 已就绪，可以进入 D-CGFS 适配和正式实验。",
        "7-point 尚未就绪。需要先补齐数据目录和预训练 SSCBM checkpoint。",
    )


def main():
    parser = argparse.ArgumentParser(description="检查第二数据集是否具备开始实验的最低条件。")
    parser.add_argument("--dataset", default="AwA2", choices=["AwA2", "PBC", "7pt"], help="选择要检查的数据集。")
    args = parser.parse_args()

    if args.dataset == "AwA2":
        check_awa2()
    elif args.dataset == "PBC":
        check_pbc()
    elif args.dataset == "7pt":
        check_7pt()


if __name__ == "__main__":
    main()
