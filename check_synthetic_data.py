import torch
import os
import argparse
import pandas as pd
from dataset_specs import build_sscbm, load_dataset_spec

# --- 配置 ---
GEN_DATA_DIR = "generated_data"
CHECKPOINT_BASE = "checkpoints/CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """解析合成数据检查参数，默认兼容旧目录，也可指定 D-CGFS 新目录。"""
    parser = argparse.ArgumentParser(description="检查 D-CGFS 合成特征文件是否能被 SSCBM 正常读取和预测")
    parser.add_argument("--dataset", default="CUB-200-2011", choices=["CUB-200-2011", "AwA2", "PBC", "7pt"])
    parser.add_argument("--checkpoint", default=None, help="Baseline SSCBM checkpoint；不填使用数据集默认路径。")
    parser.add_argument("--gen-data-dir", default=GEN_DATA_DIR)
    parser.add_argument("--num-samples", type=int, default=5)
    return parser.parse_args()


def check_quality(args=None):
    """抽查合成特征文件，确认 metadata、.pt 字段和 predict_from_features 路径一致。"""
    if args is None:
        args = parse_args()

    # 1. 加载原始 Baseline 模型
    spec = load_dataset_spec(dataset=args.dataset, checkpoint_path=args.checkpoint)
    model = build_sscbm(spec, DEVICE)

    # 2. 读取前 5 个合成的 .pt 文件进行测试
    metadata_path = os.path.join(args.gen_data_dir, "synthesized_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"未找到 {metadata_path}，请先运行 step2_synthesize_data.py。")
    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        raise ValueError(f"{metadata_path} 为空，说明没有合成样本通过过滤。")

    print(f"{'文件名':<20} | {'保存的标签':<10} | {'原始模型预测':<10} | {'是否匹配'}")
    print("-" * 60)

    for i in range(min(args.num_samples, len(metadata))):
        feat_path = os.path.join(args.gen_data_dir, metadata.iloc[i]['feat_path'])
        data = torch.load(feat_path)

        feature = data['feature'].unsqueeze(0).to(DEVICE)  # [1, 10, 10, 512]
        pos_emb = data['pos_embedding'].unsqueeze(0).to(DEVICE)
        neg_emb = data['neg_embedding'].unsqueeze(0).to(DEVICE)
        concept_probs = data['concept_probs_base'].unsqueeze(0).to(DEVICE)
        saved_label = data['label'].item()

        # 从特征和正/负概念 embedding 预测，复用 SSCBM 原始 concept-to-label 路径。
        _, task_logits = model.predict_from_features(feature, pos_emb, neg_emb, concept_probs)
        pred_label = torch.argmax(task_logits, dim=1).item()

        match = "YES" if saved_label == pred_label else "NO"
        print(f"{metadata.iloc[i]['feat_path']:<20} | {saved_label:<10} | {pred_label:<10} | {match}")


if __name__ == "__main__":
    check_quality()
