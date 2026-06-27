#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 D-CGFS 合成样本的 pair 内质量分布。

这个脚本用于定位 D-CGFS 当前落后 strong baseline 的合成数据原因：
1. top-k 保留下来的样本 target_prob 是否仍然过低；
2. 提高 pair-score target weight 后，样本质量分布是否真的改善；
3. top-k 样本中是否存在大量重复的 target/base 图像组合。

脚本只读取 step2 产物，不重新生成数据，也不训练模型。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件，并在缺失时直接报错，避免生成误导性的空诊断。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """写出结构化诊断 CSV，方便后续直接汇总到论文表格或实验记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: dict[str, str], key: str) -> float:
    """把 CSV 字符串安全转为浮点数。"""
    return float(row[key])


def quantile(values: list[float], q: float) -> float:
    """计算简单分位数；样本量固定较小，用线性插值足够。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def summarize_pair(rows: list[dict[str, str]], method: str, pair_id: int) -> dict[str, object]:
    """统计单个方法、单个 pair 的 top-k 合成样本质量。"""
    pair_rows = [row for row in rows if int(row["pair_id"]) == pair_id]
    target_probs = [to_float(row, "target_prob") for row in pair_rows]
    base_probs = [to_float(row, "base_prob") for row in pair_rows]
    concept_deltas = [to_float(row, "concept_delta") for row in pair_rows]
    scores = [to_float(row, "pair_quality_score") for row in pair_rows]

    source_pairs = [(row["target_src"], row["base_src"]) for row in pair_rows]
    unique_source_pairs = set(source_pairs)
    unique_target_src = {row["target_src"] for row in pair_rows}
    unique_base_src = {row["base_src"] for row in pair_rows}

    # 这些阈值不是过滤规则，只用于诊断：当前 target_prob 普遍极低，
    # 因此用多个数量级阈值看 top-k 是否包含足够“像目标类”的样本。
    thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    threshold_counts = {
        f"target_prob_ge_{threshold:g}": sum(value >= threshold for value in target_probs)
        for threshold in thresholds
    }

    return {
        "method": method,
        "pair_id": pair_id,
        "target_class": pair_rows[0]["target_class"] if pair_rows else "",
        "base_class": pair_rows[0]["base_class"] if pair_rows else "",
        "kept_samples": len(pair_rows),
        "unique_source_pairs": len(unique_source_pairs),
        "unique_target_src": len(unique_target_src),
        "unique_base_src": len(unique_base_src),
        "duplicate_rate": 1.0 - len(unique_source_pairs) / max(len(pair_rows), 1),
        "target_prob_min": min(target_probs) if target_probs else 0.0,
        "target_prob_p25": quantile(target_probs, 0.25),
        "target_prob_median": quantile(target_probs, 0.50),
        "target_prob_p75": quantile(target_probs, 0.75),
        "target_prob_max": max(target_probs) if target_probs else 0.0,
        "target_prob_mean": mean(target_probs) if target_probs else 0.0,
        "base_prob_median": quantile(base_probs, 0.50),
        "base_prob_max": max(base_probs) if base_probs else 0.0,
        "concept_delta_median": quantile(concept_deltas, 0.50),
        "concept_delta_mean": mean(concept_deltas) if concept_deltas else 0.0,
        "concept_delta_max": max(concept_deltas) if concept_deltas else 0.0,
        "score_median": quantile(scores, 0.50),
        "score_max": max(scores) if scores else 0.0,
        **threshold_counts,
    }


def compare_methods(summary_rows: list[dict[str, object]], baseline_name: str, candidate_name: str) -> list[dict[str, object]]:
    """比较两种合成策略在每个 pair 上的质量变化。"""
    by_key = {(row["method"], int(row["pair_id"])): row for row in summary_rows}
    pair_ids = sorted({int(row["pair_id"]) for row in summary_rows})
    rows: list[dict[str, object]] = []
    for pair_id in pair_ids:
        base = by_key[(baseline_name, pair_id)]
        cand = by_key[(candidate_name, pair_id)]
        rows.append(
            {
                "pair_id": pair_id,
                "target_class": cand["target_class"],
                "base_class": cand["base_class"],
                "target_prob_median_diff": float(cand["target_prob_median"]) - float(base["target_prob_median"]),
                "target_prob_max_diff": float(cand["target_prob_max"]) - float(base["target_prob_max"]),
                "target_prob_ge_1e-4_diff": int(cand["target_prob_ge_0.0001"]) - int(base["target_prob_ge_0.0001"]),
                "target_prob_ge_1e-3_diff": int(cand["target_prob_ge_0.001"]) - int(base["target_prob_ge_0.001"]),
                "concept_delta_mean_diff": float(cand["concept_delta_mean"]) - float(base["concept_delta_mean"]),
                "duplicate_rate_diff": float(cand["duplicate_rate"]) - float(base["duplicate_rate"]),
            }
        )
    return rows


def write_markdown(path: Path, summary_rows: list[dict[str, object]], comparison_rows: list[dict[str, object]]) -> None:
    """写出中文 Markdown 报告，便于快速判断下一步是否该改 top-k 策略。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Synthetic Quality Distribution Diagnosis\n")
    lines.append("## Pair 内 top-k 合成样本质量\n")
    lines.append(
        "| Method | Pair | Target | Kept | Unique Pairs | Dup Rate | Target Prob Median | Target Prob Max | >=1e-4 | >=1e-3 | Concept Delta Mean |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            "| {method} | {pair_id} | {target_class} | {kept} | {unique_pairs} | {dup:.2%} | {median:.3e} | {maxv:.3e} | {ge4} | {ge3} | {delta:.4f} |".format(
                method=row["method"],
                pair_id=row["pair_id"],
                target_class=row["target_class"],
                kept=row["kept_samples"],
                unique_pairs=row["unique_source_pairs"],
                dup=float(row["duplicate_rate"]),
                median=float(row["target_prob_median"]),
                maxv=float(row["target_prob_max"]),
                ge4=row["target_prob_ge_0.0001"],
                ge3=row["target_prob_ge_0.001"],
                delta=float(row["concept_delta_mean"]),
            )
        )

    lines.append("\n## target_score_w015 相对主合成策略的变化\n")
    lines.append(
        "| Pair | Target | Target Prob Median Diff | Target Prob Max Diff | >=1e-4 Diff | >=1e-3 Diff | Concept Delta Mean Diff | Dup Rate Diff |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in comparison_rows:
        lines.append(
            "| {pair_id} | {target_class} | {median:.3e} | {maxv:.3e} | {ge4:+d} | {ge3:+d} | {delta:.4f} | {dup:.2%} |".format(
                pair_id=row["pair_id"],
                target_class=row["target_class"],
                median=float(row["target_prob_median_diff"]),
                maxv=float(row["target_prob_max_diff"]),
                ge4=int(row["target_prob_ge_1e-4_diff"]),
                ge3=int(row["target_prob_ge_1e-3_diff"]),
                delta=float(row["concept_delta_mean_diff"]),
                dup=float(row["duplicate_rate_diff"]),
            )
        )

    lines.append("\n## 诊断结论\n")
    lines.append(
        "如果 target_score_w015 只提高了 target_prob 的极高分样本数量，但 target accuracy 没有提升，说明问题可能不是排序权重本身，而是 top-k 中重复样本过多或合成样本语义仍不足。"
    )
    lines.append(
        "如果某些 pair 的 duplicate rate 很高，下一步应优先加入去重或每个 target/base 源图的保留上限，而不是继续提高 synthetic loss。"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 D-CGFS 合成样本质量分布。")
    parser.add_argument("--main-dir", default="generated_data/dcgfs_pair_topk")
    parser.add_argument("--candidate-dir", default="generated_data/problem6_target_score_w015")
    parser.add_argument("--output-dir", default="diagnostic_results/synthetic_quality_distribution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [
        ("main_pair_topk_w005", Path(args.main_dir) / "synthesized_metadata.csv"),
        ("target_score_w015", Path(args.candidate_dir) / "synthesized_metadata.csv"),
    ]
    summary_rows: list[dict[str, object]] = []
    for method, path in datasets:
        rows = read_csv(path)
        pair_ids = sorted({int(row["pair_id"]) for row in rows})
        for pair_id in pair_ids:
            summary_rows.append(summarize_pair(rows, method, pair_id))

    comparison_rows = compare_methods(summary_rows, "main_pair_topk_w005", "target_score_w015")
    output_dir = Path(args.output_dir)
    summary_fields = list(summary_rows[0].keys())
    comparison_fields = list(comparison_rows[0].keys())
    write_csv(output_dir / "pair_quality_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "target_score_vs_main_diff.csv", comparison_rows, comparison_fields)
    write_markdown(output_dir / "synthetic_quality_report.md", summary_rows, comparison_rows)

    print(f"合成质量分布诊断已保存到: {output_dir}")
    print(f"- {output_dir / 'pair_quality_summary.csv'}")
    print(f"- {output_dir / 'target_score_vs_main_diff.csv'}")
    print(f"- {output_dir / 'synthetic_quality_report.md'}")


if __name__ == "__main__":
    main()
