#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 D-CGFS 与 strong baseline 在目标类/基座类上的差距。

这个脚本不重新训练模型，只读取 step4 和 step2 已经保存的 CSV 结果，
用于回答一个核心问题：D-CGFS 当前主方法到底输给 strong baseline 的哪些类、
哪些 target-base pair，以及这些失败是否和合成样本质量有关。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV，并在文件缺失时给出清晰错误，避免静默生成错误诊断。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """统一写出 CSV，保证后续论文表格和诊断记录可以直接复用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    """把 0-1 指标格式化成百分比字符串，便于 Markdown 报告阅读。"""
    return f"{value * 100:.2f}%"


def fnum(row: dict[str, str], key: str) -> float:
    """安全转换浮点数，减少主逻辑里的重复代码。"""
    return float(row[key])


def load_candidate_rows(eval_dirs: dict[str, Path]) -> list[dict[str, object]]:
    """读取每个方法相对于自身 baseline 的总体指标。

    step4 的 model_summary.csv 第一行通常是原始 SSCBM baseline，
    最后一行是当前候选方法。这里把 baseline 单独保留，避免重复统计。
    """
    rows: list[dict[str, object]] = []
    baseline_added = False
    metric_keys = [
        "overall_a_acc",
        "macro_f1",
        "balanced_accuracy",
        "worst_class_acc",
        "selected_target_acc",
        "selected_base_acc",
        "target_to_base_rate",
        "overall_c_acc",
    ]
    for method, eval_dir in eval_dirs.items():
        summary = read_csv(eval_dir / "model_summary.csv")
        if not baseline_added:
            base = summary[0]
            rows.append({"method": "SSCBM baseline", **{k: fnum(base, k) for k in metric_keys}})
            baseline_added = True
        candidate = summary[-1]
        rows.append({"method": method, **{k: fnum(candidate, k) for k in metric_keys}})
    return rows


def load_pair_rows(eval_dirs: dict[str, Path]) -> list[dict[str, object]]:
    """读取 target-base pair 级别的准确率和混淆率。"""
    rows: list[dict[str, object]] = []
    baseline_added = False
    for method, eval_dir in eval_dirs.items():
        pair_rows = read_csv(eval_dir / "target_base_confusion.csv")
        for row in pair_rows:
            model = row["model"]
            is_baseline = "Baseline" in model
            if is_baseline and baseline_added:
                continue
            out_method = "SSCBM baseline" if is_baseline else method
            rows.append(
                {
                    "method": out_method,
                    "pair_id": int(row["pair_id"]),
                    "target_class_id": int(row["target_class_id"]),
                    "target_class_name": row["target_class_name"],
                    "base_class_id": int(row["base_class_id"]),
                    "base_class_name": row["base_class_name"],
                    "target_samples": int(row["target_samples"]),
                    "target_correct": int(row["target_correct"]),
                    "target_accuracy": fnum(row, "target_accuracy"),
                    "target_pred_as_base": int(row["target_pred_as_base"]),
                    "target_to_base_rate": fnum(row, "target_to_base_rate"),
                }
            )
        baseline_added = True
    return rows


def summarize_gaps(pair_rows: list[dict[str, object]], dcgfs_name: str) -> list[dict[str, object]]:
    """计算每个 pair 上 D-CGFS 与最佳 strong baseline 的差距。

    这里的最佳 strong baseline 只按 target_accuracy 选，目的是定位
    D-CGFS 在弱势类修复能力上最需要追赶的 pair。
    """
    by_pair: dict[int, list[dict[str, object]]] = {}
    for row in pair_rows:
        by_pair.setdefault(int(row["pair_id"]), []).append(row)

    gap_rows: list[dict[str, object]] = []
    for pair_id, rows in sorted(by_pair.items()):
        dcgfs = next(row for row in rows if row["method"] == dcgfs_name)
        baseline = next(row for row in rows if row["method"] == "SSCBM baseline")
        strong = [row for row in rows if row["method"] not in {"SSCBM baseline", dcgfs_name}]
        best = max(strong, key=lambda row: float(row["target_accuracy"]))
        gap_rows.append(
            {
                "pair_id": pair_id,
                "target_class_id": dcgfs["target_class_id"],
                "target_class_name": dcgfs["target_class_name"],
                "base_class_id": dcgfs["base_class_id"],
                "base_class_name": dcgfs["base_class_name"],
                "baseline_target_acc": baseline["target_accuracy"],
                "dcgfs_target_acc": dcgfs["target_accuracy"],
                "best_strong_method": best["method"],
                "best_strong_target_acc": best["target_accuracy"],
                "dcgfs_gap_to_best": float(dcgfs["target_accuracy"]) - float(best["target_accuracy"]),
                "baseline_target_to_base": baseline["target_to_base_rate"],
                "dcgfs_target_to_base": dcgfs["target_to_base_rate"],
                "best_strong_target_to_base": best["target_to_base_rate"],
            }
        )
    return gap_rows


def merge_synthesis_diagnostics(
    gap_rows: list[dict[str, object]], pair_summary_path: Path
) -> list[dict[str, object]]:
    """把合成阶段的 pair 质量统计并入失败诊断。

    如果某个 pair 在合成阶段 target_prob 很低、concept_pass_rate 很低，
    而最终 D-CGFS 又明显落后 strong baseline，通常说明需要优先改合成质量。
    """
    summary_rows = {int(row["pair_id"]): row for row in read_csv(pair_summary_path)}
    merged: list[dict[str, object]] = []
    for row in gap_rows:
        pair_id = int(row["pair_id"])
        syn = summary_rows[pair_id]
        merged.append(
            {
                **row,
                "candidate_count": int(syn["candidate_count"]),
                "concept_pass_rate": float(syn["concept_pass_rate"]),
                "target_prob_mean": float(syn["target_prob_mean"]),
                "target_prob_max": float(syn["target_prob_max"]),
                "concept_delta_mean": float(syn["concept_delta_mean"]),
                "concept_delta_max": float(syn["concept_delta_max"]),
                "pair_quality_score_max": float(syn["pair_quality_score_max"]),
            }
        )
    return merged


def write_markdown_report(
    path: Path,
    overall_rows: list[dict[str, object]],
    merged_gap_rows: list[dict[str, object]],
    dcgfs_name: str,
) -> None:
    """写出中文诊断报告，方便每次实验后直接阅读。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(overall_rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    worst_gaps = sorted(merged_gap_rows, key=lambda row: float(row["dcgfs_gap_to_best"]))

    lines: list[str] = []
    lines.append("# Problem6 Strong Baseline Pair Diagnosis\n")
    lines.append("## 总体指标排行\n")
    lines.append(
        "| Method | Overall Acc | Macro-F1 | Balanced Acc | Target Acc | Base Acc | Target-to-Base | Concept Acc |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in ranked:
        lines.append(
            "| {method} | {overall} | {macro} | {balanced} | {target} | {base} | {t2b} | {concept} |".format(
                method=row["method"],
                overall=pct(float(row["overall_a_acc"])),
                macro=pct(float(row["macro_f1"])),
                balanced=pct(float(row["balanced_accuracy"])),
                target=pct(float(row["selected_target_acc"])),
                base=pct(float(row["selected_base_acc"])),
                t2b=pct(float(row["target_to_base_rate"])),
                concept=pct(float(row["overall_c_acc"])),
            )
        )

    lines.append("\n## D-CGFS 相对最佳强基线的 pair 级差距\n")
    lines.append(
        "| Pair | Target | Base | Baseline Acc | D-CGFS Acc | Best Strong | Best Strong Acc | Gap | D-CGFS T2B | Best T2B |"
    )
    lines.append("| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for row in worst_gaps:
        lines.append(
            "| {pair_id} | {target} | {base} | {base_acc} | {dcgfs_acc} | {best_method} | {best_acc} | {gap} | {dcgfs_t2b} | {best_t2b} |".format(
                pair_id=row["pair_id"],
                target=f"{row['target_class_id']} {row['target_class_name']}",
                base=f"{row['base_class_id']} {row['base_class_name']}",
                base_acc=pct(float(row["baseline_target_acc"])),
                dcgfs_acc=pct(float(row["dcgfs_target_acc"])),
                best_method=row["best_strong_method"],
                best_acc=pct(float(row["best_strong_target_acc"])),
                gap=pct(float(row["dcgfs_gap_to_best"])),
                dcgfs_t2b=pct(float(row["dcgfs_target_to_base"])),
                best_t2b=pct(float(row["best_strong_target_to_base"])),
            )
        )

    lines.append("\n## 合成质量关联诊断\n")
    lines.append(
        "| Pair | Target | D-CGFS Gap | Concept Pass | Target Prob Max | Concept Delta Mean | Concept Delta Max | 诊断 |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in worst_gaps:
        diagnosis = "需要优先检查"
        if float(row["target_prob_max"]) < 1e-4:
            diagnosis = "target_prob 极低，合成样本可能没有把分类边界推向目标类"
        elif float(row["concept_pass_rate"]) < 0.5:
            diagnosis = "概念过滤通过率偏低，判别概念定位或融合可能不稳定"
        elif float(row["dcgfs_gap_to_best"]) < -0.2:
            diagnosis = "strong baseline 明显更强，需要检查训练目标或保护权重"
        lines.append(
            "| {pair_id} | {target} | {gap} | {concept_pass} | {target_prob_max:.6f} | {delta_mean:.4f} | {delta_max:.4f} | {diagnosis} |".format(
                pair_id=row["pair_id"],
                target=f"{row['target_class_id']} {row['target_class_name']}",
                gap=pct(float(row["dcgfs_gap_to_best"])),
                concept_pass=pct(float(row["concept_pass_rate"])),
                target_prob_max=float(row["target_prob_max"]),
                delta_mean=float(row["concept_delta_mean"]),
                delta_max=float(row["concept_delta_max"]),
                diagnosis=diagnosis,
            )
        )

    lines.append("\n## 结论\n")
    lines.append(
        f"`{dcgfs_name}` 的主要问题不是没有修复弱势类，而是 target accuracy 的提升幅度弱于普通强基线。"
    )
    lines.append(
        "当前最值得优先处理的是合成样本分类置信度过低，以及部分 pair 的 target 修复不足。"
    )
    lines.append(
        "下一步建议在不牺牲 base preservation 的前提下，尝试增强合成样本对目标类分类边界的推动，例如调高 pair-topk 数量、加入 target-confidence rerank，或对 D-CGFS 的合成样本损失增加轻量权重。"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 problem6 中 D-CGFS 与 strong baseline 的 pair 级差距。")
    parser.add_argument("--output-dir", default="diagnostic_results/problem6_strong_baseline")
    parser.add_argument("--dcgfs-name", default="D-CGFS pair_topk_base_w015")
    parser.add_argument("--pair-summary", default="generated_data/dcgfs_pair_topk/pair_filter_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    eval_dirs = {
        "SSCBM finetune": Path("final_evaluation_results/problem6_sscbm_finetune"),
        "Oversampling": Path("final_evaluation_results/problem6_oversampling"),
        "Reweighting": Path("final_evaluation_results/problem6_reweighting"),
        "Class-balanced loss": Path("final_evaluation_results/problem6_class_balanced_loss"),
        "Feature mixup": Path("final_evaluation_results/problem6_feature_mixup"),
        args.dcgfs_name: Path("final_evaluation_results/problem6_dcgfs_pair_topk_base_w015"),
    }

    overall_rows = load_candidate_rows(eval_dirs)
    pair_rows = load_pair_rows(eval_dirs)
    gap_rows = summarize_gaps(pair_rows, args.dcgfs_name)
    merged_gap_rows = merge_synthesis_diagnostics(gap_rows, Path(args.pair_summary))

    write_csv(
        output_dir / "overall_comparison.csv",
        overall_rows,
        [
            "method",
            "overall_a_acc",
            "macro_f1",
            "balanced_accuracy",
            "worst_class_acc",
            "selected_target_acc",
            "selected_base_acc",
            "target_to_base_rate",
            "overall_c_acc",
        ],
    )
    write_csv(
        output_dir / "pair_method_comparison.csv",
        pair_rows,
        [
            "method",
            "pair_id",
            "target_class_id",
            "target_class_name",
            "base_class_id",
            "base_class_name",
            "target_samples",
            "target_correct",
            "target_accuracy",
            "target_pred_as_base",
            "target_to_base_rate",
        ],
    )
    write_csv(
        output_dir / "dcgfs_gap_to_best_strong_baseline.csv",
        merged_gap_rows,
        [
            "pair_id",
            "target_class_id",
            "target_class_name",
            "base_class_id",
            "base_class_name",
            "baseline_target_acc",
            "dcgfs_target_acc",
            "best_strong_method",
            "best_strong_target_acc",
            "dcgfs_gap_to_best",
            "baseline_target_to_base",
            "dcgfs_target_to_base",
            "best_strong_target_to_base",
            "candidate_count",
            "concept_pass_rate",
            "target_prob_mean",
            "target_prob_max",
            "concept_delta_mean",
            "concept_delta_max",
            "pair_quality_score_max",
        ],
    )
    write_markdown_report(output_dir / "diagnosis_report.md", overall_rows, merged_gap_rows, args.dcgfs_name)

    print(f"诊断结果已保存到: {output_dir}")
    print(f"- {output_dir / 'overall_comparison.csv'}")
    print(f"- {output_dir / 'pair_method_comparison.csv'}")
    print(f"- {output_dir / 'dcgfs_gap_to_best_strong_baseline.csv'}")
    print(f"- {output_dir / 'diagnosis_report.md'}")


if __name__ == "__main__":
    main()
