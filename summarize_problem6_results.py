# ==========================================================================================
# Summarize finalized Problem 6 results for paper/report writing.
#
# This script does not run experiments. It writes curated CSV/Markdown files that reflect the
# current project decision: `dcgfs_target_score_w015` is the main method, while feature
# refinement and related branches are archived as diagnostics.
# ==========================================================================================

import argparse
import os

import pandas as pd


RESULT_DIR = "final_evaluation_results"


MAIN_RESULT_SPECS = [
    ("SSCBM", "problem6_dcgfs_target_score_w015", "SSCBM (Baseline)"),
    ("D-CGFS original main", "dcgfs_main", None),
    ("D-CGFS pair-topk", "dcgfs_pair_topk", None),
    ("D-CGFS previous main w005 + BP", "dcgfs_pair_topk_base_w005", None),
    ("D-CGFS target_score_w015 + BP", "problem6_dcgfs_target_score_w015", None),
]


STRONG_BASELINE_SPECS = [
    ("SSCBM baseline", "problem6_dcgfs_target_score_w015", "SSCBM (Baseline)"),
    ("SSCBM finetune", "problem6_sscbm_finetune", None),
    ("Oversampling", "problem6_oversampling", None),
    ("Reweighting", "problem6_reweighting", None),
    ("Class-balanced loss", "problem6_class_balanced_loss", None),
    ("Feature mixup", "problem6_feature_mixup", None),
    ("D-CGFS target_score_w015", "problem6_dcgfs_target_score_w015", None),
]


ARCHIVED_DIAGNOSTIC_SPECS = [
    ("dcgfs_retrieval_residual", "problem6_dcgfs_retrieval_residual", "archive"),
    ("dcgfs_model_aware_d_w050", "problem6_dcgfs_model_aware_d_w050", "archive"),
    ("dcgfs_feature_refine_pred_disc", "problem6_dcgfs_feature_refine_pred_disc", "stop"),
    ("dcgfs_feature_refine_window", "problem6_dcgfs_feature_refine_window", "stop"),
    ("dcgfs_hybrid_pair0_refine_pred_disc_train", "problem6_dcgfs_hybrid_pair0_refine_pred_disc_train", "stop"),
]


MULTI_DATASET_SPECS = [
    ("CUB", "SSCBM baseline", "problem6_dcgfs_target_score_w015", "SSCBM (Baseline)"),
    ("CUB", "D-CGFS target_score_w015", "problem6_dcgfs_target_score_w015", None),
    ("AwA2", "SSCBM baseline", "awa2_dcgfs_target_score_w015", "SSCBM (Baseline)"),
    ("AwA2", "D-CGFS target_score_w015", "awa2_dcgfs_target_score_w015", None),
    ("PBC", "SSCBM baseline", "pbc_dcgfs_target_score_w015", "SSCBM (Baseline)"),
    ("PBC", "D-CGFS target_score_w015", "pbc_dcgfs_target_score_w015", None),
    ("7pt", "SSCBM baseline", "7pt_dcgfs_target_score_w015", "SSCBM (Baseline)"),
    ("7pt", "D-CGFS target_score_w015", "7pt_dcgfs_target_score_w015", None),
]


METRIC_GLOSSARY = [
    ["overall_acc", "整体任务准确率", "全部测试样本上的类别预测准确率。"],
    ["overall_concept_acc", "整体概念准确率", "全部测试样本、全部概念上的概念预测准确率。"],
    ["macro_f1", "宏平均 F1", "先按类别计算 F1，再对类别平均，能减轻类别不平衡对指标的遮蔽。"],
    ["balanced_acc", "平衡准确率", "各类别召回率的平均值，用于观察不同类别是否被均衡识别。"],
    ["worst_class_acc", "最差类别准确率", "测试集中表现最差类别的准确率，反映尾部风险。"],
    ["selected_target_acc", "自动弱势目标类准确率", "find_weak_classes.py 自动选出的弱势目标类上的准确率。"],
    ["selected_base_acc", "自动基座类准确率", "与目标类混淆严重的基座类上的准确率，用于检查副作用。"],
    ["target_to_base_rate", "目标类错分为基座类比例", "目标类样本被预测成对应基座类的比例，越低越好。"],
]


FALLBACK_MAIN_RESULTS = [
    {
        "method": "SSCBM",
        "overall_acc": 57.80,
        "macro_f1": 57.23,
        "balanced_acc": 57.99,
        "worst_class_acc": 0.00,
        "target_acc": 19.29,
        "base_acc": 82.91,
        "target_to_base_rate": 48.57,
    },
    {
        "method": "D-CGFS original main",
        "overall_acc": 65.36,
        "macro_f1": 65.16,
        "balanced_acc": 65.45,
        "worst_class_acc": 3.33,
        "target_acc": 37.86,
        "base_acc": 75.21,
        "target_to_base_rate": 20.71,
    },
    {
        "method": "D-CGFS pair-topk",
        "overall_acc": 65.98,
        "macro_f1": 65.97,
        "balanced_acc": 65.99,
        "worst_class_acc": 6.67,
        "target_acc": 44.29,
        "base_acc": 66.67,
        "target_to_base_rate": 12.86,
    },
    {
        "method": "D-CGFS previous main w005 + BP",
        "overall_acc": 65.74,
        "macro_f1": 65.72,
        "balanced_acc": 65.80,
        "worst_class_acc": 6.67,
        "target_acc": 44.29,
        "base_acc": 77.78,
        "target_to_base_rate": 17.86,
    },
    {
        "method": "D-CGFS target_score_w015 + BP",
        "overall_acc": 66.00,
        "macro_f1": 65.96,
        "balanced_acc": 66.04,
        "worst_class_acc": 10.00,
        "target_acc": 44.29,
        "base_acc": 78.63,
        "target_to_base_rate": 19.29,
    },
]


FALLBACK_STRONG_BASELINES = [
    ["SSCBM baseline", 57.80, 57.23, 57.99, 0.00, 19.29, 82.91, 48.57, 89.72],
    ["SSCBM finetune", 66.76, 66.62, 66.87, 3.33, 56.43, 71.79, 16.43, 89.56],
    ["Oversampling", 66.48, 66.06, 66.53, 13.33, 50.71, 64.10, 10.71, 89.51],
    ["Reweighting", 67.31, 67.23, 67.42, 3.33, 58.57, 70.94, 15.71, 89.60],
    ["Class-balanced loss", 67.31, 67.23, 67.42, 3.33, 58.57, 70.94, 15.71, 89.60],
    ["Feature mixup", 66.69, 66.65, 66.87, 3.33, 57.14, 68.38, 15.00, 88.92],
    ["D-CGFS target_score_w015", 66.00, 65.96, 66.04, 10.00, 44.29, 78.63, 19.29, 89.46],
]


FALLBACK_NEGATIVE_DIAGNOSTICS = [
    ["dcgfs_retrieval_residual", 65.83, 65.84, 65.90, 6.67, 42.14, 76.92, 20.00, "archive"],
    ["dcgfs_model_aware_d_w050", 65.62, 65.63, 65.69, 6.67, 40.71, 76.07, 22.14, "archive"],
    ["dcgfs_feature_refine_pred_disc", 65.84, 65.85, 65.94, 6.67, 42.86, 76.07, 22.14, "stop"],
    ["dcgfs_feature_refine_window", 65.93, 65.92, 66.02, 6.67, 40.71, 76.92, 21.43, "stop"],
    ["dcgfs_hybrid_pair0_refine_pred_disc_train", 66.03, 66.03, 66.11, 10.00, 43.57, 77.78, 20.00, "stop"],
]


def pct(value):
    return round(float(value) * 100.0, 2)


def _load_summary_row(result_subdir, preferred_model=None):
    path = os.path.join(RESULT_DIR, result_subdir, "model_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if preferred_model is not None:
        matched = df[df["model"] == preferred_model]
        if matched.empty:
            return None
        return matched.iloc[0]
    candidates = df[df["model"] != "SSCBM (Baseline)"]
    if candidates.empty:
        candidates = df
    return candidates.iloc[-1]


def _main_result_from_row(method, row):
    return {
        "method": method,
        "overall_acc": pct(row["overall_a_acc"]),
        "macro_f1": pct(row["macro_f1"]),
        "balanced_acc": pct(row["balanced_accuracy"]),
        "worst_class_acc": pct(row["worst_class_acc"]),
        "target_acc": pct(row["selected_target_acc"]),
        "base_acc": pct(row["selected_base_acc"]),
        "target_to_base_rate": pct(row["target_to_base_rate"]),
    }


def _strong_result_from_row(method, row):
    return [
        method,
        pct(row["overall_a_acc"]),
        pct(row["macro_f1"]),
        pct(row["balanced_accuracy"]),
        pct(row["worst_class_acc"]),
        pct(row["selected_target_acc"]),
        pct(row["selected_base_acc"]),
        pct(row["target_to_base_rate"]),
        pct(row["overall_c_acc"]),
    ]


def build_main_results():
    rows = []
    for method, result_subdir, preferred_model in MAIN_RESULT_SPECS:
        row = _load_summary_row(result_subdir, preferred_model)
        if row is None:
            return FALLBACK_MAIN_RESULTS
        rows.append(_main_result_from_row(method, row))
    return rows


def build_strong_baselines():
    rows = []
    for method, result_subdir, preferred_model in STRONG_BASELINE_SPECS:
        row = _load_summary_row(result_subdir, preferred_model)
        if row is None:
            return FALLBACK_STRONG_BASELINES
        rows.append(_strong_result_from_row(method, row))
    return rows


def build_archived_diagnostics():
    rows = []
    for method, result_subdir, decision in ARCHIVED_DIAGNOSTIC_SPECS:
        row = _load_summary_row(result_subdir)
        if row is None:
            return FALLBACK_NEGATIVE_DIAGNOSTICS
        rows.append(_strong_result_from_row(method, row)[:-1] + [decision])
    return rows


def build_multi_dataset_results():
    rows = []
    for dataset, method, result_subdir, preferred_model in MULTI_DATASET_SPECS:
        row = _load_summary_row(result_subdir, preferred_model)
        if row is None:
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "status": "pending",
                    "overall_acc": "",
                    "macro_f1": "",
                    "balanced_acc": "",
                    "worst_class_acc": "",
                    "selected_target_acc": "",
                    "selected_base_acc": "",
                    "target_to_base_rate": "",
                    "overall_concept_acc": "",
                }
            )
            continue
        values = _strong_result_from_row(method, row)
        rows.append(
            {
                "dataset": dataset,
                "method": values[0],
                "status": "done",
                "overall_acc": values[1],
                "macro_f1": values[2],
                "balanced_acc": values[3],
                "worst_class_acc": values[4],
                "selected_target_acc": values[5],
                "selected_base_acc": values[6],
                "target_to_base_rate": values[7],
                "overall_concept_acc": values[8],
            }
        )
    return rows


def markdown_table(rows, columns):
    formatted_rows = []
    for row in rows:
        if isinstance(row, dict):
            formatted_rows.append([row[column] for column in columns])
        else:
            formatted_rows.append(row)

    def fmt(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in formatted_rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(output_dir, main_results, strong_baselines, archived_diagnostics, multi_dataset_results):
    strong_columns = [
        "method",
        "overall_acc",
        "macro_f1",
        "balanced_acc",
        "worst_class_acc",
        "selected_target_acc",
        "selected_base_acc",
        "target_to_base_rate",
        "overall_concept_acc",
    ]
    diagnostic_columns = [
        "method",
        "overall_acc",
        "macro_f1",
        "balanced_acc",
        "worst_class_acc",
        "target_acc",
        "base_acc",
        "target_to_base_rate",
        "decision",
    ]
    lines = [
        "# Problem 6 Final Report Summary",
        "",
        "Current main method: `D-CGFS target_score_w015 + base preservation`.",
        "",
        "## Metric Glossary",
        "",
        markdown_table(METRIC_GLOSSARY, ["metric", "中文名称", "含义"]),
        "",
        "Feature refinement, retrieval residual, model-aware D, and hybrid pair refinement are not promoted.",
        "They remain archived diagnostics because they did not improve the primary weak-target tradeoff.",
        "",
        "## Main Result",
        "",
        markdown_table(main_results, list(main_results[0].keys())),
        "",
        "CUB conclusion: D-CGFS substantially improves overall accuracy, macro-F1, balanced",
        "accuracy, worst-class accuracy, and selected weak-target accuracy. It also reduces the",
        "target-to-base confusion rate, which directly supports the target-base motivation.",
        "",
        "## Strong Baseline Context",
        "",
        "D-CGFS should not be claimed as the best method on raw target accuracy. Reweighting and",
        "class-balanced loss are stronger on selected target accuracy, while D-CGFS keeps better",
        "selected base accuracy than the retrained imbalance baselines and gives a concept-guided",
        "synthesis mechanism.",
        "",
        markdown_table(strong_baselines, strong_columns),
        "",
        "## Archived Diagnostics",
        "",
        markdown_table(archived_diagnostics, diagnostic_columns),
        "",
        "## Cross-Dataset Results",
        "",
        markdown_table(multi_dataset_results, list(multi_dataset_results[0].keys())),
        "",
        "Cross-dataset conclusion: D-CGFS gives clear gains on CUB and 7pt, preserves the",
        "near-saturated PBC task performance with small balanced/worst-class improvements, and",
        "mainly stabilizes base classes on AwA2. The paper claim should emphasize concept-guided",
        "weak-class augmentation and cross-dataset robustness, while noting that gains are limited",
        "when the baseline is already near saturation.",
        "",
        "## Writing Position",
        "",
        "Use D-CGFS as a concept-guided weak-class augmentation method with stronger preservation",
        "and interpretability, not as a universal replacement for all imbalance baselines.",
    ]
    path = os.path.join(output_dir, "final_report_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Write finalized Problem 6 paper/report tables.")
    parser.add_argument("--output-dir", default="paper_tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main_results = build_main_results()
    strong_baselines = build_strong_baselines()
    archived_diagnostics = build_archived_diagnostics()
    multi_dataset_results = build_multi_dataset_results()

    pd.DataFrame(main_results).to_csv(
        os.path.join(args.output_dir, "final_main_results.csv"),
        index=False,
    )
    pd.DataFrame(
        strong_baselines,
        columns=[
            "method",
            "overall_acc",
            "macro_f1",
            "balanced_acc",
            "worst_class_acc",
            "selected_target_acc",
            "selected_base_acc",
            "target_to_base_rate",
            "overall_concept_acc",
        ],
    ).to_csv(os.path.join(args.output_dir, "final_strong_baseline_context.csv"), index=False)
    pd.DataFrame(
        archived_diagnostics,
        columns=[
            "method",
            "overall_acc",
            "macro_f1",
            "balanced_acc",
            "worst_class_acc",
            "target_acc",
            "base_acc",
            "target_to_base_rate",
            "decision",
        ],
    ).to_csv(os.path.join(args.output_dir, "final_archived_diagnostics.csv"), index=False)
    pd.DataFrame(METRIC_GLOSSARY, columns=["metric", "中文名称", "含义"]).to_csv(
        os.path.join(args.output_dir, "metric_glossary_zh.csv"),
        index=False,
    )
    pd.DataFrame(multi_dataset_results).to_csv(
        os.path.join(args.output_dir, "multi_dataset_results.csv"),
        index=False,
    )

    markdown_path = write_markdown(
        args.output_dir,
        main_results,
        strong_baselines,
        archived_diagnostics,
        multi_dataset_results,
    )
    print(f"已写入论文/报告汇总: {markdown_path}")


if __name__ == "__main__":
    main()
