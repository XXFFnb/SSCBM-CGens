# ==========================================================================================
# CUB global analysis for D-CGFS.
#
# This script consolidates existing `step4_final_evaluation.py` outputs into Chinese-readable
# paper/report tables. It does not train or evaluate models.
# ==========================================================================================

import argparse
import os

import pandas as pd


RESULT_DIR = "final_evaluation_results"
PAPER_DIR = "paper_tables"

MAIN_DIR = os.path.join(RESULT_DIR, "problem6_dcgfs_target_score_w015")
COMPARISON_DIRS = {
    "SSCBM 微调": os.path.join(RESULT_DIR, "problem6_sscbm_finetune"),
    "过采样 Oversampling": os.path.join(RESULT_DIR, "problem6_oversampling"),
    "重加权 Reweighting": os.path.join(RESULT_DIR, "problem6_reweighting"),
    "Class-balanced loss": os.path.join(RESULT_DIR, "problem6_class_balanced_loss"),
    "Feature Mixup": os.path.join(RESULT_DIR, "problem6_feature_mixup"),
    "D-CGFS 主方法": MAIN_DIR,
}

METRIC_INFO = [
    ("overall_a_acc", "整体任务准确率", "所有测试样本上的类别预测准确率。"),
    ("overall_c_acc", "整体概念准确率", "所有概念预测的平均准确率。"),
    ("macro_f1", "宏平均 F1", "先计算每个类别 F1，再对类别取平均；更关注类别均衡性。"),
    ("balanced_accuracy", "平衡准确率", "每个类别召回率的平均值，能减少类别样本数差异的影响。"),
    ("worst_class_acc", "最差类别准确率", "所有类别中准确率最低的类别表现。"),
    ("selected_target_acc", "自动弱势目标类准确率", "自动选出的弱势目标类集合上的准确率。"),
    ("selected_base_acc", "自动基座类准确率", "与弱势目标类配对的基座类集合上的准确率。"),
    ("target_to_base_rate", "目标类错分为基座类比例", "弱势目标类样本被预测成对应基座类的比例，越低越好。"),
]


def pct(value):
    return round(float(value) * 100.0, 2)


def load_class_names():
    path = "data/CUB_200_2011/classes.txt"
    class_names = {}
    if not os.path.exists(path):
        return class_names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id, raw_name = line.split(maxsplit=1)
            class_names[int(class_id)] = raw_name.split(".", maxsplit=1)[-1].replace("_", " ")
    return class_names


def load_model_summary(result_dir):
    path = os.path.join(result_dir, "model_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_classwise(result_dir):
    path = os.path.join(result_dir, "classwise_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_target_base_pairs():
    path = os.path.join(MAIN_DIR, "used_target_base_pairs.csv")
    if not os.path.exists(path):
        path = os.path.join(MAIN_DIR, "target_base_confusion.csv")
    pairs = pd.read_csv(path)
    return sorted(pairs["target_class_id"].astype(int).unique()), sorted(pairs["base_class_id"].astype(int).unique())


def build_metric_glossary():
    return pd.DataFrame(
        [
            {"metric": metric, "中文名称": zh_name, "解释": explanation}
            for metric, zh_name, explanation in METRIC_INFO
        ]
    )


def build_global_summary():
    rows = []
    baseline_row = load_model_summary(MAIN_DIR).iloc[0].to_dict()
    rows.append({"method": "SSCBM 原始基线", **baseline_row})

    for label, result_dir in COMPARISON_DIRS.items():
        summary = load_model_summary(result_dir)
        candidate = summary[summary["model"] != "SSCBM (Baseline)"].iloc[-1].to_dict()
        rows.append({"method": label, **candidate})

    df = pd.DataFrame(rows)
    keep_columns = ["method"] + [metric for metric, _, _ in METRIC_INFO]
    df = df[keep_columns]
    for metric, _, _ in METRIC_INFO:
        df[metric] = df[metric].map(pct)
    return df


def build_delta_summary(global_summary):
    baseline = global_summary[global_summary["method"] == "SSCBM 原始基线"].iloc[0]
    rows = []
    for _, row in global_summary.iterrows():
        if row["method"] == "SSCBM 原始基线":
            continue
        delta = {"method": row["method"]}
        for metric, _, _ in METRIC_INFO:
            delta[f"{metric}_变化"] = round(float(row[metric]) - float(baseline[metric]), 2)
        rows.append(delta)
    return pd.DataFrame(rows)


def build_classwise_global_analysis(class_names):
    target_ids, base_ids = load_target_base_pairs()
    special_ids = set(target_ids) | set(base_ids)

    classwise = load_classwise(MAIN_DIR)
    baseline = classwise[classwise["model"] == "SSCBM (Baseline)"].copy()
    candidate = classwise[classwise["model"] != "SSCBM (Baseline)"].copy()
    merged = baseline.merge(
        candidate,
        on=["class_id", "samples"],
        suffixes=("_baseline", "_dcgfs"),
    )
    merged["class_name"] = merged["class_id"].map(class_names).fillna("")
    merged["accuracy_delta"] = merged["accuracy_dcgfs"] - merged["accuracy_baseline"]
    merged["f1_delta"] = merged["f1_dcgfs"] - merged["f1_baseline"]
    merged["group"] = "普通非配对类"
    merged.loc[merged["class_id"].isin(base_ids), "group"] = "自动基座类"
    merged.loc[merged["class_id"].isin(target_ids), "group"] = "自动弱势目标类"

    def summarize(group_name, part):
        return {
            "类别组": group_name,
            "类别数": int(part["class_id"].nunique()),
            "样本数": int(part["samples"].sum()),
            "Baseline 平均类准确率": pct(part["accuracy_baseline"].mean()),
            "D-CGFS 平均类准确率": pct(part["accuracy_dcgfs"].mean()),
            "平均变化": round(pct(part["accuracy_delta"].mean()), 2),
            "中位数变化": round(pct(part["accuracy_delta"].median()), 2),
            "提升类别数": int((part["accuracy_delta"] > 1e-12).sum()),
            "下降类别数": int((part["accuracy_delta"] < -1e-12).sum()),
            "不变类别数": int((part["accuracy_delta"].abs() <= 1e-12).sum()),
        }

    summary_rows = [
        summarize("全部类别", merged),
        summarize("自动弱势目标类", merged[merged["class_id"].isin(target_ids)]),
        summarize("自动基座类", merged[merged["class_id"].isin(base_ids)]),
        summarize("非目标非基座类", merged[~merged["class_id"].isin(special_ids)]),
    ]
    group_summary = pd.DataFrame(summary_rows)

    class_detail = merged[
        [
            "class_id",
            "class_name",
            "group",
            "samples",
            "accuracy_baseline",
            "accuracy_dcgfs",
            "accuracy_delta",
            "f1_baseline",
            "f1_dcgfs",
            "f1_delta",
        ]
    ].copy()
    for column in [
        "accuracy_baseline",
        "accuracy_dcgfs",
        "accuracy_delta",
        "f1_baseline",
        "f1_dcgfs",
        "f1_delta",
    ]:
        class_detail[column] = class_detail[column].map(pct)

    top_improved = class_detail.sort_values("accuracy_delta", ascending=False).head(15)
    top_dropped = class_detail.sort_values("accuracy_delta", ascending=True).head(15)

    return group_summary, class_detail, top_improved, top_dropped


def build_target_base_detail(class_names):
    path = os.path.join(MAIN_DIR, "target_base_confusion.csv")
    df = pd.read_csv(path)
    df["target_class_name"] = df["target_class_id"].map(class_names).fillna(df["target_class_name"])
    df["base_class_name"] = df["base_class_id"].map(class_names).fillna(df["base_class_name"])
    for col in ["target_accuracy", "target_to_base_rate"]:
        df[col] = df[col].map(pct)
    return df[
        [
            "model",
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
        ]
    ]


def markdown_table(df):
    columns = list(df.columns)

    def fmt(value):
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_markdown(output_dir, metric_glossary, global_summary, delta_summary, group_summary, top_improved, top_dropped, target_base_detail):
    lines = [
        "# CUB 全局分析：D-CGFS 主方法",
        "",
        "本文档整理 CUB-200-2011 上已有评估结果，重点补充整体指标和全类别分布分析。",
        "",
        "## 指标解释",
        "",
        markdown_table(metric_glossary),
        "",
        "## 整体指标对比",
        "",
        "表中数值均为百分比。`目标类错分为基座类比例` 越低越好，其余指标越高越好。",
        "",
        markdown_table(global_summary),
        "",
        "## 相对 SSCBM 原始基线的变化",
        "",
        "表中数值为百分点变化。正数表示提升，负数表示下降。",
        "",
        markdown_table(delta_summary),
        "",
        "## 全类别分组分析",
        "",
        "这里把 200 个类别分成自动弱势目标类、自动基座类、非目标非基座类，并统计逐类准确率变化。",
        "",
        markdown_table(group_summary),
        "",
        "## 自动 target-base pair 细节",
        "",
        markdown_table(target_base_detail),
        "",
        "## 准确率提升最多的类别",
        "",
        markdown_table(top_improved),
        "",
        "## 准确率下降最多的类别",
        "",
        markdown_table(top_dropped),
        "",
        "## 结论摘要",
        "",
        "1. D-CGFS 主方法不仅提升自动弱势目标类，也提升 overall、macro-F1、balanced accuracy 和 worst-class accuracy。",
        "2. 自动基座类准确率相对 SSCBM 原始基线下降，这是弱势目标类增强带来的主要 trade-off；base preservation 的作用是减轻这种下降。",
        "3. 非目标非基座类的平均变化用于判断方法是否只服务少数 target pair，还是对整体类别分布也保持稳定。",
        "4. 后续论文表述应同时报告整体指标、目标/基座指标和全类别分布，而不是只报告 selected target accuracy。",
    ]
    path = os.path.join(output_dir, "cub_global_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Analyze CUB global metrics for D-CGFS.")
    parser.add_argument("--output-dir", default=PAPER_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = load_class_names()

    metric_glossary = build_metric_glossary()
    global_summary = build_global_summary()
    delta_summary = build_delta_summary(global_summary)
    group_summary, class_detail, top_improved, top_dropped = build_classwise_global_analysis(class_names)
    target_base_detail = build_target_base_detail(class_names)

    metric_glossary.to_csv(os.path.join(args.output_dir, "cub_metric_glossary.csv"), index=False)
    global_summary.to_csv(os.path.join(args.output_dir, "cub_global_metrics.csv"), index=False)
    delta_summary.to_csv(os.path.join(args.output_dir, "cub_global_metric_deltas.csv"), index=False)
    group_summary.to_csv(os.path.join(args.output_dir, "cub_class_group_summary.csv"), index=False)
    class_detail.to_csv(os.path.join(args.output_dir, "cub_classwise_global_detail.csv"), index=False)
    top_improved.to_csv(os.path.join(args.output_dir, "cub_top_improved_classes.csv"), index=False)
    top_dropped.to_csv(os.path.join(args.output_dir, "cub_top_dropped_classes.csv"), index=False)
    target_base_detail.to_csv(os.path.join(args.output_dir, "cub_target_base_detail.csv"), index=False)

    markdown_path = write_markdown(
        args.output_dir,
        metric_glossary,
        global_summary,
        delta_summary,
        group_summary,
        top_improved,
        top_dropped,
        target_base_detail,
    )
    print(f"已生成 CUB 全局分析: {markdown_path}")


if __name__ == "__main__":
    main()
