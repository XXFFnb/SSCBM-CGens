# ==========================================================================================
# D-CGFS Problem 6 experiment scheduler
#
# Current project policy:
# - The paper main method is `dcgfs_target_score_w015`.
# - Feature refinement, retrieval residual, model-aware D, and hybrid pair refinement are
#   archived diagnostic branches. They are intentionally not exposed as runnable groups here.
# - This script keeps only the reproducible main line, strong baselines, paper ablations,
#   reporting, and second-dataset readiness checks.
# ==========================================================================================

import argparse
import os
import subprocess

import pandas as pd

from dcgfs_config import (
    BASE_PRESERVATION_WEIGHT,
    METHOD_ACRONYM,
    PAIR_SCORE_BASE_WEIGHT,
    PAIR_SCORE_TARGET_WEIGHT,
)


PYTHON = ".venv/bin/python"
SAVE_DIR = "problem6_experiment_protocol"

MAIN_NAME = "dcgfs_target_score_w015"
MAIN_GEN_DIR = "generated_data/problem6_target_score_w015"
MAIN_CHECKPOINT = "checkpoints/problem6_dcgfs_target_score_w015.pt"
MAIN_FINAL_DIR = "final_evaluation_results/problem6_dcgfs_target_score_w015"
MAIN_EXPLAIN_DIR = "explainability_results/problem6_dcgfs_target_score_w015"
PAIR_TOPK = "500"

AWA2_AUX_DIR = "data/D-CGFS_Auxiliary_AwA2"
AWA2_NAME = "awa2_dcgfs_target_score_w015"
AWA2_GEN_DIR = "generated_data/awa2_target_score_w015"
AWA2_CHECKPOINT = "checkpoints/awa2_dcgfs_target_score_w015.pt"
AWA2_FINAL_DIR = "final_evaluation_results/awa2_dcgfs_target_score_w015"

PBC_AUX_DIR = "data/D-CGFS_Auxiliary_PBC"
PBC_NAME = "pbc_dcgfs_target_score_w015"
PBC_GEN_DIR = "generated_data/pbc_target_score_w015"
PBC_CHECKPOINT = "checkpoints/pbc_dcgfs_target_score_w015.pt"
PBC_FINAL_DIR = "final_evaluation_results/pbc_dcgfs_target_score_w015"

SEVEN_PT_AUX_DIR = "data/D-CGFS_Auxiliary_7pt"
SEVEN_PT_NAME = "7pt_dcgfs_target_score_w015"
SEVEN_PT_GEN_DIR = "generated_data/7pt_target_score_w015"
SEVEN_PT_CHECKPOINT = "checkpoints/7pt_dcgfs_target_score_w015.pt"
SEVEN_PT_FINAL_DIR = "final_evaluation_results/7pt_dcgfs_target_score_w015"


def build_pair_topk_synthesis_command(
    output_dir=MAIN_GEN_DIR,
    pair_topk=PAIR_TOPK,
    target_weight=PAIR_SCORE_TARGET_WEIGHT,
    base_weight=PAIR_SCORE_BASE_WEIGHT,
    dataset=None,
    aux_dir=None,
):
    """Build the official D-CGFS synthesis command."""
    command = [
        PYTHON,
        "step2_synthesize_data.py",
        "--ablation-mode",
        "pair_topk_filter",
        "--output-dir",
        output_dir,
        "--pair-topk",
        str(pair_topk),
        "--no-enable-quality-fallback",
        "--pair-score-target-weight",
        str(target_weight),
        "--pair-score-base-weight",
        str(base_weight),
    ]
    if dataset:
        command.extend(["--dataset", dataset])
    if aux_dir:
        command.extend(["--aux-dir", aux_dir])
    return command


def build_dcgfs_train_command(
    checkpoint,
    gen_dir=MAIN_GEN_DIR,
    enable_base_preservation=True,
    base_preservation_weight=BASE_PRESERVATION_WEIGHT,
    dataset=None,
    pair_csv=None,
):
    """Build a D-CGFS training command."""
    command = [
        PYTHON,
        "step3_balance_training.py",
        "--method",
        "dcgfs",
        "--gen-data-dir",
        gen_dir,
        "--output-checkpoint",
        checkpoint,
    ]
    if dataset:
        command.extend(["--dataset", dataset])
    if pair_csv:
        command.extend(["--pair-csv", pair_csv])
    if enable_base_preservation:
        command.extend(
            [
                "--enable-base-preservation",
                "--base-preservation-weight",
                str(base_preservation_weight),
            ]
        )
    return command


def build_final_eval_command(checkpoint, candidate_name, save_dir):
    """Build the final classification evaluation command."""
    return [
        PYTHON,
        "step4_final_evaluation.py",
        "--candidate-checkpoint",
        checkpoint,
        "--candidate-name",
        candidate_name,
        "--save-dir",
        save_dir,
    ]


def build_final_eval_command_for_dataset(checkpoint, candidate_name, save_dir, dataset=None, pair_csv=None):
    command = build_final_eval_command(checkpoint, candidate_name, save_dir)
    if dataset:
        command.extend(["--dataset", dataset])
    if pair_csv:
        command.extend(["--pair-csv", pair_csv])
    return command


def build_seeded_final_eval_command(checkpoint, candidate_name, save_dir, dataset, pair_csv, baseline_checkpoint, seed):
    command = build_final_eval_command_for_dataset(
        checkpoint=checkpoint,
        candidate_name=candidate_name,
        save_dir=save_dir,
        dataset=dataset,
        pair_csv=pair_csv,
    )
    command.extend(["--baseline-checkpoint", baseline_checkpoint, "--seed", str(seed)])
    return command


def build_explain_eval_command(checkpoint, candidate_name, save_dir):
    """Build the concept/explainability evaluation command."""
    return [
        PYTHON,
        "step5_explainability_evaluation.py",
        "--candidate-checkpoint",
        checkpoint,
        "--candidate-name",
        candidate_name,
        "--save-dir",
        save_dir,
    ]


def build_baseline_train_command(method, checkpoint):
    return [
        PYTHON,
        "step3_balance_training.py",
        "--method",
        method,
        "--output-checkpoint",
        checkpoint,
    ]


def build_record(group, name, commands, checkpoint="", eval_dir="", explain_dir=""):
    return {
        "group": group,
        "name": name,
        "commands": commands,
        "checkpoint": checkpoint,
        "eval_dir": eval_dir,
        "explain_dir": explain_dir,
    }


def build_main_records():
    return [
        build_record(
            group="main",
            name=MAIN_NAME,
            commands=[
                [PYTHON, "step1_generate_mapping.py"],
                build_pair_topk_synthesis_command(MAIN_GEN_DIR),
                [PYTHON, "check_synthetic_data.py", "--gen-data-dir", MAIN_GEN_DIR, "--num-samples", "5"],
                build_dcgfs_train_command(MAIN_CHECKPOINT, MAIN_GEN_DIR),
                build_final_eval_command(MAIN_CHECKPOINT, f"{METHOD_ACRONYM} target_score_w015", MAIN_FINAL_DIR),
                build_explain_eval_command(MAIN_CHECKPOINT, f"{METHOD_ACRONYM} target_score_w015", MAIN_EXPLAIN_DIR),
            ],
            checkpoint=MAIN_CHECKPOINT,
            eval_dir=MAIN_FINAL_DIR,
            explain_dir=MAIN_EXPLAIN_DIR,
        )
    ]


def build_awa2_main_records():
    pair_csv = os.path.join(AWA2_AUX_DIR, "target_base_pairs.csv")
    return [
        build_record(
            group="awa2_main",
            name=AWA2_NAME,
            commands=[
                [PYTHON, "find_weak_classes.py", "--dataset", "AwA2", "--save-dir", AWA2_AUX_DIR],
                [PYTHON, "step1_generate_mapping.py", "--dataset", "AwA2", "--save-dir", AWA2_AUX_DIR],
                build_pair_topk_synthesis_command(
                    output_dir=AWA2_GEN_DIR,
                    dataset="AwA2",
                    aux_dir=AWA2_AUX_DIR,
                ),
                [PYTHON, "check_synthetic_data.py", "--dataset", "AwA2", "--gen-data-dir", AWA2_GEN_DIR, "--num-samples", "5"],
                build_dcgfs_train_command(
                    checkpoint=AWA2_CHECKPOINT,
                    gen_dir=AWA2_GEN_DIR,
                    dataset="AwA2",
                    pair_csv=pair_csv,
                ),
                build_final_eval_command_for_dataset(
                    checkpoint=AWA2_CHECKPOINT,
                    candidate_name=f"{METHOD_ACRONYM} AwA2 target_score_w015",
                    save_dir=AWA2_FINAL_DIR,
                    dataset="AwA2",
                    pair_csv=pair_csv,
                ),
            ],
            checkpoint=AWA2_CHECKPOINT,
            eval_dir=AWA2_FINAL_DIR,
        )
    ]


def build_dataset_main_record(dataset, aux_dir, name, gen_dir, checkpoint, final_dir):
    pair_csv = os.path.join(aux_dir, "target_base_pairs.csv")
    return build_record(
        group=f"{dataset.lower()}_main",
        name=name,
        commands=[
            [PYTHON, "find_weak_classes.py", "--dataset", dataset, "--save-dir", aux_dir],
            [PYTHON, "step1_generate_mapping.py", "--dataset", dataset, "--save-dir", aux_dir],
            build_pair_topk_synthesis_command(
                output_dir=gen_dir,
                dataset=dataset,
                aux_dir=aux_dir,
            ),
            [PYTHON, "check_synthetic_data.py", "--dataset", dataset, "--gen-data-dir", gen_dir, "--num-samples", "5"],
            build_dcgfs_train_command(
                checkpoint=checkpoint,
                gen_dir=gen_dir,
                dataset=dataset,
                pair_csv=pair_csv,
            ),
            build_final_eval_command_for_dataset(
                checkpoint=checkpoint,
                candidate_name=f"{METHOD_ACRONYM} {dataset} target_score_w015",
                save_dir=final_dir,
                dataset=dataset,
                pair_csv=pair_csv,
            ),
        ],
        checkpoint=checkpoint,
        eval_dir=final_dir,
    )


def build_pbc_main_records():
    return [
        build_dataset_main_record(
            dataset="PBC",
            aux_dir=PBC_AUX_DIR,
            name=PBC_NAME,
            gen_dir=PBC_GEN_DIR,
            checkpoint=PBC_CHECKPOINT,
            final_dir=PBC_FINAL_DIR,
        )
    ]


def build_7pt_main_records():
    return [
        build_dataset_main_record(
            dataset="7pt",
            aux_dir=SEVEN_PT_AUX_DIR,
            name=SEVEN_PT_NAME,
            gen_dir=SEVEN_PT_GEN_DIR,
            checkpoint=SEVEN_PT_CHECKPOINT,
            final_dir=SEVEN_PT_FINAL_DIR,
        )
    ]


def build_7pt_seed_sweep_records():
    records = []
    for seed in [0, 1, 2]:
        baseline_dir = f"checkpoints/7pt_seed{seed}"
        baseline_checkpoint = os.path.join(baseline_dir, "SemiSupervisedConceptEmbeddingModel.pt")
        aux_dir = f"data/D-CGFS_Auxiliary_7pt_seed{seed}"
        pair_csv = os.path.join(aux_dir, "target_base_pairs.csv")
        gen_dir = f"generated_data/7pt_seed{seed}_target_score_w015"
        checkpoint = f"checkpoints/7pt_seed{seed}_dcgfs_target_score_w015.pt"
        final_dir = f"final_evaluation_results/7pt_seed{seed}_dcgfs_target_score_w015"
        records.append(
            build_record(
                group="7pt_seed_sweep",
                name=f"7pt_seed{seed}_dcgfs_target_score_w015",
                commands=[
                    [
                        PYTHON,
                        "main.py",
                        "--dataset",
                        "7pt",
                        "--device",
                        "cuda",
                        "--image_encoder",
                        "resnet34",
                        "--seed",
                        str(seed),
                        "--fixed_save_dir",
                        baseline_dir,
                    ],
                    [
                        PYTHON,
                        "find_weak_classes.py",
                        "--dataset",
                        "7pt",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--save-dir",
                        aux_dir,
                        "--seed",
                        str(seed),
                    ],
                    [
                        PYTHON,
                        "step1_generate_mapping.py",
                        "--dataset",
                        "7pt",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--save-dir",
                        aux_dir,
                    ],
                    build_pair_topk_synthesis_command(
                        output_dir=gen_dir,
                        dataset="7pt",
                        aux_dir=aux_dir,
                    )
                    + ["--checkpoint", baseline_checkpoint, "--seed", str(seed)],
                    [
                        PYTHON,
                        "check_synthetic_data.py",
                        "--dataset",
                        "7pt",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--gen-data-dir",
                        gen_dir,
                        "--num-samples",
                        "5",
                    ],
                    build_dcgfs_train_command(
                        checkpoint=checkpoint,
                        gen_dir=gen_dir,
                        dataset="7pt",
                        pair_csv=pair_csv,
                    )
                    + ["--checkpoint", baseline_checkpoint, "--seed", str(seed)],
                    build_seeded_final_eval_command(
                        checkpoint=checkpoint,
                        candidate_name=f"{METHOD_ACRONYM} 7pt seed{seed} target_score_w015",
                        save_dir=final_dir,
                        dataset="7pt",
                        pair_csv=pair_csv,
                        baseline_checkpoint=baseline_checkpoint,
                        seed=seed,
                    ),
                ],
                checkpoint=checkpoint,
                eval_dir=final_dir,
            )
        )
    return records


def build_cub_seed_sweep_extra_records():
    records = []
    for seed in [0, 1]:
        baseline_dir = f"checkpoints/CUB-200-2011_seed{seed}"
        baseline_checkpoint = os.path.join(baseline_dir, "SemiSupervisedConceptEmbeddingModel.pt")
        aux_dir = f"data/D-CGFS_Auxiliary_seed{seed}"
        pair_csv = os.path.join(aux_dir, "target_base_pairs.csv")
        gen_dir = f"generated_data/problem6_seed{seed}_target_score_w015"
        checkpoint = f"checkpoints/problem6_seed{seed}_dcgfs_target_score_w015.pt"
        final_dir = f"final_evaluation_results/problem6_seed{seed}_dcgfs_target_score_w015"
        records.append(
            build_record(
                group="cub_seed_sweep_extra",
                name=f"cub_seed{seed}_dcgfs_target_score_w015",
                commands=[
                    [
                        PYTHON,
                        "main.py",
                        "--dataset",
                        "CUB-200-2011",
                        "--device",
                        "cuda",
                        "--image_encoder",
                        "resnet34",
                        "--seed",
                        str(seed),
                        "--fixed_save_dir",
                        baseline_dir,
                    ],
                    [
                        PYTHON,
                        "find_weak_classes.py",
                        "--dataset",
                        "CUB-200-2011",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--save-dir",
                        aux_dir,
                        "--seed",
                        str(seed),
                    ],
                    [
                        PYTHON,
                        "step1_generate_mapping.py",
                        "--dataset",
                        "CUB-200-2011",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--save-dir",
                        aux_dir,
                    ],
                    build_pair_topk_synthesis_command(
                        output_dir=gen_dir,
                        dataset="CUB-200-2011",
                        aux_dir=aux_dir,
                    )
                    + ["--checkpoint", baseline_checkpoint, "--seed", str(seed)],
                    [
                        PYTHON,
                        "check_synthetic_data.py",
                        "--dataset",
                        "CUB-200-2011",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--gen-data-dir",
                        gen_dir,
                        "--num-samples",
                        "5",
                    ],
                    build_dcgfs_train_command(
                        checkpoint=checkpoint,
                        gen_dir=gen_dir,
                        dataset="CUB-200-2011",
                        pair_csv=pair_csv,
                    )
                    + ["--checkpoint", baseline_checkpoint, "--seed", str(seed)],
                    build_seeded_final_eval_command(
                        checkpoint=checkpoint,
                        candidate_name=f"{METHOD_ACRONYM} CUB seed{seed} target_score_w015",
                        save_dir=final_dir,
                        dataset="CUB-200-2011",
                        pair_csv=pair_csv,
                        baseline_checkpoint=baseline_checkpoint,
                        seed=seed,
                    ),
                ],
                checkpoint=checkpoint,
                eval_dir=final_dir,
            )
        )
    return records


def build_7pt_labeled_ratio_sweep_records():
    records = []
    for ratio_tag, ratio in [("r005", "0.05"), ("r020", "0.2")]:
        baseline_dir = f"checkpoints/7pt_{ratio_tag}"
        baseline_checkpoint = os.path.join(baseline_dir, "SemiSupervisedConceptEmbeddingModel.pt")
        aux_dir = f"data/D-CGFS_Auxiliary_7pt_{ratio_tag}"
        pair_csv = os.path.join(aux_dir, "target_base_pairs.csv")
        gen_dir = f"generated_data/7pt_{ratio_tag}_target_score_w015"
        checkpoint = f"checkpoints/7pt_{ratio_tag}_dcgfs_target_score_w015.pt"
        final_dir = f"final_evaluation_results/7pt_{ratio_tag}_dcgfs_target_score_w015"
        records.append(
            build_record(
                group="7pt_labeled_ratio_sweep",
                name=f"7pt_{ratio_tag}_dcgfs_target_score_w015",
                commands=[
                    [
                        PYTHON,
                        "main.py",
                        "--dataset",
                        "7pt",
                        "--device",
                        "cuda",
                        "--image_encoder",
                        "resnet34",
                        "--labeled_ratio",
                        ratio,
                        "--seed",
                        "42",
                        "--fixed_save_dir",
                        baseline_dir,
                    ],
                    [
                        PYTHON,
                        "find_weak_classes.py",
                        "--dataset",
                        "7pt",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--save-dir",
                        aux_dir,
                        "--seed",
                        "42",
                    ],
                    [
                        PYTHON,
                        "step1_generate_mapping.py",
                        "--dataset",
                        "7pt",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--save-dir",
                        aux_dir,
                    ],
                    build_pair_topk_synthesis_command(
                        output_dir=gen_dir,
                        dataset="7pt",
                        aux_dir=aux_dir,
                    )
                    + ["--checkpoint", baseline_checkpoint, "--seed", "42"],
                    [
                        PYTHON,
                        "check_synthetic_data.py",
                        "--dataset",
                        "7pt",
                        "--checkpoint",
                        baseline_checkpoint,
                        "--gen-data-dir",
                        gen_dir,
                        "--num-samples",
                        "5",
                    ],
                    build_dcgfs_train_command(
                        checkpoint=checkpoint,
                        gen_dir=gen_dir,
                        dataset="7pt",
                        pair_csv=pair_csv,
                    )
                    + ["--checkpoint", baseline_checkpoint, "--labeled-ratio", ratio, "--seed", "42"],
                    build_seeded_final_eval_command(
                        checkpoint=checkpoint,
                        candidate_name=f"{METHOD_ACRONYM} 7pt {ratio_tag} target_score_w015",
                        save_dir=final_dir,
                        dataset="7pt",
                        pair_csv=pair_csv,
                        baseline_checkpoint=baseline_checkpoint,
                        seed=42,
                    ),
                ],
                checkpoint=checkpoint,
                eval_dir=final_dir,
            )
        )
    return records


def build_strong_baseline_records():
    records = []
    for method in [
        "sscbm_finetune",
        "oversampling",
        "reweighting",
        "class_balanced_loss",
        "feature_mixup",
    ]:
        checkpoint = f"checkpoints/problem6_{method}.pt"
        eval_dir = f"final_evaluation_results/problem6_{method}"
        records.append(
            build_record(
                group="strong_baseline",
                name=method,
                commands=[
                    build_baseline_train_command(method, checkpoint),
                    build_final_eval_command(checkpoint, method, eval_dir),
                ],
                checkpoint=checkpoint,
                eval_dir=eval_dir,
            )
        )
    return records


def build_ablation_records():
    """Keep only ablations needed to support the paper method claim."""
    variants = [
        {
            "name": "dcgfs_no_conf_filter",
            "gen_dir": "generated_data/problem6_no_conf_filter",
            "checkpoint": "checkpoints/best_sscbm_dcgfs_no_conf_filter.pt",
            "eval_dir": "final_evaluation_results/problem6_dcgfs_no_conf_filter",
            "synth_cmd": [
                PYTHON,
                "step2_synthesize_data.py",
                "--ablation-mode",
                "no_conf_filter",
                "--output-dir",
                "generated_data/problem6_no_conf_filter",
            ],
            "train_base_preservation": False,
        },
        {
            "name": "dcgfs_pair_topk_no_bp",
            "gen_dir": "generated_data/problem6_pair_topk",
            "checkpoint": "checkpoints/best_sscbm_dcgfs_pair_topk.pt",
            "eval_dir": "final_evaluation_results/problem6_dcgfs_pair_topk",
            "synth_cmd": build_pair_topk_synthesis_command(
                output_dir="generated_data/problem6_pair_topk",
                target_weight=0.05,
                base_weight=0.05,
            ),
            "train_base_preservation": False,
        },
        {
            "name": "dcgfs_previous_main_w005_bp",
            "gen_dir": "generated_data/problem6_pair_topk_base_w005",
            "checkpoint": "checkpoints/best_sscbm_dcgfs_pair_topk_base_w005.pt",
            "eval_dir": "final_evaluation_results/problem6_dcgfs_pair_topk_base_w005",
            "synth_cmd": build_pair_topk_synthesis_command(
                output_dir="generated_data/problem6_pair_topk_base_w005",
                target_weight=0.05,
                base_weight=0.05,
            ),
            "train_base_preservation": True,
        },
    ]

    records = []
    for variant in variants:
        records.append(
            build_record(
                group="ablation",
                name=variant["name"],
                commands=[
                    [PYTHON, "step1_generate_mapping.py"],
                    variant["synth_cmd"],
                    [PYTHON, "check_synthetic_data.py", "--gen-data-dir", variant["gen_dir"], "--num-samples", "5"],
                    build_dcgfs_train_command(
                        checkpoint=variant["checkpoint"],
                        gen_dir=variant["gen_dir"],
                        enable_base_preservation=variant["train_base_preservation"],
                    ),
                    build_final_eval_command(variant["checkpoint"], variant["name"], variant["eval_dir"]),
                ],
                checkpoint=variant["checkpoint"],
                eval_dir=variant["eval_dir"],
            )
        )
    return records


def build_report_records():
    return [
        build_record(
            group="report",
            name="paper_results_summary",
            commands=[
                [
                    PYTHON,
                    "summarize_problem6_results.py",
                    "--output-dir",
                    "paper_tables",
                ]
            ],
            eval_dir="paper_tables",
        ),
        build_record(
            group="report",
            name="cub_global_analysis",
            commands=[
                [
                    PYTHON,
                    "analyze_cub_global_results.py",
                    "--output-dir",
                    "paper_tables",
                ]
            ],
            eval_dir="paper_tables",
        )
    ]


def build_second_dataset_check_records():
    records = []
    for dataset in ["AwA2", "PBC", "7pt"]:
        records.append(
            build_record(
                group="second_dataset_check",
                name=f"{dataset}_readiness",
                commands=[
                    [
                        PYTHON,
                        "check_second_dataset_readiness.py",
                        "--dataset",
                        dataset,
                    ]
                ],
                eval_dir="paper_tables/second_dataset_plan.md",
            )
        )
    return records


def build_experiments():
    records = []
    records.extend(build_main_records())
    records.extend(build_awa2_main_records())
    records.extend(build_pbc_main_records())
    records.extend(build_7pt_main_records())
    records.extend(build_7pt_seed_sweep_records())
    records.extend(build_cub_seed_sweep_extra_records())
    records.extend(build_7pt_labeled_ratio_sweep_records())
    records.extend(build_strong_baseline_records())
    records.extend(build_ablation_records())
    records.extend(build_report_records())
    records.extend(build_second_dataset_check_records())
    return records


def command_to_text(command):
    return " ".join(str(item) for item in command)


def save_protocol(records):
    os.makedirs(SAVE_DIR, exist_ok=True)
    rows = []
    for record in records:
        for step_idx, command in enumerate(record["commands"], start=1):
            rows.append(
                {
                    "group": record["group"],
                    "name": record["name"],
                    "step": step_idx,
                    "command": command_to_text(command),
                    "checkpoint": record.get("checkpoint", ""),
                    "eval_dir": record.get("eval_dir", ""),
                    "explain_dir": record.get("explain_dir", ""),
                }
            )
    out_path = os.path.join(SAVE_DIR, "problem6_commands.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate or run the cleaned D-CGFS Problem 6 protocol.")
    parser.add_argument("--run", action="store_true", help="Execute commands. Default only writes the protocol CSV.")
    parser.add_argument(
        "--only-group",
        default=None,
        help=(
            "Run only one group: main, awa2_main, pbc_main, 7pt_main, 7pt_seed_sweep, "
            "cub_seed_sweep_extra, 7pt_labeled_ratio_sweep, strong_baseline, ablation, "
            "report, or second_dataset_check."
        ),
    )
    args = parser.parse_args()

    records = build_experiments()
    if args.only_group:
        records = [record for record in records if record["group"] == args.only_group]
        if not records:
            raise SystemExit(f"未知实验组: {args.only_group}")

    protocol_path = save_protocol(records)
    print(f"实验协议已保存到: {protocol_path}")

    for record in records:
        print(f"\n[{record['group']}] {record['name']}")
        for command in record["commands"]:
            print(command_to_text(command))
            if args.run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
