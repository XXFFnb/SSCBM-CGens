#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================================
# D-CGFS 主流程一键运行脚本
#
# 默认模式 main：
#   跑一遍当前论文主配置，包括弱势类发现、mask 生成、target-score pair-topk 合成数据、
#   合成数据检查、base preservation(w=0.15) 平衡微调、最终分类评估和可解释性评估。
#
# 可选模式 protocol：
#   只生成问题六/问题八的大规模实验命令表，不真正训练。
#
# 可选模式 resume：
#   当 find_weak_classes.py 和 step1_generate_mapping.py 已经成功完成时，
#   从当前论文主配置的 step2 target-score pair-topk 合成数据开始继续运行。
#
# 可选模式 problem6：
#   按 run_problem6_experiments.py --run 顺序执行全部强 baseline、消融、不平衡比例、
#   概念标注比例实验。这个模式非常耗时，建议确认资源后再运行。
#
# 可选模式 pair_topk：
#   运行新的 pair 内 top-k 合成策略。该策略用于替代过严的 target_prob 绝对阈值，
#   每个 target-base pair 保留固定数量的高质量合成样本。
#
# 可选模式 pair_topk_base：
#   复用 pair_topk 合成数据，在训练阶段启用 base preservation，重点验证能否恢复
#   selected_base_acc。
#
# 可选模式 pair_topk_base_sweep：
#   复用 pair_topk 合成数据，批量扫描 base preservation 权重 0.05/0.10/0.15，
#   用于寻找目标类修复与基座类保持之间更合适的折中点。
#
# 可选模式 pair_topk_base_fine_sweep：
#   围绕当前较优的 0.15 做局部细扫，运行 0.125/0.175 两个权重。
#
# 可选模式 pair_topk_base_ultra_fine_sweep：
#   在 0.15 附近做更细粒度验证，运行 0.135/0.145/0.155/0.165。
# ==========================================================================================

MODE="${1:-main}"
PYTHON="${PYTHON:-.venv/bin/python}"

PAIR_CSV="data/D-CGFS_Auxiliary/target_base_pairs.csv"
AUX_DIR="data/D-CGFS_Auxiliary"
GEN_DIR="generated_data/dcgfs_target_score_w015"
CHECKPOINT="checkpoints/best_sscbm_dcgfs_target_score_w015.pt"
FINAL_DIR="final_evaluation_results/dcgfs_target_score_w015"
EXPLAIN_DIR="explainability_results/dcgfs_target_score_w015"
BASE_PRESERVATION_WEIGHT="0.15"
PAIR_SCORE_TARGET_WEIGHT="0.15"
PAIR_SCORE_BASE_WEIGHT="0.05"
LOG_DIR="run_logs"

mkdir -p "${LOG_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-dcgfs}"
mkdir -p "${MPLCONFIGDIR}"

run_step() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  echo
  echo "========== ${name} =========="
  echo "命令: $*"
  "$@" 2>&1 | tee "${log_file}"
  echo "日志: ${log_file}"
}

print_main_outputs() {
  echo
  echo "========== 主流程输出位置 =========="
  echo "自动 target-base pair:"
  echo "  ${PAIR_CSV}"
  echo
  echo "Step1 mask 与映射:"
  echo "  ${AUX_DIR}/concept_region_mapping.csv"
  echo "  ${AUX_DIR}/base_sample_regions.csv"
  echo "  ${AUX_DIR}/masks/"
  echo
  echo "Step2 合成数据:"
  echo "  ${GEN_DIR}/synthesized_metadata.csv"
  echo "  ${GEN_DIR}/feat_*.pt"
  echo "  ${GEN_DIR}/synthesized_images/"
  echo
  echo "Step3 训练后模型:"
  echo "  ${CHECKPOINT}"
  echo
  echo "Step4 分类评估:"
  echo "  ${FINAL_DIR}/model_summary.csv"
  echo "  ${FINAL_DIR}/target_class_accuracy.csv"
  echo "  ${FINAL_DIR}/target_base_confusion.csv"
  echo "  ${FINAL_DIR}/classwise_metrics.csv"
  echo
  echo "Step5 可解释性评估:"
  echo "  ${EXPLAIN_DIR}/concept_accuracy.csv"
  echo "  ${EXPLAIN_DIR}/heatmap_quality.csv"
  echo "  ${EXPLAIN_DIR}/concept_intervention.csv"
  echo
  echo "运行日志:"
  echo "  ${LOG_DIR}/"
}

run_main() {
  echo "开始运行 D-CGFS 当前论文主流程。"
  echo "主配置：target-score pair-topk 合成 + base preservation，pair_score_target_weight=${PAIR_SCORE_TARGET_WEIGHT}, base_preservation_weight=${BASE_PRESERVATION_WEIGHT}。"
  echo "说明：step4/step5 使用测试集做最终评估；find_weak_classes 使用验证集选择目标类和基座类。"

  run_step "01_find_weak_classes" \
    "${PYTHON}" find_weak_classes.py

  run_step "02_generate_mapping" \
    "${PYTHON}" step1_generate_mapping.py

  run_step "03_synthesize_data_target_score_w015" \
    "${PYTHON}" step2_synthesize_data.py \
      --ablation-mode pair_topk_filter \
      --output-dir "${GEN_DIR}" \
      --pair-topk 500 \
      --pair-score-target-weight "${PAIR_SCORE_TARGET_WEIGHT}" \
      --pair-score-base-weight "${PAIR_SCORE_BASE_WEIGHT}" \
      --no-enable-quality-fallback

  run_step "04_check_synthetic_data_target_score_w015" \
    "${PYTHON}" check_synthetic_data.py \
      --gen-data-dir "${GEN_DIR}" \
      --num-samples 5

  run_step "05_train_target_score_w015" \
    "${PYTHON}" step3_balance_training.py \
      --method dcgfs \
      --gen-data-dir "${GEN_DIR}" \
      --output-checkpoint "${CHECKPOINT}" \
      --enable-base-preservation \
      --base-preservation-weight "${BASE_PRESERVATION_WEIGHT}"

  run_step "06_final_evaluation_target_score_w015" \
    "${PYTHON}" step4_final_evaluation.py \
      --candidate-checkpoint "${CHECKPOINT}" \
      --candidate-name "D-CGFS target_score_w015" \
      --save-dir "${FINAL_DIR}"

  run_step "07_explainability_evaluation_target_score_w015" \
    "${PYTHON}" step5_explainability_evaluation.py \
      --candidate-checkpoint "${CHECKPOINT}" \
      --candidate-name "D-CGFS target_score_w015" \
      --save-dir "${EXPLAIN_DIR}"

  print_main_outputs
}

run_resume() {
  echo "从 step2 合成数据开始恢复运行 D-CGFS 当前论文主流程。"
  echo "主配置：target-score pair-topk 合成 + base preservation，pair_score_target_weight=${PAIR_SCORE_TARGET_WEIGHT}, base_preservation_weight=${BASE_PRESERVATION_WEIGHT}。"
  echo "前提：${PAIR_CSV}、${AUX_DIR}/concept_region_mapping.csv 和 ${AUX_DIR}/base_sample_regions.csv 已存在。"

  run_step "03_synthesize_data_target_score_w015" \
    "${PYTHON}" step2_synthesize_data.py \
      --ablation-mode pair_topk_filter \
      --output-dir "${GEN_DIR}" \
      --pair-topk 500 \
      --pair-score-target-weight "${PAIR_SCORE_TARGET_WEIGHT}" \
      --pair-score-base-weight "${PAIR_SCORE_BASE_WEIGHT}" \
      --no-enable-quality-fallback

  run_step "04_check_synthetic_data_target_score_w015" \
    "${PYTHON}" check_synthetic_data.py \
      --gen-data-dir "${GEN_DIR}" \
      --num-samples 5

  run_step "05_train_target_score_w015" \
    "${PYTHON}" step3_balance_training.py \
      --method dcgfs \
      --gen-data-dir "${GEN_DIR}" \
      --output-checkpoint "${CHECKPOINT}" \
      --enable-base-preservation \
      --base-preservation-weight "${BASE_PRESERVATION_WEIGHT}"

  run_step "06_final_evaluation_target_score_w015" \
    "${PYTHON}" step4_final_evaluation.py \
      --candidate-checkpoint "${CHECKPOINT}" \
      --candidate-name "D-CGFS target_score_w015" \
      --save-dir "${FINAL_DIR}"

  run_step "07_explainability_evaluation_target_score_w015" \
    "${PYTHON}" step5_explainability_evaluation.py \
      --candidate-checkpoint "${CHECKPOINT}" \
      --candidate-name "D-CGFS target_score_w015" \
      --save-dir "${EXPLAIN_DIR}"

  print_main_outputs
}

run_protocol() {
  echo "只生成问题六/问题八实验命令表，不执行大规模训练。"
  run_step "problem6_protocol" \
    "${PYTHON}" run_problem6_experiments.py
  echo
  echo "实验命令表:"
  echo "  problem6_experiment_protocol/problem6_commands.csv"
}

run_problem6() {
  echo "即将运行问题六/问题八全部实验。这个过程会非常耗时。"
  run_step "problem6_full_run" \
    "${PYTHON}" run_problem6_experiments.py --run
  echo
  echo "全部实验结果会分散保存到:"
  echo "  final_evaluation_results/problem6_*"
  echo "  checkpoints/problem6_*.pt"
  echo "  generated_data/dcgfs_pair_topk 或 generated_data/ablation_*"
}

run_pair_topk() {
  local gen_dir="generated_data/dcgfs_pair_topk"
  local checkpoint="checkpoints/best_sscbm_dcgfs_pair_topk.pt"
  local final_dir="final_evaluation_results/dcgfs_pair_topk"
  local explain_dir="explainability_results/dcgfs_pair_topk"

  echo "开始运行 D-CGFS pair_topk_filter 对照流程。"
  echo "说明：该模式默认假设 find_weak_classes.py 和 step1_generate_mapping.py 已完成。"
  echo "如果 target-base pair 或 mask 不存在，请先运行 find_weak_classes.py 和 step1_generate_mapping.py，或直接运行完整主流程: bash run_dcgfs_pipeline.sh main。"

  run_step "03_synthesize_data_pair_topk" \
    "${PYTHON}" step2_synthesize_data.py \
      --ablation-mode pair_topk_filter \
      --output-dir "${gen_dir}" \
      --pair-topk 500 \
      --no-enable-quality-fallback

  run_step "04_check_synthetic_data_pair_topk" \
    "${PYTHON}" check_synthetic_data.py \
      --gen-data-dir "${gen_dir}" \
      --num-samples 5

  run_step "05_train_pair_topk" \
    "${PYTHON}" step3_balance_training.py \
      --method dcgfs \
      --gen-data-dir "${gen_dir}" \
      --output-checkpoint "${checkpoint}"

  run_step "06_final_evaluation_pair_topk" \
    "${PYTHON}" step4_final_evaluation.py \
      --candidate-checkpoint "${checkpoint}" \
      --candidate-name "D-CGFS pair_topk" \
      --save-dir "${final_dir}"

  run_step "07_explainability_evaluation_pair_topk" \
    "${PYTHON}" step5_explainability_evaluation.py \
      --candidate-checkpoint "${checkpoint}" \
      --candidate-name "D-CGFS pair_topk" \
      --save-dir "${explain_dir}"

  echo
  echo "========== pair_topk 输出位置 =========="
  echo "合成数据: ${gen_dir}/synthesized_metadata.csv"
  echo "训练模型: ${checkpoint}"
  echo "分类评估: ${final_dir}/"
  echo "可解释性评估: ${explain_dir}/"
}

run_pair_topk_base() {
  local gen_dir="${GEN_DIR}"
  local checkpoint="${CHECKPOINT}"
  local final_dir="${FINAL_DIR}"
  local explain_dir="${EXPLAIN_DIR}"

  echo "开始运行 D-CGFS 当前论文主配置：target-score pair_topk + base preservation(w=0.15)。"
  echo "说明：复用 ${gen_dir} 合成数据，只重新训练和评估。"
  echo "如果 ${gen_dir}/synthesized_metadata.csv 不存在，请先运行: bash run_dcgfs_pipeline.sh resume"

  run_step "05_train_target_score_w015" \
    "${PYTHON}" step3_balance_training.py \
      --method dcgfs \
      --gen-data-dir "${gen_dir}" \
      --output-checkpoint "${checkpoint}" \
      --enable-base-preservation \
      --base-preservation-weight "0.15"

  run_step "06_final_evaluation_target_score_w015" \
    "${PYTHON}" step4_final_evaluation.py \
      --candidate-checkpoint "${checkpoint}" \
      --candidate-name "D-CGFS target_score_w015" \
      --save-dir "${final_dir}"

  run_step "07_explainability_evaluation_target_score_w015" \
    "${PYTHON}" step5_explainability_evaluation.py \
      --candidate-checkpoint "${checkpoint}" \
      --candidate-name "D-CGFS target_score_w015" \
      --save-dir "${explain_dir}"

  echo
  echo "========== target_score_w015 输出位置 =========="
  echo "复用合成数据: ${gen_dir}/synthesized_metadata.csv"
  echo "训练模型: ${checkpoint}"
  echo "分类评估: ${final_dir}/"
  echo "可解释性评估: ${explain_dir}/"
}

run_pair_topk_base_weight() {
  local weight="$1"
  local tag="$2"
  local gen_dir="generated_data/dcgfs_pair_topk"
  local checkpoint="checkpoints/best_sscbm_dcgfs_pair_topk_base_w${tag}.pt"
  local final_dir="final_evaluation_results/dcgfs_pair_topk_base_w${tag}"
  local explain_dir="explainability_results/dcgfs_pair_topk_base_w${tag}"

  echo
  echo "开始运行 D-CGFS pair_topk + base preservation 权重实验。"
  echo "说明：复用 ${gen_dir} 合成数据，只重新训练和评估。"
  echo "base_preservation_weight=${weight}"

  # 每个权重单独命名日志、模型和结果目录，防止多轮实验互相覆盖。
  run_step "05_train_pair_topk_base_w${tag}" \
    "${PYTHON}" step3_balance_training.py \
      --method dcgfs \
      --gen-data-dir "${gen_dir}" \
      --output-checkpoint "${checkpoint}" \
      --enable-base-preservation \
      --base-preservation-weight "${weight}"

  run_step "06_final_evaluation_pair_topk_base_w${tag}" \
    "${PYTHON}" step4_final_evaluation.py \
      --candidate-checkpoint "${checkpoint}" \
      --candidate-name "D-CGFS pair_topk_base_w${tag}" \
      --save-dir "${final_dir}"

  run_step "07_explainability_evaluation_pair_topk_base_w${tag}" \
    "${PYTHON}" step5_explainability_evaluation.py \
      --candidate-checkpoint "${checkpoint}" \
      --candidate-name "D-CGFS pair_topk_base_w${tag}" \
      --save-dir "${explain_dir}"

  echo
  echo "========== pair_topk_base_w${tag} 输出位置 =========="
  echo "复用合成数据: ${gen_dir}/synthesized_metadata.csv"
  echo "训练模型: ${checkpoint}"
  echo "分类评估: ${final_dir}/"
  echo "可解释性评估: ${explain_dir}/"
}

run_pair_topk_base_sweep() {
  echo "开始批量扫描 base preservation 权重。"
  echo "本模式会依次运行 0.05、0.10、0.15 三个实验。"
  echo "如果 generated_data/dcgfs_pair_topk/synthesized_metadata.csv 不存在，请先运行: bash run_dcgfs_pipeline.sh pair_topk"

  # tag 用纯数字表示权重，便于目录排序和后续汇总。
  run_pair_topk_base_weight "0.05" "005"
  run_pair_topk_base_weight "0.10" "010"
  run_pair_topk_base_weight "0.15" "015"

  echo
  echo "========== 权重扫描输出位置 =========="
  echo "分类评估:"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w005/"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w010/"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w015/"
  echo "可解释性评估:"
  echo "  explainability_results/dcgfs_pair_topk_base_w005/"
  echo "  explainability_results/dcgfs_pair_topk_base_w010/"
  echo "  explainability_results/dcgfs_pair_topk_base_w015/"
}

run_pair_topk_base_fine_sweep() {
  echo "开始围绕 base_preservation_weight=0.15 做局部细扫。"
  echo "本模式会依次运行 0.125、0.175 两个实验。"
  echo "如果 generated_data/dcgfs_pair_topk/synthesized_metadata.csv 不存在，请先运行: bash run_dcgfs_pipeline.sh pair_topk"

  # 0.15 已经跑过，本模式只补左右两侧细分点，避免重复训练。
  run_pair_topk_base_weight "0.125" "0125"
  run_pair_topk_base_weight "0.175" "0175"

  echo
  echo "========== 局部细扫输出位置 =========="
  echo "分类评估:"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w0125/"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w0175/"
  echo "可解释性评估:"
  echo "  explainability_results/dcgfs_pair_topk_base_w0125/"
  echo "  explainability_results/dcgfs_pair_topk_base_w0175/"
}

run_pair_topk_base_ultra_fine_sweep() {
  echo "开始围绕 base_preservation_weight=0.15 做更细粒度局部验证。"
  echo "本模式会依次运行 0.135、0.145、0.155、0.165 四个实验。"
  echo "如果 generated_data/dcgfs_pair_topk/synthesized_metadata.csv 不存在，请先运行: bash run_dcgfs_pipeline.sh pair_topk"

  # 0.15 已经跑过，这里只补左右更密集的点，判断局部峰值是否偏离 0.15。
  run_pair_topk_base_weight "0.135" "0135"
  run_pair_topk_base_weight "0.145" "0145"
  run_pair_topk_base_weight "0.155" "0155"
  run_pair_topk_base_weight "0.165" "0165"

  echo
  echo "========== 更细局部验证输出位置 =========="
  echo "分类评估:"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w0135/"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w0145/"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w0155/"
  echo "  final_evaluation_results/dcgfs_pair_topk_base_w0165/"
  echo "可解释性评估:"
  echo "  explainability_results/dcgfs_pair_topk_base_w0135/"
  echo "  explainability_results/dcgfs_pair_topk_base_w0145/"
  echo "  explainability_results/dcgfs_pair_topk_base_w0155/"
  echo "  explainability_results/dcgfs_pair_topk_base_w0165/"
}

case "${MODE}" in
  main)
    run_main
    ;;
  resume)
    run_resume
    ;;
  protocol)
    run_protocol
    ;;
  problem6)
    run_problem6
    ;;
  pair_topk)
    run_pair_topk
    ;;
  pair_topk_base)
    run_pair_topk_base
    ;;
  pair_topk_base_sweep)
    run_pair_topk_base_sweep
    ;;
  pair_topk_base_fine_sweep)
    run_pair_topk_base_fine_sweep
    ;;
  pair_topk_base_ultra_fine_sweep)
    run_pair_topk_base_ultra_fine_sweep
    ;;
  *)
    echo "未知模式: ${MODE}"
    echo "用法:"
    echo "  bash run_dcgfs_pipeline.sh main"
    echo "  bash run_dcgfs_pipeline.sh resume"
    echo "  bash run_dcgfs_pipeline.sh protocol"
    echo "  bash run_dcgfs_pipeline.sh problem6"
    echo "  bash run_dcgfs_pipeline.sh pair_topk"
    echo "  bash run_dcgfs_pipeline.sh pair_topk_base"
    echo "  bash run_dcgfs_pipeline.sh pair_topk_base_sweep"
    echo "  bash run_dcgfs_pipeline.sh pair_topk_base_fine_sweep"
    echo "  bash run_dcgfs_pipeline.sh pair_topk_base_ultra_fine_sweep"
    exit 1
    ;;
esac
