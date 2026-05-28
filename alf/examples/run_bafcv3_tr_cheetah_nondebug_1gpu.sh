#!/bin/bash
# Launcher for one or more BAFCv3-TR runs on 1-4 GPUs.
# Runs trust-gated runs over DELTA_TRUST_MAXES, or a single trust-disabled run.
#
# Usage: bash run_bafcv3_tr_cheetah_nondebug_1gpu.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/alf_results_v7_grad)
#   -n, --steps NUM_STEPS              Total env steps (default: 1000000)
#   -s, --seed SEED                    Seed for the run (default: 1)
#   -g, --gpu ID                       GPU id for a single CLI-selected run
#       --gpus A,B                     Comma-separated GPU ids for multi-run launch
#       --gpu-a ID                     Compatibility alias for --gpu
#       --gpu-b ID                     Compatibility alias that appends a second GPU
#       --eval VALUE                   Eval threshold (default: 115.0)
#       --eval-a VALUE                 Compatibility alias for --eval
#       --eval-b VALUE                 Ignored compatibility flag
#       --delta VALUE                  Grad threshold for a single CLI-selected run (default: 115.0)
#       --delta-a VALUE                Compatibility alias for --delta
#       --delta-b VALUE                Compatibility alias that appends a second grad threshold
#       --deltas A,B                   Comma-separated grad thresholds for multi-run launch
#       --num-feature-coords VALUE     Trust metric feature coords (default: 2)
#       --metric-interval VALUE        Trust metric update interval (default: 8)
#       --rollout-skip-cap VALUE       Max consecutive eval-gated rollout skips (default: 4)
#       --actor-extend-cap VALUE       Max consecutive grad-gated actor extensions (default: 4)
#       --enable-eval-gate BOOL        Enable eval rollout-skip gate (default: false)
#       --enable-grad-gate BOOL        Enable grad actor-extend gate (default: true)
#       --num-checkpoints VALUE        Number of checkpoints to keep (default: 5)
#       --original-algo                Disable trust metrics and both trust gates for BAFCv3-TR
#   -h, --help                         Show this help message
#
# Example:
#   bash run_bafcv3_tr_cheetah_nondebug_1gpu.sh --gpus 0,1 --deltas 40.0,115.0
#   bash run_bafcv3_tr_cheetah_nondebug_1gpu.sh --gpu 0 --eval 2 --delta 2 --rollout-skip-cap 20 --actor-extend-cap 5
#   bash run_bafcv3_tr_cheetah_nondebug_1gpu.sh --gpu 0 --seed 0 --original-algo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_tr_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi

ENV_NAME="cheetah:run"
BASE_DIR="/root/alf_results_v7_grad"
NUM_ENV_STEPS=600000
SEED=5
GPU_IDS=(0 1 2)
EVAL_TRUST_MAX=115.0 #40.0
DELTA_TRUST_MAXES=(95.0 105.0 115.0) #40.0
GPU="${GPU_IDS[0]}"
DELTA_TRUST_MAX="${DELTA_TRUST_MAXES[0]}"
NUM_FEATURE_COORDS=2
METRIC_INTERVAL=8
ROLLOUT_SKIP_CAP=4
ACTOR_EXTEND_CAP=4
ROLLOUT_SKIP_EVAL_INTERVAL=20
GRAD_GATE_EVAL_INTERVAL=20
ENABLE_EVAL_ROLLOUT_SKIP_GATE=False
ENABLE_GRAD_ACTOR_EXTEND_GATE=True
NUM_CHECKPOINTS=5
ORIGINAL_ALGO=false
USE_SINGLE_CLI_RUN=false

split_csv() {
    local value="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "${value}"
}

normalize_bool() {
    local value="${1,,}"
    case "${value}" in
        true|1|yes|y|on)
            echo "True"
            ;;
        false|0|no|n|off)
            echo "False"
            ;;
        *)
            echo "Invalid boolean value: $1 (expected true/false)" >&2
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env)
            ENV_NAME="$2"
            shift 2
            ;;
        -d|--dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -n|--steps)
            NUM_ENV_STEPS="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            USE_SINGLE_CLI_RUN=true
            shift 2
            ;;
        --gpus)
            split_csv "$2" GPU_IDS
            shift 2
            ;;
        --gpu-a)
            GPU="$2"
            USE_SINGLE_CLI_RUN=true
            shift 2
            ;;
        --gpu-b)
            GPU_IDS=("${GPU}" "$2")
            USE_SINGLE_CLI_RUN=false
            shift 2
            ;;
        --eval|--eval-a)
            EVAL_TRUST_MAX="$2"
            shift 2
            ;;
        --eval-b)
            echo "Ignoring --eval-b for single-GPU launcher" >&2
            shift 2
            ;;
        --delta|--delta-a)
            DELTA_TRUST_MAX="$2"
            USE_SINGLE_CLI_RUN=true
            shift 2
            ;;
        --delta-b)
            DELTA_TRUST_MAXES=("${DELTA_TRUST_MAX}" "$2")
            USE_SINGLE_CLI_RUN=false
            shift 2
            ;;
        --deltas)
            split_csv "$2" DELTA_TRUST_MAXES
            shift 2
            ;;
        --seed-b)
            echo "Ignoring --seed-b for single-GPU launcher" >&2
            shift 2
            ;;
        --num-feature-coords)
            NUM_FEATURE_COORDS="$2"
            shift 2
            ;;
        --metric-interval)
            METRIC_INTERVAL="$2"
            shift 2
            ;;
        --rollout-skip-cap)
            ROLLOUT_SKIP_CAP="$2"
            shift 2
            ;;
        --actor-extend-cap)
            ACTOR_EXTEND_CAP="$2"
            shift 2
            ;;
        --enable-eval-gate)
            ENABLE_EVAL_ROLLOUT_SKIP_GATE="$(normalize_bool "$2")"
            shift 2
            ;;
        --enable-grad-gate)
            ENABLE_GRAD_ACTOR_EXTEND_GATE="$(normalize_bool "$2")"
            shift 2
            ;;
        --num-checkpoints)
            NUM_CHECKPOINTS="$2"
            shift 2
            ;;
        --original-algo)
            ORIGINAL_ALGO=true
            shift
            ;;
        -h|--help)
            sed -n '5,28p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [[ "${USE_SINGLE_CLI_RUN}" == "true" ]]; then
    GPU_IDS=("${GPU}")
    DELTA_TRUST_MAXES=("${DELTA_TRUST_MAX}")
fi

if [[ ${#GPU_IDS[@]} -lt 1 || ${#GPU_IDS[@]} -gt 4 ]]; then
    echo "GPU_IDS must contain 1 to 4 entries; got ${#GPU_IDS[@]}" >&2
    exit 1
fi
if [[ "${ORIGINAL_ALGO}" != "true" && ${#GPU_IDS[@]} -ne ${#DELTA_TRUST_MAXES[@]} ]]; then
    echo "GPU_IDS and DELTA_TRUST_MAXES must have the same length" >&2
    echo "  GPU_IDS: ${GPU_IDS[*]}" >&2
    echo "  DELTA_TRUST_MAXES: ${DELTA_TRUST_MAXES[*]}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc_original_algo"
    RUN_DIR="${ROOT_BASE}/seed${SEED}"
else
    ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc_single_run"
fi

if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "Launching BAFCv3-TR trust-disabled mode (single run)"
else
    echo "Launching ${#GPU_IDS[@]} BAFCv3-TR trust-gated run(s)"
fi
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  GPUs: ${GPU_IDS[*]}"
echo "  Seed: ${SEED}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Python: ${PYTHON_BIN}"
echo "  Num checkpoints: ${NUM_CHECKPOINTS}"
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "  Trust metrics: disabled"
    echo "  Eval rollout-actor gate: disabled"
    echo "  Grad actor-extend gate: disabled"
else
    echo "  Eval threshold: ${EVAL_TRUST_MAX}"
    echo "  Grad thresholds: ${DELTA_TRUST_MAXES[*]}"
    echo "  num_feature_coords: ${NUM_FEATURE_COORDS}"
    echo "  metric_interval: ${METRIC_INTERVAL}"
    echo "  rollout_skip_cap: ${ROLLOUT_SKIP_CAP}"
    echo "  actor_extend_cap: ${ACTOR_EXTEND_CAP}"
    echo "  rollout_skip_eval_interval: ${ROLLOUT_SKIP_EVAL_INTERVAL}"
    echo "  grad_gate_eval_interval: ${GRAD_GATE_EVAL_INTERVAL}"
    echo "  Eval rollout-skip gate: ${ENABLE_EVAL_ROLLOUT_SKIP_GATE}"
    echo "  Grad actor-extend gate: ${ENABLE_GRAD_ACTOR_EXTEND_GATE}"
    echo "  Grad gate eval: ${ENABLE_GRAD_ACTOR_EXTEND_GATE}"
fi
echo ""

cd "${REPO_ROOT}"
PIDS=()

if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    GPU="${GPU_IDS[0]}"
    mkdir -p "${RUN_DIR}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_DIR}" \
        --conf_param "TrainerConfig.random_seed=${SEED}" \
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
        --conf_param "TrainerConfig.debug_summaries=True" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        --conf_param "BafcAlgorithmV3.monitor_trust_metrics=False" \
        --conf_param "BafcAlgorithmV3.enable_eval_rollout_skip_gate=False" \
        --conf_param "BafcAlgorithmV3.enable_grad_actor_extend_gate=False" \
        > "${RUN_DIR}/out.log" 2>&1 &
    PID=$!
    PIDS+=("${PID}")
    echo "Started trust-disabled run on GPU ${GPU}: PID ${PID}"
    echo "  Log: ${RUN_DIR}/out.log"
else
    for i in "${!GPU_IDS[@]}"; do
        GPU="${GPU_IDS[$i]}"
        DELTA_TRUST_MAX="${DELTA_TRUST_MAXES[$i]}"
        RUN_DIR="${ROOT_BASE}/eval${EVAL_TRUST_MAX}_delta${DELTA_TRUST_MAX}_nf${NUM_FEATURE_COORDS}_mi${METRIC_INTERVAL}_cap${ROLLOUT_SKIP_CAP}_acap${ACTOR_EXTEND_CAP}_seed${SEED}"
        mkdir -p "${RUN_DIR}"

        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
            --conf "${CONF_FILE}" \
            --root_dir "${RUN_DIR}" \
            --conf_param "TrainerConfig.random_seed=${SEED}" \
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
            --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
            --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
            --conf_param "TrainerConfig.debug_summaries=True" \
            --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
            --conf_param "TrainerConfig.grad_gate_eval=${ENABLE_GRAD_ACTOR_EXTEND_GATE}" \
            --conf_param "TrainerConfig.grad_gate_eval_interval=${GRAD_GATE_EVAL_INTERVAL}" \
            --conf_param "create_environment.env_name='${ENV_NAME}'" \
            --conf_param "BafcAlgorithmV3.monitor_trust_metrics=True" \
            --conf_param "BafcAlgorithmV3.eval_trust_max=${EVAL_TRUST_MAX}" \
            --conf_param "BafcAlgorithmV3.delta_trust_max=${DELTA_TRUST_MAX}" \
            --conf_param "BafcAlgorithmV3.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
            --conf_param "BafcAlgorithmV3.trust_metric_update_interval=${METRIC_INTERVAL}" \
            --conf_param "BafcAlgorithmV3.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
            --conf_param "BafcAlgorithmV3.grad_gate_max_consecutive_actor_extensions=${ACTOR_EXTEND_CAP}" \
            --conf_param "BafcAlgorithmV3.enable_eval_rollout_skip_gate=${ENABLE_EVAL_ROLLOUT_SKIP_GATE}" \
            --conf_param "BafcAlgorithmV3.enable_grad_actor_extend_gate=${ENABLE_GRAD_ACTOR_EXTEND_GATE}" \
            > "${RUN_DIR}/out.log" 2>&1 &

        PID=$!
        PIDS+=("${PID}")
        echo "Started delta ${DELTA_TRUST_MAX} on GPU ${GPU}: PID ${PID}"
        echo "  Log: ${RUN_DIR}/out.log"
    done
fi

echo ""
echo "Started PIDs: ${PIDS[*]}"
