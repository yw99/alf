#!/bin/bash
# Launcher for original BAFCv3 cheetah learning-rate sweep runs.
#
# Usage: bash run_bafcv3_original_cheetah_lr_sweep_nondebug_1gpu.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/alf_results_v7_original_algo_ln)
#   -n, --steps NUM_STEPS              Total env steps (default: 700000)
#   -s, --seed SEED                    Seed for each run (default: 3)
#       --actor-ln                     Enable actor layer normalization
#   -h, --help                         Show this help message
#
# Edit GPU_IDS and LEARNING_RATES below to choose the sweep.
# Each GPU_IDS[i] runs LEARNING_RATES[i].
#
# Example:
#   bash run_bafcv3_original_cheetah_lr_sweep_nondebug_1gpu.sh --seed 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi

ENV_NAME="cheetah:run"
BASE_DIR="/root/alf_results_v7_original_algo_ln"
NUM_ENV_STEPS=700000
SEED=3
GPU_IDS=(0 1 2)
LEARNING_RATES=(5e-4 1e-3 5e-3)
ACTOR_USE_LN=True

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
        --actor-ln)
            ACTOR_USE_LN=True
            shift
            ;;
        -h|--help)
            sed -n '4,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "GPU_IDS must contain at least one entry" >&2
    exit 1
fi
if [[ ${#GPU_IDS[@]} -ne ${#LEARNING_RATES[@]} ]]; then
    echo "GPU_IDS and LEARNING_RATES must have the same length" >&2
    echo "  GPU_IDS: ${GPU_IDS[*]}" >&2
    echo "  LEARNING_RATES: ${LEARNING_RATES[*]}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_original_algo_lr_sweep"

echo "Launching ${#GPU_IDS[@]} original BAFCv3 cheetah LR sweep run(s)"
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  GPUs: ${GPU_IDS[*]}"
echo "  Learning rates: ${LEARNING_RATES[*]}"
echo "  Seed: ${SEED}"
echo "  Actor layer norm: ${ACTOR_USE_LN}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Python: ${PYTHON_BIN}"
echo "  Root base: ${ROOT_BASE}"
echo ""

cd "${REPO_ROOT}"
PIDS=()

for i in "${!GPU_IDS[@]}"; do
    GPU="${GPU_IDS[$i]}"
    LEARNING_RATE="${LEARNING_RATES[$i]}"
    RUN_DIR="${ROOT_BASE}/lr${LEARNING_RATE}_seed${SEED}"
    mkdir -p "${RUN_DIR}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_DIR}" \
        --conf_param "TrainerConfig.random_seed=${SEED}" \
        --conf_param "bafcv3_actor_use_ln=${ACTOR_USE_LN}" \
        --conf_param "bafcv3_learning_rate=${LEARNING_RATE}" \
        --conf_param "TrainerConfig.num_checkpoints=20" \
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        > "${RUN_DIR}/out.log" 2>&1 &

    PID=$!
    PIDS+=("${PID}")
    echo "Started LR ${LEARNING_RATE} on GPU ${GPU}: PID ${PID}"
    echo "  Log: ${RUN_DIR}/out.log"
done

echo ""
echo "Started PIDs: ${PIDS[*]}"
