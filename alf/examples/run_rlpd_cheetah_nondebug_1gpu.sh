#!/bin/bash
# Launcher for one or more RLPD cheetah runs.
#
# Usage: bash run_rlpd_cheetah_nondebug_1gpu.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/alf_results_v7_benchmark_algo)
#   -n, --steps NUM_STEPS              Total env steps (default: 600000)
#   -s, --seed, --seeds SEEDS          Comma-separated seeds (default: 3,4,5)
#   -g, --gpu, --gpus IDS              Comma-separated GPU ids (default: 0,1,2)
#       --critic-utd VALUE             Critic update-to-data ratio (default: 3)
#   -h, --help                         Show this help message
#
# Examples:
#   bash run_rlpd_cheetah_nondebug_1gpu.sh --gpu 0 --seed 3
#   bash run_rlpd_cheetah_nondebug_1gpu.sh --gpus 0,1,2 --seeds 3,4,5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/rlpd_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi

ENV_NAME="cheetah:run"
BASE_DIR="/root/alf_results_v7_benchmark_algo"
NUM_ENV_STEPS=600000
SEEDS=(116 117 118 119)
GPUS=(0 1 2 3)
DEFAULT_CRITIC_UTD=10
CRITIC_UTD=3

parse_csv_list() {
    local raw="$1"
    local -n out="$2"

    IFS=',' read -r -a out <<< "${raw}"
    if [[ ${#out[@]} -eq 0 || -z "${out[0]}" ]]; then
        echo "Expected a non-empty comma-separated list, got: ${raw}" >&2
        exit 1
    fi
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
        -s|--seed|--seeds)
            parse_csv_list "$2" SEEDS
            shift 2
            ;;
        -g|--gpu|--gpus)
            parse_csv_list "$2" GPUS
            shift 2
            ;;
        --critic-utd|--critic_utd)
            CRITIC_UTD="$2"
            shift 2
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

if [[ ! "${CRITIC_UTD}" =~ ^[1-9][0-9]*$ ]]; then
    echo "critic_utd must be a positive integer, got: ${CRITIC_UTD}" >&2
    exit 1
fi
NUM_UPDATES_PER_TRAIN_ITER=$((1 + CRITIC_UTD))

if [[ ${#SEEDS[@]} -ne ${#GPUS[@]} ]]; then
    echo "Seed list and GPU list must have the same length." >&2
    echo "  Seeds: ${SEEDS[*]}" >&2
    echo "  GPUs: ${GPUS[*]}" >&2
    exit 1
fi

ENV_DIR="$(echo "${ENV_NAME}" | cut -d':' -f1)"
ROOT_BASE="${BASE_DIR}/${ENV_DIR}/rlpd_dmc"
RUN_BASE="${ROOT_BASE}"
if [[ "${CRITIC_UTD}" -ne "${DEFAULT_CRITIC_UTD}" ]]; then
    RUN_BASE="${ROOT_BASE}/critic_utd${CRITIC_UTD}"
fi

echo "Launching ${#SEEDS[@]} RLPD run(s)"
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  Critic UTD: ${CRITIC_UTD}"
echo "  Num updates per train iter: ${NUM_UPDATES_PER_TRAIN_ITER}"
echo "  Seeds: ${SEEDS[*]}"
echo "  GPUs: ${GPUS[*]}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Python: ${PYTHON_BIN}"
echo "  Root dir: ${RUN_BASE}"
echo ""

cd "${REPO_ROOT}"

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    GPU="${GPUS[$i]}"

    RUN_DIR="${RUN_BASE}/seed${SEED}"
    mkdir -p "${RUN_DIR}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_DIR}" \
        --conf_param "TrainerConfig.random_seed=${SEED}" \
        --conf_param "TrainerConfig.num_checkpoints=20" \
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "TrainerConfig.num_updates_per_train_iter=${NUM_UPDATES_PER_TRAIN_ITER}" \
        --conf_param "RlpdAlgorithm.critic_utd=${CRITIC_UTD}" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        > "${RUN_DIR}/out.log" 2>&1 &
    PID=$!

    echo "Started seed ${SEED} PID: ${PID} (GPU ${GPU})"
    echo "  Log: ${RUN_DIR}/out.log"
done
