#!/bin/bash
# Launcher for a single original BAFCv3 run.
#
# Usage: bash run_bafcv3_original_cheetah_run_1gpu.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/alf_results_v3)
#   -n, --steps NUM_STEPS              Total env steps (default: 1000000)
#   -s, --seed SEED                    Seed for the run (default: 0)
#   -g, --gpu ID                       GPU id for the run (default: 0)
#   -h, --help                         Show this help message
#
# Example:
#   bash run_bafcv3_original_cheetah_run_1gpu.sh --gpu 0 --seed 0

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
BASE_DIR="/root/alf_results_v3"
NUM_ENV_STEPS=1000000
SEED=0
GPU=1

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
            shift 2
            ;;
        -h|--help)
            sed -n '4,14p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

ENV_DIR="$(echo "${ENV_NAME}" | cut -d':' -f1)"
ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_original_algo"
RUN_DIR="${ROOT_BASE}/seed${SEED}"
mkdir -p "${RUN_DIR}"

echo "Launching original BAFCv3 run (single GPU)"
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  GPU: ${GPU}"
echo "  Seed: ${SEED}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Python: ${PYTHON_BIN}"
echo "  Root dir: ${RUN_DIR}"
echo ""

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
    --conf "${CONF_FILE}" \
    --root_dir "${RUN_DIR}" \
    --conf_param "debug_mode=True" \
    --conf_param "TrainerConfig.random_seed=${SEED}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    > "${RUN_DIR}/out.log" 2>&1 &
PID=$!

echo "Started Run PID: ${PID} (GPU ${GPU})"
echo "Log:"
echo "  ${RUN_DIR}/out.log"
