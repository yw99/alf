#!/bin/bash
# Debug launcher for one BAFCv3 seed using all configured GPUs via DDP.
#
# Usage: bash run_bafcv3_seed-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 10000)
#       --seed N                    Random seed (default: 0)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 5)
#       --port N                    DDP master port (default: 29500)
#   -h, --help                      Show this help message
#
# Examples:
#   bash run_bafcv3_seed-4g.sh
#   bash run_bafcv3_seed-4g.sh --seed 1 --gpus 0,1,2,3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="cheetah:run"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=14000
NUM_CHECKPOINTS=2
SEED=0
GPUS="0,1,2,3"
MASTER_PORT=29500

print_help() {
    sed -n '/^# Usage:/,/^# Examples:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
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
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --checkpoints)
            NUM_CHECKPOINTS="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use -h or --help for usage information" >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi
if [[ ! -f "${CONF_FILE}" ]]; then
    echo "Config file not found: ${CONF_FILE}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_4g_debug"
RUN_DIR="${ROOT_DIR}/seed_${SEED}"
mkdir -p "${RUN_DIR}"

cat <<EOF
Starting BAFCv3 4-GPU debug run
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Run dir: ${RUN_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seed: ${SEED}
  GPUs: ${GPUS}
  Master port: ${MASTER_PORT}
  Python: ${PYTHON_BIN}
EOF
echo ""

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
    "${PYTHON_BIN}" -m alf.bin.train \
    --conf "${CONF_FILE}" \
    --root_dir "${RUN_DIR}" \
    --conf_param "TrainerConfig.random_seed=${SEED}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${RUN_DIR}/out.log" 2>&1 &

PID=$!
echo "Launched BAFCv3 4-GPU debug run: seed ${SEED}, GPUs ${GPUS}, port ${MASTER_PORT}, PID ${PID}"
echo "Log: ${RUN_DIR}/out.log"
echo "To monitor: tail -f ${RUN_DIR}/out.log"
