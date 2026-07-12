#!/bin/bash
# Launcher script for BAFCv3 learning-rate sweeps with seeds 1 and 3.
# Each seed uses all configured GPUs via DDP, and all jobs launch in parallel.
#
# Usage: bash run_bafcv3_lr_sweep_seeds13-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 480000)
#       --critic-utd N              Critic update-to-data ratio (default: 2)
#       --learning-rates LR1,LR2    Two comma-separated learning rates (default: 3e-4,5e-4)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 10)
#   -h, --help                      Show this help message
#
# Examples:
#   bash run_bafcv3_lr_sweep_seeds13-4g.sh
#   bash run_bafcv3_lr_sweep_seeds13-4g.sh --critic-utd 3
#   bash run_bafcv3_lr_sweep_seeds13-4g.sh --learning-rates 3e-4,1e-3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=480000
NUM_CHECKPOINTS=10
CRITIC_UTD=2
LEARNING_RATES=(3e-4 5e-4)
SEEDS=(0 1)
GPUS="0,1,2,3"
BASE_PORT=29500

parse_csv_array() {
    local input="$1"
    local -n output_ref="$2"
    IFS=',' read -r -a output_ref <<< "${input}"
}

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
        --critic-utd)
            CRITIC_UTD="$2"
            shift 2
            ;;
        --learning-rates)
            parse_csv_array "$2" LEARNING_RATES
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
if [[ ! "${CRITIC_UTD}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--critic-utd must be a positive integer, got: ${CRITIC_UTD}" >&2
    exit 1
fi
if [[ ${#LEARNING_RATES[@]} -ne 2 ]]; then
    echo "--learning-rates must provide exactly two comma-separated values." >&2
    echo "  Got: ${LEARNING_RATES[*]}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_4g_lr_sweep/critic_utd${CRITIC_UTD}"

cat <<EOF
Starting BAFCv3 learning-rate sweep
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  critic_utd: ${CRITIC_UTD}
  learning rates: ${LEARNING_RATES[*]}
  GPUs: ${GPUS}
  Python: ${PYTHON_BIN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for lr_i in "${!LEARNING_RATES[@]}"; do
    LEARNING_RATE="${LEARNING_RATES[$lr_i]}"
    echo "Launching learning_rate=${LEARNING_RATE} jobs"

    for seed_i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$seed_i]}"
        MASTER_PORT=$((BASE_PORT + lr_i * ${#SEEDS[@]} + seed_i))
        RUN_DIR="${ROOT_DIR}/lr${LEARNING_RATE}/seed_${SEED}"
        mkdir -p "${RUN_DIR}"

        CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
            "${PYTHON_BIN}" -m alf.bin.train \
            --conf "${CONF_FILE}" \
            --root_dir "${RUN_DIR}" \
            --conf_param "TrainerConfig.random_seed=${SEED}" \
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
            --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
            --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
            --conf_param "BafcAlgorithmV3.critic_utd=${CRITIC_UTD}" \
            --conf_param "bafcv3_learning_rate=${LEARNING_RATE}" \
            --conf_param "create_environment.env_name='${ENV_NAME}'" \
            --distributed multi-gpu \
            > "${RUN_DIR}/out.log" 2>&1 &

        PID=$!
        PIDS+=("${PID}")
        echo "  LR ${LEARNING_RATE}, seed ${SEED}: GPUs ${GPUS}, port ${MASTER_PORT}, PID ${PID}"
        echo "    Log: ${RUN_DIR}/out.log"
    done
    echo ""
done

echo ""
echo "Launched BAFCv3 4-GPU jobs: ${PIDS[*]}"
echo "Launcher is not waiting for completion."
echo "To monitor: tail -f ${ROOT_DIR}/lr*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
