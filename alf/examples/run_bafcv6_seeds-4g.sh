#!/bin/bash
# Launcher script for BAFCv6 critic-UTD sweeps with 4 seeds.
# Each seed uses all configured GPUs via DDP; all critic_utd jobs run together.
# The launcher starts jobs in the background and exits after printing PIDs.
#
# Usage: bash run_bafcv6_seeds-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 600000)
#       --seeds CSV                 Four comma-separated seeds (default: 1,2,3,4)
#       --critic-utds CSV           Comma-separated critic_utd values (default: 2,3)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 10)
#   -h, --help                      Show this help message
#
# Example:
#   bash run_bafcv6_seeds-4g.sh --env cheetah:run --critic-utds 2,3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv6_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="cheetah:run"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
SEEDS=(1 2 3 4)
CRITIC_UTDS=(2 3)
GPUS="0,1,2,3"

parse_csv_array() {
    local input="$1"
    local -n output_ref="$2"
    IFS=',' read -r -a output_ref <<< "${input}"
}

print_help() {
    sed -n '4,17p' "$0" | sed 's/^# \{0,1\}//'
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
        --seeds)
            parse_csv_array "$2" SEEDS
            shift 2
            ;;
        --critic-utds)
            parse_csv_array "$2" CRITIC_UTDS
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
if [[ ${#SEEDS[@]} -ne 4 ]]; then
    echo "--seeds must provide exactly four comma-separated seeds." >&2
    echo "  Got: ${SEEDS[*]}" >&2
    exit 1
fi
if [[ ${#CRITIC_UTDS[@]} -eq 0 ]]; then
    echo "--critic-utds must provide at least one value." >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv6_dmc_4g"
BASE_PORT=29500

echo "Starting BAFCv6 critic-UTD sweep"
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Root dir: ${ROOT_DIR}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  Num checkpoints: ${NUM_CHECKPOINTS}"
echo "  Seeds: ${SEEDS[*]}"
echo "  critic_utd values: ${CRITIC_UTDS[*]}"
echo "  GPUs: ${GPUS}"
echo "  Critic reweighting: enabled"
echo "  Python: ${PYTHON_BIN}"
echo ""

cd "${REPO_ROOT}"

PIDS=()

for utd_i in "${!CRITIC_UTDS[@]}"; do
    CRITIC_UTD="${CRITIC_UTDS[$utd_i]}"
    echo "Launching critic_utd=${CRITIC_UTD} jobs"

    for i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$i]}"
        MASTER_PORT=$((BASE_PORT + utd_i * ${#SEEDS[@]} + i))
        RUN_DIR="${ROOT_DIR}/critic_utd${CRITIC_UTD}/seed_${SEED}"
        mkdir -p "${RUN_DIR}"

        CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
            "${PYTHON_BIN}" -m alf.bin.train \
            --conf "${CONF_FILE}" \
            --root_dir "${RUN_DIR}" \
            --conf_param "TrainerConfig.random_seed=${SEED}" \
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
            --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
            --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
            --conf_param "BafcAlgorithmV6.critic_utd=${CRITIC_UTD}" \
            --conf_param "BafcAlgorithmV6.enable_critic_reweighting=True" \
            --conf_param "create_environment.env_name='${ENV_NAME}'" \
            --distributed multi-gpu \
            > "${RUN_DIR}/out.log" 2>&1 &

        PID=$!
        PIDS+=("${PID}")
        echo "  Seed ${SEED}: GPUs ${GPUS}, port ${MASTER_PORT}, PID ${PID}"
        echo "    Log: ${RUN_DIR}/out.log"
    done
    echo ""
done

echo "Launched BAFCv6 jobs: ${PIDS[*]}"
echo "Launcher is not waiting for completion."
echo "To monitor: tail -f ${ROOT_DIR}/critic_utd*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
