#!/bin/bash
# Launcher script for BAFCv3 bootstrap sweeps with seeds 0 and 1.
# Each seed uses all configured GPUs via DDP, and all jobs launch in parallel.
#
# Usage: bash run_bafcv3_bootstrap_sweep_seed01-4g.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS              Total environment steps (default: 480000)
#       --critic-utd N                 Critic update-to-data ratio (default: 11)
#       --learning-rate LR             Learning rate for both conditions (default: 3e-4)
#       --actor-critic-pairing BOOL    Fix actor-critic pairing (default: True)
#       --num-actor-critic N           Number of actor-critic pairs (default: 10)
#       --num-attention-heads N        Transformer attention heads (default: 1)
#       --gpus CSV                     Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N                Number of checkpoints (default: 10)
#   -h, --help                         Show this help message
#
# Examples:
#   bash run_bafcv3_bootstrap_sweep_seed01-4g.sh
#   bash run_bafcv3_bootstrap_sweep_seed01-4g.sh --critic-utd 3
#   bash run_bafcv3_bootstrap_sweep_seed01-4g.sh --actor-critic-pairing false
#   bash run_bafcv3_bootstrap_sweep_seed01-4g.sh --num-actor-critic 8 --num-attention-heads 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=480000
NUM_CHECKPOINTS=10
CRITIC_UTD=5
LEARNING_RATE=3e-4
ACTOR_CRITIC_PAIRING=True
NUM_ACTOR_CRITIC=20
NUM_ATTENTION_HEADS=1
DEBUG_SUMMARIES=True
BOOTSTRAP_VALUES=(False True)
SEEDS=(0 1)
GPUS="0,1,2,3"
BASE_PORT=29500

normalize_bool() {
    case "${1,,}" in
        true)
            echo "True"
            ;;
        false)
            echo "False"
            ;;
        *)
            return 1
            ;;
    esac
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
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --actor-critic-pairing)
            if ! ACTOR_CRITIC_PAIRING="$(normalize_bool "$2")"; then
                echo "--actor-critic-pairing must be true or false, got: $2" >&2
                exit 1
            fi
            shift 2
            ;;
        --num-actor-critic)
            NUM_ACTOR_CRITIC="$2"
            shift 2
            ;;
        --num-attention-heads)
            NUM_ATTENTION_HEADS="$2"
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
if [[ ! "${NUM_ACTOR_CRITIC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-actor-critic must be a positive integer, got: ${NUM_ACTOR_CRITIC}" >&2
    exit 1
fi
if [[ ! "${NUM_ATTENTION_HEADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-attention-heads must be a positive integer, got: ${NUM_ATTENTION_HEADS}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_4g_bootstrap_sweep/critic_utd${CRITIC_UTD}/lr${LEARNING_RATE}/actor_critic_pairing${ACTOR_CRITIC_PAIRING}_num_actor_critic${NUM_ACTOR_CRITIC}_num_attention_heads${NUM_ATTENTION_HEADS}"

cat <<EOF
Starting BAFCv3 bootstrap sweep
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  bootstrap actors/critics: False/False True/True
  critic_utd: ${CRITIC_UTD}
  learning rate: ${LEARNING_RATE}
  actor_critic_pairing: ${ACTOR_CRITIC_PAIRING}
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  num_attention_heads: ${NUM_ATTENTION_HEADS}
  debug_summaries: ${DEBUG_SUMMARIES}
  GPUs: ${GPUS}
  Python: ${PYTHON_BIN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for bootstrap_i in "${!BOOTSTRAP_VALUES[@]}"; do
    BOOTSTRAP="${BOOTSTRAP_VALUES[$bootstrap_i]}"
    echo "Launching use_bootstrap_actors=${BOOTSTRAP}, use_bootstrap_critics=${BOOTSTRAP} jobs"

    for seed_i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$seed_i]}"
        MASTER_PORT=$((BASE_PORT + bootstrap_i * ${#SEEDS[@]} + seed_i))
        RUN_DIR="${ROOT_DIR}/bootstrap${BOOTSTRAP}/seed_${SEED}"
        mkdir -p "${RUN_DIR}"

        CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
            "${PYTHON_BIN}" -m alf.bin.train \
            --conf "${CONF_FILE}" \
            --root_dir "${RUN_DIR}" \
            --conf_param "TrainerConfig.random_seed=${SEED}" \
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
            --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
            --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
            --conf_param "TrainerConfig.debug_summaries=${DEBUG_SUMMARIES}" \
            --conf_param "BafcAlgorithmV3.critic_utd=${CRITIC_UTD}" \
            --conf_param "bafcv3_learning_rate=${LEARNING_RATE}" \
            --conf_param "bafcv3_actor_critic_pairing=${ACTOR_CRITIC_PAIRING}" \
            --conf_param "bafcv3_num_actor_critic=${NUM_ACTOR_CRITIC}" \
            --conf_param "bafcv3_use_bootstrap_actors=${BOOTSTRAP}" \
            --conf_param "bafcv3_use_bootstrap_critics=${BOOTSTRAP}" \
            --conf_param "bafcv3_num_attention_heads=${NUM_ATTENTION_HEADS}" \
            --conf_param "create_environment.env_name='${ENV_NAME}'" \
            --distributed multi-gpu \
            > "${RUN_DIR}/out.log" 2>&1 &

        PID=$!
        PIDS+=("${PID}")
        echo "  Bootstrap ${BOOTSTRAP}, seed ${SEED}: GPUs ${GPUS}, port ${MASTER_PORT}, PID ${PID}"
        echo "    Log: ${RUN_DIR}/out.log"
    done
    echo ""
done

echo ""
echo "Launched BAFCv3 4-GPU jobs: ${PIDS[*]}"
echo "Launcher is not waiting for completion."
echo "To monitor: tail -f ${ROOT_DIR}/bootstrap*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
