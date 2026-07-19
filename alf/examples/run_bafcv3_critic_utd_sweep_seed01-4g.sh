#!/bin/bash
# Launcher script for BAFCv3 critic-UTD sweeps with seeds 0 and 1.
# Each seed uses all configured GPUs via DDP, and all jobs launch in parallel.
# Bootstrap actors and critics are disabled for every condition.
#
# Usage: bash run_bafcv3_critic_utd_sweep_seed01-4g.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS              Total environment steps (default: 480000)
#       --critic-utds CSV              Comma-separated critic UTD values (default: 3,5,10)
#       --learning-rate LR             Learning rate for both conditions (default: 3e-4)
#       --actor-critic-pairing BOOL    Fix actor-critic pairing (default: True)
#       --num-actor-critic N           Number of actor-critic pairs (default: 10)
#       --num-sampled-critics-for-actor N
#                                      Critics averaged per actor (default: 1)
#       --num-attention-heads N        Transformer attention heads (default: 4)
#       --gpus CSV                     Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N                Number of checkpoints (default: 10)
#   -h, --help                         Show this help message
#
# Examples:
#   bash run_bafcv3_critic_utd_sweep_seed01-4g.sh
#   bash run_bafcv3_critic_utd_sweep_seed01-4g.sh --critic-utds 2,3,5
#   bash run_bafcv3_critic_utd_sweep_seed01-4g.sh --actor-critic-pairing false
#   bash run_bafcv3_critic_utd_sweep_seed01-4g.sh --actor-critic-pairing false --num-sampled-critics-for-actor 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=480000
NUM_CHECKPOINTS=10
CRITIC_UTDS=(3 5 11)
LEARNING_RATE=3e-4
ACTOR_CRITIC_PAIRING=False
NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITICS_FOR_ACTOR=1
NUM_ATTENTION_HEADS=1
DEBUG_SUMMARIES=True
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
        --critic-utds)
            parse_csv_array "$2" CRITIC_UTDS
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
        --num-sampled-critics-for-actor)
            NUM_SAMPLED_CRITICS_FOR_ACTOR="$2"
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
if [[ ${#CRITIC_UTDS[@]} -eq 0 ]]; then
    echo "--critic-utds must provide at least one value." >&2
    exit 1
fi
for critic_utd in "${CRITIC_UTDS[@]}"; do
    if [[ ! "${critic_utd}" =~ ^[1-9][0-9]*$ ]]; then
        echo "--critic-utds values must be positive integers, got: ${critic_utd}" >&2
        exit 1
    fi
done
if [[ ! "${NUM_ACTOR_CRITIC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-actor-critic must be a positive integer, got: ${NUM_ACTOR_CRITIC}" >&2
    exit 1
fi
if [[ ! "${NUM_SAMPLED_CRITICS_FOR_ACTOR}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-sampled-critics-for-actor must be a positive integer, got: ${NUM_SAMPLED_CRITICS_FOR_ACTOR}" >&2
    exit 1
fi
if (( NUM_SAMPLED_CRITICS_FOR_ACTOR > NUM_ACTOR_CRITIC )); then
    echo "--num-sampled-critics-for-actor (${NUM_SAMPLED_CRITICS_FOR_ACTOR}) cannot exceed --num-actor-critic (${NUM_ACTOR_CRITIC})" >&2
    exit 1
fi
if [[ "${ACTOR_CRITIC_PAIRING}" == "True" && "${NUM_SAMPLED_CRITICS_FOR_ACTOR}" != "1" ]]; then
    echo "--num-sampled-critics-for-actor must be 1 when --actor-critic-pairing is true" >&2
    exit 1
fi
if [[ ! "${NUM_ATTENTION_HEADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-attention-heads must be a positive integer, got: ${NUM_ATTENTION_HEADS}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_4g_critic_utd_sweep/lr${LEARNING_RATE}/actor_critic_pairing${ACTOR_CRITIC_PAIRING}_num_actor_critic${NUM_ACTOR_CRITIC}_num_sampled_critics_for_actor${NUM_SAMPLED_CRITICS_FOR_ACTOR}_num_attention_heads${NUM_ATTENTION_HEADS}"

cat <<EOF
Starting BAFCv3 critic-UTD sweep
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  bootstrap actors/critics: False/False
  critic_utd values: ${CRITIC_UTDS[*]}
  learning rate: ${LEARNING_RATE}
  actor_critic_pairing: ${ACTOR_CRITIC_PAIRING}
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  num_sampled_critics_for_actor: ${NUM_SAMPLED_CRITICS_FOR_ACTOR}
  num_attention_heads: ${NUM_ATTENTION_HEADS}
  debug_summaries: ${DEBUG_SUMMARIES}
  GPUs: ${GPUS}
  Python: ${PYTHON_BIN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for critic_utd_i in "${!CRITIC_UTDS[@]}"; do
    CRITIC_UTD="${CRITIC_UTDS[$critic_utd_i]}"
    echo "Launching critic_utd=${CRITIC_UTD} jobs"

    for seed_i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$seed_i]}"
        MASTER_PORT=$((BASE_PORT + critic_utd_i * ${#SEEDS[@]} + seed_i))
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
            --conf_param "TrainerConfig.debug_summaries=${DEBUG_SUMMARIES}" \
            --conf_param "BafcAlgorithmV3.critic_utd=${CRITIC_UTD}" \
            --conf_param "bafcv3_learning_rate=${LEARNING_RATE}" \
            --conf_param "bafcv3_actor_critic_pairing=${ACTOR_CRITIC_PAIRING}" \
            --conf_param "bafcv3_num_actor_critic=${NUM_ACTOR_CRITIC}" \
            --conf_param "bafcv3_num_sampled_critics_for_actor=${NUM_SAMPLED_CRITICS_FOR_ACTOR}" \
            --conf_param "bafcv3_use_bootstrap_actors=False" \
            --conf_param "bafcv3_use_bootstrap_critics=False" \
            --conf_param "bafcv3_num_attention_heads=${NUM_ATTENTION_HEADS}" \
            --conf_param "create_environment.env_name='${ENV_NAME}'" \
            --distributed multi-gpu \
            > "${RUN_DIR}/out.log" 2>&1 &

        PID=$!
        PIDS+=("${PID}")
        echo "  critic_utd ${CRITIC_UTD}, seed ${SEED}: GPUs ${GPUS}, port ${MASTER_PORT}, PID ${PID}"
        echo "    Log: ${RUN_DIR}/out.log"
    done
    echo ""
done

echo ""
echo "Launched BAFCv3 critic-UTD 4-GPU jobs: ${PIDS[*]}"
echo "Launcher is not waiting for completion."
echo "To monitor: tail -f ${ROOT_DIR}/critic_utd*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
