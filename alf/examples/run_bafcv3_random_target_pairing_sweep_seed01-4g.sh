#!/bin/bash
# Launch BAFCv3 random-target pairing conditions on seeds 0 and 1.
# Each job uses all configured GPUs through DDP; all six jobs run in parallel.
#
# Conditions:
#   1. pairing=False, K_actor=8,  critic_utd=11
#   2. pairing=False, K_actor=10, critic_utd=11
#   3. pairing=True,  K_actor=1,  critic_utd=3
# All conditions use an RLPD-style random critic TD target with M_target=1.
#
# Usage: bash run_bafcv3_random_target_pairing_sweep_seed01-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 480000)
#       --learning-rate LR          Learning rate (default: 3e-4)
#       --num-actor-critic N        Number of actor-critic pairs (default: 10)
#       --num-sampled-targets N     Target critics sampled per update (default: 1)
#       --num-attention-heads N     Transformer attention heads (default: 1)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 10)
#       --base-port PORT            First DDP master port (default: 29500)
#       --dry-run                   Print commands without launching jobs
#   -h, --help                      Show this help message
#
# Example:
#   bash run_bafcv3_random_target_pairing_sweep_seed01-4g.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
LEARNING_RATE=3e-4
NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITIC_TARGETS=1
NUM_ATTENTION_HEADS=1
DEBUG_SUMMARIES=True
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(0 1)

PAIRINGS=(False False True)
NUM_SAMPLED_CRITICS_FOR_ACTOR=(8 10 1)
CRITIC_UTDS=(11 11 3)

print_help() {
    sed -n '/^# Usage:/,/^# Example:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
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
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num-actor-critic)
            NUM_ACTOR_CRITIC="$2"
            shift 2
            ;;
        --num-sampled-targets)
            NUM_SAMPLED_CRITIC_TARGETS="$2"
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
        --base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=True
            shift
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
if [[ ! "${NUM_ENV_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--steps must be a positive integer, got: ${NUM_ENV_STEPS}" >&2
    exit 1
fi
if [[ ! "${NUM_CHECKPOINTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--checkpoints must be a positive integer, got: ${NUM_CHECKPOINTS}" >&2
    exit 1
fi
if [[ ! "${NUM_ACTOR_CRITIC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-actor-critic must be a positive integer, got: ${NUM_ACTOR_CRITIC}" >&2
    exit 1
fi
if (( NUM_ACTOR_CRITIC < 10 )); then
    echo "--num-actor-critic must be at least 10 for the K_actor=10 condition." >&2
    exit 1
fi
if [[ ! "${NUM_SAMPLED_CRITIC_TARGETS}" =~ ^[1-9][0-9]*$ ]] ||
        (( NUM_SAMPLED_CRITIC_TARGETS > NUM_ACTOR_CRITIC )); then
    echo "--num-sampled-targets must be between 1 and ${NUM_ACTOR_CRITIC}, got: ${NUM_SAMPLED_CRITIC_TARGETS}" >&2
    exit 1
fi
if [[ ! "${NUM_ATTENTION_HEADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-attention-heads must be a positive integer, got: ${NUM_ATTENTION_HEADS}" >&2
    exit 1
fi
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 5 > 65535 )); then
    echo "--base-port must leave room for six valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_4g_random_target_pairing_sweep/lr${LEARNING_RATE}/num_actor_critic${NUM_ACTOR_CRITIC}_num_sampled_critic_targets${NUM_SAMPLED_CRITIC_TARGETS}_num_attention_heads${NUM_ATTENTION_HEADS}"

cat <<EOF
Starting BAFCv3 random-target pairing sweep
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  learning rate: ${LEARNING_RATE}
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  use_random_critic_targets: True
  num_sampled_critic_targets: ${NUM_SAMPLED_CRITIC_TARGETS}
  num_attention_heads: ${NUM_ATTENTION_HEADS}
  bootstrap actors/critics: False/False
  debug_summaries: ${DEBUG_SUMMARIES}
  GPUs per job: ${GPUS}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
condition_index=0
for pairing_index in "${!PAIRINGS[@]}"; do
    PAIRING="${PAIRINGS[$pairing_index]}"
    ACTOR_K="${NUM_SAMPLED_CRITICS_FOR_ACTOR[$pairing_index]}"
    CRITIC_UTD="${CRITIC_UTDS[$pairing_index]}"
    CONDITION="pairing${PAIRING}_num_sampled_critics_for_actor${ACTOR_K}/critic_utd${CRITIC_UTD}"

    echo "Condition: pairing=${PAIRING}, K_actor=${ACTOR_K}, critic_utd=${CRITIC_UTD}"
    for seed_index in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$seed_index]}"
        MASTER_PORT=$((BASE_PORT + condition_index * ${#SEEDS[@]} + seed_index))
        RUN_DIR="${ROOT_DIR}/${CONDITION}/seed_${SEED}"
        COMMAND=(
            "${PYTHON_BIN}" -m alf.bin.train
            --conf "${CONF_FILE}"
            --root_dir "${RUN_DIR}"
            --conf_param "TrainerConfig.random_seed=${SEED}"
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
            --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
            --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
            --conf_param "TrainerConfig.debug_summaries=${DEBUG_SUMMARIES}"
            --conf_param "BafcAlgorithmV3.critic_utd=${CRITIC_UTD}"
            --conf_param "bafcv3_learning_rate=${LEARNING_RATE}"
            --conf_param "bafcv3_actor_critic_pairing=${PAIRING}"
            --conf_param "bafcv3_num_actor_critic=${NUM_ACTOR_CRITIC}"
            --conf_param "bafcv3_num_sampled_critics_for_actor=${ACTOR_K}"
            --conf_param "bafcv3_use_random_critic_targets=True"
            --conf_param "bafcv3_num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
            --conf_param "bafcv3_use_bootstrap_actors=False"
            --conf_param "bafcv3_use_bootstrap_critics=False"
            --conf_param "bafcv3_num_attention_heads=${NUM_ATTENTION_HEADS}"
            --conf_param "create_environment.env_name='${ENV_NAME}'"
            --distributed multi-gpu
        )

        if [[ "${DRY_RUN}" == "True" ]]; then
            printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${MASTER_PORT}"
            printf '%q ' "${COMMAND[@]}"
            printf '> %q 2>&1 &\n' "${RUN_DIR}/out.log"
        else
            mkdir -p "${RUN_DIR}"
            CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
                "${COMMAND[@]}" > "${RUN_DIR}/out.log" 2>&1 &
            PID=$!
            PIDS+=("${PID}")
            echo "  seed ${SEED}: port ${MASTER_PORT}, PID ${PID}"
            echo "    Log: ${RUN_DIR}/out.log"
        fi
    done
    ((condition_index += 1))
    echo ""
done

if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched six BAFCv3 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/pairing*/critic_utd*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
