#!/bin/bash
# Launch BAFCv3_TR and BAFCv6 random-target conditions on seeds 0 and 1.
# Each job uses all configured GPUs through DDP; all six jobs run in parallel.
#
# Conditions (all use pairing=False, K_actor=8, and critic_utd=11):
#   1. BAFCv3_TR, random critic TD targets enabled
#   2. BAFCv3_TR, random critic TD targets disabled
#   3. BAFCv6,    random critic TD targets enabled
# Random-target conditions use M_target=1 by default.
#
# Usage: bash run_bafcv3_tr_v6_random_target_sweep_seed01-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 600000)
#       --num-actor-critic N        Number of actor-critic pairs (default: 10)
#       --num-sampled-targets N     Target critics sampled per update (default: 1)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 10)
#       --base-port PORT            First DDP master port (default: 29500)
#       --dry-run                   Print commands without launching jobs
#   -h, --help                      Show this help message
#
# Example:
#   bash run_bafcv3_tr_v6_random_target_sweep_seed01-4g.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TR_CONF_FILE="${SCRIPT_DIR}/bafcv3_tr_dmc_conf.py"
V6_CONF_FILE="${SCRIPT_DIR}/bafcv6_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITICS_FOR_ACTOR=8
NUM_SAMPLED_CRITIC_TARGETS=1
CRITIC_UTD=11
NUM_UPDATES_PER_TRAIN_ITER=12
DEBUG_SUMMARIES=True
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(0 1)

CONDITION_NAMES=(
    "bafcv3_tr_random_target"
    "bafcv3_tr_no_random_target"
    "bafcv6_random_target"
)
CONF_FILES=(
    "${TR_CONF_FILE}"
    "${TR_CONF_FILE}"
    "${V6_CONF_FILE}"
)
ALGORITHM_CONFIGS=(
    "BafcAlgorithmV3"
    "BafcAlgorithmV3"
    "BafcAlgorithmV6"
)
CONFIG_PREFIXES=(
    "bafcv3_tr"
    "bafcv3_tr"
    "bafcv6"
)
USE_RANDOM_CRITIC_TARGETS=(True False True)

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
        --num-actor-critic)
            NUM_ACTOR_CRITIC="$2"
            shift 2
            ;;
        --num-sampled-targets)
            NUM_SAMPLED_CRITIC_TARGETS="$2"
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
for conf_file in "${TR_CONF_FILE}" "${V6_CONF_FILE}"; do
    if [[ ! -f "${conf_file}" ]]; then
        echo "Config file not found: ${conf_file}" >&2
        exit 1
    fi
done
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
if (( NUM_ACTOR_CRITIC < NUM_SAMPLED_CRITICS_FOR_ACTOR )); then
    echo "--num-actor-critic must be at least ${NUM_SAMPLED_CRITICS_FOR_ACTOR} for K_actor=${NUM_SAMPLED_CRITICS_FOR_ACTOR}." >&2
    exit 1
fi
if [[ ! "${NUM_SAMPLED_CRITIC_TARGETS}" =~ ^[1-9][0-9]*$ ]] ||
        (( NUM_SAMPLED_CRITIC_TARGETS > NUM_ACTOR_CRITIC )); then
    echo "--num-sampled-targets must be between 1 and ${NUM_ACTOR_CRITIC}, got: ${NUM_SAMPLED_CRITIC_TARGETS}" >&2
    exit 1
fi
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 5 > 65535 )); then
    echo "--base-port must leave room for six valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_v6_random_target_sweep_4g/num_actor_critic${NUM_ACTOR_CRITIC}_num_sampled_critics_for_actor${NUM_SAMPLED_CRITICS_FOR_ACTOR}_num_sampled_critic_targets${NUM_SAMPLED_CRITIC_TARGETS}/critic_utd${CRITIC_UTD}"

cat <<EOF
Starting BAFCv3_TR/BAFCv6 random-target sweep
  TR config: ${TR_CONF_FILE}
  v6 config: ${V6_CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  actor_critic_pairing: False
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  num_sampled_critics_for_actor: ${NUM_SAMPLED_CRITICS_FOR_ACTOR}
  num_sampled_critic_targets: ${NUM_SAMPLED_CRITIC_TARGETS}
  critic_utd: ${CRITIC_UTD}
  debug_summaries: ${DEBUG_SUMMARIES}
  GPUs per job: ${GPUS}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for condition_index in "${!CONDITION_NAMES[@]}"; do
    CONDITION_NAME="${CONDITION_NAMES[$condition_index]}"
    CONF_FILE="${CONF_FILES[$condition_index]}"
    ALGORITHM_CONFIG="${ALGORITHM_CONFIGS[$condition_index]}"
    CONFIG_PREFIX="${CONFIG_PREFIXES[$condition_index]}"
    RANDOM_TARGETS="${USE_RANDOM_CRITIC_TARGETS[$condition_index]}"

    echo "Condition: ${CONDITION_NAME}, use_random_critic_targets=${RANDOM_TARGETS}"
    for seed_index in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$seed_index]}"
        MASTER_PORT=$((BASE_PORT + condition_index * ${#SEEDS[@]} + seed_index))
        RUN_DIR="${ROOT_DIR}/${CONDITION_NAME}/seed_${SEED}"
        COMMAND=(
            "${PYTHON_BIN}" -m alf.bin.train
            --conf "${CONF_FILE}"
            --root_dir "${RUN_DIR}"
            --conf_param "TrainerConfig.random_seed=${SEED}"
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
            --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
            --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
            --conf_param "TrainerConfig.num_updates_per_train_iter=${NUM_UPDATES_PER_TRAIN_ITER}"
            --conf_param "TrainerConfig.debug_summaries=${DEBUG_SUMMARIES}"
            --conf_param "${ALGORITHM_CONFIG}.critic_utd=${CRITIC_UTD}"
            --conf_param "${CONFIG_PREFIX}_actor_critic_pairing=False"
            --conf_param "${CONFIG_PREFIX}_num_actor_critic=${NUM_ACTOR_CRITIC}"
            --conf_param "${CONFIG_PREFIX}_num_sampled_critics_for_actor=${NUM_SAMPLED_CRITICS_FOR_ACTOR}"
            --conf_param "${CONFIG_PREFIX}_use_random_critic_targets=${RANDOM_TARGETS}"
            --conf_param "${CONFIG_PREFIX}_num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
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
    echo ""
done

if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched six BAFCv3_TR/BAFCv6 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
