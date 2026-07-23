#!/bin/bash
# Launch BAFCv3 critic-bootstrap probability sweeps on seeds 0 and 1.
# Each job uses all configured GPUs through DDP; all six jobs run in parallel.
#
# Fixed condition:
#   actor_critic_pairing=True
#   use_bootstrap_actors=False
#   use_bootstrap_critics=True
#   bootstrap_mask_prob in {0.8, 0.7, 0.6}
#
# Usage: bash run_bafcv3_critic_bootstrap_prob_sweep_seed01-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 480000)
#       --critic-utd N              Critic update-to-data ratio (default: 3)
#       --learning-rate LR          Learning rate (default: 3e-4)
#       --num-actor-critic N        Number of actor-critic pairs (default: 10)
#       --num-attention-heads N     Transformer attention heads (default: 1)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 10)
#       --base-port PORT            First DDP master port (default: 29500)
#       --dry-run                   Print commands without launching jobs
#   -h, --help                      Show this help message
#
# Example:
#   bash run_bafcv3_critic_bootstrap_prob_sweep_seed01-4g.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
CRITIC_UTD=3
LEARNING_RATE=3e-4
NUM_ACTOR_CRITIC=10
NUM_ATTENTION_HEADS=1
DEBUG_SUMMARIES=True
BOOTSTRAP_PROBS=(0.8 0.7 0.6)
SEEDS=(0 1)
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False

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
        --critic-utd)
            CRITIC_UTD="$2"
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
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 5 > 65535 )); then
    echo "--base-port must leave room for six valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_4g_critic_bootstrap_prob_sweep/critic_utd${CRITIC_UTD}/lr${LEARNING_RATE}/actor_critic_pairingTrue_num_actor_critic${NUM_ACTOR_CRITIC}_num_attention_heads${NUM_ATTENTION_HEADS}"

cat <<EOF
Starting BAFCv3 critic-bootstrap probability sweep
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  bootstrap_mask_prob values: ${BOOTSTRAP_PROBS[*]}
  critic_utd: ${CRITIC_UTD}
  learning rate: ${LEARNING_RATE}
  actor_critic_pairing: True
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  num_sampled_critics_for_actor: 1
  use_bootstrap_actors: False
  use_bootstrap_critics: True
  use_random_critic_targets: False
  num_attention_heads: ${NUM_ATTENTION_HEADS}
  debug_summaries: ${DEBUG_SUMMARIES}
  GPUs per job: ${GPUS}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for prob_index in "${!BOOTSTRAP_PROBS[@]}"; do
    BOOTSTRAP_PROB="${BOOTSTRAP_PROBS[$prob_index]}"
    echo "Condition: critic bootstrap probability=${BOOTSTRAP_PROB}"

    for seed_index in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$seed_index]}"
        MASTER_PORT=$((BASE_PORT + prob_index * ${#SEEDS[@]} + seed_index))
        RUN_DIR="${ROOT_DIR}/bootstrap_mask_prob${BOOTSTRAP_PROB}/seed_${SEED}"
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
            --conf_param "BafcAlgorithmV3.bootstrap_mask_prob=${BOOTSTRAP_PROB}"
            --conf_param "bafcv3_learning_rate=${LEARNING_RATE}"
            --conf_param "bafcv3_actor_critic_pairing=True"
            --conf_param "bafcv3_num_actor_critic=${NUM_ACTOR_CRITIC}"
            --conf_param "bafcv3_num_sampled_critics_for_actor=1"
            --conf_param "bafcv3_use_random_critic_targets=False"
            --conf_param "bafcv3_use_bootstrap_actors=False"
            --conf_param "bafcv3_use_bootstrap_critics=True"
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
            echo "  probability ${BOOTSTRAP_PROB}, seed ${SEED}: port ${MASTER_PORT}, PID ${PID}"
            echo "    Log: ${RUN_DIR}/out.log"
        fi
    done
    echo ""
done

if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched six BAFCv3 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/bootstrap_mask_prob*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
