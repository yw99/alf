#!/bin/bash
# Launch two BAFCv3 replay-evaluation-sample jobs (seeds 0 and 1). Each job
# uses all configured GPUs through DistributedDataParallel.
#
# Both jobs use:
#   - replay evaluation samples from the transformed iteration batch
#   - no fixed actor-critic pairing
#   - eight sampled critics per actor
#   - one random shared critic TD target
#   - actor UTD 1 and critic UTD 11
#
# Usage: bash run_bafcv3_replay_eval_samples_seed01-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR              Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 600000)
#       --learning-rate LR          BAFCv3 learning rate (default: 3e-4)
#       --num-actor-critic N        BAFCv3 actor-critic pairs (default: 10)
#       --num-sampled-critics N     Critics sampled per actor (default: 8)
#       --num-sampled-targets N     Target critics per update (default: 1)
#       --num-eval-samples N        Actor evaluation samples (default: 512)
#       --critic-utd N              Critic UTD (default: 11)
#       --num-attention-heads N     Transformer attention heads (default: 1)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 10)
#       --base-port PORT            First DDP master port (default: 29620)
#       --http-base-port PORT       First ALF HTTP-control port (default: 18120)
#       --dry-run                   Print commands without launching jobs
#   -h, --help                      Show this help message
#
# Example:
#   bash run_bafcv3_replay_eval_samples_seed01-4g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
DEBUG_SUMMARIES=True

BAFCV3_LEARNING_RATE=3e-4
BAFCV3_ACTOR_USE_LN=False
BAFCV3_ACTOR_CRITIC_PAIRING=False
BAFCV3_NUM_ACTOR_CRITIC=10
BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR=8
BAFCV3_USE_RANDOM_CRITIC_TARGETS=True
BAFCV3_NUM_SAMPLED_CRITIC_TARGETS=1
BAFCV3_NUM_ACTOR_EVAL_SAMPLES=512
BAFCV3_NUM_ATTENTION_HEADS=1
BAFCV3_USE_BOOTSTRAP_ACTORS=False
BAFCV3_USE_BOOTSTRAP_CRITICS=False
BAFCV3_CRITIC_UTD=11
BAFCV3_NUM_UPDATES_PER_TRAIN_ITER=12

GPUS="0,1,2,3"
DDP_BASE_PORT=29620
HTTP_BASE_PORT=18120
DRY_RUN=False
SEEDS=(0 1)

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
            BAFCV3_LEARNING_RATE="$2"
            shift 2
            ;;
        --num-actor-critic)
            BAFCV3_NUM_ACTOR_CRITIC="$2"
            shift 2
            ;;
        --num-sampled-critics)
            BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR="$2"
            shift 2
            ;;
        --num-sampled-targets)
            BAFCV3_NUM_SAMPLED_CRITIC_TARGETS="$2"
            shift 2
            ;;
        --num-eval-samples)
            BAFCV3_NUM_ACTOR_EVAL_SAMPLES="$2"
            shift 2
            ;;
        --critic-utd)
            BAFCV3_CRITIC_UTD="$2"
            BAFCV3_NUM_UPDATES_PER_TRAIN_ITER=$((BAFCV3_CRITIC_UTD + 1))
            shift 2
            ;;
        --num-attention-heads)
            BAFCV3_NUM_ATTENTION_HEADS="$2"
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
            DDP_BASE_PORT="$2"
            shift 2
            ;;
        --http-base-port)
            HTTP_BASE_PORT="$2"
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

# DDP workers compile/load ALF's parallel-environment extension. Ensure the
# virtualenv's Ninja executable remains discoverable even from a detached shell.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

for value_name in NUM_ENV_STEPS NUM_CHECKPOINTS \
        BAFCV3_NUM_ACTOR_CRITIC \
        BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR \
        BAFCV3_NUM_SAMPLED_CRITIC_TARGETS \
        BAFCV3_NUM_ACTOR_EVAL_SAMPLES \
        BAFCV3_NUM_ATTENTION_HEADS \
        BAFCV3_CRITIC_UTD \
        BAFCV3_NUM_UPDATES_PER_TRAIN_ITER; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 1
    fi
done
if (( BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR > BAFCV3_NUM_ACTOR_CRITIC )); then
    echo "--num-sampled-critics must be at most ${BAFCV3_NUM_ACTOR_CRITIC}." >&2
    exit 1
fi
if (( BAFCV3_NUM_SAMPLED_CRITIC_TARGETS > BAFCV3_NUM_ACTOR_CRITIC )); then
    echo "--num-sampled-targets must be at most ${BAFCV3_NUM_ACTOR_CRITIC}." >&2
    exit 1
fi

MAX_PORT_OFFSET=$((${#SEEDS[@]} - 1))
for port_name in DDP_BASE_PORT HTTP_BASE_PORT; do
    port="${!port_name}"
    if [[ ! "${port}" =~ ^[1-9][0-9]*$ ]] || \
            (( port + MAX_PORT_OFFSET > 65535 )); then
        echo "${port_name} must leave room for two valid ports, got: ${port}" >&2
        exit 1
    fi
done
if (( DDP_BASE_PORT <= HTTP_BASE_PORT + MAX_PORT_OFFSET && \
        HTTP_BASE_PORT <= DDP_BASE_PORT + MAX_PORT_OFFSET )); then
    echo "DDP and HTTP port ranges must not overlap." >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR%/}/${ENV_DIR}/bafcv3_replay_pool_reuse_seed01_4g/num_actor_critic${BAFCV3_NUM_ACTOR_CRITIC}_num_sampled_critics_for_actor${BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR}_num_sampled_critic_targets${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}/critic_utd${BAFCV3_CRITIC_UTD}"

cat <<EOF
Starting two BAFCv3 replay-evaluation-sample jobs
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Seeds: ${SEEDS[*]}
  Num env steps: ${NUM_ENV_STEPS}
  Replay evaluation samples: ${BAFCV3_NUM_ACTOR_EVAL_SAMPLES}
  Actor-critic pairs / sampled critics: ${BAFCV3_NUM_ACTOR_CRITIC}/${BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR}
  Random target critics: ${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}
  Actor/critic UTD: 1/${BAFCV3_CRITIC_UTD}
  GPUs per job: ${GPUS}
  DDP ports: ${DDP_BASE_PORT}-$((DDP_BASE_PORT + MAX_PORT_OFFSET))
  HTTP-control ports: ${HTTP_BASE_PORT}-$((HTTP_BASE_PORT + MAX_PORT_OFFSET))
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for seed_index in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$seed_index]}"
    MASTER_PORT=$((DDP_BASE_PORT + seed_index))
    HTTP_PORT=$((HTTP_BASE_PORT + seed_index))
    RUN_DIR="${ROOT_DIR}/bafcv3_replay_eval_samples_random_target/seed_${SEED}"
    COMMAND=(
        "${PYTHON_BIN}" -m alf.bin.train
        --port "${HTTP_PORT}"
        --conf "${CONF_FILE}"
        --root_dir "${RUN_DIR}"
        --conf_param "TrainerConfig.random_seed=${SEED}"
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
        --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
        --conf_param "TrainerConfig.debug_summaries=${DEBUG_SUMMARIES}"
        --conf_param "create_environment.env_name='${ENV_NAME}'"
        --conf_param "TrainerConfig.num_updates_per_train_iter=${BAFCV3_NUM_UPDATES_PER_TRAIN_ITER}"
        --conf_param "BafcAlgorithmV3.critic_utd=${BAFCV3_CRITIC_UTD}"
        --conf_param "BafcAlgorithmV3.num_actor_eval_samples=${BAFCV3_NUM_ACTOR_EVAL_SAMPLES}"
        --conf_param "bafcv3_learning_rate=${BAFCV3_LEARNING_RATE}"
        --conf_param "bafcv3_actor_use_ln=${BAFCV3_ACTOR_USE_LN}"
        --conf_param "bafcv3_actor_critic_pairing=${BAFCV3_ACTOR_CRITIC_PAIRING}"
        --conf_param "bafcv3_num_actor_critic=${BAFCV3_NUM_ACTOR_CRITIC}"
        --conf_param "bafcv3_num_sampled_critics_for_actor=${BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR}"
        --conf_param "bafcv3_use_random_critic_targets=${BAFCV3_USE_RANDOM_CRITIC_TARGETS}"
        --conf_param "bafcv3_num_sampled_critic_targets=${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}"
        --conf_param "bafcv3_eval_samples_source='replay'"
        --conf_param "bafcv3_use_bootstrap_actors=${BAFCV3_USE_BOOTSTRAP_ACTORS}"
        --conf_param "bafcv3_use_bootstrap_critics=${BAFCV3_USE_BOOTSTRAP_CRITICS}"
        --conf_param "bafcv3_num_attention_heads=${BAFCV3_NUM_ATTENTION_HEADS}"
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
        echo "  seed ${SEED}: DDP port ${MASTER_PORT}, HTTP port ${HTTP_PORT}, PID ${PID}"
        echo "    Log: ${RUN_DIR}/out.log"
    fi
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched ${#PIDS[@]} replay 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/bafcv3_replay_eval_samples_random_target/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
