#!/bin/bash
# Launch a BAFCv3 evaluation-sample comparison on seeds 0 and 1, optionally
# including default RLPD. Each job uses all configured GPUs through DDP; four
# jobs run by default, or six when --with-rlpd is specified.
#
# Conditions:
#   1. BAFCv3 with replay evaluation samples, no fixed actor-critic pairing,
#      and an RLPD-style random critic TD target
#   2. BAFCv3 with trainable evaluation samples, no fixed actor-critic pairing,
#      and an RLPD-style random critic TD target
#   3. Default RLPD (optional; enable with --with-rlpd)
#
# Usage: bash run_bafcv3_eval_samples_sweep_seed01-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR              Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 600000)
#       --learning-rate LR          BAFCv3 learning rate (default: 3e-4)
#       --num-actor-critic N        BAFCv3 actor-critic pairs (default: 10)
#       --num-sampled-critics N     Critics sampled per BAFCv3 actor (default: 8)
#       --num-sampled-targets N     BAFCv3 target critics per update (default: 1)
#       --num-eval-samples N        BAFCv3 actor evaluation samples (default: 512)
#       --critic-utd N              BAFCv3 critic UTD (default: 11)
#       --num-attention-heads N     BAFCv3 attention heads (default: 1)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 10)
#       --base-port PORT            First DDP master port (default: 29500)
#       --http-base-port PORT       First ALF HTTP-control port (default: 18080)
#       --with-rlpd                 Also launch the default RLPD comparison
#       --dry-run                   Print commands without launching jobs
#   -h, --help                      Show this help message
#
# Example:
#   bash run_bafcv3_eval_samples_sweep_seed01-4g.sh --dry-run

set -euo pipefail

# Use headless EGL rendering for dm_control unless explicitly overridden.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Use deterministic CUBLAS workspace for reproducibility unless explicitly overridden.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAFCV3_CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
RLPD_CONF_FILE="${SCRIPT_DIR}/rlpd_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
DEBUG_SUMMARIES=True

# BAFCv3 comparison macros. Both BAFCv3 conditions differ only in their
# evaluation-sample source.
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
DDP_BASE_PORT=29500
HTTP_BASE_PORT=18080
DRY_RUN=False
RUN_RLPD=False
SEEDS=(0 1)

CONDITION_NAMES=(
    "bafcv3_replay_eval_samples_random_target"
    "bafcv3_trainable_eval_samples_random_target"
)
CONDITION_TYPES=(
    "bafcv3"
    "bafcv3"
)
EVAL_SAMPLES_SOURCES=(
    "replay"
    "trainable"
)

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
        --with-rlpd)
            RUN_RLPD=True
            shift
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
conf_files=("${BAFCV3_CONF_FILE}")
if [[ "${RUN_RLPD}" == "True" ]]; then
    conf_files+=("${RLPD_CONF_FILE}")
fi
for conf_file in "${conf_files[@]}"; do
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
for value_name in BAFCV3_NUM_ACTOR_CRITIC \
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
    echo "--num-sampled-critics must be at most ${BAFCV3_NUM_ACTOR_CRITIC}, got: ${BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR}" >&2
    exit 1
fi
if (( BAFCV3_NUM_SAMPLED_CRITIC_TARGETS > BAFCV3_NUM_ACTOR_CRITIC )); then
    echo "--num-sampled-targets must be at most ${BAFCV3_NUM_ACTOR_CRITIC}, got: ${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}" >&2
    exit 1
fi
if (( BAFCV3_NUM_UPDATES_PER_TRAIN_ITER != BAFCV3_CRITIC_UTD + 1 )); then
    echo "BAFCV3_NUM_UPDATES_PER_TRAIN_ITER must equal BAFCV3_CRITIC_UTD + actor_utd (1)." >&2
    exit 1
fi
if [[ "${RUN_RLPD}" == "True" ]]; then
    CONDITION_NAMES+=("rlpd_default")
    CONDITION_TYPES+=("rlpd")
    EVAL_SAMPLES_SOURCES+=("default")
fi
NUM_JOBS=$((${#CONDITION_NAMES[@]} * ${#SEEDS[@]}))
MAX_PORT_OFFSET=$((NUM_JOBS - 1))

for port_name in DDP_BASE_PORT HTTP_BASE_PORT; do
    port="${!port_name}"
    if [[ ! "${port}" =~ ^[1-9][0-9]*$ ]] || (( port + MAX_PORT_OFFSET > 65535 )); then
        echo "${port_name} must leave room for ${NUM_JOBS} valid ports, got: ${port}" >&2
        exit 1
    fi
done
if (( DDP_BASE_PORT <= HTTP_BASE_PORT + MAX_PORT_OFFSET && HTTP_BASE_PORT <= DDP_BASE_PORT + MAX_PORT_OFFSET )); then
    echo "DDP and HTTP port ranges must not overlap." >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR%/}/${ENV_DIR}/bafcv3_eval_samples_comparison_v2_4g/num_actor_critic${BAFCV3_NUM_ACTOR_CRITIC}_num_sampled_critics_for_actor${BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR}_num_sampled_critic_targets${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}/critic_utd${BAFCV3_CRITIC_UTD}"

cat <<EOF
Starting BAFCv3 evaluation-sample comparison
  BAFCv3 config: ${BAFCV3_CONF_FILE}
  RLPD config (when enabled): ${RLPD_CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  BAFCv3 learning rate: ${BAFCV3_LEARNING_RATE}
  BAFCv3 actor layer norm: ${BAFCV3_ACTOR_USE_LN}
  BAFCv3 actor_critic_pairing: ${BAFCV3_ACTOR_CRITIC_PAIRING}
  BAFCv3 num_actor_critic: ${BAFCV3_NUM_ACTOR_CRITIC}
  BAFCv3 num_sampled_critics_for_actor: ${BAFCV3_NUM_SAMPLED_CRITICS_FOR_ACTOR}
  BAFCv3 use_random_critic_targets: ${BAFCV3_USE_RANDOM_CRITIC_TARGETS}
  BAFCv3 num_sampled_critic_targets: ${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}
  BAFCv3 num_actor_eval_samples: ${BAFCV3_NUM_ACTOR_EVAL_SAMPLES}
  BAFCv3 critic_utd: ${BAFCV3_CRITIC_UTD}
  BAFCv3 num_updates_per_train_iter: ${BAFCV3_NUM_UPDATES_PER_TRAIN_ITER}
  BAFCv3 attention heads: ${BAFCV3_NUM_ATTENTION_HEADS}
  BAFCv3 bootstrap actors/critics: ${BAFCV3_USE_BOOTSTRAP_ACTORS}/${BAFCV3_USE_BOOTSTRAP_CRITICS}
  Include default RLPD: ${RUN_RLPD}
  debug_summaries: ${DEBUG_SUMMARIES}
  GPUs per job: ${GPUS}
  DDP ports: ${DDP_BASE_PORT}-$((DDP_BASE_PORT + MAX_PORT_OFFSET))
  HTTP-control ports: ${HTTP_BASE_PORT}-$((HTTP_BASE_PORT + MAX_PORT_OFFSET))
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for condition_index in "${!CONDITION_NAMES[@]}"; do
    CONDITION_NAME="${CONDITION_NAMES[$condition_index]}"
    CONDITION_TYPE="${CONDITION_TYPES[$condition_index]}"
    EVAL_SAMPLES_SOURCE="${EVAL_SAMPLES_SOURCES[$condition_index]}"

    if [[ "${CONDITION_TYPE}" == "bafcv3" ]]; then
        CONF_FILE="${BAFCV3_CONF_FILE}"
        echo "Condition: ${CONDITION_NAME}, eval_samples_source=${EVAL_SAMPLES_SOURCE}"
    else
        CONF_FILE="${RLPD_CONF_FILE}"
        echo "Condition: ${CONDITION_NAME}, algorithm settings=config defaults"
    fi

    for seed_index in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$seed_index]}"
        JOB_INDEX=$((condition_index * ${#SEEDS[@]} + seed_index))
        MASTER_PORT=$((DDP_BASE_PORT + JOB_INDEX))
        HTTP_PORT=$((HTTP_BASE_PORT + JOB_INDEX))
        RUN_DIR="${ROOT_DIR}/${CONDITION_NAME}/seed_${SEED}"
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
        )

        if [[ "${CONDITION_TYPE}" == "bafcv3" ]]; then
            COMMAND+=(
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
                --conf_param "bafcv3_eval_samples_source='${EVAL_SAMPLES_SOURCE}'"
                --conf_param "bafcv3_use_bootstrap_actors=${BAFCV3_USE_BOOTSTRAP_ACTORS}"
                --conf_param "bafcv3_use_bootstrap_critics=${BAFCV3_USE_BOOTSTRAP_CRITICS}"
                --conf_param "bafcv3_num_attention_heads=${BAFCV3_NUM_ATTENTION_HEADS}"
            )
        else
            # RLPD keeps critic UTD, update count, ensemble, entropy, and target
            # update settings from rlpd_dmc_conf.py.
            COMMAND+=(
                --conf_param "make_ddp_performer.find_unused_parameters=True"
            )
        fi
        COMMAND+=(--distributed multi-gpu)

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
done

if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched ${#PIDS[@]} comparison 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
