#!/bin/bash
# Launch humanoid:run and dog:run BAFCv3 control A (actor-ID encoding) on
# seeds 0, 1, 2, and 3. Each job uses four GPUs through DDP: humanoid:run
# uses GPUs 0-3 and dog:run uses GPUs 4-7. All eight jobs run concurrently
# on unique torch.distributed master ports.
# Each launch creates a fresh UTC-timestamped results directory.
#
# This preserves the BAFCv3 settings from
# run_humanoid_run_bafcv3_detach_policy_seed0123-4g.sh: pairing is off,
# eight critics are sampled for each actor update, critic UTD is 11, and TD
# targets use one randomly selected critic.
#
# Usage: bash run_humanoid_dog_run_bafcv3_actor_id_seed0123-8g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Override environment steps for both tasks
#                            Defaults: humanoid:run 600000, dog:run 800000
#       --gpus CSV           Eight distinct GPU ids (default: 0,1,2,3,4,5,6,7)
#                            First four for humanoid:run, last four for dog:run
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First of eight DDP master ports (default: 29800)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_humanoid_dog_run_bafcv3_actor_id_seed0123-8g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

TASKS=(humanoid:run dog:run)
BASE_DIR="/workspace/alf_results"
TASK_ENV_STEPS=(600000 800000)
NUM_CHECKPOINTS=10
GPUS="0,1,2,3,4,5,6,7"
BASE_PORT=29800
DRY_RUN=False
SEEDS=(0 1 2 3)

CRITIC_UTD=11
NUM_UPDATES_PER_TRAIN_ITER=12
NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITICS_FOR_ACTOR=8
NUM_SAMPLED_CRITIC_TARGETS=1
ACTOR_USE_LN=False
DEBUG_SUMMARIES=True

print_help() {
    sed -n '/^# Usage:/,/^# Example:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -n|--steps)
            TASK_ENV_STEPS=("$2" "$2")
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
for NUM_ENV_STEPS in "${TASK_ENV_STEPS[@]}"; do
    if [[ ! "${NUM_ENV_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "--steps must be a positive integer, got: ${NUM_ENV_STEPS}" >&2
        exit 1
    fi
done
if [[ ! "${NUM_CHECKPOINTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--checkpoints must be a positive integer, got: ${NUM_CHECKPOINTS}" >&2
    exit 1
fi
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT > 65528 )); then
    echo "--base-port must leave room for eight valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

# Split the host's eight GPUs into disjoint groups of four, one per task.
if [[ ! "${GPUS}" =~ ^[[:alnum:]_-]+(,[[:alnum:]_-]+){7}$ ]]; then
    echo "--gpus must contain exactly eight distinct GPU ids, got: ${GPUS}" >&2
    exit 1
fi
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
for i in "${!GPU_IDS[@]}"; do
    for ((j = 0; j < i; j++)); do
        if [[ "${GPU_IDS[$i]}" == "${GPU_IDS[$j]}" ]]; then
            echo "--gpus must contain distinct GPU ids, got: ${GPUS}" >&2
            exit 1
        fi
    done
done
TASK_GPUS=(
    "${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]},${GPU_IDS[3]}"
    "${GPU_IDS[4]},${GPU_IDS[5]},${GPU_IDS[6]},${GPU_IDS[7]}"
)


RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_PARENT="${BASE_DIR}/humanoid_dog_run/bafcv3_actor_id_seed0123_8g"
ROOT_DIR="${RESULTS_PARENT}/${RUN_ID}"

if [[ -e "${ROOT_DIR}" || -L "${ROOT_DIR}" ]]; then
    echo "Refusing to reuse existing results directory: ${ROOT_DIR}" >&2
    exit 1
fi
if [[ "${DRY_RUN}" != "True" ]]; then
    mkdir -p "${RESULTS_PARENT}"
    # Atomically reserve the destination before any job can open its log.
    if ! mkdir "${ROOT_DIR}"; then
        echo "Could not create fresh results directory: ${ROOT_DIR}" >&2
        exit 1
    fi
fi

cat <<EOF
Starting humanoid:run and dog:run BAFCv3 control A seeds 0, 1, 2, and 3
  Tasks: ${TASKS[*]}
  Root dir: ${ROOT_DIR}
  humanoid:run env steps: ${TASK_ENV_STEPS[0]}
  dog:run env steps: ${TASK_ENV_STEPS[1]}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  humanoid:run GPUs per job: ${TASK_GPUS[0]}
  dog:run GPUs per job: ${TASK_GPUS[1]}
  Threads: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}
  Critic UTD: ${CRITIC_UTD}
  Num updates per train iter: ${NUM_UPDATES_PER_TRAIN_ITER}
  Actor-critic pairing: False
  Num actor-critic pairs: ${NUM_ACTOR_CRITIC}
  Num sampled critics for actor: ${NUM_SAMPLED_CRITICS_FOR_ACTOR}
  Random critic targets: True
  Num sampled critic targets: ${NUM_SAMPLED_CRITIC_TARGETS}
  Actor-ID encoding (control A): True
  Detach actor policy input (control B): False
  Actor layer norm: ${ACTOR_USE_LN}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for TASK_INDEX in "${!TASKS[@]}"; do
    ENV_NAME="${TASKS[$TASK_INDEX]}"
    JOB_GPUS="${TASK_GPUS[$TASK_INDEX]}"
    NUM_ENV_STEPS="${TASK_ENV_STEPS[$TASK_INDEX]}"
    for i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$i]}"
        MASTER_PORT=$((BASE_PORT + TASK_INDEX * ${#SEEDS[@]} + i))
        RUN_DIR="${ROOT_DIR}/${ENV_NAME/:/_}/fixed_pairingFalse_num_sampled_critic${NUM_SAMPLED_CRITICS_FOR_ACTOR}/critic_utd${CRITIC_UTD}/seed_${SEED}"
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
            --conf_param "BafcAlgorithmV3.critic_utd=${CRITIC_UTD}"
            --conf_param "bafcv3_use_actor_id_encoding=True"
            --conf_param "bafcv3_detach_actor_policy_input=False"
            --conf_param "bafcv3_actor_use_ln=${ACTOR_USE_LN}"
            --conf_param "bafcv3_actor_critic_pairing=False"
            --conf_param "bafcv3_num_actor_critic=${NUM_ACTOR_CRITIC}"
            --conf_param "bafcv3_num_sampled_critics_for_actor=${NUM_SAMPLED_CRITICS_FOR_ACTOR}"
            --conf_param "bafcv3_use_random_critic_targets=True"
            --conf_param "bafcv3_num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
            --conf_param "make_ddp_performer.find_unused_parameters=True"
            --conf_param "create_environment.env_name='${ENV_NAME}'"
            --distributed multi-gpu
        )

        if [[ "${DRY_RUN}" == "True" ]]; then
            printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${JOB_GPUS}" "${MASTER_PORT}"
            printf '%q ' "${COMMAND[@]}"
            printf '> %q 2>&1 &\n' "${RUN_DIR}/out.log"
            continue
        fi

        mkdir -p "${RUN_DIR}"
        CUDA_VISIBLE_DEVICES="${JOB_GPUS}" MASTER_PORT="${MASTER_PORT}" \
            "${COMMAND[@]}" > "${RUN_DIR}/out.log" 2>&1 &
        PID=$!
        PIDS+=("${PID}")
        echo "  ${ENV_NAME} seed ${SEED}: GPUs ${JOB_GPUS}, port ${MASTER_PORT}, PID ${PID}"
        echo "    Log: ${RUN_DIR}/out.log"
    done
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched eight humanoid:run/dog:run BAFCv3 control A jobs (four GPUs per job): ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/*/fixed_pairingFalse_num_sampled_critic${NUM_SAMPLED_CRITICS_FOR_ACTOR}/critic_utd${CRITIC_UTD}/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
