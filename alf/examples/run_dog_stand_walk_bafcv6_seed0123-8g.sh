#!/bin/bash
# Launch dog:stand and dog:walk BAFCv6 jobs on seeds 0, 1, 2, and 3. Every job
# uses four GPUs through DDP: dog:stand uses GPUs 0-3 and dog:walk uses GPUs
# 4-7 by default. The eight jobs run in parallel on unique DDP master ports.
#
# Uses the BAFCv6 random-target and critic-reweighting settings from
# run_humanoid_stand_bafcv6_seed0123-4g.sh, with the 800000-step budget
# from run_dog_stand_walk_td3_seed0123-8g.sh.
#
# Usage: bash run_dog_stand_walk_bafcv6_seed0123-8g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 800000)
#       --num-actor-critic N Number of actor-critic pairs (default: 10)
#       --num-sampled-targets N Target critics sampled per update (default: 1)
#       --gpus CSV           Eight distinct GPU ids (default: 0,1,2,3,4,5,6,7)
#                            First four for dog:stand, last four for dog:walk
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29500)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_dog_stand_walk_bafcv6_seed0123-8g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv6_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

TASKS=(dog:stand dog:walk)
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3,4,5,6,7"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(0 1 2 3)

NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITICS_FOR_ACTOR=8
NUM_SAMPLED_CRITIC_TARGETS=1
CRITIC_UTD=11
NUM_UPDATES_PER_TRAIN_ITER=12
DEBUG_SUMMARIES=True
ENABLE_CRITIC_REWEIGHTING=True
CRITIC_REWEIGHTING_SOLVER="lbfgs_logits"
CRITIC_REWEIGHTING_SOLVER_ITERS=1
CRITIC_REWEIGHTING_NUM_FEATURE_COORDS=32
CRITIC_REWEIGHTING_NUM_TARGET_OBS=128
CRITIC_REWEIGHTING_MAX_WEIGHT=10.0

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
if [[ ! "${NUM_ACTOR_CRITIC}" =~ ^[1-9][0-9]*$ ]] ||
        (( NUM_ACTOR_CRITIC < NUM_SAMPLED_CRITICS_FOR_ACTOR )); then
    echo "--num-actor-critic must be at least ${NUM_SAMPLED_CRITICS_FOR_ACTOR}, got: ${NUM_ACTOR_CRITIC}" >&2
    exit 1
fi
if [[ ! "${NUM_SAMPLED_CRITIC_TARGETS}" =~ ^[1-9][0-9]*$ ]] ||
        (( NUM_SAMPLED_CRITIC_TARGETS > NUM_ACTOR_CRITIC )); then
    echo "--num-sampled-targets must be between 1 and ${NUM_ACTOR_CRITIC}, got: ${NUM_SAMPLED_CRITIC_TARGETS}" >&2
    exit 1
fi
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + ${#TASKS[@]} * ${#SEEDS[@]} - 1 > 65535 )); then
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

EXPERIMENT="bafcv6_seed0123_8g/num_actor_critic${NUM_ACTOR_CRITIC}_num_sampled_critics_for_actor${NUM_SAMPLED_CRITICS_FOR_ACTOR}_num_sampled_critic_targets${NUM_SAMPLED_CRITIC_TARGETS}/critic_utd${CRITIC_UTD}"

# Check every task before launching any jobs to prevent a partial run.
for ENV_NAME in "${TASKS[@]}"; do
    ROOT_DIR="${BASE_DIR}/${ENV_NAME/:/_}/${EXPERIMENT}"
    if [[ "${DRY_RUN}" != "True" ]] &&
            [[ -e "${ROOT_DIR}" || -L "${ROOT_DIR}" ]]; then
        echo "Refusing to reuse existing target directory: ${ROOT_DIR}" >&2
        echo "Change --dir or move the existing directory." >&2
        exit 1
    fi
done

cat <<EOF
Starting dog:stand and dog:walk BAFCv6 seeds 0, 1, 2, and 3
  Tasks: ${TASKS[*]}
  Base dir: ${BASE_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  dog:stand GPUs per job: ${TASK_GPUS[0]}
  dog:walk GPUs per job: ${TASK_GPUS[1]}
  CPU threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OpenBLAS=$OPENBLAS_NUM_THREADS
  actor_critic_pairing: False
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  num_sampled_critics_for_actor: ${NUM_SAMPLED_CRITICS_FOR_ACTOR}
  num_sampled_critic_targets: ${NUM_SAMPLED_CRITIC_TARGETS}
  critic_utd: ${CRITIC_UTD}
  Critic reweighting solver: ${CRITIC_REWEIGHTING_SOLVER}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for TASK_INDEX in "${!TASKS[@]}"; do
    ENV_NAME="${TASKS[$TASK_INDEX]}"
    JOB_GPUS="${TASK_GPUS[$TASK_INDEX]}"
    ROOT_DIR="${BASE_DIR}/${ENV_NAME/:/_}/${EXPERIMENT}"
    for i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$i]}"
        MASTER_PORT=$((BASE_PORT + TASK_INDEX * ${#SEEDS[@]} + i))
        RUN_DIR="${ROOT_DIR}/bafcv6_random_target/seed_${SEED}"
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
            --conf_param "BafcAlgorithmV6.critic_utd=${CRITIC_UTD}"
            --conf_param "bafcv6_actor_critic_pairing=False"
            --conf_param "bafcv6_num_actor_critic=${NUM_ACTOR_CRITIC}"
            --conf_param "bafcv6_num_sampled_critics_for_actor=${NUM_SAMPLED_CRITICS_FOR_ACTOR}"
            --conf_param "bafcv6_use_random_critic_targets=True"
            --conf_param "bafcv6_num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
            --conf_param "create_environment.env_name='${ENV_NAME}'"
            --conf_param "BafcAlgorithmV6.enable_critic_reweighting=${ENABLE_CRITIC_REWEIGHTING}"
            --conf_param "BafcAlgorithmV6.critic_reweighting_solver='${CRITIC_REWEIGHTING_SOLVER}'"
            --conf_param "BafcAlgorithmV6.critic_reweighting_solver_iters=${CRITIC_REWEIGHTING_SOLVER_ITERS}"
            --conf_param "BafcAlgorithmV6.critic_reweighting_num_feature_coords=${CRITIC_REWEIGHTING_NUM_FEATURE_COORDS}"
            --conf_param "BafcAlgorithmV6.critic_reweighting_num_target_obs=${CRITIC_REWEIGHTING_NUM_TARGET_OBS}"
            --conf_param "BafcAlgorithmV6.critic_reweighting_max_weight=${CRITIC_REWEIGHTING_MAX_WEIGHT}"
            --distributed multi-gpu
        )

        if [[ "${DRY_RUN}" == "True" ]]; then
            printf 'OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' \
                "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "${JOB_GPUS}" "${MASTER_PORT}"
            printf '%q ' "${COMMAND[@]}"
            printf '> %q 2>&1 &\n' "${RUN_DIR}/out.log"
            continue
        fi

        mkdir -p "${RUN_DIR}"
        CUDA_VISIBLE_DEVICES="${JOB_GPUS}" MASTER_PORT="${MASTER_PORT}" \
            "${COMMAND[@]}" > "${RUN_DIR}/out.log" 2>&1 &
        PID=$!
        PIDS+=("${PID}")
        echo "  ${ENV_NAME} seed ${SEED}: port ${MASTER_PORT}, PID ${PID}"
        echo "    Log: ${RUN_DIR}/out.log"
    done
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched eight dog:stand/dog:walk BAFCv6 jobs (four GPUs per job): ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
for ENV_NAME in "${TASKS[@]}"; do
    ROOT_DIR="${BASE_DIR}/${ENV_NAME/:/_}/${EXPERIMENT}"
    echo "To monitor: tail -f ${ROOT_DIR}/bafcv6_random_target/seed_*/out.log"
    echo "Results: ${ROOT_DIR}"
done
