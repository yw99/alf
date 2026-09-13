#!/bin/bash
# Launch the BAFCv6 random-target condition on humanoid:run seeds 0–3.
# Each job uses all configured GPUs through DDP; all four jobs run in parallel.
# Uses BAFCv6 with random critic targets and critic reweighting.
#
# Usage: bash run_humanoid_run_bafcv6_seed0123-4g.sh [options]
#   -d, --dir BASE_DIR              Base results directory (default: /workspace/alf_results)
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
#   bash run_humanoid_run_bafcv6_seed0123-4g.sh --dry-run

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

ENV_NAME="humanoid:run"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
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
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(0 1 2 3)

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
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 3 > 65535 )); then
    echo "--base-port must leave room for four valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ROOT_DIR="${BASE_DIR}/humanoid_run/bafcv6_seed0123_4g/num_actor_critic${NUM_ACTOR_CRITIC}_num_sampled_critics_for_actor${NUM_SAMPLED_CRITICS_FOR_ACTOR}_num_sampled_critic_targets${NUM_SAMPLED_CRITIC_TARGETS}/critic_utd${CRITIC_UTD}"

cat <<EOF
Starting BAFCv6 humanoid:run seeds 0–3
  Config: ${CONF_FILE}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  actor_critic_pairing: False
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  num_sampled_critics_for_actor: ${NUM_SAMPLED_CRITICS_FOR_ACTOR}
  num_sampled_critic_targets: ${NUM_SAMPLED_CRITIC_TARGETS}
  critic_utd: ${CRITIC_UTD}
  Critic reweighting solver: ${CRITIC_REWEIGHTING_SOLVER}
  GPUs per job: ${GPUS}
  CPU threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OpenBLAS=$OPENBLAS_NUM_THREADS
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for seed_index in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$seed_index]}"
    MASTER_PORT=$((BASE_PORT + seed_index))
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
            "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "${GPUS}" "${MASTER_PORT}"
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

if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched four BAFCv6 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/bafcv6_random_target/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
