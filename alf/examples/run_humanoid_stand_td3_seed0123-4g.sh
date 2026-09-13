#!/bin/bash
# Launch humanoid:stand TD3 jobs on seeds 0, 1, 2, and 3. Every job uses
# all four configured GPUs through DDP. The four jobs run in parallel on
# unique torch.distributed master ports.
#
# The environment budget and launch topology match
# run_humanoid_stand_rlpd_seed0123-4g.sh. TD3 is configured with the same
# 10 critics and 10 critic updates per actor update as that RLPD baseline.
# One critic is sampled for each TD target, the actor uses the mean of all 10
# critics, and critic layer normalization is enabled for stability at high UTD.
#
# Usage: bash run_humanoid_stand_td3_seed0123-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 600000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29500)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_humanoid_stand_td3_seed0123-4g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/td3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:stand"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(0 1 2 3)

NUM_CRITICS=10
NUM_SAMPLED_CRITIC_TARGETS=1
ACTOR_CRITIC_AGGREGATION="mean"
CRITIC_USE_LN=True
ACTOR_UTD=1
CRITIC_UTD=10
NUM_UPDATES_PER_TRAIN_ITER=11

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
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 3 > 65535 )); then
    echo "--base-port must leave room for four valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ROOT_DIR="${BASE_DIR}/humanoid_stand/td3_seed0123_4g/num_critics${NUM_CRITICS}_sampled_target${NUM_SAMPLED_CRITIC_TARGETS}_actor_${ACTOR_CRITIC_AGGREGATION}_critic_ln${CRITIC_USE_LN}_critic_utd${CRITIC_UTD}"

# Refuse to resume into or overwrite an existing experiment. Perform this
# check before launching any jobs so a collision cannot cause a partial run.
if [[ "${DRY_RUN}" != "True" ]] &&
        [[ -e "${ROOT_DIR}" || -L "${ROOT_DIR}" ]]; then
    echo "Refusing to reuse existing target directory: ${ROOT_DIR}" >&2
    echo "Change --dir or move the existing directory." >&2
    exit 1
fi

cat <<EOF
Starting humanoid:stand TD3 seeds 0, 1, 2, and 3
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  GPUs per job: ${GPUS}
  CPU threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OpenBLAS=$OPENBLAS_NUM_THREADS
  Num critics: ${NUM_CRITICS}
  Num sampled critic targets: ${NUM_SAMPLED_CRITIC_TARGETS}
  Actor critic aggregation: ${ACTOR_CRITIC_AGGREGATION}
  Critic layer norm: ${CRITIC_USE_LN}
  Actor UTD: ${ACTOR_UTD}
  Critic UTD: ${CRITIC_UTD}
  Num updates per train iter: ${NUM_UPDATES_PER_TRAIN_ITER}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    MASTER_PORT=$((BASE_PORT + i))
    RUN_DIR="${ROOT_DIR}/seed_${SEED}"
    COMMAND=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${CONF_FILE}"
        --root_dir "${RUN_DIR}"
        --conf_param "TrainerConfig.random_seed=${SEED}"
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
        --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
        --conf_param "TrainerConfig.num_updates_per_train_iter=${NUM_UPDATES_PER_TRAIN_ITER}"
        --conf_param "Td3Algorithm.num_critic_replicas=${NUM_CRITICS}"
        --conf_param "Td3Algorithm.num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
        --conf_param "Td3Algorithm.actor_critic_aggregation='${ACTOR_CRITIC_AGGREGATION}'"
        --conf_param "td3_critic_use_ln=${CRITIC_USE_LN}"
        --conf_param "Td3Algorithm.actor_utd=${ACTOR_UTD}"
        --conf_param "Td3Algorithm.critic_utd=${CRITIC_UTD}"
        --conf_param "make_ddp_performer.find_unused_parameters=True"
        --conf_param "create_environment.env_name='${ENV_NAME}'"
        --distributed multi-gpu
    )

    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' \
            "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "${GPUS}" "${MASTER_PORT}"
        printf '%q ' "${COMMAND[@]}"
        printf '> %q 2>&1 &\n' "${RUN_DIR}/out.log"
        continue
    fi

    mkdir -p "${RUN_DIR}"
    CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
        "${COMMAND[@]}" > "${RUN_DIR}/out.log" 2>&1 &
    PID=$!
    PIDS+=("${PID}")
    echo "  Seed ${SEED}: port ${MASTER_PORT}, PID ${PID}"
    echo "    Log: ${RUN_DIR}/out.log"
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched four humanoid:stand TD3 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
