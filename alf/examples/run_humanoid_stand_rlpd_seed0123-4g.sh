#!/bin/bash
# Launch humanoid:stand RLPD jobs on seeds 0, 1, 2, and 3. Every job uses
# all four configured GPUs through DDP. The four jobs run in parallel on
# unique torch.distributed master ports.
#
# This preserves the RLPD settings from
# run_humanoid_walk_rlpd_bafcv3_seed23-4g.sh.
#
# Usage: bash run_humanoid_stand_rlpd_seed0123-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 600000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29500)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_humanoid_stand_rlpd_seed0123-4g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/rlpd_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:stand"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(0 1 2 3)

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

ROOT_DIR="${BASE_DIR}/humanoid_stand/rlpd_seed0123_4g_rerun_20260908/critic_utd${CRITIC_UTD}"

# Refuse to resume into or overwrite an existing experiment. Check every seed
# before launching any jobs so a collision cannot result in a partial launch.
if [[ "${DRY_RUN}" != "True" ]]; then
    for seed in "${SEEDS[@]}"; do
        run_dir="${ROOT_DIR}/seed_${seed}"
        if [[ -e "${run_dir}" || -L "${run_dir}" ]]; then
            echo "Refusing to reuse existing run directory: ${run_dir}" >&2
            echo "Change the experiment path or move the existing directory." >&2
            exit 1
        fi
    done
fi

cat <<EOF
Starting humanoid:stand RLPD seeds 0, 1, 2, and 3
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  GPUs per job: ${GPUS}
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
        --conf_param "RlpdAlgorithm.critic_utd=${CRITIC_UTD}"
        --conf_param "make_ddp_performer.find_unused_parameters=True"
        --conf_param "create_environment.env_name='${ENV_NAME}'"
        --distributed multi-gpu
    )

    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${MASTER_PORT}"
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
    echo "Launched four humanoid:stand RLPD 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
