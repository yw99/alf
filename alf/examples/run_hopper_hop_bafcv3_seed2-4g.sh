#!/bin/bash
# Launch hopper:hop RLPD, BAFCv3_TR2, and BAFCv3 on seed 2 using DDP.
# This is the seed-2 counterpart of
# run_hopper_hop_rlpd_bafcv3_seed01-4g.sh.
#
# Usage: bash run_hopper_hop_bafcv3_seed2-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps (default: 800000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (wrapper default: 29506)
#       --dry-run            Print all three commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_hopper_hop_bafcv3_seed2-4g.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_hopper_hop_rlpd_bafcv3_seed01-4g.sh" \
    --seeds 2 --base-port 29506 "$@"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="hopper:hop"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
SEED=2
GPUS="0,1,2,3"
MASTER_PORT=29506
DRY_RUN=False

CRITIC_UTD=11
UPDATES_PER_ITER=12
NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITICS=8
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
        --port)
            MASTER_PORT="$2"
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
if [[ ! "${MASTER_PORT}" =~ ^[1-9][0-9]*$ ]] || (( MASTER_PORT > 65535 )); then
    echo "--port must be a valid port, got: ${MASTER_PORT}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME//:/_}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/rlpd_bafcv3_comparison_4g"
RUN_DIR="${ROOT_DIR}/bafcv3/fixed_pairingFalse_num_sampled_critic${NUM_SAMPLED_CRITICS}/critic_utd${CRITIC_UTD}/seed_${SEED}"

command=(
    "${PYTHON_BIN}" -m alf.bin.train
    --conf "${CONF_FILE}"
    --root_dir "${RUN_DIR}"
    --conf_param "TrainerConfig.random_seed=${SEED}"
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
    --conf_param "TrainerConfig.num_updates_per_train_iter=${UPDATES_PER_ITER}"
    --conf_param "make_ddp_performer.find_unused_parameters=True"
    --conf_param "create_environment.env_name='${ENV_NAME}'"
    --conf_param "TrainerConfig.debug_summaries=${DEBUG_SUMMARIES}"
    --conf_param "BafcAlgorithmV3.critic_utd=${CRITIC_UTD}"
    --conf_param "bafcv3_actor_use_ln=${ACTOR_USE_LN}"
    --conf_param "bafcv3_actor_critic_pairing=False"
    --conf_param "bafcv3_num_actor_critic=${NUM_ACTOR_CRITIC}"
    --conf_param "bafcv3_num_sampled_critics_for_actor=${NUM_SAMPLED_CRITICS}"
    --conf_param "bafcv3_use_random_critic_targets=True"
    --conf_param "bafcv3_num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
    --distributed multi-gpu
)

cat <<EOF
Starting hopper:hop BAFCv3 replacement run
  Run dir: ${RUN_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seed: ${SEED}
  GPUs: ${GPUS}
  Master port: ${MASTER_PORT}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

if [[ "${DRY_RUN}" == "True" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${MASTER_PORT}"
    printf '%q ' "${command[@]}"
    printf '> %q 2>&1 &\n' "${RUN_DIR}/out.log"
    exit 0
fi

if [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory already exists; refusing to overwrite it: ${RUN_DIR}" >&2
    exit 1
fi
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
    "${command[@]}" > "${RUN_DIR}/out.log" 2>&1 &

PID=$!
echo "Launched BAFCv3 seed ${SEED}: PID ${PID}"
echo "Log: ${RUN_DIR}/out.log"
echo "To monitor: tail -f ${RUN_DIR}/out.log"
