#!/bin/bash
# Resume dog:walk BAFCv3 TR2 checkpoints with the isolated TR3 refill path.
# The checkpoints must already be staged under the TR3 result roots.
#
# Usage: bash run_quadruped_walk_bafcv3_tr3_resume_seed01-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total recorded environment steps (default: 600000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29600)
#       --dry-run            Validate checkpoints and print commands only
#   -h, --help               Show this help message
#
# Example:
#   bash run_quadruped_walk_bafcv3_tr3_resume_seed01-4g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TR3_CONF="${SCRIPT_DIR}/bafcv3_tr3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

BASE_DIR="/workspace/alf_results"
ENV_NAME="dog:walk"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29600
DRY_RUN=False
SEEDS=(0 1)
CHECKPOINT_STEP=90090
CONDITION="bafcv3_tr3/fixed_pairingFalse_num_sampled_critic8/critic_utd11"

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
    exit 1
fi
if [[ ! -f "${TR3_CONF}" ]]; then
    echo "Config file not found: ${TR3_CONF}" >&2
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
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 1 > 65535 )); then
    echo "--base-port must leave room for two valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ROOT_DIR="${BASE_DIR}/${ENV_NAME%%:*}/rlpd_bafcv3_comparison_4g"

checkpoint_files=(
    "ckpt-${CHECKPOINT_STEP}"
    "ckpt-${CHECKPOINT_STEP}-optimizer"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer-rank0"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer-rank1"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer-rank2"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer-rank3"
    "ckpt-${CHECKPOINT_STEP}-rank-state-rank0"
    "ckpt-${CHECKPOINT_STEP}-rank-state-rank1"
    "ckpt-${CHECKPOINT_STEP}-rank-state-rank2"
    "ckpt-${CHECKPOINT_STEP}-rank-state-rank3"
)

for seed in "${SEEDS[@]}"; do
    algorithm_dir="${ROOT_DIR}/${CONDITION}/seed_${seed}/train/algorithm"
    for checkpoint_file in "${checkpoint_files[@]}"; do
        if [[ ! -f "${algorithm_dir}/${checkpoint_file}" ]]; then
            echo "Missing staged checkpoint file: ${algorithm_dir}/${checkpoint_file}" >&2
            exit 1
        fi
    done
done

cat <<EOF
BAFCv3 TR3 replay-refill resume
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Condition: ${CONDITION}
  Checkpoint: ${CHECKPOINT_STEP}
  Recorded env-step target: ${NUM_ENV_STEPS}
  Seeds: ${SEEDS[*]}
  GPUs per job: ${GPUS}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"
PIDS=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    master_port="$((BASE_PORT + index))"
    run_dir="${ROOT_DIR}/${CONDITION}/seed_${seed}"
    command=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${TR3_CONF}"
        --root_dir "${run_dir}"
        --conf_param "TrainerConfig.random_seed=${seed}"
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
        --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
        --conf_param "create_environment.env_name='${ENV_NAME}'"
        --conf_param "make_ddp_performer.find_unused_parameters=True"
        --distributed multi-gpu
    )

    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${master_port}"
        printf '%q ' "${command[@]}"
        printf '> %q 2>&1 &\n' "${run_dir}/out.log"
        continue
    fi

    CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${master_port}" \
        "${command[@]}" > "${run_dir}/out.log" 2>&1 &
    PIDS+=("$!")
    echo "  seed ${seed}: PID ${PIDS[-1]}, log ${run_dir}/out.log"
done

if [[ "${DRY_RUN}" == "True" ]]; then
    echo ""
    echo "Dry run complete; no jobs were launched."
    exit 0
fi

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
