#!/bin/bash
# Debug launcher for one BAFCv6 seed resumed from a BAFCv3 4-GPU checkpoint.
#
# Usage: bash run_bafcv6_resume_bafcv3_seed-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 30000)
#       --seed N                    Random seed (default: 0)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N             Number of checkpoints (default: 10)
#       --critic-utd N              BAFCv6 critic_utd (default: 2)
#       --lbfgs-steps N             LBFGS solver steps for reweighting (default: 3)
#       --feature-dim N             Reweighting feature dimension (default: 32)
#       --source-run-dir DIR        BAFCv3 run dir to resume from
#       --port N                    DDP master port (default: 29500)
#   -h, --help                      Show this help message
#
# Examples:
#   bash run_bafcv6_resume_bafcv3_seed-4g.sh
#   bash run_bafcv6_resume_bafcv3_seed-4g.sh --seed 0 --gpus 0,1,2,3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv6_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="cheetah:run"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=30000
NUM_CHECKPOINTS=10
SEED=0
GPUS="0,1,2,3"
MASTER_PORT=29500
CRITIC_UTD=3
LBFGS_STEPS=3
REWEIGHTING_FEATURE_DIMENSION=32
SOURCE_RUN_DIR=""

parse_csv_array() {
    local input="$1"
    local -n output_ref="$2"
    IFS=',' read -r -a output_ref <<< "${input}"
}

print_help() {
    sed -n '/^# Usage:/,/^# Examples:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

latest_checkpoint_step() {
    local ckpt_dir="$1"
    local best=""
    local path=""
    local base=""
    local step=""
    shopt -s nullglob
    for path in "${ckpt_dir}"/ckpt-[0-9]*; do
        base="$(basename "${path}")"
        [[ "${base}" =~ ^ckpt-([0-9]+)$ ]] || continue
        step="${BASH_REMATCH[1]}"
        if [[ -z "${best}" || "${step}" -gt "${best}" ]]; then
            best="${step}"
        fi
    done
    shopt -u nullglob
    [[ -n "${best}" ]] || return 1
    echo "${best}"
}

has_model_checkpoint() {
    local ckpt_dir="$1"
    local path=""
    local base=""
    shopt -s nullglob
    for path in "${ckpt_dir}"/ckpt-[0-9]*; do
        base="$(basename "${path}")"
        if [[ "${base}" =~ ^ckpt-[0-9]+$ ]]; then
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob
    return 1
}

replay_sidecar_has_replay_keys() {
    local replay_path="$1"
    "${PYTHON_BIN}" - "${replay_path}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")


def has_replay_key(nest):
    if isinstance(nest, dict):
        for key, value in nest.items():
            if isinstance(key, str) and "_replay_buffer." in key:
                return True
            if has_replay_key(value):
                return True
    return False

if not has_replay_key(checkpoint):
    sys.exit(1)
PY
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
        --seed)
            SEED="$2"
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
        --critic-utd)
            CRITIC_UTD="$2"
            shift 2
            ;;
        --lbfgs-steps)
            LBFGS_STEPS="$2"
            shift 2
            ;;
        --feature-dim)
            REWEIGHTING_FEATURE_DIMENSION="$2"
            shift 2
            ;;
        --source-run-dir)
            SOURCE_RUN_DIR="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
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

parse_csv_array "${GPUS}" GPU_IDS

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi
if [[ ! -f "${CONF_FILE}" ]]; then
    echo "Config file not found: ${CONF_FILE}" >&2
    exit 1
fi
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "--gpus must provide at least one GPU id." >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
if [[ -z "${SOURCE_RUN_DIR}" ]]; then
    SOURCE_RUN_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_dmc_4g_debug/seed_${SEED}"
fi
SOURCE_CKPT_DIR="${SOURCE_RUN_DIR}/train/algorithm"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv6_resume_bafcv3_4g_debug"
RUN_DIR="${ROOT_DIR}/critic_utd${CRITIC_UTD}/seed_${SEED}"
DST_CKPT_DIR="${RUN_DIR}/train/algorithm"
OUT_LOG="${RUN_DIR}/out.log"

if [[ ! -d "${SOURCE_CKPT_DIR}" ]]; then
    echo "Missing source checkpoint directory: ${SOURCE_CKPT_DIR}" >&2
    exit 1
fi

if ! SOURCE_STEP="$(latest_checkpoint_step "${SOURCE_CKPT_DIR}")"; then
    echo "No source model checkpoint found in ${SOURCE_CKPT_DIR}" >&2
    exit 1
fi

if [[ -d "${DST_CKPT_DIR}" ]] && has_model_checkpoint "${DST_CKPT_DIR}"; then
    echo "Destination already has a checkpoint: ${DST_CKPT_DIR}" >&2
    echo "Remove or move the destination run before starting a fresh debug resume." >&2
    exit 1
fi

required_files=(
    "${SOURCE_CKPT_DIR}/ckpt-${SOURCE_STEP}"
    "${SOURCE_CKPT_DIR}/ckpt-${SOURCE_STEP}-optimizer"
    "${SOURCE_CKPT_DIR}/ckpt-${SOURCE_STEP}-replay_buffer"
    "${SOURCE_CKPT_DIR}/ckpt-structure.json"
    "${SOURCE_CKPT_DIR}/ckpt-structure-optimizer.json"
    "${SOURCE_CKPT_DIR}/ckpt-structure-replay_buffer.json"
)
for rank in "${!GPU_IDS[@]}"; do
    required_files+=(
        "${SOURCE_CKPT_DIR}/ckpt-${SOURCE_STEP}-replay_buffer-rank${rank}")
done
for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required source checkpoint file: ${path}" >&2
        exit 1
    fi
done

replay_files=("${SOURCE_CKPT_DIR}/ckpt-${SOURCE_STEP}-replay_buffer")
for rank in "${!GPU_IDS[@]}"; do
    replay_files+=(
        "${SOURCE_CKPT_DIR}/ckpt-${SOURCE_STEP}-replay_buffer-rank${rank}")
done
for path in "${replay_files[@]}"; do
    if ! replay_sidecar_has_replay_keys "${path}"; then
        echo "Source replay checkpoint is empty or missing replay keys: ${path}" >&2
        echo "Regenerate the BAFCv3 checkpoint after the Agent replay checkpoint patch before running v6 resume." >&2
        exit 1
    fi
done

mkdir -p "${DST_CKPT_DIR}"
for path in "${required_files[@]}"; do
    cp "${path}" "${DST_CKPT_DIR}/"
done

cat <<EOF
Starting BAFCv6 4-GPU debug resume from BAFCv3
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Source run dir: ${SOURCE_RUN_DIR}
  Source checkpoint: ckpt-${SOURCE_STEP}
  Destination run dir: ${RUN_DIR}
  Num env steps target: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seed: ${SEED}
  GPUs: ${GPUS}
  Master port: ${MASTER_PORT}
  critic_utd: ${CRITIC_UTD}
  LBFGS solver steps: ${LBFGS_STEPS}
  Reweighting feature dimension: ${REWEIGHTING_FEATURE_DIMENSION}
  Python: ${PYTHON_BIN}
EOF
echo ""
echo "Staged BAFCv3 checkpoint files into ${DST_CKPT_DIR}"
echo "Log: ${OUT_LOG}"
echo ""

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
    "${PYTHON_BIN}" -m alf.bin.train \
    --conf "${CONF_FILE}" \
    --root_dir "${RUN_DIR}" \
    --conf_param "TrainerConfig.random_seed=${SEED}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "BafcAlgorithmV6.critic_utd=${CRITIC_UTD}" \
    --conf_param "BafcAlgorithmV6.enable_critic_reweighting=True" \
    --conf_param "BafcAlgorithmV6.critic_reweighting_solver_iters=${LBFGS_STEPS}" \
    --conf_param "BafcAlgorithmV6.critic_reweighting_num_feature_coords=${REWEIGHTING_FEATURE_DIMENSION}" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${OUT_LOG}" 2>&1 &

PID=$!
echo "Launched BAFCv6 resume debug run: PID ${PID}"
echo "Log: ${OUT_LOG}"
echo "To monitor: tail -f ${OUT_LOG}"
