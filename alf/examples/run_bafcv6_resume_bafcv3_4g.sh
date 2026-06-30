#!/bin/bash
# Launcher for BAFCv6 runs resumed from BAFCv3 4-GPU checkpoints.
# It stages the selected BAFCv3 checkpoint into each BAFCv6 run directory,
# then launches all requested jobs in parallel.
#
# Usage: bash run_bafcv6_resume_bafcv3_4g.sh [options]
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 600000)
#       --checkpoint-env-step N     BAFCv3 checkpoint suffix to stage (default: CHECKPOINT_ENV_STEP or 50050)
#       --v3-critic-utd N           BAFCv3 source critic_utd (default: V3_CRITIC_UTD, CRITIC_UTD, or 2)
#       --v6-critic-utd N           BAFCv6 target critic_utd (default: V6_CRITIC_UTD, CRITIC_UTD, or 2)
#       --gpus CSV                  Comma-separated GPU ids (default: GPUS or 0,1,2,3)
#       --lbfgs-steps N             LBFGS solver steps for reweighting (default: 3)
#       --feature-dim N             Reweighting feature dimension (default: 32)
#       --checkpoints N             Number of checkpoints (default: 10)
#       --base-port N               First DDP master port (default: BASE_PORT or 29500)
#       --hopper-seeds CSV          Hopper seeds (default: HOPPER_SEEDS or 0,1,2,3)
#       --cheetah-seeds CSV         Cheetah seeds (default: CHEETAH_SEEDS or 1,2,3,4)
#       --dry-run                   Validate and print planned jobs without staging or launching
#   -h, --help                      Show this help message
#
# Examples:
#   bash run_bafcv6_resume_bafcv3_4g.sh --dry-run
#   bash run_bafcv6_resume_bafcv3_4g.sh --v3-critic-utd 2 --v6-critic-utd 3 --dry-run
#   V3_CRITIC_UTD=3 V6_CRITIC_UTD=2 bash run_bafcv6_resume_bafcv3_4g.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv6_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

BASE_DIR="${BASE_DIR:-/root/numeric_results}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-600000}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-10}"
CHECKPOINT_ENV_STEP="${CHECKPOINT_ENV_STEP:-50050}"
DEFAULT_V3_CRITIC_UTD="${V3_CRITIC_UTD:-3}"
DEFAULT_V6_CRITIC_UTD="${V6_CRITIC_UTD:-2}"
V3_CRITIC_UTD="${V3_CRITIC_UTD:-${DEFAULT_V3_CRITIC_UTD}}"
V6_CRITIC_UTD="${V6_CRITIC_UTD:-${DEFAULT_V6_CRITIC_UTD}}"
GPUS="${GPUS:-0,1,2,3}"
LBFGS_STEPS="${LBFGS_STEPS:-3}"
REWEIGHTING_FEATURE_DIMENSION="${REWEIGHTING_FEATURE_DIMENSION:-32}"
BASE_PORT="${BASE_PORT:-29500}"
HOPPER_SEEDS_CSV="${HOPPER_SEEDS:-0,1,2,3}"
CHEETAH_SEEDS_CSV="${CHEETAH_SEEDS:-1,2,3,4}"
DRY_RUN=false

parse_csv_array() {
    local input="$1"
    local -n output_ref="$2"
    local item=""
    local raw=()

    output_ref=()
    IFS=',' read -r -a raw <<< "${input}"
    for item in "${raw[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -n "${item}" ]] && output_ref+=("${item}")
    done
}

print_help() {
    sed -n '/^# Usage:/,/^# Examples:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
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

env_name_for_dir() {
    local env_dir="$1"

    case "${env_dir}" in
        hopper)
            echo "hopper:hop"
            ;;
        cheetah)
            echo "cheetah:run"
            ;;
        *)
            echo "Unknown environment directory: ${env_dir}" >&2
            return 1
            ;;
    esac
}

seed_csv_for_env() {
    local env_dir="$1"

    case "${env_dir}" in
        hopper)
            echo "${HOPPER_SEEDS_CSV}"
            ;;
        cheetah)
            echo "${CHEETAH_SEEDS_CSV}"
            ;;
        *)
            echo "Unknown environment directory: ${env_dir}" >&2
            return 1
            ;;
    esac
}

required_checkpoint_files() {
    local src_ckpt_dir="$1"
    local checkpoint_step="$2"
    local rank_count="$3"
    local -n files_ref="$4"
    local rank=0

    files_ref=(
        "${src_ckpt_dir}/ckpt-${checkpoint_step}"
        "${src_ckpt_dir}/ckpt-${checkpoint_step}-optimizer"
        "${src_ckpt_dir}/ckpt-${checkpoint_step}-replay_buffer"
        "${src_ckpt_dir}/ckpt-structure.json"
        "${src_ckpt_dir}/ckpt-structure-optimizer.json"
        "${src_ckpt_dir}/ckpt-structure-replay_buffer.json"
    )
    for ((rank = 0; rank < rank_count; rank++)); do
        files_ref+=(
            "${src_ckpt_dir}/ckpt-${checkpoint_step}-replay_buffer-rank${rank}")
    done
}

validate_source_bundle() {
    local src_ckpt_dir="$1"
    local checkpoint_step="$2"
    local rank_count="$3"
    local files=()
    local path=""

    if [[ ! -d "${src_ckpt_dir}" ]]; then
        echo "Missing BAFCv3 checkpoint directory: ${src_ckpt_dir}" >&2
        return 1
    fi

    required_checkpoint_files "${src_ckpt_dir}" "${checkpoint_step}" \
        "${rank_count}" files
    for path in "${files[@]}"; do
        if [[ ! -f "${path}" ]]; then
            echo "Missing required source checkpoint file: ${path}" >&2
            return 1
        fi
    done
}

stage_checkpoint() {
    local src_ckpt_dir="$1"
    local dst_ckpt_dir="$2"
    local checkpoint_step="$3"
    local rank_count="$4"
    local files=()
    local path=""

    required_checkpoint_files "${src_ckpt_dir}" "${checkpoint_step}" \
        "${rank_count}" files

    mkdir -p "${dst_ckpt_dir}"
    for path in "${files[@]}"; do
        cp "${path}" "${dst_ckpt_dir}/"
    done
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
        --checkpoint-env-step)
            CHECKPOINT_ENV_STEP="$2"
            shift 2
            ;;
        --v3-critic-utd)
            V3_CRITIC_UTD="$2"
            shift 2
            ;;
        --v6-critic-utd)
            V6_CRITIC_UTD="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
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
        --checkpoints)
            NUM_CHECKPOINTS="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        --hopper-seeds)
            HOPPER_SEEDS_CSV="$2"
            shift 2
            ;;
        --cheetah-seeds)
            CHEETAH_SEEDS_CSV="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

parse_csv_array "${GPUS}" GPU_IDS

if [[ ! "${CHECKPOINT_ENV_STEP}" =~ ^[0-9]+$ ]]; then
    echo "--checkpoint-env-step must be a non-negative integer." >&2
    exit 1
fi
if [[ ! "${V3_CRITIC_UTD}" =~ ^[0-9]+$ ]]; then
    echo "--v3-critic-utd must be a non-negative integer." >&2
    exit 1
fi
if [[ ! "${V6_CRITIC_UTD}" =~ ^[0-9]+$ ]]; then
    echo "--v6-critic-utd must be a non-negative integer." >&2
    exit 1
fi
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "--gpus must provide at least one GPU id." >&2
    exit 1
fi
if [[ ! "${BASE_PORT}" =~ ^[0-9]+$ ]]; then
    echo "--base-port must be a non-negative integer." >&2
    exit 1
fi
if [[ ! -f "${CONF_FILE}" ]]; then
    echo "Config file not found: ${CONF_FILE}" >&2
    exit 1
fi
if [[ "${DRY_RUN}" == false && ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi

cat <<EOF
Starting BAFCv6 resume sweep from BAFCv3
  Config: ${CONF_FILE}
  Base dir: ${BASE_DIR}
  Source checkpoint: ckpt-${CHECKPOINT_ENV_STEP}
  Num env steps target: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  v3 critic_utd source: ${V3_CRITIC_UTD}
  v6 critic_utd target: ${V6_CRITIC_UTD}
  GPUs: ${GPUS}
  DDP ranks expected: ${#GPU_IDS[@]}
  Base port: ${BASE_PORT}
  Critic reweighting: enabled
  LBFGS solver steps: ${LBFGS_STEPS}
  Reweighting feature dimension: ${REWEIGHTING_FEATURE_DIMENSION}
  Python: ${PYTHON_BIN}
EOF
if [[ "${DRY_RUN}" == true ]]; then
    echo "  Mode: dry run"
fi
echo ""

cd "${REPO_ROOT}"

PIDS=()
JOB_INDEX=0
ENV_DIRS=(hopper cheetah)

for env_dir in "${ENV_DIRS[@]}"; do
    ENV_NAME="$(env_name_for_dir "${env_dir}")"
    parse_csv_array "$(seed_csv_for_env "${env_dir}")" SEEDS
    if [[ "${V3_CRITIC_UTD}" == "${V6_CRITIC_UTD}" ]]; then
        ROOT_DIR="${BASE_DIR}/${env_dir}/bafcv6_resume_bafcv3_ckpt${CHECKPOINT_ENV_STEP}_4g"
    else
        ROOT_DIR="${BASE_DIR}/${env_dir}/bafcv6_resume_bafcv3_critic_utd${V3_CRITIC_UTD}_ckpt${CHECKPOINT_ENV_STEP}_4g"
    fi

    if [[ ${#SEEDS[@]} -eq 0 ]]; then
        echo "No seeds configured for ${env_dir}." >&2
        exit 1
    fi

    echo "Environment: ${ENV_NAME}"
    echo "  Seeds: ${SEEDS[*]}"
    echo "  Root dir: ${ROOT_DIR}"

    for i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$i]}"
        MASTER_PORT=$((BASE_PORT + JOB_INDEX))
        SOURCE_CKPT_DIR="${BASE_DIR}/${env_dir}/bafcv3_dmc_4g/critic_utd${V3_CRITIC_UTD}/seed_${SEED}/train/algorithm"
        RUN_DIR="${ROOT_DIR}/critic_utd${V6_CRITIC_UTD}/seed_${SEED}"
        DST_CKPT_DIR="${RUN_DIR}/train/algorithm"
        OUT_LOG="${RUN_DIR}/out.log"

        validate_source_bundle "${SOURCE_CKPT_DIR}" \
            "${CHECKPOINT_ENV_STEP}" "${#GPU_IDS[@]}"
        if [[ -d "${DST_CKPT_DIR}" ]] && has_model_checkpoint "${DST_CKPT_DIR}"; then
            echo "Destination already has a checkpoint: ${DST_CKPT_DIR}" >&2
            echo "Move or remove it before starting a fresh resume." >&2
            exit 1
        fi

        if [[ "${DRY_RUN}" == true ]]; then
            echo "  Plan job ${JOB_INDEX}: ${env_dir} v3_critic_utd=${V3_CRITIC_UTD} v6_critic_utd=${V6_CRITIC_UTD} seed=${SEED}"
            echo "    Source: ${SOURCE_CKPT_DIR}/ckpt-${CHECKPOINT_ENV_STEP}"
            echo "    Destination: ${RUN_DIR}"
            echo "    Port: ${MASTER_PORT}"
        else
            mkdir -p "${RUN_DIR}"
            stage_checkpoint "${SOURCE_CKPT_DIR}" "${DST_CKPT_DIR}" \
                "${CHECKPOINT_ENV_STEP}" "${#GPU_IDS[@]}"

            CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
                "${PYTHON_BIN}" -m alf.bin.train \
                --conf "${CONF_FILE}" \
                --root_dir "${RUN_DIR}" \
                --conf_param "TrainerConfig.random_seed=${SEED}" \
                --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
                --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
                --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
                --conf_param "BafcAlgorithmV6.critic_utd=${V6_CRITIC_UTD}" \
                --conf_param "BafcAlgorithmV6.enable_critic_reweighting=True" \
                --conf_param "BafcAlgorithmV6.critic_reweighting_solver_iters=${LBFGS_STEPS}" \
                --conf_param "BafcAlgorithmV6.critic_reweighting_num_feature_coords=${REWEIGHTING_FEATURE_DIMENSION}" \
                --conf_param "create_environment.env_name='${ENV_NAME}'" \
                --distributed multi-gpu \
                > "${OUT_LOG}" 2>&1 &

            PID=$!
            PIDS+=("${PID}")
            echo "  Launched job ${JOB_INDEX}: v3_critic_utd=${V3_CRITIC_UTD} v6_critic_utd=${V6_CRITIC_UTD} seed=${SEED}"
            echo "    Source: ${SOURCE_CKPT_DIR}/ckpt-${CHECKPOINT_ENV_STEP}"
            echo "    Log: ${OUT_LOG}"
            echo "    Port: ${MASTER_PORT}, PID: ${PID}"
        fi

        JOB_INDEX=$((JOB_INDEX + 1))
    done
    echo ""
done

if [[ "${DRY_RUN}" == true ]]; then
    echo "Dry run complete. Total planned jobs: ${JOB_INDEX}"
else
    echo "Launched BAFCv6 resume jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
    echo "To monitor: tail -f ${BASE_DIR}/{hopper,cheetah}/$(basename "${ROOT_DIR}")/critic_utd${V6_CRITIC_UTD}/seed_*/out.log"
fi
