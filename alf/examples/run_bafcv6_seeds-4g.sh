#!/bin/bash
# Launcher script for BAFCv6 critic-UTD sweeps with 4 seeds.
# Each seed uses all configured GPUs via DDP; all jobs launch in parallel.
#
# Usage: bash run_bafcv6_seeds-4g.sh [options]
#   -e, --env ENV_NAME              DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR              Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS           Total environment steps (default: 600000)
#       --seeds CSV                 Four comma-separated seeds (default: 1,2,3,4)
#       --critic-utds CSV           Comma-separated critic_utd values (default: 2,3)
#       --gpus CSV                  Comma-separated GPU ids (default: 0,1,2,3)
#       --lbfgs-steps N             LBFGS solver steps for reweighting (default: 5)
#       --checkpoints N             Number of checkpoints (default: 10)
#       --resume-from-bafcv3-root DIR
#                                   Stage latest BAFCv3 seed checkpoints before launch
#   -h, --help                      Show this help message
#
# Example:
#   bash run_bafcv6_seeds-4g.sh --env cheetah:run --critic-utds 2,3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv6_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="cheetah:run"
# ENV_NAME="hopper:hop"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
SEEDS=(1 2 3 4)
CRITIC_UTDS=(2 3)
GPUS="0,1,2,3"
LBFGS_STEPS=3
REWEIGHTING_FEATURE_DIMENSION=32
RESUME_FROM_BAFCV3_ROOT=""
BASE_PORT=29500

parse_csv_array() {
    local input="$1"
    local -n output_ref="$2"
    IFS=',' read -r -a output_ref <<< "${input}"
}

print_help() {
    sed -n '/^# Usage:/,/^# Example:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
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

stage_bafcv3_checkpoint() {
    local seed="$1"
    local run_dir="$2"
    [[ -n "${RESUME_FROM_BAFCV3_ROOT}" ]] || return 0

    local dst_ckpt_dir="${run_dir}/train/algorithm"
    if [[ -d "${dst_ckpt_dir}" ]] && has_model_checkpoint "${dst_ckpt_dir}"; then
        echo "    Existing v6 checkpoint found; skip BAFCv3 staging."
        return 0
    fi

    local src_ckpt_dir="${RESUME_FROM_BAFCV3_ROOT}/seed_${seed}/train/algorithm"
    if [[ ! -d "${src_ckpt_dir}" ]]; then
        echo "Missing BAFCv3 checkpoint directory: ${src_ckpt_dir}" >&2
        exit 1
    fi

    local step=""
    if ! step="$(latest_checkpoint_step "${src_ckpt_dir}")"; then
        echo "No BAFCv3 model checkpoint found in ${src_ckpt_dir}" >&2
        exit 1
    fi

    mkdir -p "${dst_ckpt_dir}"
    local path=""
    for path in \
        "${src_ckpt_dir}/ckpt-${step}" \
        "${src_ckpt_dir}/ckpt-${step}-optimizer" \
        "${src_ckpt_dir}/ckpt-${step}-replay_buffer" \
        "${src_ckpt_dir}/ckpt-${step}-replay_buffer-rank"* \
        "${src_ckpt_dir}/ckpt-structure.json" \
        "${src_ckpt_dir}/ckpt-structure-optimizer.json" \
        "${src_ckpt_dir}/ckpt-structure-replay_buffer.json"; do
        [[ -e "${path}" ]] && cp "${path}" "${dst_ckpt_dir}/"
    done
    echo "    Staged BAFCv3 checkpoint step ${step} from ${src_ckpt_dir}"
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
        --seeds)
            parse_csv_array "$2" SEEDS
            shift 2
            ;;
        --critic-utds)
            parse_csv_array "$2" CRITIC_UTDS
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
        --checkpoints)
            NUM_CHECKPOINTS="$2"
            shift 2
            ;;
        --resume-from-bafcv3-root)
            RESUME_FROM_BAFCV3_ROOT="$2"
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

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi
if [[ ! -f "${CONF_FILE}" ]]; then
    echo "Config file not found: ${CONF_FILE}" >&2
    exit 1
fi
if [[ ${#SEEDS[@]} -ne 4 ]]; then
    echo "--seeds must provide exactly four comma-separated seeds." >&2
    echo "  Got: ${SEEDS[*]}" >&2
    exit 1
fi
if [[ ${#CRITIC_UTDS[@]} -eq 0 ]]; then
    echo "--critic-utds must provide at least one value." >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv6_dmc_4g"

cat <<EOF
Starting BAFCv6 critic-UTD sweep
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  critic_utd values: ${CRITIC_UTDS[*]}
  GPUs: ${GPUS}
  Critic reweighting: enabled
  LBFGS solver steps: ${LBFGS_STEPS}
  Reweighting feature dimension: ${REWEIGHTING_FEATURE_DIMENSION}
EOF
if [[ -n "${RESUME_FROM_BAFCV3_ROOT}" ]]; then
    echo "  Resume from BAFCv3 root: ${RESUME_FROM_BAFCV3_ROOT}"
fi
echo "  Python: ${PYTHON_BIN}"
echo ""

cd "${REPO_ROOT}"

PIDS=()
for utd_i in "${!CRITIC_UTDS[@]}"; do
    CRITIC_UTD="${CRITIC_UTDS[$utd_i]}"
    echo "Launching critic_utd=${CRITIC_UTD} jobs"

    for i in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$i]}"
        MASTER_PORT=$((BASE_PORT + utd_i * ${#SEEDS[@]} + i))
        RUN_DIR="${ROOT_DIR}/critic_utd${CRITIC_UTD}/seed_${SEED}"
        mkdir -p "${RUN_DIR}"
        stage_bafcv3_checkpoint "${SEED}" "${RUN_DIR}"

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
            > "${RUN_DIR}/out.log" 2>&1 &

        PID=$!
        PIDS+=("${PID}")
        echo "  Seed ${SEED}: GPUs ${GPUS}, port ${MASTER_PORT}, PID ${PID}"
        echo "    Log: ${RUN_DIR}/out.log"
    done
    echo ""
done

echo "Launched BAFCv6 jobs: ${PIDS[*]}"
echo "Launcher is not waiting for completion."
echo "To monitor: tail -f ${ROOT_DIR}/critic_utd*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
