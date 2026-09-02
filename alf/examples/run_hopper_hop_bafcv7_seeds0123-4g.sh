#!/bin/bash
# Launch the two BAFCv7 presets for hopper:hop and seeds 0,1,2,3.
#
# Usage: bash run_hopper_hop_bafcv7_seeds0123-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Environment steps per job (default: 800000)
#       --gpus CSV           GPUs used by every DDP job (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First of eight DDP ports (default: 29600)
#       --dry-run            Print all eight commands without launching
#   -h, --help               Show this help
#
# Example:
#   bash run_hopper_hop_bafcv7_seeds0123-4g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv7_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="hopper:hop"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29600
DRY_RUN=False
SEEDS=(0 1 2 3)
VARIANTS=(ensemble_base single_seeded)

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
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${CONF_FILE}" ]]; then
    echo "Config file not found: ${CONF_FILE}" >&2
    exit 1
fi
if [[ ! "${NUM_ENV_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--steps must be a positive integer" >&2
    exit 1
fi
if [[ ! "${NUM_CHECKPOINTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--checkpoints must be a positive integer" >&2
    exit 1
fi
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 7 > 65535 )); then
    echo "--base-port must leave room for eight valid ports" >&2
    exit 1
fi

ROOT_DIR="${BASE_DIR}/hopper_hop/bafcv7_episode_seeded_4g"

echo "Starting BAFCv7 hopper:hop sweep"
echo "  Config: ${CONF_FILE}"
echo "  Root dir: ${ROOT_DIR}"
echo "  Variants: ${VARIANTS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Environment steps: ${NUM_ENV_STEPS}"
echo "  GPUs per job: ${GPUS}"
echo "  Dry run: ${DRY_RUN}"
echo ""

cd "${REPO_ROOT}"
PIDS=()
port_offset=0
for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        master_port=$((BASE_PORT + port_offset))
        run_dir="${ROOT_DIR}/${variant}/seed_${seed}"
        command=(
            "${PYTHON_BIN}" -m alf.bin.train
            --conf "${CONF_FILE}"
            --root_dir "${run_dir}"
            --conf_param "bafcv7_variant='${variant}'"
            --conf_param "TrainerConfig.random_seed=${seed}"
            --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
            --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
            --conf_param "bafcv7_env_name='${ENV_NAME}'"
            --conf_param "make_ddp_performer.find_unused_parameters=True"
            --distributed multi-gpu
        )

        if [[ "${DRY_RUN}" == "True" ]]; then
            printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${master_port}"
            printf '%q ' "${command[@]}"
            printf '> %q 2>&1 &\n' "${run_dir}/out.log"
        else
            mkdir -p "${run_dir}"
            CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${master_port}" \
                "${command[@]}" > "${run_dir}/out.log" 2>&1 &
            PIDS+=("$!")
            echo "  ${variant}, seed ${seed}: port ${master_port}, PID $!"
            echo "    Log: ${run_dir}/out.log"
        fi
        ((port_offset += 1))
    done
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; emitted eight jobs and launched none."
else
    echo "Launched eight BAFCv7 four-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "Results: ${ROOT_DIR}/{ensemble_base,single_seeded}/seed_{0,1,2,3}"
