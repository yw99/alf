#!/bin/bash
# Launch the pairing-disabled quadruped:walk BAFCv3 configuration on seeds 2
# and 3. Every job uses all four configured GPUs through DDP. The two jobs run in
# parallel on unique torch.distributed master ports.
#
# Conditions:
#   1. BAFCv3 without fixed pairing, K=8 critics per actor (critic_utd=11)
#
# Usage: bash run_quadruped_walk_bafcv3_seed23-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 600000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29500)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_quadruped_walk_bafcv3_seed23-4g.sh --dry-run

set -euo pipefail

# Required by deterministic PyTorch operations backed by CuBLAS.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAFCV3_CONF="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

# Ensure DDP workers can discover the virtualenv Ninja executable.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

ENV_NAME="dog:walk"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(2 3)

BAFCV3_PAIRING_OFF_CRITIC_UTD=11
BAFCV3_UPDATES_PER_ITER=12
BAFCV3_NUM_ACTOR_CRITIC=10
BAFCV3_PAIRING_OFF_NUM_SAMPLED_CRITICS=8

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
if [[ ! -f "${BAFCV3_CONF}" ]]; then
    echo "Config file not found: ${BAFCV3_CONF}" >&2
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

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/rlpd_bafcv3_comparison_4g"

cat <<EOF
Starting pairing-disabled dog:walk BAFCv3 jobs on seeds 2 and 3
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  GPUs per job: ${GPUS}
  BAFCv3 pairing off: critic_utd=${BAFCV3_PAIRING_OFF_CRITIC_UTD}, num_sampled_critic=${BAFCV3_PAIRING_OFF_NUM_SAMPLED_CRITICS}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
launch_job() {
    local condition="$1"
    local seed="$2"
    local master_port="$3"
    local updates_per_iter="$4"
    shift 4

    local run_dir="${ROOT_DIR}/${condition}/seed_${seed}"
    local -a command=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${BAFCV3_CONF}"
        --root_dir "${run_dir}"
        --conf_param "TrainerConfig.random_seed=${seed}"
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
        --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
        --conf_param "TrainerConfig.num_updates_per_train_iter=${updates_per_iter}"
        --conf_param "make_ddp_performer.find_unused_parameters=True"
        --conf_param "create_environment.env_name='${ENV_NAME}'"
        "$@"
        --distributed multi-gpu
    )

    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${master_port}"
        printf '%q ' "${command[@]}"
        printf '> %q 2>&1 &\n' "${run_dir}/out.log"
        return
    fi

    mkdir -p "${run_dir}"
    CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${master_port}" \
        "${command[@]}" > "${run_dir}/out.log" 2>&1 &
    local pid=$!
    PIDS+=("${pid}")
    echo "  ${condition}, seed ${seed}: port ${master_port}, PID ${pid}"
    echo "    Log: ${run_dir}/out.log"
}

PAIRING_OFF_ARGS=(
    --conf_param "BafcAlgorithmV3.critic_utd=${BAFCV3_PAIRING_OFF_CRITIC_UTD}"
    --conf_param "bafcv3_actor_critic_pairing=False"
    --conf_param "bafcv3_num_actor_critic=${BAFCV3_NUM_ACTOR_CRITIC}"
    --conf_param "bafcv3_num_sampled_critics_for_actor=${BAFCV3_PAIRING_OFF_NUM_SAMPLED_CRITICS}"
)
for seed_index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$seed_index]}"
    launch_job "bafcv3/fixed_pairingFalse_num_sampled_critic${BAFCV3_PAIRING_OFF_NUM_SAMPLED_CRITICS}/critic_utd${BAFCV3_PAIRING_OFF_CRITIC_UTD}" "${seed}" "$((BASE_PORT + seed_index))" "${BAFCV3_UPDATES_PER_ITER}" "${PAIRING_OFF_ARGS[@]}"
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched two pairing-disabled 4-GPU BAFCv3 jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/bafcv3/*/*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
