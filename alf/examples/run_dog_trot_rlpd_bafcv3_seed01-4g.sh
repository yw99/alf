#!/bin/bash
# Launch the dog:trot RLPD/BAFCv3 comparison on seeds 2 and 3.
# Every job uses all four configured GPUs through DDP. The four jobs run in
# parallel on unique torch.distributed master ports.
#
# Conditions:
#   1. RLPD (critic_utd=10)
#   2. BAFCv3 without actor-critic pairing, K=8 critics per actor (critic_utd=11)
# BAFCv3 uses random critic TD targets.
#
# Usage: bash run_dog_trot_rlpd_bafcv3_seed01-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 800000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29500)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_dog_trot_rlpd_bafcv3_seed01-4g.sh --dry-run

set -euo pipefail

# Use headless EGL rendering for dm_control unless explicitly overridden.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Use deterministic CUBLAS workspace for reproducibility unless explicitly overridden.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RLPD_CONF="${SCRIPT_DIR}/rlpd_dmc_conf.py"
BAFCV3_CONF="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="dog:trot"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(2 3)

RLPD_CRITIC_UTD=10
RLPD_UPDATES_PER_ITER=11
BAFCV3_CRITIC_UTD=11
BAFCV3_UPDATES_PER_ITER=12
BAFCV3_NUM_ACTOR_CRITIC=10
BAFCV3_NUM_SAMPLED_CRITICS=8
BAFCV3_NUM_SAMPLED_CRITIC_TARGETS=1
BAFCV3_ACTOR_USE_LN=False
BAFCV3_DEBUG_SUMMARIES=True

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
for config_file in "${RLPD_CONF}" "${BAFCV3_CONF}"; do
    if [[ ! -f "${config_file}" ]]; then
        echo "Config file not found: ${config_file}" >&2
        exit 1
    fi
done
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

# Keep the task in the path so dog:trot cannot reuse another dog task's checkpoints.
ENV_DIR="${ENV_NAME//:/_}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/rlpd_bafcv3_comparison_4g"

cat <<EOF
Starting dog:trot RLPD/BAFCv3 comparison
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  GPUs per job: ${GPUS}
  RLPD: critic_utd=${RLPD_CRITIC_UTD}
  BAFCv3 pairing off, random targets: critic_utd=${BAFCV3_CRITIC_UTD}, num_sampled_critic=${BAFCV3_NUM_SAMPLED_CRITICS}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
launch_job() {
    local condition="$1"
    local seed="$2"
    local master_port="$3"
    local config_file="$4"
    local updates_per_iter="$5"
    shift 5

    local run_dir="${ROOT_DIR}/${condition}/seed_${seed}"
    local -a command=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${config_file}"
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

port_offset=0
for seed in "${SEEDS[@]}"; do
    launch_job \
        "rlpd/critic_utd${RLPD_CRITIC_UTD}" \
        "${seed}" "$((BASE_PORT + port_offset))" "${RLPD_CONF}" \
        "${RLPD_UPDATES_PER_ITER}" \
        --conf_param "RlpdAlgorithm.critic_utd=${RLPD_CRITIC_UTD}"
    ((port_offset += 1))
done

for seed in "${SEEDS[@]}"; do
    launch_job \
        "bafcv3/fixed_pairingFalse_num_sampled_critic${BAFCV3_NUM_SAMPLED_CRITICS}/critic_utd${BAFCV3_CRITIC_UTD}" \
        "${seed}" "$((BASE_PORT + port_offset))" "${BAFCV3_CONF}" \
        "${BAFCV3_UPDATES_PER_ITER}" \
        --conf_param "TrainerConfig.debug_summaries=${BAFCV3_DEBUG_SUMMARIES}" \
        --conf_param "BafcAlgorithmV3.critic_utd=${BAFCV3_CRITIC_UTD}" \
        --conf_param "bafcv3_actor_use_ln=${BAFCV3_ACTOR_USE_LN}" \
        --conf_param "bafcv3_actor_critic_pairing=False" \
        --conf_param "bafcv3_num_actor_critic=${BAFCV3_NUM_ACTOR_CRITIC}" \
        --conf_param "bafcv3_num_sampled_critics_for_actor=${BAFCV3_NUM_SAMPLED_CRITICS}" \
        --conf_param "bafcv3_use_random_critic_targets=True" \
        --conf_param "bafcv3_num_sampled_critic_targets=${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}"
    ((port_offset += 1))
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched four 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor RLPD: tail -f ${ROOT_DIR}/rlpd/*/seed_*/out.log"
echo "To monitor BAFCv3: tail -f ${ROOT_DIR}/bafcv3/*/*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
