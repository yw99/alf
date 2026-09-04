#!/bin/bash
# Launch the two BAFCv7 presets for hopper:hop on one random seed.
#
# Usage: bash run_hopper_hop_bafcv7_seed-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Environment steps per job (default: 800000)
#       --gpus CSV           GPUs used by both DDP jobs (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First of two DDP ports (default: 29600)
#       --seed SEED          Random seed for both jobs (default: 0)
#       --policy-features M  mean_log_std or action_quantiles (default: mean_log_std)
#       --dry-run            Print both commands without launching
#   -h, --help               Show this help
#
# Example:
#   bash run_hopper_hop_bafcv7_seed-4g.sh --dry-run
#   bash run_hopper_hop_bafcv7_seed-4g.sh --seed 3

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv7_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="hopper:hop"
SEED=0
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29600
DRY_RUN=False
POLICY_FEATURES="mean_log_std"
ACTOR_UTD=1
CRITIC_UTD=3
VARIANTS=(ensemble_base single_seeded)
declare -A TEMPORAL_NOISE_MIX=(
    [ensemble_base]="0.10"
    [single_seeded]="0.90"
)

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
        --seed)
            SEED="$2"
            shift 2
            ;;
        --policy-features)
            POLICY_FEATURES="$2"
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
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a nonnegative integer" >&2
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
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 1 > 65535 )); then
    echo "--base-port must leave room for two valid ports" >&2
    exit 1
fi
if [[ "${POLICY_FEATURES}" != "mean_log_std" && "${POLICY_FEATURES}" != "action_quantiles" ]]; then
    echo "--policy-features must be mean_log_std or action_quantiles" >&2
    exit 1
fi

ROOT_DIR="${BASE_DIR}/hopper_hop/bafcv7_policy_features_4g"

echo "Starting BAFCv7 hopper:hop single-seed comparison"
echo "  Config: ${CONF_FILE}"
echo "  Root dir: ${ROOT_DIR}"
echo "  Variants: ${VARIANTS[*]}"
echo "  ensemble_base lambda: ${TEMPORAL_NOISE_MIX[ensemble_base]}"
echo "  single_seeded lambda: ${TEMPORAL_NOISE_MIX[single_seeded]}"
echo "  UTD: actor=${ACTOR_UTD}, critic=${CRITIC_UTD}"
echo "  Entropy regularization: disabled"
echo "  Policy features: ${POLICY_FEATURES}"
echo "  Quantile levels (when selected): [-1,0,+1]"
echo "  Seed: ${SEED}"
echo "  Environment steps: ${NUM_ENV_STEPS}"
echo "  GPUs per job: ${GPUS}"
echo "  Dry run: ${DRY_RUN}"
echo ""

cd "${REPO_ROOT}"
PIDS=()
port_offset=0
for variant in "${VARIANTS[@]}"; do
    temporal_noise_mix="${TEMPORAL_NOISE_MIX[$variant]}"
    master_port=$((BASE_PORT + port_offset))
    run_dir="${ROOT_DIR}/${POLICY_FEATURES}/${variant}/lambda${temporal_noise_mix}/actor_utd${ACTOR_UTD}_critic_utd${CRITIC_UTD}/seed_${SEED}"
    command=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${CONF_FILE}"
        --root_dir "${run_dir}"
        --conf_param "bafcv7_variant='${variant}'"
        --conf_param "BafcAlgorithmV7.temporal_noise_mix=${temporal_noise_mix}"
        --conf_param "BafcAlgorithmV7.policy_feature_mode='${POLICY_FEATURES}'"
        --conf_param "BafcAlgorithmV7.actor_utd=${ACTOR_UTD}"
        --conf_param "BafcAlgorithmV7.critic_utd=${CRITIC_UTD}"
        --conf_param "TrainerConfig.random_seed=${SEED}"
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
        echo "  ${variant}, seed ${SEED}: port ${master_port}, PID $!"
        echo "    Log: ${run_dir}/out.log"
    fi
    ((port_offset += 1))
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; emitted two jobs and launched none."
else
    echo "Launched two BAFCv7 four-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "Results: ${ROOT_DIR}/${POLICY_FEATURES}/{ensemble_base/lambda0.10,single_seeded/lambda0.90}/actor_utd${ACTOR_UTD}_critic_utd${CRITIC_UTD}/seed_${SEED}"
