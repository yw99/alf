#!/bin/bash
# Launch the two BAFCv7 presets for dog:run and seeds 0,1.
# Sweep both mean_log_std and action_quantiles policy features.
# Each job uses four GPUs: seed 0 uses GPUs 0-3 and seed 1 uses GPUs 4-7.
# Each group runs two ensemble_base and two single_seeded jobs (both feature
# modes). All eight jobs run concurrently.
# Each launch creates a fresh UTC-timestamped results directory.
#
# Usage: bash run_dog_run_bafcv7_seeds01-8g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Environment steps per job (default: 800000)
#       --gpus CSV           Eight distinct GPU ids (default: 0,1,2,3,4,5,6,7)
#                            First four for seed 0, last four for seed 1
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First of eight DDP ports (default: 29600)
#       --disable-optimizations Use the reference BAFCv7 implementation
#       --dry-run            Print all eight commands without launching
#   -h, --help               Show this help
#
# Example:
#   bash run_dog_run_bafcv7_seeds01-8g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv7_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="dog:run"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3,4,5,6,7"
BASE_PORT=29600
DRY_RUN=False
ENABLE_OPTIMIZATIONS=True
POLICY_FEATURE_MODES=(mean_log_std action_quantiles)
ACTOR_UTD=1
CRITIC_UTD=3
SEEDS=(0 1)
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
        --disable-optimizations)
            ENABLE_OPTIMIZATIONS=False
            shift
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
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + ${#VARIANTS[@]} * ${#SEEDS[@]} * ${#POLICY_FEATURE_MODES[@]} - 1 > 65535 )); then
    echo "--base-port must leave room for eight valid ports" >&2
    exit 1
fi

# Split the host's eight GPUs into disjoint groups of four, one per seed.
if [[ ! "${GPUS}" =~ ^[[:alnum:]_-]+(,[[:alnum:]_-]+){7}$ ]]; then
    echo "--gpus must contain exactly eight distinct GPU ids, got: ${GPUS}" >&2
    exit 1
fi
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
for i in "${!GPU_IDS[@]}"; do
    for ((j = 0; j < i; j++)); do
        if [[ "${GPU_IDS[$i]}" == "${GPU_IDS[$j]}" ]]; then
            echo "--gpus must contain distinct GPU ids, got: ${GPUS}" >&2
            exit 1
        fi
    done
done
SEED_GPUS=(
    "${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]},${GPU_IDS[3]}"
    "${GPU_IDS[4]},${GPU_IDS[5]},${GPU_IDS[6]},${GPU_IDS[7]}"
)

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_PARENT="${BASE_DIR}/dog_run/bafcv7_policy_features_8g"
ROOT_DIR="${RESULTS_PARENT}/${RUN_ID}"

if [[ -e "${ROOT_DIR}" || -L "${ROOT_DIR}" ]]; then
    echo "Refusing to reuse existing results directory: ${ROOT_DIR}" >&2
    exit 1
fi
if [[ "${DRY_RUN}" != "True" ]]; then
    mkdir -p "${RESULTS_PARENT}"
    # Atomically reserve the destination before any job can open its log.
    if ! mkdir "${ROOT_DIR}"; then
        echo "Could not create fresh results directory: ${ROOT_DIR}" >&2
        exit 1
    fi
fi

echo "Starting BAFCv7 dog:run sweep"
echo "  Config: ${CONF_FILE}"
echo "  Root dir: ${ROOT_DIR}"
echo "  Variants: ${VARIANTS[*]}"
echo "  ensemble_base lambda: ${TEMPORAL_NOISE_MIX[ensemble_base]}"
echo "  single_seeded lambda: ${TEMPORAL_NOISE_MIX[single_seeded]}"
echo "  UTD: actor=${ACTOR_UTD}, critic=${CRITIC_UTD}"
echo "  Policy features: ${POLICY_FEATURE_MODES[*]}"
echo "  Eval samples: frozen"
echo "  Quantile levels (when selected): [-1,0,+1]"
echo "  Seeds: ${SEEDS[*]}"
echo "  Environment steps: ${NUM_ENV_STEPS}"
echo "  Seed 0 GPUs per job: ${SEED_GPUS[0]}"
echo "  Seed 1 GPUs per job: ${SEED_GPUS[1]}"
echo "  Each GPU group: two ensemble_base and two single_seeded jobs"
echo "  Threads: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}"
echo "  Optimizations: ${ENABLE_OPTIMIZATIONS}"
echo "  Dry run: ${DRY_RUN}"
echo ""

cd "${REPO_ROOT}"
PIDS=()
port_offset=0
for variant in "${VARIANTS[@]}"; do
    temporal_noise_mix="${TEMPORAL_NOISE_MIX[$variant]}"
    for policy_features in "${POLICY_FEATURE_MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_gpus="${SEED_GPUS[$seed]}"
            master_port=$((BASE_PORT + port_offset))
            run_dir="${ROOT_DIR}/${policy_features}/${variant}/lambda${temporal_noise_mix}/actor_utd${ACTOR_UTD}_critic_utd${CRITIC_UTD}/seed_${seed}"
            command=(
                "${PYTHON_BIN}" -m alf.bin.train
                --conf "${CONF_FILE}"
                --root_dir "${run_dir}"
                --conf_param "bafcv7_enable_optimizations=${ENABLE_OPTIMIZATIONS}"
                --conf_param "bafcv7_variant='${variant}'"
                --conf_param "BafcAlgorithmV7.temporal_noise_mix=${temporal_noise_mix}"
                --conf_param "BafcAlgorithmV7.policy_feature_mode='${policy_features}'"
                --conf_param "BafcAlgorithmV7.eval_samples_source='frozen'"
                --conf_param "BafcAlgorithmV7.actor_utd=${ACTOR_UTD}"
                --conf_param "BafcAlgorithmV7.critic_utd=${CRITIC_UTD}"
                --conf_param "TrainerConfig.random_seed=${seed}"
                --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
                --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
                --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
                --conf_param "bafcv7_env_name='${ENV_NAME}'"
                --conf_param "make_ddp_performer.find_unused_parameters=True"
                --distributed multi-gpu
            )

            if [[ "${DRY_RUN}" == "True" ]]; then
                printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${job_gpus}" "${master_port}"
                printf '%q ' "${command[@]}"
                printf '> %q 2>&1 &\n' "${run_dir}/out.log"
            else
                mkdir -p "${run_dir}"
                CUDA_VISIBLE_DEVICES="${job_gpus}" MASTER_PORT="${master_port}" \
                    "${command[@]}" > "${run_dir}/out.log" 2>&1 &
                PIDS+=("$!")
                echo "  ${variant}, ${policy_features}, seed ${seed}: GPUs ${job_gpus}, port ${master_port}, PID $!"
                echo "    Log: ${run_dir}/out.log"
            fi
            ((port_offset += 1))
        done
    done
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; emitted eight jobs and launched none."
else
    echo "Launched eight BAFCv7 four-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "Results: ${ROOT_DIR}/{mean_log_std,action_quantiles}/{ensemble_base/lambda0.10,single_seeded/lambda0.90}/actor_utd${ACTOR_UTD}_critic_utd${CRITIC_UTD}/seed_{0,1}"
