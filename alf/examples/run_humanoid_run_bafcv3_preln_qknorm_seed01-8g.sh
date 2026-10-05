#!/bin/bash
# Launch BAFCv3 pre-LN/QK-normalization experiments on humanoid:run, seeds 0-1.
# The first four GPUs run actor-LN jobs; the last four run without actor LN.
# Each group runs QK normalization off/on, each with seeds 0 and 1: four
# concurrent four-GPU DDP jobs per group, eight jobs total. All jobs use final
# transformer normalization, a slow target encoder, corrected actor gradients,
# gradient-chain diagnostics, and a math-backend diagnostic comparison.
#
# Usage: bash run_humanoid_run_bafcv3_preln_qknorm_seed01-8g.sh [options]
#   -e, --env DOMAIN:TASK    Environment name (default: humanoid:run)
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 600000)
#       --gpus CSV           Eight distinct GPU ids (default: 0,1,2,3,4,5,6,7)
#                            First four with actor LN, last four without actor LN
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First of eight DDP master ports (default: 29920)
#       --dry-run            Print commands without creating files or launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_humanoid_run_bafcv3_preln_qknorm_seed01-8g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:run"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3,4,5,6,7"
BASE_PORT=29920
DRY_RUN=False
SEEDS=(0 1)
ACTOR_LN_OPTIONS=(True False)
QK_NORM_OPTIONS=(False True)

print_help() {
    sed -n '/^# Usage:/,/^# Example:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env)
            ENV_NAME="${2:?--env requires DOMAIN:TASK}"
            shift 2
            ;;
        -d|--dir)
            BASE_DIR="${2:?--dir requires a directory}"
            shift 2
            ;;
        -n|--steps)
            NUM_ENV_STEPS="${2:?--steps requires a positive integer}"
            shift 2
            ;;
        --gpus)
            GPUS="${2:?--gpus requires eight GPU ids}"
            shift 2
            ;;
        --checkpoints)
            NUM_CHECKPOINTS="${2:?--checkpoints requires a positive integer}"
            shift 2
            ;;
        --base-port)
            BASE_PORT="${2:?--base-port requires a port}"
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

if [[ ! "${ENV_NAME}" =~ ^[[:alnum:]_]+:[[:alnum:]_]+$ ]]; then
    echo "--env must have the form DOMAIN:TASK, got: ${ENV_NAME}" >&2
    exit 1
fi
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
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( ${#BASE_PORT} > 5 || BASE_PORT > 65528 )); then
    echo "--base-port must leave room for eight valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi
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
GROUP_GPUS=(
    "${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]},${GPU_IDS[3]}"
    "${GPU_IDS[4]},${GPU_IDS[5]},${GPU_IDS[6]},${GPU_IDS[7]}"
)

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_PARENT="${BASE_DIR}/${ENV_NAME/:/_}/bafcv3_preln_qknorm_seed01_8g"
ROOT_DIR="${RESULTS_PARENT}/${RUN_ID}"
if [[ -e "${ROOT_DIR}" || -L "${ROOT_DIR}" ]]; then
    echo "Refusing to reuse existing results directory: ${ROOT_DIR}" >&2
    exit 1
fi
if [[ "${DRY_RUN}" != "True" ]]; then
    mkdir -p "${RESULTS_PARENT}"
    # Atomically reserve a fresh destination before opening any job logs.
    if ! mkdir "${ROOT_DIR}"; then
        echo "Could not create fresh results directory: ${ROOT_DIR}" >&2
        exit 1
    fi
fi

cat <<SUMMARY
Starting ${ENV_NAME} BAFCv3 pre-LN/QK-normalization experiments
  Root dir: ${ROOT_DIR}
  Num env steps per job: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds per configuration: ${SEEDS[*]}
  Actor LN on, GPUs per job: ${GROUP_GPUS[0]}
  Actor LN off, GPUs per job: ${GROUP_GPUS[1]}
  QK normalization per group: ${QK_NORM_OPTIONS[*]}
  Pre-LN / final transformer normalization / target encoder: True
  Legacy gradient: False
  Gradient-chain diagnostics / math-backend comparison: True
  Production attention backend: automatic
  Critic / actor UTD: 11 / 1; updates per iteration: 12
  Actors / critics: 10; sampled actor critics: 8; random target critics: 1
  Threads: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}
  Dry run: ${DRY_RUN}
SUMMARY

cd "${REPO_ROOT}"
PIDS=()
for GROUP_INDEX in "${!GROUP_GPUS[@]}"; do
    JOB_GPUS="${GROUP_GPUS[$GROUP_INDEX]}"
    ACTOR_USE_LN="${ACTOR_LN_OPTIONS[$GROUP_INDEX]}"
    for QK_INDEX in "${!QK_NORM_OPTIONS[@]}"; do
        NORMALIZE_QK="${QK_NORM_OPTIONS[$QK_INDEX]}"
        for SEED in "${SEEDS[@]}"; do
            MASTER_PORT=$((BASE_PORT + GROUP_INDEX * 4 + QK_INDEX * 2 + SEED))
            RUN_DIR="${ROOT_DIR}/actor_ln${ACTOR_USE_LN}/qk_norm${NORMALIZE_QK}/seed_${SEED}"
            COMMAND=(
                "${PYTHON_BIN}" -m alf.bin.train
                --conf "${CONF_FILE}"
                --root_dir "${RUN_DIR}"
                --conf_param "TrainerConfig.random_seed=${SEED}"
                --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
                --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
                --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
                --conf_param "TrainerConfig.num_updates_per_train_iter=12"
                --conf_param "TrainerConfig.debug_summaries=True"
                --conf_param "BafcAlgorithmV3.critic_utd=11"
                --conf_param "BafcAlgorithmV3.actor_utd=1"
                --conf_param "debug_mode=False"
                --conf_param "bafcv3_use_single_layer_transformer_encoder=True"
                --conf_param "bafcv3_use_actor_id_encoding=False"
                --conf_param "bafcv3_detach_actor_policy_input=False"
                --conf_param "bafcv3_transformer_norm_first=True"
                --conf_param "bafcv3_transformer_final_norm=True"
                --conf_param "bafcv3_transformer_normalize_qk=${NORMALIZE_QK}"
                --conf_param "bafcv3_num_attention_heads=1"
                --conf_param "bafcv3_use_target_actor_encoder=True"
                --conf_param "bafcv3_use_legacy_actor_gradient=False"
                --conf_param "bafcv3_debug_gradient_chain=True"
                --conf_param "bafcv3_debug_gradient_chain_compare_backends=True"
                --conf_param "bafcv3_actor_use_ln=${ACTOR_USE_LN}"
                --conf_param "bafcv3_actor_critic_pairing=False"
                --conf_param "bafcv3_num_actor_critic=10"
                --conf_param "bafcv3_num_sampled_critics_for_actor=8"
                --conf_param "bafcv3_use_random_critic_targets=True"
                --conf_param "bafcv3_num_sampled_critic_targets=1"
                --conf_param "make_ddp_performer.find_unused_parameters=True"
                --conf_param "create_environment.env_name='${ENV_NAME}'"
                --distributed multi-gpu
            )
            if [[ "${DRY_RUN}" == "True" ]]; then
                printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${JOB_GPUS}" "${MASTER_PORT}"
                printf '%q ' "${COMMAND[@]}"
                printf '> %q 2>&1 &\n' "${RUN_DIR}/out.log"
                continue
            fi
            mkdir -p "${RUN_DIR}"
            CUDA_VISIBLE_DEVICES="${JOB_GPUS}" MASTER_PORT="${MASTER_PORT}" \
                "${COMMAND[@]}" > "${RUN_DIR}/out.log" 2>&1 &
            PID=$!
            PIDS+=("${PID}")
            echo "  Actor LN ${ACTOR_USE_LN}, QK norm ${NORMALIZE_QK}, seed ${SEED}: GPUs ${JOB_GPUS}, port ${MASTER_PORT}, PID ${PID}"
            echo "    Log: ${RUN_DIR}/out.log"
        done
    done
done
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched eight ${ENV_NAME} BAFCv3 four-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/actor_ln*/qk_norm*/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
