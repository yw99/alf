#!/bin/bash
# Resume dog:walk BAFCv3 seeds 0-3 from server2/server checkpoints. Each job uses four
# configured GPUs through DDP, with a unique torch.distributed master port.
#
# BAFCv3 runs without actor-critic pairing, with K=8 critics per actor,
# critic_utd=11, and random critic TD targets. Resume from 600k total steps
# to 800k total (150k to 200k per worker).
#
# Usage: bash run_dog_walk_bafcv3_seed0123-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 800000)
#       --source-dir DIR     Seeds 0-1 source base (default: /workspace/server2_copy)
#       --source-dir-23 DIR  Seeds 2-3 source base (default: /workspace/server_copy)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29500)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_dog_walk_bafcv3_seed0123-4g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAFCV3_CONF="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="dog:walk"
BASE_DIR="/workspace/alf_results"
SOURCE_DIR="/workspace/server2_copy"
SOURCE_DIR_23="/workspace/server_copy"
CHECKPOINT_STEP=150150
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(0 1 2 3)

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
        --source-dir-23)
            SOURCE_DIR_23="$2"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="$2"
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
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 3 > 65535 )); then
    echo "--base-port must leave room for four valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME//:/_}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_resume_server_copies_seed0123_4g_to${NUM_ENV_STEPS}"
CONDITION="fixed_pairingFalse_num_sampled_critic${BAFCV3_NUM_SAMPLED_CRITICS}/critic_utd${BAFCV3_CRITIC_UTD}"

# Preserve the four-rank checkpoint topology and total-step interpretation.
if [[ ! "${GPUS}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
    echo "--gpus must specify exactly four GPU ids" >&2
    exit 1
fi
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
for ((i=0; i<4; i++)); do
    for ((j=i+1; j<4; j++)); do
        if [[ "${GPU_IDS[i]}" == "${GPU_IDS[j]}" ]]; then
            echo "--gpus must contain four distinct GPU ids" >&2
            exit 1
        fi
    done
done
if (( NUM_ENV_STEPS <= 600000 )); then
    echo "--steps must exceed the source checkpoint's 600000 total environment steps" >&2
    exit 1
fi
if [[ -e "${ROOT_DIR}" || -L "${ROOT_DIR}" ]]; then
    echo "Refusing to reuse existing target directory: ${ROOT_DIR}" >&2
    echo "Choose a new --dir base directory." >&2
    exit 1
fi

CHECKPOINT_FILES=(
    "ckpt-${CHECKPOINT_STEP}"
    "ckpt-${CHECKPOINT_STEP}-optimizer"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer-rank0"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer-rank1"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer-rank2"
    "ckpt-${CHECKPOINT_STEP}-replay_buffer-rank3"
)
source_checkpoint_dir_for_seed() {
    local seed="$1"
    if (( seed < 2 )); then
        printf '%s/dog_bafcv3_s%s/train/algorithm' "${SOURCE_DIR}" "${seed}"
    else
        printf '%s/dog_bafc_trainable_rtT_s%s/train/algorithm' "${SOURCE_DIR_23}" "${seed}"
    fi
}

# Validate every source before creating results or starting any job.
for seed in "${SEEDS[@]}"; do
    source_checkpoint_dir="$(source_checkpoint_dir_for_seed "${seed}")"
    for checkpoint_file in "${CHECKPOINT_FILES[@]}"; do
        if [[ ! -s "${source_checkpoint_dir}/${checkpoint_file}" ]]; then
            echo "Missing or empty checkpoint file: ${source_checkpoint_dir}/${checkpoint_file}" >&2
            exit 1
        fi
    done
done

cat <<EOF
Resuming dog:walk BAFCv3 on seeds 0, 1, 2, and 3
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Condition: ${CONDITION}
  Source base (seeds 0-1): ${SOURCE_DIR}
  Source base (seeds 2-3): ${SOURCE_DIR_23}
  Source checkpoint: ckpt-${CHECKPOINT_STEP} (150000 env steps per worker)
  Total env-step target: ${NUM_ENV_STEPS}
  Per-worker env-step target: $(((NUM_ENV_STEPS + 3) / 4))
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  GPUs per job: ${GPUS}
  CPU threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OpenBLAS=$OPENBLAS_NUM_THREADS
  critic_utd: ${BAFCV3_CRITIC_UTD}
  num_updates_per_train_iter: ${BAFCV3_UPDATES_PER_ITER}
  actor_critic_pairing: False
  num_actor_critic: ${BAFCV3_NUM_ACTOR_CRITIC}
  num_sampled_critics_for_actor: ${BAFCV3_NUM_SAMPLED_CRITICS}
  use_random_critic_targets: True
  num_sampled_critic_targets: ${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

# Copy complete checkpoints before starting any job. ALF automatically
# restores from root_dir/train/algorithm, including optimizer and progress.
if [[ "${DRY_RUN}" != "True" ]]; then
    mkdir -p "$(dirname "${ROOT_DIR}")"
    # Non-recursive mkdir reserves a new destination and rejects races.
    mkdir "${ROOT_DIR}"
fi
for seed in "${SEEDS[@]}"; do
    source_checkpoint_dir="$(source_checkpoint_dir_for_seed "${seed}")"
    target_checkpoint_dir="${ROOT_DIR}/${CONDITION}/seed_${seed}/train/algorithm"
    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'Would copy checkpoint %s from %q to %q\n' \
            "${CHECKPOINT_STEP}" "${source_checkpoint_dir}" "${target_checkpoint_dir}"
    else
        mkdir -p "${target_checkpoint_dir}"
        for checkpoint_file in "${CHECKPOINT_FILES[@]}"; do
            cp --reflink=auto -- "${source_checkpoint_dir}/${checkpoint_file}" "${target_checkpoint_dir}/"
        done
        printf 'Source: %s\nCheckpoint: %s\n' "${source_checkpoint_dir}" \
            "${CHECKPOINT_STEP}" > "${ROOT_DIR}/${CONDITION}/seed_${seed}/resume_source.txt"
    fi
done

PIDS=()
launch_job() {
    local seed="$1"
    local master_port="$2"
    local run_dir="${ROOT_DIR}/${CONDITION}/seed_${seed}"
    local -a command=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${BAFCV3_CONF}"
        --root_dir "${run_dir}"
        --conf_param "TrainerConfig.random_seed=${seed}"
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
        --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
        --conf_param "TrainerConfig.num_updates_per_train_iter=${BAFCV3_UPDATES_PER_ITER}"
        --conf_param "TrainerConfig.debug_summaries=${BAFCV3_DEBUG_SUMMARIES}"
        --conf_param "make_ddp_performer.find_unused_parameters=True"
        --conf_param "create_environment.env_name='${ENV_NAME}'"
        --conf_param "BafcAlgorithmV3.critic_utd=${BAFCV3_CRITIC_UTD}"
        --conf_param "bafcv3_actor_use_ln=${BAFCV3_ACTOR_USE_LN}"
        --conf_param "bafcv3_actor_critic_pairing=False"
        --conf_param "bafcv3_num_actor_critic=${BAFCV3_NUM_ACTOR_CRITIC}"
        --conf_param "bafcv3_num_sampled_critics_for_actor=${BAFCV3_NUM_SAMPLED_CRITICS}"
        --conf_param "bafcv3_use_random_critic_targets=True"
        --conf_param "bafcv3_num_sampled_critic_targets=${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}"
        --distributed multi-gpu
    )

    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' \
            "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "${GPUS}" "${master_port}"
        printf '%q ' "${command[@]}"
        printf '> %q 2>&1 &\n' "${run_dir}/out.log"
        return
    fi

    mkdir -p "${run_dir}"
    CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${master_port}" \
        "${command[@]}" > "${run_dir}/out.log" 2>&1 &
    local pid=$!
    PIDS+=("${pid}")
    echo "  seed ${seed}: port ${master_port}, PID ${pid}"
    echo "    Log: ${run_dir}/out.log"
}

for seed_index in "${!SEEDS[@]}"; do
    launch_job "${SEEDS[$seed_index]}" "$((BASE_PORT + seed_index))"
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched four resumed BAFCv3 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/${CONDITION}/seed_*/out.log"
echo "Results: ${ROOT_DIR}/${CONDITION}"
