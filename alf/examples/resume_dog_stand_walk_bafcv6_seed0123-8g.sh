#!/bin/bash
# Resume runs created by run_dog_stand_walk_bafcv6_seed0123-8g.sh.
# Seeds 0-3 for dog:stand share GPUs 0-3; seeds 0-3 for dog:walk share GPUs
# 4-7. Each job uses four DDP ranks and its original saved alf_config.py,
# including the 800000 total environment-step budget (200000 per rank).
# ALF automatically loads the latest checkpoint from each existing run.
# All eight runs are checked before launch, including per-rank replay files.
# Logs go to out.resume.<UTC timestamp>.<launcher PID>.log; out.log is kept.
#
# Usage: bash resume_dog_stand_walk_bafcv6_seed0123-8g.sh [options]
#   -d, --dir BASE_DIR       Original base directory (default: /workspace/alf_results)
#       --num-actor-critic N Original actor/critic count (default: 10)
#       --num-sampled-targets N Original target critic count (default: 1)
#       --gpus CSV           Eight distinct GPU ids (default: 0,1,2,3,4,5,6,7)
#       --base-port PORT     First of eight DDP ports (default: 29500)
#       --dry-run            Validate checkpoints and print commands only
#   -h, --help               Show this help message
#
# Example:
#   bash resume_dog_stand_walk_bafcv6_seed0123-8g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
BASE_DIR="/workspace/alf_results"
GPUS="0,1,2,3,4,5,6,7"
BASE_PORT=29500
NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITIC_TARGETS=1
TASKS=(dog:stand dog:walk)
SEEDS=(0 1 2 3)
DRY_RUN=False

fail() {
    echo "$*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir|--gpus|--base-port|--num-actor-critic|--num-sampled-targets)
            [[ $# -ge 2 && -n "$2" ]] || fail "Missing value for $1"
            case "$1" in
                -d|--dir) BASE_DIR="$2" ;;
                --gpus) GPUS="$2" ;;
                --base-port) BASE_PORT="$2" ;;
                --num-actor-critic) NUM_ACTOR_CRITIC="$2" ;;
                --num-sampled-targets) NUM_SAMPLED_CRITIC_TARGETS="$2" ;;
            esac
            shift 2
            ;;
        --dry-run)
            DRY_RUN=True
            shift
            ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# Example:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
            exit 0
            ;;
        *) fail "Unknown option: $1 (use --help)" ;;
    esac
done

[[ -x "${PYTHON_BIN}" ]] || fail "Python interpreter is not executable: ${PYTHON_BIN}"
[[ -d "${BASE_DIR}" ]] || fail "Original results directory is missing: ${BASE_DIR}"
BASE_DIR="$(cd "${BASE_DIR}" && pwd)"
[[ "${BASE_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] &&
    (( BASE_PORT <= 65528 )) || fail "--base-port must be between 1 and 65528"
[[ "${NUM_ACTOR_CRITIC}" =~ ^[1-9][0-9]{0,5}$ ]] &&
    (( NUM_ACTOR_CRITIC >= 8 )) || fail "--num-actor-critic must be at least 8"
[[ "${NUM_SAMPLED_CRITIC_TARGETS}" =~ ^[1-9][0-9]{0,5}$ ]] &&
    (( NUM_SAMPLED_CRITIC_TARGETS <= NUM_ACTOR_CRITIC )) ||
    fail "--num-sampled-targets must be between 1 and --num-actor-critic"
[[ "${GPUS}" =~ ^[[:alnum:]_-]+(,[[:alnum:]_-]+){7}$ ]] ||
    fail "--gpus must contain exactly eight distinct GPU ids"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
for i in "${!GPU_IDS[@]}"; do
    for ((j = 0; j < i; j++)); do
        [[ "${GPU_IDS[$i]}" != "${GPU_IDS[$j]}" ]] ||
            fail "Duplicate GPU id: ${GPU_IDS[$i]}"
    done
done
TASK_GPUS=(
    "${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]},${GPU_IDS[3]}"
    "${GPU_IDS[4]},${GPU_IDS[5]},${GPU_IDS[6]},${GPU_IDS[7]}"
)
EXPERIMENT="bafcv6_seed0123_8g/num_actor_critic${NUM_ACTOR_CRITIC}_num_sampled_critics_for_actor8_num_sampled_critic_targets${NUM_SAMPLED_CRITIC_TARGETS}/critic_utd11/bafcv6_random_target"

# Match Checkpointer's latest numeric model selection. Reject an incomplete
# latest checkpoint instead of silently starting fresh or dropping replay data.
RUN_DIRS=()
CHECKPOINT_STEPS=()
shopt -s nullglob
for ENV_NAME in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_DIR="${BASE_DIR}/${ENV_NAME/:/_}/${EXPERIMENT}/seed_${SEED}"
        [[ -s "${RUN_DIR}/alf_config.py" ]] || fail "Missing saved config: ${RUN_DIR}/alf_config.py"
        [[ -s "${RUN_DIR}/config_files/bafcv6_dmc_conf.py" ]] ||
            fail "Missing saved BAFCv6 config in ${RUN_DIR}/config_files"
        LATEST_STEP=-1
        for CHECKPOINT in "${RUN_DIR}"/train/algorithm/ckpt-*; do
            NAME="${CHECKPOINT##*/}"
            if [[ "${NAME}" =~ ^ckpt-([0-9]+)$ ]]; then
                STEP=$((10#${BASH_REMATCH[1]}))
                if (( STEP > LATEST_STEP )); then
                    LATEST_STEP="${STEP}"
                fi
            fi
        done
        (( LATEST_STEP >= 0 )) || fail "No model checkpoint found in ${RUN_DIR}"
        CHECKPOINT="${RUN_DIR}/train/algorithm/ckpt-${LATEST_STEP}"
        for SUFFIX in "" -optimizer -replay_buffer-rank{0..3}; do
            [[ -s "${CHECKPOINT}${SUFFIX}" && -r "${CHECKPOINT}${SUFFIX}" ]] ||
                fail "Missing, empty, or unreadable checkpoint file: ${CHECKPOINT}${SUFFIX}"
        done
        RUN_DIRS+=("${RUN_DIR}")
        CHECKPOINT_STEPS+=("${LATEST_STEP}")
    done
done

# Avoid resuming into a directory already used by a live training process.
"${PYTHON_BIN}" - "${RUN_DIRS[@]}" <<'PY'
from pathlib import Path
import sys

targets = {Path(p).resolve() for p in sys.argv[1:]}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().decode().split('\0')
        if 'alf.bin.train' not in args:
            continue
        root = None
        for i, arg in enumerate(args):
            if arg == '--root_dir':
                root = args[i + 1]
            elif arg.startswith('--root_dir='):
                root = arg.split('=', 1)[1]
        if root is None:
            continue
        root = Path(root).expanduser()
        if not root.is_absolute():
            root = (proc / 'cwd').resolve() / root
        if root.resolve() in targets:
            sys.exit(f'Refusing to resume an active run: PID {proc.name}, {root}')
    except (OSError, UnicodeError, IndexError):
        continue
PY

LOG_NAME="out.resume.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
echo "Resuming eight BAFCv6 jobs using saved configurations and existing checkpoints"
echo "  dog:stand GPUs per job: ${TASK_GPUS[0]}"
echo "  dog:walk GPUs per job: ${TASK_GPUS[1]}"
echo "  Budget and hyperparameters: preserved from each run's alf_config.py"
echo "  Dry run: ${DRY_RUN}"
cd "${REPO_ROOT}"
PIDS=()
for i in "${!RUN_DIRS[@]}"; do
    RUN_DIR="${RUN_DIRS[$i]}"
    TASK_INDEX=$((i / ${#SEEDS[@]}))
    SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"
    JOB_GPUS="${TASK_GPUS[$TASK_INDEX]}"
    MASTER_PORT=$((BASE_PORT + i))
    LOG_FILE="${RUN_DIR}/${LOG_NAME}"
    COMMAND=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${RUN_DIR}/alf_config.py"
        --root_dir "${RUN_DIR}"
        --distributed multi-gpu
        --nostore_snapshot
    )
    echo "  ${TASKS[$TASK_INDEX]} seed ${SEED}: ckpt-${CHECKPOINT_STEPS[$i]}, port ${MASTER_PORT}"
    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q nohup ' \
            "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "${JOB_GPUS}" "${MASTER_PORT}"
        printf '%q ' "${COMMAND[@]}"
        printf '> %q 2>&1 < /dev/null &\n' "${LOG_FILE}"
        continue
    fi
    CUDA_VISIBLE_DEVICES="${JOB_GPUS}" MASTER_PORT="${MASTER_PORT}" \
        nohup "${COMMAND[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &
    PIDS+=("$!")
    echo "    PID: $!; log: ${LOG_FILE}"
done
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched or run files changed."
else
    echo "Launched eight resume jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
