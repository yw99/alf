#!/bin/bash
# Run TD3v2 tasks in order: dog:run, dog:fetch.
# Seeds 0-2 run concurrently within each task; wait for all to exit, then continue.
# Failed seeds are reported without retries; the final exit code is 1 if any failed.
# Requires Linux, setsid, Bash 5.1+, and train.py's mapped-worker support.
#
# Usage: bash run_td3v2_tasks_seed012-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Override steps per seed for all tasks (default: 800000)
#       --gpus CSV           Visible GPU IDs (default: 0,1,2,3)
#       --worker-gpus CSV    Four logical GPU indices (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-http-port N   First of three HTTP ports (default: 8080)
#       --dry-run            Print commands without creating results or training
#   -h, --help               Show this help message
#
# Results: BASE_DIR/<domain>/<task>/td3v2_dmc_4g/seed_<seed>/out.log
# Existing seed directories are rejected to protect previous experiments.
# DDP rendezvous ports are allocated by train.py. Thread limits default to 1.
# Run inside tmux, then detach with Ctrl-b d to survive an SSH disconnect.
# Example:
#   bash alf/examples/run_td3v2_tasks_seed012-4g.sh --dry-run
#   bash alf/examples/run_td3v2_tasks_seed012-4g.sh --gpus 0,1,2 --worker-gpus 0,1,2,0

set -euo pipefail
# Keep background job PIDs as process-group leaders when invoking setsid.
set +m

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/td3v2_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
BASE_DIR=/workspace/alf_results
NUM_ENV_STEPS=""
NUM_CHECKPOINTS=10
GPUS=0,1,2,3
WORKER_GPUS=0,1,2,3
BASE_HTTP_PORT=8080
DRY_RUN=False
TASKS=(dog:run dog:fetch)
TASK_STEPS=(800000 800000)
SEEDS=(0 1 2)

# Algorithm defaults come from td3v2_dmc_conf.py, as in run_td3v2_seeds.sh.

die() { echo "Error: $*" >&2; exit 1; }
log() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            sed -n '/^# Usage:/,/^# Example:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
            exit 0 ;;
        --dry-run) DRY_RUN=True; shift ;;
        -d|--dir|-n|--steps|--gpus|--worker-gpus|--checkpoints|--base-http-port)
            [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
            case "$1" in
                -d|--dir) BASE_DIR="$2" ;;
                -n|--steps) NUM_ENV_STEPS="$2" ;;
                --gpus) GPUS="$2" ;;
                --worker-gpus) WORKER_GPUS="$2" ;;
                --checkpoints) NUM_CHECKPOINTS="$2" ;;
                --base-http-port) BASE_HTTP_PORT="$2" ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done

(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) || die "Bash 5.1+ is required"
command -v setsid >/dev/null || die "setsid is required"
[[ -x "$PYTHON_BIN" ]] || die "Python not executable: $PYTHON_BIN (set PYTHON_BIN)"
[[ -f "$CONF_FILE" ]] || die "Config not found: $CONF_FILE"
[[ -z "$NUM_ENV_STEPS" || "$NUM_ENV_STEPS" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer"
[[ "$NUM_CHECKPOINTS" =~ ^[1-9][0-9]*$ ]] || die "--checkpoints must be a positive integer"
[[ "$BASE_HTTP_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( BASE_HTTP_PORT <= 65533 )) || die "--base-http-port must be between 1 and 65533"
[[ "$GPUS" =~ ^[[:alnum:]_-]+(,[[:alnum:]_-]+)*$ ]] || die "--gpus must be a nonempty CSV of device IDs"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ "$WORKER_GPUS" =~ ^[0-9]+(,[0-9]+){3}$ ]] || die "--worker-gpus requires exactly four logical GPU indices"
IFS=',' read -r -a WORKER_IDS <<< "$WORKER_GPUS"
for index in "${WORKER_IDS[@]}"; do
    [[ ${#index} -le 8 ]] && (( 10#$index < ${#GPU_IDS[@]} )) || die "Worker index $index is outside --gpus $GPUS"
done
# Resolve relative paths before switching to the repository directory.
[[ "$BASE_DIR" == /* ]] || BASE_DIR="$PWD/$BASE_DIR"
[[ "$PYTHON_BIN" == /* ]] || PYTHON_BIN="$PWD/$PYTHON_BIN"

log "Tasks: ${TASKS[*]} (sequential); seeds: ${SEEDS[*]} (concurrent per task)"
log "Visible GPUs: $GPUS; four DDP workers per seed, logical mapping: $WORKER_GPUS"
log "CPU threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OpenBLAS=$OPENBLAS_NUM_THREADS"
log "Steps per seed: ${NUM_ENV_STEPS:-800000}; checkpoints: $NUM_CHECKPOINTS; results: $BASE_DIR"

# Check every destination before starting the queue; mkdir below also detects races.
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_dir="$BASE_DIR/${task/:/\/}/td3v2_dmc_4g/seed_$seed"
        [[ ! -e "$run_dir" && ! -L "$run_dir" ]] || die "Result path already exists: $run_dir. Choose another --dir."
    done
done

declare -A ACTIVE=()
FAILED_JOBS=()
cleanup() {
    local status=$? pid deadline remaining
    trap - EXIT
    trap '' INT TERM HUP
    if (( ${#ACTIVE[@]} )); then
        log "Stopping current task's jobs"
        for pid in "${!ACTIVE[@]}"; do
            # The mapped coordinator handles its separately grouped workers.
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        done
        deadline=$((SECONDS + 10))
        while (( SECONDS < deadline )); do
            remaining=0
            for pid in "${!ACTIVE[@]}"; do
                if kill -0 -- "-$pid" 2>/dev/null; then remaining=1; fi
            done
            (( remaining )) || break
            sleep 0.2
        done
        for pid in "${!ACTIVE[@]}"; do
            kill -KILL -- "-$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        done
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

cd "$REPO_ROOT"
for task_index in "${!TASKS[@]}"; do
    task="${TASKS[$task_index]}"
    task_steps="${NUM_ENV_STEPS:-${TASK_STEPS[$task_index]}}"
    log "Starting task $task ($task_steps environment steps per seed)"
    for seed in "${SEEDS[@]}"; do
        run_dir="$BASE_DIR/${task/:/\/}/td3v2_dmc_4g/seed_$seed"
        command=(
            "$PYTHON_BIN" -m alf.bin.train
            --conf "$CONF_FILE"
            --root_dir "$run_dir"
            --conf_param "TrainerConfig.random_seed=$seed"
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
            --conf_param "TrainerConfig.num_checkpoints=$NUM_CHECKPOINTS"
            --conf_param "TrainerConfig.num_env_steps=$task_steps"
            --conf_param "make_ddp_performer.find_unused_parameters=True"
            --conf_param "create_environment.env_name='$task'"
            --distributed multi-gpu
            --worker_gpus "$WORKER_GPUS"
            --port "$((BASE_HTTP_PORT + seed))"
        )
        if [[ "$DRY_RUN" == True ]]; then
            printf 'OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q CUDA_VISIBLE_DEVICES=%q ' \
                "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$GPUS"
            printf '%q ' setsid "${command[@]}"
            printf '> %q 2>&1 &\n' "$run_dir/out.log"
            continue
        fi
        mkdir -p "$(dirname "$run_dir")"
        mkdir "$run_dir"
        CUDA_VISIBLE_DEVICES="$GPUS" setsid "${command[@]}" > "$run_dir/out.log" 2>&1 &
        pid=$!
        ACTIVE[$pid]="$task seed $seed"
        log "Launched ${ACTIVE[$pid]}, PID $pid; log: $run_dir/out.log"
    done
    if [[ "$DRY_RUN" == True ]]; then
        log "Would wait for all ${#SEEDS[@]} $task seeds to exit, then continue regardless of failures"
        continue
    fi
    task_failures=0
    # All seeds are already running concurrently. Waiting by PID also retrieves
    # cached exit statuses for seeds that finished before we reached this loop.
    for pid in "${!ACTIVE[@]}"; do
        if wait "$pid"; then
            log "Completed ${ACTIVE[$pid]}"
        else
            status=$?
            FAILED_JOBS+=("${ACTIVE[$pid]} (exit $status)")
            task_failures=$((task_failures + 1))
            log "FAILED ${ACTIVE[$pid]} (exit $status); remaining seeds will continue"
        fi
        unset 'ACTIVE[$pid]'
    done
    log "Finished task $task ($task_failures failed seeds); continuing queue"
done
log "Queue complete (dry-run: $DRY_RUN); failed seeds: ${#FAILED_JOBS[@]}"
if (( ${#FAILED_JOBS[@]} )); then
    for failure in "${FAILED_JOBS[@]}"; do
        log "  FAILED $failure"
    done
    exit 1
fi
