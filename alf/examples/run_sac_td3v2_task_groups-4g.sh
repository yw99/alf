#!/bin/bash
# Run six groups sequentially, with all jobs in each group running concurrently:
#   1. SAC dog:trot, seeds 0-3, 800k environment steps each.
#   2. SAC humanoid:stand/walk/run, seed 3, 600k steps each.
#   3. TD3v2 dog:trot, seeds 0-3, 800k steps each.
#   4. TD3v2 humanoid:walk, seeds 0-3, 600k steps each.
#   5. TD3v2 humanoid:run, seeds 0-3, 600k steps each.
#   6. TD3v2 seed 3: humanoid:stand (600k) and dog:walk/stand/run/fetch (800k).
# Failed jobs are reported without retries; the final exit code is 1 if any failed.
# Requires Linux, setsid, Bash 5.1+, and train.py's mapped-worker support.
#
# Usage: bash run_sac_td3v2_task_groups-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Override environment steps for every job
#       --gpus CSV           Visible GPU IDs (default: 0,1,2,3)
#       --worker-gpus CSV    Four logical GPU indices (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-http-port N   First of five HTTP ports (default: 8080)
#       --dry-run            Print commands without creating results or training
#   -h, --help               Show this help message
#
# Results: BASE_DIR/<domain>/<task>/<sac|td3v2>_dmc_4g/seed_<seed>/out.log
# Existing seed directories are rejected to protect previous experiments.
# Each job uses four DDP workers; up to five jobs share the visible GPUs.
# HTTP ports are unique within each group and reused after the group exits.
# DDP rendezvous ports are allocated by train.py. Thread limits default to 1.
# Run inside tmux, then detach with Ctrl-b d to survive an SSH disconnect.
# Example:
#   bash alf/examples/run_sac_td3v2_task_groups-4g.sh --dry-run
#   bash alf/examples/run_sac_td3v2_task_groups-4g.sh --gpus 0,1,2 --worker-gpus 0,1,2,0

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
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
BASE_DIR=/workspace/alf_results
NUM_ENV_STEPS=""
NUM_CHECKPOINTS=10
GPUS=0,1,2,3
WORKER_GPUS=0,1,2,3
BASE_HTTP_PORT=8080
DRY_RUN=False
# Each record is: group algorithm environment seed environment_steps.
JOBS=()
add_jobs() {
    local group="$1" algorithm="$2" task="$3" steps="$4" seed
    shift 4
    for seed in "$@"; do
        JOBS+=("$group $algorithm $task $seed $steps")
    done
}
add_jobs 1 sac   dog:trot       800000 0 1 2 3
add_jobs 2 sac   humanoid:stand 600000 3
add_jobs 2 sac   humanoid:walk  600000 3
add_jobs 2 sac   humanoid:run   600000 3
add_jobs 3 td3v2 dog:trot       800000 0 1 2 3
add_jobs 4 td3v2 humanoid:walk  600000 0 1 2 3
add_jobs 5 td3v2 humanoid:run   600000 0 1 2 3
add_jobs 6 td3v2 humanoid:stand 600000 3
add_jobs 6 td3v2 dog:walk       800000 3
add_jobs 6 td3v2 dog:stand      800000 3
add_jobs 6 td3v2 dog:run        800000 3
add_jobs 6 td3v2 dog:fetch      800000 3

# Algorithm defaults come from sac_dmc_conf.py and td3v2_dmc_conf.py.

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
for algorithm in sac td3v2; do
    conf_file="$SCRIPT_DIR/${algorithm}_dmc_conf.py"
    [[ -f "$conf_file" ]] || die "Config not found: $conf_file"
done
[[ -z "$NUM_ENV_STEPS" || "$NUM_ENV_STEPS" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer"
[[ "$NUM_CHECKPOINTS" =~ ^[1-9][0-9]*$ ]] || die "--checkpoints must be a positive integer"
[[ "$BASE_HTTP_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( BASE_HTTP_PORT <= 65531 )) || die "--base-http-port must be between 1 and 65531"
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

log "Six sequential groups, ${#JOBS[@]} jobs total; jobs within each group run concurrently"
log "Visible GPUs: $GPUS; four DDP workers per job, logical mapping: $WORKER_GPUS"
log "CPU threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OpenBLAS=$OPENBLAS_NUM_THREADS"
log "Steps per job: ${NUM_ENV_STEPS:-per schedule (600000 humanoid, 800000 dog)}; checkpoints: $NUM_CHECKPOINTS; results: $BASE_DIR"

# Check every destination before starting the queue; mkdir below also detects races.
for job in "${JOBS[@]}"; do
    read -r group algorithm task seed task_steps <<< "$job"
    run_dir="$BASE_DIR/${task/:/\/}/${algorithm}_dmc_4g/seed_$seed"
    [[ ! -e "$run_dir" && ! -L "$run_dir" ]] || die "Result path already exists: $run_dir. Choose another --dir."
done

declare -A ACTIVE=()
FAILED_JOBS=()
cleanup() {
    local status=$? pid deadline remaining
    trap - EXIT
    trap '' INT TERM HUP
    if (( ${#ACTIVE[@]} )); then
        log "Stopping current group's jobs"
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
for group in 1 2 3 4 5 6; do
    log "Starting group $group"
    job_index=0
    for job in "${JOBS[@]}"; do
        read -r job_group algorithm task seed task_steps <<< "$job"
        [[ "$job_group" == "$group" ]] || continue
        task_steps="${NUM_ENV_STEPS:-$task_steps}"
        conf_file="$SCRIPT_DIR/${algorithm}_dmc_conf.py"
        run_dir="$BASE_DIR/${task/:/\/}/${algorithm}_dmc_4g/seed_$seed"
        http_port=$((BASE_HTTP_PORT + job_index))
        job_index=$((job_index + 1))
        command=(
            "$PYTHON_BIN" -m alf.bin.train
            --conf "$conf_file"
            --root_dir "$run_dir"
            --conf_param "TrainerConfig.random_seed=$seed"
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
            --conf_param "TrainerConfig.num_checkpoints=$NUM_CHECKPOINTS"
            --conf_param "TrainerConfig.num_env_steps=$task_steps"
            --conf_param "make_ddp_performer.find_unused_parameters=True"
            --conf_param "create_environment.env_name='$task'"
            --distributed multi-gpu
            --worker_gpus "$WORKER_GPUS"
            --port "$http_port"
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
        ACTIVE[$pid]="group $group $algorithm $task seed $seed"
        log "Launched ${ACTIVE[$pid]}, PID $pid; log: $run_dir/out.log"
    done
    if [[ "$DRY_RUN" == True ]]; then
        log "Would wait for all $job_index jobs in group $group to exit, then continue regardless of failures"
        continue
    fi
    group_failures=0
    # All jobs in this group are already running concurrently. Waiting by PID
    # also retrieves cached exit statuses for jobs that finished earlier.
    for pid in "${!ACTIVE[@]}"; do
        if wait "$pid"; then
            log "Completed ${ACTIVE[$pid]}"
        else
            status=$?
            FAILED_JOBS+=("${ACTIVE[$pid]} (exit $status)")
            group_failures=$((group_failures + 1))
            log "FAILED ${ACTIVE[$pid]} (exit $status); remaining jobs will continue"
        fi
        unset 'ACTIVE[$pid]'
    done
    log "Finished group $group ($group_failures failed jobs); continuing queue"
done
log "Queue complete (dry-run: $DRY_RUN); failed jobs: ${#FAILED_JOBS[@]}"
if (( ${#FAILED_JOBS[@]} )); then
    for failure in "${FAILED_JOBS[@]}"; do
        log "  FAILED $failure"
    done
    exit 1
fi
