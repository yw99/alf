#!/usr/bin/env bash
# Four optional replay-reconstruction continuations; one four-worker job per seed.
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
ALGORITHM=sac
BASE_DIR=/workspace/alf_results
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
GPUS=0,1,2,3
DRY_RUN=false
RESUME=false
RECONSTRUCT_ONLY=false
PREPARE_ONLY=false
usage() {
    echo "Usage: $0 [-d BASE_DIR] [--run-id ID] [--gpus CSV] [--dry-run] [--prepare-only] [--reconstruct-only] [--resume]"
    echo "Seeds 0-3 run concurrently, each using four workers; final total horizon is 800k."
    echo "--resume requires --run-id for a previously prepared study."
}
while (( $# )); do
    case "$1" in
        -d|--dir|--run-id|--gpus)
            (( $# >= 2 )) || { usage; exit 2; }
            case "$1" in
                -d|--dir) BASE_DIR="$2" ;;
                --run-id) RUN_ID="$2" ;;
                --gpus) GPUS="$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        --prepare-only) PREPARE_ONLY=true; shift ;;
        --reconstruct-only) RECONSTRUCT_ONLY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid run ID" >&2; exit 2; }
ROOT_DIR="$BASE_DIR/dog_walk/${ALGORITHM}_reconstruction/$RUN_ID"
cd "$REPO_ROOT"
build_command() {
    local seed="$1" source
    if [[ "$ALGORITHM" == sac ]]; then
        source="/workspace/server3_copy/dog_walk_sac_s$seed"
    elif (( seed < 2 )); then
        source="/workspace/server2_copy/dog_rlpd_s$seed"
    else
        source="/workspace/server_copy/dog_rlpd_s$seed"
    fi
    COMMAND=("$PYTHON_BIN" -m alf.bin.train_replay_reconstruction
        --source-run "$source" --root-dir "$ROOT_DIR/seed_$seed"
        --worker-gpus "$GPUS" --final-env-steps 800000)
}
if [[ "$DRY_RUN" == true ]]; then
    for seed in 0 1 2 3; do
        build_command "$seed"
        extra=(); [[ "$RESUME" == false ]] || extra+=(--resume)
        [[ "$RECONSTRUCT_ONLY" == false ]] || extra+=(--reconstruct-only)
        "${COMMAND[@]}" "${extra[@]}" --dry-run
    done
    exit 0
fi
if [[ "$RESUME" == false && -e "$ROOT_DIR" ]]; then
    echo "Study already exists; use --resume and its --run-id" >&2; exit 1
fi
[[ "$RESUME" == false || -d "$ROOT_DIR" ]] || { echo "Study does not exist" >&2; exit 1; }
mkdir -p "$ROOT_DIR"
exec 9>"$ROOT_DIR/launcher.lock"
flock -n 9 || { echo "Launcher already active" >&2; exit 1; }
# Refuse duplicate jobs from an earlier invocation of this launcher.
if [[ -f "$ROOT_DIR/launches.tsv" ]]; then
    while read -r seed pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Job still active: seed $seed PID $pid" >&2; exit 1
        fi
    done < "$ROOT_DIR/launches.tsv"
fi
for seed in 0 1 2 3; do
    build_command "$seed"
    extra=(); [[ "$RESUME" == false ]] || extra+=(--resume)
    "${COMMAND[@]}" "${extra[@]}" --prepare-only > "$ROOT_DIR/prepare.$seed.log" 2>&1
done
[[ "$PREPARE_ONLY" == false ]] || exit 0
: > "$ROOT_DIR/launches.tsv"
for seed in 0 1 2 3; do
    build_command "$seed"
    extra=(--resume); [[ "$RECONSTRUCT_ONLY" == false ]] || extra+=(--reconstruct-only)
    nohup "${COMMAND[@]}" "${extra[@]}" < /dev/null >> "$ROOT_DIR/seed_$seed/out.log" 2>&1 9>&- &
    pid=$!
    printf '%s %s\n' "$seed" "$pid" >> "$ROOT_DIR/launches.tsv"
    echo "Launched $ALGORITHM seed $seed: PID $pid; $ROOT_DIR/seed_$seed/out.log"
done
