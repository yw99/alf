#!/usr/bin/env bash
# Twelve fresh BAFCv3 -> TR2 continuations across eight GPUs.
# Seeds 0/1 share the first four GPUs; seeds 2/3 share the last four.
# Each invocation creates a new timestamped study under /workspace/alf_results.
# Sources: dog:run seeds 0/1/2/3; UTD 11 at 120k/140k/160k; skipping on.
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
BASE_DIR=/workspace/alf_results
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
GPUS=0,1,2,3,4,5,6,7
DRY_RUN=false
JOBS=(0:120:11 0:140:11 0:160:11 1:120:11 1:140:11 1:160:11
      2:120:11 2:140:11 2:160:11 3:120:11 3:140:11 3:160:11)

usage() {
    cat <<'HELP'
Usage: bash run_dog_run_bafcv3_tr2_restart_12jobs-8g.sh [options]
  -d, --dir BASE_DIR    Results base (default: /workspace/alf_results)
      --run-id ID      New study ID (default: current UTC timestamp)
      --gpus CSV       Eight distinct logical GPU indices (default: 0,1,2,3,4,5,6,7)
                       First four for seeds 0/1; last four for seeds 2/3
      --dry-run        Print preparation/training commands without creating runs
  -h, --help           Show this help

All twelve jobs start from the selected original BAFCv3 checkpoints, with a fresh
TR2 calibration, and end at absolute 200000 environment steps per rank.
Existing TR2 training directories are never reused or overwritten.
The launcher prepares the runs, starts all twelve in the background, then exits.
HELP
}
while (( $# )); do
    case "$1" in
        -d|--dir|--run-id|--gpus)
            if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                -d|--dir) BASE_DIR="$2" ;;
                --run-id) RUN_ID="$2" ;;
                --gpus) GPUS="$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid run ID" >&2; exit 2; }
[[ "$GPUS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){7}$ ]] || { echo "Expected eight GPU indices" >&2; exit 2; }
IFS=, read -r -a GPU_IDS <<< "$GPUS"
for ((i=0; i<8; i++)); do
    for ((j=0; j<i; j++)); do
        [[ "${GPU_IDS[$i]}" != "${GPU_IDS[$j]}" ]] || { echo "GPU indices must be distinct" >&2; exit 2; }
    done
done
SEED_GPUS=(
    "${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]},${GPU_IDS[3]}"
    "${GPU_IDS[4]},${GPU_IDS[5]},${GPU_IDS[6]},${GPU_IDS[7]}"
)
[[ -x "$PYTHON_BIN" ]] || { echo "Python interpreter not executable: $PYTHON_BIN" >&2; exit 1; }
cd "$REPO_ROOT"
RESULTS_PARENT="$BASE_DIR/dog_run/bafcv3_tr2_restart"
ROOT_DIR="$RESULTS_PARENT/$RUN_ID"
[[ ! -e "$ROOT_DIR" ]] || { echo "Fresh study directory already exists: $ROOT_DIR" >&2; exit 1; }

# Sets NAME, RUN_DIR and COMMAND for one seed/checkpoint/UTD combination.
build_command() {
    local seed horizon utd source source_root job_gpus
    IFS=: read -r seed horizon utd <<< "$1"
    NAME="dog_run_s${seed}_${horizon}k_utd${utd}_on"
    RUN_DIR="$ROOT_DIR/$NAME"
    if (( seed < 2 )); then
        source_root=/workspace/server2_copy
    else
        source_root=/workspace/server_copy
    fi
    job_gpus="${SEED_GPUS[$((seed / 2))]}"
    source="${source_root}/dog_run_bafcv3_rtT_s${seed}/train/algorithm/ckpt-$((horizon * 1001))"
    [[ -f "$source" ]] || { echo "Source checkpoint missing: $source" >&2; exit 1; }
    COMMAND=("$PYTHON_BIN" -m alf.bin.train_bafcv3_tr2_restart
        --source-checkpoint "$source" --root-dir "$RUN_DIR"
        --critic-utd "$utd" --rollout-skipping on
        --threshold-quantile 0.33 --calibration-repetitions 100
        --calibration-seed 20260917 --final-env-steps-per-rank 200000
        --worker-gpus "$job_gpus")
}

echo "Fresh twelve-job study: $ROOT_DIR"
echo "Seeds 0/1 use GPUs ${SEED_GPUS[0]}; seeds 2/3 use GPUs ${SEED_GPUS[1]}."
echo "All twelve jobs run concurrently, with four ranks per job."
echo "CPU threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OpenBLAS=$OPENBLAS_NUM_THREADS"
if [[ "$DRY_RUN" == true ]]; then
    for job in "${JOBS[@]}"; do
        build_command "$job"
        printf '%q ' "${COMMAND[@]}" --prepare-only
        printf '\n'
        printf 'nohup '
        printf '%q ' "${COMMAND[@]}" --resume
        printf '< /dev/null > %q 2>&1 9>&- &\n' "$RUN_DIR/out.log"
    done
    echo "Dry run: nothing created or launched."
    exit 0
fi

mkdir -p "$RESULTS_PARENT"
exec 9>"$RESULTS_PARENT/launcher.lock"
flock -n 9 || { echo "A twelve-job launcher is already running under $RESULTS_PARENT" >&2; exit 1; }
mkdir "$ROOT_DIR"
echo "$$" > "$ROOT_DIR/launcher.pid"
printf 'job\tpid\tlog\n' > "$ROOT_DIR/launches.tsv"

# Prepare all empty destinations before opening out.log or starting any trainer.
# --resume below continues these new manifests, not an old TR2 checkpoint.
for job in "${JOBS[@]}"; do
    build_command "$job"
    echo "Preparing fresh run: $NAME"
    "${COMMAND[@]}" --prepare-only > "$ROOT_DIR/prepare.$NAME.log" 2>&1
done
"$PYTHON_BIN" - "$ROOT_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
runs = []
for path in sorted(root.glob('dog_run_*/restart_manifest.json')):
    manifest = json.loads(path.read_text())
    assert not (path.parent / 'train/algorithm').exists()
    runs.append(dict(name=path.parent.name, root_dir=str(path.parent),
                     source_checkpoint=manifest['source_checkpoint'],
                     critic_utd=manifest['critic_utd'], seed=manifest['seed'],
                     out_log=str(path.parent / 'out.log')))
assert len(runs) == 12
(root / 'experiment_manifest.json').write_text(json.dumps(dict(
    start_mode='fresh_from_original_bafcv3', execution_mode='concurrent',
    runs=runs), indent=2) + '\n')
PY

PIDS=()
for job in "${JOBS[@]}"; do
    build_command "$job"
    # Detach terminal I/O and close the preparation lock in the child.
    nohup "${COMMAND[@]}" --resume < /dev/null > "$RUN_DIR/out.log" 2>&1 9>&- &
    pid=$!
    PIDS+=("$pid")
    printf '%s\t%s\t%s\n' "$NAME" "$pid" "$RUN_DIR/out.log" >> "$ROOT_DIR/launches.tsv"
    echo "Started $NAME: PID $pid; log $RUN_DIR/out.log"
done
echo "Launched twelve TR2 4-GPU jobs: ${PIDS[*]}"
echo "Launcher is not waiting for completion."
echo "PIDs and logs: $ROOT_DIR/launches.tsv"
echo "To monitor: tail -n 30 -f $ROOT_DIR/dog_run_*/out.log"
