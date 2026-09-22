#!/usr/bin/env bash
# Four default fresh BAFCv3 -> BAFCv6 continuations, all concurrent on four GPUs.
# Each invocation creates a new timestamped study under /workspace/alf_results.
# Defaults: humanoid:run seeds 0/1 at 90k/105k; override with --jobs.
# UTD 11 and immediate critic reweighting.
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
BASE_DIR=/workspace/alf_results
SOURCE_BASE_DIR=/workspace/server3_copy
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
GPUS=0,1,2,3
DRY_RUN=false
JOBS=(0:90 0:105 1:90 1:105)

usage() {
    cat <<'HELP'
Usage: bash run_humanoid_run_bafcv6_restart_6jobs-4g.sh [options]
  -d, --dir BASE_DIR    Results base (default: /workspace/alf_results)
      --source-base-dir DIR Source runs base (default: /workspace/server3_copy)
      --run-id ID      New study ID (default: current UTC timestamp)
      --gpus CSV       Four distinct logical GPU indices (default: 0,1,2,3)
      --jobs CSV       Distinct seed:checkpoint-k pairs (seeds 0--3)
                       Default: 0:90,0:105,1:90,1:105
                       Checkpoints: 15,30,45,60,75,90,105,120,135,150 (k/rank)
      --dry-run        Print preparation/training commands without creating runs
  -h, --help           Show this help

All jobs start from the selected original BAFCv3 checkpoints, with critic-loss
reweighting enabled immediately, and end at absolute 200000 environment steps per rank.
Existing V6 training directories are never reused or overwritten.
The launcher prepares the runs, starts all jobs in the background, then exits.
HELP
}
while (( $# )); do
    case "$1" in
        -d|--dir|--run-id|--gpus|--source-base-dir|--jobs)
            if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                -d|--dir) BASE_DIR="$2" ;;
                --source-base-dir) SOURCE_BASE_DIR="$2" ;;
                --run-id) RUN_ID="$2" ;;
                --gpus) GPUS="$2" ;;
                --jobs)
                    [[ "$2" =~ ^[0-9]+:[0-9]+(,[0-9]+:[0-9]+)*$ ]] || { echo "Expected comma-separated seed:checkpoint-k pairs" >&2; exit 2; }
                    IFS=, read -r -a JOBS <<< "$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
JOB_COUNT=${#JOBS[@]}
declare -A SEEN_JOBS=()
for job in "${JOBS[@]}"; do
    [[ "$job" =~ ^[0-3]:(15|30|45|60|75|90|105|120|135|150)$ ]] || { echo "Invalid job: $job" >&2; exit 2; }
    [[ -z "${SEEN_JOBS[$job]:-}" ]] || { echo "Duplicate job: $job" >&2; exit 2; }
    SEEN_JOBS[$job]=1
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid run ID" >&2; exit 2; }
[[ "$GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || { echo "Expected four GPU indices" >&2; exit 2; }
IFS=, read -r -a GPU_IDS <<< "$GPUS"
for ((i=0; i<4; i++)); do
    for ((j=0; j<i; j++)); do
        [[ "${GPU_IDS[$i]}" != "${GPU_IDS[$j]}" ]] || { echo "GPU indices must be distinct" >&2; exit 2; }
    done
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python interpreter not executable: $PYTHON_BIN" >&2; exit 1; }
cd "$REPO_ROOT"
RESULTS_PARENT="$BASE_DIR/humanoid_run/bafcv6_restart"
ROOT_DIR="$RESULTS_PARENT/$RUN_ID"
[[ ! -e "$ROOT_DIR" ]] || { echo "Fresh study directory already exists: $ROOT_DIR" >&2; exit 1; }

# Sets NAME, RUN_DIR and COMMAND for one seed/checkpoint combination.
build_command() {
    local seed horizon source
    IFS=: read -r seed horizon <<< "$1"
    NAME="humanoid_run_s${seed}_${horizon}k_utd11_reweight"
    RUN_DIR="$ROOT_DIR/$NAME"
    source="$SOURCE_BASE_DIR/humanoid_run_bafcv3_rtT_s${seed}/train/algorithm/ckpt-$((horizon * 1001))"
    [[ -f "$source" ]] || { echo "Source checkpoint missing: $source" >&2; exit 1; }
    COMMAND=("$PYTHON_BIN" -m alf.bin.train_bafcv6_restart
        --source-checkpoint "$source" --root-dir "$RUN_DIR"
        --critic-reweighting-solver lbfgs_logits
        --critic-reweighting-solver-iters 1 --critic-reweighting-num-feature-coords 32
        --critic-reweighting-num-target-obs 128 --critic-reweighting-target-obs-cache-size 512
        --critic-reweighting-max-weight 10 --critic-reweighting-ridge 1e-4
        --final-env-steps-per-rank 200000
        --worker-gpus "$GPUS")
}

echo "Fresh $JOB_COUNT-job study: $ROOT_DIR"
echo "All jobs run concurrently on GPUs $GPUS, with one rank per GPU per job."
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

# Read-only preflight of the entire grid before preparing or starting any job.
"$PYTHON_BIN" - "$SOURCE_BASE_DIR" "${JOBS[@]}" <<'PREFLIGHT'
from pathlib import Path
import sys
import torch
from alf.bin.evaluate_bafcv3_checkpoints import preconfig
from alf.utils.bafcv6_restart import fingerprint_inputs
base = Path(sys.argv[1])
for job in sys.argv[2:]:
    seed, horizon = map(int, job.split(':'))
    run = base / f'humanoid_run_bafcv3_rtT_s{seed}'
    hints = preconfig(run / 'alf_config.py')
    if hints.get('create_environment.env_name') != 'humanoid:run' or hints.get('TrainerConfig.random_seed') != seed:
        raise ValueError(f'Wrong task/seed: {run}')
    if hints.get('BafcAlgorithmV3.critic_utd') != 11 or hints.get('TrainerConfig.num_updates_per_train_iter') != 12:
        raise ValueError(f'Expected critic UTD 11 and 12 updates: {run}')
    path = run / 'train/algorithm' / f'ckpt-{horizon * 1001}'
    fingerprint_inputs(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if int(checkpoint['trainer_progress']['_env_steps']) != horizon * 1000:
        raise ValueError(f'Wrong environment-step count: {path}')

PREFLIGHT

mkdir -p "$RESULTS_PARENT"
exec 9>"$RESULTS_PARENT/launcher.lock"
flock -n 9 || { echo "A launcher is already running under $RESULTS_PARENT" >&2; exit 1; }
mkdir "$ROOT_DIR"
echo "$$" > "$ROOT_DIR/launcher.pid"
printf 'job\tpid\tlog\n' > "$ROOT_DIR/launches.tsv"

# Prepare all empty destinations before opening out.log or starting any trainer.
# --resume below continues these new manifests, not an old V6 checkpoint.
for job in "${JOBS[@]}"; do
    build_command "$job"
    echo "Preparing fresh run: $NAME"
    "${COMMAND[@]}" --prepare-only > "$ROOT_DIR/prepare.$NAME.log" 2>&1
done
"$PYTHON_BIN" - "$ROOT_DIR" "$JOB_COUNT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
runs = []
for path in sorted(root.glob('humanoid_run_*/restart_manifest.json')):
    manifest = json.loads(path.read_text())
    assert not (path.parent / 'train/algorithm').exists()
    runs.append(dict(name=path.parent.name, root_dir=str(path.parent),
                     source_checkpoint=manifest['source_checkpoint'],
                     critic_utd=11, reweighting={k: v for k, v in manifest.items() if k.startswith('critic_reweighting_')},
                     seed=manifest['seed'],
                     out_log=str(path.parent / 'out.log')))
assert len(runs) == int(sys.argv[2])
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
echo "Launched $JOB_COUNT V6 4-GPU jobs: ${PIDS[*]}"
echo "Launcher is not waiting for completion."
echo "PIDs and logs: $ROOT_DIR/launches.tsv"
echo "To monitor: tail -n 30 -f $ROOT_DIR/humanoid_run_*/out.log"
