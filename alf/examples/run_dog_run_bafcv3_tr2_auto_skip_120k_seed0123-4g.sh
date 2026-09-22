#!/usr/bin/env bash
# Fresh dog:run automatic-gate continuations, one four-GPU DDP job per seed.
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
SOURCE01=/workspace/server2_copy
SOURCE23=/workspace/server_copy
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
GPUS=0,1,2,3
SEEDS=0,1,2,3
FINAL_STEPS=200000
DRY_RUN=false
usage() {
    cat <<'HELP'
Usage: bash run_dog_run_bafcv3_tr2_auto_skip_120k_seed0123-4g.sh [options]
  -d, --dir PATH                  Results base (default /workspace/alf_results)
      --source-root-01 PATH       Source root for seeds 0/1
      --source-root-23 PATH       Source root for seeds 2/3
      --seeds CSV                 Subset of 0,1,2,3 (default all four)
      --gpus CSV                  Four distinct GPU indices (default 0,1,2,3)
      --run-id ID                 Fresh study ID (default UTC timestamp)
      --final-env-steps-per-rank N Absolute endpoint (default 200000)
      --dry-run                   Validate inputs and print; write nothing
  -h, --help                      Show help
All selected seeds are preflighted and prepared before any training starts.
Each job uses all four GPUs; the jobs run concurrently. Existing TR2 restart
comparison jobs are launched separately with the existing restart pipeline.
HELP
}
while (( $# )); do
    case "$1" in
        -d|--dir|--source-root-01|--source-root-23|--seeds|--gpus|--run-id|--final-env-steps-per-rank)
            (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                -d|--dir) BASE_DIR="$2" ;;
                --source-root-01) SOURCE01="$2" ;;
                --source-root-23) SOURCE23="$2" ;;
                --seeds) SEEDS="$2" ;;
                --gpus) GPUS="$2" ;;
                --run-id) RUN_ID="$2" ;;
                --final-env-steps-per-rank) FINAL_STEPS="$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
ROOT_DIR="$BASE_DIR/dog_run/bafcv3_tr2_auto_skip/$RUN_ID"
[[ ! -e "$ROOT_DIR" ]] || { echo "Output already exists: $ROOT_DIR" >&2; exit 1; }
cd "$REPO_ROOT"
# Read-only validation of the entire study, before preparation or launch.
"$PYTHON_BIN" - "$SOURCE01" "$SOURCE23" "$SEEDS" "$GPUS" "$FINAL_STEPS" <<'PY'
import sys
from pathlib import Path
import torch
from alf.bin.evaluate_bafcv3_checkpoints import preconfig
from alf.utils.bafcv3_restart import fingerprint_inputs
root01, root23, seed_csv, gpu_csv, final = sys.argv[1:]
seeds = [int(s) for s in seed_csv.split(',')]
gpus = [int(g) for g in gpu_csv.split(',')]
if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= {0,1,2,3}:
    raise ValueError('Expected unique seeds from 0,1,2,3')
if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0:
    raise ValueError('Expected four distinct nonnegative GPU indices')
if int(final) <= 120000:
    raise ValueError('Final horizon must exceed 120000')
for seed in seeds:
    run = Path(root01 if seed < 2 else root23) / f'dog_run_bafcv3_rtT_s{seed}'
    source = run / 'train/algorithm/ckpt-120120'
    fingerprint_inputs(source)
    hints = preconfig(run / 'alf_config.py')
    if hints.get('create_environment.env_name') != 'dog:run' or hints.get('TrainerConfig.random_seed') != seed:
        raise ValueError(f'Wrong task or seed in {run}')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if int(checkpoint['trainer_progress']['_env_steps']) != 120000:
        raise ValueError(f'Not a 120k environment-step checkpoint: {source}')
    print(f'Preflight passed: seed {seed}: {source}')
PY
IFS=, read -r -a SEED_IDS <<< "$SEEDS"
build_command() {
    local seed="$1" source_root="$SOURCE01"
    if (( seed >= 2 )); then source_root="$SOURCE23"; fi
    NAME="dog_run_s${seed}_120k_utd11_auto"
    RUN_DIR="$ROOT_DIR/$NAME"
    COMMAND=("$PYTHON_BIN" -m alf.bin.train_bafcv3_tr2_auto_skip
        --source-checkpoint "$source_root/dog_run_bafcv3_rtT_s${seed}/train/algorithm/ckpt-120120"
        --root-dir "$RUN_DIR" --critic-utd 11 --rollout-skipping off
        --threshold-quantile .33 --calibration-repetitions 100
        --calibration-seed 20260917 --worker-gpus "$GPUS"
        --final-env-steps-per-rank "$FINAL_STEPS")
}
if [[ "$DRY_RUN" == true ]]; then
    for seed in "${SEED_IDS[@]}"; do
        build_command "$seed"
        printf '%q ' "${COMMAND[@]}" --prepare-only; printf '\n'
        printf '%q ' "${COMMAND[@]}" --resume; printf '\n'
    done
    exit 0
fi
mkdir -p "$(dirname "$ROOT_DIR")"
mkdir "$ROOT_DIR"
printf 'job\tpid\tlog\n' > "$ROOT_DIR/launches.tsv"
for seed in "${SEED_IDS[@]}"; do
    build_command "$seed"
    "${COMMAND[@]}" --prepare-only > "$ROOT_DIR/prepare.$NAME.log" 2>&1
done
"$PYTHON_BIN" - "$ROOT_DIR" "$GPUS" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
runs = [json.loads(p.read_text()) for p in sorted(root.glob('dog_run_*/restart_manifest.json'))]
(root / 'experiment_manifest.json').write_text(json.dumps(dict(
    experiment='bafcv3_tr2_auto_skip', execution_mode='concurrent',
    worker_gpus=sys.argv[2], runs=runs), indent=2) + '\n')
PY
for seed in "${SEED_IDS[@]}"; do
    build_command "$seed"
    nohup "${COMMAND[@]}" --resume < /dev/null > "$RUN_DIR/out.log" 2>&1 &
    pid=$!
    printf '%s\t%s\t%s\n' "$NAME" "$pid" "$RUN_DIR/out.log" >> "$ROOT_DIR/launches.tsv"
    echo "Started $NAME: PID $pid; log $RUN_DIR/out.log"
done
