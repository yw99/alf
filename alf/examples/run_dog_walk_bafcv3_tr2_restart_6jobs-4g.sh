#!/usr/bin/env bash
# Resume the four UTD=11 dog:walk TR2 runs in place, concurrently on four GPUs.
# Extend 600k aggregate environment steps (150k/rank) to 800k (200k/rank).
# Keep the original manifests and calibrated checkpoint identity unchanged.
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
RUN_ID=20260917T054927Z
GPUS=0,1,2,3
DRY_RUN=false
JOBS=(0:75:11 0:105:11 1:75:11 1:105:11)

usage() {
    cat <<'HELP'
Usage: bash run_dog_walk_bafcv3_tr2_restart_6jobs-4g.sh [options]
  -d, --dir BASE_DIR    Results base (default: /workspace/alf_results)
      --run-id ID      Existing study ID (default: 20260917T054927Z)
      --gpus CSV       Four distinct logical GPU indices (default: 0,1,2,3)
      --dry-run        Verify checkpoints and print commands without writing files
  -h, --help           Show this help

Resume only the four UTD=11 runs (seeds 0/1, original checkpoints 75k/105k).
The absolute target is 800000 aggregate environment steps: 200000 per rank.
Existing TR2 checkpoints, optimizer, replay and calibration are restored in place.
Original manifests and out.log are retained; new logs use out.800k.log.
The launcher starts all four jobs in the background, then exits.
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
[[ "$GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || { echo "Expected four GPU indices" >&2; exit 2; }
IFS=, read -r -a GPU_IDS <<< "$GPUS"
for ((i=0; i<4; i++)); do
    for ((j=0; j<i; j++)); do
        [[ "${GPU_IDS[$i]}" != "${GPU_IDS[$j]}" ]] || { echo "GPU indices must be distinct" >&2; exit 2; }
    done
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python interpreter not executable: $PYTHON_BIN" >&2; exit 1; }
cd "$REPO_ROOT"
RESULTS_PARENT="$BASE_DIR/dog_walk/bafcv3_tr2_restart"
ROOT_DIR="$RESULTS_PARENT/$RUN_ID"
[[ -d "$ROOT_DIR" ]] || { echo "Study directory missing: $ROOT_DIR" >&2; exit 1; }
# The native trainer also needs companion tools such as Ninja on PATH.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

if [[ "$DRY_RUN" != true ]]; then
    exec 9>"$ROOT_DIR/continuation_800k.lock"
    flock -n 9 || { echo "An 800k continuation is already running" >&2; exit 1; }
fi

# Validate the whole grid before writing configuration or launching any worker.
# The original settings fingerprint is embedded in calibration checkpoint state;
# changing its manifest horizon would break faithful native checkpoint restoration.
"$PYTHON_BIN" - "$ROOT_DIR" "$DRY_RUN" "${JOBS[@]}" <<'PREFLIGHT'
import json
import sys
from pathlib import Path
import torch
from alf.utils.bafcv3_restart import validate_resume_inputs, verify_config_snapshot

root = Path(sys.argv[1]).resolve()
dry_run = sys.argv[2] == 'true'
runs = []
for job in sys.argv[3:]:
    seed, horizon, utd = map(int, job.split(':'))
    run = root / f'dog_walk_s{seed}_{horizon}k_utd{utd}_on'
    manifest = json.loads((run / 'restart_manifest.json').read_text())
    expected = dict(task='dog:walk', seed=seed, critic_utd=11,
                    rollout_skipping=True, source_env_steps=horizon * 1000,
                    final_env_steps_per_rank=150000, root_dir=str(run))
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'{run}: expected {key}={value!r}')
    verify_config_snapshot(manifest)
    checkpoints = [p for p in (run / 'train/algorithm').glob('ckpt-*')
                   if p.name[5:].isdigit()]
    if not checkpoints:
        raise ValueError(f'No TR2 checkpoint in {run}')
    checkpoint = max(checkpoints, key=lambda p: int(p.name[5:]))
    validate_resume_inputs(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    steps = int(state['trainer_progress']['_env_steps'])
    if not 150000 <= steps < 200000:
        raise ValueError(f'{run}: expected progress in [150000, 200000), got {steps}')
    calibration = state['algorithm']['_rl_algorithm._bafc_runtime.restart_calibration']
    if calibration['settings_fingerprint'] != manifest['settings_fingerprint']:
        raise ValueError(f'Checkpoint calibration does not match {run}')
    conf = (f'import alf\nalf.import_config({str(run / "restart_conf.py")!r})\n'
            '# Aggregate budget; ALF divides by four workers.\n'
            'alf.config1("TrainerConfig.num_env_steps", 800000, '
            'override_all=True, raise_if_used=False)\n')
    runs.append((run, checkpoint, steps, conf))
    print(f'{run.name}: {checkpoint.name}, {steps}/rank ({steps * 4} total) -> 800000 total')

if not dry_run:
    for run, checkpoint, steps, conf in runs:
        (run / 'continuation_800k_conf.py').write_text(conf)
    (root / 'continuation_800k_manifest.json').write_text(json.dumps(dict(
        start_mode='native_tr2_resume', final_env_steps=800000,
        final_env_steps_per_rank=200000, world_size=4,
        runs=[dict(root_dir=str(run), checkpoint=str(checkpoint),
                   resumed_env_steps_per_rank=steps) for run, checkpoint, steps, _ in runs]
    ), indent=2) + '\n')
PREFLIGHT

build_command() {
    local seed horizon utd
    IFS=: read -r seed horizon utd <<< "$1"
    NAME="dog_walk_s${seed}_${horizon}k_utd${utd}_on"
    RUN_DIR="$ROOT_DIR/$NAME"
    COMMAND=("$PYTHON_BIN" -m alf.bin.train
        --conf "$RUN_DIR/continuation_800k_conf.py" --root_dir "$RUN_DIR"
        --distributed=multi-gpu --worker_gpus "$GPUS"
        --nostore_snapshot --port=0 --alsologtostderr)
}

echo "Resume four UTD=11 jobs: $ROOT_DIR"
echo "Target: 800000 total environment steps (200000/rank), GPUs $GPUS."
if [[ "$DRY_RUN" == true ]]; then
    for job in "${JOBS[@]}"; do
        build_command "$job"
        echo "Would write $RUN_DIR/continuation_800k_conf.py (aggregate budget 800000)."
        printf 'nohup '
        printf '%q ' "${COMMAND[@]}"
        printf '< /dev/null >> %q 2>&1 &\n' "$RUN_DIR/out.800k.log"
    done
    echo "Dry run: nothing written or launched."
    exit 0
fi

printf 'job\tpid\tlog\n' > "$ROOT_DIR/launches.800k.tsv"
PIDS=()
for job in "${JOBS[@]}"; do
    build_command "$job"
    # Retain fd 9 in children so another invocation cannot overlap this batch.
    nohup "${COMMAND[@]}" < /dev/null >> "$RUN_DIR/out.800k.log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    printf '%s\t%s\t%s\n' "$NAME" "$pid" "$RUN_DIR/out.800k.log" >> "$ROOT_DIR/launches.800k.tsv"
    echo "Started $NAME: PID $pid; log $RUN_DIR/out.800k.log"
done
echo "Launched four TR2 4-GPU jobs: ${PIDS[*]}"
echo "Launcher is not waiting for completion."
echo "PIDs and logs: $ROOT_DIR/launches.800k.tsv"
