#!/bin/bash
# Evaluate the latest complete checkpoints from the dog:fetch RLPD,
# BAFCv3, and BAFCv3_TR2 seed-0/1 comparison launched by
# run_dog_fetch_rlpd_bafcv3_seed01-4g.sh.
#
# Usage: bash evaluate_dog_fetch_rlpd_bafcv3_seed01-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -o, --output-dir DIR     Evaluation output directory (default: under comparison root)
#   -n, --episodes N         Episodes per checkpoint (default: 10)
#       --eval-seed SEED     First shared evaluation seed (default: 0)
#       --checkpoint STEP    Checkpoint number or latest (default: latest)
#       --device DEVICE      Evaluation device: cpu or cuda (default: cpu)
#       --ddp-world-size N   Override inferred training DDP world size
#       --dry-run            Print the command without running it
#   -h, --help               Show this help message
#
# Example:
#   bash evaluate_dog_fetch_rlpd_bafcv3_seed01-4g.sh --episodes 10

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

BASE_DIR="/workspace/alf_results"
OUTPUT_DIR=""
NUM_EPISODES=10
EVAL_SEED=0
CHECKPOINT_STEP="latest"
DEVICE="cpu"
DDP_WORLD_SIZE=""
DRY_RUN=False

print_help() {
    sed -n '/^# Usage:/,/^# Example:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --eval-seed)
            EVAL_SEED="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --ddp-world-size)
            DDP_WORLD_SIZE="$2"
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
    exit 1
fi
if [[ ! -f "${REPO_ROOT}/alf/bin/evaluate_dog_fetch_checkpoints.py" ]]; then
    echo "Dog-fetch evaluator not found under ${REPO_ROOT}" >&2
    exit 1
fi
if [[ ! "${NUM_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--episodes must be a positive integer, got: ${NUM_EPISODES}" >&2
    exit 1
fi
if [[ ! "${EVAL_SEED}" =~ ^[0-9]+$ ]]; then
    echo "--eval-seed must be a non-negative integer, got: ${EVAL_SEED}" >&2
    exit 1
fi
if [[ "${CHECKPOINT_STEP}" != "latest" && ! "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
    echo "--checkpoint must be 'latest' or a non-negative integer, got: ${CHECKPOINT_STEP}" >&2
    exit 1
fi
if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
    echo "--device must be cpu or cuda, got: ${DEVICE}" >&2
    exit 1
fi
if [[ -n "${DDP_WORLD_SIZE}" && ! "${DDP_WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--ddp-world-size must be a positive integer, got: ${DDP_WORLD_SIZE}" >&2
    exit 1
fi

COMPARISON_ROOT="${BASE_DIR}/dog_fetch/rlpd_bafcv3_comparison_4g"
if [[ ! -d "${COMPARISON_ROOT}" ]]; then
    echo "Comparison directory not found: ${COMPARISON_ROOT}" >&2
    exit 1
fi

command=(
    "${PYTHON_BIN}" -m alf.bin.evaluate_dog_fetch_checkpoints
    --comparison_root "${COMPARISON_ROOT}"
    --num_episodes "${NUM_EPISODES}"
    --eval_seed "${EVAL_SEED}"
    --checkpoint_step "${CHECKPOINT_STEP}"
    --device "${DEVICE}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
    command+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${DDP_WORLD_SIZE}" ]]; then
    command+=(--ddp_world_size "${DDP_WORLD_SIZE}")
fi

cat <<EOF
Starting dog:fetch checkpoint-stage evaluation
  Comparison root: ${COMPARISON_ROOT}
  Output directory: ${OUTPUT_DIR:-${COMPARISON_ROOT}/checkpoint_stage_evaluation}
  Episodes per checkpoint: ${NUM_EPISODES}
  Evaluation seeds: ${EVAL_SEED} through $((EVAL_SEED + NUM_EPISODES - 1))
  Checkpoint: ${CHECKPOINT_STEP}
  Device: ${DEVICE}
  DDP world size: ${DDP_WORLD_SIZE:-infer from checkpoint sidecars}
  Dry run: ${DRY_RUN}
EOF

cd "${REPO_ROOT}"
if [[ "${DRY_RUN}" == "True" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
