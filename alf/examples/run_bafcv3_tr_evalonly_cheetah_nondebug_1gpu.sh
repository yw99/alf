#!/bin/bash
# Launcher for one or more BAFCv3-TR eval-only runs on 1-4 GPUs.
# Eval-only means:
#   - eval rollout skip gate enabled
#   - grad actor-extend gate disabled
#   - expensive grad trust metric compute skipped
#
# Usage: bash run_bafcv3_tr_evalonly_cheetah_nondebug_1gpu.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/alf_results_v5_evalonly_rolloutskip)
#   -n, --steps NUM_STEPS              Total env steps (default: 1000000)
#   -s, --seed SEED                    Seed for the run (default: 0)
#   -g, --gpu ID                       GPU id for a single CLI-selected run
#       --gpus A,B                     Compatibility alias; uses GPU A and ignores GPU B
#       --gpu-a ID                     Compatibility alias for --gpu
#       --gpu-b ID                     Ignored compatibility flag
#       --eval VALUE                   Eval threshold for a single CLI-selected run
#       --eval-a VALUE                 Compatibility alias for --eval
#       --eval-b VALUE                 Ignored compatibility flag
#       --num-feature-coords VALUE     Trust metric feature coords (default: 4)
#       --metric-interval VALUE        Trust metric update interval (default: 8)
#       --rollout-skip-cap VALUE       Max consecutive eval-gated rollout skips (default: 10)
#       --delta VALUE                  Ignored in eval-only mode
#       --actor-extend-cap VALUE       Ignored in eval-only mode
#   -h, --help                         Show this help message
#
# Example:
#   bash run_bafcv3_tr_evalonly_cheetah_nondebug_1gpu.sh --gpu 0 --eval 40 --rollout-skip-cap 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_tr_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi

ENV_NAME="cheetah:run"
BASE_DIR="/root/alf_results_v6_evalonly_oldref"
NUM_ENV_STEPS=1000000
SEED=0
GPU_IDS=(0 1 2 3)
EVAL_TRUST_MAXES=(20.0 30.0 40.0 60.0)
GPU="${GPU_IDS[0]}"
EVAL_TRUST_MAX="${EVAL_TRUST_MAXES[0]}"
NUM_FEATURE_COORDS=4
METRIC_INTERVAL=8
ROLLOUT_SKIP_CAP=10
USE_SINGLE_CLI_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env)
            ENV_NAME="$2"
            shift 2
            ;;
        -d|--dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -n|--steps)
            NUM_ENV_STEPS="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            USE_SINGLE_CLI_RUN=true
            shift 2
            ;;
        --gpus)
            GPU="$(echo "$2" | cut -d',' -f1)"
            echo "Ignoring second GPU from --gpus; using GPU ${GPU}" >&2
            USE_SINGLE_CLI_RUN=true
            shift 2
            ;;
        --gpu-a)
            GPU="$2"
            USE_SINGLE_CLI_RUN=true
            shift 2
            ;;
        --gpu-b)
            echo "Ignoring --gpu-b for single-GPU launcher" >&2
            shift 2
            ;;
        --eval|--eval-a)
            EVAL_TRUST_MAX="$2"
            USE_SINGLE_CLI_RUN=true
            shift 2
            ;;
        --eval-b)
            echo "Ignoring --eval-b for single-GPU launcher" >&2
            shift 2
            ;;
        --num-feature-coords)
            NUM_FEATURE_COORDS="$2"
            shift 2
            ;;
        --metric-interval)
            METRIC_INTERVAL="$2"
            shift 2
            ;;
        --rollout-skip-cap)
            ROLLOUT_SKIP_CAP="$2"
            shift 2
            ;;
        --delta|--delta-a|--delta-b)
            echo "Ignoring $1 in eval-only mode (grad gate is disabled)" >&2
            shift 2
            ;;
        --actor-extend-cap)
            echo "Ignoring --actor-extend-cap in eval-only mode (grad gate is disabled)" >&2
            shift 2
            ;;
        --seed-b)
            echo "Ignoring --seed-b for single-GPU launcher" >&2
            shift 2
            ;;
        -h|--help)
            sed -n '8,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [[ "${USE_SINGLE_CLI_RUN}" == "true" ]]; then
    GPU_IDS=("${GPU}")
    EVAL_TRUST_MAXES=("${EVAL_TRUST_MAX}")
fi

if [[ ${#GPU_IDS[@]} -lt 1 || ${#GPU_IDS[@]} -gt 4 ]]; then
    echo "GPU_IDS must contain 1 to 4 entries; got ${#GPU_IDS[@]}" >&2
    exit 1
fi
if [[ ${#GPU_IDS[@]} -ne ${#EVAL_TRUST_MAXES[@]} ]]; then
    echo "GPU_IDS and EVAL_TRUST_MAXES must have the same length" >&2
    echo "  GPU_IDS: ${GPU_IDS[*]}" >&2
    echo "  EVAL_TRUST_MAXES: ${EVAL_TRUST_MAXES[*]}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc_evalonly_single_run"
echo "Launching ${#GPU_IDS[@]} BAFCv3-TR eval-only run(s)"
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  GPUs: ${GPU_IDS[*]}"
echo "  Eval thresholds: ${EVAL_TRUST_MAXES[*]}"
echo "  Seed: ${SEED}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Python: ${PYTHON_BIN}"
echo "  num_feature_coords: ${NUM_FEATURE_COORDS}"
echo "  metric_interval: ${METRIC_INTERVAL}"
echo "  rollout_skip_cap: ${ROLLOUT_SKIP_CAP}"
echo "  Eval rollout-skip gate: enabled"
echo "  Grad actor-extend gate: disabled"
echo ""

cd "${REPO_ROOT}"
PIDS=()

for i in "${!GPU_IDS[@]}"; do
    GPU="${GPU_IDS[$i]}"
    EVAL_TRUST_MAX="${EVAL_TRUST_MAXES[$i]}"
    RUN_DIR="${ROOT_BASE}/eval${EVAL_TRUST_MAX}_nf${NUM_FEATURE_COORDS}_mi${METRIC_INTERVAL}_cap${ROLLOUT_SKIP_CAP}_seed${SEED}"
    mkdir -p "${RUN_DIR}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_DIR}" \
        --conf_param "TrainerConfig.random_seed=${SEED}" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "TrainerConfig.debug_summaries=True" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        --conf_param "BafcAlgorithmV3.monitor_trust_metrics=True" \
        --conf_param "BafcAlgorithmV3.eval_trust_max=${EVAL_TRUST_MAX}" \
        --conf_param "BafcAlgorithmV3.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
        --conf_param "BafcAlgorithmV3.trust_metric_update_interval=${METRIC_INTERVAL}" \
        --conf_param "BafcAlgorithmV3.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
        --conf_param "BafcAlgorithmV3.enable_eval_rollout_skip_gate=True" \
        --conf_param "BafcAlgorithmV3.enable_grad_actor_extend_gate=False" \
        > "${RUN_DIR}/out.log" 2>&1 &

    PID=$!
    PIDS+=("${PID}")
    echo "Started eval ${EVAL_TRUST_MAX} on GPU ${GPU}: PID ${PID}"
    echo "  Log: ${RUN_DIR}/out.log"
done

echo ""
echo "Started PIDs: ${PIDS[*]}"
