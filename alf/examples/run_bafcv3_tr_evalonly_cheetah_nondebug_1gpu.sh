#!/bin/bash
# Launcher for a single BAFCv3-TR eval-only run on 1 GPU.
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
#   -g, --gpu ID                       GPU id for the run (default: 0)
#       --gpus A,B                     Compatibility alias; uses GPU A and ignores GPU B
#       --gpu-a ID                     Compatibility alias for --gpu
#       --gpu-b ID                     Ignored compatibility flag
#       --eval VALUE                   Eval threshold (default: 110.0)
#       --eval-a VALUE                 Compatibility alias for --eval
#       --eval-b VALUE                 Ignored compatibility flag
#       --num-feature-coords VALUE     Trust metric feature coords (default: 4)
#       --metric-interval VALUE        Trust metric update interval (default: 8)
#       --rollout-skip-cap VALUE       Max consecutive eval-gated rollout skips (default: 20)
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
BASE_DIR="/root/alf_results_v5_evalonly_rolloutskip_rolloutbycycle_oom_test"
NUM_ENV_STEPS=1000000
SEED=0
GPU=1
EVAL_TRUST_MAX=40.0
NUM_FEATURE_COORDS=4
METRIC_INTERVAL=8
ROLLOUT_SKIP_CAP=20

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
            shift 2
            ;;
        --gpus)
            GPU="$(echo "$2" | cut -d',' -f1)"
            echo "Ignoring second GPU from --gpus; using GPU ${GPU}" >&2
            shift 2
            ;;
        --gpu-a)
            GPU="$2"
            shift 2
            ;;
        --gpu-b)
            echo "Ignoring --gpu-b for single-GPU launcher" >&2
            shift 2
            ;;
        --eval|--eval-a)
            EVAL_TRUST_MAX="$2"
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

ENV_DIR="$(echo "${ENV_NAME}" | cut -d':' -f1)"
ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc_evalonly_single_run"
RUN_DIR="${ROOT_BASE}/eval${EVAL_TRUST_MAX}_nf${NUM_FEATURE_COORDS}_mi${METRIC_INTERVAL}_cap${ROLLOUT_SKIP_CAP}_seed${SEED}"
mkdir -p "${RUN_DIR}"

echo "Launching BAFCv3-TR eval-only run (single run)"
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  GPU: ${GPU}"
echo "  Seed: ${SEED}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Python: ${PYTHON_BIN}"
echo "  Eval threshold: ${EVAL_TRUST_MAX}"
echo "  num_feature_coords: ${NUM_FEATURE_COORDS}"
echo "  metric_interval: ${METRIC_INTERVAL}"
echo "  rollout_skip_cap: ${ROLLOUT_SKIP_CAP}"
echo "  Eval rollout-skip gate: enabled"
echo "  Grad actor-extend gate: disabled"
echo "  Root dir: ${RUN_DIR}"
echo ""

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
    --conf "${CONF_FILE}" \
    --root_dir "${RUN_DIR}" \
    --conf_param "debug_mode=True" \
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

echo "Started Run PID: ${PID} (GPU ${GPU})"
echo "Log:"
echo "  ${RUN_DIR}/out.log"
