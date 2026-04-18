#!/bin/bash
# Launcher for a single BAFCv3-TR run.
# Runs either a single trust-gated run or a single trust-disabled BAFCv3-TR run.
#
# Usage: bash run_bafcv3_tr_cheetah_nondebug_1gpu.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/alf_results_v2)
#   -n, --steps NUM_STEPS              Total env steps (default: 1000000)
#   -s, --seed SEED                    Seed for the run (default: 0)
#   -g, --gpu ID                       GPU id for the run (default: 0)
#       --gpus A,B                     Compatibility alias; uses GPU A and ignores GPU B
#       --gpu-a ID                     Compatibility alias for --gpu
#       --gpu-b ID                     Ignored compatibility flag
#       --eval VALUE                   Eval threshold (default: 2.0)
#       --eval-a VALUE                 Compatibility alias for --eval
#       --eval-b VALUE                 Ignored compatibility flag
#       --delta VALUE                  Grad threshold (default: 2.0)
#       --delta-a VALUE                Compatibility alias for --delta
#       --delta-b VALUE                Ignored compatibility flag
#       --num-feature-coords VALUE     Trust metric feature coords (default: 4)
#       --metric-interval VALUE        Trust metric update interval (default: 8)
#       --rollout-hold-cap VALUE       Max consecutive eval-gated rollout-actor holds (default: 20)
#       --rollout-skip-cap VALUE       Deprecated alias for --rollout-hold-cap
#       --actor-extend-cap VALUE       Max consecutive grad-gated actor extensions (default: 5)
#       --original-algo                Disable trust metrics and both trust gates for BAFCv3-TR
#   -h, --help                         Show this help message
#
# Example:
#   bash run_bafcv3_tr_cheetah_run_1gpu.sh --gpu 0 --eval 2 --delta 2 --rollout-hold-cap 20 --actor-extend-cap 5
#   bash run_bafcv3_tr_cheetah_run_1gpu.sh --gpu 0 --seed 0 --original-algo

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
BASE_DIR="/root/alf_results_v4_full_gradonly"
NUM_ENV_STEPS=1000000
SEED=0
GPU=1
EVAL_TRUST_MAX=40.0
DELTA_TRUST_MAX=40.0
NUM_FEATURE_COORDS=4
METRIC_INTERVAL=8
ROLLOUT_HOLD_CAP=20
ACTOR_EXTEND_CAP=5
ORIGINAL_ALGO=false

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
        --delta|--delta-a)
            DELTA_TRUST_MAX="$2"
            shift 2
            ;;
        --delta-b)
            echo "Ignoring --delta-b for single-GPU launcher" >&2
            shift 2
            ;;
        --seed-b)
            echo "Ignoring --seed-b for single-GPU launcher" >&2
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
        --rollout-hold-cap|--rollout-skip-cap)
            ROLLOUT_HOLD_CAP="$2"
            shift 2
            ;;
        --actor-extend-cap)
            ACTOR_EXTEND_CAP="$2"
            shift 2
            ;;
        --original-algo)
            ORIGINAL_ALGO=true
            shift
            ;;
        -h|--help)
            sed -n '5,27p' "$0" | sed 's/^# \{0,1\}//'
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
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc_original_algo"
    RUN_DIR="${ROOT_BASE}/seed${SEED}"
else
    ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc_single_run"
    RUN_DIR="${ROOT_BASE}/eval${EVAL_TRUST_MAX}_delta${DELTA_TRUST_MAX}_cap${ROLLOUT_HOLD_CAP}_acap${ACTOR_EXTEND_CAP}_seed${SEED}"
fi
mkdir -p "${RUN_DIR}"

if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "Launching BAFCv3-TR trust-disabled mode (single run)"
else
    echo "Launching BAFCv3-TR trust-gated run (single run)"
fi
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  GPU: ${GPU}"
echo "  Seed: ${SEED}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Python: ${PYTHON_BIN}"
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "  Trust metrics: disabled"
    echo "  Eval rollout-actor gate: disabled"
    echo "  Grad actor-extend gate: disabled"
else
    echo "  Eval threshold: ${EVAL_TRUST_MAX}"
    echo "  Grad threshold: ${DELTA_TRUST_MAX}"
    echo "  num_feature_coords: ${NUM_FEATURE_COORDS}"
    echo "  metric_interval: ${METRIC_INTERVAL}"
    echo "  rollout_hold_cap: ${ROLLOUT_HOLD_CAP}"
    echo "  actor_extend_cap: ${ACTOR_EXTEND_CAP}"
fi
echo "  Root dir: ${RUN_DIR}"
echo ""

if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_DIR}" \
        --conf_param "TrainerConfig.random_seed=${SEED}" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "TrainerConfig.debug_summaries=True" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        --conf_param "BafcAlgorithmV3.monitor_trust_metrics=False" \
        --conf_param "BafcAlgorithmV3.enable_eval_rollout_skip_gate=False" \
        --conf_param "BafcAlgorithmV3.enable_grad_actor_extend_gate=False" \
        > "${RUN_DIR}/out.log" 2>&1 &
else
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_DIR}" \
        --conf_param "TrainerConfig.random_seed=${SEED}" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "TrainerConfig.debug_summaries=True" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        --conf_param "BafcAlgorithmV3.monitor_trust_metrics=True" \
        --conf_param "BafcAlgorithmV3.eval_trust_max=${EVAL_TRUST_MAX}" \
        --conf_param "BafcAlgorithmV3.delta_trust_max=${DELTA_TRUST_MAX}" \
        --conf_param "BafcAlgorithmV3.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
        --conf_param "BafcAlgorithmV3.trust_metric_update_interval=${METRIC_INTERVAL}" \
        --conf_param "BafcAlgorithmV3.eval_gate_max_consecutive_rollout_actor_holds=${ROLLOUT_HOLD_CAP}" \
        --conf_param "BafcAlgorithmV3.grad_gate_max_consecutive_actor_extensions=${ACTOR_EXTEND_CAP}" \
        > "${RUN_DIR}/out.log" 2>&1 &
fi
PID=$!

echo "Started Run PID: ${PID} (GPU ${GPU})"
echo "Log:"
echo "  ${RUN_DIR}/out.log"
