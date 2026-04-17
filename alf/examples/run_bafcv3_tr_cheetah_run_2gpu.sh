#!/bin/bash
# Launcher for BAFCv3-TR runs.
# Runs either a 2-job threshold sweep or a single trust-disabled BAFCv3-TR job.
#
# Usage: bash run_bafcv3_tr_cheetah_run_2gpu.sh [options]
#   -e, --env ENV_NAME                 DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR                 Base results directory (default: /root/alf_results_v2)
#   -n, --steps NUM_STEPS              Total env steps (default: 1000000)
#   -s, --seed SEED                    Seed for run A (default: 0)
#       --seed-b SEED                  Seed for run B (default: same as --seed)
#   -g, --gpus A,B                     GPU ids for run A and B (default: 0,1)
#       --gpu-a ID                     GPU id for run A (overrides --gpus)
#       --gpu-b ID                     GPU id for run B (overrides --gpus)
#       --eval-a VALUE                 Eval threshold for run A (default: 2.0)
#       --delta-a VALUE                Grad threshold for run A (default: 4.0)
#       --eval-b VALUE                 Eval threshold for run B (default: 5.0)
#       --delta-b VALUE                Grad threshold for run B (default: 10.0)
#       --num-feature-coords VALUE     Trust metric feature coords (default: 4)
#       --metric-interval VALUE        Trust metric update interval (default: 8)
#       --rollout-hold-cap VALUE       Max consecutive eval-gated rollout-actor holds (default: 20)
#       --rollout-skip-cap VALUE       Deprecated alias for --rollout-hold-cap
#       --actor-extend-cap VALUE       Max consecutive grad-gated actor extensions (default: 5)
#       --original-algo                Disable trust metrics and both trust gates for BAFCv3-TR; launch one job on GPU A
#   -h, --help                         Show this help message
#
# Example:
#   bash run_bafcv3_tr_cheetah_run_2gpu.sh --gpus 0,1 --eval-a 2 --delta-a 4 --eval-b 5 --delta-b 10 --rollout-hold-cap 20 --actor-extend-cap 5
#   bash run_bafcv3_tr_cheetah_run_2gpu.sh --gpu-a 0 --seed 0 --original-algo

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
BASE_DIR="/root/alf_results_v3"
NUM_ENV_STEPS=1000000
SEED_A=0
SEED_B=1
GPU_A=0
GPU_B=1
EVAL_TRUST_MAX_A=20.0
DELTA_TRUST_MAX_A=20.0
EVAL_TRUST_MAX_B=5.0
DELTA_TRUST_MAX_B=5.0
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
            SEED_A="$2"
            shift 2
            ;;
        --seed-b)
            SEED_B="$2"
            shift 2
            ;;
        -g|--gpus)
            GPU_A="$(echo "$2" | cut -d',' -f1)"
            GPU_B="$(echo "$2" | cut -d',' -f2)"
            shift 2
            ;;
        --gpu-a)
            GPU_A="$2"
            shift 2
            ;;
        --gpu-b)
            GPU_B="$2"
            shift 2
            ;;
        --eval-a)
            EVAL_TRUST_MAX_A="$2"
            shift 2
            ;;
        --delta-a)
            DELTA_TRUST_MAX_A="$2"
            shift 2
            ;;
        --eval-b)
            EVAL_TRUST_MAX_B="$2"
            shift 2
            ;;
        --delta-b)
            DELTA_TRUST_MAX_B="$2"
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
            sed -n '5,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [[ -z "${SEED_B}" ]]; then
    SEED_B="${SEED_A}"
fi

ENV_DIR="$(echo "${ENV_NAME}" | cut -d':' -f1)"
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc_original_algo"
    RUN_DIR="${ROOT_BASE}/seed${SEED_A}"
else
    ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc_threshold_sweep"
    RUN_A_DIR="${ROOT_BASE}/A_eval${EVAL_TRUST_MAX_A}_delta${DELTA_TRUST_MAX_A}_cap${ROLLOUT_HOLD_CAP}_acap${ACTOR_EXTEND_CAP}_seed${SEED_A}"
    RUN_B_DIR="${ROOT_BASE}/B_eval${EVAL_TRUST_MAX_B}_delta${DELTA_TRUST_MAX_B}_cap${ROLLOUT_HOLD_CAP}_acap${ACTOR_EXTEND_CAP}_seed${SEED_B}"
fi
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    mkdir -p "${RUN_DIR}"
else
    mkdir -p "${RUN_A_DIR}" "${RUN_B_DIR}"
fi

if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "Launching BAFCv3-TR trust-disabled mode (single run)"
else
    echo "Launching BAFCv3-TR threshold sweep (2 parallel runs)"
fi
echo "  Config: ${CONF_FILE}"
echo "  Environment: ${ENV_NAME}"
echo "  Num env steps: ${NUM_ENV_STEPS}"
echo "  Repo root: ${REPO_ROOT}"
echo "  Python: ${PYTHON_BIN}"
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "  Trust metrics: disabled"
    echo "  Eval rollout-actor gate: disabled"
    echo "  Grad actor-extend gate: disabled"
else
    echo "  num_feature_coords: ${NUM_FEATURE_COORDS}"
    echo "  metric_interval: ${METRIC_INTERVAL}"
    echo "  rollout_hold_cap: ${ROLLOUT_HOLD_CAP}"
    echo "  actor_extend_cap: ${ACTOR_EXTEND_CAP}"
fi
echo ""
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "Run: GPU ${GPU_A}, seed ${SEED_A}"
else
    echo "Run A: GPU ${GPU_A}, seed ${SEED_A}, eval ${EVAL_TRUST_MAX_A}, delta ${DELTA_TRUST_MAX_A}"
fi
if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "  Root dir: ${RUN_DIR}"
else
    echo "  Root dir: ${RUN_A_DIR}"
    echo "Run B: GPU ${GPU_B}, seed ${SEED_B}, eval ${EVAL_TRUST_MAX_B}, delta ${DELTA_TRUST_MAX_B}"
    echo "  Root dir: ${RUN_B_DIR}"
fi
echo ""

if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_A}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_DIR}" \
        --conf_param "debug_mode=True" \
        --conf_param "TrainerConfig.random_seed=${SEED_A}" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "TrainerConfig.debug_summaries=True" \
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        --conf_param "BafcAlgorithmV3.monitor_trust_metrics=False" \
        --conf_param "BafcAlgorithmV3.enable_eval_rollout_skip_gate=False" \
        --conf_param "BafcAlgorithmV3.enable_grad_actor_extend_gate=False" \
        > "${RUN_DIR}/out.log" 2>&1 &
    PID=$!
else
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_A}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_A_DIR}" \
        --conf_param "debug_mode=True" \
        --conf_param "TrainerConfig.random_seed=${SEED_A}" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "TrainerConfig.debug_summaries=True" \
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        --conf_param "BafcAlgorithmV3.monitor_trust_metrics=True" \
        --conf_param "BafcAlgorithmV3.eval_trust_max=${EVAL_TRUST_MAX_A}" \
        --conf_param "BafcAlgorithmV3.delta_trust_max=${DELTA_TRUST_MAX_A}" \
        --conf_param "BafcAlgorithmV3.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
        --conf_param "BafcAlgorithmV3.trust_metric_update_interval=${METRIC_INTERVAL}" \
        --conf_param "BafcAlgorithmV3.eval_gate_max_consecutive_rollout_actor_holds=${ROLLOUT_HOLD_CAP}" \
        --conf_param "BafcAlgorithmV3.grad_gate_max_consecutive_actor_extensions=${ACTOR_EXTEND_CAP}" \
        > "${RUN_A_DIR}/out.log" 2>&1 &
    PID_A=$!

    CUDA_VISIBLE_DEVICES="${GPU_B}" "${PYTHON_BIN}" -m alf.bin.train \
        --conf "${CONF_FILE}" \
        --root_dir "${RUN_B_DIR}" \
        --conf_param "debug_mode=True" \
        --conf_param "TrainerConfig.random_seed=${SEED_B}" \
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
        --conf_param "TrainerConfig.debug_summaries=True" \
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
        --conf_param "create_environment.env_name='${ENV_NAME}'" \
        --conf_param "BafcAlgorithmV3.monitor_trust_metrics=True" \
        --conf_param "BafcAlgorithmV3.eval_trust_max=${EVAL_TRUST_MAX_B}" \
        --conf_param "BafcAlgorithmV3.delta_trust_max=${DELTA_TRUST_MAX_B}" \
        --conf_param "BafcAlgorithmV3.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
        --conf_param "BafcAlgorithmV3.trust_metric_update_interval=${METRIC_INTERVAL}" \
        --conf_param "BafcAlgorithmV3.eval_gate_max_consecutive_rollout_actor_holds=${ROLLOUT_HOLD_CAP}" \
        --conf_param "BafcAlgorithmV3.grad_gate_max_consecutive_actor_extensions=${ACTOR_EXTEND_CAP}" \
        > "${RUN_B_DIR}/out.log" 2>&1 &
    PID_B=$!
fi

if [[ "${ORIGINAL_ALGO}" == "true" ]]; then
    echo "Started Run PID: ${PID} (GPU ${GPU_A})"
    echo "Log:"
    echo "  ${RUN_DIR}/out.log"
else
    echo "Started Run A PID: ${PID_A} (GPU ${GPU_A})"
    echo "Started Run B PID: ${PID_B} (GPU ${GPU_B})"
    echo "Logs:"
    echo "  ${RUN_A_DIR}/out.log"
    echo "  ${RUN_B_DIR}/out.log"
fi
