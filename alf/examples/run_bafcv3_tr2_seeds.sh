#!/bin/bash
# Launcher script for running BAFCv3-TR2 with 4 seeds in parallel
# Seeds 0,1 share GPUs 0,1; Seeds 2,3 share GPUs 2,3
# Each seed uses 2 GPUs via DDP for 2 parallel environments
#
# Usage: bash run_bafcv3_tr2_seeds.sh [options]
#   -e, --env ENV_NAME        DMC environment (default: hopper:hop)
#   -d, --dir BASE_DIR        Base results directory (default: /workspace/results)
#   -n, --steps NUM_STEPS     Total environment steps (default: 1000000)
#   -h, --help                Show this help message
#
# Examples:
#   bash run_bafcv3_tr2_seeds.sh -e walker:walk
#   bash run_bafcv3_tr2_seeds.sh -e walker:walk -n 500000
#   bash run_bafcv3_tr2_seeds.sh --env hopper:hop --steps 2000000 --dir /my/results

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_tr2_dmc_conf.py"
# ENV_NAME="hopper:hop"
ENV_NAME="cheetah:run"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=400000
NUM_CHECKPOINTS=10
EVAL_TRUST_MAX=40.0
NUM_FEATURE_COORDS=4
METRIC_INTERVAL=8
ROLLOUT_SKIP_CAP=3
ROLLOUT_SKIP_EVAL_INTERVAL=60
FREEZE_EVAL_SAMPLES=False
ACTOR_USE_LN=False

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        -h|--help)
            head -15 "$0" | tail -13
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Extract domain name for directory (e.g., hopper:hop -> hopper)
ENV_DIR=$(echo "$ENV_NAME" | cut -d':' -f1)
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_tr2_dmc"

echo "Starting BAFCv3-TR2 training with 4 seeds"
echo "  Config: $CONF_FILE"
echo "  Environment: $ENV_NAME"
echo "  Root dir: $ROOT_DIR"
echo "  Num env steps: $NUM_ENV_STEPS"
echo "  Num checkpoints: $NUM_CHECKPOINTS"
echo "  Eval threshold: $EVAL_TRUST_MAX"
echo "  num_feature_coords: $NUM_FEATURE_COORDS"
echo "  metric_interval: $METRIC_INTERVAL"
echo "  rollout_skip_cap: $ROLLOUT_SKIP_CAP"
echo "  rollout_skip_eval_interval: $ROLLOUT_SKIP_EVAL_INTERVAL"
echo "  freeze_eval_samples: $FREEZE_EVAL_SAMPLES"
echo "  Actor layer norm: $ACTOR_USE_LN"
echo "  Eval rollout-skip gate: enabled"
echo "  Grad actor-extend gate: disabled"
echo ""

mkdir -p "${ROOT_DIR}/seed_0" "${ROOT_DIR}/seed_1" "${ROOT_DIR}/seed_2" "${ROOT_DIR}/seed_3"

# Seeds 0,1 share GPUs 0,1 (different MASTER_PORT)
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29500 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_0" \
    --conf_param "TrainerConfig.random_seed=0" \
    --conf_param "bafcv3_tr2_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_0/out.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29501 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_1" \
    --conf_param "TrainerConfig.random_seed=1" \
    --conf_param "bafcv3_tr2_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_1/out.log" 2>&1 &
PID1=$!

# Seeds 2,3 share GPUs 2,3 (different MASTER_PORT)
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29502 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_2" \
    --conf_param "TrainerConfig.random_seed=2" \
    --conf_param "bafcv3_tr2_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_2/out.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29503 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_3" \
    --conf_param "TrainerConfig.random_seed=3" \
    --conf_param "bafcv3_tr2_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_3/out.log" 2>&1 &
PID3=$!

echo "Seed 0 running on GPUs 0,1 port 29500 (PID: $PID0)"
echo "Seed 1 running on GPUs 0,1 port 29501 (PID: $PID1)"
echo "Seed 2 running on GPUs 2,3 port 29502 (PID: $PID2)"
echo "Seed 3 running on GPUs 2,3 port 29503 (PID: $PID3)"
echo ""
echo "To monitor: tail -f ${ROOT_DIR}/seed_*/out.log"
