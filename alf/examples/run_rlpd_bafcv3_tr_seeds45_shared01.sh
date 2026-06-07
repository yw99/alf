#!/bin/bash
# Launcher script for running RLPD and BAFCv3-TR seeds 4,5 in parallel
# All jobs share GPUs 0,1 and use distinct MASTER_PORT values.
#
# Usage: bash run_rlpd_bafcv3_tr_seeds45_shared01.sh [options]
#   -e, --env ENV_NAME        DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR        Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS     Total environment steps (default: 400000)
#   -h, --help                Show this help message
#
# Examples:
#   bash run_rlpd_bafcv3_tr_seeds45_shared01.sh -e walker:walk
#   bash run_rlpd_bafcv3_tr_seeds45_shared01.sh -e walker:walk -n 500000
#   bash run_rlpd_bafcv3_tr_seeds45_shared01.sh --env hopper:hop --steps 2000000 --dir /my/results

print_usage() {
    sed -n '4,14p' "$0"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLPD_CONF_FILE="${SCRIPT_DIR}/rlpd_dmc_conf.py"
BAFCV3_TR_CONF_FILE="${SCRIPT_DIR}/bafcv3_tr_dmc_conf.py"
ENV_NAME="cheetah:run"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=400000
NUM_CHECKPOINTS=10

CRITIC_UTD=3
NUM_UPDATES_PER_TRAIN_ITER=4

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
            print_usage
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
RLPD_ROOT_DIR="${BASE_DIR}/${ENV_DIR}/rlpd_dmc/critic_utd${CRITIC_UTD}"
BAFCV3_TR_ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc"

echo "Starting RLPD critic_utd=${CRITIC_UTD} and BAFCv3-TR training with seeds 4,5"
echo "  Environment: $ENV_NAME"
echo "  Num env steps: $NUM_ENV_STEPS"
echo "  Num checkpoints: $NUM_CHECKPOINTS"
echo ""
echo "RLPD settings"
echo "  Config: $RLPD_CONF_FILE"
echo "  Root dir: $RLPD_ROOT_DIR"
echo "  Critic UTD: $CRITIC_UTD"
echo "  Num updates per train iter: $NUM_UPDATES_PER_TRAIN_ITER"
echo ""
echo "BAFCv3-TR settings"
echo "  Config: $BAFCV3_TR_CONF_FILE"
echo "  Root dir: $BAFCV3_TR_ROOT_DIR"
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

mkdir -p \
    "${RLPD_ROOT_DIR}/seed_4" \
    "${RLPD_ROOT_DIR}/seed_5" \
    "${BAFCV3_TR_ROOT_DIR}/seed_4" \
    "${BAFCV3_TR_ROOT_DIR}/seed_5"

# RLPD seeds 4,5 share GPUs 0,1 (different MASTER_PORT)
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29504 python -m alf.bin.train \
    --conf "$RLPD_CONF_FILE" \
    --root_dir "${RLPD_ROOT_DIR}/seed_4" \
    --conf_param "TrainerConfig.random_seed=4" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.num_updates_per_train_iter=${NUM_UPDATES_PER_TRAIN_ITER}" \
    --conf_param "RlpdAlgorithm.critic_utd=${CRITIC_UTD}" \
    --conf_param "make_ddp_performer.find_unused_parameters=True" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${RLPD_ROOT_DIR}/seed_4/out.log" 2>&1 &
PID_RLPD_4=$!

CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29505 python -m alf.bin.train \
    --conf "$RLPD_CONF_FILE" \
    --root_dir "${RLPD_ROOT_DIR}/seed_5" \
    --conf_param "TrainerConfig.random_seed=5" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.num_updates_per_train_iter=${NUM_UPDATES_PER_TRAIN_ITER}" \
    --conf_param "RlpdAlgorithm.critic_utd=${CRITIC_UTD}" \
    --conf_param "make_ddp_performer.find_unused_parameters=True" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${RLPD_ROOT_DIR}/seed_5/out.log" 2>&1 &
PID_RLPD_5=$!

# BAFCv3-TR seeds 4,5 share GPUs 0,1 (different MASTER_PORT)
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29506 python -m alf.bin.train \
    --conf "$BAFCV3_TR_CONF_FILE" \
    --root_dir "${BAFCV3_TR_ROOT_DIR}/seed_4" \
    --conf_param "TrainerConfig.random_seed=4" \
    --conf_param "bafcv3_tr_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3.enable_grad_actor_extend_gate=False" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${BAFCV3_TR_ROOT_DIR}/seed_4/out.log" 2>&1 &
PID_BAFCV3_TR_4=$!

CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29507 python -m alf.bin.train \
    --conf "$BAFCV3_TR_CONF_FILE" \
    --root_dir "${BAFCV3_TR_ROOT_DIR}/seed_5" \
    --conf_param "TrainerConfig.random_seed=5" \
    --conf_param "bafcv3_tr_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3.enable_grad_actor_extend_gate=False" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${BAFCV3_TR_ROOT_DIR}/seed_5/out.log" 2>&1 &
PID_BAFCV3_TR_5=$!

echo "RLPD seed 4 running on GPUs 0,1 port 29504 (PID: $PID_RLPD_4)"
echo "RLPD seed 5 running on GPUs 0,1 port 29505 (PID: $PID_RLPD_5)"
echo "BAFCv3-TR seed 4 running on GPUs 0,1 port 29506 (PID: $PID_BAFCV3_TR_4)"
echo "BAFCv3-TR seed 5 running on GPUs 0,1 port 29507 (PID: $PID_BAFCV3_TR_5)"
echo ""
echo "To monitor RLPD: tail -f ${RLPD_ROOT_DIR}/seed_*/out.log"
echo "To monitor BAFCv3-TR: tail -f ${BAFCV3_TR_ROOT_DIR}/seed_*/out.log"
