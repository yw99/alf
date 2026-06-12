#!/bin/bash
# Launcher script for running RLPD seed 4 and BAFCv3-TR seed 4 in parallel.
# Both jobs share one 2-GPU DDP group, using distinct MASTER_PORT values.
#
# Usage: bash run_rlpd_bafcv3_tr_seed4_2gpu.sh [options]
#   -e, --env ENV_NAME        DMC environment (default: cheetah:run)
#   -d, --dir BASE_DIR        Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS     Total environment steps (default: 400000)
#   -g, --gpus GPU_IDS        Comma-separated GPU ids (default: 0,1)
#       --base-port PORT      First MASTER_PORT to use (default: 29520)
#   -h, --help                Show this help message
#
# Examples:
#   bash run_rlpd_bafcv3_tr_seed4_2gpu.sh
#   bash run_rlpd_bafcv3_tr_seed4_2gpu.sh --gpus 2,3 --base-port 29530
#   bash run_rlpd_bafcv3_tr_seed4_2gpu.sh --env hopper:hop --steps 500000

print_usage() {
    sed -n '4,16p' "$0"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLPD_CONF_FILE="${SCRIPT_DIR}/rlpd_dmc_conf.py"
BAFCV3_TR_CONF_FILE="${SCRIPT_DIR}/bafcv3_tr_dmc_conf.py"

ENV_NAME="cheetah:run"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=400000
NUM_CHECKPOINTS=10
GPU_IDS="0,1"
BASE_PORT=29520
SEED=4

CRITIC_UTD=10
NUM_UPDATES_PER_TRAIN_ITER=11

EVAL_TRUST_MAX=40.0
NUM_FEATURE_COORDS=4
METRIC_INTERVAL=8
ROLLOUT_SKIP_CAP=3
ROLLOUT_SKIP_EVAL_INTERVAL=60
FREEZE_EVAL_SAMPLES=False
ACTOR_USE_LN=False

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
        -g|--gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
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

ENV_DIR=$(echo "$ENV_NAME" | cut -d':' -f1)
RLPD_ROOT_DIR="${BASE_DIR}/${ENV_DIR}/rlpd_dmc/critic_utd${CRITIC_UTD}"
BAFCV3_TR_ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_tr_dmc"
RLPD_PORT=${BASE_PORT}
BAFCV3_TR_PORT=$((BASE_PORT + 1))

echo "Starting RLPD seed ${SEED} and BAFCv3-TR seed ${SEED} on GPUs ${GPU_IDS}"
echo "  Environment: $ENV_NAME"
echo "  Num env steps: $NUM_ENV_STEPS"
echo "  Num checkpoints: $NUM_CHECKPOINTS"
echo "  RLPD port: $RLPD_PORT"
echo "  BAFCv3-TR port: $BAFCV3_TR_PORT"
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
    "${RLPD_ROOT_DIR}/seed_${SEED}" \
    "${BAFCV3_TR_ROOT_DIR}/seed_${SEED}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} MASTER_PORT=${RLPD_PORT} python -m alf.bin.train \
    --conf "$RLPD_CONF_FILE" \
    --root_dir "${RLPD_ROOT_DIR}/seed_${SEED}" \
    --conf_param "TrainerConfig.random_seed=${SEED}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.num_updates_per_train_iter=${NUM_UPDATES_PER_TRAIN_ITER}" \
    --conf_param "RlpdAlgorithm.critic_utd=${CRITIC_UTD}" \
    --conf_param "make_ddp_performer.find_unused_parameters=True" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${RLPD_ROOT_DIR}/seed_${SEED}/out.log" 2>&1 &
PID_RLPD=$!

CUDA_VISIBLE_DEVICES=${GPU_IDS} MASTER_PORT=${BAFCV3_TR_PORT} python -m alf.bin.train \
    --conf "$BAFCV3_TR_CONF_FILE" \
    --root_dir "${BAFCV3_TR_ROOT_DIR}/seed_${SEED}" \
    --conf_param "TrainerConfig.random_seed=${SEED}" \
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
    > "${BAFCV3_TR_ROOT_DIR}/seed_${SEED}/out.log" 2>&1 &
PID_BAFCV3_TR=$!

echo "RLPD seed ${SEED} running on GPUs ${GPU_IDS} port ${RLPD_PORT} (PID: $PID_RLPD)"
echo "BAFCv3-TR seed ${SEED} running on GPUs ${GPU_IDS} port ${BAFCV3_TR_PORT} (PID: $PID_BAFCV3_TR)"
echo ""
echo "To monitor RLPD: tail -f ${RLPD_ROOT_DIR}/seed_${SEED}/out.log"
echo "To monitor BAFCv3-TR: tail -f ${BAFCV3_TR_ROOT_DIR}/seed_${SEED}/out.log"
