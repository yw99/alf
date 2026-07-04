#!/bin/bash
# Launcher script for running BAFCv3-TR2 with 4 seeds in parallel
# The configured seeds share GPUs 0,1,2,3
# Each seed uses 4 GPUs via DDP for 4 parallel environments
#
# Usage: bash run_bafcv3_tr2_seeds-4g.sh [options]
#   -e, --env ENV_NAME        DMC environment (default: hopper:hop)
#   -d, --dir BASE_DIR        Base results directory (default: /root/numeric_results)
#   -n, --steps NUM_STEPS     Total environment steps (default: 400000)
#       --critic-utd N        Critic update-to-data ratio (default: 3)
#       --eval-trust-max X    Eval trust threshold (default: 40.0)
#   -h, --help                Show this help message
#
# Examples:
#   bash run_bafcv3_tr2_seeds-4g.sh -e walker:walk
#   bash run_bafcv3_tr2_seeds-4g.sh -e walker:walk -n 500000
#   bash run_bafcv3_tr2_seeds-4g.sh --env hopper:hop --steps 2000000 --dir /my/results
#   EVAL_TRUST_MAX=30.0 bash run_bafcv3_tr2_seeds-4g.sh --critic-utd 3
#   bash run_bafcv3_tr2_seeds-4g.sh --critic-utd 3 --eval-trust-max 30.0

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_tr2_dmc_conf.py"
ENV_NAME="hopper:hop"
# ENV_NAME="cheetah:run"
BASE_DIR="/root/numeric_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
EVAL_TRUST_MAX="${EVAL_TRUST_MAX:-40.0}"
NUM_FEATURE_COORDS=4
METRIC_INTERVAL=8
ROLLOUT_SKIP_CAP=4
ROLLOUT_SKIP_EVAL_INTERVAL=60
FREEZE_EVAL_SAMPLES=False
ACTOR_USE_LN=False
SEEDS=(0 1 2 3)
CRITIC_UTD=3
ENABLE_CRITIC_REWEIGHTING=False
CRITIC_REWEIGHTING_BETA=None
CRITIC_REWEIGHTING_RIDGE=None
CRITIC_REWEIGHTING_SOLVER_ITERS=5
CRITIC_REWEIGHTING_MAX_WEIGHT=20.0

print_help() {
    sed -n '/^# Usage:/,/^# Examples:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

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
        --critic-utd)
            CRITIC_UTD="$2"
            shift 2
            ;;
        --eval-trust-max)
            EVAL_TRUST_MAX="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [[ ! "${CRITIC_UTD}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--critic-utd must be a positive integer, got: ${CRITIC_UTD}" >&2
    exit 1
fi

if [[ ! "${EVAL_TRUST_MAX}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--eval-trust-max must be a non-negative number, got: ${EVAL_TRUST_MAX}" >&2
    exit 1
fi

# Extract domain name for directory (e.g., hopper:hop -> hopper)
ENV_DIR=$(echo "$ENV_NAME" | cut -d':' -f1)
ROOT_BASE="${BASE_DIR}/${ENV_DIR}/bafcv3_tr2_dmc_4g"
ROOT_DIR="${ROOT_BASE}/critic_utd${CRITIC_UTD}/eval${EVAL_TRUST_MAX}"

echo "Starting BAFCv3-TR2 training with 4 seeds on shared GPUs 0,1,2,3"
echo "  Config: $CONF_FILE"
echo "  Environment: $ENV_NAME"
echo "  Root dir: $ROOT_DIR"
echo "  Num env steps: $NUM_ENV_STEPS"
echo "  Num checkpoints: $NUM_CHECKPOINTS"
echo "  Seeds: ${SEEDS[*]}"
echo "  critic_utd: $CRITIC_UTD"
echo "  Eval threshold: $EVAL_TRUST_MAX"
echo "  num_feature_coords: $NUM_FEATURE_COORDS"
echo "  metric_interval: $METRIC_INTERVAL"
echo "  rollout_skip_cap: $ROLLOUT_SKIP_CAP"
echo "  rollout_skip_eval_interval: $ROLLOUT_SKIP_EVAL_INTERVAL"
echo "  freeze_eval_samples: $FREEZE_EVAL_SAMPLES"
echo "  Actor layer norm: $ACTOR_USE_LN"
echo "  enable_critic_reweighting: $ENABLE_CRITIC_REWEIGHTING"
echo "  critic_reweighting_beta: $CRITIC_REWEIGHTING_BETA"
echo "  critic_reweighting_ridge: $CRITIC_REWEIGHTING_RIDGE"
echo "  critic_reweighting_solver_iters: $CRITIC_REWEIGHTING_SOLVER_ITERS"
echo "  critic_reweighting_max_weight: $CRITIC_REWEIGHTING_MAX_WEIGHT"
echo "  Eval rollout-skip gate: enabled"
echo "  Grad actor-extend gate: disabled"
echo ""

mkdir -p "${ROOT_DIR}/seed_${SEEDS[0]}" "${ROOT_DIR}/seed_${SEEDS[1]}" "${ROOT_DIR}/seed_${SEEDS[2]}" "${ROOT_DIR}/seed_${SEEDS[3]}"

# The configured seeds share GPUs 0,1,2,3 (different MASTER_PORT)
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29500 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_${SEEDS[0]}" \
    --conf_param "TrainerConfig.random_seed=${SEEDS[0]}" \
    --conf_param "bafcv3_tr2_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3TR2.critic_utd=${CRITIC_UTD}" \
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False" \
    --conf_param "BafcAlgorithmV3TR2.enable_critic_reweighting=${ENABLE_CRITIC_REWEIGHTING}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_beta=${CRITIC_REWEIGHTING_BETA}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_ridge=${CRITIC_REWEIGHTING_RIDGE}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_solver_iters=${CRITIC_REWEIGHTING_SOLVER_ITERS}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_max_weight=${CRITIC_REWEIGHTING_MAX_WEIGHT}" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_${SEEDS[0]}/out.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_${SEEDS[1]}" \
    --conf_param "TrainerConfig.random_seed=${SEEDS[1]}" \
    --conf_param "bafcv3_tr2_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3TR2.critic_utd=${CRITIC_UTD}" \
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False" \
    --conf_param "BafcAlgorithmV3TR2.enable_critic_reweighting=${ENABLE_CRITIC_REWEIGHTING}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_beta=${CRITIC_REWEIGHTING_BETA}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_ridge=${CRITIC_REWEIGHTING_RIDGE}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_solver_iters=${CRITIC_REWEIGHTING_SOLVER_ITERS}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_max_weight=${CRITIC_REWEIGHTING_MAX_WEIGHT}" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_${SEEDS[1]}/out.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29502 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_${SEEDS[2]}" \
    --conf_param "TrainerConfig.random_seed=${SEEDS[2]}" \
    --conf_param "bafcv3_tr2_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3TR2.critic_utd=${CRITIC_UTD}" \
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False" \
    --conf_param "BafcAlgorithmV3TR2.enable_critic_reweighting=${ENABLE_CRITIC_REWEIGHTING}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_beta=${CRITIC_REWEIGHTING_BETA}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_ridge=${CRITIC_REWEIGHTING_RIDGE}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_solver_iters=${CRITIC_REWEIGHTING_SOLVER_ITERS}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_max_weight=${CRITIC_REWEIGHTING_MAX_WEIGHT}" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_${SEEDS[2]}/out.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29503 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_${SEEDS[3]}" \
    --conf_param "TrainerConfig.random_seed=${SEEDS[3]}" \
    --conf_param "bafcv3_tr2_actor_use_ln=${ACTOR_USE_LN}" \
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False" \
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "TrainerConfig.debug_summaries=True" \
    --conf_param "TrainerConfig.rollout_skip_eval=True" \
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True" \
    --conf_param "BafcAlgorithmV3TR2.critic_utd=${CRITIC_UTD}" \
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}" \
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}" \
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}" \
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${FREEZE_EVAL_SAMPLES}" \
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True" \
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False" \
    --conf_param "BafcAlgorithmV3TR2.enable_critic_reweighting=${ENABLE_CRITIC_REWEIGHTING}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_beta=${CRITIC_REWEIGHTING_BETA}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_ridge=${CRITIC_REWEIGHTING_RIDGE}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_solver_iters=${CRITIC_REWEIGHTING_SOLVER_ITERS}" \
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_max_weight=${CRITIC_REWEIGHTING_MAX_WEIGHT}" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_${SEEDS[3]}/out.log" 2>&1 &
PID3=$!

echo "Seed ${SEEDS[0]} running on GPUs 0,1,2,3 port 29500 (PID: $PID0)"
echo "Seed ${SEEDS[1]} running on GPUs 0,1,2,3 port 29501 (PID: $PID1)"
echo "Seed ${SEEDS[2]} running on GPUs 0,1,2,3 port 29502 (PID: $PID2)"
echo "Seed ${SEEDS[3]} running on GPUs 0,1,2,3 port 29503 (PID: $PID3)"
echo ""
echo "To monitor: tail -f ${ROOT_DIR}/seed_*/out.log"
