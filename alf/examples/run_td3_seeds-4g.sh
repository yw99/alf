#!/bin/bash
# Launcher script for running TD3 with 2 seeds in parallel
# Seeds 0,1 share GPUs 0,1,2,3
# Each seed uses 4 GPUs via DDP for 4 parallel environments
#
# Usage: bash run_td3_seeds.sh [options]
#   -e, --env ENV_NAME        DMC environment (default: hopper:hop)
#   -d, --dir BASE_DIR        Base results directory (default: /workspace/results)
#   -n, --steps NUM_STEPS     Total environment steps (default: 1000000)
#   -h, --help                Show this help message
#
# Examples:
#   bash run_td3_seeds-4g.sh -e walker:walk
#   bash run_td3_seeds-4g.sh -e walker:walk -n 500000
#   bash run_td3_seeds-4g.sh --env hopper:hop --steps 2000000 --dir /my/results

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${SCRIPT_DIR}/td3_dmc_conf.py"
ENV_NAME="hopper:hop"
BASE_DIR="/workspace/results-4g"
NUM_ENV_STEPS=1000000

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
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/td3_dmc"

echo "Starting TD3 training with 2 seeds"
echo "  Config: $CONF_FILE"
echo "  Environment: $ENV_NAME"
echo "  Root dir: $ROOT_DIR"
echo "  Num env steps: $NUM_ENV_STEPS"
echo ""

mkdir -p "${ROOT_DIR}/seed_0" "${ROOT_DIR}/seed_1"

# Seeds 0,1 share GPUs 0,1,2,3 (different MASTER_PORT)
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29500 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_0" \
    --conf_param "TrainerConfig.random_seed=0" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_0/out.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_1" \
    --conf_param "TrainerConfig.random_seed=1" \
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_1/out.log" 2>&1 &
PID1=$!

echo "Seed 0 running on GPUs 0,1,2,3 port 29500 (PID: $PID0)"
echo "Seed 1 running on GPUs 0,1,2,3 port 29501 (PID: $PID1)"


# mkdir -p "${ROOT_DIR}/seed_2" "${ROOT_DIR}/seed_3"
#
# # Seeds 2,3 share GPUs 0,1,2,3 (different MASTER_PORT)
# CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29502 python -m alf.bin.train \
#     --conf "$CONF_FILE" \
#     --root_dir "${ROOT_DIR}/seed_2" \
#     --conf_param "TrainerConfig.random_seed=2" \
#     --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
#     --conf_param "create_environment.env_name='${ENV_NAME}'" \
#     --distributed multi-gpu \
#     > "${ROOT_DIR}/seed_2/out.log" 2>&1 &
# PID2=$!
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29503 python -m alf.bin.train \
#     --conf "$CONF_FILE" \
#     --root_dir "${ROOT_DIR}/seed_3" \
#     --conf_param "TrainerConfig.random_seed=3" \
#     --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}" \
#     --conf_param "create_environment.env_name='${ENV_NAME}'" \
#     --distributed multi-gpu \
#     > "${ROOT_DIR}/seed_3/out.log" 2>&1 &
# PID3=$!
#
# echo "Seed 2 running on GPUs 0,1,2,3 port 29502 (PID: $PID2)"
# echo "Seed 3 running on GPUs 0,1,2,3 port 29503 (PID: $PID3)"

echo ""
echo "To monitor: tail -f ${ROOT_DIR}/seed_*/out.log"
