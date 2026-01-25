#!/bin/bash
# Launcher script for running TD3v2 (original paper version) with 4 seeds in parallel
# Seeds 0,1 share GPUs 0,1; Seeds 2,3 share GPUs 2,3
# Each seed uses 2 GPUs via DDP for 2 parallel environments
#
# Usage: bash run_td3v2_seeds.sh [env_name] [root_dir]
#   env_name: DMC environment (default: hopper:hop)
#   root_dir: Base results directory (default: /workspace/results)

CONF_FILE="alf/examples/td3v2_dmc_conf.py"
ENV_NAME=${1:-"hopper:hop"}
BASE_DIR=${2:-"/workspace/results"}

# Extract domain name for directory (e.g., hopper:hop -> hopper)
ENV_DIR=$(echo "$ENV_NAME" | cut -d':' -f1)
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/td3v2_dmc"

echo "Starting TD3v2 training with 4 seeds"
echo "  Config: $CONF_FILE"
echo "  Environment: $ENV_NAME"
echo "  Root dir: $ROOT_DIR"
echo ""

mkdir -p "${ROOT_DIR}/seed_0" "${ROOT_DIR}/seed_1" "${ROOT_DIR}/seed_2" "${ROOT_DIR}/seed_3"

# Seeds 0,1 share GPUs 0,1 (different MASTER_PORT)
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29500 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_0" \
    --conf_param "TrainerConfig.random_seed=0" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_0/out.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29501 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_1" \
    --conf_param "TrainerConfig.random_seed=1" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_1/out.log" 2>&1 &
PID1=$!

# Seeds 2,3 share GPUs 2,3 (different MASTER_PORT)
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29502 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_2" \
    --conf_param "TrainerConfig.random_seed=2" \
    --conf_param "create_environment.env_name='${ENV_NAME}'" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_2/out.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29503 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_3" \
    --conf_param "TrainerConfig.random_seed=3" \
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
