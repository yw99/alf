#!/bin/bash
# Launcher script for running BafcV3 with 2 seeds in parallel
# Each seed uses 2 GPUs via DDP for 2 parallel environments

CONF_FILE="alf/examples/bafcv3_dmc_conf.py"
ROOT_DIR=${1:-"/workspace/results/bafcv3_dmc"}

echo "Starting BafcV3 training with 2 seeds"
echo "  Config: $CONF_FILE"
echo "  Root dir: $ROOT_DIR"
echo ""

mkdir -p "${ROOT_DIR}/seed_0" "${ROOT_DIR}/seed_1"

CUDA_VISIBLE_DEVICES=0,1 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_0" \
    --conf_param "TrainerConfig.random_seed=0" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_0/out.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=2,3 python -m alf.bin.train \
    --conf "$CONF_FILE" \
    --root_dir "${ROOT_DIR}/seed_1" \
    --conf_param "TrainerConfig.random_seed=1" \
    --distributed multi-gpu \
    > "${ROOT_DIR}/seed_1/out.log" 2>&1 &
PID1=$!

echo "Seed 0 running on GPUs 0,1 (PID: $PID0)"
echo "Seed 1 running on GPUs 2,3 (PID: $PID1)"
echo ""
echo "To monitor: tail -f ${ROOT_DIR}/seed_*/out.log"
