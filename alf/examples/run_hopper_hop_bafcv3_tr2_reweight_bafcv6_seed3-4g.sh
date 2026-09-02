#!/bin/bash
# Launch hopper:hop BAFCv3_TR2 with critic reweighting and BAFCv6 on one seed.
# BAFCv6 is the BAFCv3-style condition augmented with snapshot-feature critic
# sample reweighting; unlike TR2, it has no trust-metric rollout gate.
# Both jobs use all four configured GPUs through DDP and run in parallel.
#
# Shared settings match run_hopper_hop_rlpd_bafcv3_seed01-4g.sh:
# pairing=False, K_actor=8, critic_utd=11, and random critic TD targets with
# M_target=1. TR2 also keeps that launcher's trust-gate settings, with critic
# reweighting enabled. BAFCv6 uses the reweighting settings from
# run_bafcv3_tr_v6_random_target_sweep_seed01-4g.sh.
#
# Usage: bash run_hopper_hop_bafcv3_tr2_reweight_bafcv6_seed3-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 800000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29610)
#       --seed SEED          Random seed for both jobs (default: 3)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_hopper_hop_bafcv3_tr2_reweight_bafcv6_seed3-4g.sh --dry-run

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TR2_CONF_FILE="${SCRIPT_DIR}/bafcv3_tr2_dmc_conf.py"
V6_CONF_FILE="${SCRIPT_DIR}/bafcv6_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="hopper:hop"
SEED=3
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29610
DRY_RUN=False

CRITIC_UTD=11
NUM_UPDATES_PER_TRAIN_ITER=12
NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITICS_FOR_ACTOR=8
NUM_SAMPLED_CRITIC_TARGETS=1
DEBUG_SUMMARIES=True

TR2_EVAL_TRUST_MAX=30.0
TR2_EVAL_TRUST_MAX_DECAY=True
TR2_NUM_FEATURE_COORDS=4
TR2_METRIC_INTERVAL=8
TR2_ROLLOUT_SKIP_CAP=3
TR2_ROLLOUT_SKIP_EVAL_INTERVAL=60
TR2_FREEZE_EVAL_SAMPLES=False
TR2_CRITIC_REWEIGHTING_BETA=None
TR2_CRITIC_REWEIGHTING_RIDGE=None
TR2_CRITIC_REWEIGHTING_SOLVER_ITERS=1
TR2_CRITIC_REWEIGHTING_MAX_WEIGHT=10.0

V6_CRITIC_REWEIGHTING_SOLVER="lbfgs_logits"
V6_CRITIC_REWEIGHTING_SOLVER_ITERS=1
V6_CRITIC_REWEIGHTING_NUM_FEATURE_COORDS=32
V6_CRITIC_REWEIGHTING_NUM_TARGET_OBS=128
V6_CRITIC_REWEIGHTING_MAX_WEIGHT=10.0

print_help() {
    sed -n '/^# Usage:/,/^# Example:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -n|--steps)
            NUM_ENV_STEPS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --checkpoints)
            NUM_CHECKPOINTS="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=True
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use -h or --help for usage information" >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
for conf_file in "${TR2_CONF_FILE}" "${V6_CONF_FILE}"; do
    if [[ ! -f "${conf_file}" ]]; then
        echo "Config file not found: ${conf_file}" >&2
        exit 1
    fi
done
if [[ ! "${NUM_ENV_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--steps must be a positive integer, got: ${NUM_ENV_STEPS}" >&2
    exit 1
fi
if [[ ! "${NUM_CHECKPOINTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--checkpoints must be a positive integer, got: ${NUM_CHECKPOINTS}" >&2
    exit 1
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a nonnegative integer, got: ${SEED}" >&2
    exit 1
fi
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 1 > 65535 )); then
    echo "--base-port must leave room for two valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME//:/_}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_tr2_reweight_bafcv6_seed${SEED}_4g"
TR2_RUN_DIR="${ROOT_DIR}/bafcv3_tr2_reweight/fixed_pairingFalse_num_sampled_critic${NUM_SAMPLED_CRITICS_FOR_ACTOR}/critic_utd${CRITIC_UTD}/seed_${SEED}"
V6_RUN_DIR="${ROOT_DIR}/bafcv6_reweight/fixed_pairingFalse_num_sampled_critic${NUM_SAMPLED_CRITICS_FOR_ACTOR}/critic_utd${CRITIC_UTD}/seed_${SEED}"

cat <<EOF
Starting hopper:hop TR2-reweight/BAFCv6 comparison
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Seed: ${SEED}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  critic_utd: ${CRITIC_UTD}
  actor_critic_pairing: False
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  num_sampled_critics_for_actor: ${NUM_SAMPLED_CRITICS_FOR_ACTOR}
  random critic targets: True
  num_sampled_critic_targets: ${NUM_SAMPLED_CRITIC_TARGETS}
  TR2 critic reweighting: True
  BAFCv6 critic reweighting: True
  GPUs per job: ${GPUS}
  Ports: ${BASE_PORT}, $((BASE_PORT + 1))
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

COMMON_ARGS=(
    --conf_param "TrainerConfig.random_seed=${SEED}"
    --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
    --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
    --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
    --conf_param "TrainerConfig.num_updates_per_train_iter=${NUM_UPDATES_PER_TRAIN_ITER}"
    --conf_param "TrainerConfig.debug_summaries=${DEBUG_SUMMARIES}"
    --conf_param "make_ddp_performer.find_unused_parameters=True"
    --conf_param "create_environment.env_name='${ENV_NAME}'"
)

TR2_COMMAND=(
    "${PYTHON_BIN}" -m alf.bin.train
    --conf "${TR2_CONF_FILE}"
    --root_dir "${TR2_RUN_DIR}"
    "${COMMON_ARGS[@]}"
    --conf_param "TrainerConfig.rollout_skip_eval=True"
    --conf_param "TrainerConfig.rollout_skip_eval_interval=${TR2_ROLLOUT_SKIP_EVAL_INTERVAL}"
    --conf_param "BafcAlgorithmV3TR2.critic_utd=${CRITIC_UTD}"
    --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True"
    --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${TR2_EVAL_TRUST_MAX}"
    --conf_param "BafcAlgorithmV3TR2.enable_eval_trust_max_decay=${TR2_EVAL_TRUST_MAX_DECAY}"
    --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${TR2_NUM_FEATURE_COORDS}"
    --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${TR2_METRIC_INTERVAL}"
    --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${TR2_ROLLOUT_SKIP_CAP}"
    --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=${TR2_FREEZE_EVAL_SAMPLES}"
    --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True"
    --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False"
    --conf_param "BafcAlgorithmV3TR2.enable_critic_reweighting=True"
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_beta=${TR2_CRITIC_REWEIGHTING_BETA}"
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_ridge=${TR2_CRITIC_REWEIGHTING_RIDGE}"
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_solver_iters=${TR2_CRITIC_REWEIGHTING_SOLVER_ITERS}"
    --conf_param "BafcAlgorithmV3TR2.critic_reweighting_max_weight=${TR2_CRITIC_REWEIGHTING_MAX_WEIGHT}"
    --conf_param "bafcv3_tr2_actor_use_ln=False"
    --conf_param "bafcv3_tr2_actor_critic_pairing=False"
    --conf_param "bafcv3_tr2_num_actor_critic=${NUM_ACTOR_CRITIC}"
    --conf_param "bafcv3_tr2_num_sampled_critics_for_actor=${NUM_SAMPLED_CRITICS_FOR_ACTOR}"
    --conf_param "bafcv3_tr2_use_random_critic_targets=True"
    --conf_param "bafcv3_tr2_num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
    --distributed multi-gpu
)

V6_COMMAND=(
    "${PYTHON_BIN}" -m alf.bin.train
    --conf "${V6_CONF_FILE}"
    --root_dir "${V6_RUN_DIR}"
    "${COMMON_ARGS[@]}"
    --conf_param "BafcAlgorithmV6.critic_utd=${CRITIC_UTD}"
    --conf_param "BafcAlgorithmV6.enable_critic_reweighting=True"
    --conf_param "BafcAlgorithmV6.critic_reweighting_solver='${V6_CRITIC_REWEIGHTING_SOLVER}'"
    --conf_param "BafcAlgorithmV6.critic_reweighting_solver_iters=${V6_CRITIC_REWEIGHTING_SOLVER_ITERS}"
    --conf_param "BafcAlgorithmV6.critic_reweighting_num_feature_coords=${V6_CRITIC_REWEIGHTING_NUM_FEATURE_COORDS}"
    --conf_param "BafcAlgorithmV6.critic_reweighting_num_target_obs=${V6_CRITIC_REWEIGHTING_NUM_TARGET_OBS}"
    --conf_param "BafcAlgorithmV6.critic_reweighting_max_weight=${V6_CRITIC_REWEIGHTING_MAX_WEIGHT}"
    --conf_param "bafcv6_actor_use_ln=False"
    --conf_param "bafcv6_actor_critic_pairing=False"
    --conf_param "bafcv6_num_actor_critic=${NUM_ACTOR_CRITIC}"
    --conf_param "bafcv6_num_sampled_critics_for_actor=${NUM_SAMPLED_CRITICS_FOR_ACTOR}"
    --conf_param "bafcv6_use_random_critic_targets=True"
    --conf_param "bafcv6_num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
    --distributed multi-gpu
)

launch_job() {
    local name="$1"
    local port="$2"
    local run_dir="$3"
    shift 3
    local -a command=("$@")

    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${port}"
        printf '%q ' "${command[@]}"
        printf '> %q 2>&1 &\n' "${run_dir}/out.log"
        return
    fi

    mkdir -p "${run_dir}"
    CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${port}" \
        "${command[@]}" > "${run_dir}/out.log" 2>&1 &
    local pid=$!
    PIDS+=("${pid}")
    echo "  ${name}: port ${port}, PID ${pid}"
    echo "    Log: ${run_dir}/out.log"
}

PIDS=()
launch_job "BAFCv3_TR2 reweight, seed ${SEED}" "${BASE_PORT}" \
    "${TR2_RUN_DIR}" "${TR2_COMMAND[@]}"
launch_job "BAFCv6 reweight, seed ${SEED}" "$((BASE_PORT + 1))" \
    "${V6_RUN_DIR}" "${V6_COMMAND[@]}"

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched two 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor TR2: tail -f ${TR2_RUN_DIR}/out.log"
echo "To monitor BAFCv6: tail -f ${V6_RUN_DIR}/out.log"
echo "Results: ${ROOT_DIR}"
