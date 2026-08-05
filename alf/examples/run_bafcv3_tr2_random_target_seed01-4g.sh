#!/bin/bash
# Launch the BAFCv3_TR2 random-critic-TD-target condition on seeds 0 and 1.
# Each job uses all configured GPUs through DDP; both jobs run in parallel.
#
# Usage: bash run_bafcv3_tr2_random_target_seed01-4g.sh [options]
#   -e, --env ENV_NAME       DMC environment (default: humanoid:walk)
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps (default: 600000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29500)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message

set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONF_FILE="${SCRIPT_DIR}/bafcv3_tr2_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="humanoid:walk"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=600000
NUM_CHECKPOINTS=10
NUM_ACTOR_CRITIC=10
NUM_SAMPLED_CRITICS_FOR_ACTOR=8
NUM_SAMPLED_CRITIC_TARGETS=1
CRITIC_UTD=11
NUM_UPDATES_PER_TRAIN_ITER=12
EVAL_TRUST_MAX=15
EVAL_TRUST_MAX_DECAY=True
NUM_FEATURE_COORDS=4
METRIC_INTERVAL=8
ROLLOUT_SKIP_CAP=3
ROLLOUT_SKIP_EVAL_INTERVAL=60
GPUS="0,1,2,3"
BASE_PORT=29500
DRY_RUN=False
SEEDS=(0 1)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env) ENV_NAME="$2"; shift 2 ;;
        -d|--dir) BASE_DIR="$2"; shift 2 ;;
        -n|--steps) NUM_ENV_STEPS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --checkpoints) NUM_CHECKPOINTS="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=True; shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
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
if [[ ! -f "${CONF_FILE}" ]]; then
    echo "Config file not found: ${CONF_FILE}" >&2
    exit 1
fi
if [[ ! "${NUM_ENV_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--steps must be a positive integer, got: ${NUM_ENV_STEPS}" >&2
    exit 1
fi
if [[ ! "${NUM_CHECKPOINTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--checkpoints must be a positive integer, got: ${NUM_CHECKPOINTS}" >&2
    exit 1
fi
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + 1 > 65535 )); then
    echo "--base-port must leave room for two valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

ENV_DIR="${ENV_NAME%%:*}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/bafcv3_tr2_random_target_4g/num_actor_critic${NUM_ACTOR_CRITIC}_num_sampled_critics_for_actor${NUM_SAMPLED_CRITICS_FOR_ACTOR}_num_sampled_critic_targets${NUM_SAMPLED_CRITIC_TARGETS}/critic_utd${CRITIC_UTD}/eval${EVAL_TRUST_MAX}_decay"

cat <<EOF
Starting BAFCv3_TR2 random-target training
  Config: ${CONF_FILE}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  actor_critic_pairing: False
  num_actor_critic: ${NUM_ACTOR_CRITIC}
  num_sampled_critics_for_actor: ${NUM_SAMPLED_CRITICS_FOR_ACTOR}
  use_random_critic_targets: True
  num_sampled_critic_targets: ${NUM_SAMPLED_CRITIC_TARGETS}
  critic_utd: ${CRITIC_UTD}
  Eval trust max: ${EVAL_TRUST_MAX}
  Eval trust max decay: ${EVAL_TRUST_MAX_DECAY}
  GPUs per job: ${GPUS}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"
PIDS=()
for seed_index in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$seed_index]}"
    MASTER_PORT=$((BASE_PORT + seed_index))
    RUN_DIR="${ROOT_DIR}/seed_${SEED}"
    COMMAND=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${CONF_FILE}"
        --root_dir "${RUN_DIR}"
        --conf_param "TrainerConfig.random_seed=${SEED}"
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
        --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
        --conf_param "TrainerConfig.num_updates_per_train_iter=${NUM_UPDATES_PER_TRAIN_ITER}"
        --conf_param "TrainerConfig.debug_summaries=True"
        --conf_param "TrainerConfig.rollout_skip_eval=True"
        --conf_param "TrainerConfig.rollout_skip_eval_interval=${ROLLOUT_SKIP_EVAL_INTERVAL}"
        --conf_param "BafcAlgorithmV3TR2.critic_utd=${CRITIC_UTD}"
        --conf_param "bafcv3_tr2_actor_critic_pairing=False"
        --conf_param "bafcv3_tr2_num_actor_critic=${NUM_ACTOR_CRITIC}"
        --conf_param "bafcv3_tr2_num_sampled_critics_for_actor=${NUM_SAMPLED_CRITICS_FOR_ACTOR}"
        --conf_param "bafcv3_tr2_use_random_critic_targets=True"
        --conf_param "bafcv3_tr2_num_sampled_critic_targets=${NUM_SAMPLED_CRITIC_TARGETS}"
        --conf_param "bafcv3_tr2_actor_use_ln=False"
        --conf_param "BafcAlgorithmV3TR2.monitor_trust_metrics=True"
        --conf_param "BafcAlgorithmV3TR2.eval_trust_max=${EVAL_TRUST_MAX}"
        --conf_param "BafcAlgorithmV3TR2.enable_eval_trust_max_decay=${EVAL_TRUST_MAX_DECAY}"
        --conf_param "BafcAlgorithmV3TR2.trust_metric_num_feature_coords=${NUM_FEATURE_COORDS}"
        --conf_param "BafcAlgorithmV3TR2.trust_metric_update_interval=${METRIC_INTERVAL}"
        --conf_param "BafcAlgorithmV3TR2.eval_gate_max_consecutive_rollout_skips=${ROLLOUT_SKIP_CAP}"
        --conf_param "BafcAlgorithmV3TR2.freeze_eval_samples=False"
        --conf_param "BafcAlgorithmV3TR2.enable_eval_rollout_skip_gate=True"
        --conf_param "BafcAlgorithmV3TR2.enable_grad_actor_extend_gate=False"
        --conf_param "BafcAlgorithmV3TR2.enable_critic_reweighting=False"
        --conf_param "BafcAlgorithmV3TR2.critic_reweighting_beta=None"
        --conf_param "BafcAlgorithmV3TR2.critic_reweighting_ridge=None"
        --conf_param "BafcAlgorithmV3TR2.critic_reweighting_solver_iters=1"
        --conf_param "BafcAlgorithmV3TR2.critic_reweighting_max_weight=10.0"
        --conf_param "create_environment.env_name='${ENV_NAME}'"
        --distributed multi-gpu
    )

    if [[ "${DRY_RUN}" == "True" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${MASTER_PORT}"
        printf '%q ' "${COMMAND[@]}"
        printf '> %q 2>&1 &\n' "${RUN_DIR}/out.log"
    else
        mkdir -p "${RUN_DIR}"
        CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" \
            "${COMMAND[@]}" > "${RUN_DIR}/out.log" 2>&1 &
        PIDS+=("$!")
        echo "seed ${SEED}: port ${MASTER_PORT}, PID ${PIDS[-1]}"
        echo "  Log: ${RUN_DIR}/out.log"
    fi
done

if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched two BAFCv3_TR2 random-target 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/seed_*/out.log"
echo "Results: ${ROOT_DIR}"
