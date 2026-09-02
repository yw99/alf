#!/bin/bash
# Launch BAFCv3 on hopper:hop for seeds 0, 2, and 3 with critic_utd=3 and either one
# or eight sampled critics per actor. Each job uses all configured GPUs through
# DDP, and the six jobs run in parallel on unique master ports.
#
# Usage: bash run_hopper_hop_bafcv3_utd3_k1_seeds0123-4g.sh [options]
#   -d, --dir BASE_DIR       Base results directory (default: /workspace/alf_results)
#   -n, --steps NUM_STEPS    Total environment steps per job (default: 800000)
#       --gpus CSV           Comma-separated GPU ids (default: 0,1,2,3)
#       --checkpoints N      Number of checkpoints (default: 10)
#       --base-port PORT     First DDP master port (default: 29520)
#       --seeds CSV          Comma-separated random seeds (default: 0,2,3)
#       --sampled-critics CSV
#                            Critics sampled per actor (default: 1,8)
#       --dry-run            Print commands without launching jobs
#   -h, --help               Show this help message
#
# Example:
#   bash run_hopper_hop_bafcv3_utd3_k1_seeds0123-4g.sh --dry-run

set -euo pipefail

# Use headless EGL rendering for dm_control unless explicitly overridden.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Use deterministic CUBLAS workspace for reproducibility unless explicitly overridden.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAFCV3_CONF="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ENV_NAME="hopper:hop"
BASE_DIR="/workspace/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
GPUS="0,1,2,3"
BASE_PORT=29520
DRY_RUN=False
SEEDS=(0 2 3)

BAFCV3_CRITIC_UTD=3
# Inherited from bafcv3_dmc_conf.py; kept here for output-path provenance.
BAFCV3_UPDATES_PER_ITER=12
BAFCV3_NUM_ACTOR_CRITIC=10
BAFCV3_NUM_SAMPLED_CRITICS=(1 8)
BAFCV3_NUM_SAMPLED_CRITIC_TARGETS=1
BAFCV3_ACTOR_USE_LN=False
BAFCV3_DEBUG_SUMMARIES=True

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
        --seeds)
            IFS=',' read -r -a SEEDS <<< "$2"
            shift 2
            ;;
        --sampled-critics)
            IFS=',' read -r -a BAFCV3_NUM_SAMPLED_CRITICS <<< "$2"
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
    echo "Set PYTHON_BIN to a working interpreter, or create ${REPO_ROOT}/.venv." >&2
    exit 1
fi
if [[ ! -f "${BAFCV3_CONF}" ]]; then
    echo "Config file not found: ${BAFCV3_CONF}" >&2
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
if (( ${#SEEDS[@]} == 0 )); then
    echo "--seeds must contain at least one seed" >&2
    exit 1
fi
for seed in "${SEEDS[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "--seeds must be a comma-separated list of nonnegative integers, got: ${SEEDS[*]}" >&2
        exit 1
    fi
done
if (( ${#BAFCV3_NUM_SAMPLED_CRITICS[@]} == 0 )); then
    echo "--sampled-critics must contain at least one value" >&2
    exit 1
fi
for num_sampled_critics in "${BAFCV3_NUM_SAMPLED_CRITICS[@]}"; do
    if [[ ! "${num_sampled_critics}" =~ ^[1-9][0-9]*$ ]] || \
            (( num_sampled_critics > BAFCV3_NUM_ACTOR_CRITIC )); then
        echo "--sampled-critics values must be integers from 1 to ${BAFCV3_NUM_ACTOR_CRITIC}, got: ${BAFCV3_NUM_SAMPLED_CRITICS[*]}" >&2
        exit 1
    fi
done
NUM_JOBS=$((${#SEEDS[@]} * ${#BAFCV3_NUM_SAMPLED_CRITICS[@]}))
if [[ ! "${BASE_PORT}" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + NUM_JOBS - 1 > 65535 )); then
    echo "--base-port must leave room for ${NUM_JOBS} valid ports, got: ${BASE_PORT}" >&2
    exit 1
fi

# Reuse the comparison hierarchy. Include the update count so these runs cannot
# resume the existing UTD3/K=8 runs that used four updates per train iteration.
ENV_DIR="${ENV_NAME//:/_}"
ROOT_DIR="${BASE_DIR}/${ENV_DIR}/rlpd_bafcv3_comparison_4g"

cat <<EOF
Starting hopper:hop BAFCv3 sampled-critic sweep
  Config: ${BAFCV3_CONF}
  Environment: ${ENV_NAME}
  Root dir: ${ROOT_DIR}
  Num env steps: ${NUM_ENV_STEPS}
  Num checkpoints: ${NUM_CHECKPOINTS}
  Seeds: ${SEEDS[*]}
  GPUs per job: ${GPUS}
  critic_utd: ${BAFCV3_CRITIC_UTD}
  Updates per train iteration: ${BAFCV3_UPDATES_PER_ITER} (config default)
  Sampled critics per actor: ${BAFCV3_NUM_SAMPLED_CRITICS[*]}
  Dry run: ${DRY_RUN}
EOF
echo ""

cd "${REPO_ROOT}"

PIDS=()
port_offset=0
for num_sampled_critics in "${BAFCV3_NUM_SAMPLED_CRITICS[@]}"; do
    condition="bafcv3/fixed_pairingFalse_num_sampled_critic${num_sampled_critics}/critic_utd${BAFCV3_CRITIC_UTD}/num_updates_per_train_iter${BAFCV3_UPDATES_PER_ITER}"

    for seed in "${SEEDS[@]}"; do
        master_port=$((BASE_PORT + port_offset))
        run_dir="${ROOT_DIR}/${condition}/seed_${seed}"
        command=(
            "${PYTHON_BIN}" -m alf.bin.train
            --conf "${BAFCV3_CONF}"
            --root_dir "${run_dir}"
            --conf_param "TrainerConfig.random_seed=${seed}"
            --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
            --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
            --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
            --conf_param "TrainerConfig.debug_summaries=${BAFCV3_DEBUG_SUMMARIES}"
            --conf_param "make_ddp_performer.find_unused_parameters=True"
            --conf_param "create_environment.env_name='${ENV_NAME}'"
            --conf_param "BafcAlgorithmV3.critic_utd=${BAFCV3_CRITIC_UTD}"
            --conf_param "bafcv3_actor_use_ln=${BAFCV3_ACTOR_USE_LN}"
            --conf_param "bafcv3_actor_critic_pairing=False"
            --conf_param "bafcv3_num_actor_critic=${BAFCV3_NUM_ACTOR_CRITIC}"
            --conf_param "bafcv3_num_sampled_critics_for_actor=${num_sampled_critics}"
            --conf_param "bafcv3_use_random_critic_targets=True"
            --conf_param "bafcv3_num_sampled_critic_targets=${BAFCV3_NUM_SAMPLED_CRITIC_TARGETS}"
            --distributed multi-gpu
        )

        if [[ "${DRY_RUN}" == "True" ]]; then
            printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "${master_port}"
            printf '%q ' "${command[@]}"
            printf '> %q 2>&1 &\n' "${run_dir}/out.log"
        else
            mkdir -p "${run_dir}"
            CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${master_port}" \
                "${command[@]}" > "${run_dir}/out.log" 2>&1 &
            pid=$!
            PIDS+=("${pid}")
            echo "  BAFCv3 K=${num_sampled_critics}, seed ${seed}: port ${master_port}, PID ${pid}"
            echo "    Log: ${run_dir}/out.log"
        fi

        ((port_offset += 1))
    done
done

echo ""
if [[ "${DRY_RUN}" == "True" ]]; then
    echo "Dry run complete; no jobs were launched."
else
    echo "Launched ${NUM_JOBS} BAFCv3 4-GPU jobs: ${PIDS[*]}"
    echo "Launcher is not waiting for completion."
fi
echo "To monitor: tail -f ${ROOT_DIR}/bafcv3/fixed_pairingFalse_num_sampled_critic*/critic_utd${BAFCV3_CRITIC_UTD}/num_updates_per_train_iter${BAFCV3_UPDATES_PER_ITER}/seed_*/out.log"
echo "Results: ${ROOT_DIR}/bafcv3/fixed_pairingFalse_num_sampled_critic*/critic_utd${BAFCV3_CRITIC_UTD}/num_updates_per_train_iter${BAFCV3_UPDATES_PER_ITER}"
