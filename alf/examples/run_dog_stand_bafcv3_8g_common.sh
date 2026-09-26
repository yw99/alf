#!/bin/bash
# Shared implementation for the two dog:stand 8-GPU launchers. Source only.

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAFCV3_CONF="${SCRIPT_DIR}/bafcv3_dmc_conf.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
BASE_DIR="/root/alf_results"
NUM_ENV_STEPS=800000
NUM_CHECKPOINTS=10
INITIAL_COLLECT_STEPS=""
GPUS="0,1,2,3,4,5,6,7"
DRY_RUN=False

case "${BAFCV3_SCENARIO}" in
    four_utd11)
        ROOT_NAME="bafcv3_seed0123_8g_eval_samples_frozen"
        BASE_PORT=29600
        BASE_HTTP_PORT=18080
        JOB_COUNT=4
        ;;
    mixed_utd11_utd3)
        ROOT_NAME="bafcv3_utd11_utd3_seed0123_8g_eval_samples_frozen"
        BASE_PORT=29620
        BASE_HTTP_PORT=18100
        JOB_COUNT=8
        ;;
    *)
        echo "Unknown scenario: ${BAFCV3_SCENARIO}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

print_help() {
    cat <<EOF
Usage: bash $(basename "${BASH_SOURCE[1]}") [options]
  -d, --dir BASE_DIR          Base results directory (default: /root/alf_results)
  -n, --steps NUM_STEPS       Job-wide environment steps (default: 800000)
      --gpus CSV              GPU ids (default: 0,1,2,3,4,5,6,7)
      --checkpoints N         Number of checkpoints (default: 10)
      --initial-collect-steps N  Job-wide initial collection steps (default: config value 10000)
      --base-port PORT        First DDP master port (default: ${BASE_PORT})
      --base-http-port PORT   First HTTP server port (default: ${BASE_HTTP_PORT})
      --dry-run               Print commands without launching
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir) BASE_DIR="$2"; shift 2 ;;
        -n|--steps) NUM_ENV_STEPS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --checkpoints) NUM_CHECKPOINTS="$2"; shift 2 ;;
        --initial-collect-steps) INITIAL_COLLECT_STEPS="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; shift 2 ;;
        --base-http-port) BASE_HTTP_PORT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=True; shift ;;
        -h|--help) print_help; return 0 2>/dev/null || exit 0 ;;
        *) echo "Unknown option: $1" >&2; print_help >&2; return 1 2>/dev/null || exit 1 ;;
    esac
done

if [[ ! -x "${PYTHON_BIN}" || ! -f "${BAFCV3_CONF}" ]]; then
    echo "Missing Python interpreter or BAFCv3 config" >&2
    return 1 2>/dev/null || exit 1
fi
for value in "${NUM_ENV_STEPS}" "${NUM_CHECKPOINTS}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Steps and checkpoints must be positive integers" >&2
        return 1 2>/dev/null || exit 1
    fi
done
if [[ -n "${INITIAL_COLLECT_STEPS}" && ! "${INITIAL_COLLECT_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--initial-collect-steps must be a positive integer" >&2
    return 1 2>/dev/null || exit 1
fi
for value in "${BASE_PORT}" "${BASE_HTTP_PORT}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] || (( value + JOB_COUNT - 1 > 65535 )); then
        echo "Port base must leave room for ${JOB_COUNT} valid ports" >&2
        return 1 2>/dev/null || exit 1
    fi
done

ROOT_DIR="${BASE_DIR}/dog_stand/${ROOT_NAME}/fixed_pairingFalse_num_sampled_critic8"
echo "Launching ${JOB_COUNT} BAFCv3 dog:stand jobs: ${BAFCV3_SCENARIO}"
echo "  Root: ${ROOT_DIR}"
echo "  GPUs per job: ${GPUS}; job-wide steps: ${NUM_ENV_STEPS}"
echo "  Initial collection: ${INITIAL_COLLECT_STEPS:-config default}; eval samples: frozen; dry run: ${DRY_RUN}"

cd "${REPO_ROOT}"
PIDS=()

launch_job() {
    local seed="$1" critic_utd="$2" updates_per_iter="$3" index="$4"
    local run_dir="${ROOT_DIR}/critic_utd${critic_utd}/seed_${seed}"
    local -a command=(
        "${PYTHON_BIN}" -m alf.bin.train
        --conf "${BAFCV3_CONF}"
        --root_dir "${run_dir}"
        --port "$((BASE_HTTP_PORT + index))"
        --conf_param "TrainerConfig.random_seed=${seed}"
        --conf_param "TrainerConfig.confirm_checkpoint_upon_crash=False"
        --conf_param "TrainerConfig.num_checkpoints=${NUM_CHECKPOINTS}"
        --conf_param "TrainerConfig.num_env_steps=${NUM_ENV_STEPS}"
        --conf_param "TrainerConfig.num_updates_per_train_iter=${updates_per_iter}"
        --conf_param "TrainerConfig.debug_summaries=True"
        --conf_param "make_ddp_performer.find_unused_parameters=True"
        --conf_param "create_environment.env_name='dog:stand'"
        --conf_param "BafcAlgorithmV3.critic_utd=${critic_utd}"
        --conf_param "bafcv3_eval_samples_source='frozen'"
        --conf_param "bafcv3_actor_use_ln=False"
        --conf_param "bafcv3_actor_critic_pairing=False"
        --conf_param "bafcv3_num_actor_critic=10"
        --conf_param "bafcv3_num_sampled_critics_for_actor=8"
        --conf_param "bafcv3_use_random_critic_targets=True"
        --conf_param "bafcv3_num_sampled_critic_targets=1"
    )
    if [[ -n "${INITIAL_COLLECT_STEPS}" ]]; then
        command+=(--conf_param "TrainerConfig.initial_collect_steps=${INITIAL_COLLECT_STEPS}")
    fi
    command+=(--distributed multi-gpu)

    if [[ "${DRY_RUN}" == True ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q ' "${GPUS}" "$((BASE_PORT + index))"
        printf '%q ' "${command[@]}"
        printf '> %q 2>&1 &\n' "${run_dir}/out.log"
        return
    fi

    mkdir -p "${run_dir}"
    CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="$((BASE_PORT + index))" \
        "${command[@]}" > "${run_dir}/out.log" 2>&1 &
    PIDS+=("$!")
    echo "  critic_utd=${critic_utd} seed=${seed}: PID ${PIDS[-1]}, DDP port $((BASE_PORT + index)), HTTP port $((BASE_HTTP_PORT + index))"
    echo "    ${run_dir}/out.log"
}

index=0
for seed in 0 1 2 3; do
    launch_job "${seed}" 11 12 "${index}"
    index=$((index + 1))
done
if [[ "${BAFCV3_SCENARIO}" == mixed_utd11_utd3 ]]; then
    for seed in 0 1 2 3; do
        launch_job "${seed}" 3 4 "${index}"
        index=$((index + 1))
    done
fi

if [[ "${DRY_RUN}" == True ]]; then
    echo "Dry run complete; no jobs launched."
else
    echo "Launched ${JOB_COUNT} jobs: ${PIDS[*]}"
    echo "Launcher does not wait for completion."
fi
echo "Results: ${ROOT_DIR}"
