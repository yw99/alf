#!/bin/bash
# Submit with:
#   sbatch hpc_sbatchs/bafc_dmc_16env.sbatch

#SBATCH --job-name=alf-bafc-dmc-16env
#SBATCH --account=torch_pr_741_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=96:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${SUBMIT_DIR}"
if [[ ! -d "${REPO_ROOT}/alf" && -d "${SUBMIT_DIR}/.." && -d "${SUBMIT_DIR}/../alf" ]]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
fi

PYTHON_BIN="/scratch/${USER}/conda-envs/alf-bafc/bin/python"
CONF_FILE="${REPO_ROOT}/alf/examples/bafcv3_dmc_conf.py"
ROOT_DIR="/scratch/${USER}/alf_results/bafc_v3_dmc_debug"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found at ${PYTHON_BIN}"
    exit 1
fi
if [[ ! -f "${CONF_FILE}" ]]; then
    echo "Config file not found at ${CONF_FILE}"
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    exit 1
fi

mkdir -p "${ROOT_DIR}"
cd "${REPO_ROOT}"

export PATH="/scratch/${USER}/conda-envs/alf-bafc/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR="${REPO_ROOT}/.torch_extensions"
# Required for deterministic CuBLAS ops when deterministic algorithms are enabled.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "Running BAFC DMC train:"
echo "  submit_dir: ${SUBMIT_DIR}"
echo "  repo: ${REPO_ROOT}"
echo "  conf: ${CONF_FILE}"
echo "  root: ${ROOT_DIR}"
echo "  python: ${PYTHON_BIN}"
echo "  num_parallel_environments: 16"

"${PYTHON_BIN}" -m alf.bin.train \
    --conf="${CONF_FILE}" \
    --root_dir="${ROOT_DIR}" \
    --conf_param="create_environment.num_parallel_environments=16" \
    --conf_param="TrainerConfig.debug_summaries=True" \
    --conf_param="TrainerConfig.summarize_grads_and_vars=True" \
    --conf_param="TrainerConfig.summarize_gradient_noise_scale=False" \
    --conf_param="TrainerConfig.summarize_action_distributions=True" \
    --alsologtostderr \
    --conf_param="TrainerConfig.num_checkpoints=5" \
    --distributed=none

echo "BAFC run completed. Output root_dir: ${ROOT_DIR}"
