#!/bin/bash
# Submit with:
#   sbatch hpc_sbatchs/smoke_bafcv3_dmc_cpu.sbatch

#SBATCH --job-name=alf-smoke-bafcv3-cpu
#SBATCH --account=torch_pr_741_general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${SUBMIT_DIR}"
if [[ ! -d "${REPO_ROOT}/alf" && -d "${SUBMIT_DIR}/.." && -d "${SUBMIT_DIR}/../alf" ]]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
fi

PYTHON_BIN="/scratch/${USER}/conda-envs/alf-bafc/bin/python"
CONF_FILE="${REPO_ROOT}/alf/examples/bafcv3_dmc_conf.py"
ROOT_DIR="/scratch/yw3210/alf_results/bafc_smoke"

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCH_EXTENSIONS_DIR="${REPO_ROOT}/.torch_extensions"
# Required for deterministic CuBLAS ops when deterministic algorithms are enabled.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "Running smoke train:"
echo "  submit_dir: ${SUBMIT_DIR}"
echo "  repo: ${REPO_ROOT}"
echo "  conf: ${CONF_FILE}"
echo "  root: ${ROOT_DIR}"
echo "  python: ${PYTHON_BIN}"

"${PYTHON_BIN}" -m alf.bin.train \
    --conf="${CONF_FILE}" \
    --root_dir="${ROOT_DIR}" \
    --conf_param="create_environment.env_name='hopper:hop'" \
    --conf_param="TrainerConfig.random_seed=0" \
    --conf_param="TrainerConfig.num_env_steps=2000" \
    --conf_param="num_actor_critic=2" \
    --conf_param="debug_mode=True" \
    --distributed=none

echo "Smoke run completed. Output root_dir: ${ROOT_DIR}"
