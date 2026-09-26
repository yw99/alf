#!/bin/bash
# Launch eight dog:stand BAFCv3 jobs on all eight GPUs: seeds 0-3 at
# critic_utd=11 (12 updates/iteration) and seeds 0-3 at critic_utd=3
# (4 updates/iteration). Random eval samples are frozen after initialization.
# Each job has its own DDP and HTTP ports.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAFCV3_SCENARIO="mixed_utd11_utd3"
source "${SCRIPT_DIR}/run_dog_stand_bafcv3_8g_common.sh" "$@"
