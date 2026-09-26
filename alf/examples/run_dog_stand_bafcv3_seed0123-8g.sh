#!/bin/bash
# Launch four dog:stand BAFCv3 jobs (seeds 0-3), each using all eight GPUs, with frozen eval samples.
# Run with --dry-run to inspect commands. Use --initial-collect-steps for a
# short throughput pilot without changing the normal 10,000-step warmup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAFCV3_SCENARIO="four_utd11"
source "${SCRIPT_DIR}/run_dog_stand_bafcv3_8g_common.sh" "$@"
