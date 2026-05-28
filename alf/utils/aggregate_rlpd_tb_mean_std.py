# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregate TensorBoard scalar curves across RLPD seed runs.

Edit the hardcoded config below, then run from the repo root:

    python alf/utils/aggregate_rlpd_tb_mean_std.py

The script reuses ``aggregate_tb_mean_std.py`` and writes TensorBoard event
files with mean, mean-std, mean+std, and std curves.
"""

from __future__ import annotations

import os

try:
    from alf.utils import aggregate_tb_mean_std as tb_agg
except ModuleNotFoundError:
    import aggregate_tb_mean_std as tb_agg


# ---------------------------------------------------------------------------
# Hardcoded RLPD benchmark config. Edit this block for new aggregation jobs.
# ---------------------------------------------------------------------------

OUTPUT_ROOT = "/root/alf_results_v7_benchmark_algo/tb_aggregated"
OVERWRITE_OUTPUT = True

RUN_ROOT = "/root/alf_results_v7_benchmark_algo/hopper/rlpd_dmc"
ENV_NAME = "hopper"

# Relative variant roots under RUN_ROOT. The empty string is the baseline layout:
#   RUN_ROOT/seed3, RUN_ROOT/seed4, ...
RLPD_VARIANTS = [
    {
        "name": "rlpd_dmc",
        "relative_dir": "",
    },
    {
        "name": "rlpd_dmc_critic_utd3",
        "relative_dir": "critic_utd3",
    },
]

CURVES = [
    {
        "tag": "Metrics/AverageReturn",
        "subdir": "train",
    },
    {
        "tag": "Metrics_vs_EnvironmentSteps/AverageReturn",
        "subdir": "train",
    },
    {
        "tag": "Metrics/AverageEpisodeLength",
        "subdir": "train",
    },
    {
        "tag": "Metrics_vs_EnvironmentSteps/AverageEpisodeLength",
        "subdir": "train",
    },
]


def _build_target(variant: dict) -> dict:
    variant_root = os.path.join(RUN_ROOT, variant["relative_dir"])
    return {
        "name": "%s_%s" % (ENV_NAME, variant["name"]),
        "run_glob": os.path.join(variant_root, "seed*"),
        "curves": CURVES,
    }


TARGETS = [_build_target(variant) for variant in RLPD_VARIANTS]


def main() -> None:
    tb_agg.OVERWRITE_OUTPUT = OVERWRITE_OUTPUT
    for target in TARGETS:
        tb_agg.aggregate_target(target, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
