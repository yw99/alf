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
import re
from typing import Iterable

try:
    from alf.utils import aggregate_tb_mean_std as tb_agg
except ModuleNotFoundError:
    import aggregate_tb_mean_std as tb_agg


# ---------------------------------------------------------------------------
# Hardcoded RLPD benchmark config. Edit this block for new aggregation jobs.
# ---------------------------------------------------------------------------

OUTPUT_ROOT = "/root/alf_results_v7_benchmark_algo/tb_aggregated"
OVERWRITE_OUTPUT = True

MANUAL_RUN_SPEC_ENV_NAME = "cheetah"
# Optional comma-separated `label:path` specs for explicit manual aggregation.
# When non-empty, labels matching `<prefix>_sN` are grouped by prefix and
# replace the default RLPD_VARIANTS targets below.
MANUAL_RUN_SPECS = (
    "rlpdc3_s3:/root/alf_results_v7_benchmark_algo/cheetah/"
    "rlpd_dmc/critic_utd3/seed3,"
    "rlpdc3_s4:/root/alf_results_v7_benchmark_algo/cheetah/"
    "rlpd_dmc/critic_utd3/seed4,"
    "rlpdc3_s5:/root/alf_results_v7_benchmark_algo/cheetah/"
    "rlpd_dmc/critic_utd3/seed5"
)

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


def _parse_manual_run_specs(run_specs: str | Iterable[str]) -> list[tuple[str, str]]:
    if isinstance(run_specs, str):
        raw_specs = run_specs.split(",")
    else:
        raw_specs = run_specs

    parsed = []
    for raw_spec in raw_specs:
        spec = raw_spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError("Manual run spec must be `label:path`: %r" % spec)
        label, path = spec.split(":", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError("Manual run spec must be `label:path`: %r" % spec)
        parsed.append((label, path))
    return parsed


def _build_manual_run_spec_targets(
        specs: Iterable[tuple[str, str]]) -> list[dict]:
    label_re = re.compile(r"^(?P<prefix>.+)_s(?P<seed>\d+)$")
    grouped: dict[str, list[tuple[int, str]]] = {}

    for label, path in specs:
        match = label_re.match(label)
        if not match:
            continue
        prefix = match.group("prefix")
        seed = int(match.group("seed"))
        grouped.setdefault(prefix, []).append((seed, path))

    targets = []
    for prefix, seed_paths in grouped.items():
        seed_paths = sorted(seed_paths, key=lambda item: item[0])
        seed_label = "_".join("s%d" % seed for seed, _ in seed_paths)
        targets.append({
            "name": "%s_%s_%s" %
                    (MANUAL_RUN_SPEC_ENV_NAME, prefix, seed_label),
            "run_dirs": [path for _, path in seed_paths],
            "curves": CURVES,
        })
    return targets


def _build_targets() -> list[dict]:
    manual_specs = _parse_manual_run_specs(MANUAL_RUN_SPECS)
    if manual_specs:
        manual_targets = _build_manual_run_spec_targets(manual_specs)
        if not manual_targets:
            raise ValueError(
                "MANUAL_RUN_SPECS did not contain any <prefix>_sN entries")
        return manual_targets

    return [_build_target(variant) for variant in RLPD_VARIANTS]


TARGETS = _build_targets()


def main() -> None:
    tb_agg.OVERWRITE_OUTPUT = OVERWRITE_OUTPUT
    for target in TARGETS:
        tb_agg.aggregate_target(target, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
