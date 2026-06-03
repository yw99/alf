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

"""Plot hardcoded cheetah BAFCv3 and RLPD TensorBoard curves.

Run from the repo root:

    python alf/utils/plot_cheetah_bafcv3_rlpd_curves.py

Each output figure overlays the mean curve for each run group and shades +/-1
std for the group. Curves are aligned on each group's overlapping step range,
reusing the TensorBoard loading and interpolation code from
``aggregate_tb_mean_std.py``.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from alf.utils import aggregate_tb_mean_std as tb_agg
except ModuleNotFoundError:
    import aggregate_tb_mean_std as tb_agg


BAFC_ROOT = (
    "/root/alf_results_v7_eval/cheetah/"
    "bafcv3_tr_dmc_evalonly_single_run"
)
RLPD_ROOT = (
    "/root/alf_results_v7_benchmark_algo/cheetah/"
    "rlpd_dmc/critic_utd3"
)
OUTPUT_ROOT = "/root/alf_results_v7_eval/plots_cheetah_bafcv3_rlpd_curves"

SERIES = [
    {
        "label": "eval40",
        "run_dirs": [
            os.path.join(BAFC_ROOT, "eval40.0_nf4_mi8_cap4_seed3"),
            os.path.join(BAFC_ROOT, "eval40.0_nf4_mi8_cap4_seed4"),
        ],
    },
    {
        "label": "eval50",
        "run_dirs": [
            os.path.join(BAFC_ROOT, "eval50.0_nf4_mi8_cap4_seed2"),
            os.path.join(BAFC_ROOT, "eval50.0_nf4_mi8_cap4_seed4"),
        ],
    },
    {
        "label": "eval60",
        "run_dirs": [
            os.path.join(BAFC_ROOT, "eval60.0_nf4_mi8_cap4_seed3"),
            os.path.join(BAFC_ROOT, "eval60.0_nf4_mi8_cap4_seed4"),
        ],
    },
    {
        "label": "RLPD",
        "run_dirs": [
            os.path.join(RLPD_ROOT, "seed3"),
            os.path.join(RLPD_ROOT, "seed4"),
        ],
    },
]

CURVES = [
    {
        "tag": "Metrics/AverageReturn",
        "subdir": "train",
        "title": "Average Return",
        "ylabel": "Average Return",
        "output": "metrics_average_return.png",
    },
    {
        "tag": "Metrics_vs_EnvironmentSteps/AverageReturn",
        "subdir": "train",
        "title": "Average Return vs Environment Steps",
        "ylabel": "Average Return",
        "output": "metrics_vs_environment_steps_average_return.png",
    },
    {
        "tag": "Metrics/AverageEpisodeLength",
        "subdir": "train",
        "title": "Average Episode Length",
        "ylabel": "Average Episode Length",
        "output": "metrics_average_episode_length.png",
    },
    {
        "tag": "Metrics_vs_EnvironmentSteps/AverageEpisodeLength",
        "subdir": "train",
        "title": "Average Episode Length vs Environment Steps",
        "ylabel": "Average Episode Length",
        "output": "metrics_vs_environment_steps_average_episode_length.png",
    },
    {
        "tag": "rollout_skip_eval/average_return",
        "subdir": "eval",
        "title": "Rollout Skip Eval Average Return",
        "ylabel": "Average Return",
        "output": "rollout_skip_eval_average_return.png",
    },
    {
        "tag": "rollout_skip_eval/relative_average_return_change",
        "subdir": "eval",
        "title": "Relative Average Return Change",
        "ylabel": "Relative Average Return Change",
        "output": "rollout_skip_eval_relative_average_return_change.png",
    },
]


def _source_logdirs(run_dirs: list[str], subdir: str) -> list[str]:
    return [os.path.join(run_dir, subdir) for run_dir in run_dirs]


def _check_run_dirs() -> None:
    missing = [
        run_dir for series in SERIES for run_dir in series["run_dirs"]
        if not os.path.isdir(run_dir)
    ]
    if missing:
        raise ValueError("Missing run directories:\n  " + "\n  ".join(missing))


def _plot_curve(curve: dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    tag = curve["tag"]
    subdir = curve["subdir"]
    plotted = 0

    for series in SERIES:
        logdirs = _source_logdirs(series["run_dirs"], subdir)
        label = series["label"]
        try:
            aggregate = tb_agg.aggregate_scalar(logdirs, tag)
        except ValueError as exc:
            print("skip %s for %s: %s" % (tag, label, exc))
            continue

        ax.plot(aggregate.steps, aggregate.mean, linewidth=2, label=label)
        color = ax.lines[-1].get_color()
        ax.fill_between(aggregate.steps,
                        aggregate.mean_minus_std,
                        aggregate.mean_plus_std,
                        color=color,
                        alpha=0.18,
                        linewidth=0)
        plotted += 1
        print("%s | %s | points=%d | range=%d -> %d | final mean/std=%.6g / %.6g"
              % (label, tag, aggregate.steps.size, int(aggregate.steps[0]),
                 int(aggregate.steps[-1]), aggregate.mean[-1],
                 aggregate.std[-1]))

    if plotted == 0:
        raise ValueError("No readable series for tag %r" % tag)

    ax.set_title(curve["title"])
    ax.set_xlabel("TensorBoard step")
    ax.set_ylabel(curve["ylabel"])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    output_path = os.path.join(OUTPUT_ROOT, curve["output"])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote: %s" % output_path)
    return output_path


def main() -> None:
    _check_run_dirs()
    for curve in CURVES:
        _plot_curve(curve)


if __name__ == "__main__":
    main()
