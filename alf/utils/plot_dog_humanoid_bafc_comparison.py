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

"""Plot the selected Dog, Dog Fetch, Dog Run, and Humanoid BAFC experiments.

The experiment selection intentionally combines runs copied from two hosts
with runs produced locally. Run from the repository root, for example::

    python alf/utils/plot_dog_humanoid_bafc_comparison.py

For each environment, the script writes an AverageReturn comparison, including
BAFCv6 where runs are available. It also writes BAFC_TR trust diagnostics for
the three environments with BAFC_TR runs, the two-seed Humanoid comparison,
and focused BAFCv3-vs-RLPD AverageReturn plots for Dog Fetch, Dog Run, Dog Walk,
Humanoid Walk, and Hopper Hop. Curves are aligned on their overlapping
environment-step range, linearly interpolated, and plotted as the unsmoothed
across-seed mean with a population +/-1 standard deviation band.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


RETURN_TAG = "Metrics_vs_EnvironmentSteps/AverageReturn"
EVAL_TRUST_OVER_MAX_TAG = "BafcAlgorithmV3TR2/eval_trust_over_max"
EVAL_TRUST_METRIC_TAG = "BafcAlgorithmV3TR2/eval_trust_metric"
INITIAL_EVAL_TRUST_THRESHOLD = 30.0

_PLOTTED_TAGS = (
    RETURN_TAG,
    EVAL_TRUST_OVER_MAX_TAG,
    EVAL_TRUST_METRIC_TAG,
)
_SCALAR_CURVE_CACHE: dict[tuple[str, str], ScalarCurve] = {}

ALGORITHM_COLORS = {
    "RLPD": "tab:blue",
    "Ours": "tab:orange",
    "BAFCv3": "tab:orange",
    "BAFC_TR": "tab:green",
    "BAFCv6": "tab:red",
}

FOCUSED_ALGORITHM_COLORS = {
    "RLPD": ALGORITHM_COLORS["Ours"],
    "Ours": ALGORITHM_COLORS["RLPD"],
}


@dataclass(frozen=True)
class ScalarCurve:
    steps: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class AggregateCurve:
    steps: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @property
    def mean_minus_std(self) -> np.ndarray:
        return self.mean - self.std

    @property
    def mean_plus_std(self) -> np.ndarray:
        return self.mean + self.std


def _display_list(items: Iterable[str], limit: int = 12) -> str:
    items = list(items)
    shown = items[:limit]
    suffix = "" if len(items) <= limit else "\n  ..."
    return "\n  " + "\n  ".join(shown) + suffix


def _read_scalar_curve(logdir: str, tag: str) -> ScalarCurve:
    if not os.path.isdir(logdir):
        raise ValueError("TensorBoard log directory does not exist: %s" % logdir)

    cache_key = (os.path.realpath(logdir), tag)
    if cache_key in _SCALAR_CURVE_CACHE:
        return _SCALAR_CURVE_CACHE[cache_key]

    event_acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    event_acc.Reload()
    scalar_tags = event_acc.Tags().get("scalars", [])
    if tag not in scalar_tags:
        raise ValueError(
            "Required scalar tag %r is missing from %s. Available scalar tags:%s"
            % (tag, logdir, _display_list(scalar_tags)))

    # EventAccumulator has already parsed the entire event file. Extract and
    # cache all metrics used by this script so BAFC_TR logs are not parsed
    # again for each of the three figures.
    for scalar_tag in set(_PLOTTED_TAGS + (tag,)).intersection(scalar_tags):
        events = event_acc.Scalars(scalar_tag)
        if not events:
            continue

        # Resumed runs can contain more than one value at a step. Match the
        # repository aggregation helper by retaining the last observed value.
        by_step = {}
        for event in events:
            by_step[int(event.step)] = float(event.value)
        steps = np.array(sorted(by_step), dtype=np.float64)
        values = np.array([by_step[int(step)] for step in steps],
                          dtype=np.float64)
        _SCALAR_CURVE_CACHE[(os.path.realpath(logdir), scalar_tag)] = (
            ScalarCurve(steps=steps, values=values))

    if cache_key not in _SCALAR_CURVE_CACHE:
        raise ValueError("Required scalar tag %r has no events in %s" %
                         (tag, logdir))
    return _SCALAR_CURVE_CACHE[cache_key]


def aggregate_curves(curves: list[ScalarCurve],
                     source_names: list[str], tag: str) -> AggregateCurve:
    """Align curves over their intersection and calculate mean/population std."""
    if not curves:
        raise ValueError("No curves were supplied for tag %r" % tag)
    if len(curves) != len(source_names):
        raise ValueError("Curve/source count mismatch for tag %r" % tag)

    overlap_start = max(curve.steps[0] for curve in curves)
    overlap_end = min(curve.steps[-1] for curve in curves)
    if overlap_start > overlap_end:
        ranges = ("%s: [%d, %d]" % (name, curve.steps[0], curve.steps[-1])
                  for name, curve in zip(source_names, curves))
        raise ValueError("No overlapping step range for tag %r:%s" %
                         (tag, _display_list(ranges)))

    reference = max(curves, key=lambda curve: curve.steps.size)
    reference_steps = reference.steps[
        (reference.steps >= overlap_start) & (reference.steps <= overlap_end)]
    common_steps = np.unique(
        np.concatenate((np.array([overlap_start, overlap_end]), reference_steps)))

    aligned = []
    for curve in curves:
        if curve.steps.size == 1:
            values = np.full_like(common_steps, curve.values[0])
        else:
            values = np.interp(common_steps, curve.steps, curve.values)
        aligned.append(values)
    values = np.stack(aligned)
    return AggregateCurve(steps=common_steps,
                          mean=np.mean(values, axis=0),
                          std=np.std(values, axis=0, ddof=0))


def aggregate_scalar(run_dirs: list[str], tag: str) -> AggregateCurve:
    logdirs = [os.path.join(run_dir, "train") for run_dir in run_dirs]
    missing = [path for path in logdirs if not os.path.isdir(path)]
    if missing:
        raise ValueError("Missing required TensorBoard directories:%s" %
                         _display_list(missing))
    curves = [_read_scalar_curve(logdir, tag) for logdir in logdirs]
    return aggregate_curves(curves, logdirs, tag)


def build_run_groups(workspace_root: str, local_results_root: str,
                     server_copy_root: str,
                     server2_copy_root: str) -> dict[str, dict[str, list[str]]]:
    """Return the fixed experiment mapping, with every list ordered by seed."""
    dog_local = os.path.join(local_results_root, "dog",
                             "rlpd_bafcv3_comparison_4g")
    dog_tr = os.path.join(
        dog_local, "bafcv3_tr2_no_cadence_trust_max_decay",
        "fixed_pairingFalse_num_sampled_critic8", "critic_utd11")

    hum_common = os.path.join(
        local_results_root, "humanoid",
        "bafcv3_eval_samples_rlpd_comparison_4g",
        "num_actor_critic10_num_sampled_critics_for_actor8_"
        "num_sampled_critic_targets1", "critic_utd11")
    hum_bafcv3 = os.path.join(
        local_results_root, "humanoid",
        "bafcv3_eval_samples_comparison_v2_4g",
        "num_actor_critic10_num_sampled_critics_for_actor8_"
        "num_sampled_critic_targets1", "critic_utd11",
        "bafcv3_trainable_eval_samples_random_target")

    groups = {
        "dog": {
            "RLPD": [
                os.path.join(server2_copy_root, "dog_rlpd_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root, "dog_rlpd_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFCv3": [
                os.path.join(server2_copy_root, "dog_bafcv3_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_bafc_trainable_rtT_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFC_TR": [os.path.join(dog_tr, "seed_%d" % seed)
                        for seed in range(4)],
        },
        "dog_fetch": {
            "RLPD": [
                os.path.join(server2_copy_root,
                             "dog_fetch_rlpd_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_fetch_rlpd_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFCv3": [
                os.path.join(server2_copy_root,
                             "dog_fetch_bafcv3_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_fetch_bafcv3_rtT_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFC_TR": [
                os.path.join(server2_copy_root,
                             "dog_fetch_bafcv3_tr2_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_fetch_bafcv3_tr2_rtT_s%d" % seed)
                for seed in (2, 3)
            ],
        },
        "dog_run": {
            "RLPD": [
                os.path.join(server2_copy_root,
                             "dog_run_rlpd_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_run_rlpd_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFCv3": [
                os.path.join(server2_copy_root,
                             "dog_run_bafcv3_rtT_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_run_bafcv3_rtT_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFCv6": [
                os.path.join(server2_copy_root,
                             "dog_run_bafcv6_reweight_rtT_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_run_bafcv6_reweight_rtT_s%d" % seed)
                for seed in (2, 3)
            ],
        },
        "humanoid": {
            "RLPD": [
                os.path.join(hum_common, "rlpd_default", "seed_%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server2_copy_root, "hum_rlpd_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFCv3": [os.path.join(hum_bafcv3, "seed_%d" % seed)
                       for seed in (0, 1)] + [
                           os.path.join(server2_copy_root,
                                        "hum_bafcv3_s%d" % seed)
                           for seed in (2, 3)
                       ],
            "BAFC_TR": [
                os.path.join(server_copy_root,
                             "hum_tr2_nocad_eval30_nodecay_rtT_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server2_copy_root,
                             "hum_bafcv3_tr2_s%d" % seed)
                for seed in (2, 3)
            ],
        },
        "humanoid_seed01": {
            "RLPD": [
                os.path.join(hum_common, "rlpd_default", "seed_%d" % seed)
                for seed in (0, 1)
            ],
            "BAFCv3": [os.path.join(hum_bafcv3, "seed_%d" % seed)
                       for seed in (0, 1)],
            "BAFC_TR": [
                os.path.join(server_copy_root,
                             "hum_tr2_nocad_eval30_nodecay_rtT_s%d" % seed)
                for seed in (0, 1)
            ],
            "BAFCv6": [
                os.path.join(server_copy_root,
                             "hum_bafcv6_trainable_rtT_s%d" % seed)
                for seed in (0, 1)
            ],
        },
    }
    # Keep this argument in the public builder so callers can provide one root
    # and derive all defaults in the same way as the CLI.
    del workspace_root
    return groups


def build_rlpd_ours_run_groups(
        workspace_root: str, local_results_root: str, server_copy_root: str,
        server2_copy_root: str) -> dict[str, dict[str, list[str]]]:
    """Return RLPD/Ours groups for the five focused comparison plots.

    Dog Fetch, Dog Run, Dog Walk (``dog``), and Humanoid Walk
    (``humanoid``) use seeds 0--3. Hopper Hop uses seeds 0--2, with BAFCv3
    critic UTD 11 for seeds 0--1 and critic UTD 3 for seed 2.
    """
    existing = build_run_groups(workspace_root, local_results_root,
                                server_copy_root, server2_copy_root)

    hopper_root = os.path.join(local_results_root, "hopper_hop",
                               "rlpd_bafcv3_comparison_4g")
    hopper_rlpd = os.path.join(hopper_root, "rlpd", "critic_utd10")
    hopper_bafcv3 = os.path.join(
        hopper_root, "bafcv3", "fixed_pairingFalse_num_sampled_critic8")
    hopper_ours = [
        os.path.join(hopper_bafcv3, "critic_utd11", "seed_%d" % seed)
        for seed in (0, 1)
    ] + [os.path.join(hopper_bafcv3, "critic_utd3", "seed_2")]

    return {
        "dog_fetch": {
            "Ours": existing["dog_fetch"]["BAFCv3"],
            "RLPD": existing["dog_fetch"]["RLPD"],
        },
        "dog_run": {
            "Ours": existing["dog_run"]["BAFCv3"],
            "RLPD": existing["dog_run"]["RLPD"],
        },
        "dog": {
            "Ours": existing["dog"]["BAFCv3"],
            "RLPD": existing["dog"]["RLPD"],
        },
        "humanoid": {
            "Ours": existing["humanoid"]["BAFCv3"],
            "RLPD": existing["humanoid"]["RLPD"],
        },
        "hopper_hop": {
            "Ours": hopper_ours,
            "RLPD": [
                os.path.join(hopper_rlpd, "seed_%d" % seed)
                for seed in (0, 1, 2)
            ],
        },
    }


def _plot_aggregate(ax: plt.Axes, aggregate: AggregateCurve, label: str,
                    color: str | None = None) -> None:
    color = color or ALGORITHM_COLORS[label]
    ax.plot(aggregate.steps, aggregate.mean, color=color, linewidth=2,
            label=label)
    ax.fill_between(aggregate.steps, aggregate.mean_minus_std,
                    aggregate.mean_plus_std, color=color, alpha=0.18,
                    linewidth=0)


def _print_summary(env: str, label: str, tag: str, seed_count: int,
                   aggregate: AggregateCurve) -> None:
    print("%s | %s | seeds=%d | %s | range=%d -> %d | "
          "final mean/std=%.6g / %.6g" %
          (env, label, seed_count, tag, int(aggregate.steps[0]),
           int(aggregate.steps[-1]), aggregate.mean[-1], aggregate.std[-1]))


def _format_environment_steps(value: float, _position: int | None) -> str:
    if abs(value) < 1000:
        return "%g" % value
    return "%gk" % (value / 1000)


def _finish_plot(fig: plt.Figure, ax: plt.Axes, title: str, ylabel: str,
                 output_path: str, xlabel: str = "Environment steps",
                 human_readable_x_ticks: bool = False) -> str:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    if human_readable_x_ticks:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_environment_steps))
    else:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote: %s" % output_path)
    return output_path


def plot_average_return(env: str, groups: dict[str, list[str]],
                        output_root: str, title: str | None = None,
                        filename: str | None = None,
                        xlabel: str = "Environment steps",
                        ylabel: str = "Average Return",
                        colors: dict[str, str] | None = None,
                        human_readable_x_ticks: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for label, run_dirs in groups.items():
        aggregate = aggregate_scalar(run_dirs, RETURN_TAG)
        color = colors[label] if colors is not None else None
        _plot_aggregate(ax, aggregate, label, color=color)
        _print_summary(env, label, RETURN_TAG, len(run_dirs), aggregate)
    if title is None:
        title = "%s Average Return vs Environment Steps" % env.capitalize()
    filename = filename or "%s_average_return_vs_env_steps.png" % env
    return _finish_plot(fig, ax, title, ylabel,
                        os.path.join(output_root, filename), xlabel=xlabel,
                        human_readable_x_ticks=human_readable_x_ticks)


def plot_eval_trust_over_max(env: str, bafc_tr_dirs: list[str],
                             output_root: str) -> str:
    aggregate = aggregate_scalar(bafc_tr_dirs, EVAL_TRUST_OVER_MAX_TAG)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    _plot_aggregate(ax, aggregate, "BAFC_TR")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
               label="Threshold")
    _print_summary(env, "BAFC_TR", EVAL_TRUST_OVER_MAX_TAG,
                   len(bafc_tr_dirs), aggregate)
    return _finish_plot(
        fig, ax,
        "%s BAFC_TR Evaluation Trust / Maximum vs Environment Steps" %
        env.capitalize(), "Evaluation trust / maximum",
        os.path.join(output_root,
                     "%s_eval_trust_over_max_vs_env_steps.png" % env))


def plot_raw_eval_trust_metric(env: str, bafc_tr_dirs: list[str],
                               initial_threshold: float,
                               output_root: str) -> str:
    """Plot raw evaluation trust against its non-decayed initial threshold."""
    aggregate = aggregate_scalar(bafc_tr_dirs, EVAL_TRUST_METRIC_TAG)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    _plot_aggregate(ax, aggregate, "BAFC_TR")
    ax.axhline(initial_threshold,
               color="black",
               linestyle="--",
               linewidth=1.5,
               label="Initial threshold (no decay)")
    _print_summary(env, "BAFC_TR", EVAL_TRUST_METRIC_TAG,
                   len(bafc_tr_dirs), aggregate)
    return _finish_plot(
        fig, ax,
        "%s BAFC_TR Raw Evaluation Trust vs Environment Steps" %
        env.capitalize(), "Raw evaluation trust metric",
        os.path.join(output_root,
                     "%s_eval_trust_metric_vs_env_steps.png" % env))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default="/workspace")
    parser.add_argument("--local-results-root", default=None,
                        help="Defaults to <workspace-root>/alf_results.")
    parser.add_argument("--server-copy-root", default=None,
                        help="Defaults to <workspace-root>/server_copy.")
    parser.add_argument("--server2-copy-root", default=None,
                        help="Defaults to <workspace-root>/server2_copy.")
    parser.add_argument("--output-root", default=None,
                        help=("Defaults to <local-results-root>/"
                              "plots_dog_humanoid_bafc_comparison."))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    local_results_root = args.local_results_root or os.path.join(
        args.workspace_root, "alf_results")
    server_copy_root = args.server_copy_root or os.path.join(
        args.workspace_root, "server_copy")
    server2_copy_root = args.server2_copy_root or os.path.join(
        args.workspace_root, "server2_copy")
    output_root = args.output_root or os.path.join(
        local_results_root, "plots_dog_humanoid_bafc_comparison")
    groups = build_run_groups(args.workspace_root, local_results_root,
                              server_copy_root, server2_copy_root)
    rlpd_ours_groups = build_rlpd_ours_run_groups(
        args.workspace_root, local_results_root, server_copy_root,
        server2_copy_root)

    for env in ("dog", "dog_fetch", "dog_run", "humanoid"):
        plot_average_return(env, groups[env], output_root)
    for env in ("dog", "dog_fetch", "humanoid"):
        plot_eval_trust_over_max(env, groups[env]["BAFC_TR"], output_root)
        plot_raw_eval_trust_metric(
            env, groups[env]["BAFC_TR"], INITIAL_EVAL_TRUST_THRESHOLD,
            output_root)
    plot_average_return(
        "humanoid_seed01", groups["humanoid_seed01"], output_root,
        title="Humanoid Average Return vs Environment Steps (Seeds 0-1)",
        filename="humanoid_seed01_average_return_vs_env_steps.png")
    for env in ("dog_fetch", "dog_run", "dog", "humanoid", "hopper_hop"):
        plot_average_return(
            env, rlpd_ours_groups[env], output_root,
            title="",
            xlabel="Environment Steps",
            ylabel="Average Episodic Return",
            colors=FOCUSED_ALGORITHM_COLORS,
            human_readable_x_ticks=True,
            filename="%s_rlpd_vs_ours_average_return_vs_env_steps.png" % env)


if __name__ == "__main__":
    main()
