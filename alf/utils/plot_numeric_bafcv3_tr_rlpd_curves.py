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

"""Plot numeric-result BAFCv3-TR and RLPD TensorBoard curves.

Run from the repo root:

    python alf/utils/plot_numeric_bafcv3_tr_rlpd_curves.py --env cheetah

The script writes two PNG figures:

* BAFCv3-TR vs RLPD AverageReturn over environment steps.
* BAFCv3-TR rollout-skip signed absolute AverageReturn change.

Each figure plots the across-seed mean and shades +/-1 std. By default, the
2-GPU run set uses seeds 1,2,3 and the 4-GPU run set uses seeds 0,1,2,3. Curves
are aligned on the overlapping step range and interpolated using the same logic
as ``aggregate_tb_mean_std.py``.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

try:
    from alf.utils import aggregate_tb_mean_std as tb_agg
except ModuleNotFoundError:
    import aggregate_tb_mean_std as tb_agg


DEFAULT_BASE_DIR = "/root/numeric_results"
DEFAULT_ENV = "hopper" #"cheetah"
DEFAULT_RUN_SET = "2g"
DEFAULT_SEEDS_BY_RUN_SET = {
    "2g": "1,2,3",
    "4g": "0,1,2,3",
}

RETURN_TAG = "Metrics_vs_EnvironmentSteps/AverageReturn"
START_RETURN_TAG = "rollout_skip_eval/start_average_return"
END_RETURN_TAG = "rollout_skip_eval/end_average_return"
ABS_CHANGE_TAG = "rollout_skip_eval/absolute_average_return_change"


def _display_list(items: Iterable[str], limit: int = 12) -> str:
    items = list(items)
    shown = items[:limit]
    suffix = "" if len(items) <= limit else "\n  ..."
    return "\n  " + "\n  ".join(shown) + suffix


def _parse_seeds(raw: str) -> list[int]:
    seeds = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            seed = int(piece)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Seeds must be comma-separated integers: %r" % raw) from exc
        seeds.append(seed)

    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _env_dir(env: str) -> str:
    return env.split(":", 1)[0]


def _default_seeds(run_set: str) -> list[int]:
    return _parse_seeds(DEFAULT_SEEDS_BY_RUN_SET[run_set])


def _default_output_root(base_dir: str, env: str, run_set: str) -> str:
    output_name = "plots_bafcv3_tr_rlpd_curves"
    if run_set == "4g":
        output_name += "_4g"
    return os.path.join(base_dir, _env_dir(env), output_name)


def _build_run_dirs(base_dir: str, env: str, run_set: str,
                    seeds: list[int]) -> tuple[list[str], list[str]]:
    env_dir = _env_dir(env)
    if run_set == "4g":
        bafc_run_name = "bafcv3_tr_dmc_4g"
        rlpd_run_name = "rlpd_dmc_4g"
    else:
        bafc_run_name = "bafcv3_tr_dmc"
        rlpd_run_name = "rlpd_dmc"

    bafc_root = os.path.join(base_dir, env_dir, bafc_run_name)
    rlpd_root = os.path.join(base_dir, env_dir, rlpd_run_name, "critic_utd3")
    bafc_dirs = [os.path.join(bafc_root, "seed_%d" % seed) for seed in seeds]
    rlpd_dirs = [os.path.join(rlpd_root, "seed_%d" % seed) for seed in seeds]
    return bafc_dirs, rlpd_dirs


def _source_logdirs(run_dirs: list[str], subdir: str) -> list[str]:
    return [os.path.join(run_dir, subdir) for run_dir in run_dirs]


def _check_dirs(paths: Iterable[str], description: str) -> None:
    missing = [path for path in paths if not os.path.isdir(path)]
    if missing:
        raise ValueError("Missing %s:%s" %
                         (description, _display_list(missing)))


def _read_scalar_curve(logdir: str, tag: str) -> tb_agg.ScalarCurve:
    if not os.path.isdir(logdir):
        raise ValueError("Scalar log directory does not exist: %s" % logdir)

    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    scalar_tags = event_acc.Tags().get("scalars", [])
    if tag not in scalar_tags:
        raise ValueError("Tag %r is missing from %s. Available scalar tags:%s" %
                         (tag, logdir, _display_list(scalar_tags)))

    events = event_acc.Scalars(tag)
    if not events:
        raise ValueError("Tag %r has no scalar events in %s" % (tag, logdir))

    # Keep the last observed value for duplicate steps, matching the aggregate
    # helper's behavior for resumed runs.
    by_step = {}
    for event in events:
        by_step[int(event.step)] = float(event.value)

    steps = np.array(sorted(by_step), dtype=np.float64)
    values = np.array([by_step[int(step)] for step in steps], dtype=np.float64)
    return tb_agg.ScalarCurve(steps=steps, values=values)


def _read_absolute_change_curve(logdir: str) -> tb_agg.ScalarCurve:
    start_curve = _read_scalar_curve(logdir, START_RETURN_TAG)
    end_curve = _read_scalar_curve(logdir, END_RETURN_TAG)

    count = min(start_curve.steps.size, end_curve.steps.size)
    if count == 0:
        raise ValueError("No paired rollout-skip return events in %s" % logdir)

    if start_curve.steps.size != end_curve.steps.size:
        print("warning: truncating unmatched rollout-skip events in %s "
              "(start=%d end=%d paired=%d)" %
              (logdir, start_curve.steps.size, end_curve.steps.size, count))

    steps = end_curve.steps[:count]
    values = end_curve.values[:count] - start_curve.values[:count]
    return tb_agg.ScalarCurve(steps=steps, values=values)


def _aggregate_curves(curves: list[tb_agg.ScalarCurve], source_names: list[str],
                      tag: str) -> tb_agg.AggregateCurve:
    if not curves:
        raise ValueError("No curves were provided for tag %r" % tag)
    if len(curves) == 1:
        print("  warning: only one run for %s; std will be zero" % tag)

    overlap_start = max(curve.steps[0] for curve in curves)
    overlap_end = min(curve.steps[-1] for curve in curves)
    if overlap_start > overlap_end:
        raise ValueError(
            "No overlapping step range for tag %r. Per-run ranges:%s" %
            (tag, _display_list("%s: [%d, %d]" %
                                (name, curve.steps[0], curve.steps[-1])
                                for name, curve in zip(source_names, curves))))

    reference_curve = max(curves, key=lambda c: len(c.steps))
    in_overlap = reference_curve.steps[
        (reference_curve.steps >= overlap_start) &
        (reference_curve.steps <= overlap_end)]
    common_steps = np.unique(
        np.concatenate(
            [np.array([overlap_start, overlap_end], dtype=np.float64),
             in_overlap]))
    if common_steps.size == 0:
        raise ValueError("No common steps could be built for tag %r" % tag)

    aligned_values = []
    for curve in curves:
        if curve.steps.size == 1:
            values = np.full_like(common_steps, curve.values[0])
        else:
            values = np.interp(common_steps, curve.steps, curve.values)
        aligned_values.append(values)

    values = np.stack(aligned_values, axis=0)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    return tb_agg.AggregateCurve(steps=common_steps,
                                 mean=mean,
                                 std=std,
                                 mean_minus_std=mean - std,
                                 mean_plus_std=mean + std)


def _plot_aggregate(ax: plt.Axes, aggregate: tb_agg.AggregateCurve,
                    label: str) -> None:
    ax.plot(aggregate.steps, aggregate.mean, linewidth=2, label=label)
    color = ax.lines[-1].get_color()
    ax.fill_between(aggregate.steps,
                    aggregate.mean_minus_std,
                    aggregate.mean_plus_std,
                    color=color,
                    alpha=0.18,
                    linewidth=0)


def _finish_plot(fig: plt.Figure, ax: plt.Axes, title: str, xlabel: str,
                 ylabel: str, output_path: str) -> str:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote: %s" % output_path)
    return output_path


def _print_aggregate_summary(label: str, tag: str,
                             aggregate: tb_agg.AggregateCurve) -> None:
    print("%s | %s | points=%d | range=%d -> %d | final mean/std=%.6g / %.6g"
          % (label, tag, aggregate.steps.size, int(aggregate.steps[0]),
             int(aggregate.steps[-1]), aggregate.mean[-1],
             aggregate.std[-1]))


def _plot_average_return(env: str, bafc_dirs: list[str], rlpd_dirs: list[str],
                         output_root: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for label, run_dirs in (("BAFCv3-TR", bafc_dirs), ("RLPD", rlpd_dirs)):
        logdirs = _source_logdirs(run_dirs, "train")
        aggregate = tb_agg.aggregate_scalar(logdirs, RETURN_TAG)
        _plot_aggregate(ax, aggregate, label)
        _print_aggregate_summary(label, RETURN_TAG, aggregate)

    return _finish_plot(
        fig=fig,
        ax=ax,
        title="%s Average Return vs Environment Steps" % _env_dir(env),
        xlabel="Environment steps",
        ylabel="Average Return",
        output_path=os.path.join(output_root,
                                 "average_return_vs_environment_steps.png"),
    )


def _plot_absolute_return_change(env: str, bafc_dirs: list[str],
                                 output_root: str) -> str:
    eval_logdirs = _source_logdirs(bafc_dirs, "eval")
    curves = [_read_absolute_change_curve(logdir) for logdir in eval_logdirs]
    aggregate = _aggregate_curves(curves, eval_logdirs, ABS_CHANGE_TAG)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    _plot_aggregate(ax, aggregate, "BAFCv3-TR")
    _print_aggregate_summary("BAFCv3-TR", ABS_CHANGE_TAG, aggregate)

    return _finish_plot(
        fig=fig,
        ax=ax,
        title="%s Rollout-Skip Absolute Average Return Change" % _env_dir(env),
        xlabel="Rollout-skip end step",
        ylabel="End - Start Average Return",
        output_path=os.path.join(
            output_root, "rollout_skip_eval_absolute_average_return_change.png"
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot numeric-result BAFCv3-TR and RLPD curves.")
    parser.add_argument("--env", default=DEFAULT_ENV,
                        help="Environment directory/name (default: %(default)s).")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                        help="Base result directory (default: %(default)s).")
    parser.add_argument("--run-set", default=DEFAULT_RUN_SET,
                        choices=sorted(DEFAULT_SEEDS_BY_RUN_SET),
                        help=("Run directory preset. Use 2g for "
                              "bafcv3_tr_dmc/rlpd_dmc or 4g for "
                              "bafcv3_tr_dmc_4g/rlpd_dmc_4g "
                              "(default: %(default)s)."))
    parser.add_argument("--seeds", type=_parse_seeds,
                        default=None,
                        help=("Comma-separated seed IDs. Defaults to 1,2,3 "
                              "for --run-set 2g and 0,1,2,3 for --run-set "
                              "4g."))
    parser.add_argument("--output-root", default=None,
                        help=("Output directory. Defaults to "
                              "<base-dir>/<env>/plots_bafcv3_tr_rlpd_curves "
                              "for 2g and the same path with _4g appended "
                              "for 4g."))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = args.seeds if args.seeds is not None else _default_seeds(
        args.run_set)
    output_root = args.output_root or _default_output_root(
        args.base_dir, args.env, args.run_set)
    bafc_dirs, rlpd_dirs = _build_run_dirs(args.base_dir, args.env,
                                           args.run_set, seeds)

    _check_dirs(bafc_dirs, "BAFCv3-TR run directories")
    _check_dirs(rlpd_dirs, "RLPD run directories")
    _check_dirs(_source_logdirs(bafc_dirs, "train"),
                "BAFCv3-TR train log directories")
    _check_dirs(_source_logdirs(rlpd_dirs, "train"),
                "RLPD train log directories")
    _check_dirs(_source_logdirs(bafc_dirs, "eval"),
                "BAFCv3-TR eval log directories")

    print("env: %s" % args.env)
    print("run_set: %s" % args.run_set)
    print("seeds: %s" % ",".join(str(seed) for seed in seeds))
    print("output: %s" % output_root)

    _plot_average_return(args.env, bafc_dirs, rlpd_dirs, output_root)
    _plot_absolute_return_change(args.env, bafc_dirs, output_root)


if __name__ == "__main__":
    main()
