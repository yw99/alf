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

"""Plot numeric-result BAFCv3-TR, BAFCv3, RLPD, BAFCv6, and BAFCv3-TR2 curves.

Run from the repo root:

    python alf/utils/plot_numeric_bafcv3_tr_rlpd_bafcv6_curves.py --env cheetah

The script writes three PNG figures:

* BAFCv3-TR vs BAFCv3 cu3 vs RLPD vs BAFCv6 cu2 AverageReturn over
  environment steps, plus BAFCv3-TR2 and BAFCv6 resume when available.
* BAFCv3-TR rollout-skip signed absolute AverageReturn change, plus
  BAFCv3-TR2 when available.
* BAFCv3-TR rollout-skip relative AverageReturn change, plus BAFCv3-TR2 when
  available.

Each figure plots the across-seed mean and shades +/-1 std. This script is
4-GPU focused and defaults to seeds 1,2,3,4. Additional BAFCv3 and BAFCv6
numeric runs are included only in the AverageReturn figure. The BAFCv3-TR2
AverageReturn curve is optional and is plotted when all selected seed train
logs are available. BAFCv3-TR2 rollout-skip curves are optional and are plotted
when all selected seed eval logs and rollout-skip scalar tags are available.
BAFCv6 resume curves are optional and are included only in the AverageReturn
figure. Curves are aligned on the overlapping step range and interpolated using
the same logic as ``aggregate_tb_mean_std.py``.
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
DEFAULT_ENV = "cheetah"
DEFAULT_SEEDS = "1,2,3,4"
DEFAULT_BAFCV3_CRITIC_UTDS = "3"
DEFAULT_BAFCV3_TR2_CRITIC_UTD = 3
DEFAULT_RLPD_CRITIC_UTD = 10
DEFAULT_BAFCV6_CRITIC_UTDS = "2"

RETURN_TAG = "Metrics_vs_EnvironmentSteps/AverageReturn"
START_RETURN_TAG = "rollout_skip_eval/start_average_return"
END_RETURN_TAG = "rollout_skip_eval/end_average_return"
ABS_CHANGE_TAG = "rollout_skip_eval/absolute_average_return_change"
REL_CHANGE_TAG = "rollout_skip_eval/relative_average_return_change"
ROLLOUT_SKIP_EVAL_TAGS = (START_RETURN_TAG, END_RETURN_TAG, REL_CHANGE_TAG)


def _display_list(items: Iterable[str], limit: int = 12) -> str:
    items = list(items)
    shown = items[:limit]
    suffix = "" if len(items) <= limit else "\n  ..."
    return "\n  " + "\n  ".join(shown) + suffix


def _parse_int_csv(raw: str, description: str) -> list[int]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "%s must be comma-separated integers: %r" %
                (description, raw)) from exc
        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one %s value is required" % description)
    return values


def _parse_seeds(raw: str) -> list[int]:
    return _parse_int_csv(raw, "Seeds")


def _parse_critic_utds(raw: str) -> list[int]:
    return _parse_int_csv(raw, "critic_utds")


def _env_dir(env: str) -> str:
    return env.split(":", 1)[0]


def _default_output_root(base_dir: str, env: str) -> str:
    return os.path.join(base_dir, _env_dir(env),
                        "plots_bafcv3_tr_rlpd_bafcv6_curves_4g")


def _build_seed_dirs(root: str, seeds: list[int]) -> list[str]:
    return [os.path.join(root, "seed_%d" % seed) for seed in seeds]


def _build_bafcv3_tr_dirs(base_dir: str, env: str,
                          seeds: list[int]) -> list[str]:
    root = os.path.join(base_dir, _env_dir(env), "bafcv3_tr_dmc_4g")
    return _build_seed_dirs(root, seeds)


def _build_bafcv3_tr2_dirs(base_dir: str, env: str, seeds: list[int],
                           critic_utd: int) -> list[str]:
    root = os.path.join(base_dir, _env_dir(env), "bafcv3_tr2_dmc_4g",
                        "critic_utd%d" % critic_utd)
    return _build_seed_dirs(root, seeds)


def _build_bafcv3_dirs_by_critic_utd(
        base_dir: str, env: str, seeds: list[int],
        critic_utds: list[int]) -> dict[int, list[str]]:
    root = os.path.join(base_dir, _env_dir(env), "bafcv3_dmc_4g")
    return {
        critic_utd: _build_seed_dirs(
            os.path.join(root, "critic_utd%d" % critic_utd), seeds)
        for critic_utd in critic_utds
    }


def _build_rlpd_dirs(base_dir: str, env: str, seeds: list[int],
                     critic_utd: int) -> list[str]:
    root = os.path.join(base_dir, _env_dir(env), "rlpd_dmc_4g",
                        "critic_utd%d" % critic_utd)
    return _build_seed_dirs(root, seeds)


def _build_bafcv6_dirs_by_critic_utd(
        base_dir: str, env: str, seeds: list[int],
        critic_utds: list[int]) -> dict[int, list[str]]:
    root = os.path.join(base_dir, _env_dir(env), "bafcv6_dmc_4g")
    return {
        critic_utd: _build_seed_dirs(
            os.path.join(root, "critic_utd%d" % critic_utd), seeds)
        for critic_utd in critic_utds
    }


def _build_bafcv6_resume_dirs(base_dir: str, env: str,
                              seeds: list[int]) -> list[str]:
    root = os.path.join(
        base_dir, _env_dir(env),
        "bafcv6_resume_bafcv3_critic_utd3_ckpt37538_4g",
        "critic_utd2")
    return _build_seed_dirs(root, seeds)


def _source_logdirs(run_dirs: list[str], subdir: str) -> list[str]:
    return [os.path.join(run_dir, subdir) for run_dir in run_dirs]


def _check_dirs(paths: Iterable[str], description: str) -> None:
    missing = [path for path in paths if not os.path.isdir(path)]
    if missing:
        raise ValueError("Missing %s:%s" %
                         (description, _display_list(missing)))


def _has_dirs(paths: Iterable[str]) -> bool:
    return all(os.path.isdir(path) for path in paths)


def _has_scalar_tags(logdirs: Iterable[str], tags: Iterable[str]) -> bool:
    for logdir in logdirs:
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()
        scalar_tags = set(event_acc.Tags().get("scalars", []))
        if any(tag not in scalar_tags for tag in tags):
            return False
    return True


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


def _plot_average_return(
        env: str, bafcv3_tr_dirs: list[str],
        optional_bafcv3_tr2_dirs: list[str] | None,
        optional_bafcv6_resume_dirs: list[str] | None,
        bafcv3_dirs_by_critic_utd: dict[int, list[str]],
        rlpd_dirs: list[str],
        bafcv6_dirs_by_critic_utd: dict[int, list[str]], output_root: str,
        rlpd_critic_utd: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    series = [("BAFCv3-TR", bafcv3_tr_dirs)]
    series.extend(
        ("BAFCv3 cu%d" % critic_utd, run_dirs)
        for critic_utd, run_dirs in sorted(bafcv3_dirs_by_critic_utd.items()))
    series.append(("RLPD cu%d" % rlpd_critic_utd, rlpd_dirs))
    series.extend(
        ("BAFCv6 cu%d" % critic_utd, run_dirs)
        for critic_utd, run_dirs in sorted(bafcv6_dirs_by_critic_utd.items()))
    if optional_bafcv6_resume_dirs is not None:
        series.append(("BAFCv6 resume ckpt37k cu2",
                       optional_bafcv6_resume_dirs))
    if optional_bafcv3_tr2_dirs is not None:
        series.append(("BAFCv3-TR2", optional_bafcv3_tr2_dirs))

    for label, run_dirs in series:
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


def _plot_absolute_return_change(
        env: str, bafcv3_dirs: list[str],
        optional_bafcv3_tr2_dirs: list[str] | None,
        output_root: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    series = [("BAFCv3-TR", bafcv3_dirs)]
    if optional_bafcv3_tr2_dirs is not None:
        series.append(("BAFCv3-TR2", optional_bafcv3_tr2_dirs))

    for label, run_dirs in series:
        eval_logdirs = _source_logdirs(run_dirs, "eval")
        curves = [_read_absolute_change_curve(logdir)
                  for logdir in eval_logdirs]
        aggregate = _aggregate_curves(curves, eval_logdirs, ABS_CHANGE_TAG)
        _plot_aggregate(ax, aggregate, label)
        _print_aggregate_summary(label, ABS_CHANGE_TAG, aggregate)

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


def _plot_relative_return_change(
        env: str, bafcv3_dirs: list[str],
        optional_bafcv3_tr2_dirs: list[str] | None,
        output_root: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    series = [("BAFCv3-TR", bafcv3_dirs)]
    if optional_bafcv3_tr2_dirs is not None:
        series.append(("BAFCv3-TR2", optional_bafcv3_tr2_dirs))

    for label, run_dirs in series:
        eval_logdirs = _source_logdirs(run_dirs, "eval")
        curves = [_read_scalar_curve(logdir, REL_CHANGE_TAG)
                  for logdir in eval_logdirs]
        aggregate = _aggregate_curves(curves, eval_logdirs, REL_CHANGE_TAG)
        _plot_aggregate(ax, aggregate, label)
        _print_aggregate_summary(label, REL_CHANGE_TAG, aggregate)

    return _finish_plot(
        fig=fig,
        ax=ax,
        title="%s Rollout-Skip Relative Average Return Change" % _env_dir(env),
        xlabel="Rollout-skip end step",
        ylabel="Relative Average Return Change",
        output_path=os.path.join(
            output_root, "rollout_skip_eval_relative_average_return_change.png"
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Plot numeric-result BAFCv3-TR, BAFCv3, RLPD, BAFCv6, "
                     "and optional BAFCv3-TR2 4-GPU curves."))
    parser.add_argument("--env", default=DEFAULT_ENV,
                        help="Environment directory/name (default: %(default)s).")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                        help="Base result directory (default: %(default)s).")
    parser.add_argument("--seeds", type=_parse_seeds,
                        default=_parse_seeds(DEFAULT_SEEDS),
                        help="Comma-separated seed IDs (default: %(default)s).")
    parser.add_argument("--bafcv3-critic-utds", type=_parse_critic_utds,
                        default=_parse_critic_utds(
                            DEFAULT_BAFCV3_CRITIC_UTDS),
                        help=("Comma-separated BAFCv3 critic UTD directory "
                              "suffixes (default: %(default)s)."))
    parser.add_argument("--bafcv3-tr2-critic-utd", type=int,
                        default=DEFAULT_BAFCV3_TR2_CRITIC_UTD,
                        help=("BAFCv3-TR2 critic UTD directory suffix "
                              "(default: %(default)s)."))
    parser.add_argument("--rlpd-critic-utd", type=int,
                        default=DEFAULT_RLPD_CRITIC_UTD,
                        help=("RLPD critic UTD directory suffix "
                              "(default: %(default)s)."))
    parser.add_argument("--bafcv6-critic-utds", type=_parse_critic_utds,
                        default=_parse_critic_utds(
                            DEFAULT_BAFCV6_CRITIC_UTDS),
                        help=("Comma-separated BAFCv6 critic UTD directory "
                              "suffixes (default: %(default)s)."))
    parser.add_argument("--output-root", default=None,
                        help=("Output directory. Defaults to "
                              "<base-dir>/<env>/"
                              "plots_bafcv3_tr_rlpd_bafcv6_curves_4g."))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = args.output_root or _default_output_root(args.base_dir,
                                                           args.env)
    bafcv3_tr_dirs = _build_bafcv3_tr_dirs(args.base_dir, args.env,
                                           args.seeds)
    bafcv3_tr2_dirs = _build_bafcv3_tr2_dirs(
        args.base_dir, args.env, args.seeds, args.bafcv3_tr2_critic_utd)
    bafcv6_resume_dirs = _build_bafcv6_resume_dirs(args.base_dir, args.env,
                                                   args.seeds)
    bafcv3_dirs_by_critic_utd = _build_bafcv3_dirs_by_critic_utd(
        args.base_dir, args.env, args.seeds, args.bafcv3_critic_utds)
    rlpd_dirs = _build_rlpd_dirs(args.base_dir, args.env, args.seeds,
                                 args.rlpd_critic_utd)
    bafcv6_dirs_by_critic_utd = _build_bafcv6_dirs_by_critic_utd(
        args.base_dir, args.env, args.seeds, args.bafcv6_critic_utds)

    _check_dirs(bafcv3_tr_dirs, "BAFCv3-TR run directories")
    for critic_utd, bafcv3_dirs in sorted(
            bafcv3_dirs_by_critic_utd.items()):
        _check_dirs(bafcv3_dirs,
                    "BAFCv3 critic_utd%d run directories" % critic_utd)
    _check_dirs(rlpd_dirs, "RLPD run directories")
    for critic_utd, bafcv6_dirs in sorted(
            bafcv6_dirs_by_critic_utd.items()):
        _check_dirs(bafcv6_dirs,
                    "BAFCv6 critic_utd%d run directories" % critic_utd)

    _check_dirs(_source_logdirs(bafcv3_tr_dirs, "train"),
                "BAFCv3-TR train log directories")
    for critic_utd, bafcv3_dirs in sorted(
            bafcv3_dirs_by_critic_utd.items()):
        _check_dirs(_source_logdirs(bafcv3_dirs, "train"),
                    "BAFCv3 critic_utd%d train log directories" % critic_utd)
    _check_dirs(_source_logdirs(rlpd_dirs, "train"),
                "RLPD train log directories")
    for critic_utd, bafcv6_dirs in sorted(
            bafcv6_dirs_by_critic_utd.items()):
        _check_dirs(_source_logdirs(bafcv6_dirs, "train"),
                    "BAFCv6 critic_utd%d train log directories" % critic_utd)

    optional_bafcv3_tr2_dirs = None
    if _has_dirs(bafcv3_tr2_dirs):
        bafcv3_tr2_train_dirs = _source_logdirs(bafcv3_tr2_dirs, "train")
        if _has_dirs(bafcv3_tr2_train_dirs):
            optional_bafcv3_tr2_dirs = bafcv3_tr2_dirs
        else:
            print("skip BAFCv3-TR2: incomplete selected seed train log "
                  "directories")
    else:
        print("skip BAFCv3-TR2: incomplete selected seed run directories")

    optional_bafcv6_resume_dirs = None
    if _has_dirs(bafcv6_resume_dirs):
        bafcv6_resume_train_dirs = _source_logdirs(bafcv6_resume_dirs,
                                                   "train")
        if _has_dirs(bafcv6_resume_train_dirs):
            optional_bafcv6_resume_dirs = bafcv6_resume_dirs
        else:
            print("skip BAFCv6 resume ckpt37k cu2: incomplete selected seed "
                  "train log directories")
    else:
        print("skip BAFCv6 resume ckpt37k cu2: incomplete selected seed run "
              "directories")

    _check_dirs(_source_logdirs(bafcv3_tr_dirs, "eval"),
                "BAFCv3-TR eval log directories")

    optional_bafcv3_tr2_rollout_dirs = None
    bafcv3_tr2_eval_dirs = _source_logdirs(bafcv3_tr2_dirs, "eval")
    if _has_dirs(bafcv3_tr2_eval_dirs):
        if _has_scalar_tags(bafcv3_tr2_eval_dirs, ROLLOUT_SKIP_EVAL_TAGS):
            optional_bafcv3_tr2_rollout_dirs = bafcv3_tr2_dirs
        else:
            print("skip BAFCv3-TR2 rollout-skip: incomplete selected seed "
                  "eval scalar tags")
    else:
        print("skip BAFCv3-TR2 rollout-skip: incomplete selected seed eval "
              "directories")

    print("env: %s" % args.env)
    print("run_set: 4g")
    print("seeds: %s" % ",".join(str(seed) for seed in args.seeds))
    print("bafcv3_critic_utds: %s" %
          ",".join(str(utd) for utd in args.bafcv3_critic_utds))
    print("bafcv3_tr2_critic_utd: %d" % args.bafcv3_tr2_critic_utd)
    print("rlpd_critic_utd: %d" % args.rlpd_critic_utd)
    print("bafcv6_critic_utds: %s" %
          ",".join(str(utd) for utd in args.bafcv6_critic_utds))
    print("bafcv3_tr2: %s" %
          ("included" if optional_bafcv3_tr2_dirs is not None else "skipped"))
    print("bafcv6_resume_ckpt37k_cu2: %s" %
          ("included"
           if optional_bafcv6_resume_dirs is not None else "skipped"))
    print("bafcv3_tr2_rollout_skip: %s" %
          ("included"
           if optional_bafcv3_tr2_rollout_dirs is not None else "skipped"))
    print("output: %s" % output_root)
    print("rollout_skip: BAFCv3-TR plus BAFCv3-TR2 when available")

    _plot_average_return(args.env, bafcv3_tr_dirs, optional_bafcv3_tr2_dirs,
                         optional_bafcv6_resume_dirs,
                         bafcv3_dirs_by_critic_utd, rlpd_dirs,
                         bafcv6_dirs_by_critic_utd, output_root,
                         args.rlpd_critic_utd)
    _plot_absolute_return_change(args.env, bafcv3_tr_dirs,
                                 optional_bafcv3_tr2_rollout_dirs,
                                 output_root)
    _plot_relative_return_change(args.env, bafcv3_tr_dirs,
                                 optional_bafcv3_tr2_rollout_dirs,
                                 output_root)


if __name__ == "__main__":
    main()
