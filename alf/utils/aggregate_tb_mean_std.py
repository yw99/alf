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

"""Aggregate TensorBoard scalar curves across seed runs.

This is intentionally a hardcoded experiment helper. Edit the config block below
to choose eval thresholds, run directories, and TensorBoard scalar tags, then run:

    python alf/utils/aggregate_tb_mean_std.py

The script writes aggregate TensorBoard event files. For each scalar tag, the
``mean``, ``mean_minus_std``, and ``mean_plus_std`` runs all use the original tag
so TensorBoard overlays them in one scalar chart. A separate ``std`` run writes a
``<tag>_std`` scalar for inspecting the standard deviation directly.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Hardcoded experiment config. Edit this block for new aggregation jobs.
# ---------------------------------------------------------------------------

OUTPUT_ROOT = "/root/alf_results_v7_eval/tb_aggregated"
OVERWRITE_OUTPUT = True

RUN_ROOT = (
    "/root/alf_results_v7_eval/hopper/"
    "bafcv3_tr_dmc_evalonly_single_run"
)
# Values should match the directory fragment after `eval`, e.g. eval20.0.
EVAL_THRESHOLDS = ["25.0", "30.0", "40.0"]
NUM_FEATURE_COORDS = 4
METRIC_INTERVAL = 8
# Set to None to aggregate all cap values, or to an int/string such as 3 to
# match only one rollout skip cap.
ROLLOUT_SKIP_CAP = 4
# Set to None to aggregate all actor-LN variants. With None, the run glob
# matches both directories with `_lnTrue`/`_lnFalse` and older directories with
# no `_ln...` suffix.
ACTOR_USE_LN = None
EXCLUDE_SUBSTRINGS = ["freezeEvalSamples", "seed5"]

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
        "tag": "rollout_skip_eval/average_return",
        "subdir": "eval",
    },
]


def _build_target(eval_threshold: str) -> dict:
    cap_pattern = "*" if ROLLOUT_SKIP_CAP is None else str(ROLLOUT_SKIP_CAP)
    cap_label = "any" if ROLLOUT_SKIP_CAP is None else str(ROLLOUT_SKIP_CAP)
    if ACTOR_USE_LN is None:
        actor_ln_label = "any"
        seed_pattern = "seed*"
    else:
        actor_ln_label = str(ACTOR_USE_LN)
        seed_pattern = "seed*_ln%s" % actor_ln_label

    run_pattern = (
        "eval%s_nf%d_mi%d_cap%s_%s" %
        (eval_threshold, NUM_FEATURE_COORDS, METRIC_INTERVAL,
         cap_pattern, seed_pattern))
    return {
        "name": (
            "hopper_bafcv3_tr_eval%s_nf%d_mi%d_cap%s_ln%s" %
            (eval_threshold, NUM_FEATURE_COORDS, METRIC_INTERVAL,
             cap_label, actor_ln_label)),
        "run_glob": os.path.join(RUN_ROOT, run_pattern),
        "exclude_substrings": EXCLUDE_SUBSTRINGS,
        "curves": CURVES,
    }


MANUAL_TARGETS = [
    {
        "name": "hopper_bafcv3_tr_eval25_seed4_eval20_seed3_nf4_mi8_cap4",
        "run_dirs": [
            os.path.join(RUN_ROOT, "eval25.0_nf4_mi8_cap4_seed4_lnTrue"),
            os.path.join(RUN_ROOT, "eval20.0_nf4_mi8_cap4_seed3"),
        ],
        "curves": CURVES,
    },
]


TARGETS = [_build_target(threshold) for threshold in EVAL_THRESHOLDS] + MANUAL_TARGETS


@dataclass(frozen=True)
class ScalarCurve:
    steps: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class AggregateCurve:
    steps: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    mean_minus_std: np.ndarray
    mean_plus_std: np.ndarray


def _slugify(text: str) -> str:
    """Convert a TensorBoard tag or path fragment into a stable directory name."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return slug or "curve"


def _display_list(items: Iterable[str], limit: int = 12) -> str:
    items = list(items)
    shown = items[:limit]
    suffix = "" if len(items) <= limit else "\n  ..."
    return "\n  " + "\n  ".join(shown) + suffix


def _expand_run_dirs(target: dict) -> list[str]:
    run_dirs = []
    if target.get("run_glob"):
        run_dirs.extend(glob.glob(target["run_glob"]))
    if target.get("run_dirs"):
        run_dirs.extend(target["run_dirs"])

    # Deduplicate after normalizing, then sort for reproducible output.
    run_dirs = sorted({os.path.abspath(path) for path in run_dirs})
    exclude_substrings = target.get("exclude_substrings", [])
    if exclude_substrings:
        run_dirs = [
            path for path in run_dirs
            if not any(part in path for part in exclude_substrings)
        ]

    if not run_dirs:
        raise ValueError(
            "No run directories matched target %r. run_glob=%r run_dirs=%r" %
            (target.get("name"), target.get("run_glob"), target.get("run_dirs")))

    missing = [path for path in run_dirs if not os.path.isdir(path)]
    if missing:
        raise ValueError("These run directories do not exist:%s" %
                         _display_list(missing))

    return run_dirs


def _read_scalar_curve(logdir: str, tag: str) -> ScalarCurve:
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

    # Keep the last observed value for duplicate steps. Duplicate steps can happen
    # when a run is resumed and multiple event files live in the same directory.
    by_step = {}
    for event in events:
        by_step[int(event.step)] = float(event.value)

    steps = np.array(sorted(by_step), dtype=np.float64)
    values = np.array([by_step[int(step)] for step in steps], dtype=np.float64)
    return ScalarCurve(steps=steps, values=values)


def _source_logdirs(run_dirs: list[str], subdir: str | None) -> list[str]:
    if not subdir:
        return run_dirs
    return [os.path.join(run_dir, subdir) for run_dir in run_dirs]


def aggregate_scalar(logdirs: list[str], tag: str) -> AggregateCurve:
    """Read ``tag`` from ``logdirs`` and compute aligned mean/std curves."""
    if not logdirs:
        raise ValueError("No log directories were provided for tag %r" % tag)

    curves = [_read_scalar_curve(logdir, tag) for logdir in logdirs]
    if len(curves) == 1:
        print("  warning: only one run for %s; std will be zero" % tag)

    overlap_start = max(curve.steps[0] for curve in curves)
    overlap_end = min(curve.steps[-1] for curve in curves)
    if overlap_start > overlap_end:
        raise ValueError(
            "No overlapping step range for tag %r. Per-run ranges:%s" %
            (tag, _display_list("%s: [%d, %d]" %
                                (logdir, curve.steps[0], curve.steps[-1])
                                for logdir, curve in zip(logdirs, curves))))

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
    return AggregateCurve(steps=common_steps,
                          mean=mean,
                          std=std,
                          mean_minus_std=mean - std,
                          mean_plus_std=mean + std)


def _write_scalar_series(logdir: str, tag: str, steps: np.ndarray,
                         values: np.ndarray) -> None:
    writer = SummaryWriter(logdir)
    try:
        for step, value in zip(steps, values):
            writer.add_scalar(tag, float(value), int(round(float(step))))
        writer.flush()
    finally:
        writer.close()


def write_aggregate_curve(output_dir: str, tag: str,
                          aggregate: AggregateCurve) -> None:
    """Write mean, std, and +/-std bounds as TensorBoard event runs."""
    _write_scalar_series(os.path.join(output_dir, "mean"), tag,
                         aggregate.steps, aggregate.mean)
    _write_scalar_series(os.path.join(output_dir, "mean_minus_std"), tag,
                         aggregate.steps, aggregate.mean_minus_std)
    _write_scalar_series(os.path.join(output_dir, "mean_plus_std"), tag,
                         aggregate.steps, aggregate.mean_plus_std)
    _write_scalar_series(os.path.join(output_dir, "std"), "%s_std" % tag,
                         aggregate.steps, aggregate.std)


def _prepare_target_output(output_root: str, target_name: str) -> str:
    target_output = os.path.abspath(os.path.join(output_root, target_name))
    root = os.path.abspath(output_root)
    if not target_output.startswith(root + os.sep):
        raise ValueError("Refusing to write outside OUTPUT_ROOT: %s" %
                         target_output)

    if OVERWRITE_OUTPUT and os.path.isdir(target_output):
        shutil.rmtree(target_output)
    os.makedirs(target_output, exist_ok=True)
    return target_output


def aggregate_target(target: dict, output_root: str = OUTPUT_ROOT) -> None:
    target_name = target["name"]
    run_dirs = _expand_run_dirs(target)
    target_output = _prepare_target_output(output_root, _slugify(target_name))

    print("target: %s" % target_name)
    print("  runs:%s" % _display_list(run_dirs))
    print("  output: %s" % target_output)

    for curve_cfg in target["curves"]:
        tag = curve_cfg["tag"]
        subdir = curve_cfg.get("subdir")
        source_dirs = _source_logdirs(run_dirs, subdir)
        curve_slug = _slugify("%s_%s" % (subdir, tag) if subdir else tag)
        curve_output = os.path.join(target_output, curve_slug)

        print("  curve: %s%s" % (tag, " from %s/" % subdir if subdir else ""))
        aggregate = aggregate_scalar(source_dirs, tag)
        write_aggregate_curve(curve_output, tag, aggregate)

        print("    points: %d" % aggregate.steps.size)
        print("    step range: %d -> %d" %
              (int(round(float(aggregate.steps[0]))),
               int(round(float(aggregate.steps[-1])))))
        print("    final mean/std: %.6g / %.6g" %
              (aggregate.mean[-1], aggregate.std[-1]))
        print("    wrote: %s" % curve_output)


def main() -> None:
    for target in TARGETS:
        aggregate_target(target, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
