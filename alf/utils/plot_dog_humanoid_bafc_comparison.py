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

"""Plot selected Dog and Humanoid tasks plus Hopper Hop BAFC experiments.

The experiment selection combines runs copied from server-copy directories. Server 3 runs
use shorthand directories under <workspace-root>/server3_copy.
Run from the repository root, for example::

    python alf/utils/plot_dog_humanoid_bafc_comparison.py
    python alf/utils/plot_dog_humanoid_bafc_comparison.py --task dog_trot
    python alf/utils/plot_dog_humanoid_bafc_comparison.py --task dog_trot_trust
    python alf/utils/plot_dog_humanoid_bafc_comparison.py --task dog_walk_tr2_resume

For each environment, the script writes an AverageReturn comparison, including
BAFCv6 (labeled Ours_reweight) where runs are available. The main Dog and
Humanoid Walk/Run comparisons show SAC+, TD3+, Ours, and Ours_reweight
in that legend order, excluding TR2. Humanoid Run BAFCv6
uses seeds 0--3 from <workspace-root>/server7_copy. Dog Fetch, Walk,
Trot, and Stand BAFCv6 use seeds 0--3 from server2_copy, server9_copy,
server_copy, and server9_copy, respectively. RLPD is labeled
SAC+ throughout; algorithm colors are shared across plot families. It also writes BAFC_TR trust diagnostics for
the three environments with BAFC_TR runs, the two-seed Humanoid comparison,
and focused Ours-vs-SAC+ AverageReturn plots for Dog Fetch, Dog Run,
Dog Stand, Dog Trot, Dog Walk, Humanoid Walk, Humanoid Run, Humanoid Stand,
and Hopper Hop. Each task with all five algorithms also gets an Ours/SAC+/
TD3+/SAC/TD3 comparison in that legend order. Additional SAC/TD3/SAC+ comparisons use seeds 0--3 where available,
with TD3v2 labeled TD3 and RLPD labeled SAC+. Humanoid Walk, Run, and Stand
also include TD3, labeled TD3+, using seeds 0--3 in both comparisons
(server8, server4, and server3 copies). SAC and TD3v2 are discovered
across server copies; the longest training budget with all four seeds
is preferred. Unavailable four-seed curves are reported and omitted.
Dog Walk and Stand also include four-seed TD3+ runs from server9_copy
in both comparisons, as do Dog Fetch, Run, and Trot from server3_copy.
All Dog plots use exactly 0--200,000 environment steps, and all Humanoid
plots use exactly 0--150,000, including specialized comparisons and diagnostics.
Dog Walk BAFCv3 and RLPD use
the extended800k and reconstruction800k runs from server3_copy; the SAC
baseline prefers the four reconstruction800k runs from the same copy.
Continuation curves prepend the matching original seed history at absolute steps.
Humanoid Stand return plots use only the first 150,000 environment steps.
A separate seeds 0--1
comparison includes the TR2 resume study 20260917T054927Z; continuation curves
start at their recorded absolute environment steps and use only seed overlap.
A companion plot shows local rollout skip percentages between logged samples
(roughly 1k environment steps), placed at interval midpoints, using counter
differences rather than cumulative skip fractions.
The dog_trot_trust plot selects one checkpoint per seed by final return
at the shared last logged step (ties retain BAFCv3), from 120k/140k/160k
TR2 restarts or 200k (unchanged BAFCv3). It prepends BAFCv3 through the
restart step. This is a retrospective best-checkpoint comparison; the
selection scores and source paths are saved beside the plot as JSON.
Dog Run also compares 120k/140k/160k and auto-120k restarts, with BAFCv3
as the no-restart option. Humanoid Run compares 90k/105k BAFCv6 restarts
with unrestarted BAFCv3 and full BAFCv6 (0k). Restart histories come from
BAFCv3; a selected 0k run retains its own history and uses the full task
horizon without enable markers. TR2 selections never include v6 candidates. These title-free selection plots start 30k before the earliest selected restart
and mark each selected enable step with a faint dashed vertical line. Run them via
--tasks dog_walk_trust dog_run_trust dog_trot_trust humanoid_run_trust.
Dog Walk selects BAFCv3, full TR2, 75k/105k UTD11 restarts, and the
available 105k UTD3 restarts for seeds 0--1. Only candidates reaching the
full baseline endpoint are eligible. Seed 2 is temporarily pinned to
105k/UTD11 while its 75k continuation finishes. Dog Walk selection return
and local skip-rate plots use a 70k--200k window; the skip-rate plot averages
the selected four seeds, with zero skipping before each enable checkpoint.
Curves are aligned on
their overlapping
environment-step range, linearly interpolated, and plotted as the unsmoothed
across-seed mean with a population +/-1 standard deviation band.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
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
ENV_STEPS_TAG = "Metrics/EnvironmentSteps"
SKIP_COUNT_TAG = "BafcAlgorithmV3TR2/rollout_skip_due_eval_gate_count"
ROLLOUT_COUNT_TAG = "BafcAlgorithmV3TR2/rollout_opportunity_count"
INITIAL_EVAL_TRUST_THRESHOLD = 30.0
PLOT_TASKS = ("dog", "dog_fetch", "dog_run", "dog_stand", "dog_trot",
              "humanoid", "humanoid_run", "humanoid_stand", "hopper_hop")

_PLOTTED_TAGS = (
    RETURN_TAG,
    ENV_STEPS_TAG,
    SKIP_COUNT_TAG,
    ROLLOUT_COUNT_TAG,
    EVAL_TRUST_OVER_MAX_TAG,
    EVAL_TRUST_METRIC_TAG,
)
_SCALAR_CURVE_CACHE: dict[tuple[str, str], ScalarCurve] = {}

ALGORITHM_LABELS = {"RLPD": "SAC+", "BAFCv6": "Ours_reweight"}

ALGORITHM_COLORS = {
    "SAC+": "tab:orange",
    "Ours": "tab:blue",
    "Ours_trust": "tab:brown",
    "BAFCv3": "tab:blue",
    "SAC": "tab:green",
    "TD3": "tab:red",
    "TD3+": "tab:purple",
    "BAFC_TR": "tab:brown",
    "Ours_reweight": "tab:pink",
    "BAFCv3_TR2_reweight": "tab:cyan",
    "BAFCv7": "tab:purple",
    "BAFC_nCritic1": "tab:blue",
    "BAFC_nCritic8": "tab:green",
}
for alias, label in ALGORITHM_LABELS.items():
    ALGORITHM_COLORS[alias] = ALGORITHM_COLORS[label]

BASELINE_ALGORITHM_COLORS = {
    label: ALGORITHM_COLORS[label] for label in ("SAC", "TD3", "SAC+", "TD3+")
}
FOCUSED_ALGORITHM_COLORS = {
    label: ALGORITHM_COLORS[label] for label in ("SAC+", "TD3+", "Ours")
}


def _algorithm_label(label: str) -> tuple[str, int | None]:
    """Resolve display aliases while retaining individual-seed information."""
    match = re.fullmatch(r"(.+)_s(\d+)", label)
    algorithm, seed = (match.group(1), int(match.group(2))) if match else (label, None)
    return ALGORITHM_LABELS.get(algorithm, algorithm), seed


def _display_algorithm_label(label: str) -> str:
    algorithm, seed = _algorithm_label(label)
    return algorithm if seed is None else "%s_s%d" % (algorithm, seed)


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


def _read_run_curve(run_dir: str, tag: str) -> ScalarCurve:
    """Prepend original history to the copied Dog Walk continuation logs."""
    curve = _read_scalar_curve(os.path.join(run_dir, "train"), tag)
    match = re.fullmatch(
        r"dog_walk_(bafcv3_extended800k|rlpd_reconstruction800k|"
        r"sac_reconstruction800k)_s([0123])", os.path.basename(run_dir))
    if not match or curve.steps[0] == 0:
        return curve
    experiment, seed_text = match.groups()
    seed = int(seed_text)
    workspace_root = os.path.dirname(os.path.dirname(run_dir))
    if experiment == "sac_reconstruction800k":
        predecessor = os.path.join(workspace_root, "server3_copy",
                                   "dog_walk_sac_s%d" % seed)
    else:
        server = "server2_copy" if seed < 2 else "server_copy"
        if experiment == "rlpd_reconstruction800k":
            name = "dog_rlpd_s%d" % seed
        else:
            name = ("dog_bafcv3_s%d" if seed < 2 else
                    "dog_bafc_trainable_rtT_s%d") % seed
        predecessor = os.path.join(workspace_root, server, name)
    history = _read_scalar_curve(os.path.join(predecessor, "train"), tag)
    # Continuations retain absolute steps and take precedence on overlap.
    before_resume = history.steps < curve.steps[0]
    return ScalarCurve(
        steps=np.concatenate((history.steps[before_resume], curve.steps)),
        values=np.concatenate((history.values[before_resume], curve.values)))


def aggregate_scalar(run_dirs: list[str], tag: str) -> AggregateCurve:
    logdirs = [os.path.join(run_dir, "train") for run_dir in run_dirs]
    missing = [path for path in logdirs if not os.path.isdir(path)]
    if missing:
        raise ValueError("Missing required TensorBoard directories:%s" %
                         _display_list(missing))
    curves = [_read_run_curve(run_dir, tag) for run_dir in run_dirs]
    return aggregate_curves(curves, logdirs, tag)


def build_run_groups(
        workspace_root: str, local_results_root: str, server_copy_root: str,
        server2_copy_root: str,
        server4_copy_root: str | None = None) -> dict[str, dict[str, list[str]]]:
    """Return the fixed experiment mapping, with every list ordered by seed."""
    server4_copy_root = server4_copy_root or os.path.join(
        workspace_root, "server4_copy")
    server3_copy_root = os.path.join(workspace_root, "server3_copy")

    groups = {
        "dog": {
            "RLPD": [
                os.path.join(server3_copy_root,
                             "dog_walk_rlpd_reconstruction800k_s%d" % seed)
                for seed in range(4)
            ],
            "BAFCv3": [
                os.path.join(server3_copy_root,
                             "dog_walk_bafcv3_extended800k_s%d" % seed)
                for seed in range(4)
            ],
            "BAFC_TR": [os.path.join(server3_copy_root,
                                     "dog_walk_bafcv3_tr2_decay_s%d" % seed)
                        for seed in range(4)],
            "BAFCv6": [
                os.path.join(workspace_root, "server9_copy",
                             "dog_walk_bafcv6_s%d" % seed)
                for seed in range(4)
            ],
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
            "BAFCv6": [
                os.path.join(server2_copy_root,
                             "dog_fetch_bafcv6_s%d" % seed)
                for seed in range(4)
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
        "dog_stand": {
            "RLPD": [
                os.path.join(server_copy_root,
                             "dog_stand_rlpd_s%d" % seed)
                for seed in range(4)
            ],
            "BAFCv3": [
                os.path.join(server2_copy_root,
                             "dog_stand_bafcv3_rtT_s%d" % seed)
                for seed in range(4)
            ],
            "BAFCv6": [
                os.path.join(workspace_root, "server9_copy",
                             "dog_stand_bafcv6_s%d" % seed)
                for seed in range(4)
            ],
        },
        "dog_trot": {
            "RLPD": [
                os.path.join(server2_copy_root,
                             "dog_trot_rlpd_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_trot_rlpd_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFCv3": [
                os.path.join(server2_copy_root,
                             "dog_trot_bafcv3_rtT_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server_copy_root,
                             "dog_trot_bafcv3_rtT_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFCv6": [
                os.path.join(server_copy_root,
                             "dog_trot_bafcv6_s%d" % seed)
                for seed in range(4)
            ],
        },
        "humanoid": {
            "RLPD": [
                os.path.join(server3_copy_root, "humanoid_walk_rlpd_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server2_copy_root, "hum_rlpd_s%d" % seed)
                for seed in (2, 3)
            ],
            "BAFCv3": [os.path.join(server3_copy_root,
                                    "humanoid_walk_bafcv3_rtT_s%d" % seed)
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
            "BAFCv6": [
                os.path.join(server_copy_root,
                             "hum_bafcv6_trainable_rtT_s%d" % seed)
                for seed in (0, 1)
            ] + [
                os.path.join(server2_copy_root,
                             "humanoid_walk_bafcv6_s%d" % seed)
                for seed in (2, 3)
            ],
        },
        "humanoid_run": {
            "RLPD": [
                os.path.join(server4_copy_root,
                             "humanoid_run_rlpd_s%d" % seed)
                for seed in range(4)
            ],
            "BAFCv3": [
                os.path.join(server3_copy_root, "humanoid_run_bafcv3_rtT_s%d" % seed)
                for seed in range(4)
            ],
            "BAFCv6": [
                os.path.join(workspace_root, "server7_copy",
                             "humanoid_run_bafcv6_s%d" % seed)
                for seed in range(4)
            ],
        },
        "humanoid_stand": {
            "RLPD": [
                os.path.join(server4_copy_root,
                             "humanoid_stand_rlpd_s%d" % seed)
                for seed in range(4)
            ],
            "BAFCv3": [
                os.path.join(server_copy_root,
                             "humanoid_stand_bafcv3_rtT_s%d" % seed)
                for seed in range(4)
            ],
        },
        "humanoid_seed01": {
            "RLPD": [
                os.path.join(server3_copy_root, "humanoid_walk_rlpd_s%d" % seed)
                for seed in (0, 1)
            ],
            "BAFCv3": [os.path.join(server3_copy_root,
                                    "humanoid_walk_bafcv3_rtT_s%d" % seed)
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
    # Retained for compatibility; local_results_root now only sets output defaults.
    del local_results_root
    return groups


def build_rlpd_ours_run_groups(
        workspace_root: str, local_results_root: str, server_copy_root: str,
        server2_copy_root: str,
        server4_copy_root: str | None = None) -> dict[str, dict[str, list[str]]]:
    """Return RLPD/Ours groups with available Dog and Humanoid TD3+ runs.

    Dog Fetch, Dog Run, Dog Stand, Dog Trot, Dog Walk (``dog``), Humanoid
    Walk (``humanoid``), Humanoid Run, and Humanoid Stand use seeds 0--3. Hopper Hop uses
    seeds 0--2, with BAFCv3 critic UTD 11 for seeds 0--1 and critic UTD 3 for seed 2.
    """
    existing = build_run_groups(workspace_root, local_results_root,
                                server_copy_root, server2_copy_root,
                                server4_copy_root)

    server3_copy_root = os.path.join(workspace_root, "server3_copy")
    server4_copy_root = server4_copy_root or os.path.join(
        workspace_root, "server4_copy")
    hopper_ours = [
        os.path.join(server3_copy_root, "hopper_hop_bafcv3_nCritic8_utd11_s%d" % seed)
        for seed in (0, 1)
    ] + [os.path.join(server3_copy_root,
                      "hopper_hop_bafcv3_nCritic8_utd3_focused_s2")]

    return {
        "dog_fetch": {
            "Ours": existing["dog_fetch"]["BAFCv3"],
            "RLPD": existing["dog_fetch"]["RLPD"],
            "TD3+": [
                os.path.join(server3_copy_root, "dog_fetch_td3_s%d" % seed)
                for seed in range(4)
            ],
        },
        "dog_run": {
            "Ours": existing["dog_run"]["BAFCv3"],
            "RLPD": existing["dog_run"]["RLPD"],
            "TD3+": [
                os.path.join(server3_copy_root, "dog_run_td3_s%d" % seed)
                for seed in range(4)
            ],
        },
        "dog_stand": {
            "Ours": existing["dog_stand"]["BAFCv3"],
            "RLPD": existing["dog_stand"]["RLPD"],
            "TD3+": [
                os.path.join(workspace_root, "server9_copy",
                             "dog_stand_td3_s%d" % seed)
                for seed in range(4)
            ],
        },
        "dog_trot": {
            "Ours": existing["dog_trot"]["BAFCv3"],
            "RLPD": existing["dog_trot"]["RLPD"],
            "TD3+": [
                os.path.join(server3_copy_root, "dog_trot_td3_s%d" % seed)
                for seed in range(4)
            ],
        },
        "dog": {
            "Ours": existing["dog"]["BAFCv3"],
            "RLPD": existing["dog"]["RLPD"],
            "TD3+": [
                os.path.join(workspace_root, "server9_copy",
                             "dog_walk_td3_s%d" % seed)
                for seed in range(4)
            ],
        },
        "humanoid": {
            "Ours": existing["humanoid"]["BAFCv3"],
            "RLPD": existing["humanoid"]["RLPD"],
            "TD3+": [
                os.path.join(workspace_root, "server8_copy",
                             "humanoid_walk_td3_s%d" % seed)
                for seed in range(4)
            ],
        },
        "humanoid_run": {
            "Ours": existing["humanoid_run"]["BAFCv3"],
            "RLPD": existing["humanoid_run"]["RLPD"],
            "TD3+": [
                os.path.join(server4_copy_root, "humanoid_run_td3_s%d" % seed)
                for seed in range(4)
            ],
        },
        "humanoid_stand": {
            "Ours": existing["humanoid_stand"]["BAFCv3"],
            "RLPD": existing["humanoid_stand"]["RLPD"],
            "TD3+": [
                os.path.join(server3_copy_root, "humanoid_stand_td3_s%d" % seed)
                for seed in range(4)
            ],
        },
        "hopper_hop": {
            "Ours": hopper_ours,
            "RLPD": [
                os.path.join(server3_copy_root, "hopper_hop_rlpd_s%d" % seed)
                for seed in (0, 1, 2)
            ],
        },
    }


def build_baseline_run_groups(
        workspace_root: str, rlpd_groups: dict[str, dict[str, list[str]]],
        server_roots: Iterable[str],
        tasks: Iterable[str] = PLOT_TASKS) -> dict[str, dict[str, list[str]]]:
    """Discover four-seed SAC/TD3v2 sets and reuse available RLPD seeds.

    RLPD uses seeds 0--3, except Hopper Hop which has seeds 0--2.

    Accept copied names such as ``dog_walk_td3v2_s0`` and
    ``dog_fetch_sac_800k_s0``. Prefer the largest training budget
    having all four seeds, without mixing budgets or counting copies twice.
    Unnamed budgets are read from saved alf_config.py when available.
    Conflicting copies of the same run fail with their paths for inspection.
    """
    roots = sorted(set(os.path.realpath(root) for root in [
        *glob.glob(os.path.join(workspace_root, "server*copy")),
        *server_roots,
    ]))
    entries = [os.path.join(root, name) for root in roots
               if os.path.isdir(root) for name in sorted(os.listdir(root))]
    groups = {}
    for task in tasks:
        aliases = {"dog": ("dog", "dog_walk"),
                   "humanoid": ("hum", "humanoid", "humanoid_walk")}.get(
                       task, (task,))
        groups[task] = {}
        for algorithm, label in (("sac", "SAC"), ("td3v2", "TD3")):
            if task == "dog" and algorithm == "sac":
                extended = [os.path.join(
                    workspace_root, "server3_copy",
                    "dog_walk_sac_reconstruction800k_s%d" % seed)
                            for seed in range(4)]
                if all(os.path.isdir(os.path.join(run, "train"))
                       for run in extended):
                    groups[task][label] = extended
                    continue
            pattern = re.compile(
                r"(?:%s)_%s(?:_(\d+)([km]))?_s([0123])$" %
                ("|".join(map(re.escape, aliases)), algorithm))
            candidates = {}
            for path in entries:
                match = pattern.fullmatch(os.path.basename(path))
                if not match or not os.path.isdir(os.path.join(path, "train")):
                    continue
                amount, unit, seed = match.groups()
                budget = (int(amount) * (1000 if unit == "k" else 1000000)
                          if amount else 0)
                # Some copied run names omit the training budget. Read the
                # saved literal configuration without executing experiment code.
                config_path = os.path.join(path, "alf_config.py")
                if not amount and os.path.isfile(config_path):
                    with open(config_path) as config_file:
                        config_budget = re.search(
                            r"['\"]TrainerConfig\.num_env_steps['\"]\s*:\s*(\d+)",
                            config_file.read())
                    if config_budget:
                        budget = int(config_budget.group(1))
                candidates.setdefault(budget, {}).setdefault(
                    int(seed), set()).add(os.path.realpath(path))
            complete = [budget for budget, seeds in candidates.items()
                        if set(seeds) == {0, 1, 2, 3}]
            if not complete:
                print("%s: omitting %s; no complete seeds 0-3 in server copies"
                      % (task, label))
                continue
            seeds = candidates[max(complete)]
            runs = []
            for seed in range(4):
                paths = sorted(seeds[seed])
                if len(paths) != 1:
                    raise ValueError(
                        "Ambiguous %s %s seed %d copies:%s" %
                        (task, label, seed, _display_list(paths)))
                runs.append(paths[0])
            groups[task][label] = runs
        groups[task]["SAC+"] = rlpd_groups[task]["RLPD"][:4]
        if "TD3+" in rlpd_groups[task]:
            groups[task]["TD3+"] = rlpd_groups[task]["TD3+"][:4]
    return groups


def build_additional_run_groups(
        workspace_root: str, local_results_root: str, server_copy_root: str,
        server2_copy_root: str,
        server4_copy_root: str) -> dict[str, dict[str, list[str]]]:
    """Return the run mappings for the requested specialized comparisons."""
    existing = build_run_groups(workspace_root, local_results_root,
                                server_copy_root, server2_copy_root,
                                server4_copy_root)
    server3_copy_root = os.path.join(workspace_root, "server3_copy")
    hopper_seeds = (0, 2, 3)

    humanoid_reweight = [
        os.path.join(server4_copy_root,
                     "humanoid_walk_bafcv3_tr2_reweight_s%d" % seed)
        for seed in range(4)
    ]
    humanoid_individual = {}
    for label, run_dirs in (("RLPD", existing["humanoid"]["RLPD"]),
                            ("BAFCv3_TR2_reweight", humanoid_reweight)):
        for seed, run_dir in enumerate(run_dirs):
            humanoid_individual["%s_s%d" % (label, seed)] = [run_dir]

    return {
        "hopper_hop_ncritic": {
            "RLPD": [
                os.path.join(server3_copy_root, "hopper_hop_rlpd_s%d" % seed)
                for seed in hopper_seeds
            ],
            "BAFC_nCritic1": [
                os.path.join(server3_copy_root,
                             "hopper_hop_bafcv3_nCritic1_utd3_updates12_s%d" % seed)
                for seed in hopper_seeds
            ],
            "BAFC_nCritic8": [
                os.path.join(server3_copy_root,
                             "hopper_hop_bafcv3_nCritic8_utd3_updates12_s%d" % seed)
                for seed in hopper_seeds
            ],
        },
        "humanoid_reweight": {
            "RLPD": existing["humanoid"]["RLPD"],
            "BAFCv3_TR2_reweight": humanoid_reweight,
        },
        "humanoid_reweight_individual": humanoid_individual,
        "hopper_hop_seed0_v7": {
            "BAFCv7": [
                os.path.join(
                    server_copy_root,
                    "hopper_hop_bafcv7_ensemble_base_lambda010_s0")
            ],
            "RLPD": [os.path.join(server3_copy_root, "hopper_hop_rlpd_s0")],
            "Ours": [os.path.join(server3_copy_root,
                                  "hopper_hop_bafcv3_nCritic1_utd3_updates12_s0")],
        },
    }


def _plot_aggregate(ax: plt.Axes, aggregate: AggregateCurve, label: str,
                    color: str | None = None) -> None:
    algorithm, seed = _algorithm_label(label)
    color = color or ALGORITHM_COLORS[algorithm]
    linestyle = "-" if seed is None else ("-", "--", "-.", ":")[seed % 4]
    ax.plot(aggregate.steps, aggregate.mean, color=color, linewidth=2,
            label=_display_algorithm_label(label), linestyle=linestyle)
    ax.fill_between(aggregate.steps, aggregate.mean_minus_std,
                    aggregate.mean_plus_std, color=color, alpha=0.18,
                    linewidth=0)


def _print_summary(env: str, label: str, tag: str, seed_count: int,
                   aggregate: AggregateCurve) -> None:
    print("%s | %s | seeds=%d | %s | range=%d -> %d | "
          "final mean/std=%.6g / %.6g" %
          (env, _display_algorithm_label(label), seed_count, tag, int(aggregate.steps[0]),
           int(aggregate.steps[-1]), aggregate.mean[-1], aggregate.std[-1]))


def _format_environment_steps(value: float, _position: int | None) -> str:
    if abs(value) < 1000:
        return "%g" % value
    return "%gk" % (value / 1000)


def _finish_plot(fig: plt.Figure, ax: plt.Axes, title: str, ylabel: str,
                 output_path: str, xlabel: str = "Environment steps",
                 human_readable_x_ticks: bool = False,
                 xlim: tuple[float, float] | None = None) -> str:
    # Apply the same exact horizon to every comparison and diagnostic family.
    task = os.path.basename(output_path)
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif task.startswith("dog_"):
        ax.set_xlim(0, 200_000)
    elif task.startswith("humanoid_"):
        ax.set_xlim(0, 150_000)
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


def build_comparison_group(runs: dict[str, list[str]],
                           td3_runs: list[str]) -> dict[str, list[str]]:
    """Select the main comparison curves in a shared legend order, without TR2."""
    return {
        "SAC+": runs["RLPD"],
        "TD3+": td3_runs,
        "Ours": runs["BAFCv3"],
        "Ours_reweight": runs["BAFCv6"],
    }


def plot_ours_vs_baselines(env: str, focused: dict[str, list[str]],
                           baseline: dict[str, list[str]],
                           output_root: str,
                           workspace_root: str | None = None) -> str | None:
    """Plot all five algorithms with a consistent legend order and task horizon."""
    ours = focused["Ours"]
    if env == "dog_stand" and workspace_root is not None:
        ours = [os.path.join(
            workspace_root, "h200_copy", "dog_stand",
            "bafcv3_seed0123_8g_4rank_frozen_eval_samples",
            "fixed_pairingFalse_num_sampled_critic8", "critic_utd11",
            "seed_%d" % seed) for seed in range(4)]
    available = {**baseline, "Ours": ours}
    labels = ("Ours", "SAC+", "TD3+", "SAC", "TD3")
    missing = [label for label in labels if label not in available]
    if missing:
        print("%s: skipping five-algorithm comparison; missing %s" %
              (env, ", ".join(missing)))
        return None
    xlim = None
    if env == "dog_stand" and workspace_root is not None:
        xlim = (0, float(aggregate_scalar(ours, RETURN_TAG).steps[-1]))
    return plot_average_return(
        env, {label: available[label] for label in labels}, output_root,
        title={"dog": "Dog Walk", "humanoid": "Humanoid Walk"}.get(
            env, env.replace("_", " ").title()),
        xlabel="Environment Steps", ylabel="Average Episodic Return",
        colors=ALGORITHM_COLORS, human_readable_x_ticks=True,
        filename="%s_ours_sacplus_td3plus_sac_td3_average_return_vs_env_steps.png"
                 % env, xlim=xlim)


def plot_average_return(env: str, groups: dict[str, list[str]],
                        output_root: str, title: str | None = None,
                        filename: str | None = None,
                        xlabel: str = "Environment steps",
                        ylabel: str = "Average Return",
                        colors: dict[str, str] | None = None,
                        human_readable_x_ticks: bool = False,
                        xlim: tuple[float, float] | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for label, run_dirs in groups.items():
        aggregate = aggregate_scalar(run_dirs, RETURN_TAG)
        if xlim is not None:
            visible = (aggregate.steps >= xlim[0]) & (aggregate.steps <= xlim[1])
            aggregate = AggregateCurve(aggregate.steps[visible],
                                       aggregate.mean[visible],
                                       aggregate.std[visible])
        if env == "humanoid_stand":
            visible = aggregate.steps <= 150_000
            aggregate = AggregateCurve(aggregate.steps[visible],
                                       aggregate.mean[visible],
                                       aggregate.std[visible])
        color = colors[label] if colors is not None else None
        _plot_aggregate(ax, aggregate, label, color=color)
        _print_summary(env, label, RETURN_TAG, len(run_dirs), aggregate)
    if any(_algorithm_label(label)[0] == "Ours_reweight" for label in groups):
        title = ""
    elif title is None:
        title = "%s Average Return vs Environment Steps" % (
            env.replace("_", " ").title())
    filename = filename or "%s_average_return_vs_env_steps.png" % env
    return _finish_plot(fig, ax, title, ylabel,
                        os.path.join(output_root, filename), xlabel=xlabel,
                        human_readable_x_ticks=human_readable_x_ticks, xlim=xlim)


def select_trust_restart(baseline: ScalarCurve,
                         restarts: dict[int, ScalarCurve]):
    """Select by return at a shared endpoint, then splice baseline history.

    The 200k option retains BAFCv3, including on ties. Selection is per seed,
    never a pointwise maximum. The caller supplies curves capped at 200k.
    """
    endpoint = min(c.steps[-1] for c in [baseline, *restarts.values()])
    if endpoint < max(c.steps[0] for c in [baseline, *restarts.values()]):
        raise ValueError("Restart candidates have no common step range")
    scores = {200000: float(np.interp(endpoint, baseline.steps, baseline.values))}
    scores.update({step: float(np.interp(endpoint, c.steps, c.values))
                   for step, c in sorted(restarts.items())})
    chosen = max(scores, key=scores.get)
    if chosen == 200000:
        curve = baseline
    else:
        restart = restarts[chosen]
        before = baseline.steps < chosen
        after = restart.steps > chosen
        curve = ScalarCurve(
            np.concatenate((baseline.steps[before], [chosen], restart.steps[after])),
            np.concatenate((baseline.values[before],
                            [np.interp(chosen, baseline.steps, baseline.values)],
                            restart.values[after])))
    visible = curve.steps <= endpoint
    return ScalarCurve(curve.steps[visible], curve.values[visible]), chosen, endpoint, scores



def _selection_window(curve: ScalarCurve, start: float, end: float) -> ScalarCurve:
    """Crop to the displayed window, interpolating boundaries without extrapolation."""
    start, end = max(start, curve.steps[0]), min(end, curve.steps[-1])
    if start > end:
        raise ValueError("Selection curve has no data in the displayed window")
    steps = np.unique(np.r_[start, curve.steps[
        (curve.steps > start) & (curve.steps < end)], end])
    return ScalarCurve(steps, np.interp(steps, curve.steps, curve.values))


def plot_run_restart_selection(env: str, groups: dict[str, list[str]],
                                workspace_root: str, output_root: str,
                                comparison_groups: dict[str, list[str]] | None = None,
                                local_results_root: str | None = None) -> str:
    """Compare BAFCv3 with retrospective per-seed restart/no-restart choices."""
    is_dog = env in ("dog", "dog_run")
    output_prefix = "dog_walk" if env == "dog" else env
    if env not in ("dog", "dog_run", "humanoid_run"):
        raise ValueError("Unsupported restart comparison: %s" % env)
    horizon = 200000 if is_dog else 150000
    # Both TR2 and reweight restarts inherit the original BAFCv3 checkpoints.
    base_label = "BAFCv3"
    baseline_dirs = groups["BAFCv3"]
    source_dirs = groups[base_label]
    candidates, baselines, sources = [], [], []

    def read(path):
        curve = _read_run_curve(path, RETURN_TAG)
        return _selection_window(curve, curve.steps[0], horizon)

    for seed in range(4):
        paths = {}
        if env == "dog":
            paths["0k_tr2"] = (0, groups["BAFC_TR"][seed])
            if seed < 2:
                root = os.path.join(local_results_root or os.path.join(
                    workspace_root, "alf_results"), "dog_walk", "bafcv3_tr2_restart",
                    "20260917T054927Z")
            else:
                root = os.path.join(workspace_root, "server2_copy",
                    "dog_walk_bafcv3_tr2_restart", "20260922T162241Z")
            # Temporary: seed 2's 75k continuation is unfinished. Reconsider
            # it when complete; keep the requested 105k/UTD11 choice for now.
            checkpoints = [(105, 11)] if seed == 2 else [(75, 11), (105, 11)]
            if seed < 2:
                checkpoints.append((105, 3))
            for checkpoint, utd in checkpoints:
                paths["%dk_utd%d" % (checkpoint, utd)] = (checkpoint * 1000,
                    os.path.join(root, "dog_walk_s%d_%dk_utd%d_on" %
                                 (seed, checkpoint, utd)))
        elif is_dog:
            for checkpoint in (120, 140, 160):
                paths["%dk" % checkpoint] = (checkpoint * 1000, os.path.join(
                    workspace_root, "server9_copy", "dog_run_bafcv3_tr2_restart",
                    "20260922T021158Z", "dog_run_s%d_%dk_utd11_on" % (seed, checkpoint)))
            paths["auto120k"] = (120000, os.path.join(
                workspace_root, "server3_copy",
                "dog_run_bafcv3_tr2_auto_restart120k_utd11_s%d" % seed))
        else:
            paths["0k_v6"] = (0, groups["BAFCv6"][seed])
            for checkpoint in (90, 105):
                paths["%dk" % checkpoint] = (checkpoint * 1000, os.path.join(
                    workspace_root, "server_copy" if seed < 2 else "server7_copy",
                    "humanoid_run_bafcv6_reweight_restart%dk_utd11_s%d" % (checkpoint, seed)))
        baselines.append(read(baseline_dirs[seed]))
        sources.append(read(source_dirs[seed]))
        candidates.append({name: (step, path, read(path))
                           for name, (step, path) in paths.items()})
    excluded = [{} for _ in range(4)]
    if env == "dog":
        # Compare at the full baseline horizon; unfinished candidates must not
        # truncate every curve or be extrapolated beyond their recorded data.
        endpoint = min(c.steps[-1] for c in baselines)
        for seed, candidate in enumerate(candidates):
            excluded[seed] = {name: float(item[2].steps[-1])
                              for name, item in candidate.items()
                              if item[2].steps[-1] < endpoint}
            candidates[seed] = {name: item for name, item in candidate.items()
                                if name not in excluded[seed]}
            if excluded[seed]:
                print("dog_walk | seed=%d | unfinished candidates omitted: %s" %
                      (seed, excluded[seed]))
        if "105k_utd11" not in candidates[2]:
            raise ValueError("Pinned Dog Walk seed 2 105k run has not reached the comparison step")
    else:
        # Other tasks retain the shared endpoint across all candidates.
        endpoint = min(c.steps[-1] for c in baselines + sources + [
            item[2] for candidate in candidates for item in candidate.values()])
    if endpoint < max([c.steps[0] for c in baselines] +
                      [c.steps[0] for candidate in candidates
                       for _, _, c in candidate.values()]):
        raise ValueError("Restart candidates have no common comparison step")
    selected, selections = [], []
    for seed, (source, candidate) in enumerate(zip(sources, candidates)):
        scores = {base_label: float(np.interp(endpoint, source.steps, source.values))}
        scores.update({name: float(np.interp(endpoint, c.steps, c.values))
                       for name, (_, _, c) in candidate.items()})
        winner = ("105k_utd11" if env == "dog" and seed == 2 else
                  max(scores, key=scores.get))  # No restart wins ties.
        checkpoint = None
        chosen = source
        if winner != base_label:
            checkpoint, _, restart = candidate[winner]
            if checkpoint == 0:
                chosen = restart
            else:
                before, after = source.steps < checkpoint, restart.steps > checkpoint
                chosen = ScalarCurve(
                    np.r_[source.steps[before], checkpoint, restart.steps[after]],
                    np.r_[source.values[before], np.interp(checkpoint, source.steps,
                           source.values), restart.values[after]])
        selected.append(chosen)
        selections.append(dict(seed=seed, selected=winner,
            selected_checkpoint=checkpoint, selection_step=float(endpoint),
            candidate_returns=scores, baseline=baseline_dirs[seed],
            forced_selection=(env == "dog" and seed == 2),
            excluded_incomplete_candidates=excluded[seed],
            continuation_source=source_dirs[seed],
            restarts={name: path for name, (_, path, _) in candidate.items()}))
        print("%s selection | seed=%d | selected=%s | return=%.6g" %
              (env, seed, winner, scores[winner]))
    enabled_steps = sorted({s["selected_checkpoint"] for s in selections
                            if s["selected_checkpoint"] is not None})
    start = max(0, min(enabled_steps) - 30000) if enabled_steps else 0
    if env == "dog":
        start = 70000
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, output_prefix + "_trust_selection.json"), "w") as f:
        json.dump(dict(selection_rule="retrospective per-seed highest return at "
                       "shared final step; no restart wins ties; Dog Walk seed 2 "
                       "temporarily pinned to 105k/UTD11",
                       display_window=[start, horizon], enabled_steps=enabled_steps,
                       seeds=selections), f, indent=2)
        f.write("\n")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    # Reference algorithms are displayed only; they do not enter checkpoint selection.
    for label, runs in (comparison_groups or {}).items():
        aggregate = aggregate_curves(
            [_selection_window(read(run), start, horizon) for run in runs],
            runs, RETURN_TAG)
        _plot_aggregate(ax, aggregate, label)
        _print_summary(env, label, RETURN_TAG, len(runs), aggregate)
    for label, curves in (("Ours", baselines),
                          ("Ours_trust" if is_dog else "Ours_reweight", selected)):
        aggregate = aggregate_curves(
            [_selection_window(c, start, endpoint) for c in curves],
            baseline_dirs, RETURN_TAG)
        _plot_aggregate(ax, aggregate, label)
        _print_summary(env, label, RETURN_TAG, 4, aggregate)
    for step in ([] if 0 in enabled_steps else enabled_steps):
        ax.axvline(step, color="gray", linestyle="--", alpha=0.4, linewidth=1.2)
    return _finish_plot(fig, ax, "", "Average Episodic Return",
        os.path.join(output_root, output_prefix + (
            "_trust_average_return_vs_env_steps.png" if is_dog else
            "_reweight_selection_average_return_vs_env_steps.png")),
        xlabel="Environment Steps", human_readable_x_ticks=True,
        xlim=(start, horizon))


def plot_dog_trot_trust(baseline_dirs: list[str], workspace_root: str,
                        local_results_root: str, output_root: str) -> str:
    """Compare BAFCv3 with retrospective per-seed TR2 checkpoint selection."""
    if len(baseline_dirs) != 4:
        raise ValueError("Dog Trot trust comparison requires seeds 0--3")
    baselines, selected, selections = [], [], []

    def read(run_dir):
        curve = _read_scalar_curve(os.path.join(run_dir, "train"), RETURN_TAG)
        visible = curve.steps <= 200000
        if not np.any(visible):
            raise ValueError("No returns at or before 200k: %s" % run_dir)
        return ScalarCurve(curve.steps[visible], curve.values[visible])

    for seed, baseline_dir in enumerate(baseline_dirs):
        baseline = read(baseline_dir)
        paths = {}
        for checkpoint in (120, 140, 160):
            if seed < 2:
                path = os.path.join(
                    local_results_root, "dog_trot", "bafcv3_tr2_restart",
                    "20260920T014756Z",
                    "dog_trot_s%d_%dk_utd11_on" % (seed, checkpoint))
            else:
                path = os.path.join(
                    workspace_root, "server7_copy",
                    "dog_trot_bafcv3_tr2_s%d_%dk_utd11_on" % (seed, checkpoint))
            paths[checkpoint * 1000] = path
        curve, chosen, endpoint, scores = select_trust_restart(
            baseline, {step: read(path) for step, path in paths.items()})
        baselines.append(baseline)
        selected.append(curve)
        selections.append(dict(seed=seed, selected_checkpoint=chosen,
                               selection_step=endpoint, candidate_returns=scores,
                               baseline=baseline_dir, restarts=paths))
        print("dog_trot trust | seed=%d | selected=%dk | final return=%.6g" %
              (seed, chosen // 1000, scores[chosen]))

    enabled_steps = sorted({s["selected_checkpoint"] for s in selections
                            if s["selected_checkpoint"] != 200000})
    start = max(0, min(enabled_steps) - 30000) if enabled_steps else 0
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "dog_trot_trust_selection.json"), "w") as f:
        json.dump(dict(selection_rule="per-seed highest return at shared final step; "
                       "BAFCv3 wins ties", display_window=[start, 200000],
                       enabled_steps=enabled_steps, seeds=selections), f, indent=2)
        f.write("\n")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for label, curves in (("Ours", baselines), ("Ours_trust", selected)):
        aggregate = aggregate_curves(
            [_selection_window(curve, start, 200000) for curve in curves],
            baseline_dirs, RETURN_TAG)
        _plot_aggregate(ax, aggregate, label)
        _print_summary("dog_trot", label, RETURN_TAG, 4, aggregate)
    for step in ([] if 0 in enabled_steps else enabled_steps):
        ax.axvline(step, color="gray", linestyle="--", alpha=0.4, linewidth=1.2)
    return _finish_plot(
        fig, ax, "", "Average Episodic Return",
        os.path.join(output_root, "dog_trot_trust_average_return_vs_env_steps.png"),
        xlabel="Environment Steps", human_readable_x_ticks=True,
        xlim=(start, 200000))


def build_dog_walk_tr2_resume_groups(local_results_root: str):
    """Share run selection and colors between return and skipping plots."""
    groups, colors = {}, {}
    study_root = os.path.join(local_results_root, "dog_walk",
                              "bafcv3_tr2_restart", "20260917T054927Z")
    for horizon, utd, color in ((75, 11, "tab:green"),
                                 (105, 3, "tab:red"),
                                 (105, 11, "tab:brown")):
        label = "TR2: %dk, UTD %d, skipping on" % (horizon, utd)
        groups[label] = [
            os.path.join(study_root, "dog_walk_s%d_%dk_utd%d_on" %
                         (seed, horizon, utd)) for seed in (0, 1)
        ]
        colors[label] = color
    return groups, colors


def rollout_skip_frequency(env_steps: ScalarCurve, skipped: ScalarCurve,
                           opportunities: ScalarCurve) -> ScalarCurve:
    """Local skip percentage at interval midpoints on the environment axis.

    Match counters to recorded environment steps by summary step. Differences
    exclude the unknown interval before the first common sample. Repeated
    environment coordinates (no rollout) are collapsed to their latest counts.
    """
    common = np.intersect1d(env_steps.steps,
                           np.intersect1d(skipped.steps, opportunities.steps))
    if len(common) < 2:
        raise ValueError("Need at least two matched rollout-counter samples")
    env = env_steps.values[np.searchsorted(env_steps.steps, common)]
    skip = skipped.values[np.searchsorted(skipped.steps, common)]
    total = opportunities.values[np.searchsorted(opportunities.steps, common)]
    if np.any(np.diff(env) < 0):
        raise ValueError("Environment steps decreased in rollout counters")
    keep = np.r_[np.diff(env) > 0, True]
    env, skip, total = env[keep], skip[keep], total[keep]
    delta_skip, delta_total = np.diff(skip), np.diff(total)
    if (np.any(delta_skip < 0) or np.any(delta_total < delta_skip)):
        raise ValueError("Invalid or reset rollout counters")
    valid = delta_total > 0
    if not np.any(valid):
        raise ValueError("No rollout opportunities between logged samples")
    return ScalarCurve(steps=((env[:-1] + env[1:]) / 2)[valid],
                       values=100 * delta_skip[valid] / delta_total[valid])



def plot_dog_walk_selected_skip_frequency(output_root: str) -> str:
    """Mean local rollout skip rate for the same four selected Dog Walk runs."""
    with open(os.path.join(output_root, "dog_walk_trust_selection.json")) as f:
        selection = json.load(f)
    start, horizon = selection["display_window"]
    curves, paths = [], []
    for entry in selection["seeds"]:
        checkpoint = entry["selected_checkpoint"]
        if checkpoint is None:
            curves.append(ScalarCurve(np.array([start, horizon], dtype=float),
                                      np.zeros(2)))
            paths.append(entry["baseline"])
            continue
        run = entry["restarts"][entry["selected"]]
        paths.append(run)
        curve = rollout_skip_frequency(*[
            _read_scalar_curve(os.path.join(run, "train"), tag) for tag in
            (ENV_STEPS_TAG, SKIP_COUNT_TAG, ROLLOUT_COUNT_TAG)])
        # No TR2 skipping occurs before enablement. Keep the logged local
        # rates after enablement, rather than cumulative skip fractions.
        if checkpoint > start:
            curve = ScalarCurve(np.r_[start, checkpoint, curve.steps],
                                np.r_[0., 0., curve.values])
        curves.append(_selection_window(curve, start, horizon))
    aggregate = aggregate_curves(curves, paths, "Rollout skip frequency (%)")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.set_ylim(0, 100)
    _plot_aggregate(ax, aggregate, "Ours_trust")
    for step in selection["enabled_steps"]:
        if step > 0:
            ax.axvline(step, color="gray", linestyle="--", alpha=0.4, linewidth=1.2)
    _print_summary("dog", "Ours_trust", "Rollout skip frequency (%)", 4, aggregate)
    return _finish_plot(fig, ax, "", "Rollout opportunities skipped (%)",
        os.path.join(output_root, "dog_walk_trust_skip_frequency_vs_env_steps.png"),
        xlabel="Environment Steps", human_readable_x_ticks=True,
        xlim=(start, horizon))


def plot_dog_walk_tr2_skip_frequency(local_results_root: str,
                                    output_root: str) -> str:
    """Plot interval skip frequency, averaged over seeds 0--1."""
    groups, colors = build_dog_walk_tr2_resume_groups(local_results_root)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.set_ylim(0, 100)
    for label, runs in groups.items():
        curves = []
        for run in runs:
            logdir = os.path.join(run, "train")
            curves.append(rollout_skip_frequency(*[
                _read_scalar_curve(logdir, tag) for tag in
                (ENV_STEPS_TAG, SKIP_COUNT_TAG, ROLLOUT_COUNT_TAG)]))
        aggregate = aggregate_curves(curves, runs, "Rollout skip frequency (%)")
        _plot_aggregate(ax, aggregate, label, colors[label])
        _print_summary("dog", label, "Rollout skip frequency (%)", 2, aggregate)
    return _finish_plot(
        fig, ax, "Dog Walk TR2 Rollout Skipping (Seeds 0–1)",
        "Skipped rollout opportunities per logged interval (%)",
        os.path.join(output_root,
                     "dog_walk_tr2_resume_seed01_skip_frequency_vs_env_steps.png"),
        xlabel="Environment Steps", human_readable_x_ticks=True)


def plot_dog_walk_tr2_resume(
        focused_groups: dict[str, list[str]], local_results_root: str,
        output_root: str) -> str:
    """Compare seeds 0--1, retaining absolute steps for TR2 continuations."""
    groups = {"SAC+" if label == "RLPD" else label: runs[:2]
              for label, runs in focused_groups.items()}
    colors = dict(FOCUSED_ALGORITHM_COLORS)
    resume_groups, resume_colors = build_dog_walk_tr2_resume_groups(local_results_root)
    groups.update(resume_groups)
    colors.update(resume_colors)
    return plot_average_return(
        "dog", groups, output_root,
        title="Dog Walk TR2 Resume (Seeds 0–1)",
        xlabel="Environment Steps", ylabel="Average Episodic Return",
        colors=colors, human_readable_x_ticks=True,
        filename="dog_walk_tr2_resume_seed01_average_return_vs_env_steps.png")


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
    parser.add_argument(
        "--task", "--tasks", dest="tasks", nargs="+", action="extend",
        choices=PLOT_TASKS + ("dog_walk_tr2_resume", "dog_trot_trust", "dog_run_trust", "dog_walk_trust",
                             "humanoid_run_trust"),
        help=("Generate only the named task(s); may be repeated. "
              "Defaults to every task."))
    parser.add_argument("--workspace-root", default="/workspace")
    parser.add_argument("--local-results-root", default=None,
                        help="Output base directory; defaults to <workspace-root>/alf_results. "
                             "Server 3 inputs use <workspace-root>/server3_copy.")
    parser.add_argument("--server-copy-root", default=None,
                        help="Defaults to <workspace-root>/server_copy.")
    parser.add_argument("--server2-copy-root", default=None,
                        help="Defaults to <workspace-root>/server2_copy.")
    parser.add_argument("--server4-copy-root", default=None,
                        help="Defaults to <workspace-root>/server4_copy.")
    parser.add_argument("--server5-copy-root", default=None,
                        help="Defaults to <workspace-root>/server5_copy.")
    parser.add_argument("--server6-copy-root", default=None,
                        help="Defaults to <workspace-root>/server6_copy.")
    parser.add_argument("--output-root", default=None,
                        help=("Defaults to <local-results-root>/"
                              "plots_dog_humanoid_bafc_comparison."))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_tasks = set(args.tasks or PLOT_TASKS)
    print("plot tasks: %s" % ", ".join(
        task for task in PLOT_TASKS + ("dog_walk_tr2_resume", "dog_trot_trust", "dog_run_trust", "dog_walk_trust",
                             "humanoid_run_trust")
        if task in selected_tasks))

    local_results_root = args.local_results_root or os.path.join(
        args.workspace_root, "alf_results")
    server_copy_root = args.server_copy_root or os.path.join(
        args.workspace_root, "server_copy")
    server2_copy_root = args.server2_copy_root or os.path.join(
        args.workspace_root, "server2_copy")
    server4_copy_root = args.server4_copy_root or os.path.join(
        args.workspace_root, "server4_copy")
    output_root = args.output_root or os.path.join(
        local_results_root, "plots_dog_humanoid_bafc_comparison")
    groups = build_run_groups(
        args.workspace_root, local_results_root, server_copy_root,
        server2_copy_root, server4_copy_root)
    rlpd_ours_groups = build_rlpd_ours_run_groups(
        args.workspace_root, local_results_root, server_copy_root,
        server2_copy_root, server4_copy_root)
    additional_groups = build_additional_run_groups(
        args.workspace_root, local_results_root, server_copy_root,
        server2_copy_root, server4_copy_root)

    baseline_groups = build_baseline_run_groups(
        args.workspace_root, rlpd_ours_groups,
        [server_copy_root, server2_copy_root, server4_copy_root,
         args.server5_copy_root or os.path.join(args.workspace_root, "server5_copy"),
         args.server6_copy_root or os.path.join(args.workspace_root, "server6_copy")],
        tasks=[task for task in PLOT_TASKS if task in selected_tasks])
    for env, baseline in baseline_groups.items():
        seed_count = max(map(len, baseline.values()))
        seed_suffix = "".join(map(str, range(seed_count)))
        plot_average_return(
            env, baseline, output_root,
            title="%s (Seeds 0-%d)" % ({
                "dog": "Dog Walk", "humanoid": "Humanoid Walk"
            }.get(env, env.replace("_", " ").title()),
                seed_count - 1),
            xlabel="Environment Steps", ylabel="Average Episodic Return",
            colors=BASELINE_ALGORITHM_COLORS, human_readable_x_ticks=True,
            filename="%s_sac_td3_sacplus_seed%s_average_return_vs_env_steps.png" % (
                env, seed_suffix))
        plot_ours_vs_baselines(env, rlpd_ours_groups[env], baseline, output_root,
                              workspace_root=args.workspace_root)

    for env in ("dog", "dog_fetch", "dog_run", "dog_stand", "dog_trot",
                "humanoid", "humanoid_run", "humanoid_stand"):
        if env in selected_tasks:
            comparison = groups[env]
            if env != "humanoid_stand":
                comparison = build_comparison_group(
                    comparison, rlpd_ours_groups[env]["TD3+"])
            plot_average_return(env, comparison, output_root)
    for env in ("dog", "dog_fetch", "humanoid"):
        if env not in selected_tasks:
            continue
        plot_eval_trust_over_max(env, groups[env]["BAFC_TR"], output_root)
        plot_raw_eval_trust_metric(
            env, groups[env]["BAFC_TR"], INITIAL_EVAL_TRUST_THRESHOLD,
            output_root)
    if "humanoid" in selected_tasks:
        plot_average_return(
            "humanoid_seed01", groups["humanoid_seed01"], output_root,
            title="Humanoid Average Return vs Environment Steps (Seeds 0-1)",
            filename="humanoid_seed01_average_return_vs_env_steps.png")

    for env in ("dog_fetch", "dog_run", "dog_stand", "dog_trot", "dog",
                "humanoid", "humanoid_run", "humanoid_stand", "hopper_hop"):
        if env not in selected_tasks:
            continue
        plot_average_return(
            env, {"SAC+" if label == "RLPD" else label: runs
                  for label, runs in rlpd_ours_groups[env].items()},
            output_root, title="",
            xlabel="Environment Steps", ylabel="Average Episodic Return",
            colors=FOCUSED_ALGORITHM_COLORS, human_readable_x_ticks=True,
            filename="%s_rlpd_vs_ours_average_return_vs_env_steps.png" % env)

    for env in ("dog", "dog_run", "humanoid_run"):
        selection_task = "dog_walk_trust" if env == "dog" else env + "_trust"
        if {env, selection_task}.intersection(selected_tasks):
            references = None
            if env == "humanoid_run":
                references = {"SAC+": rlpd_ours_groups[env]["RLPD"],
                              "TD3+": rlpd_ours_groups[env]["TD3+"]}
            plot_run_restart_selection(env, groups[env], args.workspace_root,
                                       output_root, comparison_groups=references,
                                       local_results_root=local_results_root)
            if env == "dog":
                plot_dog_walk_selected_skip_frequency(output_root)

    if {"dog_trot", "dog_trot_trust"}.intersection(selected_tasks):
        plot_dog_trot_trust(rlpd_ours_groups["dog_trot"]["Ours"],
                            args.workspace_root, local_results_root, output_root)

    if {"dog", "dog_walk_tr2_resume"}.intersection(selected_tasks):
        plot_dog_walk_tr2_resume(rlpd_ours_groups["dog"], local_results_root,
                                 output_root)
        plot_dog_walk_tr2_skip_frequency(local_results_root, output_root)

    if "hopper_hop" in selected_tasks:
        plot_average_return(
            "hopper_hop_ncritic", additional_groups["hopper_hop_ncritic"],
            output_root,
            title="Hopper Hop BAFC Sampled-Critic Comparison (Seeds 0, 2, 3)",
            filename="hopper_hop_ncritic_average_return_vs_env_steps.png")
        plot_average_return(
            "hopper_hop_seed0_v7",
            additional_groups["hopper_hop_seed0_v7"], output_root,
            title="Hopper Hop Seed 0 BAFCv7 Comparison",
            filename="hopper_hop_seed0_bafcv7_average_return_vs_env_steps.png")

    if "humanoid" in selected_tasks:
        plot_average_return(
            "humanoid_reweight", additional_groups["humanoid_reweight"],
            output_root,
            title="Humanoid Walk SAC+ vs BAFCv3 TR2 Reweight (Seeds 0-3)",
            filename=("humanoid_rlpd_vs_bafcv3_tr2_reweight_"
                      "average_return_vs_env_steps.png"))
        plot_average_return(
            "humanoid_reweight_individual",
            additional_groups["humanoid_reweight_individual"], output_root,
            title=("Humanoid Walk SAC+ vs BAFCv3 TR2 Reweight: "
                   "Individual Runs"),
            filename=("humanoid_rlpd_vs_bafcv3_tr2_reweight_individual_"
                      "average_return_vs_env_steps.png"))


if __name__ == "__main__":
    main()
