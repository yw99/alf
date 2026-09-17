"""Calibrate preliminary, frozen TR2 thresholds from saved rank-max metrics.

This command does not load networks or launch training. Its outputs use the
saved offline measurement protocol, not a resumed TR2 startup calibration.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def quantile_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('threshold quantile must be finite and in [0, 1]')
    return value


def calibrate_threshold(rank_max_values, quantile=0.33):
    """Quantile of per-repetition maxima, not maximum of rank quantiles.

    The empirical fraction can differ from the requested quantile because of
    finite sample counts, interpolation, and ties. It is not a rollout-skip rate.
    """
    quantile = quantile_value(quantile)
    values = np.asarray(rank_max_values, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values < 0)):
        raise ValueError('Expected a nonempty vector of finite, nonnegative rank-max metrics')
    threshold = float(np.quantile(values, quantile, method='linear'))
    return {
        'threshold_quantile': quantile,
        'eval_trust_max': threshold,
        'calibration_repetitions': len(values),
        'empirical_threshold_pass_fraction': float(np.mean(values <= threshold)),
    }


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def select_thresholds(study, quantile=0.33, tasks=(), seeds=(), checkpoints=(),
                      sample_count=128, ridge=1e-4):
    study = Path(study)
    quantile_value(quantile)
    if sample_count < 1 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Sample count and ridge must be positive and finite')
    manifest = json.loads((study / 'manifest.json').read_text())
    jobs = {(r['run'], Path(r['checkpoint']).name): r for r in manifest['jobs']}
    metadata = {}
    for row in read_csv(study / 'checkpoint_means.csv'):
        key = row['run'], row['checkpoint']
        if key in metadata:
            raise ValueError(f'Duplicate checkpoint metadata: {key}')
        metadata[key] = row
    grouped = {}
    for row in read_csv(study / 'cross_rank.csv'):
        if int(row['n']) != sample_count or float(row['ridge']) != ridge:
            continue
        key = row['run'], row['checkpoint']
        if key not in metadata:
            raise ValueError(f'Missing checkpoint metadata: {key}')
        meta = metadata[key]
        if tasks and meta['task'] not in tasks:
            continue
        if seeds and int(meta['seed']) not in seeds:
            continue
        if checkpoints and int(row['checkpoint'].removeprefix('ckpt-')) not in checkpoints:
            continue
        job = jobs.get(key)
        if job is None or job['status'] != 'ok' or job.get('missing_ranks'):
            raise ValueError(f'Complete, successful rank inputs are required: {key}')
        repetition = int(row['repetition'])
        values = grouped.setdefault(key, {})
        if repetition in values:
            raise ValueError(f'Duplicate repetition {repetition}: {key}')
        values[repetition] = float(row['rank_max'])
    if not grouped:
        raise ValueError('No complete cross-rank measurements match the requested filters/settings')
    results = []
    for key, repetitions in sorted(grouped.items()):
        meta = metadata[key]
        expected = int(meta['repetitions'])
        if set(repetitions) != set(range(expected)):
            raise ValueError(f'Missing or unexpected repetitions: {key}; expected 0..{expected - 1}')
        if 'num_ranks' in meta and int(meta['num_ranks']) != len(jobs[key]['expected_ranks']):
            raise ValueError(f'Rank count disagrees with manifest: {key}')
        result = {k: meta[k] for k in ('run', 'task', 'seed', 'variant', 'checkpoint', 'env_steps')}
        result.update(seed=int(meta['seed']), env_steps=int(meta['env_steps']),
                      protocol='reconstructed_cache', covariance_sample_count=sample_count,
                      ridge=ridge, rank_reduction='max', rollout_skip_sync_mode='min',
                      enable_eval_trust_max_decay=False, calibration_stage='offline_preliminary')
        result.update(calibrate_threshold([repetitions[i] for i in sorted(repetitions)], quantile))
        results.append(result)
    # Explicitly requested combinations must not disappear silently.
    selected_seeds = seeds or sorted({r['seed'] for r in results})
    selected_tasks = tasks or sorted({r['task'] for r in results})
    if checkpoints:
        for task in selected_tasks:
            for seed in selected_seeds:
                for step in checkpoints:
                    if not any(r['task'] == task and r['seed'] == seed and
                               r['checkpoint'] == f'ckpt-{step}' for r in results):
                        raise ValueError(f'No complete measurements for task={task}, seed={seed}, checkpoint={step}')
    return results


def fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size': path.stat().st_size}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--threshold-quantile', type=quantile_value, default=0.33)
    p.add_argument('--task', nargs='+', default=[])
    p.add_argument('--seed-filter', type=int, nargs='+', default=[])
    p.add_argument('--checkpoint', type=int, nargs='+', default=[],
                   help='Checkpoint filename suffixes, e.g. 75075 105105; not environment steps')
    p.add_argument('--sample-count', type=int, default=128)
    p.add_argument('--ridge', type=float, default=1e-4)
    return p


def main():
    args = parser().parse_args()
    study = args.study.resolve()
    results = select_thresholds(study, args.threshold_quantile, args.task,
                                args.seed_filter, args.checkpoint, args.sample_count, args.ridge)
    now = dt.datetime.now(dt.timezone.utc)
    output = args.output or study / 'calibrations' / (
        now.strftime('%Y%m%dT%H%M%S.%fZ') + '_q' + str(args.threshold_quantile).replace('.', 'p'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / 'thresholds.csv'
    tmp = output / 'thresholds.csv.tmp'
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    os.replace(tmp, csv_path)
    record = {
        'created_utc': now.isoformat(), 'command': sys.argv,
        'calibration_stage': 'offline_preliminary',
        'condition': 'max_r C_r <= eval_trust_max, in addition to existing controller eligibility and skip caps',
        'threshold_policy': 'Quantile of per-repetition rank maxima; freeze each threshold during continuation',
        'quantile_interpolation': 'linear', 'threshold_quantile': args.threshold_quantile,
        'interpretation': 'Initial metric-threshold pass calibration, not a realized rollout-skip fraction or equal critic-accuracy guarantee',
        'limitations': 'Recalibrate with frozen networks through the actual resumed TR2 metric path and actual sampling/cache settings before training. These results use saved offline reconstructed-cache measurements.',
        'numpy_version': np.__version__, 'python_version': sys.version,
        'sources': [fingerprint(study / name) for name in
                    ('manifest.json', 'cross_rank.csv', 'checkpoint_means.csv')],
        'calibrator': fingerprint(__file__), 'thresholds': results,
    }
    tmp = output / 'calibration.json.tmp'
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, output / 'calibration.json')
    print(f'OUTPUT {output}')
    for row in results:
        print(f"{row['task']} seed {row['seed']} {row['checkpoint']}: "
              f"threshold={row['eval_trust_max']:.4f}, "
              f"empirical passes={row['empirical_threshold_pass_fraction']:.1%} "
              f"({row['calibration_repetitions']} repetitions)")


if __name__ == '__main__':
    main()
