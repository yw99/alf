"""Offline BAFCv3 actor separation over saved environment steps.

Run directly (no environments or replay buffers are loaded)::

    .venv/bin/python alf/utils/plot_actor_encoding_distances.py

Defaults match the dog:stand/dog:run BAFCv3 runs in the comparison plot.
Each checkpoint uses its own saved evaluation samples. Distances are
within a checkpoint, never between encodings in changing latent coordinate
systems. Action differences on these synthetic probes are descriptive, not a
measurement of visitation-weighted behavior or causal representation utility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def pairwise_metrics(encodings, actions):
    """Return symmetric matrices; undefined zero-vector cosine is NaN.

    actions has shape [probe, actor, action]. Action distance is RMS over
    probes and action coordinates, retaining the action's original units.
    """
    z = np.asarray(encodings, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    if z.ndim != 2 or a.ndim != 3 or a.shape[1] != len(z):
        raise ValueError('Expected [actor, feature] and [probe, actor, action]')
    if not np.isfinite(z).all() or not np.isfinite(a).all():
        raise ValueError('Nonfinite encodings/actions')
    l2 = np.linalg.norm(z[:, None] - z[None, :], axis=-1)
    norms = np.linalg.norm(z, axis=-1)
    denominator = norms[:, None] * norms[None, :]
    similarity = np.divide(z @ z.T, denominator,
                           out=np.full_like(l2, np.nan), where=denominator > 0)
    cosine = np.clip(1 - similarity, 0, 2)
    action_rms = np.sqrt(np.mean(
        (a[:, :, None, :] - a[:, None, :, :]) ** 2, axis=(0, 3)))
    return {'l2': l2, 'cosine': cosine, 'action_rms': action_rms}


def worker(args):
    sys.path.insert(0, str(REPO))
    import torch
    from alf.bin.evaluate_bafcv3_checkpoints import (
        load_model, tensor_fingerprint, digest)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    checkpoint = Path(args.checkpoint)
    run = checkpoint.parent.parent.parent
    with torch.inference_mode():
        alg, _, metadata = load_model(
            {'config': str(run / 'alf_config.py')}, str(checkpoint), 'cpu',
            allow_frozen_eval_samples=True)
        if alg._use_actor_id_encoding:
            raise ValueError('Actor-ID embeddings are not functional encodings')
        before = tensor_fingerprint(alg)
        z, _ = alg._encode_actor_policies()
        if z.shape[0] != 10:
            raise ValueError(f'Expected 10 actors, got {z.shape[0]}')
        actions = alg._actor_networks(alg._actor_eval_samples)[0]
        matrices = pairwise_metrics(z.cpu().numpy(), actions.cpu().numpy())
        if before != tensor_fingerprint(alg):
            raise RuntimeError('Diagnostic mutated the saved model')
    # NPZ preserves undefined cosine values without invalid JSON numbers.
    np.savez_compressed(
        args.result, encodings=z.cpu().numpy(),
        encoding_norms=z.norm(dim=-1).cpu().numpy(),
        **matrices, metadata=json.dumps({
            'run': str(run), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': digest(checkpoint),
            'config_sha256': {str(p.relative_to(run)): digest(p) for p in
                              [run / 'alf_config.py'] +
                              sorted((run / 'config_files').glob('*.py'))},
            'task': metadata['task'], 'seed': metadata['seed'],
            'env_steps': metadata['env_steps'],
            'global_step': metadata['global_step'],
            'variant': metadata['variant'],
            'actor_eval_type': alg._actor_eval_type,
            'probe_protocol': f'checkpoint_saved_{alg._eval_samples_source}_samples',
            'eval_samples_source': alg._eval_samples_source,
            'eval_samples_sha256': hashlib.sha256(
                alg._actor_eval_samples.detach().cpu().numpy().tobytes()).hexdigest(),
            'state_unchanged': True}))


def default_runs(root):
    for task in ('stand', 'run'):
        for seed in range(4):
            server = 'server2_copy' if task == 'stand' or seed < 2 else 'server_copy'
            yield root / server / f'dog_{task}_bafcv3_rtT_s{seed}'


def summarize(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    groups = {}
    rows = []
    summary = []
    i, j = np.triu_indices(10, k=1)
    keys = ('l2', 'cosine', 'action_rms')
    labels = ('Encoding Euclidean distance', 'Encoding cosine distance',
              'Action RMS difference (saved probes)')
    for path in sorted((output / 'raw').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            item = {k: data[k].copy() for k in keys}
            item['encoding_norms'] = data['encoding_norms'].copy()
        item.update(meta)
        groups.setdefault(meta['run'], []).append(item)
        for index, (a, b) in enumerate(zip(i, j)):
            rows.append(dict(task=meta['task'], seed=meta['seed'],
                             run=meta['run'], checkpoint=meta['checkpoint'],
                             env_steps=meta['env_steps'], actor_i=int(a), actor_j=int(b),
                             **{k: item[k][a, b] for k in keys}))
    if not groups:
        raise ValueError('No successfully analyzed checkpoints')
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    tasks = sorted({r['task'] for rr in groups.values() for r in rr})
    overview, overview_axes = plt.subplots(len(tasks), 3,
        figsize=(15, 4 * len(tasks)), squeeze=False)
    report = ['# Actor encoding separation', '',
              'All 45 unordered pairs of 10 actors; no diagonal/self pairs.',
              'Encoding-distance axes use a symmetric log scale (linear below 1e-14); heatmaps share a first/last observed scale within each run.',
              'Environment steps come from saved trainer progress, not checkpoint filenames.',
              'Curves are unsmoothed. Overview lines are pair means per seed; shaded bands show the pair minimum–maximum, not confidence intervals.',
              'Actors and encoder can change across checkpoints; saved probes follow each run\'s configured trainable/frozen source. Actions are measured on those synthetic probes, not environment rollouts.',
              'Nonzero separation shows distinguishable encodings. It does not prove better returns or that the critic uses the encoding; that requires a controlled intervention.', '',
              '| Task | Seed | Checkpoints | Env-step range | Mean L2 first → last | Mean cosine first → last | Mean action RMS first → last |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for run, rr in sorted(groups.items()):
        rr.sort(key=lambda r: r['env_steps'])
        task, seed = rr[0]['task'], rr[0]['seed']
        x = np.array([r['env_steps'] for r in rr])
        name = f"{task.replace(':', '_')}_s{seed}_{hashlib.sha256(run.encode()).hexdigest()[:6]}"
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for col, (key, label) in enumerate(zip(keys, labels)):
            y = np.stack([r[key][i, j] for r in rr])
            axes[0, col].plot(x, y, alpha=.3, linewidth=.8)
            axes[0, col].plot(x, np.nanmean(y, axis=1), color='black', linewidth=2, label='45-pair mean')
            axes[0, col].set(xlabel='Environment steps', ylabel=label)
            axes[0, col].legend()
            if key != 'action_rms':
                axes[0, col].set_yscale('symlog', linthresh=1e-14)
            ax = overview_axes[tasks.index(task), col]
            line, = ax.plot(x, np.nanmean(y, axis=1), label=f'seed {seed}')
            ax.fill_between(x, np.nanmin(y, axis=1), np.nanmax(y, axis=1),
                            color=line.get_color(), alpha=.10)
            ax.set(title=task, xlabel='Environment steps', ylabel=label)
            ax.legend()
            if key != 'action_rms':
                ax.set_yscale('symlog', linthresh=1e-14)
        # A shared observed scale exposes small first/last differences.
        vmax = max(1e-14, np.nanmax(rr[0]['cosine']),
                   np.nanmax(rr[-1]['cosine']))
        for col, r in enumerate((rr[0], rr[-1])):
            im = axes[1, col].imshow(r['cosine'], vmin=0, vmax=vmax, cmap='viridis')
            axes[1, col].set(title=f"Cosine distance at {r['env_steps']:,} steps",
                             xlabel='Actor ID', ylabel='Actor ID', xticks=range(10), yticks=range(10))
            fig.colorbar(im, ax=axes[1, col])
        axes[1, 2].plot(x, np.stack([r['encoding_norms'] for r in rr]))
        axes[1, 2].set(xlabel='Environment steps', ylabel='Encoding norm (each actor)')
        for ax in [*axes[0], axes[1, 2]]:
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k'))
        fig.suptitle(f'{task}, seed {seed}')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(figures / f'{name}.{ext}', dpi=160)
        plt.close(fig)
        ends = [f"{np.nanmean(rr[0][k][i,j]):.4g} → {np.nanmean(rr[-1][k][i,j]):.4g}" for k in keys]
        report.append(f"| {task} | {seed} | {len(rr)} | {x[0]:,}–{x[-1]:,} | " + ' | '.join(ends) + ' |')
        for r in rr:
            record = {k: r[k] for k in ('task', 'seed', 'run', 'checkpoint', 'env_steps')}
            for k in keys:
                values = r[k][i, j]
                record.update({f'{k}_mean': np.nanmean(values),
                               f'{k}_min': np.nanmin(values),
                               f'{k}_max': np.nanmax(values)})
            summary.append(record)
    for ax in overview_axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k'))
    overview.tight_layout()
    for ext in ('png', 'pdf'):
        overview.savefig(figures / f'overview.{ext}', dpi=180)
    plt.close(overview)
    for filename, records in [('pairs.csv', rows), ('summary.csv', summary)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output / 'report.md').write_text('\n'.join(report) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, default=Path('/workspace'))
    parser.add_argument('--run', type=Path, action='append', help='Explicit run directory (repeatable)')
    parser.add_argument('--output', type=Path, default=REPO / 'artifacts/actor_encoding_distances')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--checkpoint', help=argparse.SUPPRESS)
    parser.add_argument('--result', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.checkpoint:
        worker(args)
        return
    if args.summarize_only:
        summarize(args.output)
        return
    if args.workers < 1 or args.cpu_threads < 1:
        parser.error('workers and cpu-threads must be positive')
    # Refuse to mix old results and a new study.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'raw').mkdir()
    (args.output / 'logs').mkdir()
    runs = args.run or list(default_runs(args.workspace_root))
    jobs, missing = [], []
    for run in runs:
        cps = sorted((run / 'train/algorithm').glob('ckpt-*'))
        cps = [p for p in cps if re.fullmatch(r'ckpt-\d+', p.name)]
        if not cps:
            missing.append(str(run))
        jobs.extend(cps)
    manifest = {'invocation': sys.argv, 'missing_runs': missing, 'checkpoints': [],
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def launch(cp):
        key = hashlib.sha256(str(cp).encode()).hexdigest()[:16]
        with (args.output / 'logs' / f'{key}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--checkpoint', str(cp), '--result', str(args.output / 'raw' / f'{key}.npz'),
                '--cpu-threads', str(args.cpu_threads)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        print(f"{'OK' if result.returncode == 0 else 'ERROR'} {cp}", flush=True)
        return {'path': str(cp), 'status': 'ok' if result.returncode == 0 else 'error',
                'log': f'logs/{key}.log'}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(launch, jobs):
            manifest['checkpoints'].append(record)
            (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summarize(args.output)
    if missing or any(r['status'] != 'ok' for r in manifest['checkpoints']):
        raise SystemExit('Incomplete study: see manifest.json and logs')


if __name__ == '__main__':
    main()
