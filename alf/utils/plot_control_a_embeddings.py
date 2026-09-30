"""Read learned Control A actor-ID embeddings directly from checkpoints.

    .venv/bin/python alf/utils/plot_control_a_embeddings.py

Discovers saved configs enabling bafcv3_use_actor_id_encoding under server9_copy.
No models/environments are constructed. Distances use all 45 unordered actor
pairs; temporal displacement is measured from the earliest SAVED checkpoint,
not initialization. Raw vectors, matrices, per-pair CSVs and provenance are saved.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch

EMBEDDING_KEY = '_rl_algorithm._actor_id_embedding.weight'


def preconfig(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'pre_config':
                return ast.literal_eval(node.args[0])
    return {}


def geometry(z):
    """Pair matrices and centered participation rank; compute in float64."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] != 10 or not np.isfinite(z).all():
        raise ValueError('Expected ten finite embedding vectors')
    l2 = np.linalg.norm(z[:, None] - z[None, :], axis=-1)
    norms = np.linalg.norm(z, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = np.clip(1 - np.divide(z @ z.T, denominator,
        out=np.full_like(l2, np.nan), where=denominator > 0), 0, 2)
    variance = np.linalg.svd(z - z.mean(axis=0), compute_uv=False) ** 2
    rank = float(variance.sum() ** 2 / (variance ** 2).sum()) if variance.sum() > 0 else 0.
    return l2, cosine, norms, rank


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('/workspace/server9_copy'))
    parser.add_argument('--run', type=Path, action='append', help='Explicit run directory, repeatable')
    parser.add_argument('--output', type=Path, default=Path('artifacts/control_a_embedding_distances'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    runs = args.run or sorted(p.parent for p in args.source_root.rglob('alf_config.py')
        if preconfig(p).get('bafcv3_use_actor_id_encoding', False))
    if not runs:
        raise ValueError('No Control A configs found')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'raw').mkdir()
    (args.output / 'figures').mkdir()
    manifest = dict(embedding_key=EMBEDDING_KEY, source_sha256=digest(Path(__file__)),
        protocol='Direct saved actor-ID table; float64 geometry; temporal reference is first saved checkpoint',
        torch_version=torch.__version__, runs=[])
    summary, pairs = [], []
    endpoints = []
    ix = np.triu_indices(10, 1)
    overview, overview_axes = plt.subplots(2, 3, figsize=(15, 8))
    for run in runs:
        conf = preconfig(run / 'alf_config.py')
        if not conf.get('bafcv3_use_actor_id_encoding', False):
            raise ValueError(f'Not Control A: {run}')
        task, seed = conf['create_environment.env_name'], conf['TrainerConfig.random_seed']
        cps = [p for p in (run / 'train/algorithm').glob('ckpt-*') if re.fullmatch(r'ckpt-\d+', p.name)]
        if not cps:
            raise ValueError(f'No model checkpoints: {run}')
        observations, provenance = [], []
        for cp in sorted(cps, key=lambda p: int(p.name[5:])):
            before = cp.stat()
            data = torch.load(cp, map_location='cpu', weights_only=True)
            z = data['algorithm'][EMBEDDING_KEY].detach().numpy().copy()
            step = int(data['trainer_progress']['_env_steps'])
            observations.append((step, z))
            checksum = digest(cp)
            after = cp.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Checkpoint changed during read: {cp}')
            provenance.append(dict(path=str(cp), env_steps=step, sha256=checksum))
            del data
        observations.sort(key=lambda x: x[0])
        steps = np.array([x[0] for x in observations])
        if len(set(steps)) != len(steps):
            raise ValueError(f'Duplicate environment steps: {run}')
        z = np.stack([x[1] for x in observations])
        l2, cosine, norms, rank = map(np.array, zip(*(geometry(v) for v in z)))
        displacement = np.linalg.norm(z.astype(np.float64) - z[0].astype(np.float64), axis=-1)
        previous = np.concatenate((np.zeros((1, 10)),
            np.linalg.norm(np.diff(z.astype(np.float64), axis=0), axis=-1)))
        relative = displacement.mean(axis=1) / norms[0].mean()
        name = f"{task.replace(':', '_')}_s{seed}_{hashlib.sha256(str(run).encode()).hexdigest()[:6]}"
        meta = dict(run=str(run), task=task, seed=seed, first_saved_env_steps=int(steps[0]),
                    config=conf, config_sha256=digest(run / 'alf_config.py'), checkpoints=provenance,
                    embedding_shape=list(z.shape[1:]))
        manifest['runs'].append(meta)
        np.savez_compressed(args.output / 'raw' / f'{name}.npz', env_steps=steps,
            encodings=z, l2=l2, cosine=cosine, norms=norms, centered_participation_rank=rank,
            displacement_from_first=displacement, displacement_from_previous=previous,
            metadata=json.dumps(meta))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for col, (values, label) in enumerate(((l2[:, ix[0], ix[1]], 'Euclidean distance'),
                                             (cosine[:, ix[0], ix[1]], 'Cosine distance'),
                                             (norms, 'Embedding norm'))):
            axes[0, col].plot(steps, values, alpha=.4, linewidth=.8)
            axes[0, col].plot(steps, values.mean(axis=1), color='black', label='Mean')
            axes[0, col].set(ylabel=label)
            axes[0, col].legend()
            ax = overview_axes[0, col]
            line, = ax.plot(steps, values.mean(axis=1), label=f'{task} seed {seed}')
            ax.fill_between(steps, values.min(axis=1), values.max(axis=1), color=line.get_color(), alpha=.12)
            ax.set(ylabel=label)
        vmax = max(float(cosine[0].max()), float(cosine[-1].max()), 1e-12)
        for col, index in enumerate((0, -1)):
            im = axes[1, col].imshow(cosine[index], vmin=0, vmax=vmax)
            axes[1, col].set(title=f'Cosine at {steps[index]:,} steps', xlabel='Actor ID',
                             ylabel='Actor ID', xticks=range(10), yticks=range(10))
            fig.colorbar(im, ax=axes[1, col])
        axes[1, 2].plot(steps, displacement)
        axes[1, 2].set(ylabel='L2 displacement from first saved checkpoint')
        label = f'{task} seed {seed}'
        overview_axes[1, 0].plot(steps, displacement.mean(axis=1), label=label)
        overview_axes[1, 1].plot(steps, rank, label=label)
        overview_axes[1, 2].plot(steps, relative, label=label)
        for ax in [*axes[0], axes[1, 2]]:
            ax.set(xlabel='Environment steps')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
        fig.suptitle(f'Control A learned embeddings: {task}, seed {seed}')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(args.output / 'figures' / f'{name}.{ext}', dpi=160)
        plt.close(fig)
        for t, step in enumerate(steps):
            row = dict(task=task, seed=seed, run=str(run), env_steps=int(step),
                norm_mean=float(norms[t].mean()), norm_min=float(norms[t].min()), norm_max=float(norms[t].max()),
                centered_participation_rank=float(rank[t]),
                displacement_from_first_mean=float(displacement[t].mean()),
                displacement_from_previous_mean=float(previous[t].mean()),
                relative_displacement_from_first=float(relative[t]))
            for key, matrices in (('l2', l2), ('cosine', cosine)):
                values = matrices[t][ix]
                row.update({f'{key}_mean':float(values.mean()),f'{key}_min':float(values.min()),f'{key}_max':float(values.max())})
            summary.append(row)
            for a, b in zip(*ix):
                pairs.append(dict(task=task,seed=seed,run=str(run),env_steps=int(step),actor_i=int(a),actor_j=int(b),
                    l2=float(l2[t,a,b]),cosine=float(cosine[t,a,b])))
        endpoints.append(dict(run=str(run), first=summary[-len(steps)], last=summary[-1]))
        print(f'OK {task} seed {seed}: {len(steps)} checkpoints, L2 {l2[0][ix].mean():.5g} -> {l2[-1][ix].mean():.5g}', flush=True)
    overview_axes[1, 0].set(ylabel='Mean L2 displacement from first saved checkpoint')
    overview_axes[1, 1].set(ylabel='Centered participation rank (maximum 9)')
    overview_axes[1, 2].set(ylabel='Mean displacement / initial saved mean norm')
    for ax in overview_axes.flat:
        ax.set(xlabel='Environment steps')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    overview.suptitle('Control A actor-ID embeddings; top-row bands: range across actors/pairs')
    overview.tight_layout()
    for ext in ('png','pdf'):
        overview.savefig(args.output / 'figures' / f'overview.{ext}', dpi=170)
    plt.close(overview)
    write_csv(args.output / 'summary.csv', summary)
    write_csv(args.output / 'pairs.csv', pairs)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'endpoints.json').write_text(json.dumps(endpoints, indent=2))


if __name__ == '__main__':
    main()
