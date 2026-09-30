"""Compare fresh BAFCv3 initialization with the earliest saved checkpoint.

Uses saved configs and seeds, fresh actors/encoder/probes, and no environment.
This is a reproducible initialization experiment, not recovery of historical
step-zero weights: trainer RNG consumption/device/code may have differed.

    .venv/bin/python alf/utils/compare_actor_encoding_initialization.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from plot_actor_encoding_distances import default_runs, pairwise_metrics

REPO = Path(__file__).resolve().parents[2]


def worker(run, result):
    sys.path.insert(0, str(REPO))
    import random
    import torch
    import alf
    from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3
    from alf.algorithms.config import TrainerConfig
    from alf.tensor_specs import TensorSpec, BoundedTensorSpec
    from alf.utils.common import parse_conf_file
    from alf.bin.evaluate_bafcv3_checkpoints import digest, tensor_fingerprint

    torch.set_num_threads(2)
    parse_conf_file(str(run / 'alf_config.py'), create_env=False)
    if alf.get_config_value('Agent.rl_algorithm_cls') is not BafcAlgorithmV3:
        raise ValueError('Only BAFCv3 is supported')
    # These selected studies save the first checkpoint at global step 20020.
    checkpoint = run / 'train/algorithm/ckpt-20020'
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = {k.removeprefix('_rl_algorithm.'): v
             for k, v in data['algorithm'].items()
             if k.startswith('_rl_algorithm.')}
    steps = int(data['trainer_progress']['_env_steps'])
    if steps != 20000:
        raise ValueError(f'Expected 20k environment steps, found {steps}')
    observation_spec = TensorSpec((state['_actor_eval_samples'].shape[-1],))
    action_spec = BoundedTensorSpec(
        (state['_actor_networks._action_layer._bias'].shape[-1],),
        minimum=-1., maximum=1.)
    config = TrainerConfig(root_dir=str(result.parent))
    seed = int(config.random_seed)

    def construct():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return BafcAlgorithmV3(observation_spec=observation_spec,
                               action_spec=action_spec, config=config).eval()

    def measure(alg):
        before = tensor_fingerprint(alg)
        with torch.no_grad():
            z, _ = alg._encode_actor_policies()
            actions = alg._actor_networks(alg._actor_eval_samples)[0]
        if z.shape[0] != 10:
            raise ValueError('Expected 10 actors')
        if before != tensor_fingerprint(alg):
            raise RuntimeError('Inference changed model state')
        return dict(encodings=z.numpy(), encoding_norms=z.norm(dim=-1).numpy(),
                    **pairwise_metrics(z.numpy(), actions.numpy()))

    alg = construct()
    if alg._eval_samples_source != 'trainable' or alg._use_actor_id_encoding:
        raise ValueError('Expected functional encoding with trainable probes')
    fresh_hash = tensor_fingerprint(alg)
    initial = measure(alg)
    repeated = construct()
    repeat_metrics = measure(repeated)
    if fresh_hash != tensor_fingerprint(repeated):
        raise RuntimeError('Seeded initialization is not reproducible')
    for key in initial:
        np.testing.assert_array_equal(initial[key], repeat_metrics[key])
    del repeated
    alg.load_state_dict(state, strict=True)
    trained = measure(alg)
    paths = [run / 'alf_config.py', checkpoint]
    paths += sorted((run / 'config_files').glob('*.py'))
    metadata = dict(run=str(run), task=alf.get_config_value('create_environment.env_name'),
        seed=seed, checkpoint=str(checkpoint), checkpoint_env_steps=steps,
        initialization_protocol='seed_reset_immediately_before_direct_BAFCv3_construction_on_CPU',
        historical_initialization_recovered=False, repeated_initialization_identical=True,
        initial_model_sha256=fresh_hash, state_unchanged=True,
        initial_probes='fresh constructor samples; no checkpoint weights loaded',
        checkpoint_probes='checkpoint saved trainable samples',
        torch_version=torch.__version__, input_sha256={str(p): digest(p) for p in paths})
    np.savez_compressed(result, metadata=json.dumps(metadata),
        **{f'initial_{k}': v for k, v in initial.items()},
        **{f'checkpoint_{k}': v for k, v in trained.items()})


def summarize(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = []
    pairs = []
    ix = np.triu_indices(10, 1)
    datasets = []
    for path in sorted((output / 'raw').glob('*.npz')):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d['metadata']))
            arrays = {k: d[k].copy() for k in d.files if k != 'metadata'}
        datasets.append((meta, arrays))
        for stage in ('initial', 'checkpoint'):
            row = dict(task=meta['task'], seed=meta['seed'], stage=stage,
                       env_steps=0 if stage == 'initial' else meta['checkpoint_env_steps'])
            for key in ('l2', 'cosine', 'action_rms'):
                v = arrays[f'{stage}_{key}'][ix]
                row[f'{key}_mean'] = float(v.mean())
                row[f'{key}_min'] = float(v.min())
                row[f'{key}_max'] = float(v.max())
            row['encoding_norm_mean'] = float(arrays[f'{stage}_encoding_norms'].mean())
            records.append(row)
            for a, b in zip(*ix):
                pairs.append(dict(task=meta['task'], seed=meta['seed'], stage=stage,
                    actor_i=int(a), actor_j=int(b), **{k: float(arrays[f'{stage}_{k}'][a, b])
                    for k in ('l2', 'cosine', 'action_rms')}))
    if not datasets:
        raise ValueError('No successful measurements')
    tasks = sorted({m['task'] for m, _ in datasets})
    fig, axes = plt.subplots(len(tasks), 3, figsize=(12, 4 * len(tasks)), squeeze=False)
    for row, task in enumerate(tasks):
        for col, (metric, label) in enumerate(zip(('l2', 'cosine', 'action_rms'),
                ('Encoding Euclidean distance', 'Encoding cosine distance', 'Action RMS difference'))):
            ax = axes[row, col]
            for m, d in sorted(datasets, key=lambda item: item[0]['seed']):
                if m['task'] != task:
                    continue
                y = np.array([d[f'{s}_{metric}'][ix] for s in ('initial', 'checkpoint')])
                line, = ax.plot([0, 1], y.mean(axis=1), 'o--', label=f"seed {m['seed']}")
                ax.fill_between([0, 1], y.min(axis=1), y.max(axis=1),
                                alpha=.1, color=line.get_color())
            ax.set(title=task, ylabel=label, xticks=[0, 1],
                   xticklabels=['Fresh initialization*', 'Saved 20k checkpoint'])
            if metric != 'action_rms':
                ax.set_yscale('log')
            ax.legend()
    fig.suptitle('45 actor pairs: means and pair ranges\n*Seeded reconstruction; dashed connectors do not show collapse timing')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'initialization_vs_20k.{ext}', dpi=180)
    plt.close(fig)
    for filename, rr in [('summary.csv', records), ('pairs.csv', pairs)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rr[0]))
            writer.writeheader()
            writer.writerows(rr)
    report = ['# Fresh initialization versus saved 20k encodings', '',
        'Fresh actors, encoder and evaluation probes are constructed on CPU using each saved run configuration and seed. No trained weights are used in the initialization measurements. Each construction is repeated and checked for identical model state and numerical results.', '',
        '**This is not recovered historical step-zero state.** RNG is reset immediately before direct BAFCv3 construction; the original trainer may consume randomness before construction, use a different device, or have used different source/library versions. Therefore this tests whether the configured initialization itself collapses, rather than claiming exact original initial weights.', '',
        'Both stages use their own evaluation probes. All 45 unordered pairs are retained; plot bands are pair ranges, not confidence intervals. Dashed lines connect observations and do not locate the onset of collapse.', '',
        '| Task | Seed | Mean L2: fresh → 20k | Mean cosine: fresh → 20k |',
        '|---|---:|---:|---:|']
    for m, d in sorted(datasets, key=lambda item: (item[0]['task'], item[0]['seed'])):
        cells = [f"{d[f'initial_{k}'][ix].mean():.5g} → {d[f'checkpoint_{k}'][ix].mean():.5g}" for k in ('l2', 'cosine')]
        report.append(f"| {m['task']} | {m['seed']} | " + ' | '.join(cells) + ' |')
    (output / 'report.md').write_text('\n'.join(report) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'artifacts/actor_encoding_initialization')
    parser.add_argument('--run', type=Path, action='append')
    parser.add_argument('--workspace-root', type=Path, default=Path('/workspace'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.result)
        return
    if args.summarize_only:
        summarize(args.output)
        return
    if args.workers < 1:
        parser.error('--workers must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'raw').mkdir()
    (args.output / 'logs').mkdir()
    runs = args.run or list(default_runs(args.workspace_root))
    def launch(run):
        key = hashlib.sha256(str(run).encode()).hexdigest()[:12]
        with (args.output / 'logs' / f'{key}.log').open('w') as log:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--worker', str(run), '--result', str(args.output / 'raw' / f'{key}.npz')],
                stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        print(f"{'OK' if p.returncode == 0 else 'ERROR'} {run}", flush=True)
        return dict(run=str(run), status='ok' if p.returncode == 0 else 'error', log=f'logs/{key}.log')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(launch, runs))
    sources = [Path(__file__), Path(__file__).with_name('plot_actor_encoding_distances.py'),
               REPO / 'alf/algorithms/bafc_algorithm_v3.py', REPO / 'alf/networks/transformer_networks.py']
    (args.output / 'manifest.json').write_text(json.dumps(dict(results=results,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        invocation=sys.argv), indent=2))
    summarize(args.output)
    if any(r['status'] != 'ok' for r in results):
        raise SystemExit('Some runs failed; see manifest and logs')


if __name__ == '__main__':
    main()
