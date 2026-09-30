"""Compare humanoid:run actor-ID control A with the standard BAFCv3 runs.

    .venv/bin/python alf/utils/plot_humanoid_run_control_a.py

Uses unsmoothed TensorBoard AverageReturn versus environment steps. All eight
curves are aligned on their common observed range, capped at 150k steps.
"""
import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from plot_dog_humanoid_bafc_comparison import _read_scalar_curve, RETURN_TAG


def settings(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'pre_config':
                return ast.literal_eval(node.args[0])
    raise ValueError(f'No saved pre_config: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, default=Path('/workspace'))
    parser.add_argument('--control-root', type=Path,
                        help='Study directory containing seed_0 through seed_3')
    parser.add_argument('--output', type=Path, default=Path('artifacts/humanoid_run_control_a'))
    args = parser.parse_args()
    root = args.control_root
    if root is None:
        candidates = sorted((args.workspace_root / 'server9_copy/humanoid_run_bafcv3_actor_id').glob(
            '*/fixed_pairingFalse_num_sampled_critic8/critic_utd11'))
        if not candidates:
            raise ValueError('No Control A study found')
        root = candidates[-1]
    groups = {
        'BAFCv3': [args.workspace_root / f'server3_copy/humanoid_run_bafcv3_rtT_s{s}' for s in range(4)],
        'Control A (actor ID)': [root / f'seed_{s}' for s in range(4)]}
    curves, configs = {}, {}
    for label, runs in groups.items():
        curves[label], configs[label] = [], []
        for seed, run in enumerate(runs):
            conf = settings(run / 'alf_config.py')
            assert conf['create_environment.env_name'] == 'humanoid:run'
            assert conf['TrainerConfig.random_seed'] == seed
            assert conf.get('bafcv3_use_actor_id_encoding', False) == (label != 'BAFCv3')
            curves[label].append(_read_scalar_curve(str(run / 'train'), RETURN_TAG))
            configs[label].append(conf)
    ignored = {'bafcv3_use_actor_id_encoding', 'bafcv3_detach_actor_policy_input'}
    for seed in range(4):
        a, b = [configs[label][seed] for label in groups]
        if {k:v for k,v in a.items() if k not in ignored} != {k:v for k,v in b.items() if k not in ignored}:
            raise ValueError(f'Unexpected pre_config differences for seed {seed}')
    all_curves = [c for cc in curves.values() for c in cc]
    start = max(c.steps[0] for c in all_curves)
    end = min(150000, *(c.steps[-1] for c in all_curves))
    if end <= start:
        raise ValueError('No shared nonempty range')
    tail_start = max(start, end - 10000)
    grid = np.unique(np.concatenate(([start, end, tail_start], *[
        c.steps[(c.steps >= start) & (c.steps <= end)] for c in all_curves])))
    aligned = {label: np.stack([np.interp(grid, c.steps, c.values) for c in cc])
               for label, cc in curves.items()}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records = []
    colors = {'BAFCv3': 'tab:blue', 'Control A (actor ID)': 'tab:orange'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, values in aligned.items():
        mean, sd = values.mean(axis=0), values.std(axis=0)
        axes[0].plot(grid, mean, label=label, color=colors[label])
        axes[0].fill_between(grid, mean-sd, mean+sd, alpha=.18, color=colors[label])
        for seed, y in enumerate(values):
            tail = grid >= tail_start
            records.append(dict(algorithm=label, seed=seed, shared_start=float(start), shared_end=float(end),
                final_return=float(y[-1]),
                last_10k_mean=float(np.trapz(y[tail], grid[tail]) / (end-tail_start)),
                mean_return_over_training=float(np.trapz(y, grid) / (end-start))))
    delta = aligned['Control A (actor ID)'] - aligned['BAFCv3']
    for seed in range(4):
        axes[1].plot(grid, delta[seed], alpha=.65, linewidth=1, label=f'Seed {seed}')
    axes[1].plot(grid, delta.mean(axis=0), color='black', linewidth=2, label='Mean')
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[0].set(ylabel='AverageReturn', title='humanoid:run — mean ± population SD (4 seeds)')
    axes[1].set(ylabel='Control A − BAFCv3 return', title='Matched-seed differences')
    for ax in axes:
        ax.set(xlabel='Environment steps', xlim=(0,150000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(output / f'humanoid_run_control_a_vs_bafcv3.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for seed, ax in enumerate(axes.flat):
        for label, values in aligned.items():
            ax.plot(grid, values[seed], color=colors[label], label=label)
        ax.set(title=f'Seed {seed}', xlabel='Environment steps', ylabel='AverageReturn',xlim=(0,150000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(output / f'per_seed.{ext}', dpi=180)
    plt.close(fig)
    with (output / 'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    source = {label: [dict(run=str(run), first_step=float(c.steps[0]),last_step=float(c.steps[-1]),
        samples=len(c.steps), config=conf,
        event_files=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                     for p in sorted((run/'train').glob('events*'))])
        for run,c,conf in zip(groups[label], curves[label], configs[label])] for label in groups}
    (output/'manifest.json').write_text(json.dumps(dict(control_study=str(root),sources=source,
        tag=RETURN_TAG,shared_start=float(start),shared_end=float(end)),indent=2))
    (output/'curves.json').write_text(json.dumps({label:[dict(seed=s,steps=c.steps.tolist(),values=c.values.tolist())
        for s,c in enumerate(cc)] for label,cc in curves.items()}))
    report=['# Humanoid run: Control A versus BAFCv3','',
        'Control A replaces the functional encoder with learned actor-ID embeddings. Embeddings learn from critic loss; the policy-encoding actor-gradient path is absent. This changes more than representation separation alone.',
        f'Four seeds per arm, shared observed range {start:,.0f}–{end:,.0f} environment steps. Unsmoothed curves, linear interpolation, no extrapolation. Bands are population SD across seeds, not confidence intervals.',
        'Saved top-level settings match after excluding the two ablation flags. The saved config adds support for those flags; historical training-source equivalence has not been established.', '',
        '| Metric | BAFCv3 mean ± SD | Control A mean ± SD | Mean difference |',
        '|---|---:|---:|---:|']
    for key in ('final_return','last_10k_mean','mean_return_over_training'):
        vals=[np.array([r[key] for r in records if r['algorithm']==label]) for label in groups]
        report.append(f'| {key} | {vals[0].mean():.2f} ± {vals[0].std():.2f} | {vals[1].mean():.2f} ± {vals[1].std():.2f} | {vals[1].mean()-vals[0].mean():+.2f} |')
    report+=['','The last-10k metric is a time-weighted average of the logged return curve, not a new episode evaluation. Mean return over training is normalized trapezoidal AUC on the shared range. These are descriptive four-seed results; they do not establish statistical equivalence or a causal collapse mechanism.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
