"""Compare saved dog:stand actor distances with trainable and frozen probes.

Requires outputs from plot_actor_encoding_distances.py for both studies.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def read_study(root, source):
    runs = {}
    for path in sorted((root / 'raw').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta['task'] != 'dog:stand':
                continue
            if meta['probe_protocol'] != f'checkpoint_saved_{source}_samples':
                raise ValueError(f'Unexpected probe source: {path}')
            seed, step = meta['seed'], meta['env_steps']
            if step in runs.setdefault(seed, {}):
                raise ValueError(f'Duplicate seed/step: {path}')
            runs[seed][step] = dict(metadata=meta, file=str(path),
                **{k:data[k].copy() for k in ('l2','cosine','action_rms','encoding_norms')})
    if set(runs) != set(range(4)):
        raise ValueError(f'Expected seeds 0–3 in {root}')
    if source == 'frozen':
        for seed, rr in runs.items():
            if len({r['metadata']['eval_samples_sha256'] for r in rr.values()}) != 1:
                raise ValueError(f'Frozen probes changed across checkpoints for seed {seed}')
    return runs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainable',type=Path,default=Path('artifacts/actor_encoding_distances'))
    parser.add_argument('--frozen',type=Path,default=Path('artifacts/actor_encoding_distances_frozen'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/dog_stand_frozen_vs_trainable'))
    args=parser.parse_args()
    studies={'trainable':read_study(args.trainable,'trainable'),
             'frozen':read_study(args.frozen,'frozen')}
    steps=sorted(set.intersection(*[set(rr) for runs in studies.values() for rr in runs.values()]))
    if not steps:
        raise ValueError('No shared checkpoint environment steps')
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    ix=np.triu_indices(10,1)
    keys=('l2','cosine','action_rms')
    labels=('Encoding Euclidean distance','Encoding cosine distance','Action RMS on saved probes')
    colors={'trainable':'tab:blue','frozen':'tab:orange'}
    rows=[]
    overview, overview_axes=plt.subplots(1,3,figsize=(15,4.5))
    detail, axes=plt.subplots(4,3,figsize=(15,13),sharex=True)
    for mode,runs in studies.items():
        for col,(key,label) in enumerate(zip(keys,labels)):
            means=[]
            for seed,rr in sorted(runs.items()):
                values=np.stack([rr[s][key][ix] for s in steps])
                means.append(values.mean(axis=1))
                ax=axes[seed,col]
                ax.plot(steps,values.mean(axis=1),color=colors[mode],label=mode)
                ax.fill_between(steps,values.min(axis=1),values.max(axis=1),color=colors[mode],alpha=.15)
                ax.set(title=f'Seed {seed}',ylabel=label)
            means=np.array(means)
            overview_axes[col].plot(steps,means.mean(axis=0),color=colors[mode],label=mode)
            overview_axes[col].fill_between(steps,means.min(axis=0),means.max(axis=0),color=colors[mode],alpha=.15)
            overview_axes[col].set(ylabel=label)
        for seed,rr in sorted(runs.items()):
            for step in steps:
                r=rr[step]
                row=dict(mode=mode,seed=seed,env_steps=step,
                         encoding_norm_mean=float(r['encoding_norms'].mean()))
                for k in keys:
                    v=r[k][ix]
                    row.update({f'{k}_mean':float(v.mean()),f'{k}_min':float(v.min()),f'{k}_max':float(v.max())})
                rows.append(row)
    for grid in (axes,overview_axes.reshape(1,3)):
        for row in grid:
            for col,ax in enumerate(row):
                if col<2:
                    ax.set_yscale('symlog',linthresh=1e-14)
                ax.set(xlabel='Environment steps',xlim=(0,200000))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
                ax.grid(alpha=.2)
                ax.legend()
    overview.suptitle('dog:stand — mean of 45 pairs, averaged over 4 seeds; bands: range of seed means')
    detail.suptitle('dog:stand — 45-pair means; bands: pair minimum–maximum within each seed')
    for fig,name in ((overview,'overview'),(detail,'per_seed')):
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(out/f'{name}.{ext}',dpi=170)
        plt.close(fig)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(shared_steps=steps, frozen_probes_constant_per_seed=True,
        inputs={mode:[r['file'] for rr in runs.values() for r in rr.values()] for mode,runs in studies.items()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report=['# Dog stand: frozen versus trainable encoder probes','',
        f'Four seeds per condition; {len(steps)} common checkpoints from {steps[0]:,} to {steps[-1]:,} environment steps. All 45 unordered actor pairs per checkpoint. No interpolation between unmatched checkpoints.',
        'Frozen probe tensors have identical SHA-256 hashes across every saved checkpoint within each seed. Saved experiment settings differ only in the probe-source flag; historical executable-source equivalence is not established.',
        'Both conditions use their own saved probes. Action RMS values describe these different synthetic input sets, not matched physical observations or rollout behavior. Representation distances do not establish causal usefulness or improved returns.', '',
        '| Mode | Seed | Mean L2 first → last | Mean cosine first → last | Mean action RMS first → last |',
        '|---|---:|---:|---:|---:|']
    for mode,runs in studies.items():
        for seed,rr in sorted(runs.items()):
            endpoints=[f"{rr[steps[0]][k][ix].mean():.5g} → {rr[steps[-1]][k][ix].mean():.5g}" for k in keys]
            report.append(f'| {mode} | {seed} | '+' | '.join(endpoints)+' |')
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))


if __name__=='__main__':
    main()
