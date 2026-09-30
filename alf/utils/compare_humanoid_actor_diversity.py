"""Compare original BAFCv3 and Control A replay-action/representation diversity.

    .venv/bin/python alf/utils/compare_humanoid_actor_diversity.py

Consumes saved offline diagnostics; never reads training checkpoints or events.
Both arms use their own replay-state distributions and checkpoint normalization.
Functional encodings use saved evaluation probes; Control A uses its ID table.
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


def insert(records, key, value):
    if key in records:
        raise ValueError(f'Duplicate task/seed/step: {key}')
    records[key] = value


def load_original(root):
    records = {}
    for path in sorted((root/'raw').glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']))
            if m['encoding_type']!='functional':
                raise ValueError('Expected original functional encodings')
            insert(records,(m['task'],m['seed'],m['env_steps']),dict(
                replay=d['replay_rms'].copy(),l2=d['encoding_l2'].copy(),cosine=d['encoding_cosine'].copy(),
                norms=d['encoding_norms'].copy(),probe=d['saved_probe_rms'].copy(),sources=[str(path)]))
    return records


def load_control(replay_root,embedding_root):
    records={}
    for path in sorted((replay_root/'raw').glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']))
            insert(records,(m['task'],m['seed'],m['env_steps']),dict(
                replay=d['replay_rms'].copy(),probe=d['saved_probe_rms'].copy(),sources=[str(path)]))
    for path in sorted((embedding_root/'raw').glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']))
            for t,step in enumerate(d['env_steps']):
                key=(m['task'],m['seed'],int(step))
                if key not in records:
                    continue
                if 'l2' in records[key]:
                    raise ValueError(f'Duplicate Control A embedding: {key}')
                records[key].update(l2=d['l2'][t].copy(),cosine=d['cosine'][t].copy(),norms=d['norms'][t].copy())
                records[key]['sources'].append(str(path))
    if any('l2' not in r for r in records.values()):
        raise ValueError('Missing Control A embedding measurements')
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,default=Path('artifacts/humanoid_run_bafcv3_replay_and_encodings'))
    parser.add_argument('--control-replay',type=Path,default=Path('artifacts/control_a_replay_action_rms'))
    parser.add_argument('--control-embeddings',type=Path,default=Path('artifacts/control_a_embedding_distances'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/humanoid_run_actor_diversity_comparison'))
    args=parser.parse_args()
    arms={'BAFCv3':load_original(args.original),'Control A':load_control(args.control_replay,args.control_embeddings)}
    if set(arms['BAFCv3'])!=set(arms['Control A']):
        raise ValueError('Checkpoint keys differ; refusing an incomplete matched comparison')
    keys=sorted(arms['BAFCv3'])
    if {k[0] for k in keys}!={'humanoid:run'} or {k[1] for k in keys}!=set(range(4)):
        raise ValueError('Expected humanoid:run seeds 0–3')
    steps=sorted({k[2] for k in keys})
    if len(keys)!=4*len(steps):
        raise ValueError('Unequal checkpoint coverage across seeds')
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    ix=np.triu_indices(10,1)
    colors={'BAFCv3':'tab:blue','Control A':'tab:orange'}
    overview,axes=plt.subplots(2,2,figsize=(12,8))
    detail,detail_axes=plt.subplots(4,3,figsize=(15,12),sharex=True)
    metrics=('replay','l2','cosine','norms')
    labels=('Action RMS on own replay states','Encoding Euclidean distance','Encoding cosine distance','Encoding norm')
    rows,pairs,endpoints=[],[],[]
    for label,records in arms.items():
        for col,(metric,ylabel) in enumerate(zip(metrics,labels)):
            means=[]
            for seed in range(4):
                v=np.stack([records[('humanoid:run',seed,s)][metric] for s in steps])
                if metric in ('l2','cosine'):
                    v=v[:,ix[0],ix[1]]
                means.append(v.mean(axis=1))
                if col<3:
                    ax=detail_axes[seed,col]
                    ax.plot(steps,v.mean(axis=1),color=colors[label],label=label)
                    ax.fill_between(steps,v.min(axis=1),v.max(axis=1),color=colors[label],alpha=.15)
                    ax.set(title=f'Seed {seed}',ylabel=ylabel)
            means=np.array(means)
            ax=axes.flat[col]
            ax.plot(steps,means.mean(axis=0),color=colors[label],label=label)
            ax.fill_between(steps,means.min(axis=0),means.max(axis=0),color=colors[label],alpha=.15)
            ax.set(ylabel=ylabel)
        for task,seed,step in keys:
            r=records[(task,seed,step)]
            row=dict(algorithm=label,task=task,seed=seed,env_steps=step,
                     replay_rms_mean=float(r['replay'].mean()),replay_rms_min=float(r['replay'].min()),
                     replay_rms_max=float(r['replay'].max()),saved_probe_rms_mean=float(r['probe'].mean()),
                     encoding_norm_mean=float(r['norms'].mean()))
            for metric in ('l2','cosine'):
                v=r[metric][ix]
                row.update({f'encoding_{metric}_mean':float(v.mean()),f'encoding_{metric}_min':float(v.min()),f'encoding_{metric}_max':float(v.max())})
            rows.append(row)
            if step in (steps[0],steps[-1]):
                endpoints.append(row)
            for n,(a,b) in enumerate(zip(*ix)):
                pairs.append(dict(algorithm=label,task=task,seed=seed,env_steps=step,
                    actor_i=int(a),actor_j=int(b),replay_rms=float(r['replay'][n]),
                    encoding_l2=float(r['l2'][a,b]),encoding_cosine=float(r['cosine'][a,b])))
    for n,ax in enumerate(axes.flat):
        if n in (1,2,3):
            ax.set_yscale('symlog',linthresh=1e-14)
    for seed_axes in detail_axes:
        for col,ax in enumerate(seed_axes):
            if col>0:
                ax.set_yscale('symlog',linthresh=1e-14)
    for ax in [*axes.flat,*detail_axes.flat]:
        ax.set(xlabel='Environment steps')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
        ax.grid(alpha=.2)
        ax.legend()
    overview.suptitle('humanoid:run — 4-seed mean; bands: range of seed means\nActions evaluated on each arm’s own replay states')
    detail.suptitle('humanoid:run — 45-pair mean; bands: pair minimum–maximum')
    for fig,name in ((overview,'overview'),(detail,'per_seed')):
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(out/f'{name}.{ext}',dpi=170)
        plt.close(fig)
    for name,data in (('summary',rows),('pairs',pairs)):
        with (out/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (out/'endpoints.json').write_text(json.dumps(endpoints,indent=2))
    manifest=dict(steps=steps,seeds=list(range(4)),
        interpretation='Same measurement protocol and checkpoint steps; different replay distributions and saved normalization per arm. Native functional encodings use saved probes, not replay states. Separation is descriptive and does not establish causal performance benefit.',
        inputs={label:sorted({p for r in records.values() for p in r['sources']}) for label,records in arms.items()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps([r for r in endpoints if r['env_steps']==steps[-1]],indent=2))


if __name__=='__main__':
    main()
