"""Offline BAFCv3 actor action and representation differences on visited replay states.

    .venv/bin/python alf/utils/plot_replay_actor_action_rms.py

Samples up to 4096 nonterminal observations without replacement per replay
rank at each checkpoint. All actors see exactly the same sampled states,
normalized with the saved checkpoint statistics. Rank mean squared differences
are weighted by eligible replay population, then square-rooted. No exploration
noise, rollout, training, or normalizer updates are performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def sample_replay(state, count, rng):
    """Uniform sample over populated FIRST/MID slots across all environments."""
    prefix = '_replay_buffer.'
    obs = state[prefix + 'time_step|observation']
    types = state[prefix + 'time_step|step_type']
    sizes = state[prefix + '_current_size']
    positions = state[prefix + '_current_pos']
    capacity = obs.shape[1]
    eligible = []
    for env in range(len(sizes)):
        size = int(sizes[env])
        if not 0 <= size <= capacity:
            raise ValueError('Invalid ring buffer size')
        slots = (torch.arange(size) + int(positions[env]) - size) % capacity
        valid_types = types[env, slots]
        if not torch.isin(valid_types, torch.tensor([0, 1, 2])).all():
            raise ValueError('Invalid populated replay step type')
        slots = slots[valid_types != 2]
        eligible.extend((env, int(slot)) for slot in slots)
    if not eligible:
        raise ValueError('No nonterminal replay states')
    population = len(eligible)
    indices = np.asarray(eligible, dtype=np.int64)[rng.choice(population, min(count, population), replace=False)]
    return obs[indices[:, 0], indices[:, 1]].clone(), population, indices


def action_pair_mse(actions):
    """MSE over states and action coordinates for each unordered actor pair."""
    if actions.ndim != 3 or actions.shape[1] != 10 or not torch.isfinite(actions).all():
        raise ValueError('Expected finite [state, 10 actors, action] tensor')
    i, j = torch.triu_indices(10, 10, offset=1)
    delta = actions.double()[:, i] - actions.double()[:, j]
    return delta.square().mean(dim=(0, 2)).cpu().numpy()


def pooled_rms(rank_mse, populations):
    """Pool squared distances before square root; weights are buffer populations."""
    return np.sqrt(np.average(rank_mse, axis=0, weights=populations))


def worker(args):
    sys.path.insert(0, str(REPO))
    from alf.bin.evaluate_bafcv3_checkpoints import load_model, normalize, tensor_fingerprint, stable_seed, digest
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cp = args.checkpoint
    run = cp.parents[2]
    shards = sorted(cp.parent.glob(cp.name + '-replay_buffer-rank*'))
    shards = [p for p in shards if re.search(r'rank\d+$', p.name)]
    # These studies have four DDP ranks; never count legacy rank-0 duplicate.
    if {int(p.name.rsplit('rank', 1)[1]) for p in shards} != {0, 1, 2, 3}:
        raise ValueError(f'Expected replay ranks 0–3 for {cp}')
    with torch.inference_mode():
        alg, norm, meta = load_model({'config': str(run / 'alf_config.py')}, str(cp), 'cpu')
        encoding_type = 'actor_id' if alg._use_actor_id_encoding else 'functional'
        if encoding_type != args.encoding_type:
            raise ValueError(f'Expected {args.encoding_type} encoding, found {encoding_type}')
        before = tensor_fingerprint(alg), tensor_fingerprint(norm)
        rank_mse, populations, rank_details = [], [], []
        sampled_indices = {}
        for path in shards:
            rank = int(path.name.rsplit('rank', 1)[1])
            stat = path.stat()
            data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)['algorithm']
            sample_seed = stable_seed(args.sample_seed, str(cp), rank)
            obs, population, indices = sample_replay(data, args.samples_per_rank,
                                                      np.random.default_rng(sample_seed))
            del data
            actions = []
            for batch in obs.split(256):
                x = normalize(norm, batch, 'cpu')
                actions.append(alg._actor_networks(x)[0])
            actions = torch.cat(actions)
            rank_mse.append(action_pair_mse(actions))
            populations.append(population)
            sampled_indices[f'rank{rank}_sampled_env_slot'] = indices
            checksum = digest(path)
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Replay file changed during analysis')
            rank_details.append(dict(rank=rank, eligible_states=population, sampled_states=len(obs),
                                     sample_seed=sample_seed, path=str(path), sha256=checksum))
        saved_probe_actions = alg._actor_networks(alg._actor_eval_samples)[0]
        probe_rms = np.sqrt(action_pair_mse(saved_probe_actions))
        from plot_actor_encoding_distances import pairwise_metrics
        encodings, _ = alg._encode_actor_policies()
        representation = pairwise_metrics(encodings.cpu().numpy(), saved_probe_actions.cpu().numpy())
        after = tensor_fingerprint(alg), tensor_fingerprint(norm)
        if before != after:
            raise RuntimeError('Model or normalization state changed')
        meta = dict(run=str(run), checkpoint=str(cp), task=meta['task'], seed=meta['seed'],
            env_steps=meta['env_steps'], ranks=rank_details, checkpoint_sha256=digest(cp),
            config_sha256={str(p):digest(p) for p in [run / 'alf_config.py', *sorted((run/'config_files').glob('*.py'))]},
            normalization='shared saved checkpoint statistics, frozen; not historical per-rank statistics',
            sampling='uniform without replacement per rank; FIRST/MID only; population-weighted MSE pooling',
            encoding_type=encoding_type,
            probe_comparator='saved synthetic probes; inactive only in Control A',
            encoding_protocol='native checkpoint encoding; saved evaluation probes for functional encoding',
            state_unchanged=True, action_dim=actions.shape[-1], torch_version=torch.__version__)
        np.savez_compressed(args.result, replay_rms=pooled_rms(rank_mse, populations),
            saved_probe_rms=probe_rms, rank_mse=np.stack(rank_mse),
            encodings=encodings.cpu().numpy(), encoding_l2=representation['l2'],
            encoding_cosine=representation['cosine'],
            encoding_norms=encodings.norm(dim=-1).cpu().numpy(),
            **sampled_indices, metadata=json.dumps(meta))


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import csv
    groups, rows, pairs = {}, [], []
    ix = list(zip(*np.triu_indices(10, 1)))
    for path in sorted((out / 'raw').glob('*.npz')):
        with np.load(path, allow_pickle=False) as d:
            m = json.loads(str(d['metadata']))
            replay, probe = d['replay_rms'].copy(), d['saved_probe_rms'].copy()
        groups.setdefault(m['run'], []).append((m, replay, probe))
        rows.append(dict(task=m['task'], seed=m['seed'], env_steps=m['env_steps'],
                         replay_rms_mean=float(replay.mean()), replay_rms_min=float(replay.min()),
                         replay_rms_max=float(replay.max()), saved_probe_rms_mean=float(probe.mean()),
                         sampled_states=sum(r['sampled_states'] for r in m['ranks']),
                         eligible_states=sum(r['eligible_states'] for r in m['ranks'])))
        for index, (a, b) in enumerate(ix):
            pairs.append(dict(task=m['task'], seed=m['seed'],env_steps=m['env_steps'],
                actor_i=int(a),actor_j=int(b),replay_rms=float(replay[index]),saved_probe_rms=float(probe[index])))
    if not groups:
        raise ValueError('No successful checkpoint results')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    detail, detail_axes = plt.subplots(len(groups), 2, figsize=(12, 3 * len(groups)), squeeze=False)
    endpoints = []
    for row, (run, records) in enumerate(sorted(groups.items())):
        records.sort(key=lambda r:r[0]['env_steps'])
        x = [r[0]['env_steps'] for r in records]
        replay, probe = (np.stack([r[index] for r in records]) for index in (1,2))
        label=f"seed {records[0][0]['seed']}"
        for col, values in enumerate((replay, probe)):
            line, = axes[col].plot(x, values.mean(axis=1), label=label)
            axes[col].fill_between(x, values.min(axis=1), values.max(axis=1), alpha=.12, color=line.get_color())
            ax=detail_axes[row,col]
            ax.plot(x,values,alpha=.35,linewidth=.8)
            ax.plot(x,values.mean(axis=1),color='black',label='45-pair mean')
            ax.set(title=f"{label}: {'replay states' if col == 0 else 'saved synthetic probes'}")
        endpoints.append(dict(run=run,seed=records[0][0]['seed'],first_step=x[0],last_step=x[-1],
            first_replay_mean=float(replay[0].mean()),last_replay_mean=float(replay[-1].mean()),
            last_replay_min=float(replay[-1].min()),last_replay_max=float(replay[-1].max()),
            first_probe_mean=float(probe[0].mean()),last_probe_mean=float(probe[-1].mean())))
    axes[0].set(title='Replay states: four ranks')
    axes[1].set(title='Saved synthetic probes')
    for ax in [*axes, *detail_axes.flat]:
        ax.set(xlabel='Environment steps',ylabel='Pairwise action RMS',ylim=(0,2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
        ax.grid(alpha=.2)
        ax.legend()
    modes = {r[0].get('encoding_type', 'actor_id') for rr in groups.values() for r in rr}
    study_label = 'BAFCv3 functional encoder' if modes == {'functional'} else 'Control A actor ID'
    fig.suptitle(f'{study_label} — 45-pair mean; bands: pair minimum–maximum')
    for f,name in ((fig,'overview'),(detail,'per_seed')):
        f.tight_layout()
        for ext in ('png','pdf'):
            f.savefig(out/f'{name}.{ext}',dpi=170)
        plt.close(f)
    for name,data in (('summary',rows),('pairs',pairs)):
        with (out/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (out/'endpoints.json').write_text(json.dumps(endpoints,indent=2))
    print(json.dumps(endpoints,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/control_a_replay_action_rms'))
    parser.add_argument('--embedding-study',type=Path,default=Path('artifacts/control_a_embedding_distances'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--run',type=Path,action='append',help='Explicit run directory (repeatable); overrides embedding-study inventory')
    parser.add_argument('--encoding-type',choices=('actor_id','functional'),default='actor_id')
    parser.add_argument('--samples-per-rank',type=int,default=4096)
    parser.add_argument('--sample-seed',type=int,default=0)
    parser.add_argument('--checkpoint',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.samples_per_rank<1 or args.workers<1:
        parser.error('Sample count and workers must be positive')
    if args.checkpoint:
        worker(args);return
    if args.summarize_only:
        summarize(args.output);return
    if args.run:
        jobs=[]
        for run in args.run:
            cps=sorted(p for p in (run/'train/algorithm').glob('ckpt-*') if re.fullmatch(r'ckpt-\d+',p.name))
            if not cps:
                raise ValueError(f'No model checkpoints: {run}')
            jobs.extend(str(p.resolve()) for p in cps)
    else:
        sources=json.loads((args.embedding_study/'manifest.json').read_text())
        jobs=[cp['path'] for run in sources['runs'] for cp in run['checkpoints']]
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'raw').mkdir();(args.output/'logs').mkdir()
    manifest=dict(protocol=__doc__,encoding_type=args.encoding_type,samples_per_rank=args.samples_per_rank,sample_seed=args.sample_seed,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoints=[])
    def launch(cp):
        key=hashlib.sha256(cp.encode()).hexdigest()[:16]
        with (args.output/'logs'/f'{key}.log').open('w') as log:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',cp,
                '--result',str((args.output/'raw'/f'{key}.npz').resolve()),
                '--samples-per-rank',str(args.samples_per_rank),'--sample-seed',str(args.sample_seed),
                '--encoding-type',args.encoding_type],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        print('OK' if p.returncode==0 else 'ERROR',cp,flush=True)
        return dict(path=cp,status='ok' if p.returncode==0 else 'error',log=f'logs/{key}.log')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(launch,jobs):
            manifest['checkpoints'].append(item)
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summarize(args.output)
    if any(r['status']!='ok' for r in manifest['checkpoints']):
        raise SystemExit('Incomplete analysis; see manifest and logs')


if __name__=='__main__':
    main()
