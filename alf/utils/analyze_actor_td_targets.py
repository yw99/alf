"""Compare humanoid BAFCv3/control TD regression targets on saved replay.

Run with .venv/bin/python alf/utils/analyze_actor_td_targets.py
Enumerates the ten possible shared target-head draws (K=1), samples valid
one-step replay transitions, and uses the native TD loss to construct targets.
Current Q uses the same stored action for every actor, as in critic training.
"""
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
ARMS = ('BAFCv3', 'Control A', 'Control B')


def sample_transitions(state, count, rng):
    """Sample chronological pairs, excluding LAST starts/resets/newest slots."""
    import torch
    prefix = '_replay_buffer.'
    fields = {k[len(prefix):]: v for k, v in state.items()}
    if any('discounted_return' in k for k in fields):
        raise ValueError('Monte Carlo lower bounds require explicit handling')
    types = fields['time_step|step_type']
    capacity = types.shape[1]
    eligible = []
    for env, (size, pos) in enumerate(zip(fields['_current_size'], fields['_current_pos'])):
        size, pos = int(size), int(pos)
        if not 0 <= size <= capacity:
            raise ValueError('Invalid ring size')
        slots = (torch.arange(size) + pos - size) % capacity
        t = types[env, slots]
        if not torch.isin(t, torch.tensor([0, 1, 2])).all():
            raise ValueError('Invalid populated step type')
        valid = (t[:-1] != 2) & (t[1:] != 0)
        eligible.extend((env, int(a), int(b)) for a, b in zip(slots[:-1][valid], slots[1:][valid]))
    if not eligible:
        raise ValueError('No valid replay transitions')
    indices = np.asarray(eligible, dtype=np.int64)[rng.choice(len(eligible), min(count, len(eligible)), replace=False)]
    env, start, end = indices.T
    result = {}
    for name in ('observation', 'reward', 'discount', 'step_type'):
        v = fields['time_step|' + name]
        result[name] = torch.stack([v[env, start], v[env, end]])
    result['action'] = fields['rollout_info|rl|action'][env, start].clone()
    return result, len(eligible), indices


def target_moments(targets, current):
    """Moments before rank pooling: [transition, actor, head] in both inputs.

    Target head is shared between actors; online critic and target draw are
    separate axes when computing regression error, because every online critic
    receives the same randomly selected target.
    """
    y, q = np.asarray(targets, dtype=np.float64), np.asarray(current, dtype=np.float64)
    if y.shape != q.shape or y.ndim != 3 or y.shape[1:] != (10, 10):
        raise ValueError('Expected matching [state, 10 actors, 10 heads]')
    if not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite target or prediction')
    i, j = np.triu_indices(10, 1)
    ym, qm = y.mean(axis=2), q.mean(axis=2)
    return dict(target_pair_mse=((y[:, i] - y[:, j])**2).mean(axis=(0, 2)),
                expected_target_pair_mse=((ym[:, i] - ym[:, j])**2).mean(axis=0),
                current_q_pair_mse=((q[:, i] - q[:, j])**2).mean(axis=(0, 2)),
                actor_mean=y.mean(axis=(0, 2)), actor_second=(y*y).mean(axis=(0, 2)),
                head_variance=y.var(axis=2).mean(axis=0),
                actor_variance=y.var(axis=1).mean(axis=(0, 1)),
                td_error_mse=((y*y).mean(axis=2)+(q*q).mean(axis=2)-2*ym*qm).mean(axis=0))


def worker(args):
    sys.path.insert(0, str(REPO))
    import torch
    from alf.algorithms.bafc_algorithm_v3 import BafcCriticState, BafcInfo
    from alf.bin.evaluate_bafcv3_checkpoints import load_model, normalize, tensor_fingerprint, stable_seed, digest
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cp = args.checkpoint
    run = cp.parents[2]
    shards = [p for p in sorted(cp.parent.glob(cp.name+'-replay_buffer-rank*')) if re.search(r'rank\d+$', p.name)]
    if {int(p.name.rsplit('rank', 1)[1]) for p in shards} != {0, 1, 2, 3}:
        raise ValueError('Expected four replay ranks')
    stats = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in [cp, *shards]}
    with torch.inference_mode():
        alg, norm, meta = load_model({'config': str(run/'alf_config.py')}, str(cp), 'cpu')
        arm = 'Control A' if alg._use_actor_id_encoding else 'Control B' if alg._detach_actor_policy_input else 'BAFCv3'
        if arm != args.arm or alg._num_actor_critic != 10 or alg.has_multidim_reward():
            raise ValueError('Unexpected arm, actor count or reward dimension')
        if not alg._use_random_critic_targets or alg._num_sampled_critic_targets != 1:
            raise ValueError('This analysis requires one uniformly selected shared target head')
        losses = alg._critic_losses
        if any(l._lambda != 0 or l._normalize_target or not torch.equal(l.gamma, losses[0].gamma) for l in losses):
            raise ValueError('Requires unnormalized one-step TD targets and common gamma')
        before = tensor_fingerprint(alg), tensor_fingerprint(norm)
        encoding, _ = alg._encode_actor_policies()
        rank_details, populations, moments, saved = [], [], [], {}
        parity_errors = []
        for path in shards:
            rank = int(path.name.rsplit('rank', 1)[1])
            raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)['algorithm']
            sample_seed = stable_seed(args.sample_seed, str(cp), rank, 'td_transitions')
            samples, population, indices = sample_transitions(raw, args.samples_per_rank, np.random.default_rng(sample_seed))
            del raw
            target_batches, current_batches = [], []
            for start in range(0, len(indices), 64):
                stop = start+64
                obs = normalize(norm, samples['observation'][0, start:stop], 'cpu')
                nxt = normalize(norm, samples['observation'][1, start:stop], 'cpu')
                act = samples['action'][start:stop]
                n = len(obs)
                enc = encoding.repeat(n, 1)
                next_actions = alg._actor_networks(nxt)[0].reshape(n*10, -1)
                target_q = alg._target_critic_networks((enc, (nxt.repeat_interleave(10, dim=0), next_actions)))[0].reshape(n, 10, 10)
                current_q = alg._critic_networks((enc, (obs.repeat_interleave(10, dim=0), act.repeat_interleave(10, dim=0))))[0].reshape(n, 10, 10)
                info = BafcInfo(**{k: samples[k][:, start:stop] for k in ('reward', 'discount', 'step_type')})
                y = losses[0].compute_td_target(info, torch.stack([torch.zeros_like(target_q), target_q]))[0]
                manual = info.reward[1, :, None, None] + losses[0].gamma*info.discount[1, :, None, None]*target_q
                torch.testing.assert_close(y, manual, rtol=0, atol=0)
                # Verify actual native training forward and selection, once per rank.
                if start == 0:
                    torch.manual_seed(173)
                    _, native = alg._critic_train_step(nxt, BafcCriticState((), ()), BafcInfo(action=act), next_actions)
                    torch.manual_seed(173)
                    selected = alg._select_critic_targets(target_q)
                    torch.testing.assert_close(selected, native.target_critic, rtol=2e-5, atol=2e-5)
                    torch.manual_seed(173)
                    head = int(torch.randperm(10)[0])
                    native_y = losses[0].compute_td_target(info, torch.stack([torch.zeros_like(native.target_critic), native.target_critic]))[0]
                    torch.testing.assert_close(y[:, :, head], native_y, rtol=2e-5, atol=2e-5)
                    # At the current state native critic receives identical replay actions.
                    _, native_current = alg._critic_train_step(obs, BafcCriticState((), ()), BafcInfo(action=act), alg._actor_networks(obs)[0].reshape(n*10, -1))
                    torch.testing.assert_close(current_q, native_current.critic, rtol=2e-5, atol=2e-5)
                    parity_errors.append(float((y[:, :, head]-native_y).abs().max()))
                target_batches.append(y.cpu().numpy())
                current_batches.append(current_q.cpu().numpy())
            y, q = np.concatenate(target_batches), np.concatenate(current_batches)
            moments.append(target_moments(y, q)); populations.append(population)
            saved.update({f'rank{rank}_targets': y, f'rank{rank}_current_q': q,
                          f'rank{rank}_env_start_end': indices,
                          f'rank{rank}_next_reward': samples['reward'][1].numpy(),
                          f'rank{rank}_next_discount': samples['discount'][1].numpy()})
            rank_details.append(dict(rank=rank, eligible_transitions=population, sampled_transitions=len(indices),
                sample_seed=sample_seed, terminal_fraction=float((samples['step_type'][1]==2).float().mean()),
                zero_discount_fraction=float((samples['discount'][1]==0).float().mean()),
                path=str(path), sha256=digest(path)))
        pooled = {k: np.average([m[k] for m in moments], axis=0, weights=populations) for k in moments[0]}
        if before != (tensor_fingerprint(alg), tensor_fingerprint(norm)):
            raise RuntimeError('Model or normalizer state changed')
        if any((p.stat().st_size, p.stat().st_mtime_ns) != s for p, s in stats.items()):
            raise RuntimeError('Checkpoint or replay changed during analysis')
        meta = dict(arm=arm, task=meta['task'], seed=meta['seed'], env_steps=meta['env_steps'],
            checkpoint=str(cp), checkpoint_sha256=digest(cp), ranks=rank_details,
            gamma=float(losses[0].gamma), sampled_target_heads=1, enumerated_target_heads=10,
            target_definition='y[i,c] = reward[t+1] + gamma * discount[t+1] * Q_target[c](s[t+1], actor_i(s[t+1]), encoding_i)',
            target_selection='one uniform target critic shared by all actors and transitions in each training batch; enumerate all 10 choices',
            sampling='uniform valid one-step transitions without replacement per rank; population-weighted moment pooling',
            current_q_definition='Q_online[j](s[t], stored_rollout_action[t], encoding_i)',
            normalization='saved shared checkpoint observation statistics, frozen',
            interpretation='recomputed checkpoint targets, not historically recorded minibatch targets; each run uses its own replay',
            mc_lower_bound=False, native_parity_max_abs_error=max(parity_errors),
            state_unchanged=True, source_files_unchanged=True,
            config_sha256={str(p): digest(p) for p in [run/'alf_config.py', *sorted((run/'config_files').glob('*.py'))]})
        np.savez_compressed(args.result, **pooled, **saved, metadata=json.dumps(meta))


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    rows, actors, pairs, records = [], [], [], {}
    pair_ids = list(zip(*np.triu_indices(10, 1)))
    for path in sorted((out/'raw').glob('*.npz')):
        with np.load(path, allow_pickle=False) as d:
            m = json.loads(str(d['metadata']))
            base = dict(arm=m['arm'], seed=m['seed'], env_steps=m['env_steps'])
            ypair, epair, qpair = [np.sqrt(d[k]) for k in ('target_pair_mse', 'expected_target_pair_mse', 'current_q_pair_mse')]
            scale = float(np.sqrt(d['actor_second'].mean()))
            row = dict(**base, target_mean=float(d['actor_mean'].mean()), target_value_rms=scale,
                target_pair_rms_mean=float(ypair.mean()), target_pair_rms_min=float(ypair.min()), target_pair_rms_max=float(ypair.max()),
                relative_target_pair_rms=float(ypair.mean()/scale), expected_target_pair_rms_mean=float(epair.mean()),
                current_q_pair_rms_mean=float(qpair.mean()),
                target_head_sd=float(np.sqrt(d['head_variance'].mean())),
                shared_head_actor_variance=float(d['actor_variance']),
                expected_target_actor_variance=float(d['expected_target_pair_mse'].mean()*9/20),
                td_error_rmse=float(np.sqrt(d['td_error_mse'].mean())),
                sampled_transitions=sum(r['sampled_transitions'] for r in m['ranks']),
                eligible_transitions=sum(r['eligible_transitions'] for r in m['ranks']),
                native_parity_max_abs_error=m['native_parity_max_abs_error'])
            rows.append(row)
            actor_rows = [dict(**base, actor=i, target_mean=float(d['actor_mean'][i]),
                target_sd=float(np.sqrt(max(0, d['actor_second'][i]-d['actor_mean'][i]**2))),
                target_value_rms=float(np.sqrt(d['actor_second'][i])),
                target_head_sd=float(np.sqrt(d['head_variance'][i])),
                td_error_rmse=float(np.sqrt(d['td_error_mse'][i]))) for i in range(10)]
            actors.extend(actor_rows)
            key = (m['arm'], m['seed'], m['env_steps'])
            if key in records: raise ValueError(f'Duplicate checkpoint: {key}')
            records[key] = (row, actor_rows)
            for k, (i, j) in enumerate(pair_ids):
                pairs.append(dict(**base, actor_i=int(i), actor_j=int(j), target_rms=float(ypair[k]),
                                  expected_target_rms=float(epair[k]), current_q_rms=float(qpair[k])))
    if not records: raise ValueError('No results')
    common = sorted(set.intersection(*[{k[2] for k in records if k[:2] == (a,s)} for a in ARMS for s in range(4)]))
    if not common: raise ValueError('No complete common checkpoints')
    for name, data in (('summary', rows), ('actors', actors), ('pairs', pairs)):
        data.sort(key=lambda r: (ARMS.index(r['arm']), r['seed'], r['env_steps']))
        with (out/f'{name}.csv').open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    colors=dict(zip(ARMS, ('tab:blue', 'tab:orange', 'tab:green')))
    fig, axes=plt.subplots(2, 3, figsize=(15, 8))
    metrics=(('target_pair_rms_mean','Actor-pair TD target RMS'),
             ('relative_target_pair_rms','Actor-pair RMS / target RMS'),
             ('current_q_pair_rms_mean','Current Q actor-pair RMS (same replay action)'),
             ('expected_target_pair_rms_mean','Actor-pair RMS after averaging target heads'), ('target_head_sd','Target-head SD (fixed actor/state)'),
             ('td_error_rmse','Critic TD-error RMSE'))
    for arm in ARMS:
        for ax, (metric, label) in zip(axes.flat, metrics):
            values=np.array([[records[arm,s,t][0][metric] for t in common] for s in range(4)])
            ax.plot(common, values.mean(axis=0), color=colors[arm], label=arm)
            ax.fill_between(common, values.min(axis=0), values.max(axis=0), color=colors[arm], alpha=.13)
            ax.set(ylabel=label)
            if metric=='current_q_pair_rms_mean': ax.set_yscale('symlog',linthresh=1e-7)
    detail, dax=plt.subplots(4,3,figsize=(15,12),sharex=True)
    means, maxes=plt.subplots(4,3,figsize=(15,12),sharex=True)
    centered, caxes=plt.subplots(4,3,figsize=(15,12),sharex=True)
    for col, arm in enumerate(ARMS):
        for seed in range(4):
            steps=sorted(k[2] for k in records if k[:2]==(arm,seed))
            rr=[records[arm,seed,t][0] for t in steps]
            ax=dax[seed,col]
            ax.plot(steps,[r['target_pair_rms_mean'] for r in rr],label='TD target pair RMS')
            ax.fill_between(steps,[r['target_pair_rms_min'] for r in rr],[r['target_pair_rms_max'] for r in rr],alpha=.15,label='45-pair range')
            ax.plot(steps,[r['current_q_pair_rms_mean'] for r in rr],label='Current Q pair RMS')
            ax.plot(steps,[r['expected_target_pair_rms_mean'] for r in rr],linestyle='--',label='Head-averaged target pair RMS')
            ax.set(title=f'{arm}, seed {seed}',ylabel='RMS difference')
            ax=maxes[seed,col]
            for actor in range(10):
                ax.plot(steps,[records[arm,seed,t][1][actor]['target_mean'] for t in steps],label=str(actor),linewidth=1)
            ax.set(title=f'{arm}, seed {seed}',ylabel='Mean TD target')
            ax=caxes[seed,col]
            for actor in range(10):
                ax.plot(steps,[records[arm,seed,t][1][actor]['target_mean']-records[arm,seed,t][0]['target_mean'] for t in steps],label=str(actor),linewidth=1)
            ax.axhline(0,color='black',linewidth=.5)
            ax.set(title=f'{arm}, seed {seed}',ylabel='Actor mean minus ten-actor mean')
    for ax in [*axes.flat,*dax.flat,*maxes.flat,*caxes.flat]:
        ax.set(xlabel='Environment steps');ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
        ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2 if ax in [*maxes.flat,*caxes.flat] else 1)
    fig.suptitle('humanoid:run — TD regression targets on saved replay\n4-seed mean; bands: seed range; all ten shared target-head choices enumerated')
    detail.suptitle('Actor target differences vs critic predictions on identical replay (s, a)')
    means.suptitle('TD target mean for each of the ten actors — each run uses its own replay')
    centered.suptitle('Actor-specific mean TD target offsets — subtraction is within each checkpoint and seed')
    for f,name in ((fig,'overview'),(detail,'per_seed'),(means,'actor_means'),(centered,'actor_mean_offsets')):
        f.tight_layout()
        for ext in ('png','pdf'): f.savefig(out/f'{name}.{ext}',dpi=170)
        plt.close(f)
    endpoint=dict(env_steps=common[-1], common_checkpoints=common, rows=[r for r in rows if r['env_steps']==common[-1]])
    (out/'matched_endpoint.json').write_text(json.dumps(endpoint,indent=2))
    for arm in ARMS:
        rr=[r for r in endpoint['rows'] if r['arm']==arm]
        print(arm,{k:float(np.mean([r[k] for r in rr])) for k,_ in metrics},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/humanoid_actor_td_targets'))
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--samples-per-rank',type=int,default=512)
    p.add_argument('--sample-seed',type=int,default=0)
    p.add_argument('--checkpoint',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--arm',choices=ARMS,help=argparse.SUPPRESS)
    p.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--resume',action='store_true',help='Reuse validated completed results in an existing output directory')
    args=p.parse_args()
    if min(args.workers,args.samples_per_rank)<1:p.error('Positive workers and samples required')
    if args.checkpoint:worker(args);return
    if args.summarize_only:summarize(args.output);return
    sources=dict(zip(ARMS,[REPO/'artifacts'/x/'manifest.json' for x in
        ('humanoid_run_bafcv3_replay_and_encodings','control_a_replay_action_rms','control_b_replay_and_encodings')]))
    jobs=[]
    for arm,path in sources.items():
        manifest=json.loads(path.read_text())
        if any(r['status']!='ok' for r in manifest['checkpoints']):raise ValueError(f'Incomplete inventory: {path}')
        jobs.extend((arm,cp['path']) for cp in manifest['checkpoints'])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    (args.output/'raw').mkdir(exist_ok=args.resume);(args.output/'logs').mkdir(exist_ok=args.resume)
    previous = json.loads((args.output/'manifest.json').read_text()) if args.resume else {}
    if previous and (previous['samples_per_rank'] != args.samples_per_rank or previous['sample_seed'] != args.sample_seed):
        raise ValueError('Resume sampling settings differ')
    manifest=dict(input_studies={a:str(path) for a,path in sources.items()},checkpoints=[],
        samples_per_rank=args.samples_per_rank,sample_seed=args.sample_seed,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prior_source_sha256=previous.get('source_sha256'),protocol=__doc__)
    def launch(job):
        arm,cp=job;key=hashlib.sha256(cp.encode()).hexdigest()[:16]
        result=args.output/'raw'/f'{key}.npz'
        if args.resume and result.exists():
            with np.load(result,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
                if meta['arm'] != arm or meta['checkpoint'] != cp or not meta['state_unchanged'] or not meta['source_files_unchanged']:
                    raise ValueError(f'Incompatible cached result: {result}')
                for rank in meta['ranks']:
                    if rank['sampled_transitions'] != min(args.samples_per_rank,rank['eligible_transitions']):
                        raise ValueError(f'Incompatible sample count: {result}')
            print('REUSED',arm,cp,flush=True)
            return dict(arm=arm,path=cp,status='ok',log=f'logs/{key}.log',reused=True)
        with (args.output/'logs'/f'{key}.log').open('w') as log:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',cp,'--arm',arm,
                '--samples-per-rank',str(args.samples_per_rank),'--sample-seed',str(args.sample_seed),
                '--result',str((args.output/'raw'/f'{key}.npz').resolve())],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        print('OK' if proc.returncode==0 else 'ERROR',arm,cp,flush=True)
        return dict(arm=arm,path=cp,status='ok' if proc.returncode==0 else 'error',log=f'logs/{key}.log')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(launch,jobs):
            manifest['checkpoints'].append(r)
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if any(r['status']!='ok' for r in manifest['checkpoints']):raise SystemExit('Incomplete analysis: see logs')
    summarize(args.output)


if __name__=='__main__':main()
