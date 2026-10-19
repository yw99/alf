"""Read-only actor embedding geometry for active BAFCv3 jobs.

    .venv/bin/python alf/utils/analyze_running_actor_embeddings.py

Snapshots completed model checkpoints (>60 seconds old). Strictly restores saved
configs; evaluates online and target encoders on native saved-probe tokens.
Also compares deterministic actions and actor-conditioned critic predictions on
shared replay states (all four ranks). Never starts an environment or training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PAIRS = np.triu_indices(10, 1)
BLOCKS = ('online', 'target', 'input', 'first_token')


def geometry(values):
    """Pair geometry and centered covariance spectrum across the ten actors."""
    x=np.asarray(values,dtype=np.float64).reshape(10,-1)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite embedding')
    norms=np.linalg.norm(x,axis=1)
    l2=np.array([np.linalg.norm(x[i]-x[j]) for i,j in zip(*PAIRS)])
    denominator=norms[PAIRS[0]]*norms[PAIRS[1]]
    dots=np.einsum('ij,ij->i',x[PAIRS[0]],x[PAIRS[1]])
    cosine=1-np.divide(dots,denominator,out=np.full(45,np.nan),where=denominator>0)
    cosine=np.clip(cosine,0,2)
    center=x-x.mean(axis=0)
    eigen=np.maximum(np.linalg.eigvalsh(center@center.T)[::-1],0)
    # Centering bounds rank by 9; drop only numerical eigensolver roundoff.
    eigen[eigen<eigen[0]*1e-12]=0
    total=eigen.sum()
    p=eigen[eigen>0]/total if total>0 else np.array([])
    erank=float(np.exp(-(p*np.log(p)).sum())) if total>0 else 0.
    energy=float(np.square(x).sum())
    return dict(l2=l2,rms=l2/np.sqrt(x.shape[1]),cosine=cosine,norms=norms,
        relative_l2=l2/np.sqrt(np.square(norms).mean()) if energy>0 else np.full(45,np.nan),
        centered_singular_values=np.sqrt(eigen),effective_rank=np.array(erank),
        centered_energy_fraction=np.array(total/energy if energy>0 else 0.),
        leading_centered_variance_fraction=np.array(eigen[0]/total if total>0 else 0.))


def active_runs():
    result={}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: argv=path.read_bytes().decode().split('\0')
        except (OSError,UnicodeError): continue
        if 'alf.bin.train' not in argv or '--root_dir' not in argv: continue
        run=argv[argv.index('--root_dir')+1]
        result.setdefault(run,[]).append(int(path.parent.name))
    if not result: raise ValueError('No running alf.bin.train jobs found')
    return result


def worker(args):
    sys.path.insert(0,str(REPO))
    import torch
    from alf.bin.evaluate_bafcv3_checkpoints import load_model, normalize, tensor_fingerprint, stable_seed, digest
    from alf.utils.plot_replay_actor_action_rms import sample_replay, action_pair_mse, pooled_rms
    from alf.algorithms.bafc_algorithm_v3 import BafcCriticState,BafcInfo
    torch.set_num_threads(2);torch.manual_seed(0)
    cp=args.checkpoint;run=cp.parents[2]
    paths=[cp]+[p for p in sorted(cp.parent.glob(cp.name+'-replay_buffer-rank*')) if re.search(r'rank\d+$',p.name)]
    if len(paths)!=5: raise ValueError('Expected four replay ranks')
    stats={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
    with torch.inference_mode():
        alg,norm,meta=load_model({'config':str(run/'alf_config.py')},str(cp),'cpu',allow_frozen_eval_samples=True)
        if alg._num_actor_critic!=10 or alg._actor_eval_type!='last_two' or alg._use_actor_id_encoding:
            raise ValueError('Requires ten functional last_two actors')
        if not alg._use_target_actor_encoder: raise ValueError('Target encoder required for this study')
        before=(tensor_fingerprint(alg),tensor_fingerprint(norm))
        z,eval_out,tokens=alg._encode_actor_policies(return_tokens=True)
        tz=alg._target_actor_encoder(tokens.detach())[0]
        # Check native encoder input and first-token readout with its capture path.
        with alg._actor_encoder.capture_gradient_chain() as capture:
            captured_z=alg._actor_encoder(tokens)[0]
        torch.testing.assert_close(capture['input'],tokens,rtol=0,atol=0)
        torch.testing.assert_close(captured_z,z,rtol=2e-5,atol=2e-5)
        source=capture.get('final_norm',capture['stack_output'])
        if alg._actor_encoder._core_fc is not None or alg._actor_encoder._core_size!=1:
            raise ValueError('Unexpected encoder readout')
        torch.testing.assert_close(captured_z,source[:,0],rtol=0,atol=0)
        # Current runs have dropout=0 and LayerNorm; verify training mode directly.
        alg._actor_encoder.train()
        train_z=alg._actor_encoder(tokens)[0]
        alg._actor_encoder.eval()
        torch.testing.assert_close(train_z,z,rtol=2e-5,atol=2e-5)
        geometry_arrays={}
        for name,values in dict(online=z,target=tz,input=tokens,first_token=tokens[:,0]).items():
            geometry_arrays.update({name+'_'+key:v for key,v in geometry(values.cpu().numpy()).items()})
        probe_actions=alg._actor_networks(alg._actor_eval_samples)[0]
        torch.testing.assert_close(probe_actions,eval_out[-1],rtol=0,atol=0)
        saved=dict(online_encodings=z.cpu().numpy(),target_encodings=tz.cpu().numpy(),
                   probe_action_pair_rms=np.sqrt(action_pair_mse(probe_actions)),
                   online_target_actor_l2=(z-tz).norm(dim=-1).cpu().numpy())
        # Native final-norm input/output geometry on the selected token.
        for name in ('positioned_input','stack_output','final_norm'):
            if name in capture:
                saved.update({f'stage_{name}_{key}':v for key,v in geometry(capture[name][:,0].cpu().numpy()).items()})
        rank_action,rank_q,pops,ranks=[],[],[],[]
        parity=[]
        for path in paths[1:]:
            rank=int(path.name.rsplit('rank',1)[1])
            raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['algorithm']
            seed=stable_seed(str(cp),rank,'running_embedding_analysis')
            obs,pop,indices=sample_replay(raw,args.samples_per_rank,np.random.default_rng(seed))
            act=raw['_replay_buffer.rollout_info|rl|action'][indices[:,0],indices[:,1]].clone()
            del raw
            actions,qs=[],[]
            for start in range(0,len(obs),64):
                x=normalize(norm,obs[start:start+64],'cpu');a=act[start:start+64]
                action=alg._actor_networks(x)[0]
                q=alg._critic_networks((z.repeat(len(x),1),(x.repeat_interleave(10,dim=0),a.repeat_interleave(10,dim=0))))[0].reshape(len(x),10,10)
                if start==0 and rank==0:
                    _,native=alg._critic_train_step(x,BafcCriticState((),()),BafcInfo(action=a),action.reshape(len(x)*10,-1))
                    torch.testing.assert_close(q,native.critic,rtol=2e-5,atol=2e-5)
                    parity.append(float((q-native.critic).abs().max()))
                actions.append(action);qs.append(q)
            actions=torch.cat(actions);qs=torch.cat(qs)
            rank_action.append(action_pair_mse(actions));rank_q.append(action_pair_mse(qs));pops.append(pop)
            saved[f'rank{rank}_env_slot']=indices
            ranks.append(dict(rank=rank,path=str(path),sha256=digest(path),eligible_states=pop,sampled_states=len(obs),sample_seed=seed))
        saved.update(replay_action_pair_rms=pooled_rms(rank_action,pops),replay_q_pair_rms=pooled_rms(rank_q,pops))
        if before!=(tensor_fingerprint(alg),tensor_fingerprint(norm)): raise RuntimeError('Model or normalizer mutated')
        if any(stats[str(p)]!=[p.stat().st_size,p.stat().st_mtime_ns] for p in paths): raise RuntimeError('Source changed during analysis')
        m=dict(run=str(run),checkpoint=str(cp),checkpoint_sha256=digest(cp),source_stats=stats,
            task=meta['task'],seed=meta['seed'],env_steps=meta['env_steps'],global_step=meta['global_step'],ranks=ranks,
            actor_eval_type=alg._actor_eval_type,eval_samples_source=alg._eval_samples_source,
            input_shape=list(tokens.shape),embedding_shape=list(z.shape),
            eval_samples_sha256=hashlib.sha256(alg._actor_eval_samples.cpu().numpy().tobytes()).hexdigest(),
            actual_transformer_layers=len(alg._actor_encoder._transformer.layers),
            norm_first=alg._actor_encoder._transformer.layers[0].norm_first,
            final_norm=alg._actor_encoder._transformer.norm is not None,actor_use_ln=alg._actor_use_ln,
            state_unchanged=True,source_files_unchanged=True,native_input_and_readout_verified=True,
            train_eval_max_abs_error=float((train_z-z).abs().max()),native_critic_max_abs_error=max(parity),
            target_protocol='target encoder on same current actor/probe tokens as native critic training',
            replay_protocol='shared checkpoint observation normalizer; identical sampled replay (s,a) across actors; population-weighted MSE across ranks',
            config_sha256={str(p):digest(p) for p in [run/'alf_config.py',*sorted((run/'config_files').glob('*.py'))]})
        np.savez_compressed(args.result,**geometry_arrays,**saved,metadata=json.dumps(m))


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    rows,pairs,records,stages=[],[],{},[]
    for path in sorted((out/'raw').glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']));base=dict(task=m['task'],seed=m['seed'],env_steps=m['env_steps'])
            key=(m['seed'],m['env_steps'])
            if key in records: raise ValueError('Duplicate seed/env-step record')
            records[key]={k:d[k].copy() for k in d.files if k!='metadata'}
            row=dict(**base)
            for block in BLOCKS:
                for metric in ('l2','rms','cosine','relative_l2','norms'):
                    v=d[f'{block}_{metric}']
                    for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:row[f'{block}_{metric}_{stat}']=float(fn(v))
                for metric in ('effective_rank','centered_energy_fraction','leading_centered_variance_fraction'):
                    row[f'{block}_{metric}']=float(d[f'{block}_{metric}'])
                for k,(i,j) in enumerate(zip(*PAIRS)):
                    pairs.append(dict(**base,block=block,actor_i=int(i),actor_j=int(j),**{metric:float(d[f'{block}_{metric}'][k]) for metric in ('l2','rms','cosine','relative_l2')}))
            for metric in ('probe_action_pair_rms','replay_action_pair_rms','replay_q_pair_rms','online_target_actor_l2'):
                row[metric+'_mean']=float(d[metric].mean());row[metric+'_min']=float(d[metric].min());row[metric+'_max']=float(d[metric].max())
            rows.append(row)
            for stage in ('first_token','stage_positioned_input','stage_stack_output','stage_final_norm'):
                stages.append(dict(**base,stage=stage,pair_l2_mean=float(d[stage+'_l2'].mean()),pair_rms_mean=float(d[stage+'_rms'].mean()),pair_cosine_mean=float(d[stage+'_cosine'].mean()),norm_mean=float(d[stage+'_norms'].mean()),effective_rank=float(d[stage+'_effective_rank'])))
    if not rows:raise ValueError('No results')
    rows.sort(key=lambda r:(r['seed'],r['env_steps']))
    for name,data in (('summary',rows),('pairs',pairs),('stages',stages)):
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    seeds=sorted({r['seed'] for r in rows})
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    specs=[('online_l2_mean','Online embedding: mean pair L2'),('target_l2_mean','Target embedding: mean pair L2'),
           ('online_cosine_mean','Online embedding: mean cosine distance'),('online_effective_rank','Centered effective rank (maximum 9)'),
           ('replay_action_pair_rms_mean','Replay actions: mean pair RMS'),('replay_q_pair_rms_mean','Replay Q(s,a,z): mean actor-pair RMS')]
    detail,daxes=plt.subplots(len(seeds),3,figsize=(15,3*len(seeds)),squeeze=False)
    heat,haxes=plt.subplots(len(seeds),2,figsize=(10,4*len(seeds)),squeeze=False)
    stagefig,saxes=plt.subplots(2,2,figsize=(12,8))
    for row,seed in enumerate(seeds):
        stage_ax=saxes.flat[row]
        for stage,label in [('first_token','Raw first token'),('stage_stack_output','Before final LayerNorm'),('stage_final_norm','Final embedding')]:
            ss=sorted([r for r in stages if r['seed']==seed and r['stage']==stage],key=lambda r:r['env_steps'])
            stage_ax.plot([r['env_steps'] for r in ss],[r['pair_cosine_mean'] for r in ss],marker='o',label=label)
        stage_ax.set(title=f'Seed {seed}',ylabel='Mean cosine distance',yscale='log')
        rr=[r for r in rows if r['seed']==seed];steps=[r['env_steps'] for r in rr]
        for ax,(metric,label) in zip(axes.flat,specs):
            ax.plot(steps,[r[metric] for r in rr],marker='o',label=f'Seed {seed}')
            ax.set(ylabel=label)
        for block in ('online','target'):
            ax=daxes[row,0];ax.plot(steps,[r[f'{block}_l2_mean'] for r in rr],label=block)
            ax.fill_between(steps,[r[f'{block}_l2_min'] for r in rr],[r[f'{block}_l2_max'] for r in rr],alpha=.15)
        dax=daxes[row,1]
        for block in ('first_token','input','online','target'):
            dax.plot(steps,[r[f'{block}_rms_mean'] for r in rr],label=block)
        dax.set_yscale('log')
        dax=daxes[row,2]
        dax.plot(steps,[r['online_centered_energy_fraction'] for r in rr],label='Online centered energy / total')
        dax.plot(steps,[r['online_leading_centered_variance_fraction'] for r in rr],label='PC1 share of centered variance')
        for ax,label in zip(daxes[row],('Pair L2: mean and pair range','Input/output pair RMS','Common-vector and rank diagnostics')):
            ax.set(title=f'Seed {seed}: {label}')
        for col,block in enumerate(('online','target')):
            v=records[seed,steps[-1]][block+'_l2'];mat=np.zeros((10,10));mat[PAIRS]=v;mat+=mat.T
            vmax=max(records[seed,steps[-1]][b+'_l2'].max() for b in ('online','target'))
            ax=haxes[row,col];im=ax.imshow(mat,vmin=0,vmax=vmax,cmap='viridis')
            ax.set(title=f'Seed {seed}: {block}, {steps[-1]/1000:g}k steps',xlabel='Actor',ylabel='Actor',xticks=range(10),yticks=range(10))
            heat.colorbar(im,ax=ax,label='Embedding L2')
    for ax in [*axes.flat,*daxes.flat,*saxes.flat]:
        ax.set(xlabel='Environment steps');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
    axes[1,0].set_ylim(0,9.5)
    fig.suptitle('humanoid:walk launcher runs — 45 actor pairs, four seeds\nSaved checkpoints; online and Polyak target encoder; replay uses 4 rank buffers')
    detail.suptitle('Embedding separation and compression across checkpoints')
    heat.suptitle('Latest completed checkpoints: online and target actor embedding distances')
    stagefig.suptitle('Directional alignment appears inside the transformer, before final LayerNorm')
    for f,name in ((fig,'overview'),(detail,'per_seed'),(heat,'latest_pairwise'),(stagefig,'encoder_stages')):
        f.tight_layout()
        for ext in ('png','pdf'):f.savefig(out/f'{name}.{ext}',dpi=170)
        plt.close(f)
    late_rows=[r for r in rows if r['env_steps']>=75000]
    if late_rows:
        late,laxes=plt.subplots(2,3,figsize=(15,8))
        late_specs=[('online_l2_mean','Mean embedding pair L2','log'),
                    ('online_cosine_mean','Mean embedding cosine distance','log'),
                    ('online_norms_mean','Mean embedding norm','linear'),
                    ('online_relative_l2_mean','Pair L2 / RMS embedding norm','log'),
                    ('replay_q_pair_rms_mean','Critic actor-pair RMS on identical replay (s,a)','log'),
                    ('online_effective_rank','Centered effective rank (maximum 9)','linear')]
        for seed in seeds:
            rr=[r for r in late_rows if r['seed']==seed]
            for ax,(metric,label,scale) in zip(laxes.flat,late_specs):
                ax.plot([r['env_steps'] for r in rr],[r[metric] for r in rr],marker='o',label=f'Seed {seed}')
                ax.set(xlabel='Environment steps',ylabel=label,yscale=scale)
        for ax in laxes.flat:
            ax.grid(alpha=.2);ax.legend(fontsize=8)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
        late.suptitle('Later checkpoints: distinguish directional alignment from shrinking embedding scale')
        late.tight_layout()
        for ext in ('png','pdf'):late.savefig(out/f'late_training.{ext}',dpi=170)
        plt.close(late)
    endpoints=[max([r for r in rows if r['seed']==s],key=lambda r:r['env_steps']) for s in seeds]
    (out/'endpoints.json').write_text(json.dumps(endpoints,indent=2))
    common=sorted(set.intersection(*[{r['env_steps'] for r in rows if r['seed']==seed} for seed in seeds]))
    comparison=[]
    for latest in endpoints:
        seed=latest['seed']
        first=min([r for r in rows if r['seed']==seed],key=lambda r:r['env_steps'])
        reference=next((r for r in rows if r['seed']==seed and r['env_steps']==75000),None)
        item=dict(seed=seed,latest_env_steps=latest['env_steps'],first_env_steps=first['env_steps'],
                  first_l2_mean=first['online_l2_mean'],latest_l2_mean=latest['online_l2_mean'],
                  l2_drop_from_first_fraction=1-latest['online_l2_mean']/first['online_l2_mean'])
        if reference:
            for metric in ('online_l2_mean','online_cosine_mean','online_norms_mean','online_relative_l2_mean','online_effective_rank','replay_action_pair_rms_mean','replay_q_pair_rms_mean'):
                item[metric+'_at_75k']=reference[metric]
                item[metric+'_latest']=latest[metric]
                item[metric+'_ratio_since_75k']=latest[metric]/reference[metric]
        comparison.append(item)
    (out/'comparison_since_75k.json').write_text(json.dumps(comparison,indent=2))
    (out/'matched_endpoint.json').write_text(json.dumps(dict(env_steps=common[-1],rows=[r for r in rows if r['env_steps']==common[-1]]),indent=2))
    for r in endpoints: print({k:r[k] for k in ['seed','env_steps','online_l2_mean','online_l2_min','online_cosine_mean','online_effective_rank','target_l2_mean','replay_action_pair_rms_mean','replay_q_pair_rms_mean']},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',type=Path)
    p.add_argument('--output',type=Path,default=REPO/'artifacts/running_actor_embedding_collapse_20261006')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--samples-per-rank',type=int,default=256)
    p.add_argument('--checkpoint',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args()
    if a.checkpoint:worker(a);return
    if a.summarize_only:summarize(a.output);return
    if min(a.workers,a.samples_per_rank)<1:p.error('Positive workers and samples required')
    runs={str(r):[] for r in a.run} if a.run else active_runs()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():raise ValueError('Use a fresh study output directory')
    (a.output/'raw').mkdir();(a.output/'logs').mkdir()
    jobs=[];snapshot=[]
    for run,pids in sorted(runs.items()):
        for cp in sorted((Path(run)/'train/algorithm').glob('ckpt-*'),key=lambda p:p.name):
            if not re.fullmatch(r'ckpt-\d+',cp.name) or time.time()-cp.stat().st_mtime<60:continue
            jobs.append(cp);snapshot.append(dict(path=str(cp),size=cp.stat().st_size,mtime_ns=cp.stat().st_mtime_ns,pids=pids))
    manifest=dict(captured_utc=datetime.now(timezone.utc).isoformat(),runs=runs,snapshot=snapshot,checkpoints=[],samples_per_rank=a.samples_per_rank,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol=__doc__)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def launch(cp):
        key=hashlib.sha256(str(cp).encode()).hexdigest()[:16]
        with (a.output/'logs'/f'{key}.log').open('w') as log:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(cp),'--result',str((a.output/'raw'/f'{key}.npz').resolve()),'--samples-per-rank',str(a.samples_per_rank)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        print('OK' if proc.returncode==0 else 'ERROR',cp,flush=True)
        return dict(path=str(cp),status='ok' if proc.returncode==0 else 'error',log=f'logs/{key}.log')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(launch,jobs):
            manifest['checkpoints'].append(r);(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if any(r['status']!='ok' for r in manifest['checkpoints']):raise SystemExit('Incomplete analysis; see logs')
    summarize(a.output)


if __name__=='__main__':main()
