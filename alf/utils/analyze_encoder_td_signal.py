"""Decompose encoder TD gradients and test actor-conditioning interventions.

Offline: freezes sampled Bellman targets and model parameters; no optimizer step.
All ten possible target-head draws are enumerated, shared between actors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[2]


def decompose(q,y):
    """q:[N,actor,online head], y:[N,actor,target head]."""
    r=q.double().unsqueeze(-1)-y.double().unsqueeze(2)
    common=r.mean(dim=1,keepdim=True)
    return r.square().mean(),common.square().mean(),(r-common).square().mean()


def gradnorm(gs):
    return float(sum(g.double().square().sum() for g in gs if g is not None).sqrt())


def dot(a,b):
    return float(sum((x.double()*y.double()).sum() for x,y in zip(a,b) if x is not None and y is not None))


def worker(a):
    sys.path.insert(0,str(REPO))
    import torch
    from alf.bin.evaluate_bafcv3_checkpoints import load_model,normalize,tensor_fingerprint,stable_seed,digest
    from alf.utils.analyze_actor_td_targets import sample_transitions
    from alf.algorithms.bafc_algorithm_v3 import BafcCriticState,BafcInfo
    torch.set_num_threads(2);torch.manual_seed(0)
    cp=a.checkpoint;run=cp.parents[2]
    alg,norm,meta=load_model({'config':str(run/'alf_config.py')},str(cp),'cpu')
    if not alg._use_target_actor_encoder or alg._num_sampled_critic_targets!=1:
        raise ValueError('Requires native target encoder and K=1')
    before=(tensor_fingerprint(alg),tensor_fingerprint(norm))
    # Match ordinary critic training: current actors frozen, probes/encoder trainable.
    alg._actor_networks.requires_grad_(False)
    alg._actor_eval_samples.requires_grad_(True)
    samples=[];ranks=[];indices_saved={};stats={}
    for rank in range(4):
        path=Path(str(cp)+f'-replay_buffer-rank{rank}')
        stats[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)
        raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['algorithm']
        seed=stable_seed(str(cp),rank,'encoder_td_signal')
        s,pop,idx=sample_transitions(raw,a.samples_per_rank,np.random.default_rng(seed));del raw
        samples.append(s);indices_saved[f'rank{rank}_env_start_end']=idx
        ranks.append(dict(rank=rank,path=str(path),sha256=digest(path),population=pop,sampled=len(idx),sample_seed=seed))
    if len({r['population'] for r in ranks})!=1:
        raise ValueError('Equal rank populations required for this equally weighted gradient diagnostic')
    sample={k:torch.cat([s[k] for s in samples],dim=0 if k=='action' else 1) for k in samples[0]}
    obs=normalize(norm,sample['observation'][0],'cpu');nxt=normalize(norm,sample['observation'][1],'cpu')
    act=sample['action'];n=len(act)
    z,_,tokens=alg._encode_actor_policies(return_tokens=True)
    with torch.no_grad():
        tz=alg._target_actor_encoder(tokens.detach())[0]
        tq=[]
        for start in range(0,n,64):
            x=nxt[start:start+64]
            aa=alg._actor_networks(x)[0].reshape(len(x)*10,-1)
            tq.append(alg._target_critic_networks((tz.repeat(len(x),1),(x.repeat_interleave(10,dim=0),aa)))[0].reshape(len(x),10,10))
        tq=torch.cat(tq)
        info=BafcInfo(**{k:sample[k] for k in ('reward','discount','step_type')})
        target=alg._critic_losses[0].compute_td_target(info,torch.stack([torch.zeros_like(tq),tq]))[0]
    def predict(encoding):
        qs=[]
        for start in range(0,n,64):
            x=obs[start:start+64];aa=act[start:start+64]
            qs.append(alg._critic_networks((encoding.repeat(len(x),1),(x.repeat_interleave(10,dim=0),aa.repeat_interleave(10,dim=0))))[0].reshape(len(x),10,10))
        return torch.cat(qs)
    q=predict(z)
    total,common,contrast=decompose(q,target)
    torch.testing.assert_close(total,common+contrast,rtol=1e-12,atol=1e-14)
    # Native training forward parity plus exact loss reduction for one selected draw.
    with torch.no_grad():
        torch.manual_seed(123)
        _,native=alg._critic_train_step(obs[:64],BafcCriticState((),()),BafcInfo(action=act[:64]),alg._actor_networks(obs[:64])[0].reshape(-1,act.shape[-1]))
        torch.testing.assert_close(q[:64],native.critic,rtol=2e-5,atol=2e-5)
        torch.manual_seed(123)
        _,native_next=alg._critic_train_step(nxt[:64],BafcCriticState((),()),BafcInfo(action=act[:64]),alg._actor_networks(nxt[:64])[0].reshape(-1,act.shape[-1]))
        torch.manual_seed(123);selected=alg._select_critic_targets(tq[:64])
        torch.testing.assert_close(selected,native_next.target_critic,rtol=2e-5,atol=2e-5)
        local_info=BafcInfo(**{k:sample[k][:,:64] for k in ('reward','discount','step_type')})
        selected_values=torch.stack([torch.zeros_like(selected),selected])
        y_native=alg._critic_losses[0].compute_td_target(local_info,selected_values)[0]
        scalar=sum(l(local_info,torch.stack([q[:64,:,j].detach(),torch.zeros_like(q[:64,:,j])]),selected_values).loss.mean() for j,l in enumerate(alg._critic_losses))
        # Actor average, sum over 10 heads, zero-padded final time step => factor 5.
        torch.testing.assert_close(scalar,5*(q[:64]-y_native[:,:,None]).square().mean(),rtol=2e-5,atol=2e-5)
    params=list(alg._actor_encoder.parameters())
    gcommon=torch.autograd.grad(common,[z,*params],retain_graph=True,allow_unused=True)
    gcontrast=torch.autograd.grad(contrast,[z,*params],retain_graph=True,allow_unused=True)
    gz=gcommon[0]+gcontrast[0]
    gc,gx=gcommon[1:],gcontrast[1:]
    gt=[(c if c is not None else 0)+(x if x is not None else 0) for c,x in zip(gc,gx)]
    gc_norm,gx_norm,gt_norm=map(gradnorm,(gc,gx,gt))
    # Local gradient direction, not a training trajectory or actual Adam update.
    idx=torch.triu_indices(10,10,1)
    zn=torch.nn.functional.normalize(z.double(),dim=1)
    cosine=(1-(zn[idx[0]]*zn[idx[1]]).sum(-1)).mean()
    gcos=torch.autograd.grad(cosine,params,retain_graph=True,allow_unused=True)
    norm_gcos=gradnorm(gcos)
    directional={}
    for label,g,gn in [('total',gt,gt_norm),('common',gc,gc_norm),('actor_centered',gx,gx_norm)]:
        directional[f'negative_gradient_cosine_distance_derivative_{label}']=-dot(gcos,g)
        directional[f'gradient_cosine_with_separation_{label}']=dot(gcos,g)/(norm_gcos*gn) if norm_gcos*gn>0 else 0.
    with torch.no_grad():
        centered=q-q.mean(dim=1,keepdim=True)
        ycenter=target-target.mean(dim=1,keepdim=True)
        expected_y=target.mean(dim=2)
        qcentroid=predict(z.mean(dim=0,keepdim=True).expand_as(z))
        centroid_loss,_,_=decompose(qcentroid,target)
        # Expected random permutation of actor labels, with frozen targets.
        permutation_loss=common+centered.double().square().mean()+ycenter.double().square().mean()
        # Projection/remove common embedding component, without editing any weights.
        centered_encoding=z-z.mean(dim=0,keepdim=True)
        qcentered=predict(centered_encoding)
        centered_encoding_loss,_,_=decompose(qcentered,target)
        scale_rows=[]
        for scale in (0.,.5,1.,2.):
            zz=z.mean(dim=0,keepdim=True)+scale*centered_encoding
            qq=q if scale==1 else qcentroid if scale==0 else predict(zz)
            lt,lc,lx=decompose(qq,target)
            scale_rows.append(dict(scale=scale,td_mse=float(lt),common_mse=float(lc),actor_centered_mse=float(lx)))
        row=dict(seed=meta['seed'],env_steps=meta['env_steps'],samples=n,
            td_mse=float(total),common_mse=float(common),actor_centered_mse=float(contrast),
            actor_centered_loss_fraction=float(contrast/total),
            target_head_variance=float(target.double().var(dim=2,unbiased=False).mean()),
            expected_target_actor_variance=float(expected_y.double().var(dim=1,unbiased=False).mean()),
            actor_prediction_variance=float(q.double().var(dim=1,unbiased=False).mean()),
            target_pair_rms_mean=float(((target[:,idx[0]].double()-target[:,idx[1]].double()).square().mean(dim=(0,2)).sqrt()).mean()),
            common_gradient_norm=gc_norm,actor_centered_gradient_norm=gx_norm,total_gradient_norm=gt_norm,
            common_to_actor_centered_gradient_norm_ratio=gc_norm/gx_norm,
            gradient_common_contrast_cosine=dot(gc,gx)/(gc_norm*gx_norm),
            embedding_gradient_common_energy_fraction=float(gz.mean(dim=0).square().sum()*10/gz.square().sum()),
            centroid_td_mse=float(centroid_loss),centroid_loss_relative_change=float(centroid_loss/total-1),
            permuted_td_mse=float(permutation_loss),permuted_loss_relative_change=float(permutation_loss/total-1),
            centered_encoding_td_mse=float(centered_encoding_loss),**directional)
    if before!=(tensor_fingerprint(alg),tensor_fingerprint(norm)):raise RuntimeError('Model or normalizer changed')
    if any(stats[str(p)]!=(p.stat().st_size,p.stat().st_mtime_ns) for p in map(Path,stats)):raise RuntimeError('Replay changed')
    result=dict(checkpoint=str(cp),checkpoint_sha256=digest(cp),row=row,ranks=ranks,scale_interventions=scale_rows,
        protocol='Uniform replay transitions, all shared target-head draws, exact squared TD loss decomposition; frozen targets for every intervention; no optimizer step.',
        gradient_convention='Mean over transitions, actors, online and target heads. Native two-time-step padded aggregate loss is 5 times this mean.',
        direction_convention='Negative loss-gradient dot separation-gradient: negative means infinitesimal SGD would contract cosine distance at this checkpoint. This is not actual Adam history.',
        state_unchanged=True,native_forward_target_and_loss_parity=True)
    a.result.write_text(json.dumps(result,indent=2))
    np.savez_compressed(a.result.with_suffix('.npz'),predictions=q.detach().numpy(),targets=target.numpy(),encoding=z.detach().numpy(),encoding_gradient=gz.detach().numpy(),**indices_saved)
    print(json.dumps(row),flush=True)


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(p.read_text())['row'] for p in (out/'raw').glob('*.json')]
    rows.sort(key=lambda r:(r['seed'],r['env_steps']))
    for r in rows:
        learnable=r['td_mse']-r['target_head_variance']
        r['learnable_td_mse']=learnable
        r['centroid_learnable_loss_relative_change']=(r['centroid_td_mse']-r['td_mse'])/learnable
        r['permuted_learnable_loss_relative_change']=(r['permuted_td_mse']-r['td_mse'])/learnable
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    colors=plt.get_cmap('tab10').colors
    for seed in range(4):
        rr=[r for r in rows if r['seed']==seed]
        for ax,(key,label) in zip(axes.flat[:2],[(
            'common_to_actor_centered_gradient_norm_ratio','Common / actor-centered encoder gradient norm'),
            ('embedding_gradient_common_energy_fraction','Common-direction share of embedding-gradient energy')]):
            ax.plot([r['env_steps']/1000 for r in rr],[r[key] for r in rr],marker='o',label=f'Seed {seed}',color=colors[seed])
            ax.set(xlabel='Environment steps (thousands)',ylabel=label,xticks=[15,75])
    axes[0,0].set_yscale('log');axes[0,1].set_ylim(0,1.02)
    latest=[r for r in rows if r['env_steps']==75000]
    x=np.arange(4)
    for ax,key,pkey,title in [(axes[1,0],'centroid_loss_relative_change','permuted_loss_relative_change','Expected TD loss'),
                             (axes[1,1],'centroid_learnable_loss_relative_change','permuted_learnable_loss_relative_change','TD loss after removing target-head variance')]:
        ax.bar(x-.18,[100*r[key] for r in latest],.36,label='Replace all encodings with centroid')
        ax.bar(x+.18,[100*r[pkey] for r in latest],.36,label='Randomly permute actor encodings')
        ax.set(xticks=x,xticklabels=[str(r['seed']) for r in latest],xlabel='Seed at 75k',ylabel='Loss increase (%)',title=title,yscale='log')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Encoder TD supervision and frozen-target actor-conditioning interventions\n512 replay transitions/checkpoint; all ten shared target-head draws; gradients before Adam preconditioning')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'td_objective_diagnostics.{ext}',dpi=170)
    plt.close(fig)
    (out/'endpoint_comparison.json').write_text(json.dumps(latest,indent=2))
    print(json.dumps([{k:r[k] for k in ['seed','common_to_actor_centered_gradient_norm_ratio','centroid_loss_relative_change','centroid_learnable_loss_relative_change']} for r in latest],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPO/'artifacts/encoder_collapse_cause_20261006/td_signal')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--samples-per-rank',type=int,default=128)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.checkpoint:worker(a);return
    if a.summarize_only:summarize(a.output);return
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'raw').mkdir();(a.output/'logs').mkdir()
    root=Path('/workspace/alf_results/humanoid_walk/bafcv3_preln_actorln_seed0123_4g/20261005T041317Z')
    jobs=[root/f'seed_{seed}'/'train/algorithm'/f'ckpt-{step}' for seed in range(4) for step in (15015,75075)]
    def launch(cp):
        key=hashlib.sha256(str(cp).encode()).hexdigest()[:16]
        with (a.output/'logs'/f'{key}.log').open('w') as log:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(cp),'--result',str((a.output/'raw'/f'{key}.json').resolve()),'--samples-per-rank',str(a.samples_per_rank)],stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
        print('OK' if proc.returncode==0 else 'ERROR',cp,flush=True)
        return dict(checkpoint=str(cp),status='ok' if proc.returncode==0 else 'error',log=f'logs/{key}.log')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:completed=list(pool.map(launch,jobs))
    (a.output/'manifest.json').write_text(json.dumps(dict(checkpoints=completed,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    if any(r['status']!='ok' for r in completed):raise SystemExit('See failed logs')
    rows=[json.loads(p.read_text())['row'] for p in (a.output/'raw').glob('*.json')]
    rows.sort(key=lambda r:(r['seed'],r['env_steps']))
    with (a.output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summarize(a.output)


if __name__=='__main__':main()
