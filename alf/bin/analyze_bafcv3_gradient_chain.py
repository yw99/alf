"""Reconstruct BAFCv3 actor chain-rule terms from read-only saved checkpoints.

Runs no environments and performs no optimizer updates. Uses saved probe states
and reproducible rank-local replay samples; it cannot recover the failing batch.
Results retain loss reductions, both connected-feature surrogate branches, a
nonduplicated policy branch, backend comparisons, and historical event metrics.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import traceback
import concurrent.futures

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from alf.bin.evaluate_bafcv3_checkpoints import (
    load_model, normalize, preconfig, stable_seed, tensor_fingerprint, digest)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def number(x):
    x = float(x)
    return x if math.isfinite(x) else str(x)


def stats(x):
    x = x.detach().double()
    finite = torch.isfinite(x)
    return dict(shape=list(x.shape), numel=x.numel(),
                nonfinite=int((~finite).sum()),
                l2=number(x.norm()), rms=number(x.square().mean().sqrt()),
                max_abs=number(x.abs().max()), mean_abs=number(x.abs().mean()))


def vector(parts, params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for p, g in zip(params, parts)])


def vjp(output, inputs, cotangent):
    return torch.autograd.grad(output, inputs, grad_outputs=cotangent,
                               retain_graph=True, allow_unused=True)


def compare(a, b):
    a, b = a.detach().double(), b.detach().double()
    na, nb = a.norm(), b.norm()
    return dict(cosine=number((a @ b) / (na * nb).clamp_min(1e-300)),
                sum_ratio=number((a+b).norm()/(na+nb).clamp_min(1e-300)),
                relative_error=number((a-b).norm()/nb.clamp_min(1e-300)))


def sample_replay(state, sequence_count, seed):
    """ALF default stratified sampling, length two; chronological ring indexing."""
    s = {k.removeprefix('_replay_buffer.'):v for k,v in state.items()
         if k.startswith('_replay_buffer.')}
    sizes, positions = s['_current_size'], s['_current_pos']
    if sizes.numel() != 1:
        raise ValueError('This study expects one environment per DDP rank')
    if int(sizes[0]) < 2:
        raise ValueError('Too few saved replay observations')
    rng = torch.Generator().manual_seed(seed)
    fractions = (torch.arange(sequence_count) + torch.rand(sequence_count, generator=rng))/sequence_count
    starts = (fractions*(int(sizes[0])-1)).long()+int(positions[0])-int(sizes[0])
    obs = s['time_step|observation']
    idx = (starts[:,None]+torch.arange(2)) % obs.shape[1]
    # Training reshapes [time, batch, obs] to [time*batch, obs].
    sampled = obs[0,idx].transpose(0,1).reshape(-1,obs.shape[-1])
    types = s['time_step|step_type'][0,idx].transpose(0,1).reshape(-1)
    return sampled, types, dict(absolute_starts=starts.tolist(), ring_indices=idx.tolist(),
                                size=int(sizes[0]), position=int(positions[0]))


def diagnose(alg, obs, step_types, matching, backend):
    """Compute directional Jacobian products, never dense parameter Jacobians."""
    ctx = sdpa_kernel(SDPBackend.MATH) if backend == 'math' else contextlib.nullcontext()
    params = list(alg._actor_networks.parameters())
    names = [n for n,_ in alg._actor_networks.named_parameters()]
    stage = {}
    hooks = []
    layer = alg._actor_encoder._transformer.layers[0]
    def hook(name, inputs=False):
        def record(module, args, output):
            value = args[0] if inputs else output
            if isinstance(value, tuple): value = value[0]
            stage[name] = value
        return record
    for name,module in [('attention',layer.self_attn),('norm1',layer.norm1),
                        ('ff1',layer.linear1),('ff2',layer.linear2),('norm2',layer.norm2)]:
        hooks.append(module.register_forward_hook(hook(name+'_output')))
        if name in ('norm1','norm2'):
            hooks.append(module.register_forward_hook(hook(name+'_input', True)))
    try:
        with ctx:
            action = alg._actor_networks(obs)[0]
            eval_out = alg._actor_networks(alg._actor_eval_samples.detach(),full_neurons=True)[0][-2:]
            h, r = eval_out
            tokens = alg._tokenize_actor_out(eval_out)
            z = alg._actor_encoder(tokens)[0]
            k,n = matching.shape
            b,m = obs.shape[0], h.shape[0]
            enc = z[matching][:,None,:,:].expand(k,b,n,z.shape[-1]).reshape(k*b,n,z.shape[-1])
            matched_action = torch.gather(action[None].expand(k,*action.shape), 2,
                matching[:,None,:,None].expand(k,b,n,action.shape[-1]))
            critic_obs = obs[None,:,None,:].expand(k,b,n,obs.shape[-1]).reshape(k*b,n,obs.shape[-1])
            q = alg._critic_networks((enc,(critic_obs,matched_action.reshape(k*b,n,-1))))[0]
            objective = q.sum()/k
            labels = ['dqda','dQdz','dQdtokens','dqde_leaf_0','dqde_leaf_1']+list(stage)
            tensors = [action,z,tokens,h,r]+list(stage.values())
            gradients = torch.autograd.grad(objective,tensors,retain_graph=True,allow_unused=True)
            gd = dict(zip(labels,gradients))
            ga,gz,gt,gh,gr = gradients[:5]
            # The token tensor is the independent cut through the actor graph.
            direct_h, direct_r = gt.permute(2,0,1).split([h.shape[-1],r.shape[-1]],-1)
            via_r = vjp(r,[h],direct_r)[0]
            if via_r is None: via_r=torch.zeros_like(h)
            # Current original surrogate loss averaging: action over B; probes over M*N.
            action_weights = (step_types != 2).to(ga.dtype)[:,None,None]/b
            g_action = vector(vjp(action,params,ga*action_weights),params)
            g_hidden = vector(vjp(h,params,gh/(m*n)),params)
            g_output = vector(vjp(r,params,gr/(m*n)),params)
            g_policy_exact = vector(vjp(tokens,params,gt/(m*n)),params)
            g_duplicate = g_hidden+g_output-g_policy_exact
            # Compare raw objective chain-rule reconstruction without surrogate reweighting/masks.
            g_direct = vector(torch.autograd.grad(objective/b,params,retain_graph=True,allow_unused=True),params)
            g_chain = vector(vjp(action,params,ga/b),params)+g_policy_exact*(m*n/b)
            current_policy=g_hidden+g_output
            total=g_action+current_policy
            correct_total=g_action+g_policy_exact
            # ReLU gate identifies whether large dqde lives on inactive hidden units.
            hidden_active=gh*(h>0)
            terms = dict(dqda=ga,dQdz=gz,dQdtokens=gt,dqde_leaf_0=gh,dqde_leaf_1=gr,
                         hidden_direct=direct_h,hidden_via_output=via_r,
                         hidden_after_relu_gate=hidden_active,
                         actor_action=g_action,actor_hidden=g_hidden,actor_output=g_output,
                         actor_policy_current=current_policy,actor_policy_correct=g_policy_exact,
                         actor_duplicate=g_duplicate,actor_total_current=total,
                         actor_total_correct=correct_total,actor_objective_direct=g_direct)
            terms.update({'stage_grad/'+k:v for k,v in gd.items() if k in stage and v is not None})
            result={'terms':{k:stats(v) for k,v in terms.items()},
                    'activations':{k:stats(v) for k,v in dict(q=q,hidden=h,probe_action=r,embedding=z,**stage).items()},
                    'checks':{'chain_vs_direct':compare(g_chain,g_direct),
                              'hidden_chain':compare(direct_h.reshape(-1)+via_r.reshape(-1),gh.reshape(-1)),
                              'output_partial':compare(direct_r.reshape(-1),gr.reshape(-1))},
                    'balance':{'action_vs_policy':compare(g_action,current_policy),
                               'hidden_vs_output':compare(g_hidden,g_output),
                               'current_vs_correct':compare(total,correct_total)},
                    'gains':{'encoder':number(gt.double().norm()/gz.double().norm().clamp_min(1e-300)),
                             'hidden_to_parameters':number(g_hidden.double().norm()/(gh.double().norm()/(m*n)).clamp_min(1e-300)),
                             'action_to_parameters':number(g_action.double().norm()/(ga.double().norm()/b).clamp_min(1e-300)),
                             'relu_retained':number(hidden_active.double().norm()/gh.double().norm().clamp_min(1e-300))},
                    'batch':{'B':b,'M':m,'N':n,'K':k,'last_steps':int((step_types==2).sum()),
                             'action_reduction':1/b,'policy_reduction':1/(m*n)},
                    'per_actor_layer':{}}
            offset=0
            for name,p in zip(names,params):
                count=p.numel()
                result['per_actor_layer'][name]={key:stats(val[offset:offset+count]) for key,val in
                    [('action',g_action),('hidden',g_hidden),('output',g_output),('policy_correct',g_policy_exact),('total',total)]}
                offset+=count
            # A stable cotangent isolates the encoder from the critic signal.
            return result,{k:v.detach().cpu() for k,v in terms.items() if k in
                           ('dqda','dQdz','dqde_leaf_0','dqde_leaf_1','actor_total_current','actor_policy_current',
                            'actor_action','actor_hidden','actor_output','actor_policy_correct','actor_total_correct')}
    finally:
        for hook_handle in hooks:hook_handle.remove()


def worker(request):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=request['device']
    torch.cuda.set_device(device)
    import alf
    alf.summary.enable_summary(False)
    alg,normalizer,metadata=load_model({'config':request['config']},request['checkpoint'],device)
    alg.train()
    alg._actor_networks.requires_grad_(True)
    alg._actor_eval_samples.requires_grad_(False)
    before=(tensor_fingerprint(alg),tensor_fingerprint(normalizer))
    result={'status':'running','checkpoint':request['checkpoint'],'run_id':request['run_id'],
            'metadata':metadata,'device':device,'torch':torch.__version__,'settings':request,
            'samples':[],'input_hashes':{request['checkpoint']:digest(request['checkpoint'])}}
    result['metadata']['aggregate_env_steps']=metadata['env_steps']*4
    started=time.time()
    rank_sums={}
    for rank,shard in sorted(request['shards'].items()):
        result['input_hashes'][shard]=digest(shard)
        data=torch.load(shard,map_location='cpu',weights_only=True)['algorithm']
        for repeat in range(request['repetitions']):
            seed=stable_seed('chain-v1',request['run_id'],rank,repeat)
            obs,types,sampling=sample_replay(data,request['sequence_count'],seed)
            obs=normalize(normalizer,obs,device)
            types=types.to(device)
            torch.manual_seed(seed)
            matching=alg._sample_actor_critic_matchings(device)
            sample={'rank':int(rank),'repeat':repeat,'seed':seed,'sampling':sampling,
                    'matching':matching.cpu().tolist(),'backends':{}}
            tensors={}
            for backend in ['default','math']:
                torch.manual_seed(seed)
                diag,tensor=diagnose(alg,obs,types,matching,backend)
                sample['backends'][backend]=diag;tensors[backend]=tensor
                accumulator=rank_sums.setdefault((repeat,backend),{'count':0,'sums':{},'norm_sums':{}})
                accumulator['count']+=1
                for term,value in tensor.items():
                    if not term.startswith('actor_'):continue
                    accumulator['sums'][term]=accumulator['sums'].get(term,0)+value.double()
                    accumulator['norm_sums'][term]=accumulator['norm_sums'].get(term,0.)+float(value.double().norm())
            if int(rank)==0 and repeat==0:
                from unittest.mock import patch
                with sdpa_kernel(SDPBackend.MATH), patch.object(alg,'_sample_actor_critic_matchings',return_value=matching):
                    action=alg._actor_networks(obs)[0]
                    _,info=alg._actor_train_step(obs,action,action.detach()[:,0],(),())
                    original_loss=(info.loss*(types!=2)).mean()+info.extra.eval_action_loss.mean()
                    original_gradient=vector(torch.autograd.grad(original_loss,list(alg._actor_networks.parameters()),allow_unused=True),list(alg._actor_networks.parameters())).cpu()
                sample['original_implementation_check']=compare(tensors['math']['actor_total_current'],-original_gradient)
                if sample['original_implementation_check']['relative_error']>0.01:
                    raise AssertionError(sample['original_implementation_check'])
            sample['backend_comparison']={k:compare(tensors['default'][k].reshape(-1),tensors['math'][k].reshape(-1)) for k in tensors['default']}
            result['samples'].append(sample)
            save(request['output'],result)
            print(json.dumps({'checkpoint':request['checkpoint'],'rank':rank,'repeat':repeat,
                 'default_gh_max':sample['backends']['default']['terms']['dqde_leaf_0']['max_abs'],
                 'math_gh_max':sample['backends']['math']['terms']['dqde_leaf_0']['max_abs'],
                 'gh_relative_error':sample['backend_comparison']['dqde_leaf_0']['relative_error']}),flush=True)
            del tensors
    result['ddp_reconstructions']=[]
    for (repeat,backend),acc in rank_sums.items():
        means={k:v/acc['count'] for k,v in acc['sums'].items()}
        result['ddp_reconstructions'].append({'repeat':repeat,'backend':backend,'ranks':acc['count'],
            'terms':{k:stats(v) for k,v in means.items()},
            'rank_retained_ratio':{k:number(v.norm()/(acc['norm_sums'][k]/acc['count'])) for k,v in means.items()},
            'balance':compare(means['actor_action'],means['actor_policy_current'])})
    after=(tensor_fingerprint(alg),tensor_fingerprint(normalizer))
    result['state_unchanged']=before==after
    if before!=after:raise RuntimeError('Diagnostic changed model or normalizer state')
    result['status']='ok';result['seconds']=time.time()-started
    save(request['output'],result)


def inventory(base):
    runs=[]
    for task in ['humanoid_run','humanoid_walk']:
        for log in sorted((Path(base)/task).glob('bafcv3_single*/20*/seed_*/out.log')):
            errors=[l for l in log.read_text(errors='replace').splitlines() if 'MainProcess -' in l]
            if not errors:continue
            run=log.parent;hints=preconfig(run/'alf_config.py')
            group='actor_ln' if hints.get('bafcv3_actor_use_ln',False) else 'no_actor_ln'
            rid=f'{task}_{group}_{run.name}'
            cps=[]
            for cp in (run/'train/algorithm').glob('ckpt-*'):
                if not re.fullmatch(r'ckpt-\d+',cp.name):continue
                shards={re.search(r'rank(\d+)$',p.name)[1]:str(p) for p in cp.parent.glob(cp.name+'-replay_buffer-rank*')}
                cps.append({'path':str(cp),'step':int(cp.name[5:]),'shards':shards})
            runs.append({'id':rid,'path':str(run),'task':hints.get('create_environment.env_name'),
                         'seed':hints.get('TrainerConfig.random_seed'),'actor_ln':group=='actor_ln',
                         'failure':errors[-1],'checkpoints':sorted(cps,key=lambda c:c['step'])})
    return runs


def extract_events(runs,output):
    from tensorboard.compat.proto.event_pb2 import Event
    histories={}
    wanted=['Metrics/EnvironmentSteps','Metrics/AverageReturn',
            'actor_gradients/dqda_abs/mean','actor_gradients/dqda/value',
            'actor_gradients/dqde_leaf_0_abs/value','actor_gradients/dqde_leaf_0_abs/mean',
            'actor_gradients/dqde_leaf_1_abs/value','actor_gradients/dqde_leaf_0_l2_norm/mean',
            'loss/rl.actor.eval_action_loss']
    for run in runs:
        rows={}
        for file in sorted((Path(run['path'])/'train').glob('events.out*')):
            with file.open('rb') as f:
                while True:
                    header=f.read(12)
                    if len(header)!=12:break
                    n=struct.unpack('<Q',header[:8])[0];data=f.read(n);f.read(4)
                    if len(data)!=n:break
                    if not any(w.encode() in data for w in wanted):continue
                    e=Event.FromString(data)
                    for v in e.summary.value:
                        if v.tag not in wanted:continue
                        row=rows.setdefault(e.step,{'iteration':e.step,'wall_time':e.wall_time})
                        if v.HasField('simple_value'):row[v.tag]=number(v.simple_value)
                        elif v.HasField('histo'):row[v.tag]={'min':number(v.histo.min),'max':number(v.histo.max),'count':v.histo.num}
        histories[run['id']]=sorted(rows.values(),key=lambda r:r['iteration'])
    save(Path(output)/'historical_events.json',histories)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='/workspace/alf_results')
    p.add_argument('--output',default='artifacts/bafcv3_gradient_chain_20261003')
    p.add_argument('--gpus',nargs='+',type=int,default=[4,5,6,7])
    p.add_argument('--repetitions',type=int,default=2)
    p.add_argument('--sequence-count',type=int,default=64)
    p.add_argument('--worker-request')
    p.add_argument('--limit',type=int)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.worker_request:
        req=json.loads(Path(args.worker_request).read_text())
        try:worker(req)
        except Exception:
            save(req['output']+'.error.json',{'error':traceback.format_exc()})
            raise
        return
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    runs=inventory(args.source)
    manifest={'runs':runs,'command':sys.argv,'script_sha256':digest(__file__),
              'repetitions':args.repetitions,'sequence_count':args.sequence_count,
              'method':'Checkpoint reconstruction, not replay of the historical failing minibatch. Rank-0 checkpoint normalization is shared across replay shards.',
              'jobs':[]}
    for run in runs:
        for cp in run['checkpoints']:
            target=out/'checkpoints'/f'{run["id"]}_{cp["step"]}.json'
            req={'run_id':run['id'],'config':str(Path(run['path'])/'alf_config.py'),
                 'checkpoint':cp['path'],'shards':cp['shards'],'output':str(target),
                 'repetitions':args.repetitions,'sequence_count':args.sequence_count,
                 'script_sha256':manifest['script_sha256']}
            manifest['jobs'].append(req)
    jobs=manifest['jobs'][:args.limit] if args.limit else manifest['jobs']
    save(out/'manifest.json',manifest)
    extract_events(runs,out)
    def run_queue(gpu,queue):
        for req in queue:
            target=Path(req['output'])
            if args.resume and target.exists():
                old=json.loads(target.read_text())
                if old['status']=='ok' and old['settings']['script_sha256']==req['script_sha256'] and old['settings']['repetitions']==req['repetitions']:continue
            req={**req,'device':f'cuda:{gpu}'}
            request_path=out/'requests'/target.name
            save(request_path,req)
            log=out/'logs'/target.with_suffix('.log').name;log.parent.mkdir(exist_ok=True)
            with log.open('w') as stream:
                result=subprocess.run([sys.executable,'-m','alf.bin.analyze_bafcv3_gradient_chain','--worker-request',str(request_path)],stdout=stream,stderr=subprocess.STDOUT)
            print('DONE' if result.returncode==0 else 'ERROR',req['run_id'],Path(req['checkpoint']).name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(args.gpus)) as executor:
        futures=[executor.submit(run_queue,gpu,jobs[i::len(args.gpus)]) for i,gpu in enumerate(args.gpus)]
        for f in futures:f.result()


if __name__=='__main__':main()
