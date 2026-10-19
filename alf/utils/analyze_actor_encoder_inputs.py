"""Analyze native last_two actor-encoder inputs for humanoid BAFCv3 controls.

    .venv/bin/python alf/utils/analyze_actor_encoder_inputs.py

Uses checkpoint inventories from the earlier BAFCv3/Control A/Control B replay
studies. Measures saved-probe hidden activations, post-squash actions, their
concatenation, and the first token. Control A bypasses the encoder: its inputs
are reconstructed diagnostics, while its native output is its learned ID table.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BLOCKS = ('hidden', 'action', 'last_two', 'first_token', 'native_output')


def block_metrics(values):
    """Actor-first tensor geometry; RMS normalizes for different block widths."""
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    if len(x) != 10 or not np.isfinite(x).all():
        raise ValueError('Expected ten finite actor tensors')
    norms = np.linalg.norm(x, axis=1)
    l2 = np.zeros((10, 10))
    for i in range(10):
        for j in range(i + 1, 10):
            l2[i, j] = l2[j, i] = np.linalg.norm(x[i] - x[j])
    den = norms[:, None] * norms[None, :]
    cosine = np.clip(1 - np.divide(x @ x.T, den, out=np.full((10, 10), np.nan), where=den > 0), 0, 2)
    return dict(l2=l2, rms=l2 / np.sqrt(x.shape[1]), cosine=cosine,
                actor_value_rms=norms / np.sqrt(x.shape[1]),
                actor_zero_fraction=(x == 0).mean(axis=1))


def worker(args):
    sys.path.insert(0, str(REPO))
    import torch
    from alf.bin.evaluate_bafcv3_checkpoints import load_model, tensor_fingerprint, digest
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cp = args.checkpoint
    run = cp.parents[2]
    stat = cp.stat()
    with torch.inference_mode():
        alg, _, metadata = load_model({'config': str(run / 'alf_config.py')}, str(cp), 'cpu')
        arm = ('Control A' if alg._use_actor_id_encoding else
               'Control B' if alg._detach_actor_policy_input else 'BAFCv3')
        if arm != args.arm or alg._actor_eval_type != 'last_two':
            raise ValueError(f'Unexpected configuration: {arm}, {alg._actor_eval_type}')
        before = tensor_fingerprint(alg)
        probes = alg._actor_eval_samples
        full, _ = alg._actor_networks(probes, full_neurons=True)
        hidden, action = full[-2:]
        torch.testing.assert_close(action, alg._actor_networks(probes)[0], rtol=0, atol=0)
        tokens = alg._tokenize_actor_out([hidden, action])
        # Check against the actual module input, rather than assuming token layout.
        captured = []
        hook = alg._actor_encoder.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
        try:
            native_output, native_eval = alg._encode_actor_policies()
        finally:
            hook.remove()
        if arm == 'Control A':
            if captured:
                raise ValueError('Actor-ID path unexpectedly used the functional encoder')
            torch.testing.assert_close(native_output, alg._actor_id_embedding.weight, rtol=0, atol=0)
        else:
            if len(captured) != 1:
                raise ValueError('Expected exactly one functional encoder invocation')
            torch.testing.assert_close(captured[0], tokens, rtol=0, atol=0)
            for actual, expected in zip(native_eval, (hidden, action)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        h = hidden.permute(1, 2, 0).cpu().numpy()
        a = action.permute(1, 2, 0).cpu().numpy()
        t = tokens.cpu().numpy()
        arrays = dict(hidden=h, action=a, last_two=t, first_token=t[:, :1],
                      native_output=native_output.cpu().numpy())
        results = {f'{block}_{metric}': v for block, values in arrays.items()
                   for metric, v in block_metrics(values).items()}
        pair_indices = list(zip(*np.triu_indices(10, 1)))
        per_token = np.stack([np.sqrt(np.mean((t[i].astype(float)-t[j].astype(float))**2,axis=1))
                              for i,j in pair_indices],axis=1)
        if tensor_fingerprint(alg) != before:
            raise RuntimeError('Analysis changed model state')
        metadata = dict(arm=arm, task=metadata['task'], seed=metadata['seed'],env_steps=metadata['env_steps'],
            checkpoint=str(cp),checkpoint_sha256=digest(cp),actor_eval_type=alg._actor_eval_type,
            token_shape=list(t.shape),hidden_shape=list(h.shape),action_shape=list(a.shape),
            output_shape=list(native_output.shape),
            input_is_used_by_training=arm!='Control A',native_input_verified=arm!='Control A',
            actor_id_bypass_verified=arm=='Control A',state_unchanged=True,
            input_definition='last hidden layer activations + post-squash actions, before positional encoding',
            first_token_definition='first neuron of final hidden layer, evaluated across all saved probes',
            control_a_definition='reconstructed hypothetical functional input on unused saved probes; native output is learned actor-ID table',
            evaluation_probe_sha256=hashlib.sha256(probes.cpu().numpy().tobytes()).hexdigest(),
            token_sha256=hashlib.sha256(t.tobytes()).hexdigest(),
            config_sha256={str(p):digest(p) for p in [run/'alf_config.py',*sorted((run/'config_files').glob('*.py'))]})
        after = cp.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Checkpoint changed during analysis')
        np.savez_compressed(args.result, **results, per_token_pair_rms=per_token,metadata=json.dumps(metadata))


def summarize(out):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    records = {}
    rows, pairs = [], []
    ix=np.triu_indices(10,1)
    for path in sorted((out/'raw').glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']))
            key=(m['arm'],m['seed'],m['env_steps'])
            if key in records:raise ValueError(f'Duplicate record {key}')
            records[key]={k:d[k].copy() for k in d.files if k!='metadata'}
        r=records[key]
        for block in BLOCKS:
            row=dict(arm=m['arm'],seed=m['seed'],env_steps=m['env_steps'],block=block,
                     pair_rms_mean=float(r[f'{block}_rms'][ix].mean()),
                     pair_rms_min=float(r[f'{block}_rms'][ix].min()),pair_rms_max=float(r[f'{block}_rms'][ix].max()),
                     pair_l2_mean=float(r[f'{block}_l2'][ix].mean()),
                     pair_cosine_mean=float(np.nanmean(r[f'{block}_cosine'][ix])),
                     undefined_cosine_pairs=int(np.isnan(r[f'{block}_cosine'][ix]).sum()),
                     actor_value_rms_mean=float(r[f'{block}_actor_value_rms'].mean()),
                     actor_zero_fraction_mean=float(r[f'{block}_actor_zero_fraction'].mean()))
            rows.append(row)
            for i,j in zip(*ix):
                pairs.append(dict(arm=m['arm'],seed=m['seed'],env_steps=m['env_steps'],block=block,
                    actor_i=int(i),actor_j=int(j),rms=float(r[f'{block}_rms'][i,j]),
                    l2=float(r[f'{block}_l2'][i,j]),cosine=float(r[f'{block}_cosine'][i,j])))
    if not records:raise ValueError('No successful checkpoints')
    colors={'BAFCv3':'tab:blue','Control A':'tab:orange','Control B':'tab:green'}
    arms=[a for a in colors if any(k[0]==a for k in records)]
    common=sorted(set.intersection(*[{k[2] for k in records if k[:2]==(arm,seed)} for arm in arms for seed in range(4)]))
    if not common:raise ValueError('No shared four-seed checkpoint steps')
    # Keep each component visible; combining hidden and action coordinates changes weighting.
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    specs=[('hidden','rms','Last hidden activation: pair RMS'),('action','rms','Post-squash action: pair RMS'),
           ('last_two','rms','Combined last_two: pair RMS'),('first_token','rms','First token: pair RMS'),
           ('last_two','cosine','Combined last_two: cosine distance'),('native_output','rms','Native output: pair RMS')]
    detail,daxes=plt.subplots(4,4,figsize=(17,12),sharex=True)
    for arm in arms:
        for ax,(block,metric,label) in zip(axes.flat,specs):
            values=np.array([[np.nanmean(records[(arm,seed,s)][f'{block}_{metric}'][ix]) for s in common] for seed in range(4)])
            ax.plot(common,values.mean(axis=0),color=colors[arm],label=arm)
            ax.fill_between(common,values.min(axis=0),values.max(axis=0),color=colors[arm],alpha=.13)
            ax.set(ylabel=label)
            if (block,metric)==('native_output','rms'):ax.set_yscale('symlog',linthresh=1e-14)
        for seed in range(4):
            steps=sorted(k[2] for k in records if k[:2]==(arm,seed))
            for col,block in enumerate(('hidden','action','last_two','native_output')):
                values=np.array([records[(arm,seed,s)][f'{block}_rms'][ix] for s in steps])
                ax=daxes[seed,col]
                ax.plot(steps,values.mean(axis=1),color=colors[arm],label=arm)
                ax.fill_between(steps,values.min(axis=1),values.max(axis=1),color=colors[arm],alpha=.13)
                ax.set(title=f'Seed {seed}: {block}',ylabel='Pair RMS')
                if block=='native_output':ax.set_yscale('symlog',linthresh=1e-14)
    for ax in [*axes.flat,*daxes.flat]:
        ax.set(xlabel='Environment steps')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('humanoid:run encoder inputs on saved probes — 4-seed mean; bands: range of seed means\nControl A inputs are hypothetical; its native output is the actor-ID embedding')
    detail.suptitle('All saved checkpoints; 45-pair means and pair ranges\nControl A bypasses the functional encoder')
    for f,name in ((fig,'overview'),(detail,'per_seed')):
        f.tight_layout()
        for ext in ('png','pdf'):f.savefig(out/f'{name}.{ext}',dpi=170)
        plt.close(f)
    for name,data in (('summary',rows),('pairs',pairs)):
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    endpoint=[r for r in rows if r['env_steps']==common[-1]]
    (out/'matched_endpoint.json').write_text(json.dumps(dict(env_steps=common[-1],rows=endpoint),indent=2,allow_nan=True))
    print('Common checkpoints:',common)
    for arm in arms:
        print(arm,{b:np.mean([r['pair_rms_mean'] for r in endpoint if r['arm']==arm and r['block']==b]) for b in BLOCKS},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/humanoid_actor_encoder_inputs'))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--checkpoint',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--arm',choices=('BAFCv3','Control A','Control B'),help=argparse.SUPPRESS)
    parser.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.checkpoint:worker(args);return
    if args.summarize_only:summarize(args.output);return
    if args.workers<1:parser.error('--workers must be positive')
    sources={'BAFCv3':REPO/'artifacts/humanoid_run_bafcv3_replay_and_encodings/manifest.json',
             'Control A':REPO/'artifacts/control_a_replay_action_rms/manifest.json',
             'Control B':REPO/'artifacts/control_b_replay_and_encodings/manifest.json'}
    jobs=[]
    for arm,path in sources.items():
        manifest=json.loads(path.read_text())
        if any(r['status']!='ok' for r in manifest['checkpoints']):raise ValueError(f'Incomplete source study: {path}')
        jobs.extend((arm,cp['path']) for cp in manifest['checkpoints'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'raw').mkdir();(args.output/'logs').mkdir()
    manifest=dict(input_studies={a:str(p) for a,p in sources.items()},checkpoints=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol='Saved-probe last_two inputs, before positional encoding; separate hidden/action RMS. Control A input is hypothetical.')
    def launch(job):
        arm,cp=job;key=hashlib.sha256(cp.encode()).hexdigest()[:16]
        with (args.output/'logs'/f'{key}.log').open('w') as log:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',cp,'--arm',arm,
                '--result',str((args.output/'raw'/f'{key}.npz').resolve())],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        print('OK' if p.returncode==0 else 'ERROR',arm,cp,flush=True)
        return dict(arm=arm,path=cp,status='ok' if p.returncode==0 else 'error',log=f'logs/{key}.log')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(launch,jobs):
            manifest['checkpoints'].append(r)
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summarize(args.output)
    if any(r['status']!='ok' for r in manifest['checkpoints']):raise SystemExit('Incomplete analysis: see logs')


if __name__=='__main__':main()
