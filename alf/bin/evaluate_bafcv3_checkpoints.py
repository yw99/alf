"""Reproducible, environment-free BAFCv3 checkpoint diagnostics.

Example::

    python -m alf.bin.evaluate_bafcv3_checkpoints --device auto

Sources are read-only. Results are incremental and can be resumed with
``--output DIR --resume``. A separate process loads each checkpoint's saved
configuration. See the generated report for the interpretation of the proxies.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import glob
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

VERSION = 3
THRESHOLDS = [2, 10, 20, 30, 50, 100, 200]
REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(tmp, 'wt', encoding='utf8') as f:
        json.dump(value, f, allow_nan=False, sort_keys=True)
    os.replace(tmp, path)


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return json.load(f)


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def source_identity():
    paths = [Path(__file__), REPO / 'alf/algorithms/bafc_algorithm_v3.py',
             REPO / 'alf/algorithms/bafc_algorithm_v3_tr2.py',
             REPO / 'alf/algorithms/data_transformer.py',
             REPO / 'alf/utils/value_ops.py']
    hashes = {str(p.relative_to(REPO)): digest(p) for p in paths}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    return {'version': VERSION, 'git_commit': git, 'files': hashes}


def preconfig(path):
    """Read task/seed hints without executing a saved configuration."""
    result = {}
    try:
        for node in ast.walk(ast.parse(Path(path).read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'pre_config' and node.args:
                    try:
                        result.update(ast.literal_eval(node.args[0]))
                    except (ValueError, TypeError):
                        pass
    except (OSError, SyntaxError):
        pass
    return result


def discover(pattern, tasks=(), seeds=(), steps=()):
    runs = []
    for root in sorted(glob.glob(pattern)):
        root = Path(root)
        candidates = {p.parent for p in root.rglob('alf_config.py')}
        # Empty directories are important evidence of incomplete server copies.
        candidates.update(p for p in root.rglob('*') if p.is_dir()
                          and re.search(r'bafcv3(?![0-9])', p.name, re.I)
                          and not re.search(r'bafcv3[_-]?tr', p.name, re.I))
        for run in sorted(candidates):
            conf = run / 'alf_config.py'
            texts = []
            for p in [conf] + sorted((run / 'config_files').glob('*.py')):
                if p.is_file():
                    texts.append(p.read_text(errors='replace'))
            text = '\n'.join(texts)
            is_v3 = bool(re.search(r'\bBafcAlgorithmV3\b', text))
            if text and not is_v3:
                continue
            hints = preconfig(conf)
            task = hints.get('create_environment.env_name')
            seed = hints.get('TrainerConfig.random_seed')
            if tasks and task not in tasks:
                continue
            if seeds and seed not in seeds:
                continue
            models = sorted((run / 'train/algorithm').glob('ckpt-*'))
            models = [p for p in models if re.fullmatch(r'ckpt-\d+', p.name)
                      and (not steps or int(p.name[5:]) in steps)]
            checkpoints = []
            for model in sorted(models, key=lambda p: int(p.name[5:])):
                shards = {int(p.name.rsplit('rank', 1)[1]): str(p)
                          for p in model.parent.glob(model.name + '-replay_buffer-rank*')
                          if re.search(r'rank\d+$', p.name)}
                legacy = Path(str(model) + '-replay_buffer')
                if not shards and legacy.exists():
                    shards = {0: str(legacy)}
                checkpoints.append({'path': str(model), 'step': int(model.name[5:]),
                                    'shards': shards, 'legacy_only': bool(shards) and
                                    all('rank' not in p for p in shards.values())})
            runs.append({'path': str(run), 'config': str(conf), 'task_hint': task,
                         'seed_hint': seed, 'checkpoints': checkpoints,
                         'status': 'discovered' if checkpoints else 'pending_no_checkpoints'})
    return runs


def chronological_replay(state):
    """Return one chronological tensor dictionary per environment."""
    prefix = '_replay_buffer.'
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    sizes, positions = state['_current_size'], state['_current_pos']
    fields = {k: v for k, v in state.items() if not k.startswith('_')}
    capacity = next(iter(fields.values())).shape[1]
    result = []
    for env in range(sizes.numel()):
        size = int(sizes[env])
        if size < 0 or size > capacity:
            raise ValueError('Invalid replay size')
        indices = (torch.arange(size) + int(positions[env]) - size) % capacity
        result.append({k: v[env, indices] for k, v in fields.items()})
    return result


def replay_windows(episodes, horizon):
    """Valid starts and endpoints, never crossing reset or ring boundaries."""
    result = []
    for env, data in enumerate(episodes):
        types = data['time_step|step_type'].tolist()
        n = len(types)
        for start in range(n - 1):
            if types[start] == 2 or types[start + 1] == 0:
                continue
            end = start
            for j in range(start + 1, min(n, start + horizon + 1)):
                if types[j] == 0:
                    break
                end = j
                if types[j] == 2:
                    break
            if end - start == horizon or types[end] == 2:
                result.append((env, start, end))
    return result


def accumulated_rewards(data, start, end, gamma):
    total, factor = 0., 1.
    for j in range(start + 1, end + 1):
        total += factor * float(data['time_step|reward'][j])
        factor *= gamma * float(data['time_step|discount'][j])
    return total, factor


class TrustAdapter:
    """Delegate the metric to TR2 without constructing or training TR2."""
    def __init__(self, algorithm):
        from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2
        for name in ('_ensure_group_action', '_critic_feature_head_index',
                     '_compute_snapshot_feature_map', '_compute_feature_inv_cov',
                     '_compute_weighted_feature_norm', '_compute_eval_trust_from_features',
                     '_extract_eval_action', '_compute_actor_encoding'):
            setattr(self, name, getattr(BafcAlgorithmV3TR2, name).__get__(self))
        for name in ('_actor_networks', '_actor_eval_samples', '_actor_eval_type',
                     '_actor_encoder', '_num_actor_critic', '_tokenize_actor_out'):
            setattr(self, name, getattr(algorithm, name))
        self._snapshot_critic_networks = algorithm._critic_networks
        self._trust_cov_reg = 1e-4


def tensor_fingerprint(module):
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        if isinstance(value, torch.Tensor):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def variant_configuration(settings):
    """Keep training/configuration variants separate while excluding seed/path."""
    selectors = ('BafcAlgorithmV3.', 'TransformerEncoder.', 'OneStepTDLoss.',
                 'ObservationNormalizer.', 'Agent.', '.optimizers.', 'TrainerConfig.')
    excluded = ('TrainerConfig.random_seed', 'TrainerConfig.root_dir')
    return {k: re.sub(r' at 0x[0-9a-fA-F]+', '', repr(v))
            for k, v in settings.items()
            if any(tag in k for tag in selectors)
            and not any(k.endswith(tag) for tag in excluded)}


def load_model(run, model, device):
    import alf
    from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3
    from alf.algorithms.config import TrainerConfig
    from alf.algorithms.data_transformer import create_data_transformer
    from alf.tensor_specs import TensorSpec, BoundedTensorSpec
    from alf.utils.common import parse_conf_file
    parse_conf_file(run['config'], create_env=False)
    if alf.get_config_value('Agent.rl_algorithm_cls') is not BafcAlgorithmV3:
        raise ValueError('Saved algorithm is not BAFCv3')
    task = alf.get_config_value('create_environment.env_name')
    if ':' not in task:
        raise ValueError('Only saved vector DM Control tasks are supported')
    data = torch.load(model, map_location='cpu', weights_only=True)
    state = data['algorithm']
    s = {k.removeprefix('_rl_algorithm.'): v for k, v in state.items()
         if k.startswith('_rl_algorithm.')}
    observation_dim = s['_actor_eval_samples'].shape[-1]
    action_dim = s['_actor_networks._action_layer._bias'].shape[-1]
    config = TrainerConfig(root_dir=str(Path(model).parent))
    obs_spec = TensorSpec((observation_dim,))
    # ALF's vector DM Control actor actions use the suite's [-1, 1] bounds.
    alg = BafcAlgorithmV3(observation_spec=obs_spec,
                         action_spec=BoundedTensorSpec((action_dim,), minimum=-1., maximum=1.),
                         config=config)
    if alg._eval_samples_source != 'trainable':
        raise ValueError('Replay-sourced actor encodings need a separate diagnostic protocol')
    incompatible = alg.load_state_dict(s)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(str(incompatible))
    normalizer = create_data_transformer(config.data_transformer_ctor, obs_spec)
    ns = {k.removeprefix('_data_transformer.'): v for k, v in state.items()
          if k.startswith('_data_transformer.')}
    incompatible = normalizer.load_state_dict(ns)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(str(incompatible))
    from alf.algorithms.data_transformer import ObservationNormalizer
    if not isinstance(normalizer, ObservationNormalizer):
        raise ValueError('Unsupported transformer: explicitly implement its frozen replay semantics')
    alg.to(device).eval()
    normalizer.eval()  # Keep non-buffer ALF averaging constants on CPU.
    settings = dict(alf.get_operative_configs() + alf.get_inoperative_configs())
    variant_settings = variant_configuration(settings)
    variant = hashlib.sha256(json.dumps(variant_settings, sort_keys=True).encode()).hexdigest()[:12]
    metadata = {'task': task, 'seed': int(config.random_seed), 'variant': variant,
                'config_values': jsonable(settings), 'variant_config': variant_settings,
                'observation_dim': observation_dim, 'action_dim': action_dim,
                'action_bounds': [-1, 1], 'global_step': int(data['global_step']),
                'env_steps': int(data['trainer_progress']['_env_steps']),
                'trainer_progress': jsonable(data['trainer_progress']),
                'aggregate_env_steps': None,
                'gamma': float(alg._critic_losses[0].gamma),
                'reconstruction': 'Current actor/current critic used as reference/snapshot; not historical TR2 state'}
    return alg, normalizer, metadata


def normalize(normalizer, obs, device):
    return normalizer._normalizer.normalize(obs.cpu(), normalizer._clipping).to(device)


def features(adapter, encoding, normalizer, obs, action, device, batch_size,
             normalized=False, target=False):
    out = []
    for start in range(0, len(obs), batch_size):
        x = obs[start:start + batch_size].to(device)
        if not normalized:
            x = normalize(normalizer, x, device)
        a = adapter._actor_networks(x)[0] if target else action[start:start + batch_size].to(device)
        out.append(adapter._compute_snapshot_feature_map(x, encoding, a))
    return torch.cat(out)


def covariance_scores(adapter, behavior, heldout, targets, ridges):
    """Float32 TR2 computation plus independent double-precision diagnostics."""
    by_group = behavior.permute(1, 0, 2)
    cov32 = by_group.transpose(1, 2) @ by_group / behavior.shape[0]
    b64 = by_group.double().cpu()
    cov64 = b64.transpose(1, 2) @ b64 / behavior.shape[0]
    if not torch.isfinite(cov64).all():
        raise ValueError('Nonfinite behavior covariance; check replay and network outputs')
    solver = 'torch.linalg.eigh:L'
    try:
        eigen, basis = torch.linalg.eigh(cov64)
    except torch.linalg.LinAlgError:
        # Both triangles represent the same finite symmetric covariance.
        # A different LAPACK path can resolve convergence failures near rank
        # deficiency without changing the ridge or discarding observations.
        eigen, basis = torch.linalg.eigh(cov64, UPLO='U')
        solver = 'torch.linalg.eigh:U_after_L_failure'
    eigen = eigen.clamp_min(0)
    all_features = {'in_sample': behavior, 'heldout': heldout, **targets}
    # E[(phi dot eigenvector)^2] allows all ridge values to share one eigensolve.
    projected = {k: (v.permute(1, 0, 2).double().cpu() @ basis).square().mean(1)
                 for k, v in all_features.items()}
    rows = []
    for ridge in ridges:
        adapter._trust_cov_reg = ridge
        inv32 = adapter._compute_feature_inv_cov(behavior)
        scores = {k: adapter._compute_weighted_feature_norm(v, inv32).square().mean(0)
                  for k, v in all_features.items()}
        scores64 = {k: (p / (eigen + ridge)).sum(-1) for k, p in projected.items()}
        relative_error = {k: ((scores[k].double().cpu() - scores64[k]).abs() /
                              scores64[k].abs().clamp_min(1e-12)).tolist()
                          for k in scores}
        row = {'ridge': ridge, 'covariance_solver': solver,
               'scores': jsonable(scores), 'scores_float64': jsonable(scores64),
               'relative_float32_error': relative_error,
               'effective_dimension': jsonable((eigen / (eigen + ridge)).sum(-1)),
               'regularized_condition': jsonable((eigen[:, -1] + ridge) / (eigen[:, 0] + ridge)),
               'inverse_residual_max': float(((cov32 + ridge * torch.eye(cov32.shape[-1], device=cov32.device)) @ inv32 -
                                               torch.eye(cov32.shape[-1], device=cov32.device)).abs().max())}
        rows.append(row)
    return rows, eigen.tolist()


def residual_summary(error):
    a = error.double().cpu()
    flat = a.flatten()
    return {'count': len(a), 'mae': float(flat.abs().mean()),
            'rmse': float(flat.square().mean().sqrt()), 'bias': float(flat.mean()),
            'abs_quantiles': torch.quantile(flat.abs(), torch.tensor([.5, .9, .99], dtype=torch.float64)).tolist(),
            'per_actor_critic_mae': a.abs().mean(0).tolist(),
            'per_actor_critic_rmse': a.square().mean(0).sqrt().tolist(),
            'per_actor_critic_bias': a.mean(0).tolist()}


def q_values(alg, encoding, obs, action, target=False):
    groups = alg._num_actor_critic
    actor_encoding = encoding.repeat(len(obs), 1)
    critic_obs = obs.repeat_interleave(groups, dim=0)
    if target:
        critic_action = alg._actor_networks(obs)[0].reshape(-1, action.shape[-1])
        network = alg._target_critic_networks
    else:
        critic_action = action.repeat_interleave(groups, dim=0)
        network = alg._critic_networks
    q = network((actor_encoding, (critic_obs, critic_action)))[0]
    q = q.reshape(len(obs), groups, groups)
    if target:
        q = alg._select_critic_targets(q)
        if q.ndim == 2:
            q = q.unsqueeze(-1).expand(-1, -1, groups)
    return q


def critic_proxies(alg, encoding, normalizer, episodes, options, device, seed):
    output = {'interpretation': 'Replay consistency residuals, not held-out policy-value accuracy'}
    for horizon, label in [(1, 'td'), (32, 'replay_32_step')]:
        candidates = replay_windows(episodes, horizon)
        rng = np.random.default_rng(stable_seed(seed, label))
        indices = rng.choice(len(candidates), min(options['critic_samples'], len(candidates)), replace=False)
        chosen = [candidates[i] for i in indices]
        if not chosen:
            output[label] = {'status': 'unavailable_no_valid_sequences'}
            continue
        errors, disagreement, q_scales = [], [], []
        counts = {'terminated': 0, 'time_limit': 0, 'horizon': 0}
        torch.manual_seed(stable_seed(seed, label, 'target_critic'))
        for offset in range(0, len(chosen), options['batch_size']):
            items = chosen[offset:offset + options['batch_size']]
            obs = torch.stack([episodes[e]['time_step|observation'][s] for e, s, t in items])
            act = torch.stack([episodes[e]['action'][s] for e, s, t in items]).to(device)
            nxt = torch.stack([episodes[e]['time_step|observation'][t] for e, s, t in items])
            accumulated = [accumulated_rewards(episodes[e], s, t, float(alg._critic_losses[0].gamma))
                           for e, s, t in items]
            rewards, factors = torch.tensor(accumulated, device=device).unbind(-1)
            q = q_values(alg, encoding, normalize(normalizer, obs, device), act)
            target = q_values(alg, encoding, normalize(normalizer, nxt, device), act, target=True)
            y = rewards[:, None, None] + factors[:, None, None] * target
            errors.append((y - q).cpu())
            disagreement.append(q.std(-1, unbiased=False).cpu())
            q_scales.append(q.cpu())
            for e, s, t in items:
                data = episodes[e]
                ended = int(data['time_step|step_type'][t]) == 2
                counts['terminated' if ended and float(data['time_step|discount'][t]) == 0
                       else 'time_limit' if ended else 'horizon'] += 1
        summary = residual_summary(torch.cat(errors))
        summary.update({'status': 'ok', 'endpoint_counts': counts,
                        'critic_disagreement_mean': float(torch.cat(disagreement).mean()),
                        'q_rms': float(torch.cat(q_scales).square().mean().sqrt()),
                        'sample_index_hash': hashlib.sha256(repr(chosen).encode()).hexdigest()})
        output[label] = summary
    return output


def evaluate_rank(alg, normalizer, state, rank, options, device, job_seed):
    episodes = chronological_replay(state)
    adapter = TrustAdapter(alg)
    encoding = adapter._compute_actor_encoding(alg._actor_networks).detach()
    obs = torch.cat([d['time_step|observation'] for d in episodes])
    action = torch.cat([d['action'] for d in episodes])
    recent_per_env = (512 + len(episodes) - 1) // len(episodes)
    recent = torch.cat([d['time_step|observation'][-recent_per_env:] for d in episodes])[-512:]
    if not len(obs):
        raise ValueError('Empty replay shard')
    pools = {'reconstructed_cache': features(adapter, encoding, normalizer, recent, None, device,
                                             options['batch_size'], target=True)}
    saved = alg._reweighting_target_observation_cache
    if rank == 0 and isinstance(saved, torch.Tensor) and len(saved):
        pools['saved_cache'] = features(adapter, encoding, normalizer, saved, None, device,
                                       options['batch_size'], normalized=True, target=True)
    eligible = [n for n in options['sample_counts'] if len(obs) >= 2 * n]
    output = {'rank': rank, 'replay_size': len(obs), 'num_environments': len(episodes),
              'normalizer_provenance': 'Shared checkpoint normalizer, not recovered historical rank-local state',
              'protocols': list(pools), 'unavailable_sample_counts': [n for n in options['sample_counts'] if n not in eligible],
              'coverage': [], 'spectra': []}
    rng = np.random.default_rng(job_seed)
    for rep in range(options['repetitions']):
        if not eligible:
            break
        largest = max(eligible)
        sample = rng.choice(len(obs), 2 * largest, replace=False)
        phi = features(adapter, encoding, normalizer, obs[sample], action[sample], device,
                       options['batch_size'])
        targets = {k: v[torch.as_tensor(rng.choice(len(v), min(128, len(v)), replace=False), device=device)]
                   for k, v in pools.items()}
        for n in eligible:
            rows, eigen = covariance_scores(adapter, phi[:n], phi[largest:largest + n], targets, options['ridges'])
            output['spectra'].append({'repetition': rep, 'n': n, 'eigenvalues': eigen})
            for row in rows:
                row.update({'repetition': rep, 'n': n})
                output['coverage'].append(row)
    output['critic_proxies'] = critic_proxies(alg, encoding, normalizer, episodes, options, device, job_seed)
    return output


def input_paths(run, checkpoint):
    paths = [Path(run['config']), Path(checkpoint['path'])]
    paths += sorted((Path(run['path']) / 'config_files').glob('*.py'))
    paths += [Path(p) for p in checkpoint['shards'].values()]
    return sorted(set(paths))


def fingerprints(paths):
    out = {}
    for p in paths:
        before = p.stat()
        sha = digest(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Input is still changing: {p}')
        out[str(p)] = {'size': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': sha}
    return out


def unchanged(inputs):
    return all(Path(p).exists() and Path(p).stat().st_size == v['size']
               and Path(p).stat().st_mtime_ns == v['mtime_ns'] for p, v in inputs.items())


def cache_matches(result, inputs, options, code):
    return result.get('inputs') == inputs and result.get('options') == options and result.get('code') == code


def worker(request):
    run, cp, options = request['run'], request['checkpoint'], request['options']
    torch.set_num_threads(options['cpu_threads'])
    device = options['device']
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = fingerprints(input_paths(run, cp))
    target = Path(request['result'])
    if request['resume'] and target.exists():
        prior = read_json(target)
        if prior.get('status') == 'ok' and cache_matches(prior, inputs, options, request['code']):
            print('RESUMED', cp['path'], flush=True)
            return
    started = time.time()
    with torch.no_grad():
        alg, normalizer, metadata = load_model(run, cp['path'], device)
        before = [tensor_fingerprint(alg), tensor_fingerprint(normalizer)]
        output = {'status': 'ok', 'run': run['path'], 'checkpoint': cp['path'], 'metadata': metadata,
                  'inputs': inputs, 'options': options, 'code': request['code'], 'device_used': device,
                  'ranks': [], 'rank_errors': {}, 'dependencies': {'torch': torch.__version__, 'numpy': np.__version__,
                  'python': sys.version}, 'device_name': torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU'}
        for rank, path in sorted(cp['shards'].items(), key=lambda kv: int(kv[0])):
            rank = int(rank)
            print('RANK_START', cp['path'], rank, flush=True)
            try:
                state = torch.load(path, map_location='cpu', weights_only=True)['algorithm']
                result = evaluate_rank(alg, normalizer, state, rank, options, device,
                                       stable_seed(options['seed'], run['path'], cp['step'], rank))
                output['ranks'].append(result)
                del state
            except Exception as exc:
                output['rank_errors'][str(rank)] = {'error': str(exc), 'traceback': traceback.format_exc()}
                print(traceback.format_exc(), flush=True)
            print('RANK_DONE', cp['path'], rank, flush=True)
        if not output['ranks']:
            output['status'] = 'error'
        elif output['rank_errors']:
            output['status'] = 'partial'
        after = [tensor_fingerprint(alg), tensor_fingerprint(normalizer)]
        if before != after:
            raise RuntimeError('Evaluation mutated model or normalizer state')
        if not unchanged(inputs):
            raise RuntimeError('Inputs changed during evaluation; result is not committed')
        output['state_unchanged'] = True
        output['duration_seconds'] = time.time() - started
        atomic_json(target, output)
        print('CHECKPOINT_DONE', cp['path'], round(output['duration_seconds'], 1), output['status'], flush=True)


def read_curves(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    curves = {}
    wanted = ['Metrics_vs_EnvironmentSteps/AverageReturn', 'Metrics/AverageReturn',
              'Metrics/EnvironmentSteps', 'BafcAlgorithmV3TR2/eval_trust_metric',
              'BafcAlgorithmV3TR2/eval_trust_metric/effective',
              'BafcAlgorithmV3TR2/rollout_skip_due_eval_gate_count',
              'BafcAlgorithmV3TR2/rollout_opportunity_count']
    for kind in ('train', 'eval'):
        folder = Path(run) / kind
        if not list(folder.glob('events*')):
            continue
        try:
            events = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
            for tag in wanted:
                if tag not in events.Tags().get('scalars', []):
                    continue
                by_step = {}
                for event in sorted(events.Scalars(tag), key=lambda x: x.wall_time):
                    by_step[event.step] = [event.step, event.value, event.wall_time]
                curves[kind + '/' + tag] = [by_step[s] for s in sorted(by_step)]
        except Exception as exc:
            curves[kind + '/error'] = str(exc)
    return curves


def near_return(curves, step):
    values = curves.get('train/Metrics_vs_EnvironmentSteps/AverageReturn', [])
    if not values or step < values[0][0] or step > values[-1][0] + 1000:
        return None, None
    # Nearest logged measurement only; never synthesize an interpolated return.
    x = min(values, key=lambda x: abs(x[0] - step))
    if abs(x[0] - step) > 1000:
        return None, None
    return x[1], x[0]


def summary_rows(result, curves):
    rows = []
    meta = result['metadata']
    ret, ret_step = near_return(curves, meta['env_steps'])
    for rank in result['ranks']:
        groups = {}
        for entry in rank['coverage']:
            for protocol in rank['protocols']:
                groups.setdefault((entry['n'], entry['ridge'], protocol), []).append(entry)
        for (n, ridge, protocol), entries in groups.items():
            scores = np.array([np.mean(e['scores'][protocol]) for e in entries])
            held = np.array([np.mean(e['scores']['heldout']) for e in entries])
            effective = np.array([np.mean(e['effective_dimension']) for e in entries])
            td = rank['critic_proxies'].get('td', {})
            multi = rank['critic_proxies'].get('replay_32_step', {})
            row = {'run': result['run'], 'task': meta['task'], 'seed': meta['seed'], 'variant': meta['variant'],
                   'checkpoint': Path(result['checkpoint']).name, 'env_steps': meta['env_steps'],
                   'global_step': meta['global_step'], 'rank': rank['rank'], 'protocol': protocol,
                   'n': n, 'ridge': ridge, 'repetitions': len(entries),
                   'target_mean': float(scores.mean()), 'target_sd': float(scores.std(ddof=1)) if len(scores)>1 else 0.,
                   'target_median': float(np.median(scores)), 'target_p05': float(np.quantile(scores, .05)),
                   'target_p95': float(np.quantile(scores, .95)), 'heldout_mean': float(held.mean()),
                   'effective_dim_mean': float(effective.mean()),
                   'target_heldout_ratio': float(np.mean(scores / np.maximum(held, 1e-12))),
                   'float32_relative_error_max': max(max(e['relative_float32_error'][protocol]) for e in entries),
                   'td_rmse': td.get('rmse'), 'td_mae': td.get('mae'), 'td_bias': td.get('bias'),
                   'q_rms': td.get('q_rms'), 'replay32_rmse': multi.get('rmse'),
                   'training_return': ret, 'return_logged_env_step': ret_step}
            for threshold in THRESHOLDS:
                row['pass_' + str(threshold)] = float(np.mean(scores <= threshold))
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cross_rank(result, expected):
    if not expected or set(expected) != {r['rank'] for r in result['ranks']}:
        return []
    groups = {}
    for rank in result['ranks']:
        for row in rank['coverage']:
            key = row['repetition'], row['n'], row['ridge']
            groups.setdefault(key, []).append(np.mean(row['scores']['reconstructed_cache']))
    out = []
    for (rep, n, ridge), values in groups.items():
        if len(values) != len(expected):
            continue
        out.append({'run': result['run'], 'checkpoint': Path(result['checkpoint']).name,
                    'env_steps': result['metadata']['env_steps'], 'repetition': rep, 'n': n, 'ridge': ridge,
                    'rank_min': min(values), 'rank_mean': float(np.mean(values)), 'rank_max': max(values),
                    **{'all_ranks_pass_' + str(t): int(max(values) <= t) for t in THRESHOLDS}})
    return out



def tr2_skip_windows(curves_by_run):
    """Counter deltas aligned on global step; never infer skips from trust alone."""
    rows = []
    base = 'train/BafcAlgorithmV3TR2/'
    for run, curves in curves_by_run.items():
        skips = {int(v[0]): v[1] for v in curves.get(base+'rollout_skip_due_eval_gate_count', [])}
        opportunities = {int(v[0]): v[1] for v in curves.get(base+'rollout_opportunity_count', [])}
        env = curves.get('train/Metrics/EnvironmentSteps', [])
        steps = sorted(set(skips) & set(opportunities))
        for a, b in zip(steps, steps[1:]):
            ds, do = skips[b]-skips[a], opportunities[b]-opportunities[a]
            if do <= 0 or ds < 0 or ds > do:
                continue
            before = [v for v in env if v[0] <= b]
            rows.append({'run':run, 'global_step_start':a, 'global_step_end':b,
                         'env_steps_end':before[-1][1] if before else None,
                         'skip_count':ds, 'opportunity_count':do, 'skip_fraction':ds/do})
    return rows

def collapse_checkpoint_rows(rows):
    """Average ranks without mislabelling one rank's sampling SD as pooled SD."""
    metadata_keys = ('run', 'task', 'seed', 'variant', 'checkpoint', 'env_steps',
                     'global_step', 'protocol', 'n', 'ridge', 'repetitions',
                     'training_return', 'return_logged_env_step')
    out = {k: rows[0][k] for k in metadata_keys if k in rows[0]}
    out['num_ranks'] = len(rows)
    for key in ('target_mean', 'heldout_mean', 'effective_dim_mean',
                'target_heldout_ratio', 'td_rmse', 'q_rms', 'replay32_rmse'):
        values = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    # Independent diagnostic sampling across ranks; not uncertainty over seeds.
    out['sampling_sd_of_rank_mean'] = float(
        np.sqrt(sum(r['target_sd'] ** 2 for r in rows)) / len(rows))
    return out


def summarize(output, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    curves_by_run = {r['path']: read_curves(r['path']) for r in manifest['runs']}
    # TR2 files are complementary evidence, never classified as BAFCv3 checkpoints.
    tr2 = {}
    for root in glob.glob(manifest['source_glob']):
        for conf in Path(root).rglob('alf_config.py'):
            run = conf.parent
            if 'tr2' in str(run).lower():
                tr2[str(run)] = read_curves(run)
    atomic_json(output / 'curves.json', {'bafcv3': curves_by_run, 'tr2': tr2})
    manifest['event_files'] = {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                                      'sha256': digest(p)}
                               for run in list(curves_by_run) + list(tr2)
                               for p in Path(run).rglob('events*') if p.is_file()}
    windows = tr2_skip_windows(tr2)
    write_csv(output / 'tr2_skip_windows.csv', windows)
    tr2_findings, tr2_summary = [], []
    for run in tr2:
        ww = [w for w in windows if w['run'] == run and w['env_steps_end'] is not None]
        if not ww:
            continue
        last_step = max(w['env_steps_end'] for w in ww)
        total_skips = sum(w['skip_count'] for w in ww)
        total_opportunities = sum(w['opportunity_count'] for w in ww)
        early = [w for w in ww if w['env_steps_end'] <= .25 * last_step]
        early_skips = sum(w['skip_count'] for w in early)
        late = [w for w in ww if w['env_steps_end'] > .5 * last_step]
        early_opportunities = sum(w['opportunity_count'] for w in early)
        late_opportunities = sum(w['opportunity_count'] for w in late)
        early_fraction = early_skips / early_opportunities if early_opportunities else None
        late_fraction = sum(w['skip_count'] for w in late) / late_opportunities if late_opportunities else None
        early_share = early_skips / total_skips if total_skips else None
        record = {'run':run, 'observed_env_steps_end':last_step, 'recorded_skip_count':total_skips,
                  'recorded_opportunity_count':total_opportunities,
                  'skip_fraction':total_skips/total_opportunities,
                  'first_quarter_skip_share':early_share,
                  'first_quarter_skip_fraction':early_fraction, 'second_half_skip_fraction':late_fraction}
        tr2_summary.append(record)
        tr2_findings.append(f"- **{Path(run).name}**: {total_skips:,.0f} recorded skips over {total_opportunities:,.0f} opportunities "
                            f"({100*record['skip_fraction']:.1f}%). First-quarter share of skips: "
                            f"{100*early_share:.1f}%" if early_share is not None else f"- {Path(run).name}: no recorded skips.")
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot([w['env_steps_end'] for w in ww], [w['skip_fraction'] for w in ww])
        ax.set(title=Path(run).name + ': actual counter increments', xlabel='Logged environment steps', ylabel='Windowed skip fraction')
        ax.grid(alpha=.2)
        (output/'figures').mkdir(exist_ok=True)
        name = 'tr2_' + hashlib.sha256(run.encode()).hexdigest()[:10]
        for ext in ('png','pdf'): fig.savefig(output/'figures'/f'{name}.{ext}')
        plt.close(fig)
    write_csv(output / 'tr2_summary.csv', tr2_summary)
    rows, cross, statuses = [], [], []
    for item in manifest['jobs']:
        p = output / item['result']
        if p.exists():
            result = read_json(p)
            if 'metadata' in result:
                rows += summary_rows(result, curves_by_run.get(result['run'], {}))
                cross += cross_rank(result, item['expected_ranks'])
                item['status'] = ('partial_missing_ranks' if item.get('missing_ranks') and result['status'] == 'ok' else result['status'])
                item['rank_errors'] = result['rank_errors']
        statuses.append({'run': item['run'], 'checkpoint': item['checkpoint'], 'status': item['status'],
                         'missing_ranks': ','.join(map(str,item.get('missing_ranks',[]))), 'result': item['result']})
    write_csv(output / 'summary.csv', rows)
    write_csv(output / 'cross_rank.csv', cross)
    write_csv(output / 'inventory.csv', statuses)
    primary = [r for r in rows if r['n'] == 128 and r['ridge'] == 1e-4 and r['protocol'] == 'reconstructed_cache']
    grouped = {}
    for row in primary:
        grouped.setdefault(row['run'], []).append(row)
    findings, candidates, run_points, sensitivities = [], [], [], []
    for run, group in sorted(grouped.items()):
        steps = sorted(set(r['env_steps'] for r in group))
        collapsed = []
        for s in steps:
            rr = [r for r in group if r['env_steps'] == s]
            c = collapse_checkpoint_rows(rr)
            collapsed.append(c)
        title = f"{group[0]['task']} seed {group[0]['seed']} ({group[0]['variant']})"
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for key, label in [('target_mean','Target'),('heldout_mean','Held-out behavior'),('effective_dim_mean','Effective dimension')]:
            axes[0,0].plot(steps,[r[key] for r in collapsed], marker='o',label=label)
        axes[0,0].set_ylabel('Metric'); axes[0,0].legend()
        axes[0,1].plot(steps,[r['target_heldout_ratio'] for r in collapsed],marker='o')
        axes[0,1].set_ylabel('Target / held-out behavior')
        for key,label in [('td_rmse','One-step'),('replay32_rmse','32-step')]:
            axes[1,0].plot(steps,[r[key] for r in collapsed],marker='o',label=label)
        axes[1,0].set_ylabel('Replay residual RMSE'); axes[1,0].legend()
        curve = curves_by_run[run].get('train/Metrics_vs_EnvironmentSteps/AverageReturn',[])
        if curve:
            axes[1,1].plot([v[0] for v in curve],[v[1] for v in curve])
        axes[1,1].set_ylabel('Training return (logged)')
        for ax in axes.flat:
            ax.set_xlabel('Logged environment steps'); ax.grid(alpha=.2)
        fig.suptitle(title + '\nReconstructed caches; rank means; n=128, ridge=1e-4')
        name = hashlib.sha256(run.encode()).hexdigest()[:10]
        (output/'figures').mkdir(exist_ok=True)
        for ext in ('png','pdf'):fig.savefig(output/'figures'/f'{name}.{ext}')
        plt.close(fig)
        run_points.extend(collapsed)
        first,last = collapsed[0],collapsed[-1]
        primary_lookup = {(r['env_steps'], r['rank']): r for r in group}
        for alt in rows:
            key = alt['env_steps'], alt['rank']
            if alt['run'] != run or alt['protocol'] != 'reconstructed_cache' or key not in primary_lookup:
                continue
            base = primary_lookup[key]
            sensitivities.append({'run': run, 'task': alt['task'], 'seed': alt['seed'],
                                  'env_steps': alt['env_steps'], 'rank': alt['rank'],
                                  'n': alt['n'], 'ridge': alt['ridge'],
                                  'target_relative_to_primary': alt['target_mean'] / max(base['target_mean'], 1e-12),
                                  'heldout_relative_to_primary': alt['heldout_mean'] / max(base['heldout_mean'], 1e-12)})
        findings.append(f"- **{title}**: {len(steps)} checkpoints, env steps {steps[0]:,}–{steps[-1]:,}; target metric {first['target_mean']:.2f} → {last['target_mean']:.2f}, held-out baseline {first['heldout_mean']:.2f} → {last['heldout_mean']:.2f}, effective dimension {first['effective_dim_mean']:.2f} → {last['effective_dim_mean']:.2f}. [Plot](figures/{name}.png).")
        if len(collapsed) >= 3:
            correlations = {}
            for key in ('effective_dim_mean', 'heldout_mean', 'target_heldout_ratio', 'td_rmse'):
                x = np.array([r['target_mean'] for r in collapsed])
                y = np.array([r[key] for r in collapsed])
                if np.std(x) > 0 and np.std(y) > 0:
                    correlations[key] = float(np.corrcoef(x, y)[0, 1])
            findings.append('  Descriptive within-run correlations with raw trust (no significance claim): ' +
                            ', '.join(f'{k}={v:.2f}' for k, v in correlations.items()) + '.')
        for i in range(2,len(collapsed)-1):
            window=collapsed[i-2:i+1];now=collapsed[i];future=collapsed[i+1:min(i+3,len(collapsed))]
            metrics=np.array([r['target_mean'] for r in window]);stability=float(np.ptp(metrics)/max(np.mean(metrics),1e-12))
            returns=[r['training_return'] for r in future if r['training_return'] is not None]
            gain=max(returns)-now['training_return'] if returns and now['training_return'] is not None else None
            candidates.append({'run':run,'task':now['task'],'seed':now['seed'],'variant':now['variant'],
                               'env_steps':now['env_steps'],'checkpoint':now['checkpoint'],
                               'three_checkpoint_relative_range':stability,'subsequent_return_gain':gain,
                               'stable_and_improving':stability<=.2 and gain is not None and gain>0})
    candidates.sort(key=lambda x:(x['run'],not x['stable_and_improving'],x['three_checkpoint_relative_range']))
    write_csv(output/'restart_candidates.csv',candidates)
    write_csv(output/'sensitivity.csv',sensitivities)
    write_csv(output/'checkpoint_means.csv',run_points)
    task_groups = {}
    for row in run_points:
        task_groups.setdefault((row['task'], row['variant'], row['env_steps']), []).append(row)
    task_rows = []
    for (task, variant, step), points in sorted(task_groups.items()):
        if len({p['seed'] for p in points}) != len(points):
            continue  # Ambiguous duplicate seed runs stay separate, never receive extra weight.
        row = {'task': task, 'variant': variant, 'env_steps': step, 'num_seeds': len(points),
               'seeds': ','.join(str(p['seed']) for p in points)}
        for key in ('target_mean', 'heldout_mean', 'effective_dim_mean', 'td_rmse', 'target_heldout_ratio'):
            values = [p[key] for p in points]
            row[key] = float(np.mean(values))
            row[key + '_seed_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        task_rows.append(row)
    write_csv(output/'task_summary.csv',task_rows)
    for task, variant in sorted({(r['task'], r['variant']) for r in task_rows}):
        rr = [r for r in task_rows if r['task'] == task and r['variant'] == variant]
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for key, label in [('target_mean','Target'),('heldout_mean','Held-out behavior'),('effective_dim_mean','Effective dimension')]:
            x = np.array([r['env_steps'] for r in rr]); y = np.array([r[key] for r in rr])
            sd = np.array([r[key + '_seed_sd'] or 0 for r in rr])
            ax.plot(x, y, marker='o', label=label); ax.fill_between(x, y-sd, y+sd, alpha=.15)
        ax.set(title=f'{task} ({variant}): equal-seed means ± seed SD', xlabel='Logged environment steps', ylabel='Metric')
        ax.legend(); ax.grid(alpha=.2)
        name = 'task_' + task.replace(':','_') + '_' + variant
        for ext in ('png','pdf'): fig.savefig(output/'figures'/f'{name}.{ext}')
        plt.close(fig)
    available_tr2=bool(tr2_summary)
    report=['# Offline BAFCv3 checkpoint diagnostics','',
            'This is a retrospective offline study. It cannot establish the benefit of delayed rollout skipping or recover historical TR2 metrics.', '',
            '## Coverage', '',f"Discovered {len(manifest['runs'])} BAFCv3/candidate runs and {len(manifest['jobs'])} checkpoints. "
            f"Results: {sum(j['status']=='ok' for j in manifest['jobs'])} successful, "
            f"{sum(j['status']!='ok' for j in manifest['jobs'])} partial/pending/error. See [inventory](inventory.csv) and [manifest](manifest.json).",'']
    report += [f"- {r['path']}: {r['status']}" for r in manifest['runs'] if not r['checkpoints']]
    report += ['', '## Task and seed observations','']+findings
    report += ['', '## Candidate restart horizons','',
               'Exploratory stability means a three-checkpoint relative range ≤20%; improvement means a positive logged-return gain at either of the next two checkpoints. This is hindsight selection, not a test of convergence. Rank candidates within their run/configuration using [candidate table](restart_candidates.csv).']
    for run in grouped:
        good=[r for r in candidates if r['run']==run and r['stable_and_improving']]
        report.append(f"- {Path(run).name}: "+(', '.join(str(r['env_steps']) for r in good[:3]) if good else 'no stable, still-improving window meets this exploratory rule.'))
    report += ['', '## Historical TR2 skip timing', ''] + (tr2_findings or ['No usable aligned skip/opportunity counters are currently available.'])
    report += ['', 'Counter windows that cross the first-quarter boundary are assigned by their end step; first-quarter fractions are approximate at logging resolution. See [TR2 summary](tr2_summary.csv) and [windows](tr2_skip_windows.csv).']
    report += ['', '## Interpretation and limits','',
               '- A small raw metric is a coverage estimate, not a critic-accuracy certificate. Compare it with effective dimension, held-out behavior baselines, and residuals.',
               '- See [sample/ridge sensitivity](sensitivity.csv), [checkpoint means](checkpoint_means.csv), and [equal-seed task summaries](task_summary.csv).',
               '- Increasing held-out baselines or strong sample-count/ridge sensitivity support a representation/estimation explanation. A rising target-to-held-out ratio is evidence consistent with additional target/replay mismatch; it is not causal identification.',
               '- TD and 32-step errors are behavior-replay consistency proxies, not independent on-policy value errors. Their absolute scale also changes with predicted return scale; Q RMS is included.',
               '- Current checkpoint networks reconstruct reference/snapshot networks. Saved rank-0 caches retain historical normalization; reconstructed caches use the shared checkpoint normalizer and are a separate protocol.',
               '- Repetition SD describes sampling variability; ranks and repetitions are not independent training seeds. Separate variants are never combined. Missing tasks/seeds limit generalization.',
               '- Threshold-pass probabilities do not include feedback, cadence, skip caps, or refresh staleness and must not be called realized skip rates.',
               '- Historical TR2 early-skip concentration remains unverified.' if not available_tr2 else '- Historical TR2 event curves are stored separately in curves.json; inspect counter increments and effective metric on aligned step axes.',
               '', '## Reproduction','',
               'Settings, input SHA-256 hashes, source hashes, software/device information and raw repetition/ensemble results are preserved in the manifest and compressed checkpoint results. Re-run with the same --output and --resume; changed inputs/settings/code invalidate cached results.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    atomic_json(output/'manifest.json',manifest)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-glob',default='/workspace/server*_copy')
    p.add_argument('--output')
    p.add_argument('--task',action='append',default=[])
    p.add_argument('--seed-filter',type=int,action='append',default=[])
    p.add_argument('--checkpoint',type=int,action='append',default=[])
    p.add_argument('--device',default='auto')
    p.add_argument('--repetitions',type=int,default=20)
    p.add_argument('--sample-counts',type=int,nargs='+',default=[128,512,2048])
    p.add_argument('--ridges',type=float,nargs='+',default=[1e-5,1e-4,1e-3])
    p.add_argument('--critic-samples',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=20260917)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-rescan',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--worker-request',help=argparse.SUPPRESS)
    return p


def main():
    args=parser().parse_args()
    if args.worker_request:
        request=read_json(args.worker_request)
        try:worker(request)
        except Exception as exc:
            atomic_json(request['result'],{'status':'error','error':str(exc),'traceback':traceback.format_exc()})
            raise
        return
    output=Path(args.output or REPO/'artifacts/bafcv3_offline_eval'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        summarize(output,read_json(output/'manifest.json'));return
    if (output/'manifest.json').exists() and not args.resume:
        raise ValueError('Output exists; use --resume or a new directory')
    options={key:getattr(args,key) for key in ('device','repetitions','sample_counts','ridges','critic_samples','batch_size','cpu_threads','seed')}
    if args.repetitions<1 or min(args.sample_counts)<1 or min(args.ridges)<=0 or args.critic_samples<1:
        raise ValueError('Sampling counts, repetitions and ridges must be positive')
    code=source_identity()
    manifest={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'source_glob':args.source_glob,
              'options':options,'code':code,'command':sys.argv,'filters':{'tasks':args.task,'seeds':args.seed_filter,'checkpoints':args.checkpoint},'runs':[],'jobs':[]}
    completed={}
    print('OUTPUT',output,flush=True)
    for scan in range(1 if args.no_rescan else 2):
        runs=discover(args.source_glob,args.task,args.seed_filter,args.checkpoint)
        manifest['runs']=runs
        for run in runs:
            expected=sorted({int(rank) for cp in run['checkpoints'] for rank in cp['shards']})
            for cp in run['checkpoints']:
                ident=hashlib.sha256(cp['path'].encode()).hexdigest()[:16]
                result=f'checkpoints/{ident}.json.gz'
                paths=input_paths(run,cp)
                snapshot=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths if p.exists()]
                if completed.get(cp['path'])==snapshot:
                    for existing in manifest['jobs']:
                        if existing['checkpoint'] == cp['path']:
                            existing['expected_ranks'] = expected
                            existing['missing_ranks'] = sorted(set(expected)-set(map(int, cp['shards'])))
                    continue
                item={'run':run['path'],'checkpoint':cp['path'],'result':result,'expected_ranks':expected,
                      'missing_ranks':sorted(set(expected)-set(map(int,cp['shards']))),'status':'pending_no_replay'}
                manifest['jobs']=[j for j in manifest['jobs'] if j['checkpoint']!=cp['path']]+[item]
                atomic_json(output/'manifest.json',manifest)
                if not cp['shards']:continue
                request={'run':run,'checkpoint':cp,'options':options,'code':code,'resume':args.resume or scan>0,
                         'result':str(output/result)}
                req=output/'requests'/f'{ident}.json';atomic_json(req,request)
                log=output/'logs'/f'{ident}.log';log.parent.mkdir(exist_ok=True)
                print('CHECKPOINT_START',cp['path'],flush=True)
                with log.open('a') as stream:
                    process=subprocess.run([sys.executable,'-m','alf.bin.evaluate_bafcv3_checkpoints','--worker-request',str(req)],
                                           cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
                item['status']=read_json(output/result).get('status','error') if (output/result).exists() else 'error'
                if item['missing_ranks'] and item['status']=='ok':item['status']='partial_missing_ranks'
                completed[cp['path']]=snapshot
                print('CHECKPOINT_END',cp['path'],item['status'],flush=True)
                atomic_json(output/'manifest.json',manifest)
        print('SCAN_COMPLETE',scan+1,flush=True)
    summarize(output,manifest)
    print('REPORT',output/'report.md',flush=True)


if __name__=='__main__':
    main()
