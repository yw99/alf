"""Strict, opt-in BAFCv3 -> TR2 training-state migration.

The supported input is a vector DM Control BAFCv3 Agent with trainable actor
encoding samples, uniform replay and an ObservationNormalizer. Unsupported
schemas fail closed rather than silently resetting training state.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

import alf
from alf.algorithms.agent import Agent, AgentInfo
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3
from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2, BafcInfo
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import Experience, TimeStep, StepType
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.utils import checkpoint_utils
from alf.bin.evaluate_bafcv3_checkpoints import atomic_json, digest, jsonable, stable_seed

VERSION = 2
WORLD_SIZE = 4
_RUNTIME = '_rl_algorithm._bafc_runtime.'


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def distributed():
    return dist.is_available() and dist.is_initialized()


def collective_call(fn):
    """Report local errors on every rank before the next collective."""
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = [error]
    if distributed():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f'Restart failed collectively: {errors}')
    return result


@contextlib.contextmanager
def isolated_rng(seed=None):
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if cuda is not None:
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def validate_options(options):
    q = options['threshold_quantile']
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('threshold_quantile must be finite and in [0, 1]')
    if options['calibration_repetitions'] < 1:
        raise ValueError('calibration_repetitions must be positive')
    utd = options['critic_utd']
    if utd < 1 or 12 % (utd + 1):
        raise ValueError('critic_utd + actor_utd(1) must divide 12 exactly')
    if options['final_env_steps_per_rank'] < 1:
        raise ValueError('final_env_steps_per_rank must be positive')


def fingerprint_inputs(source):
    source = Path(source).resolve()
    if not source.name.startswith('ckpt-') or not source.name[5:].isdigit():
        raise ValueError('Specify an exact ckpt-N model file')
    run = source.parents[2]
    files = [source, Path(str(source) + '-optimizer')]
    files += [Path(str(source) + f'-replay_buffer-rank{r}') for r in range(WORLD_SIZE)]
    files += [run / 'alf_config.py'] + sorted((run / 'config_files').rglob('*.py'))
    missing = [str(p) for p in files if not p.is_file() or not p.stat().st_size]
    if missing:
        raise ValueError(f'Missing required inputs (no rank fallback): {missing}')
    shards = list(source.parent.glob(source.name + '-replay_buffer-rank*'))
    expected = {str(source) + f'-replay_buffer-rank{r}' for r in range(WORLD_SIZE)}
    if set(map(str, shards)) != expected:
        raise ValueError('Expected exactly four replay ranks (0..3)')
    return {str(p): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files}


def settings_fingerprint(settings, inputs):
    return hashlib.sha256(json.dumps(dict(settings=settings, inputs=inputs,
                                         version=VERSION), sort_keys=True).encode()).hexdigest()


def verify_config_snapshot(options):
    root = Path(options['root_dir'])
    for relative, expected in options.get('config_hashes', {}).items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Restart configuration changed: {relative}')


def configure(options):
    """Called by the generated ALF configuration, after importing source config."""
    validate_options(options)
    verify_config_snapshot(options)
    if alf.get_config_value('Agent.rl_algorithm_cls') is not BafcAlgorithmV3:
        raise ValueError('The source configuration must select BAFCv3')
    if alf.get_config_value('BafcAlgorithmV3.eval_samples_source') != 'trainable':
        raise ValueError('Only trainable evaluation samples are supported')
    source_parameters = inspect.signature(BafcAlgorithmV3).parameters
    target_parameters = inspect.signature(BafcAlgorithmV3TR2).parameters
    injected = {'self', 'observation_spec', 'action_spec', 'reward_spec', 'env',
                'config', 'debug_summaries', 'name'}
    values = {}
    for name in source_parameters.keys() & target_parameters.keys() - injected:
        values[name] = alf.get_config_value('BafcAlgorithmV3.' + name)
    values.update(actor_utd=1, critic_utd=options['critic_utd'],
                  rollout_cycles_per_collect=12 // (options['critic_utd'] + 1),
                  checkpoint_replay_buffer=True, restart_options=options,
                  trust_cov_reg=1e-4, trust_metric_num_obs=128,
                  trust_metric_target_obs_cache_size=512,
                  trust_metric_update_interval=8, monitor_trust_metrics=True,
                  enable_eval_rollout_skip_gate=options['rollout_skipping'],
                  rollout_skip_sync_mode='min', enable_eval_trust_max_decay=False,
                  enable_grad_actor_extend_gate=False, enable_critic_reweighting=False,
                  eval_gate_max_consecutive_rollout_skips=3)
    alf.config('BafcAlgorithmV3TR2', **values)
    alf.config1('Agent.rl_algorithm_cls', BafcAlgorithmV3TR2, override_all=True, raise_if_used=False)
    # ALF divides this aggregate budget by the world size exactly once.
    for name, value in dict(num_updates_per_train_iter=12,
                            num_env_steps=WORLD_SIZE * options['final_env_steps_per_rank'],
                            debug_summaries=True, confirm_checkpoint_upon_crash=False).items():
        alf.config1('TrainerConfig.' + name, value, override_all=True, raise_if_used=False)



class RestartReplayBuffer(ReplayBuffer):
    """Retain saved step types and reject windows across simulator resets.

    Ordinary ReplayBuffer loading marks the newest entry LAST. That would alter
    the saved data. Instead, retain it and explicitly exclude the missing
    transition across each restart. Boundary positions are absolute ring indices.
    """
    def __init__(self, *args, restart_boundaries=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self._prioritized_sampling or self._keep_episodic_info:
            raise ValueError('Restart replay requires uniform, non-episodic replay')
        if restart_boundaries is None:
            restart_boundaries = torch.empty((0, 2), dtype=torch.int64)
        self.register_buffer('_restart_boundaries', restart_boundaries.clone().cpu())

    def add_batch(self, batch, env_ids=None):
        # TR2 adds live trust telemetry to rollout info. It is recomputed during
        # learning and has no historical values in BAFC replay. Keep the saved
        # replay schema instead of inventing past metric observations.
        rl_info = getattr(batch.rollout_info, 'rl', None)
        if rl_info is not None:
            batch = batch._replace(rollout_info=batch.rollout_info._replace(
                rl=rl_info._replace(eval_trust_metric=(), grad_trust_metric=())))
        return super().add_batch(batch, env_ids)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        types = state_dict[prefix + 'time_step|step_type']
        positions = state_dict[prefix + '_current_pos']
        env = torch.arange(len(positions), device="cpu")
        newest = types[env, (positions - 1) % types.shape[1]].clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffer.step_type[env, (positions - 1) % types.shape[1]] = newest

    def mark_restart(self):
        rows = [[e, int(pos)] for e, pos in enumerate(self._current_pos) if pos > 0]
        boundaries = self._restart_boundaries.tolist() + rows
        boundaries = sorted(set(tuple(row) for row in boundaries))
        # Discard boundaries whose left side has been overwritten.
        boundaries = [row for row in boundaries if row[1] > int(
            self._current_pos[row[0]] - self._current_size[row[0]])]
        self._restart_boundaries = torch.tensor(boundaries, dtype=torch.int64, device="cpu").reshape(-1, 2)

    @torch.no_grad()
    def get_batch(self, batch_size, batch_length):
        samples, infos, count = [], [], 0
        for _ in range(100):
            sample, info = super().get_batch(batch_size - count, batch_length)
            valid = ~(sample.step_type[:, 1:] == StepType.FIRST).any(dim=1)
            for env, boundary in self._restart_boundaries.tolist():
                valid &= ~((info.env_ids == env) & (info.positions < boundary)
                           & (info.positions + batch_length > boundary))
            n = int(valid.sum())
            if n:
                samples.append(alf.nest.map_structure(lambda x: x[valid], sample))
                infos.append(alf.nest.map_structure(lambda x: x[valid] if isinstance(x, torch.Tensor) else x, info))
                count += n
            if count == batch_size:
                return (alf.nest.map_structure(lambda *x: torch.cat(x), *samples),
                        alf.nest.map_structure(lambda *x: torch.cat(x) if isinstance(x[0], torch.Tensor) else x[0], *infos))
        raise ValueError('Replay has insufficient valid sequence starts')


def materialize_replay(agent, replay_state):
    """Construct schema from saved tensors, without rollout or adding a sample."""
    prefix = '_replay_buffer.'
    fields = {k[len(prefix):]: v for k, v in replay_state.items() if k.startswith(prefix)}
    allowed = {'_current_size', '_current_pos', '_restart_boundaries', 'action',
               'rollout_info|rl|action', 'time_step|discount', 'time_step|env_id',
               'time_step|env_info|num_env_frames', 'time_step|observation',
               'time_step|prev_action', 'time_step|reward', 'time_step|step_type'}
    if set(fields) - allowed or allowed - {'_restart_boundaries'} - set(fields):
        raise ValueError(f'Unsupported replay schema: {sorted(fields)}')
    n, capacity = fields['time_step|step_type'].shape
    if not ((fields['_current_size'] >= 2) & (fields['_current_size'] <= capacity)).all():
        raise ValueError('Invalid or insufficient replay sizes')
    if not (fields['_current_pos'] >= fields['_current_size']).all():
        raise ValueError('Invalid replay positions')
    config = agent._config
    if agent._is_rnn:
        raise ValueError('Recurrent warm starts need a separate state migration')
    if config.priority_replay or config.whole_replay_buffer_training or config.clear_replay_buffer:
        raise ValueError('Unsupported replay training configuration')
    if agent._replay_buffer_num_envs is not None:
        if (n != agent._replay_buffer_num_envs or capacity != agent._replay_buffer_max_length):
            raise ValueError('Saved replay capacity/environment count differs from destination')
    def field(name):
        return fields[name][:, 0].clone()
    ts = TimeStep(**{name: field('time_step|' + name) for name in
                     ('step_type', 'reward', 'discount', 'observation', 'prev_action', 'env_id')},
                  env_info={'num_env_frames': field('time_step|env_info|num_env_frames')})
    sample = Experience(time_step=ts, action=field('action'),
                        rollout_info=AgentInfo(rl=BafcInfo(action=field('rollout_info|rl|action'))))
    from alf.utils import dist_utils
    agent._experience_spec = dist_utils.extract_spec(sample, from_dim=1)
    agent._exp_contains_step_type = True
    agent.set_replay_buffer(n, capacity, prioritized_sampling=False)
    agent._replay_buffer = RestartReplayBuffer(
        data_spec=dist_utils.to_distribution_param_spec(agent._experience_spec),
        num_environments=n, max_length=capacity, prioritized_sampling=False,
        num_earliest_frames_ignored=agent._num_earliest_frames_ignored,
        restart_boundaries=fields.get('_restart_boundaries'),
        name=agent._name + '_replay_buffer')
    agent._observers.append(lambda exp: agent._replay_buffer.add_batch(exp, exp.env_id))
    checkpoint_utils.enable_checkpoint(agent._replay_buffer, True)
    result = dict(replay_state)
    result.setdefault(prefix + '_restart_boundaries', agent._replay_buffer._restart_boundaries)
    return result


def clone_empty_optimizer(optimizer):
    # torch.optim.Optimizer.__getstate__ drops ALF wrapper attributes when
    # deepcopying the optimizer itself. Preserve the complete constructor state.
    if optimizer.state or any(g['params'] for g in optimizer.param_groups):
        raise ValueError('Expected a pristine optimizer before materialization')
    result = object.__new__(type(optimizer))
    result.__dict__ = copy.deepcopy(optimizer.__dict__)
    return result


def optimizer_groups(agent):
    agent._setup_optimizers()
    names = {p: n for n, p in agent.named_parameters()}
    return [[[names[p] for p in group['params']] for group in opt.param_groups]
            for opt in agent.optimizers()]


def remap_optimizers(source, target, saved):
    """Translate serialized IDs through names, including optimizer group identity."""
    source_groups, target_groups = optimizer_groups(source), optimizer_groups(target)
    source_keys = [k for k, v in source.state_dict().items() if isinstance(v, dict)
                   and 'param_groups' in v and 'state' in v]
    target_state = target.state_dict()
    target_keys = [k for k, v in target_state.items() if isinstance(v, dict)
                   and 'param_groups' in v and 'state' in v]
    if source_keys != target_keys or len(source_keys) != len(source_groups):
        raise ValueError('Optimizer ownership changed')
    parameters = dict(target.named_parameters())
    result, audit = {}, {}
    for key, sg, tg in zip(source_keys, source_groups, target_groups):
        old, new = saved[key], target_state[key]
        if len(sg) != len(old['param_groups']) or len(sg) != len(tg):
            raise ValueError('Optimizer group count mismatch')
        states, groups, mapping = {}, [], {}
        for sn, tn, og, ng in zip(sg, tg, old['param_groups'], new['param_groups']):
            if len(sn) != len(og['params']) or set(sn) != set(tn):
                raise ValueError(f'Optimizer group parameters differ: {key}')
            ids = dict(zip(sn, og['params']))
            group = copy.deepcopy(og)
            group['params'] = list(ng['params'])
            groups.append(group)
            for name, new_id in zip(tn, ng['params']):
                old_id = ids[name]
                if old_id not in old['state']:
                    raise ValueError(f'Missing optimizer state for {name}')
                state = old['state'][old_id]
                for moment in ('exp_avg', 'exp_avg_sq'):
                    if (state[moment].shape != parameters[name].shape or
                            state[moment].dtype != parameters[name].dtype):
                        raise ValueError(f'Optimizer moment mismatch: {name}.{moment}')
                states[new_id] = state
                mapping[name] = {'source_id': old_id, 'destination_id': new_id,
                                 'step': float(state['step'])}
        if len(states) != len(old['state']):
            raise ValueError('Unmapped optimizer state')
        result[key] = dict(state=states, param_groups=groups)
        audit[key] = mapping
    return result, audit


def assert_equal_state(expected, actual, label):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or expected.shape != actual.shape or not torch.equal(expected.cpu(), actual.cpu()):
            raise ValueError(f'{label}: tensors differ')
    elif isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError(f'{label}: keys differ')
        for k in expected:
            assert_equal_state(expected[k], actual[k], label + '.' + str(k))
    elif isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise ValueError(f'{label}: lengths differ')
        for i, (a, b) in enumerate(zip(expected, actual)):
            assert_equal_state(a, b, label + '.' + str(i))
    elif expected != actual:
        raise ValueError(f'{label}: {expected!r} != {actual!r}')


def migrate(agent, options, rank):
    source_path = options['source_checkpoint']
    checkpoint = load(source_path)
    optimizer = load(source_path + '-optimizer')['algorithm']
    replay = load(source_path + f'-replay_buffer-rank{rank}')['algorithm']
    # The original BAFC constructor/config remains registered alongside TR2.
    # A fresh optimizer avoids sharing mutable parameter groups with destination.
    with isolated_rng():
        source = Agent(observation_spec=agent.observation_spec,
                       action_spec=agent.action_spec, reward_spec=agent._reward_spec,
                       config=agent._config, rl_algorithm_cls=BafcAlgorithmV3,
                       optimizer=clone_empty_optimizer(agent.default_optimizer))
        source.load_state_dict({**checkpoint['algorithm'], **optimizer}, strict=True)
        mapped_opts, opt_audit = remap_optimizers(source, agent, optimizer)
    source_utd = source._rl_algorithm._critic_utd
    del source
    target = agent.state_dict()
    original = checkpoint['algorithm']
    known_runtime = {'training_started', 'train_mode', 'rollout_actor_id',
                     'actor_update_counter', 'critic_update_counter',
                     'reweighting_target_observation_cache'}
    unknown = {k[len(_RUNTIME):] for k in original if k.startswith(_RUNTIME)} - known_runtime
    if unknown:
        raise ValueError(f'Unrecognized source runtime fields: {sorted(unknown)}')
    for name, value in original.items():
        if name.startswith(_RUNTIME):
            continue
        if name not in target:
            raise ValueError(f'Unexpected source state: {name}')
        if not isinstance(value, torch.Tensor) or target[name].shape != value.shape or target[name].dtype != value.dtype:
            raise ValueError(f'Incompatible state: {name}')
        target[name] = value
    added = []
    for name in target:
        if name in original or name.startswith(_RUNTIME) or name in mapped_opts:
            continue
        replacements = (('_reference_actor_networks.', '_actor_networks.'),
                        ('_snapshot_critic_networks.', '_critic_networks.'))
        for new, old in replacements:
            if new in name:
                target[name] = original[name.replace(new, old)]
                added.append(name)
                break
        else:
            raise ValueError(f'Unexplained new state: {name}')
    fields = ('training_started', 'train_mode', 'rollout_actor_id',
              'actor_update_counter', 'critic_update_counter')
    for name in fields:
        if _RUNTIME + name not in original:
            raise ValueError(f'Missing source runtime state: {name}')
        target[_RUNTIME + name] = original[_RUNTIME + name]
    target.update(mapped_opts)
    replay = materialize_replay(agent, replay)
    target.update(replay)
    agent.load_state_dict(target, strict=True)
    restored = agent.state_dict()
    for name, value in original.items():
        if not name.startswith(_RUNTIME):
            assert_equal_state(value, restored[name], name)
    for name, value in {**replay, **mapped_opts}.items():
        assert_equal_state(value, restored[name], name)
    alg = agent._rl_algorithm
    if source_utd != alg._critic_utd:
        alg._train_mode = TrainMode.critic
        alg._critic_phase_offset = alg._critic_update_counter
    alg._completed_cycles_since_rollout = alg._rollout_cycles_per_collect
    alg._apply_train_mode_grad_flags()
    normalizer = agent._data_transformer
    if not isinstance(normalizer, ObservationNormalizer) or normalizer._fields is not None:
        raise ValueError('Only ObservationNormalizer is supported')
    cache_key = _RUNTIME + 'reweighting_target_observation_cache'
    if rank == 0 and isinstance(original.get(cache_key), torch.Tensor):
        cache = original[cache_key]
        protocol = 'saved_normalized_rank0_cache'
    else:
        buffer = agent._replay_buffer
        chunks = []
        for env, size in enumerate(buffer._current_size):
            size = min(int(size), 512)
            indices = (torch.arange(size, device="cpu") + int(buffer._current_pos[env]) - size) % buffer._max_length
            chunks.append(buffer._buffer.observation[env, indices])
        raw = torch.cat(chunks)[-512:]
        cache = normalize(normalizer, raw)
        protocol = 'reconstructed_cache_with_shared_checkpoint_normalizer'
    if cache.ndim != 2 or len(cache) == 0 or not torch.isfinite(cache).all():
        raise ValueError('Invalid target observation cache')
    alg._target_metric_observation_cache = cache[-512:].to(alg._actor_eval_samples.device)
    agent._replay_buffer.mark_restart()
    audit = dict(rank=rank, source_critic_utd=source_utd,
                 destination_critic_utd=alg._critic_utd, optimizer_name_mapping=opt_audit,
                 exact_saved_tensor_and_optimizer_check=True, cache_protocol=protocol,
                 initialized_network_keys=added,
                 source_global_step=int(checkpoint['global_step']),
                 source_env_steps=int(checkpoint['trainer_progress']['_env_steps']),
                 reset=['simulator/partial episode', 'new TR2 gate/cadence controller',
                        'unsaved target updater counter (source period is 1)',
                        'training RNG (deterministic new stream)'],
                 limitations=['Historical rank-local normalization unavailable; shared rank0 state used',
                              'Historical simulator and RNG state unavailable'])
    if alf.get_config_value('BafcAlgorithmV3.target_critic_period') != 1:
        raise ValueError('An unsaved target updater phase is unsupported for period != 1')
    return checkpoint, audit


def normalize(normalizer, observations):
    # This is exactly the transform's normalize operation without its updates.
    observations = observations.to(next(normalizer.buffers()).device)
    return normalizer._normalizer.normalize(observations, normalizer._clipping)


def state_digest(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((tuple(x.shape), x.dtype)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def quantile_rank_maxima(values, quantile):
    """Aggregate ranks before the quantile; repetitions are diagnostic draws."""
    from alf.bin.calibrate_bafcv3_tr2_thresholds import calibrate_threshold
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError('Expected repetitions by ranks')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative metrics on every rank')
    maxima = values.max(axis=1)
    return maxima, calibrate_threshold(maxima, quantile)['eval_trust_max']


def calibrate(agent, options, rank):
    alg = agent._rl_algorithm
    if alg._restart_calibration is not None:
        raise ValueError('Refusing to recalibrate a calibrated checkpoint')
    source_hash = options['inputs'][options['source_checkpoint']]['sha256']
    seed = stable_seed(source_hash, options['calibration_seed'], rank)
    before = collective_call(lambda: state_digest(agent.state_dict()))
    modes = {module: module.training for module in agent.modules()}
    values = []
    start = time.monotonic()
    with isolated_rng(seed), torch.no_grad():
        agent.eval()
        try:
            for _ in range(options['calibration_repetitions']):
                def measure():
                    exp, _ = agent._replay_buffer.get_batch(
                        agent._config.mini_batch_size, agent._config.mini_batch_length)
                    obs = normalize(agent._data_transformer, exp.observation)
                    device = alg._actor_eval_samples.device
                    value = float(alg._compute_eval_trust_metric(
                        obs.to(device), exp.rollout_info.rl.action.to(device)))
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f'Invalid trust metric: {value}')
                    return value
                value = collective_call(measure)
                gathered = [value]
                if distributed():
                    gathered = [None] * dist.get_world_size()
                    dist.all_gather_object(gathered, value)
                values.append(gathered)
        finally:
            for module, mode in modes.items():
                module.training = mode
    after = collective_call(lambda: state_digest(agent.state_dict()))
    collective_call(lambda: assert_equal_state(before, after, 'frozen calibration state'))
    maxima, threshold = quantile_rank_maxima(values, options['threshold_quantile'])
    if distributed():
        threshold_box = [threshold if rank == 0 else None]
        dist.broadcast_object_list(threshold_box, src=0)
        threshold = threshold_box[0]
    result = dict(version=VERSION, quantile=options['threshold_quantile'],
                  repetitions=options['calibration_repetitions'], seed=options['calibration_seed'],
                  source_sha256=source_hash, threshold=threshold,
                  aggregation='maximum_across_ranks_before_linear_quantile',
                  rank_values=values, effective_values=maxima.tolist(),
                  initial_eligible_fraction=float((maxima <= threshold).mean()),
                  seconds=time.monotonic() - start, frozen_state_verified=True,
                  settings_fingerprint=options['settings_fingerprint'])
    if distributed():
        box = [result if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        result = box[0]
    alg._restart_calibration = result
    alg._eval_trust_max = threshold
    alg._last_eval_trust = torch.tensor(values[-1][rank], device=alg._actor_eval_samples.device)
    alg._refresh_eval_trust_aggregates()
    # A fresh metric is due at the first resumed actor update.
    alg._trust_metric_update_counter = 0
    return result


def validate_resume_inputs(path):
    required = [path, Path(str(path) + '-optimizer')]
    required += [Path(str(path) + f'{suffix}{r}') for r in range(WORLD_SIZE)
                 for suffix in ('-replay_buffer-rank', '-rank-state-rank')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError(f'Incomplete TR2 restart checkpoint: {missing}')


def materialize_ddp(agent):
    """Register both update phases without running a training forward.

    Lazy DDP creation in critic mode would omit the temporarily frozen actor
    parameters, silently leaving subsequent actor gradients unsynchronized.
    """
    if agent._ddp_activated_rank < 0:
        return
    from alf.utils.distributed import make_ddp_performer
    key = '_compute_train_info_and_loss_info'
    performers = getattr(agent, '_ddp_performer_map', {})
    if key in performers:
        return
    parameters = [(p, p.requires_grad) for opt in agent.optimizers()
                  for group in opt.param_groups for p in group['params']]
    buffers = [(b, b.detach().clone()) for n, b in agent.named_buffers()
               if '_replay_buffer.' not in n]
    try:
        for p, _ in parameters:
            p.requires_grad_(True)
        with isolated_rng():
            method = getattr(type(agent), key).__wrapped__
            performers[key] = make_ddp_performer(agent, method, find_unused_parameters=True)
        agent._ddp_performer_map = performers
    finally:
        for p, requires_grad in parameters:
            p.requires_grad_(requires_grad)
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)


def save_initial(checkpointer, step, rank, root):
    """Publish a completion marker only after every rank has saved its sidecars."""
    staging = Path(checkpointer._ckpt_dir) / '.restart-staging'
    staged = checkpoint_utils.Checkpointer(ckpt_dir=str(staging), **checkpointer._modules)
    collective_call(lambda: staged.save(step, ddp_rank=rank))
    def publish():
        if rank != 0:
            return
        model_name = f'ckpt-{step}'
        # Publish the discoverable model file last, after all required sidecars.
        for path in sorted(staging.iterdir(), key=lambda p: p.name == model_name):
            os.replace(path, Path(checkpointer._ckpt_dir) / path.name)
        staging.rmdir()
        atomic_json(Path(root) / 'restart_ready.json', {'global_step': step, 'world_size': WORLD_SIZE})
    collective_call(publish)
    checkpointer._global_step = step


def restore(agent, progress, metrics, checkpointer, options, rank):
    if not distributed() or dist.get_world_size() != WORLD_SIZE:
        raise ValueError('Faithful restarts require all four source replay ranks')
    root = Path(options['root_dir'])
    if checkpointer.has_checkpoint():
        step = checkpointer._get_latest_checkpoint_step()
        path = Path(checkpointer._ckpt_dir) / f'ckpt-{step}'
        def prepare():
            validate_resume_inputs(path)
            state = load(path)['algorithm']
            metadata = state.get(_RUNTIME + 'restart_calibration')
            if not metadata or metadata['settings_fingerprint'] != options['settings_fingerprint']:
                raise ValueError('Checkpoint calibration/settings do not match this restart')
            replay = load(str(path) + f'-replay_buffer-rank{rank}')['algorithm']
            materialize_replay(agent, replay)
        collective_call(prepare)
        step = collective_call(lambda: checkpointer.load(ddp_rank=rank, strict=True))
        collective_call(lambda: assert_equal_state(load(path)['metrics'], metrics.state_dict(), 'resumed metrics'))
        agent._replay_buffer.mark_restart()
        materialize_ddp(agent)
        audit = dict(rank=rank, resumed=True, global_step=int(step), recalibrated=False)
        collective_call(lambda: atomic_json(root / f'resume_audit_rank{rank}.json', audit))
    else:
        checkpoint, audit = collective_call(lambda: migrate(agent, options, rank))
        collective_call(lambda: progress.load_state_dict(checkpoint['trainer_progress'], strict=True))
        collective_call(lambda: metrics.load_state_dict(checkpoint['metrics'], strict=True))
        step = int(checkpoint['global_step'])
        progress.update()
        alf.summary.set_global_counter(step)
        # Start new reproducible RNG streams independently of the experiment arm.
        seed = stable_seed(options['calibration_seed'], options['inputs'][options['source_checkpoint']]['sha256'], rank, 'training')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        materialize_ddp(agent)
        result = calibrate(agent, options, rank)
        collective_call(lambda: atomic_json(root / f'migration_rank{rank}.json', audit))
        def save_metadata():
            if rank == 0:
                atomic_json(root / 'calibration.json', result)
                atomic_json(root / 'resolved_config.json', jsonable(dict(
                    alf.get_operative_configs() + alf.get_inoperative_configs())))
        collective_call(save_metadata)
        save_initial(checkpointer, step, rank, root)
    progress.update()
    alf.summary.set_global_counter(int(step))
    return int(step)


def restore_trainer(trainer, checkpointer):
    agent = trainer._algorithm
    restore(agent, trainer._trainer_progress, torch.nn.ModuleList(agent.get_metrics()),
            checkpointer, agent._rl_algorithm._restart_options, trainer._rank)


@torch.no_grad()
def compare_cpu_gpu_metric(agent):
    """Check native float32 calculations on identical observations and actions.

    Fix the target subset explicitly: CPU and CUDA randperm need not generate
    identical samples from the same seed. Do not compare unrelated random draws.
    """
    from alf.algorithms.data_transformer import create_data_transformer
    alg = agent._rl_algorithm
    before = state_digest(agent.state_dict())
    saved_cache = alg._target_metric_observation_cache
    modes = {m: m.training for m in agent.modules()}
    try:
        with isolated_rng(1781):
            agent.eval()
            experience, _ = agent._replay_buffer.get_batch(64, 2)
            observations = normalize(agent._data_transformer, experience.observation)
            actions = experience.rollout_info.rl.action
            target = saved_cache[:128].detach().clone()
            alg._target_metric_observation_cache = target
            gpu = float(alg._compute_eval_trust_metric(observations, actions))
            with alf.device('cpu'):
                config = copy.copy(agent._config)
                config.data_transformer = create_data_transformer(
                    config.data_transformer_ctor, agent.observation_spec)
                cpu_alg = BafcAlgorithmV3TR2(observation_spec=agent.observation_spec,
                    action_spec=agent.action_spec, reward_spec=agent._reward_spec, config=config)
                cpu_alg.load_state_dict(alg.state_dict(), strict=True)
                cpu_alg._target_metric_observation_cache = target.cpu()
                cpu_alg.eval()
                cpu = float(cpu_alg._compute_eval_trust_metric(observations.cpu(), actions.cpu()))
            error = abs(cpu - gpu)
            relative = error / max(abs(cpu), 1e-8)
            # Float32 pinv on ill-conditioned covariance is backend dependent.
            # This tolerance is fixed before running the real-input comparisons.
            if error > 1e-3 + .005 * abs(cpu):
                raise ValueError(f'CPU/GPU metric mismatch: {cpu} versus {gpu}')
            result = dict(cpu=cpu, gpu=gpu, relative_error=relative,
                          atol=1e-3, rtol=.005, identical_inputs=True)
    finally:
        alg._target_metric_observation_cache = saved_cache
        for module, mode in modes.items():
            module.training = mode
    assert_equal_state(before, state_digest(agent.state_dict()), 'comparison state')
    return result
