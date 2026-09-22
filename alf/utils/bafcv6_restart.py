"""Opt-in, four-rank BAFCv3 -> BAFCv6 training-state migration.

Only saved state can be restored: source simulator/RNG and historical rank-local
normalizers are unavailable. See docs/bafcv6_restart.md for the contract.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import alf
from alf.algorithms.agent import Agent, AgentInfo
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3
from alf.algorithms.bafc_algorithm_v6 import BafcAlgorithmV6, BafcInfo
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.data_structures import Experience, TimeStep
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.utils import checkpoint_utils
from alf.bin.evaluate_bafcv3_checkpoints import atomic_json, jsonable, stable_seed
from alf.utils.bafcv3_restart import (
    WORLD_SIZE, RestartReplayBuffer, assert_equal_state, clone_empty_optimizer,
    collective_call, distributed, fingerprint_inputs, isolated_rng, load,
    materialize_ddp, normalize, remap_optimizers, save_initial, state_digest,
    verify_config_snapshot)

VERSION = 1
PIPELINE = 'bafcv3_to_bafcv6'
_RUNTIME = '_rl_algorithm._bafc_runtime.'
REWEIGHTING_DEFAULTS = dict(
    critic_reweighting_beta=None, critic_reweighting_ridge=1e-4,
    critic_reweighting_solver='lbfgs_logits', critic_reweighting_solver_iters=1,
    critic_reweighting_max_weight=10., critic_reweighting_num_feature_coords=32,
    critic_reweighting_num_target_obs=128,
    critic_reweighting_target_obs_cache_size=512)


def validate_options(options):
    if options['final_env_steps_per_rank'] < 1:
        raise ValueError('final_env_steps_per_rank must be positive')
    for key in ('critic_reweighting_ridge', 'critic_reweighting_max_weight'):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    beta = options['critic_reweighting_beta']
    if beta is not None and (not math.isfinite(beta) or beta < 0):
        raise ValueError('critic_reweighting_beta must be finite and nonnegative')
    for key in ('critic_reweighting_solver_iters', 'critic_reweighting_num_feature_coords',
                'critic_reweighting_num_target_obs', 'critic_reweighting_target_obs_cache_size'):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if options['critic_reweighting_solver'] not in ('lbfgs_logits', 'projected_gradient_fw'):
        raise ValueError('Unsupported critic_reweighting_solver')


def settings_fingerprint(settings, inputs):
    return hashlib.sha256(json.dumps(dict(settings=settings, inputs=inputs,
        version=VERSION, pipeline=PIPELINE), sort_keys=True).encode()).hexdigest()


def restart_metadata(options):
    return dict(pipeline=PIPELINE, version=VERSION,
                settings_fingerprint=options['settings_fingerprint'],
                activation_env_step=options['source_env_steps'])


def configure(options):
    validate_options(options)
    verify_config_snapshot(options)
    if options.get('pipeline') != PIPELINE or options.get('version') != VERSION:
        raise ValueError('Unsupported V6 restart manifest')
    if alf.get_config_value('Agent.rl_algorithm_cls') is not BafcAlgorithmV3:
        raise ValueError('Source configuration must select BAFCv3')
    if alf.get_config_value('BafcAlgorithmV3.eval_samples_source') != 'trainable':
        raise ValueError('Only trainable evaluation samples are supported')
    if alf.get_config_value('BafcAlgorithmV3.target_critic_period') != 1:
        raise ValueError('Unsaved target updater phase requires source period 1')
    source = inspect.signature(BafcAlgorithmV3).parameters
    target = inspect.signature(BafcAlgorithmV6).parameters
    injected = {'self', 'observation_spec', 'action_spec', 'reward_spec', 'env', 'config', 'name'}
    values = {key: alf.get_config_value('BafcAlgorithmV3.' + key)
              for key in source.keys() & target.keys() - injected}
    if (values['actor_utd'] != 1 or values['critic_utd'] != 11 or
            alf.get_config_value('TrainerConfig.num_updates_per_train_iter') != 12):
        raise ValueError('V1 requires the source 11-critic/1-actor, 12-update schedule')
    values.update({key: options[key] for key in REWEIGHTING_DEFAULTS})
    values.update(enable_critic_reweighting=True, checkpoint_replay_buffer=True)
    alf.config('BafcAlgorithmV6', **values)
    alf.config('BafcAlgorithmV6Restart', restart_options=options)
    alf.config1('Agent.rl_algorithm_cls', BafcAlgorithmV6Restart,
                override_all=True, raise_if_used=False)
    # ALF divides the aggregate budget by the world size once.
    for key, value in dict(num_env_steps=WORLD_SIZE * options['final_env_steps_per_rank'],
                           num_iterations=0, debug_summaries=True,
                           confirm_checkpoint_upon_crash=False).items():
        alf.config1('TrainerConfig.' + key, value, override_all=True, raise_if_used=False)


@alf.configurable
class BafcAlgorithmV6Restart(BafcAlgorithmV6):
    """V6 mathematics with opt-in restore and rank-local checkpoint hooks."""

    def __init__(self, *args, restart_options, **kwargs):
        self._restart_options = copy.deepcopy(restart_options)
        super().__init__(*args, **kwargs)

    def _restore_restart_checkpoint(self, trainer, checkpointer):
        restore(trainer._algorithm, trainer._trainer_progress,
                torch.nn.ModuleList(trainer._algorithm.get_metrics()),
                checkpointer, self._restart_options, trainer._rank)

    def _save_bafc_runtime_state(self, destination, prefix):
        super()._save_bafc_runtime_state(destination, prefix)
        destination[self._bafc_runtime_key(prefix, 'v6_restart')] = restart_metadata(self._restart_options)

    def _restore_bafc_runtime_state(self, runtime_state):
        metadata = runtime_state.pop('v6_restart', None)
        if metadata != restart_metadata(self._restart_options):
            raise ValueError('Checkpoint metadata does not match this V6 restart')
        super()._restore_bafc_runtime_state(runtime_state)

    def _rank_local_checkpoint_state(self):
        runtime = {}
        self._save_bafc_runtime_state(runtime, '')
        return dict(version=VERSION, runtime=copy.deepcopy(runtime),
                    python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)

    def _load_rank_local_checkpoint_state(self, state):
        if state is None or state.get('version') != VERSION:
            raise ValueError('Missing or incompatible V6 rank-local state')
        runtime = self._pop_bafc_runtime_state(copy.deepcopy(state['runtime']), '')
        self._restore_bafc_runtime_state(runtime)
        random.setstate(state['python_rng_state'])
        np.random.set_state(state['numpy_rng_state'])
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if state['cuda_rng_state'] is not None:
            if not torch.cuda.is_available():
                raise ValueError('CUDA rank-state must resume on CUDA')
            torch.cuda.set_rng_state(state['cuda_rng_state'].cpu())

    def _synchronize_trainer_control(self, termination_due, periodic_checkpoint_due, checkpoint_requested):
        flags = (termination_due, periodic_checkpoint_due, checkpoint_requested)
        if not distributed():
            return tuple(map(bool, flags)) + (False,)
        device = self._actor_eval_samples.device if dist.get_backend() == 'nccl' else 'cpu'
        values = torch.tensor(flags, dtype=torch.int32, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        return tuple(bool(v) for v in values.tolist()) + (True,)


class V6RestartReplayBuffer(RestartReplayBuffer):
    """Reuse boundary handling without TR2's trust-telemetry stripping."""

    def add_batch(self, batch, env_ids=None):
        return ReplayBuffer.add_batch(self, batch, env_ids)


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
    agent._replay_buffer = V6RestartReplayBuffer(
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


def migrate(agent, options, rank):
    source_path = options['source_checkpoint']
    checkpoint = load(source_path)
    optimizer = load(source_path + '-optimizer')['algorithm']
    replay = load(source_path + f'-replay_buffer-rank{rank}')['algorithm']
    # The original BAFC constructor/config remains registered alongside V6.
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
    for name in added:
        assert_equal_state(target[name], restored[name], name)
    for name in fields:
        assert_equal_state(original[_RUNTIME + name], restored[_RUNTIME + name], name)
    alg = agent._rl_algorithm
    if source_utd != alg._critic_utd:
        raise ValueError('Source and destination UTD differ')
    alg._apply_train_mode_grad_flags()
    normalizer = agent._data_transformer
    if not isinstance(normalizer, ObservationNormalizer) or normalizer._fields is not None:
        raise ValueError('Only ObservationNormalizer is supported')
    cache_key = _RUNTIME + 'reweighting_target_observation_cache'
    cache_size = options['critic_reweighting_target_obs_cache_size']
    if rank == 0 and isinstance(original.get(cache_key), torch.Tensor):
        cache = original[cache_key]
        protocol = 'saved_normalized_rank0_cache'
    else:
        buffer = agent._replay_buffer
        chunks = []
        for env, size in enumerate(buffer._current_size):
            size = min(int(size), cache_size)
            indices = (torch.arange(size, device="cpu") + int(buffer._current_pos[env]) - size) % buffer._max_length
            chunks.append(buffer._buffer.observation[env, indices])
        raw = torch.cat(chunks)[-cache_size:]
        cache = normalize(normalizer, raw)
        protocol = 'reconstructed_cache_with_shared_checkpoint_normalizer'
    if cache.ndim != 2 or len(cache) == 0 or not torch.isfinite(cache).all():
        raise ValueError('Invalid target observation cache')
    alg._reweighting_target_observation_cache = cache[-cache_size:].to(alg._actor_eval_samples.device)
    agent._replay_buffer.mark_restart()
    audit = dict(rank=rank, source_critic_utd=source_utd,
                 destination_critic_utd=alg._critic_utd, optimizer_name_mapping=opt_audit,
                 exact_saved_tensor_and_optimizer_check=True, cache_protocol=protocol,
                 initialized_network_keys=added,
                 source_global_step=int(checkpoint['global_step']),
                 source_env_steps=int(checkpoint['trainer_progress']['_env_steps']),
                 activation_env_step=options['source_env_steps'],
                 initial_optimizer_steps=0, initial_environment_steps=0,
                 reset=['simulator/partial episode', 'new V6 reference actor and snapshot critic',
                        'unsaved target updater counter (source period is 1)',
                        'training RNG (deterministic new stream)'],
                 limitations=['Historical rank-local normalization unavailable; shared rank0 state used',
                              'Historical simulator and RNG state unavailable'])
    if alf.get_config_value('BafcAlgorithmV3.target_critic_period') != 1:
        raise ValueError('An unsaved target updater phase is unsupported for period != 1')
    return checkpoint, audit


def validate_resume_inputs(path):
    required = [path, Path(str(path) + '-optimizer')]
    required += [Path(str(path) + f'{suffix}{rank}') for rank in range(WORLD_SIZE)
                 for suffix in ('-replay_buffer-rank', '-rank-state-rank')]
    missing = [str(p) for p in required if not p.is_file() or not p.stat().st_size]
    if missing:
        raise ValueError(f'Incomplete V6 restart checkpoint: {missing}')


def restore(agent, progress, metrics, checkpointer, options, rank):
    if not distributed() or dist.get_world_size() != WORLD_SIZE:
        raise ValueError('V6 restarts require all four source replay ranks')
    root = Path(options['root_dir'])
    if checkpointer.has_checkpoint():
        step = checkpointer._get_latest_checkpoint_step()
        path = Path(checkpointer._ckpt_dir) / f'ckpt-{step}'
        def prepare_resume():
            validate_resume_inputs(path)
            state = load(path)['algorithm']
            if state.get(_RUNTIME + 'v6_restart') != restart_metadata(options):
                raise ValueError('Checkpoint metadata does not match this V6 restart')
            # V6's legacy loader can synthesize missing networks; native resumes
            # must instead reject incomplete V6 checkpoints.
            for prefix in ('_reference_actor_networks.', '_snapshot_critic_networks.'):
                expected = {k for k in agent.state_dict() if prefix in k}
                if not expected.issubset(state):
                    raise ValueError(f'Incomplete native V6 network state: {prefix}')
            materialize_replay(agent, load(str(path) + f'-replay_buffer-rank{rank}')['algorithm'])
        collective_call(prepare_resume)
        step = collective_call(lambda: checkpointer.load(ddp_rank=rank, strict=True))
        collective_call(lambda: assert_equal_state(load(path)['metrics'], metrics.state_dict(), 'resumed metrics'))
        agent._replay_buffer.mark_restart()
        materialize_ddp(agent)
        audit = dict(rank=rank, resumed=True, global_step=int(step), migrated=False)
        collective_call(lambda: atomic_json(root / f'resume_audit_rank{rank}.json', audit))
    else:
        checkpoint, audit = collective_call(lambda: migrate(agent, options, rank))
        collective_call(lambda: progress.load_state_dict(checkpoint['trainer_progress'], strict=True))
        collective_call(lambda: metrics.load_state_dict(checkpoint['metrics'], strict=True))
        step = int(checkpoint['global_step'])
        if int(progress._env_steps) != options['source_env_steps']:
            raise ValueError('Source environment step differs from manifest')
        progress.update()
        alf.summary.set_global_counter(step)
        seed = stable_seed(options['restart_seed'], options['inputs'][options['source_checkpoint']]['sha256'], rank, 'training')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        materialize_ddp(agent)
        collective_call(lambda: atomic_json(root / f'migration_rank{rank}.json', audit))
        def save_metadata():
            if rank == 0:
                atomic_json(root / 'resolved_config.json', jsonable(dict(
                    alf.get_operative_configs() + alf.get_inoperative_configs())))
        collective_call(save_metadata)
        save_initial(checkpointer, step, rank, root)
    progress.update()
    alf.summary.set_global_counter(int(step))
    return int(step)
