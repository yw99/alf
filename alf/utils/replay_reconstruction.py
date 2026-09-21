"""Opt-in historical replay reconstruction for the archived dog:walk baselines.

Nothing is configured on import. ``configure`` is called only by the generated
restart configuration. Training, losses and replay sampling are inherited
unchanged. Legacy normalizer/RNG/environment history cannot be recovered.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.data_transformer import ObservationNormalizer, IdentityDataTransformer
from alf.algorithms.rlpd_algorithm import RlpdAlgorithm, TrainMode
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.data_structures import make_experience
from alf.trainers.policy_trainer import RLTrainer
from alf.utils import common, dist_utils
from alf.utils.checkpoint_utils import Checkpointer, enable_checkpoint

MIXTURE = ((45045, 2500), (60060, 15000), (75075, 15000),
           (90090, 15000), (105105, 15000), (120120, 15000),
           (135135, 15000), (150150, 7500))
WORLD_SIZE = 4


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def _digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def _collective(fn):
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f'Replay reconstruction failed: {errors}')
    return result


def _rng_state():
    np_state = np.random.get_state()
    return dict(python=random.getstate(), numpy_name=np_state[0],
                numpy_keys=torch.tensor(np_state[1].astype(np.int64), device="cpu"),
                numpy_tail=list(np_state[2:]), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _set_rng(state):
    random.setstate(state['python'])
    np.random.set_state((state['numpy_name'],
                         state['numpy_keys'].cpu().numpy().astype(np.uint32),
                         *state['numpy_tail']))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextlib.contextmanager
def _isolated_rng(seed=None):
    state = _rng_state()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        yield
    finally:
        _set_rng(state)


def _assert_equal(a, b, label='state'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a.cpu(), b.cpu()):
            raise ValueError(f'Changed {label}')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError(f'Changed keys in {label}')
        for k in a:
            _assert_equal(a[k], b[k], f'{label}.{k}')
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f'Changed length in {label}')
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_equal(x, y, f'{label}.{i}')
    elif a != b:
        raise ValueError(f'Changed {label}: {a} != {b}')


def _learner_state(agent):
    return {k: v for k, v in agent.state_dict().items()
            if not k.startswith('_replay_buffer.')}


def _runtime(agent):
    alg = agent._rl_algorithm
    state = dict(training_started=alg._training_started,
                 target_counter=alg._update_target._counter)
    if type(alg) is RlpdAlgorithm:
        state.update(actor=alg._actor_update_counter,
                     critic=alg._critic_update_counter, mode=int(alg._train_mode.value))
    return state


def _set_runtime(agent, state):
    alg = agent._rl_algorithm
    alg._training_started = state['training_started']
    alg._update_target._counter = state['target_counter']
    if type(alg) is RlpdAlgorithm:
        alg._actor_update_counter = state['actor']
        alg._critic_update_counter = state['critic']
        alg._train_mode = TrainMode(state['mode'])


def _infer_legacy_phase(agent):
    """Only the archived full 10-critic/1-actor iteration boundary is supported."""
    alg = agent._rl_algorithm
    alg._training_started = True
    alg._update_target._counter = 0  # source target period is exactly one
    if type(alg) is not RlpdAlgorithm:
        return
    if (alg._actor_utd, alg._critic_utd) != (1, 10):
        raise ValueError('Expected RLPD actor_utd=1, critic_utd=10')
    names = {p: n for n, p in agent.named_parameters()}
    counts = {'actor': set(), 'critic': set(), 'alpha': set()}
    for opt in agent.optimizers():
        for p, state in opt.state.items():
            name = names[p]
            kind = ('actor' if '._actor_network.' in name else
                    'critic' if '._critic_networks.' in name else
                    'alpha' if name.endswith('._log_alpha') else None)
            if kind:
                counts[kind].add(int(state['step']))
    if any(len(v) != 1 for v in counts.values()):
        raise ValueError(f'Inconsistent optimizer counters: {counts}')
    actor, critic, alpha = (next(iter(counts[k])) for k in ('actor', 'critic', 'alpha'))
    # The very first standard forward updates actor/alpha but only increments
    # RLPD's critic Python counter. Subsequent complete cycles have 10:1 UTD.
    if actor != alpha or critic != 10 * (actor - 1) or critic <= 0:
        raise ValueError(f'Not a supported full RLPD cycle boundary: {counts}')
    alg._actor_update_counter = actor - 1
    alg._critic_update_counter = critic
    alg._train_mode = TrainMode.critic


class ReconstructionAgent(Agent):
    """Only adds rank-local checkpoint hooks; no rollout or update overrides."""

    def _rank_local_checkpoint_state(self):
        return dict(version=1, rng=_rng_state(), runtime=_runtime(self),
                    normalizer=copy.deepcopy(self._data_transformer.state_dict()))

    def _load_rank_local_checkpoint_state(self, state):
        if state is None:
            return  # legacy source, explicitly handled by the restorer
        if state['version'] != 1:
            raise ValueError('Unsupported reconstruction rank-state version')
        self._data_transformer.load_state_dict(state['normalizer'], strict=True)
        _set_runtime(self, state['runtime'])
        _set_rng(state['rng'])


def configure(options):
    """Explicitly select the restart trainer after importing the saved config."""
    for path, expected in options['inputs'].items():
        if _digest(path) != expected:
            raise ValueError(f'Source input changed: {path}')
    cls = alf.get_config_value('Agent.rl_algorithm_cls')
    if cls not in (SacAlgorithm, RlpdAlgorithm):
        raise ValueError('Only exact SacAlgorithm/RlpdAlgorithm classes are supported')
    if alf.get_config_value('create_environment.env_name') != 'dog:walk':
        raise ValueError('This protocol supports dog:walk only')
    from functools import partial
    # The training parent already registers _train. A spawned environment worker
    # may only import this configuration, so register the extension there too.
    try:
        alf.get_config_value('_train.trainer_class')
    except ValueError:
        from alf.bin import train  # noqa: F401
    for name, value in {
            'TrainerConfig.algorithm_ctor': ReconstructionAgent,
            'TrainerConfig.num_env_steps': options['final_env_steps'],
            'TrainerConfig.confirm_checkpoint_upon_crash': False,
            '_train.trainer_class': partial(ReconstructionTrainer, options=options),
    }.items():
        alf.config1(name, value, override_all=True, raise_if_used=False)


class ReconstructionTrainer(RLTrainer):
    """All training remains in RLTrainer; only restore and save bookkeeping differ."""

    def __init__(self, *args, options, **kwargs):
        self._reconstruction_options = options
        super().__init__(*args, **kwargs)

    def _restore_checkpoint(self):
        cp = Checkpointer(ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
                          algorithm=self._algorithm,
                          metrics=torch.nn.ModuleList(self._algorithm.get_metrics()),
                          trainer_progress=self._trainer_progress)
        ReplayReconstructionRestorer(self._reconstruction_options).restore(self, cp)
        self._checkpointer = cp

    def _save_checkpoint(self, sync_ddp=True):
        step = int(alf.summary.get_global_counter())
        save = lambda: self._checkpointer.save(step, ddp_rank=self._rank)
        if sync_ddp:
            # A final save can reuse a periodic checkpoint step. Invalidate the
            # old marker before touching any of its files.
            def invalidate():
                if self._rank <= 0:
                    (Path(self._train_dir) / 'algorithm' /
                     f'complete-{step}.json').unlink(missing_ok=True)
            _collective(invalidate)
            _collective(save)
            self._publish_complete(step)
        else:
            save()  # Never publish an uncoordinated crash checkpoint.

    def _publish_complete(self, step):
        def publish():
            if self._rank <= 0:
                directory = Path(self._train_dir) / 'algorithm'
                files = _checkpoint_files(directory, step, WORLD_SIZE)
                if not all(p.is_file() for p in files):
                    raise ValueError('Cannot publish incomplete distributed checkpoint')
                _json(directory / f'complete-{step}.json',
                      dict(step=step, files={p.name: p.stat().st_size for p in files}))
        _collective(publish)

    def train(self):
        if not self._reconstruction_options.get('reconstruct_only'):
            return super().train()
        try:
            self._restore_checkpoint()
        finally:
            self._close()


def _checkpoint_files(directory, step, world_size):
    base = Path(directory) / f'ckpt-{step}'
    return [base, Path(str(base) + '-optimizer')] + [
        Path(str(base) + suffix + str(r)) for r in range(world_size)
        for suffix in ('-replay_buffer-rank', '-rank-state-rank')]


def _latest_complete(directory):
    candidates = []
    for marker in Path(directory).glob('complete-*.json'):
        data = json.loads(marker.read_text())
        required = _checkpoint_files(directory, data['step'], WORLD_SIZE)
        if set(data['files']) != {p.name for p in required}:
            raise ValueError(f'Invalid completion manifest: {marker}')
        if not all(p.is_file() and p.stat().st_size == data['files'][p.name]
                   for p in required):
            raise ValueError(f'Damaged completed checkpoint: {marker}')
        candidates.append(data['step'])
    return max(candidates) if candidates else None


def _new_collector(agent):
    # Clone the pristine optimizer before learner restoration binds parameters.
    opt = agent.default_optimizer
    if opt.state or any(g['params'] for g in opt.param_groups):
        raise ValueError('Collector must be constructed before optimizer materialization')
    fresh = object.__new__(type(opt))
    fresh.__dict__ = copy.deepcopy(opt.__dict__)
    config = copy.copy(agent._config)
    config.data_transformer = copy.deepcopy(agent._data_transformer)
    return Agent(observation_spec=agent.observation_spec,
                 action_spec=agent.action_spec, reward_spec=agent._reward_spec,
                 config=config, optimizer=fresh)


def _load_collector(collector, path):
    state = collector.state_dict()
    model = _load(path)['algorithm']
    keys = {k for k in state if '_optimizers.' not in k}
    if keys != set(model):
        raise ValueError(f'Historical model schema mismatch: {path}')
    state.update(model)
    collector.load_state_dict(state, strict=True)
    collector.eval()


def _frozen_timestep(collector, timestep):
    normalizer = collector._data_transformer
    if isinstance(normalizer, IdentityDataTransformer):
        return timestep
    if type(normalizer) is not ObservationNormalizer or normalizer._fields is not None:
        raise ValueError('Unsupported observation transformer')
    return timestep._replace(observation=normalizer._normalizer.normalize(
        timestep.observation, normalizer._clipping))


def _materialize_specs(agent, collector):
    """Infer schemas without environment interaction or learner forward passes."""
    env = agent._env
    # Use tensor specs, not env.reset/current_time_step (which can collect).
    ts = alf.nest.map_structure(
        lambda spec: spec.zeros(outer_dims=(env.batch_size,)), env.time_step_spec())
    ts = ts._replace(env_info=alf.nest.map_structure(
        lambda spec: spec.zeros(outer_dims=(env.batch_size,)), env.env_info_spec()))
    with torch.no_grad():
        ps = collector.rollout_step(_frozen_timestep(collector, ts),
                                    collector.get_initial_rollout_state(env.batch_size))
        exp = make_experience(ts.cpu(), alf.layers.to_float32(ps),
                              collector.get_initial_rollout_state(env.batch_size))
    exp = common.prune_exp_replay_state(exp, agent._use_rollout_state,
                                        agent.rollout_state_spec, agent.train_state_spec)
    agent._set_replay_buffer(exp)  # constructs only: no samples are inserted
    enable_checkpoint(agent._replay_buffer, True)
    # RLPD must see a full joint-info spec, but only on this disposable model.
    collector.train()
    with torch.enable_grad():
        step = collector.train_step(_frozen_timestep(collector, ts),
                                    collector.get_initial_train_state(env.batch_size), ps.info)
    agent._train_info_spec = dist_utils.extract_spec(step.info)
    collector.eval()


def _materialize_ddp(agent):
    if agent._ddp_activated_rank < 0:
        return
    from alf.utils.distributed import make_ddp_performer
    key = '_compute_train_info_and_loss_info'
    flags = [(p, p.requires_grad) for opt in agent.optimizers()
             for group in opt.param_groups for p in group['params']]
    buffers = [(b, b.clone()) for n, b in agent.named_buffers()
               if not n.startswith('_replay_buffer.')]
    try:
        for p, _ in flags:
            p.requires_grad_(True)
        with _isolated_rng():
            method = getattr(type(agent), key).__wrapped__
            if not hasattr(agent, '_ddp_performer_map'):
                agent._ddp_performer_map = {}
            agent._ddp_performer_map[key] = make_ddp_performer(
                agent, method, find_unused_parameters=True)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        with torch.no_grad():
            for b, saved in buffers:
                b.copy_(saved)


class ReplayReconstructionRestorer:
    """Load learner first, reconstruct replay in isolation, then resume normally."""

    def __init__(self, options):
        self.options = options

    def _validate(self, agent):
        c, a = agent._config, agent._rl_algorithm
        if type(agent) is not ReconstructionAgent or type(a) not in (SacAlgorithm, RlpdAlgorithm):
            raise ValueError('Reconstruction requires its opt-in Agent and exact SAC/RLPD')
        if (agent._is_rnn or agent._env.batch_size != 1 or c.async_unroll
                or c.priority_replay or c.whole_replay_buffer_training
                or c.clear_replay_buffer or c.use_rollout_state
                or not c.temporally_independent_train_step
                or c.mini_batch_length != 2 or c.unroll_length != 1
                or c.enable_amp or c.num_iterations or agent.has_offline
                or c.replay_buffer_length != 100000
                or alf.get_config_value('create_environment.num_spare_envs') != 0):
            raise ValueError('Unsupported replay/training configuration')
        if c.num_updates_per_train_iter != (11 if type(a) is RlpdAlgorithm else 1):
            raise ValueError('Unexpected update ratio')
        if (a._update_target._period() != 1 or a._update_target._delayed_update
                or a._repr_alg is not None or a._prior_actor is not None):
            raise ValueError('Unsupported target update or representation configuration')
        if type(a) is RlpdAlgorithm and a._use_bootstrap_critics:
            raise ValueError('Bootstrap masks are unsupported')
        if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
            raise ValueError('Continuation requires exactly four distributed workers')

    def restore(self, trainer, checkpointer):
        agent = trainer._algorithm
        _collective(lambda: self._validate(agent))
        root = Path(self.options['root_dir'])
        rank = trainer._rank
        source_dir = Path(self.options['source_run']) / 'train/algorithm'
        source = source_dir / 'ckpt-150150'
        with _isolated_rng():
            collector = _collective(lambda: _new_collector(agent))
        latest = _collective(lambda: _latest_complete(root / 'train/algorithm'))
        source_cp = Checkpointer(ckpt_dir=str(source_dir), algorithm=agent,
                                metrics=torch.nn.ModuleList(agent.get_metrics()),
                                trainer_progress=trainer._trainer_progress)
        # Load before schema inference; optimizer groups are constructed by load.
        step = _collective(lambda: source_cp.load(150150, including_replay_buffer=False,
                                                 ddp_rank=rank))
        alf.summary.set_global_counter(step)
        trainer._trainer_progress.update()
        _collective(lambda: _infer_legacy_phase(agent))
        with _isolated_rng():
            _collective(lambda: _load_collector(collector, source))
            _collective(lambda: _materialize_specs(agent, collector))
        if latest is not None:
            step = _collective(lambda: checkpointer.load(latest, ddp_rank=rank))
            agent.reset_state()
        else:
            before = copy.deepcopy(_learner_state(agent))
            metrics = copy.deepcopy(torch.nn.ModuleList(agent.get_metrics()).state_dict())
            phase = _runtime(agent)
            progress_before = copy.deepcopy(trainer._trainer_progress.state_dict())
            counter_before = int(alf.summary.get_global_counter())
            audit = self._reconstruct(agent, collector, source_dir, rank)
            _collective(lambda: _assert_equal(before, _learner_state(agent)))
            _collective(lambda: _assert_equal(phase, _runtime(agent), 'runtime'))
            _collective(lambda: _assert_equal(progress_before, trainer._trainer_progress.state_dict(), 'progress'))
            _collective(lambda: _assert_equal(counter_before, int(alf.summary.get_global_counter()), 'global counter'))
            _collective(lambda: _assert_equal(metrics, torch.nn.ModuleList(agent.get_metrics()).state_dict(), 'metrics'))
            _collective(lambda: _json(root / f'reconstruction-rank{rank}.json', audit))
        del collector
        _materialize_ddp(agent)
        alf.summary.set_global_counter(step)
        trainer._trainer_progress.update()
        trainer._checkpointer = checkpointer
        if latest is None:
            trainer._save_checkpoint(sync_ddp=True)

    def _reconstruct(self, agent, collector, source_dir, rank):
        seed = int.from_bytes(hashlib.sha256(
            f"{self.options['reconstruction_seed']}:{type(agent._rl_algorithm).__name__}:"
            f"{self.options['seed']}:{rank}".encode()).digest()[:4], 'little')
        env = agent._env
        start = time.monotonic()
        counts, interactions = [], 0
        with _isolated_rng(seed), torch.no_grad():
            # ParallelAlfEnvironment accepts a list; local tensor wrappers
            # accept a scalar. Inspect wrappers rather than retrying seed calls.
            from alf.environments.alf_wrappers import AlfEnvironmentBaseWrapper
            from alf.environments.fast_parallel_environment import FastParallelEnvironment
            from alf.environments.parallel_environment import ParallelAlfEnvironment
            base = env
            while isinstance(base, AlfEnvironmentBaseWrapper):
                base = base.wrapped_env()
            parallel = isinstance(base, (FastParallelEnvironment, ParallelAlfEnvironment))
            _collective(lambda: env.seed([seed] if parallel else seed))
            ts = _collective(env.reset)
            state = collector.get_initial_rollout_state(env.batch_size)
            def collect_chunk(size):
                nonlocal ts, state, interactions
                for _ in range(size):
                    state = common.reset_state_if_necessary(
                        state, collector.get_initial_rollout_state(env.batch_size), ts.is_first())
                    ps = collector.rollout_step(_frozen_timestep(collector, ts), state)
                    action = common.detach(ps.output)
                    next_ts = env.step(action)
                    if agent._overwrite_policy_output:
                        ps = ps._replace(output=next_ts.prev_action)
                    agent.observe_for_replay(make_experience(
                        ts.cpu(), alf.layers.to_float32(ps), alf.layers.to_float32(state)))
                    interactions += int((~ts.is_last()).sum())
                    ts, state = next_ts, ps.state

            for checkpoint, quota in MIXTURE:
                _collective(lambda: _load_collector(collector, source_dir / f'ckpt-{checkpoint}'))
                # Coordinate bounded chunks: a fast rank must not spend the
                # entire reconstruction waiting in one collective, nor can a
                # rank-local collection error strand its peers at a later call.
                for offset in range(0, quota, 1000):
                    _collective(lambda: collect_chunk(min(1000, quota - offset)))
                counts.append(dict(checkpoint=checkpoint, entries=quota))
                _collective(lambda: _json(
                    Path(self.options['root_dir']) / f'reconstruction-progress-rank{rank}.json',
                    dict(completed=counts, env_interactions=interactions)))
            agent._current_time_step = ts
            agent._current_policy_state = agent.get_initial_rollout_state(env.batch_size)
            agent._current_transform_state = agent.get_initial_transform_state(env.batch_size)
        if int(agent._replay_buffer.total_size) != sum(n for _, n in MIXTURE):
            raise ValueError('Incorrect reconstructed replay occupancy')
        return dict(rank=rank, seed=seed, policies=counts, env_interactions=interactions,
                    entries=int(agent._replay_buffer.total_size), seconds=time.monotonic()-start,
                    limitations=['Reconstructed, not original replay',
                                 'Legacy rank-local normalization/RNG/environment state unavailable'])
