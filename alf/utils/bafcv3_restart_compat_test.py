"""Compatibility and real migration tests for old/new BAFCv3 checkpoints."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import alf
from alf.algorithms.agent import AgentInfo
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3, BafcInfo
from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2, BafcInfo as TR2Info
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import Experience, TimeStep
from alf.experience_replayers.replay_buffer import BatchInfo
from alf.tensor_specs import TensorSpec
from alf.trainers.policy_trainer import TrainerProgress
from alf.utils.bafcv3_restart import (
    assert_equal_state, isolated_rng, materialize_replay, migrate, state_digest)
from alf.utils.bafcv3_restart_compat import RUNTIME_PREFIX as PREFIX, migrate_runtime_state
from alf.utils.bafcv3_restart_test import small_agent
from alf.utils.checkpoint_utils import Checkpointer


class CompatibilityTest(unittest.TestCase):
    def source(self):
        return {PREFIX + name: torch.tensor(value) for name, value in dict(
            training_started=True, train_mode=1, rollout_actor_id=0,
            actor_update_counter=10, critic_update_counter=110).items()}

    def convert(self, source, **kwargs):
        options = dict(target_critic_period=1, target_critic_use_ema=False)
        options.update(kwargs)
        return migrate_runtime_state(source, **options)

    def test_old_and_new_without_input_or_rng_mutation(self):
        for saved in (None, 0, torch.tensor(3, dtype=torch.int32)):
            source = self.source()
            if saved is not None:
                source[PREFIX + 'target_updater_counter'] = saved
            before = copy.deepcopy(source)
            py, np_rng, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
            result, audit = self.convert(source)
            self.assertEqual(int(result[PREFIX + 'target_updater_counter']),
                             0 if saved is None else int(saved))
            self.assertEqual(audit['target_updater_counter_protocol'],
                             'legacy_period_one_zero_fallback' if saved is None else 'restored')
            assert_equal_state(before, source, 'input')
            self.assertEqual(py, random.getstate())
            np.testing.assert_equal(np_rng, np.random.get_state())
            self.assertTrue(torch.equal(cpu, torch.get_rng_state()))

    def test_rejects_malformed_and_unknown_state(self):
        for counter in (-1, True, 1., '1', torch.tensor(-1),
                        torch.tensor(True), torch.tensor(1.), torch.tensor([1])):
            with self.subTest(counter=counter), self.assertRaisesRegex(ValueError, 'scalar integer'):
                self.convert(dict(self.source(), **{PREFIX + 'target_updater_counter': counter}))
        with self.assertRaisesRegex(ValueError, 'Unrecognized source runtime'):
            self.convert(dict(self.source(), **{PREFIX + 'unknown': 0}))
        source = self.source()
        del source[PREFIX + 'actor_update_counter']
        with self.assertRaisesRegex(ValueError, 'Missing source runtime'):
            self.convert(source)

    def test_rejects_unsupported_updaters(self):
        for period in (0, 2, 1., True, lambda: 1):
            with self.assertRaisesRegex(ValueError, 'target_critic_period=1'):
                self.convert(self.source(), target_critic_period=period)
        with self.assertRaisesRegex(ValueError, 'target_critic_use_ema=False'):
            self.convert(self.source(), target_critic_use_ema=True)
        with self.assertRaisesRegex(ValueError, 'intermediate'):
            self.convert(dict(self.source(), **{PREFIX + 'target_updater_recent_models': []}))

    def test_rejection_precedes_source_construction_and_sidecar_loading(self):
        with mock.patch('alf.utils.bafcv3_restart.load', return_value={'algorithm': self.source()}) as load, \
                mock.patch('alf.utils.bafcv3_restart.alf.get_config_value', side_effect=[2, False]), \
                mock.patch('alf.utils.bafcv3_restart.Agent') as ctor:
            with self.assertRaisesRegex(ValueError, 'target_critic_period=1'):
                migrate(mock.Mock(), {'source_checkpoint': 'unused'}, 0)
            ctor.assert_not_called()
            load.assert_called_once_with('unused')


class MigrationTest(unittest.TestCase):
    def setUp(self):
        alf.set_default_device('cpu')
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def agent(self, cls, utd=11):
        agent = small_agent(cls, self.tmp.name, utd)
        agent._rl_algorithm._checkpoint_replay_buffer = True
        agent._data_transformer = ObservationNormalizer(TensorSpec((4,)))
        agent._config.data_transformer = agent._data_transformer
        return agent

    def experience(self):
        action = torch.randn(4, 2, 2).tanh()
        return Experience(time_step=TimeStep(
            step_type=torch.ones(4, 2, dtype=torch.int32), reward=torch.randn(4, 2),
            discount=torch.ones(4, 2), observation=torch.randn(4, 2, 4),
            prev_action=torch.zeros(4, 2, 2), env_id=torch.zeros(4, 2, dtype=torch.int64),
            env_info={'num_env_frames': torch.ones(4, 2, dtype=torch.int64)}),
            action=action, rollout_info=AgentInfo(rl=BafcInfo(action=action)))

    def update(self, agent, exp):
        info = BatchInfo(env_ids=torch.zeros(4, dtype=torch.int64),
                         positions=torch.zeros(4, dtype=torch.int64))
        return agent._train_experience(exp, info, 1, 4, 2, False, False)

    def test_migration_training_and_resume(self):
        for legacy in (True, False):
            for utd in (3, 11):
                with self.subTest(legacy=legacy, critic_utd=utd), isolated_rng(71):
                    source = self.agent(BafcAlgorithmV3)
                    exp = self.experience()
                    source.set_replay_buffer(1, 16)
                    sample = alf.nest.map_structure(lambda x: x[:1, 0], exp)
                    source._set_replay_buffer(sample)
                    for _ in range(8):
                        source._replay_buffer.add_batch(sample)
                    # Populate real optimizer moments and normalizer statistics.
                    self.update(source, exp)
                    source._rl_algorithm._training_started = True
                    source._rl_algorithm._train_mode = TrainMode.critic
                    source._rl_algorithm._critic_update_counter = 110
                    source._rl_algorithm._actor_update_counter = 10
                    source._rl_algorithm._apply_train_mode_grad_flags()
                    source._rl_algorithm._update_target_critic._counter = 2
                    progress = TrainerProgress()
                    progress.set_termination_criterion(0, 200000)
                    progress._env_steps.fill_(120000)
                    progress._iter_num.fill_(100)
                    folder = Path(self.tmp.name) / f'source-{legacy}-{utd}'
                    Checkpointer(str(folder), algorithm=source, trainer_progress=progress).save(120120, ddp_rank=0)
                    path = str(folder / 'ckpt-120120')
                    checkpoint = torch.load(path, weights_only=True)
                    if legacy:
                        del checkpoint['algorithm'][PREFIX + 'target_updater_counter']
                        torch.save(checkpoint, path)
                    before = state_digest(checkpoint)
                    target = self.agent(BafcAlgorithmV3TR2, utd)
                    # Only construction is adapted to the tiny fixture networks;
                    # loading, optimizer remapping, replay and migration are real.
                    with mock.patch('alf.utils.bafcv3_restart.Agent', side_effect=lambda **kw: self.agent(BafcAlgorithmV3)), \
                            mock.patch('alf.utils.bafcv3_restart.alf.get_config_value', side_effect=[1, False]):
                        migrated, audit = migrate(target, {'source_checkpoint': path}, 0)
                    self.assertEqual(before, state_digest(migrated))
                    self.assertEqual(before, state_digest(torch.load(path, weights_only=True)))
                    self.assertEqual(audit['runtime_compatibility']['target_updater_counter_protocol'],
                                     'legacy_period_one_zero_fallback' if legacy else 'restored')
                    self.assertEqual(any('target updater counter' in x for x in audit['reset']), legacy)
                    restored_progress = TrainerProgress()
                    restored_progress.load_state_dict(migrated['trainer_progress'])
                    assert_equal_state(progress.state_dict(), restored_progress.state_dict(), 'progress')
                    assert_equal_state(source._data_transformer.state_dict(), target._data_transformer.state_dict(), 'normalizer')
                    assert_equal_state(source.default_optimizer.state_dict(), target.default_optimizer.state_dict(), 'optimizer')
                    for name, value in source.named_parameters():
                        torch.testing.assert_close(value, dict(target.named_parameters())[name], rtol=0, atol=0)
                    for name, value in source._replay_buffer.state_dict().items():
                        assert_equal_state(value, target._replay_buffer.state_dict()[name],
                                           'replay.' + name)
                    rl = target._rl_algorithm
                    self.assertEqual(rl._update_target_critic._counter, 0 if legacy else 2)
                    self.assertEqual((rl._actor_update_counter, rl._critic_update_counter), (10, 110))
                    self.assertEqual(rl._critic_phase_offset, 110 if utd == 3 else 0)
                    exp = exp._replace(rollout_info=AgentInfo(rl=TR2Info(action=exp.action)))
                    target._rl_algorithm._restart_options = {'test': True}
                    modes = set()
                    actor_before = [p.clone() for p in rl._actor_networks.parameters()]
                    critic_before = [p.clone() for p in rl._critic_networks.parameters()]
                    for _ in range(utd + 1):
                        modes.add(rl._train_mode)
                        self.update(target, exp)
                    self.assertEqual(modes, {TrainMode.actor, TrainMode.critic})
                    self.assertEqual((rl._actor_update_counter, rl._critic_update_counter), (11, 110 + utd))
                    for before_params, network in ((actor_before, rl._actor_networks), (critic_before, rl._critic_networks)):
                        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_params, network.parameters())))
                    from alf.utils.bafcv3_auto_skip import AutoSkipController, DEFAULTS
                    rl._restart_calibration = {'threshold': .25}
                    rl._auto_skip_controller = AutoSkipController(dict(DEFAULTS), 200000)
                    rl._auto_skip_controller.observe(120000, 100., 1.)
                    saved = copy.deepcopy(target.state_dict())
                    resume = self.agent(BafcAlgorithmV3TR2, utd)
                    resume._rl_algorithm._restart_options = {'test': True}
                    resume._rl_algorithm._auto_skip_controller = AutoSkipController(dict(DEFAULTS), 200000)
                    replay = {k: v for k, v in saved.items() if k.startswith('_replay_buffer.')}
                    materialize_replay(resume, replay)
                    resume.load_state_dict(saved)
                    self.assertEqual(resume._rl_algorithm._update_target_critic._counter, rl._update_target_critic._counter)
                    self.assertEqual(resume._rl_algorithm._restart_calibration, rl._restart_calibration)
                    self.assertEqual(resume._rl_algorithm._auto_skip_controller.state_dict(), rl._auto_skip_controller.state_dict())
                    # Continue both copies with the same draws after restoration.
                    for agent in (target, resume):
                        with isolated_rng(101):
                            self.update(agent, exp)
                    for name, value in target.named_parameters():
                        torch.testing.assert_close(value, dict(resume.named_parameters())[name], rtol=0, atol=0)
                    assert_equal_state(target.default_optimizer.state_dict(), resume.default_optimizer.state_dict(), 'resumed optimizer')


if __name__ == '__main__':
    unittest.main()
