"""Restart invariants. Real checkpoint and GPU tests run through the CLI."""
import copy
from functools import partial
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.agent import Agent, AgentInfo
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3, BafcInfo as SourceInfo
from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2, BafcInfo
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import Experience, TimeStep, StepType
from alf.experience_replayers.replay_buffer import BatchInfo
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils.bafcv3_restart import (
    RestartReplayBuffer, assert_equal_state, clone_empty_optimizer,
    fingerprint_inputs, isolated_rng, remap_optimizers, state_digest,
    validate_options, validate_resume_inputs, quantile_rank_maxima, save_initial)


def small_agent(cls, root, critic_utd=11):
    config = TrainerConfig(root_dir=root, mini_batch_size=4, mini_batch_length=2,
        initial_collect_steps=0, num_updates_per_train_iter=12,
        whole_replay_buffer_training=False, clear_replay_buffer=False)
    kwargs = dict(actor_utd=1, critic_utd=critic_utd, num_actor_critic=3,
        actor_critic_pairing=False, num_sampled_critics_for_actor=2,
        use_random_critic_targets=True, num_actor_eval_samples=4,
        actor_encoding_dim=8, obs_action_encoding_dim=8,
        actor_network_cls=partial(alf.networks.ActorFCNetwork, fc_layer_params=(8,)),
        critic_network_cls=partial(alf.networks.FuncCriticNetwork,
            obs_action_joint_fc_layer_params=(8,), actor_obs_action_joint_fc_layer_params=(8,)),
        actor_encoder_cls=partial(alf.networks.TransformerEncoder, num_layers=1, num_attention_heads=1))
    if cls is BafcAlgorithmV3TR2:
        kwargs.update(enable_eval_rollout_skip_gate=False, monitor_trust_metrics=False)
    return Agent(observation_spec=TensorSpec((4,)),
        action_spec=BoundedTensorSpec((2,), minimum=-1., maximum=1.), config=config,
        rl_algorithm_cls=partial(cls, **kwargs), optimizer=alf.optimizers.Adam(lr=3e-4))


class RestartTest(unittest.TestCase):
    def setUp(self):
        alf.set_default_device('cpu')
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_options(self):
        options = dict(threshold_quantile=.33, calibration_repetitions=100,
                       critic_utd=3, final_env_steps_per_rank=150000)
        validate_options(options)
        for name, value in [('threshold_quantile', float('nan')),
                            ('threshold_quantile', 1.1), ('critic_utd', 4),
                            ('calibration_repetitions', 0), ('final_env_steps_per_rank', 0)]:
            with self.assertRaises(ValueError):
                validate_options(dict(options, **{name: value}))

    def test_max_precedes_quantile_and_ties(self):
        values = [[1., 9., 1., 9.], [9., 1., 9., 1.]]
        _, threshold = quantile_rank_maxima(values, .33)
        self.assertEqual(threshold, 9.)
        self.assertGreater(threshold, np.quantile(values, .33, axis=0).max())
        self.assertEqual(quantile_rank_maxima([[2, 2]] * 100, .33)[1], 2.)
        self.assertEqual(quantile_rank_maxima([[0, 0]], 0)[1], 0.)
        for values in ([], [[1, float('nan')]], [[1, -1]]):
            with self.assertRaises(ValueError):
                quantile_rank_maxima(values, .33)

    def test_rng_isolation_and_determinism(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        draws = []
        for _ in range(2):
            with isolated_rng(123):
                draws.append((random.random(), float(np.random.rand()), torch.rand(10)))
        self.assertEqual(draws[0][:2], draws[1][:2])
        self.assertTrue(torch.equal(draws[0][2], draws[1][2]))
        self.assertEqual(before[0], random.getstate())
        self.assertTrue(np.array_equal(before[1][1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))

    def test_optimizer_clone_keeps_alf_wrapper(self):
        opt = alf.optimizers.Adam(lr=.003, gradient_clipping=2., grad_accumulation_steps=2)
        clone = clone_empty_optimizer(opt)
        self.assertEqual(clone._gradient_clipping, 2.)
        self.assertEqual(clone._grad_accumulation_steps, 2)
        self.assertFalse(clone._ignore_param_not_requiring_grad)
        clone.add_param_group({'params': [torch.nn.Parameter(torch.ones(2))]})
        with self.assertRaises(ValueError):
            clone_empty_optimizer(clone)
        self.assertEqual(len(opt.param_groups), 1)

    def test_optimizer_remap_uses_names_not_position(self):
        def make(reverse):
            a = Algorithm(train_state_spec=(), optimizer=alf.optimizers.Adam(lr=.003))
            for name in (('b', 'a') if reverse else ('a', 'b')):
                setattr(a, name, torch.nn.Parameter(torch.ones(3)))
            a._setup_optimizers()
            return a
        source, target = make(False), make(True)
        (source.a.sum() + 2 * source.b.sum()).backward()
        source.default_optimizer.step()
        saved = {'_optimizers.0': source.default_optimizer.state_dict()}
        mapped, audit = remap_optimizers(source, target, saved)
        self.assertNotEqual(audit['_optimizers.0']['a']['source_id'], audit['_optimizers.0']['a']['destination_id'])
        target.default_optimizer.load_state_dict(mapped['_optimizers.0'])
        for name in ('a', 'b'):
            assert_equal_state(source.default_optimizer.state[getattr(source, name)],
                target.default_optimizer.state[getattr(target, name)], name)

    def test_phase_offsets_preserve_cumulative_counts(self):
        for utd in (3, 11):
            agent = small_agent(BafcAlgorithmV3TR2, self.tmp.name, utd)
            alg = agent._rl_algorithm
            alg._training_started = True
            alg._train_mode = TrainMode.critic
            alg._critic_update_counter, alg._actor_update_counter = 101, 9
            alg._critic_phase_offset = 101
            for _ in range(24):
                if alg._train_mode == TrainMode.critic:
                    alg._critic_update_counter += 1
                else:
                    alg._actor_update_counter += 1
                alg._update_train_mode()
            cycles = 24 // (utd + 1)
            self.assertEqual(alg._critic_update_counter, 101 + cycles * utd)
            self.assertEqual(alg._actor_update_counter, 9 + cycles)
            restored = small_agent(BafcAlgorithmV3TR2, self.tmp.name, utd)
            restored.load_state_dict(agent.state_dict())
            self.assertEqual(restored._rl_algorithm._critic_phase_offset, 101)

    def _buffer(self, capacity=8, boundaries=None, telemetry=False):
        info = AgentInfo(rl=BafcInfo(action=TensorSpec((1,)))) if telemetry else ()
        return RestartReplayBuffer(data_spec=Experience(time_step=TimeStep(
            step_type=TensorSpec((), torch.int32), reward=TensorSpec(()),
            discount=TensorSpec(()), observation=TensorSpec((1,)),
            prev_action=TensorSpec((1,)), env_id=TensorSpec((), torch.int32)), rollout_info=info),
            num_environments=1, max_length=capacity, restart_boundaries=boundaries,
            enable_checkpoint=True)

    def _add(self, buffer, value, step=StepType.MID, discount=1., info=()):
        buffer.add_batch(Experience(time_step=TimeStep(
            step_type=torch.tensor([step], dtype=torch.int32), reward=torch.tensor([float(value)]),
            discount=torch.tensor([discount]), observation=torch.tensor([[float(value)]]),
            prev_action=torch.zeros(1, 1), env_id=torch.zeros(1, dtype=torch.int32)), rollout_info=info))

    def test_partial_wrapped_replay_and_restart_boundaries(self):
        for size in (5, 13):
            b = self._buffer()
            for i in range(size):
                self._add(b, i)
            b.mark_restart()
            saved = copy.deepcopy(b.state_dict())
            c = self._buffer(boundaries=saved['_restart_boundaries'])
            c.load_state_dict(saved)
            assert_equal_state(saved, c.state_dict(), 'replay')
            self._add(c, 100, StepType.FIRST)
            self._add(c, 101)
            sample, info = c.get_batch(1000, 2)
            self.assertFalse((sample.step_type[:, 1] == StepType.FIRST).any())
            self.assertFalse((info.positions == size - 1).any())
            self.assertTrue(torch.all(sample.reward[:, 1] - sample.reward[:, 0] == 1))

    def test_terminal_time_limit_and_next_reward_alignment(self):
        for discount in (0., 1.):
            b = self._buffer()
            self._add(b, 0)
            self._add(b, 3, StepType.LAST, discount)
            sample, _ = b.get_batch(32, 2)
            self.assertTrue(torch.all(sample.reward[:, 1] == 3))
            self.assertTrue(torch.all(sample.discount[:, 1] == discount))
            self.assertTrue(torch.all(sample.step_type[:, 1] == StepType.LAST))

    def test_live_tr2_telemetry_does_not_change_replay_schema(self):
        b = self._buffer(telemetry=True)
        info = AgentInfo(rl=BafcInfo(action=torch.ones(1, 1),
            eval_trust_metric=torch.tensor([100.]), grad_trust_metric=torch.tensor([1.])))
        self._add(b, 0, info=info)
        self._add(b, 1, info=info)
        sample, _ = b.get_batch(4, 2)
        self.assertEqual(sample.rollout_info.rl.eval_trust_metric, ())
        self.assertTrue(torch.all(sample.rollout_info.rl.action == 1))

    def test_missing_shards_fail(self):
        path = Path(self.tmp.name) / 'run/train/algorithm/ckpt-1'
        path.parent.mkdir(parents=True)
        path.write_text('x')
        with self.assertRaisesRegex(ValueError, 'Missing required inputs'):
            fingerprint_inputs(path)
        with self.assertRaisesRegex(ValueError, 'Incomplete TR2'):
            validate_resume_inputs(path)

    def test_gate_disabled_control_refreshes_real_metric(self):
        alg = small_agent(BafcAlgorithmV3TR2, self.tmp.name)._rl_algorithm
        alg._restart_options = {'test': True}
        alg._debug_summaries = True
        alg._monitor_trust_metrics = True
        alg._last_update_had_actor_step = True
        alg._trust_metric_update_counter = 0
        alg._rollout_skip_sync_mode = 'min'
        with mock.patch.object(alg, '_compute_eval_trust_metric', return_value=torch.tensor(7.)):
            alg.after_update(TimeStep(observation=torch.zeros(2, 4)), BafcInfo(action=torch.zeros(2, 2)))
        self.assertEqual(float(alg._last_eval_trust_effective), 7.)
        self.assertEqual(alg._restart_rank_metrics.tolist(), [7.])
        alg._last_eval_trust = torch.tensor(float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'Invalid restart trust metric'):
            alg._refresh_eval_trust_aggregates()

    def test_no_preparatory_train_iter(self):
        from alf.trainers.policy_trainer import Trainer
        trainer = mock.Mock()
        trainer._algorithm._rl_algorithm._restart_options = {'enabled': True}
        with mock.patch('alf.utils.bafcv3_restart.restore_trainer') as restore:
            Trainer._restore_checkpoint(trainer, mock.Mock())
        restore.assert_called_once()
        trainer._algorithm.train_iter.assert_not_called()

    def test_fixed_minibatches_complete_11_1_cycle_matches_source(self):
        source, target = [small_agent(c, self.tmp.name) for c in (BafcAlgorithmV3, BafcAlgorithmV3TR2)]
        source._rl_algorithm._training_started = True
        source._rl_algorithm._critic_update_counter = 110
        source._rl_algorithm._actor_update_counter = 10
        source._rl_algorithm._train_mode = TrainMode.critic
        source._rl_algorithm._apply_train_mode_grad_flags()
        old, new = source.state_dict(), target.state_dict()
        for key in new:
            if key in old:
                new[key] = old[key]
            elif '_reference_actor_networks.' in key:
                new[key] = old[key.replace('_reference_actor_networks.', '_actor_networks.')]
            elif '_snapshot_critic_networks.' in key:
                new[key] = old[key.replace('_snapshot_critic_networks.', '_critic_networks.')]
        target.load_state_dict(new)
        target._rl_algorithm._restart_options = {'test': True}
        torch.manual_seed(91)
        ts = TimeStep(step_type=torch.ones(4, 2, dtype=torch.int32), reward=torch.randn(4, 2),
            discount=torch.ones(4, 2), observation=torch.randn(4, 2, 4),
            prev_action=torch.zeros(4, 2, 2), env_id=torch.zeros(4, 2, dtype=torch.int32))
        action = torch.randn(4, 2, 2).tanh()
        for agent, cls in ((source, SourceInfo), (target, BafcInfo)):
            exp = Experience(time_step=ts, action=action, rollout_info=AgentInfo(rl=cls(action=action)))
            agent.set_replay_buffer(1, 8)
            agent._set_replay_buffer(alf.nest.map_structure(lambda x: x[:, 0], exp))
            for update in range(12):
                if agent is source:
                    agent._train_info_spec = None
                with isolated_rng(17 + update):
                    info = BatchInfo(env_ids=torch.zeros(4, dtype=torch.int64), positions=torch.zeros(4, dtype=torch.int64))
                    agent._train_experience(exp, info, 1, 4, 2, False, False)
        for name, parameter in source.named_parameters():
            torch.testing.assert_close(parameter, dict(target.named_parameters())[name], atol=1e-7, rtol=1e-6)
        assert_equal_state(source.default_optimizer.state_dict(), target.default_optimizer.state_dict(), 'optimizer after cycle')

    def test_prepare_resume_settings_and_snapshot_invalidation(self):
        from alf.bin.train_bafcv3_tr2_restart import parser, prepare
        run = Path(self.tmp.name) / 'source'
        model = run / 'train/algorithm/ckpt-1'
        model.parent.mkdir(parents=True)
        (run / 'config_files').mkdir()
        (run / 'alf_config.py').write_text('import alf\n')
        (run / 'config_files/config.py').write_text('# saved source\n')
        torch.save(dict(global_step=torch.tensor(1), trainer_progress={'_env_steps': torch.tensor(1)}), model)
        for suffix in ['-optimizer', *[f'-replay_buffer-rank{r}' for r in range(4)]]:
            Path(str(model) + suffix).write_text('fake fixture; only inventory is read')
        root = Path(self.tmp.name) / 'destination'
        args = parser().parse_args(['--source-checkpoint', str(model), '--root-dir', str(root), '--prepare-only'])
        prepared = prepare(args)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            prepare(args)
        args.resume = True
        self.assertEqual(prepare(args)['settings_fingerprint'], prepared['settings_fingerprint'])
        args.critic_utd = 11
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            prepare(args)
        args.critic_utd = 3
        (root / 'source_config/config_files/config.py').write_text('# modified\n')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            prepare(args)
        (root / 'source_config/config_files/config.py').write_text('# saved source\n')
        Path(str(model) + '-optimizer').write_text('changed source')
        with self.assertRaisesRegex(ValueError, 'Inputs or experiment settings changed'):
            prepare(args)

    def test_failed_initial_save_does_not_publish_model(self):
        cp = mock.Mock(_ckpt_dir=self.tmp.name, _modules={})
        with mock.patch('alf.utils.bafcv3_restart.checkpoint_utils.Checkpointer') as ctor:
            ctor.return_value.save.side_effect = OSError('simulated disk error')
            with self.assertRaisesRegex(RuntimeError, 'simulated disk error'):
                save_initial(cp, 1, 0, self.tmp.name)
        self.assertFalse((Path(self.tmp.name) / 'ckpt-1').exists())
        self.assertFalse((Path(self.tmp.name) / 'restart_ready.json').exists())


if __name__ == '__main__':
    unittest.main()
