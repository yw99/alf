"""V6 warm-start invariants; real four-rank checks use the restart CLI."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import alf
from alf.algorithms.agent import AgentInfo
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3, BafcInfo as SourceInfo
from alf.algorithms.bafc_algorithm_v6 import BafcInfo
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import Experience, TimeStep
from alf.experience_replayers.replay_buffer import BatchInfo
from alf.utils.bafcv3_restart_test import small_agent as source_small_agent
from alf.utils.bafcv6_restart import (
    BafcAlgorithmV6Restart, REWEIGHTING_DEFAULTS, V6RestartReplayBuffer,
    collective_call,
    assert_equal_state, isolated_rng, materialize_replay, restart_metadata,
    state_digest, validate_options, validate_resume_inputs)

OPTIONS = dict(settings_fingerprint='fixture', source_env_steps=120000,
               final_env_steps_per_rank=200000, **REWEIGHTING_DEFAULTS)


def small_agent(cls, root):
    if cls is BafcAlgorithmV6Restart:
        from functools import partial
        cls = partial(cls, restart_options=OPTIONS, enable_critic_reweighting=False)
    return source_small_agent(cls, root)


def _missing_shard_worker(rank, rendezvous, root):
    import datetime
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=rendezvous,
        rank=rank, world_size=4, timeout=datetime.timedelta(seconds=30))
    try:
        def read_shard():
            (Path(root) / f'replay-rank{rank}').read_bytes()
        try:
            collective_call(read_shard)
        except RuntimeError as error:
            if 'replay-rank2' not in str(error):
                raise
            (Path(root) / f'failed-rank{rank}').write_text(str(error))
        else:
            raise AssertionError('Missing shard did not fail collectively')
    finally:
        torch.distributed.destroy_process_group()


class V6RestartTest(unittest.TestCase):
    def setUp(self):
        alf.set_default_device('cpu')
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_fixed_minibatches_complete_11_1_cycle_matches_source(self):
        source, target = [small_agent(c, self.tmp.name) for c in (BafcAlgorithmV3, BafcAlgorithmV6Restart)]
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

    def test_first_critic_update_uses_weights(self):
        from alf.algorithms.bafc_algorithm_v6 import BafcState, BafcCriticReweightingInfo
        agent = small_agent(BafcAlgorithmV6Restart, self.tmp.name)
        alg = agent._rl_algorithm
        alg._enable_critic_reweighting = True
        alg._training_started = True
        alg._critic_update_counter, alg._actor_update_counter = 110, 10
        alg._train_mode = TrainMode.critic
        alg._apply_train_mode_grad_flags()
        inputs = TimeStep(step_type=torch.ones(4, dtype=torch.int32),
                          observation=torch.randn(4, 4), reward=torch.zeros(4),
                          discount=torch.ones(4))
        weights = torch.tensor([.25, .5, 1.25, 2.])
        with mock.patch.object(alg, '_compute_critic_sample_weights',
                return_value=(weights, BafcCriticReweightingInfo())) as compute:
            step = alg.train_step(inputs, alg.get_initial_train_state(4),
                                  BafcInfo(action=torch.randn(4, 2).tanh()))
        compute.assert_called_once()
        torch.testing.assert_close(step.info.critic.critic_sample_weight, weights)
        self.assertEqual(alg._critic_update_counter, 111)
        self.assertEqual(alg._actor_update_counter, 10)

    def test_nonuniform_weights_scale_loss(self):
        from alf.algorithms.bafc_algorithm_v6 import BafcCriticInfo
        from alf.data_structures import LossInfo
        alg = small_agent(BafcAlgorithmV6Restart, self.tmp.name)._rl_algorithm
        weights = torch.tensor([[.25, .5], [1.25, 2.]])
        n = alg._num_actor_critic
        class UnitLoss:
            def __call__(self, *args, **kwargs):
                return LossInfo(loss=torch.ones(2, 2))
        alg._critic_losses = [UnitLoss() for _ in range(n)]
        info = BafcInfo(critic=BafcCriticInfo(
            critic=torch.zeros(2, 2, n, n), target_critic=torch.zeros(2, 2, n),
            critic_sample_weight=weights), bootstrap_mask=torch.ones(2, 2, n))
        torch.testing.assert_close(alg._calc_critic_loss(info).loss, weights * n)

    def test_ordinary_restore_fallback(self):
        from alf.trainers.policy_trainer import Trainer
        trainer = mock.Mock()
        trainer._algorithm._rl_algorithm = None
        cp = mock.Mock()
        cp.has_checkpoint.return_value = True
        cp.load.return_value = 123
        with mock.patch('alf.summary.set_global_counter') as counter:
            Trainer._restore_checkpoint(trainer, cp)
        trainer._algorithm.train_iter.assert_called_once_with()
        cp.load.assert_called_once_with(ddp_rank=trainer._rank)
        counter.assert_called_once_with(123)

    def test_missing_shard_fails_on_all_four_ranks(self):
        root = Path(self.tmp.name)
        for rank in (0, 1, 3):
            (root / f'replay-rank{rank}').write_bytes(b'shard')
        torch.multiprocessing.spawn(_missing_shard_worker,
            args=('file://' + str(root / 'rendezvous'), str(root)), nprocs=4, join=True)
        self.assertEqual(len(list(root.glob('failed-rank*'))), 4)

    def test_launcher_dry_run_grid_and_no_writes(self):
        import subprocess
        import shlex
        from alf.bin.train_bafcv6_restart import parser
        root = Path(self.tmp.name)
        sources = root / 'sources'
        for seed in (0, 1):
            run = sources / f'dog_run_bafcv3_rtT_s{seed}' / 'train/algorithm'
            run.mkdir(parents=True)
            for horizon in (120, 140, 160):
                (run / f'ckpt-{horizon * 1001}').touch()
        output = root / 'output'
        script = Path(__file__).resolve().parents[1] / 'examples/run_dog_run_bafcv6_restart_6jobs-4g.sh'
        result = subprocess.run(['bash', str(script), '--dry-run', '--dir', str(output),
            '--source-base-dir', str(sources), '--run-id', 'test'], check=True, capture_output=True, text=True)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('nohup ')]
        self.assertEqual(len(commands), 6)
        args = [parser().parse_args(cmd[4:cmd.index('<')]) for cmd in commands]
        self.assertEqual(len({arg.root_dir for arg in args}), 6)
        self.assertEqual({Path(arg.source_checkpoint).name for arg in args},
                         {'ckpt-120120', 'ckpt-140140', 'ckpt-160160'})
        self.assertTrue(all(arg.final_env_steps_per_rank == 200000 and arg.resume for arg in args))
        self.assertTrue(all(arg.critic_reweighting_solver_iters == 1 for arg in args))
        self.assertFalse(output.exists())

    def test_options(self):
        validate_options(OPTIONS)
        for key, value in [('critic_reweighting_beta', float('nan')),
                           ('critic_reweighting_ridge', 0),
                           ('critic_reweighting_solver_iters', 0),
                           ('critic_reweighting_max_weight', float('inf')),
                           ('critic_reweighting_solver', 'unknown')]:
            with self.assertRaises(ValueError):
                validate_options(dict(OPTIONS, **{key: value}))

    def test_opt_in_restore_does_not_train(self):
        from alf.trainers.policy_trainer import Trainer
        alg = small_agent(BafcAlgorithmV6Restart, self.tmp.name)._rl_algorithm
        trainer = mock.Mock()
        trainer._algorithm._rl_algorithm = alg
        trainer._algorithm.get_metrics.return_value = []
        cp = mock.Mock()
        with mock.patch('alf.utils.bafcv6_restart.restore') as restore:
            Trainer._restore_checkpoint(trainer, cp)
        restore.assert_called_once()
        trainer._algorithm.train_iter.assert_not_called()
        self.assertIs(trainer._checkpointer, cp)

    def test_rank_local_state_and_metadata(self):
        a = small_agent(BafcAlgorithmV6Restart, self.tmp.name)
        alg = a._rl_algorithm
        alg._reweighting_target_observation_cache = torch.randn(5, 4)
        local = alg._rank_local_checkpoint_state()
        expected = state_digest(local)
        torch.rand(12)
        alg._reweighting_target_observation_cache = ()
        alg._load_rank_local_checkpoint_state(local)
        self.assertEqual(state_digest(alg._rank_local_checkpoint_state()), expected)
        with self.assertRaises(ValueError):
            alg._load_rank_local_checkpoint_state(None)
        state = a.state_dict()
        state['_rl_algorithm._bafc_runtime.v6_restart']['settings_fingerprint'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            a.load_state_dict(state)

    def test_missing_native_shards_fail(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete V6'):
            validate_resume_inputs(Path(self.tmp.name) / 'ckpt-1')

    def test_v6_replay_materialization_and_insertion(self):
        agent = small_agent(BafcAlgorithmV6Restart, self.tmp.name)
        count, capacity = 1, 8
        fields = {'_current_size': torch.tensor([5]), '_current_pos': torch.tensor([5]),
                  'action': torch.zeros(count, capacity, 2),
                  'rollout_info|rl|action': torch.zeros(count, capacity, 2)}
        for key, v in dict(step_type=torch.ones(count, capacity, dtype=torch.int32),
                           reward=torch.zeros(count, capacity), discount=torch.ones(count, capacity),
                           observation=torch.randn(count, capacity, 4),
                           prev_action=torch.zeros(count, capacity, 2),
                           env_id=torch.zeros(count, capacity, dtype=torch.int32)).items():
            fields['time_step|' + key] = v
        fields['time_step|env_info|num_env_frames'] = torch.zeros(count, capacity, dtype=torch.int64)
        replay = {'_replay_buffer.' + k: v for k, v in fields.items()}
        restored = materialize_replay(agent, replay)
        agent.load_state_dict(dict(agent.state_dict(), **restored))
        buffer = agent._replay_buffer
        buffer.mark_restart()
        self.assertIsInstance(buffer, V6RestartReplayBuffer)
        sample = Experience(time_step=TimeStep(
            step_type=torch.zeros(1, dtype=torch.int32), reward=torch.zeros(1),
            discount=torch.ones(1), observation=torch.zeros(1, 4),
            prev_action=torch.zeros(1, 2), env_id=torch.zeros(1, dtype=torch.int32),
            env_info={'num_env_frames': torch.zeros(1, dtype=torch.int64)}),
            action=torch.zeros(1, 2), rollout_info=AgentInfo(rl=BafcInfo(action=torch.zeros(1, 2))))
        buffer.add_batch(sample)
        sample = sample._replace(time_step=sample.time_step._replace(step_type=torch.ones(1,dtype=torch.int32)))
        buffer.add_batch(sample)
        batch, info = buffer.get_batch(100, 2)
        self.assertFalse((info.positions == 4).any())
        self.assertEqual(type(batch.rollout_info.rl), BafcInfo)

    def test_prepare_resume_settings_and_snapshot_invalidation(self):
        from alf.bin.train_bafcv6_restart import parser, prepare
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
        args.critic_reweighting_solver_iters = 5
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            prepare(args)
        args.critic_reweighting_solver_iters = 1
        (root / 'source_config/config_files/config.py').write_text('# modified\n')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            prepare(args)
        (root / 'source_config/config_files/config.py').write_text('# saved source\n')
        Path(str(model) + '-optimizer').write_text('changed source')
        with self.assertRaisesRegex(ValueError, 'Inputs or experiment settings changed'):
            prepare(args)


if __name__ == '__main__':
    unittest.main()
