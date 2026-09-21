"""Isolation and restore tests. Real-source four-rank checks are opt-in.

Run the normal tests with ``python -m unittest alf.utils.replay_reconstruction_test``.
Set ALF_RECONSTRUCTION_SOURCE to an archived SAC/RLPD run to additionally test
real DM Control collection, Gloo training, and replay reload on four CPU ranks.
Only test-local quotas and warm-up are reduced; production defaults stay intact.
"""
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
import alf
from alf.algorithms.agent import Agent
from alf.algorithms.rlpd_algorithm import RlpdAlgorithm
from alf.trainers.policy_trainer import RLTrainer
from alf.utils import replay_reconstruction as rr


class ReconstructionTest(unittest.TestCase):
    def test_training_methods_are_inherited(self):
        for name in ('train_iter', 'train_step', 'rollout_step', 'calc_loss', 'after_update'):
            self.assertIs(getattr(rr.ReconstructionAgent, name), getattr(Agent, name))
        self.assertIs(rr.ReconstructionTrainer._train, RLTrainer._train)
        # Module import does not replace base classes or select a trainer.
        from alf.bin import train
        self.assertIs(inspect.signature(train._train).parameters['trainer_class'].default, RLTrainer)

    def test_trainer_default_and_opt_in(self):
        from alf.bin import train
        config = mock.Mock(ml_type='rl', ddp_paras_check_interval=0)
        raw = train._train
        with mock.patch.object(train.common, 'get_conf_file', return_value='unused'), \
             mock.patch.object(train.policy_trainer, 'TrainerConfig', return_value=config), \
             mock.patch.object(train, 'FLAGS', mock.Mock(as_remote_trainer=False, as_remote_unroller=False)), \
             mock.patch.object(RLTrainer, '__init__', return_value=None) as init, \
             mock.patch.object(RLTrainer, 'train') as run:
            raw('/tmp/test-default')
            init.assert_called_once_with(config, -1, None)
            run.assert_called_once()
            other = mock.Mock()
            raw('/tmp/test-custom', trainer_class=other)
            other.assert_called_once_with(config, -1, None)
            other.return_value.train.assert_called_once()

    def test_rng_isolation(self):
        before = rr._rng_state()
        with rr._isolated_rng(123):
            torch.rand(20)
        rr._assert_equal(before, rr._rng_state())

    def test_mixture(self):
        self.assertEqual(sum(n for _, n in rr.MIXTURE), 100000)
        self.assertEqual([s for s, _ in rr.MIXTURE], sorted(s for s, _ in rr.MIXTURE))

    def test_completion_requires_all_rank_files(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(rr._latest_complete(d))
            files = rr._checkpoint_files(d, 1, 4)
            for p in files:
                p.write_bytes(b'checkpoint')
            # Unpublished complete-looking files must never be selected.
            self.assertIsNone(rr._latest_complete(d))
            rr._json(Path(d)/'complete-1.json', dict(step=1, files={p.name:p.stat().st_size for p in files}))
            self.assertEqual(rr._latest_complete(d), 1)
            files[-1].unlink()
            with self.assertRaisesRegex(ValueError, 'Damaged'):
                rr._latest_complete(d)

    def test_phase_rejects_inconsistent_optimizer_counts(self):
        # Use real source smoke tests for the valid phase and the next full cycle.
        alg = object.__new__(RlpdAlgorithm)
        torch.nn.Module.__init__(alg)
        alg._update_target = mock.Mock(_counter=0)
        alg._actor_utd, alg._critic_utd = 1, 10
        agent = mock.Mock(_rl_algorithm=alg)
        agent.named_parameters.return_value = []
        agent.optimizers.return_value = []
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            rr._infer_legacy_phase(agent)

    @unittest.skipUnless(os.environ.get('ALF_RECONSTRUCTION_SOURCE'), 'Set ALF_RECONSTRUCTION_SOURCE for real-source integration')
    def test_real_source_four_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, '-m', 'alf.utils.replay_reconstruction_test',
                            '--distributed-smoke', os.environ['ALF_RECONSTRUCTION_SOURCE'], d], check=True)


def _worker(rank, source, root, rendezvous, second):
    import torch.distributed as dist
    from alf.bin import train
    from alf.algorithms.config import TrainerConfig
    from alf.utils import common
    from alf.utils.per_process_context import PerProcessContext
    os.environ['MUJOCO_GL'] = 'egl'
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4)
    PerProcessContext().set_distributed(rank=rank, local_rank=-1, num_processes=4)
    alf.config('create_environment', nonparallel=not bool(os.environ.get('ALF_RECONSTRUCTION_PARALLEL_ENV')))
    common.parse_conf_file(str(Path(root)/'reconstruction_conf.py'))
    config = TrainerConfig(root_dir=root, conf_file=str(Path(root)/'reconstruction_conf.py'))
    opts = json.loads((Path(root)/'reconstruction_manifest.json').read_text())
    trainer_ctor = alf.get_config_value('_train.trainer_class')
    assert trainer_ctor.func is rr.ReconstructionTrainer
    trainer = trainer_ctor(config, rank)
    agent, progress = trainer._algorithm, trainer._trainer_progress
    env = agent._env
    uninterrupted = rr._new_collector(agent) if not second else None
    # Test-local reduction: production validation still sees the original config.
    agent.set_replay_buffer(1, 64, False)
    rr.MIXTURE = tuple((s, 8) for s, _ in rr.MIXTURE)
    with mock.patch.object(agent, 'train_iter', side_effect=AssertionError('pre-load train_iter')), \
         mock.patch.object(agent, 'train_step', side_effect=AssertionError('learner spec inference')):
        if second:
            with mock.patch.object(rr.ReplayReconstructionRestorer, '_reconstruct', side_effect=AssertionError('recollection')):
                trainer._restore_checkpoint()
        else:
            trainer._restore_checkpoint()
    assert int(agent._replay_buffer.total_size) == 64
    assert int(progress._env_steps) == (150001 if second else 150000)
    # Collector quotas and handoff remain uncounted in learner metrics.
    assert int(agent.get_step_metrics()[1].result()) == (150001 if second else 150000)
    config.initial_collect_steps = 0  # test-only warm-up reduction
    if uninterrupted is not None:
        # A native, fresh Agent exercises the real initialization update. Its
        # observed Python counters must agree with optimizer-based inference.
        uninterrupted._experience_spec = agent._experience_spec
        uninterrupted._replay_buffer = agent._replay_buffer
        uninterrupted._config.initial_collect_steps = 0
        uninterrupted.train_from_replay_buffer(update_global_counter=False)
        uninterrupted.train_from_replay_buffer(update_global_counter=False)
        actual = rr._runtime(uninterrupted)
        rr._infer_legacy_phase(uninterrupted)
        rr._assert_equal(actual, rr._runtime(uninterrupted), 'uninterrupted phase')
        del uninterrupted
    before = rr._runtime(agent)
    agent.train_iter()
    after = rr._runtime(agent)
    if type(agent._rl_algorithm) is RlpdAlgorithm:
        assert after['actor'] == before['actor'] + 1
        assert after['critic'] == before['critic'] + 10
        assert after['mode'] == before['mode']
    for name, p in agent.named_parameters():
        reference = p.detach().clone()
        dist.broadcast(reference, src=0)
        assert torch.allclose(reference, p, atol=1e-6), name
    # Record the bounded test update, not an 800k production result.
    alf.summary.set_global_counter(150151)
    progress.update(150151, agent.get_step_metrics()[1].result())
    trainer._save_checkpoint()
    env.close()
    dist.destroy_process_group()


def distributed_smoke(source, root):
    from alf.bin.train_replay_reconstruction import parser, prepare
    import torch.multiprocessing as mp
    args = parser().parse_args(['--source-run', source, '--root-dir', root, '--prepare-only'])
    prepare(args)
    for second in (False, True):
        rendezvous = str(Path(root)/('gloo-second' if second else 'gloo-first'))
        mp.start_processes(_worker, args=(source, root, rendezvous, second),
                           nprocs=4, join=True, start_method='spawn')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--distributed-smoke':
        distributed_smoke(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
