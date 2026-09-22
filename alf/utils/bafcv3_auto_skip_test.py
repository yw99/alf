"""Controller, checkpoint, and distributed activation regression tests."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from alf.utils.bafcv3_auto_skip import AutoSkipController, DEFAULTS, after_iteration


def tiny_settings(**kwargs):
    return dict(DEFAULTS, sample_steps=1, window_steps=5, warmup_fraction=0., **kwargs)


def distributed_worker(rank, root):
    from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2
    from alf.algorithms.rlpd_algorithm import TrainMode
    from alf.utils.bafcv3_restart import quantile_rank_maxima
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + root + '/rendezvous',
                            rank=rank, world_size=4)
    try:
        c = AutoSkipController(tiny_settings(), 100)
        alg = SimpleNamespace(_auto_skip_controller=c,
            _restart_options={'root_dir': root}, _last_eval_trust_rank_max=4.,
            _enable_eval_rollout_skip_gate=False, _eval_trust_max=0.,
            _eval_gate_consecutive_rollout_skips=0, _trust_metric_update_counter=5,
            _training_started=True, _train_mode=TrainMode.critic,
            _completed_cycles_since_rollout=1, _rollout_cycles_per_collect=1,
            _restart_calibration={'threshold': 0.},
            _eval_gate_max_consecutive_rollout_skips=3, _last_eval_trust=1.)
        alg._current_eval_trust_max = lambda: alg._eval_trust_max
        state = {'step': 0}
        def metric(name, value):
            return SimpleNamespace(name=name, result=lambda: value())
        agent = SimpleNamespace(_rl_algorithm=alg, get_metrics=lambda: [
            metric('EnvironmentSteps', lambda: state['step']),
            metric('AverageReturn', lambda: 100. if rank == 0 else -1000.)])
        calls = []
        def measure(agent, options, rank, seed_context):
            calls.append(seed_context)
            gathered = [None]*4
            dist.all_gather_object(gathered, float(rank+1))
            maxima, threshold = quantile_rank_maxima([gathered]*100, .33)
            return dict(threshold=threshold)
        with mock.patch('alf.utils.bafcv3_restart.measure_calibration', side_effect=measure):
            for t in range(17):
                state['step'] = t
                assert not BafcAlgorithmV3TR2._local_rollout_skip_proposal(alg)['should_skip']
                after_iteration(agent)
            assert c.activation_step == 16 and alg._eval_trust_max == 4.
            assert BafcAlgorithmV3TR2._local_rollout_skip_proposal(alg)['should_skip']
            # Native runtime restore installs the active threshold and gate.
            saved = c.state_dict()
            c.load_state_dict(saved)
            after_iteration(agent)
            assert len(calls) == 1
            states = [None]*4
            dist.all_gather_object(states, c.state_dict())
            assert all(s == states[0] for s in states)
    finally:
        dist.destroy_process_group()


class AutoSkipTest(unittest.TestCase):
    def test_constant_curve_and_two_checks(self):
        c = AutoSkipController(tiny_settings(), 100)
        for t in range(16):
            records = c.observe(t, 100., 2.)
        self.assertEqual(c.streak, 1)
        self.assertIsNone(c.activation_step)
        r = c.observe(16, 100., 2.)[-1]
        self.assertTrue(r['active'])
        self.assertEqual(c.activation_step, 16)
        self.assertEqual(r['recent_gain'], 0.)

    def test_zero_invalid_and_missing_trust(self):
        for invalid in [None, float('nan'), float('inf'), -1.]:
            c = AutoSkipController(tiny_settings(), 100)
            for t in range(16):c.observe(t, 100., 0.)
            c.observe(16, 100., invalid)
            self.assertEqual(c.streak, 0)
            self.assertIsNone(c.activation_step)
        c = AutoSkipController(tiny_settings(), 100)
        for t in range(17):c.observe(t, 100., 0.)
        self.assertEqual(c.activation_step, 16)
        c = AutoSkipController(tiny_settings(), 100)
        for t in range(16):c.observe(t, 100., 0. if t <= 10 else 1.)
        self.assertEqual(c.streak, 0)

    def test_warmup_and_no_duplicate_samples(self):
        settings = tiny_settings();settings['warmup_fraction'] = .5
        c = AutoSkipController(settings, 100)
        for t in range(51):
            c.observe(t, 100., 1.)
            n = len(c.samples)
            self.assertEqual(c.observe(t, 100., 1.), [])
            self.assertEqual(len(c.samples), n)
        self.assertIsNone(c.activation_step)
        c.observe(51, 100., 1.)
        self.assertEqual(c.activation_step, 51)

    def test_causal_crossing(self):
        c = AutoSkipController(DEFAULTS, 200000)
        c.observe(100100, 10., 1.)
        c.observe(102100, 99., 9.)
        self.assertEqual(c.samples, [(101000, 10., 1.), (102000, 10., 1.)])
        c.observe(103000, 20., 2.)
        self.assertEqual(c.samples[-1], (103000, 20., 2.))
        with self.assertRaises(ValueError):c.observe(102000, 20., 2.)

    def test_return_and_trust_boundaries(self):
        def evaluate(values, trust):
            c = AutoSkipController(tiny_settings(), 100)
            for t in range(16):
                r = c.observe(t, values[max(0,t-1)//5], trust[max(0,t-1)//5])
            return r[-1]
        r = evaluate([100., 110., 121.], [1., 10., 11.])
        self.assertTrue(r['return_pass']);self.assertTrue(r['trust_pass'])
        self.assertFalse(evaluate([100., 110., 122.], [1., 10., 11.])['return_pass'])
        self.assertTrue(evaluate([100., 100., 98.], [1., 10., 11.])['return_pass'])
        self.assertFalse(evaluate([100., 100., 97.9], [1., 10., 11.])['return_pass'])
        self.assertFalse(evaluate([100., 110., 121.], [1., 10., 11.1])['trust_pass'])

    def test_checkpoint_roundtrips(self):
        from alf.utils.bafcv3_restart_test import small_agent
        from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2
        import alf
        alf.set_default_device('cpu');torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as root:
            for steps in [5,16,17]:
                agent = small_agent(BafcAlgorithmV3TR2, root)
                alg = agent._rl_algorithm
                c = AutoSkipController(tiny_settings(), 100)
                for t in range(steps):c.observe(t, 100., 1.)
                if c.activation_step is not None:c.calibration = {'threshold': 42.}
                alg._auto_skip_controller = c
                alg._restart_calibration = {'threshold': 3.}
                state = copy.deepcopy(agent.state_dict())
                other = small_agent(BafcAlgorithmV3TR2, root)
                other._rl_algorithm._auto_skip_controller = AutoSkipController(tiny_settings(), 100)
                other.load_state_dict(state, strict=True)
                restored = other._rl_algorithm
                self.assertEqual(restored._auto_skip_controller.state_dict(), c.state_dict())
                self.assertEqual(restored._enable_eval_rollout_skip_gate, c.calibration is not None)
                self.assertEqual(restored._eval_trust_max, 42. if c.calibration else 3.)

    def test_four_rank_activation(self):
        with tempfile.TemporaryDirectory() as root:
            torch.multiprocessing.spawn(distributed_worker, args=(root,), nprocs=4, join=True)
            audit = json.loads((Path(root)/'auto_skip_activation.json').read_text())
            self.assertEqual(audit['activation_step'], 16)
            self.assertEqual(audit['threshold'], 4.)

    def test_bad_settings_and_missing_returns(self):
        for override in [dict(window_steps=4500), dict(sample_steps=0),
                         dict(consecutive_checks=0), dict(warmup_fraction=1.1),
                         dict(trust_growth_limit=float('nan'))]:
            with self.assertRaises(ValueError):
                AutoSkipController(dict(DEFAULTS, **override), 200000)
        c = AutoSkipController(tiny_settings(), 100)
        for t in range(16):c.observe(t, 100., 1.)
        c.observe(16, None, 1.)
        self.assertEqual(c.streak, 0)
        self.assertIsNone(c.activation_step)

    def test_frozen_activation_calibration_and_legacy_guard(self):
        from alf.utils.bafcv3_restart import measure_calibration, calibrate
        class Learner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('_actor_eval_samples', torch.zeros(1))
                self._restart_calibration = {'threshold': 99.}
            def _compute_eval_trust_metric(self, obs, action):
                return torch.rand(()) + 1.
        agent = torch.nn.Module()
        agent._rl_algorithm = Learner()
        agent._data_transformer = torch.nn.Identity()
        agent._config = SimpleNamespace(mini_batch_size=2, mini_batch_length=2)
        def batch(*args):
            exp = SimpleNamespace(observation=torch.randn(2,2),
                rollout_info=SimpleNamespace(rl=SimpleNamespace(action=torch.randn(2,2))))
            return exp, None
        agent._replay_buffer = SimpleNamespace(get_batch=batch)
        agent.train();agent._rl_algorithm.eval()
        options = dict(source_checkpoint='source', inputs={'source': {'sha256': 'abc'}},
            calibration_seed=1, calibration_repetitions=100,
            threshold_quantile=.33, settings_fingerprint='settings')
        before = torch.get_rng_state().clone()
        with mock.patch('alf.utils.bafcv3_restart.normalize', side_effect=lambda tr,x:x):
            result = measure_calibration(agent, options, 0, seed_context=('auto_skip', 120000))
            again = measure_calibration(agent, options, 0, seed_context=('auto_skip', 120000))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(agent.training);self.assertFalse(agent._rl_algorithm.training)
        self.assertEqual(agent._rl_algorithm._restart_calibration, {'threshold': 99.})
        self.assertEqual(result['rank_values'], again['rank_values'])
        self.assertTrue(result['frozen_state_verified'])
        self.assertEqual(len(result['rank_values']), 100)
        with self.assertRaisesRegex(ValueError, 'Refusing to recalibrate'):
            calibrate(agent, options, 0)

    def test_activation_settings_fingerprinted(self):
        from alf.utils.bafcv3_restart import settings_fingerprint
        legacy = dict(rollout_skipping=False)
        auto = dict(legacy, auto_skip=dict(DEFAULTS))
        changed = copy.deepcopy(auto)
        changed['auto_skip']['window_steps'] = 10000
        self.assertNotEqual(settings_fingerprint(legacy, {}), settings_fingerprint(auto, {}))
        self.assertNotEqual(settings_fingerprint(auto, {}), settings_fingerprint(changed, {}))

    def test_cli_defaults(self):
        from alf.bin.train_bafcv3_tr2_auto_skip import parser
        args = parser().parse_args(['--source-checkpoint','/source','--root-dir','/dest'])
        self.assertEqual(args.auto_skip, DEFAULTS)
        self.assertEqual(args.rollout_skipping, 'off')
        self.assertEqual(args.critic_utd, 11)


if __name__ == '__main__':
    unittest.main()
