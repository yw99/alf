# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
"""Reference-equivalence tests for V7-only computation reuse."""
import copy
import itertools
import unittest

import torch
import alf
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.networks.bafc_v7_critic_network import BafcV7FuncCriticNetwork
from alf.tensor_specs import TensorSpec

from alf.utils.bafcv7_benchmark_utils import (
    FLAGS, make_algorithm, make_batch, compute_loss)


class BafcV7OptimizationTest(unittest.TestCase):
    def assert_close(self, a, b):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6)

    def compare_gradients(self, ref, fast):
        for (name, a), (other, b) in zip(ref.named_parameters(), fast.named_parameters()):
            self.assertEqual(name, other)
            if a.grad is None or b.grad is None:
                self.assertIs(a.grad, b.grad, name)
            else:
                self.assert_close(a.grad, b.grad)

    def test_update_sequence_and_optimizer_checkpoint(self):
        for variant, mode, source in itertools.product(
                ['ensemble_base', 'single_seeded'], ['mean_log_std', 'action_quantiles'],
                ['frozen', 'trainable']):
            with self.subTest(variant=variant, mode=mode, source=source):
                torch.manual_seed(12)
                ref = make_algorithm(False, variant, mode, source).double()
                fast = make_algorithm(True, variant, mode, source).double()
                fast.load_state_dict(copy.deepcopy(ref.state_dict()), strict=True)
                self.assertEqual(list(ref.state_dict()), list(fast.state_dict()))
                self.assertTrue(fast._critic_networks.supports_v7_fast_paths)
                ro = torch.optim.Adam(ref.parameters(), lr=3e-4)
                fo = torch.optim.Adam(fast.parameters(), lr=3e-4)
                # Includes initial joint, three critics, actor and subsequent cycles.
                for step in range(12):
                    batch = make_batch(ref, unique=[1, 2, 8][step % 3])
                    ro.zero_grad(set_to_none=True)
                    fo.zero_grad(set_to_none=True)
                    rng = torch.get_rng_state()
                    lr, ir = compute_loss(ref, batch)
                    torch.set_rng_state(rng)
                    lf, iff = compute_loss(fast, batch)
                    self.assert_close(lr, lf)
                    lr.backward()
                    lf.backward()
                    self.compare_gradients(ref, fast)
                    ro.step()
                    fo.step()
                    ref.after_update(None, ir)
                    fast.after_update(None, iff)
                    for a, b in zip(ref.parameters(), fast.parameters()):
                        self.assert_close(a, b)
                # Load both directions; optimizer indices/parameter order are unchanged.
                ref.load_state_dict(copy.deepcopy(fast.state_dict()), strict=True)
                ro.load_state_dict(copy.deepcopy(fo.state_dict()))
                fast.load_state_dict(copy.deepcopy(ref.state_dict()), strict=True)
                fo.load_state_dict(copy.deepcopy(ro.state_dict()))
                for pa, pb in zip(ro.state.values(), fo.state.values()):
                    for key in pa:
                        self.assert_close(pa[key], pb[key])
                self.assertIsNone(fast._probe_cache)

    def test_float32_losses_and_gradients(self):
        for variant, mode, source, phase in itertools.product(
                ['ensemble_base', 'single_seeded'], ['mean_log_std', 'action_quantiles'],
                ['frozen', 'trainable'], ['initial', 'critic', 'actor']):
            with self.subTest(variant=variant, mode=mode, source=source, phase=phase):
                torch.manual_seed(73)
                ref = make_algorithm(False, variant, mode, source)
                torch.manual_seed(73)
                fast = make_algorithm(True, variant, mode, source)
                for a,b in zip(ref.parameters(), fast.parameters()):
                    torch.testing.assert_close(a,b, rtol=0, atol=0)
                for alg in (ref, fast):
                    if phase != 'initial':
                        alg._critic_update_counter=1
                        alg._train_mode=TrainMode.actor if phase=='actor' else TrainMode.critic
                        alg._apply_train_mode_grad_flags()
                batch=make_batch(ref, unique=2)
                rng=torch.get_rng_state()
                lr,_=compute_loss(ref,batch)
                torch.set_rng_state(rng)
                lf,_=compute_loss(fast,batch)
                self.assert_close(lr,lf)
                lr.backward()
                lf.backward()
                self.compare_gradients(ref,fast)

    def test_switches_independently(self):
        for flag, variant in itertools.product(
                FLAGS, ['ensemble_base', 'single_seeded']):
            with self.subTest(flag=flag, variant=variant):
                ref = make_algorithm(False, variant)
                fast = make_algorithm(True, variant, flags=[flag])
                fast.load_state_dict(copy.deepcopy(ref.state_dict()))
                for alg in (ref, fast):
                    alg._critic_update_counter = 1
                    alg._apply_train_mode_grad_flags()
                batch = make_batch(ref, unique=2)
                rng = torch.get_rng_state()
                lr, _ = compute_loss(ref, batch)
                torch.set_rng_state(rng)
                lf, _ = compute_loss(fast, batch)
                self.assert_close(lr, lf)
                lr.backward()
                lf.backward()
                self.compare_gradients(ref, fast)

    def test_unsupported_critic_falls_back(self):
        net = BafcV7FuncCriticNetwork(
            (TensorSpec((8,)), (TensorSpec((4,)), TensorSpec((2,)))),
            obs_action_joint_fc_layer_params=(8,),
            actor_obs_action_joint_fc_layer_params=(8,),
            actor_obs_action_combiner=alf.layers.NestConcat(dim=-1),
            use_fc_bn=True).make_parallel(3)
        self.assertFalse(net.supports_v7_fast_paths)
        net((torch.randn(4,3,8),(torch.randn(4,3,4),torch.randn(4,3,2))))

    def test_cache_lifecycle(self):
        alg = make_algorithm(True)
        alg._critic_update_counter = 1
        alg._apply_train_mode_grad_flags()
        a = alg._probe_output(alg._actor_eval_samples)
        compute_loss(alg, make_batch(alg))
        self.assertIs(a, alg._probe_output(alg._actor_eval_samples))
        self.assertIsNone(a.mean.grad_fn)
        self.assertTrue(all(not t.requires_grad for t in a.neurons))
        with torch.no_grad():
            next(alg._actor_networks.parameters()).add_(.01)
        b = alg._probe_output(alg._actor_eval_samples)
        self.assertIsNot(a, b)
        alg.load_state_dict(copy.deepcopy(alg.state_dict()))
        self.assertIsNone(alg._probe_cache)
        alg._probe_output(alg._actor_eval_samples)
        alg.to(dtype=torch.float64)
        self.assertIsNone(alg._probe_cache)
        alg._train_mode = TrainMode.actor
        alg._apply_train_mode_grad_flags()
        c = alg._probe_output(alg._actor_eval_samples)
        self.assertIsNotNone(c.mean.grad_fn)
        self.assertIsNone(alg._probe_cache)

    def test_dedup_counts_and_dropout_fallback(self):
        for dropout in [0., .1]:
            alg = make_algorithm(True, dropout=dropout)
            alg._critic_update_counter = 1
            alg._apply_train_mode_grad_flags()
            compute_loss(alg, make_batch(alg, unique=2))
            self.assertEqual(alg._last_seed_counts, (2,8) if dropout==0 else None)
            if dropout==0:
                compute_loss(alg, make_batch(alg, unique=8))
                self.assertEqual(alg._last_seed_counts, (8,8))

    def test_critic_operations(self):
        alg = make_algorithm(True, variant='ensemble_base')
        net = alg._critic_networks
        b, a, c = 5, 3, 3
        enc = torch.randn(b, a, 8, requires_grad=True)
        obs = torch.randn(b, 4, requires_grad=True)
        action = torch.randn(b, a, 2, requires_grad=True)
        full = net.actor_critic_product(enc, obs, action)
        ids = torch.tensor([2,0])
        self.assert_close(net.selected_target_values(enc, obs, action, ids),
                          full.index_select(2, ids))
        self.assert_close(net.paired_actor_values(enc, obs, action),
                          full[:, torch.arange(a), torch.arange(c)])
        shared = action[:,0]
        self.assert_close(net.actor_critic_product(enc, obs, shared, True),
                          net.actor_critic_product(enc, obs, shared, False))
        net.selected_target_values(enc, obs, action, ids).sum().backward()
        for name, p in net.named_parameters():
            if p.grad is not None:
                if '._ln.' in name:
                    self.assertEqual(p.grad.reshape(c,-1)[1].abs().sum().item(), 0)
                else:
                    self.assertEqual(p.grad[1].abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
