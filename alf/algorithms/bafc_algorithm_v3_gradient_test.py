# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
from unittest import mock

import torch

import alf
from alf.algorithms import bafc_algorithm_v3_test as _fixtures
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3, BafcInfo
from alf.utils.checkpoint_utils import Checkpointer


class _NonlinearCritic(torch.nn.Module):
    """A smooth objective with different action/probe gradients per critic."""

    def forward(self, inputs, state=()):
        encoding, (_, action) = inputs
        coefficients = torch.arange(1, action.shape[-2] + 1,
                                    dtype=action.dtype, device=action.device)
        value = (action.sin().sum(-1) + .1 * encoding.square().sum(-1)
                 + .3 * encoding[..., 0] * action[..., 0])
        return value * coefficients, state


class BafcAlgorithmV3GradientTest(alf.test.TestCase):
    # Reuse fixture constructors without inheriting and rerunning its test suite.
    _make_alg = _fixtures.BafcAlgorithmV3CheckpointTest._make_alg
    _make_agent = _fixtures.BafcAlgorithmV3CheckpointTest._make_agent

    def setUp(self):
        super().setUp()
        patcher = mock.patch('alf.algorithms.algorithm.psutil.Process',
                             return_value=_fixtures._DummyProcess())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _gradient_alg(self, **kwargs):
        torch.manual_seed(812)
        alg = self._make_alg(use_single_layer_transformer_encoder=True,
                             actor_encoding_dim=8, **kwargs).double()
        alg._actor_encoder.eval()  # No dropout, including reference forwards.
        alg._critic_networks = _NonlinearCritic()
        return alg

    def _critic_values(self, alg, observation, action, encoding, matching):
        """Evaluate critic/actor pairs explicitly rather than copying gathering."""
        rows = []
        for actor_ids in matching:
            matched_actions = torch.stack([action[:, i] for i in actor_ids], 1)
            matched_encodings = torch.stack([encoding[i] for i in actor_ids], 0)
            matched_encodings = matched_encodings.unsqueeze(0).expand(
                observation.shape[0], -1, -1)
            q_value = alg._critic_networks((matched_encodings,
                                           (observation, matched_actions)))[0]
            rows.append(q_value)
        return torch.stack(rows)

    def _reference_loss(self, alg, observation, mask, valid, matching,
                        clip=None, legacy=False, action_only=False):
        """Direct differentiation or an independent clipped-cotangent reference.

        The action branch averages over replay rows and selected critics. The
        probe branch sums replay rows and averages only actors/probe samples and
        selected critics. LAST and bootstrap masks apply to the action branch.
        """
        action = alg._actor_networks(observation)[0]
        encoding, eval_action, tokens = alg._encode_actor_policies(
            return_tokens=True)
        k = len(matching)
        batch_size, num_actors = action.shape[:2]
        actor_weights = (mask / alg._bootstrap_mask_prob
                         if alg._use_bootstrap_actors else torch.ones_like(mask))
        matched_weights = torch.stack([
            torch.stack([actor_weights[:, i] for i in ids], 1)
            for ids in matching])
        if not legacy and clip is None:
            # Isolate the two paths in F, then differentiate ordinary objectives.
            action_values = self._critic_values(
                alg, observation, action, encoding.detach(), matching)
            action_loss = -(action_values * matched_weights
                            * valid[None, :, None]).sum() / (k * batch_size)
            if action_only:
                return action_loss
            probe_values = self._critic_values(
                alg, observation, action.detach(), encoding, matching)
            return action_loss - probe_values.sum() / (
                k * num_actors * alg._num_actor_eval_samples)

        objective = self._critic_values(
            alg, observation, action, encoding, matching).sum() / k
        leaves = alf.nest.flatten(eval_action)
        if alg._actor_eval_type == 'full':
            leaves = leaves[1:]
        probe_inputs = leaves if legacy else [tokens]
        gradients = torch.autograd.grad(
            objective, [action] + probe_inputs, retain_graph=True)
        if clip is not None:
            gradients = [gradient.clamp(-clip, clip) for gradient in gradients]
        action_loss = -(gradients[0].detach() * action
                        * actor_weights[..., None] * valid[:, None, None]).sum()
        action_loss = action_loss / batch_size
        if action_only:
            return action_loss
        if legacy:
            probe_loss = -sum((gradient.detach() * leaf).sum()
                             for gradient, leaf in zip(gradients[1:], leaves))
        else:
            token_gradient = gradients[1].detach()
            if alg._actor_eval_type == 'full':
                # The direct raw-input path is intentionally absent.
                token_gradient = token_gradient.clone()
                token_gradient[:, :eval_action[0].shape[-1]] = 0
            probe_loss = -(token_gradient * tokens).sum()
        return action_loss + probe_loss / (
            num_actors * alg._num_actor_eval_samples)

    def _actual_gradients(self, alg, observation, mask, valid, matching):
        action = alg._actor_networks(observation)[0]
        with mock.patch.object(alg, '_sample_actor_critic_matchings',
                               return_value=matching):
            _, info = alg._actor_train_step(
                observation, action, torch.zeros_like(action[:, 0]), mask, ())
        loss = ((info.loss * valid).mean()
                + info.extra.eval_action_loss.mean())
        gradients = torch.autograd.grad(
            loss, tuple(alg._actor_networks.parameters()), allow_unused=True)
        self.assertTrue(all(parameter.grad is None
                            for parameter in alg._actor_encoder.parameters()))
        return gradients

    def _assert_gradients_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            if a is None or b is None:
                self.assertIsNone(a)
                self.assertIsNone(b)
            else:
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)

    def test_corrected_all_modes_match_direct_autograd_with_distinct_reductions(self):
        for eval_type in ('full', 'exclude_input', 'last_two', 'output'):
            for pairing in (False, True):
                with self.subTest(eval_type=eval_type, pairing=pairing):
                    alg = self._gradient_alg(
                        actor_eval_type=eval_type, actor_critic_pairing=pairing,
                        num_sampled_critics_for_actor=1 if pairing else 2,
                        use_bootstrap_actors=True)
                    matching = (torch.tensor([[0, 1, 2]]) if pairing else
                                torch.tensor([[2, 0, 1], [1, 2, 0]]))
                    observation = torch.randn(4, 4, dtype=torch.float64)
                    mask = torch.tensor([[1., 0., 1.], [0., 1., 0.],
                                         [1., 1., 0.], [0., 0., 1.]],
                                        dtype=torch.float64)
                    valid = torch.tensor([1., 0., 1., 1.], dtype=torch.float64)
                    actual = self._actual_gradients(
                        alg, observation, mask, valid, matching)
                    reference = self._reference_loss(
                        alg, observation, mask, valid, matching)
                    expected = torch.autograd.grad(
                        reference, tuple(alg._actor_networks.parameters()))
                    self._assert_gradients_close(actual, expected)
                    self.assertGreater(sum(gradient.norm().item()
                                           for gradient in actual), 0)

    def test_clipping_independent_token_partials_and_action_only(self):
        for eval_type in ('full', 'exclude_input', 'last_two', 'output'):
            for action_only in (False, True):
                with self.subTest(eval_type=eval_type, action_only=action_only):
                    alg = self._gradient_alg(
                        actor_eval_type=eval_type, actor_critic_pairing=False,
                        num_sampled_critics_for_actor=2,
                        detach_actor_policy_input=action_only,
                        dqda_clipping=.01, use_bootstrap_actors=True)
                    matching = torch.tensor([[2, 0, 1], [1, 2, 0]])
                    observation = torch.randn(3, 4, dtype=torch.float64)
                    mask = torch.tensor([[1., 0., 1.], [0., 1., 0.],
                                         [1., 1., 0.]], dtype=torch.float64)
                    valid = torch.tensor([1., 0., 1.], dtype=torch.float64)
                    actual = self._actual_gradients(
                        alg, observation, mask, valid, matching)
                    reference = self._reference_loss(
                        alg, observation, mask, valid, matching,
                        clip=.01, action_only=action_only)
                    expected = torch.autograd.grad(
                        reference, tuple(alg._actor_networks.parameters()))
                    self._assert_gradients_close(actual, expected)

    def test_legacy_reproduces_connected_leaf_reinjection(self):
        for eval_type in ('full', 'exclude_input', 'last_two', 'output'):
            with self.subTest(eval_type=eval_type):
                alg = self._gradient_alg(
                    actor_eval_type=eval_type, actor_critic_pairing=False,
                    num_sampled_critics_for_actor=2,
                    use_legacy_actor_gradient=True)
                matching = torch.tensor([[2, 0, 1], [1, 2, 0]])
                observation = torch.randn(3, 4, dtype=torch.float64)
                mask = torch.ones(3, 3, dtype=torch.float64)
                valid = torch.ones(3, dtype=torch.float64)
                actual = self._actual_gradients(
                    alg, observation, mask, valid, matching)
                expected = torch.autograd.grad(
                    self._reference_loss(alg, observation, mask, valid,
                                         matching, legacy=True),
                    tuple(alg._actor_networks.parameters()))
                self._assert_gradients_close(actual, expected)
                corrected = torch.autograd.grad(
                    self._reference_loss(alg, observation, mask, valid, matching),
                    tuple(alg._actor_networks.parameters()))
                difference = sum((a - b).norm().item()
                                 for a, b in zip(actual, corrected))
                if eval_type == 'output':
                    self.assertLess(difference, 1e-8)
                else:
                    self.assertGreater(difference, 1e-7)

    def test_dqde_weight_scales_only_probe_gradient_after_clipping(self):
        for legacy in (False, True):
            for clip in (None, .01):
                with self.subTest(legacy=legacy, clip=clip):
                    kwargs = dict(
                        actor_eval_type='last_two', actor_critic_pairing=False,
                        num_sampled_critics_for_actor=2,
                        use_legacy_actor_gradient=legacy,
                        dqda_clipping=clip, use_bootstrap_actors=True)
                    alg = self._gradient_alg(**kwargs)
                    observation = torch.randn(3, 4, dtype=torch.float64)
                    matching = torch.tensor([[2, 0, 1], [1, 2, 0]])
                    mask = torch.tensor([[1., 0., 1.], [0., 1., 0.],
                                         [1., 1., 0.]], dtype=torch.float64)
                    valid = torch.tensor([1., 0., 1.], dtype=torch.float64)
                    parameters = tuple(alg._actor_networks.parameters())
                    action = torch.autograd.grad(
                        self._reference_loss(
                            alg, observation, mask, valid, matching,
                            clip=clip, legacy=legacy, action_only=True),
                        parameters)
                    total = torch.autograd.grad(
                        self._reference_loss(
                            alg, observation, mask, valid, matching,
                            clip=clip, legacy=legacy), parameters)
                    probe = tuple(t - a for t, a in zip(total, action))
                    self.assertGreater(sum(g.norm().item() for g in probe), 0)
                    default = self._actual_gradients(
                        alg, observation, mask, valid, matching)
                    self._assert_gradients_close(default, total)
                    for weight in (0., .5, 1.):
                        with self.subTest(weight=weight):
                            weighted_alg = self._gradient_alg(
                                dqde_weight=weight, **kwargs)
                            actual = self._actual_gradients(
                                weighted_alg, observation, mask, valid, matching)
                            expected = tuple(a + weight * p
                                             for a, p in zip(action, probe))
                            self._assert_gradients_close(actual, expected)
                            if weight == 1.:
                                self._assert_gradients_close(actual, default)

    def test_dqde_weight_scales_gradient_chain_probe_branches(self):
        for legacy in (False, True):
            branches = []
            for weight in (1., .5):
                with self.subTest(legacy=legacy, weight=weight):
                    alg = self._gradient_alg(
                        actor_eval_type='last_two', dqde_weight=weight,
                        use_legacy_actor_gradient=legacy,
                        debug_summaries=True, debug_gradient_chain=True)
                    observation = torch.randn(3, 4, dtype=torch.float64)
                    action = alg._actor_networks(observation)[0]
                    with mock.patch.object(
                            alf.summary, 'should_record_summaries',
                            return_value=True), \
                         mock.patch.object(alf.summary, 'scalar'), \
                         mock.patch.object(alf.summary, 'histogram'):
                        _, info = alg._actor_train_step(
                            observation, action, torch.zeros_like(action[:, 0]),
                            torch.ones(3, 3, dtype=torch.float64), ())
                    losses = alg._gradient_chain_pending
                    parameters = tuple(alg._actor_networks.parameters())
                    gradients = [torch.autograd.grad(
                        loss.mean(), parameters, retain_graph=True,
                        allow_unused=True, materialize_grads=True)
                                 for loss in losses]
                    actual_probe = torch.autograd.grad(
                        info.extra.eval_action_loss.mean(), parameters)
                    self._assert_gradients_close(
                        actual_probe, tuple(h + o for h, o in
                                            zip(gradients[1], gradients[2])))
                    branches.append(gradients)
            for i, (unweighted, weighted) in enumerate(zip(*branches)):
                scale = 1. if i == 0 else .5
                self._assert_gradients_close(
                    weighted, tuple(scale * gradient for gradient in unweighted))

    def test_dqde_weight_rejects_negative_and_nonfinite_values(self):
        for weight in (-.5, float('nan'), float('inf'), float('-inf')):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, 'dqde_weight'):
                    self._make_alg(dqde_weight=weight)

    def test_full_mode_excludes_direct_raw_probe_path(self):
        alg = self._gradient_alg(actor_eval_type='full')
        observation = torch.randn(3, 4, dtype=torch.float64)
        action = alg._actor_networks(observation)[0]
        matching = torch.tensor([[0, 1, 2]])
        with mock.patch.object(alg, '_sample_actor_critic_matchings',
                               return_value=matching):
            _, info = alg._actor_train_step(
                observation, action, torch.zeros(3, 2, dtype=torch.float64),
                torch.ones(3, 3, dtype=torch.float64), ())
        actual = torch.autograd.grad(info.extra.eval_action_loss.mean(),
                                     alg._actor_eval_samples)[0]
        leaves = alg._actor_networks(alg._actor_eval_samples,
                                     full_neurons=True)[0]
        # Differentiate a real objective with only the raw-input edge detached.
        tokens = torch.cat([leaves[0].detach(), *leaves[1:]], dim=-1).permute(1, 2, 0)
        encoding = alg._actor_encoder(tokens)[0]
        loss = -self._critic_values(alg, observation, action.detach(), encoding,
                                    matching).sum() / (3 * alg._num_actor_eval_samples)
        expected = torch.autograd.grad(loss, alg._actor_eval_samples)[0]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
        # An uncut objective must differ, otherwise this would not test the edge.
        encoding = alg._encode_actor_policies()[0]
        uncut_loss = -self._critic_values(alg, observation, action.detach(), encoding,
                                         matching).sum() / (3 * alg._num_actor_eval_samples)
        uncut = torch.autograd.grad(uncut_loss, alg._actor_eval_samples)[0]
        self.assertGreater((uncut - expected).norm().item(), 1e-8)

    def test_centered_surrogate_large_finite_cotangent(self):
        for dtype in (torch.float32, torch.float64):
            value = torch.tensor([1e10, -1e10], dtype=dtype, requires_grad=True)
            cotangent = torch.tensor([3e27, -4e27], dtype=dtype,
                                     requires_grad=True)
            loss = BafcAlgorithmV3._linear_gradient_surrogate(cotangent, value)
            self.assertTensorEqual(loss, torch.zeros_like(loss))
            loss.sum().backward()
            self.assertTensorEqual(value.grad, -cotangent.detach())
            self.assertIsNone(cotangent.grad)

    def test_target_encoder_copy_preserves_initialization_rng_and_optimizer(self):
        torch.manual_seed(701)
        original = self._make_alg()
        rng = torch.get_rng_state()
        torch.manual_seed(701)
        target = self._make_alg(use_target_actor_encoder=True)
        self.assertTensorEqual(rng, torch.get_rng_state())
        for name, value in original.state_dict().items():
            self.assertTensorEqual(value, target.state_dict()[name])
        for online, frozen in zip(target._actor_encoder.parameters(),
                                  target._target_actor_encoder.parameters()):
            self.assertTensorEqual(online, frozen)
            self.assertNotEqual(online.data_ptr(), frozen.data_ptr())
            self.assertFalse(frozen.requires_grad)
        with self.assertRaisesRegex(AssertionError, 'use_actor_id_encoding'):
            self._make_alg(use_target_actor_encoder=True,
                           use_actor_id_encoding=True)
        agent = self._make_agent(use_target_actor_encoder=True)
        agent._default_optimizer = alf.optimizers.Adam(lr=1e-3)
        agent._setup_optimizers()
        owned = {id(parameter) for optimizer in agent.optimizers()
                 for group in optimizer.param_groups for parameter in group['params']}
        self.assertTrue(all(id(parameter) not in owned for parameter in
                            agent._rl_algorithm._target_actor_encoder.parameters()))

    def test_target_encoder_reuses_detached_tokens_and_has_no_gradients(self):
        alg = self._make_alg(use_target_actor_encoder=True)
        alg._actor_encoder.eval()
        alg._target_actor_encoder.eval()
        observation = torch.randn(4, 4)
        action = alg._actor_networks(observation)[0].reshape(-1, 2)
        captured = {}
        with torch.no_grad():
            for parameter in alg._target_actor_encoder.parameters():
                parameter.add_(.05)

        def capture_output(module, inputs, output):
            captured['target_output'] = output[0]

        def capture_critic_input(module, inputs):
            captured['target_critic_encoding'] = inputs[0][0]

        def capture(name):
            def hook(module, inputs):
                captured[name] = inputs[0]
            return hook

        handles = [alg._actor_encoder.register_forward_pre_hook(capture('online')),
                   alg._target_actor_encoder.register_forward_pre_hook(capture('target')),
                   alg._target_actor_encoder.register_forward_hook(capture_output),
                   alg._target_critic_networks.register_forward_pre_hook(capture_critic_input)]
        try:
            with mock.patch.object(alg._actor_networks, 'forward',
                                   wraps=alg._actor_networks.forward) as forward:
                _, info = alg._critic_train_step(
                    observation, alg.get_initial_train_state(4).critic,
                    BafcInfo(action=torch.zeros(4, 2)), action)
            self.assertEqual(forward.call_count, 1)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTensorEqual(captured['online'], captured['target'])
        self.assertEqual(captured['online'].data_ptr(), captured['target'].data_ptr())
        self.assertFalse(captured['target'].requires_grad)
        self.assertTensorEqual(captured['target_critic_encoding'],
                               captured['target_output'].repeat(4, 1))
        self.assertFalse(torch.equal(captured['target_output'],
                                     alg._actor_encoder(captured['online'])[0]))
        self.assertFalse(info.target_critic.requires_grad)
        info.critic.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in
                            alg._actor_encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in
                            alg._target_actor_encoder.parameters()))

    def test_target_encoder_polyak_period(self):
        alg = self._make_alg(use_target_actor_encoder=True,
                             target_critic_tau=.25, target_critic_period=2)
        before = [parameter.clone() for parameter in
                  alg._target_actor_encoder.parameters()]
        with torch.no_grad():
            for parameter in alg._actor_encoder.parameters():
                parameter.add_(2.)
        alg._update_target_critic()
        for initial, target in zip(before, alg._target_actor_encoder.parameters()):
            self.assertTensorEqual(initial, target)
        alg._update_target_critic()
        for initial, online, target in zip(
                before, alg._actor_encoder.parameters(),
                alg._target_actor_encoder.parameters()):
            self.assertTensorClose(target, .75 * initial + .25 * online)

    def test_target_encoder_checkpoint_continuation_including_delayed_updates(self):
        for delayed in (False, True):
            with self.subTest(delayed=delayed):
                kwargs = dict(use_target_actor_encoder=True,
                              target_critic_period=3, target_critic_tau=.25,
                              target_critic_use_ema=delayed)
                source = self._make_alg(**kwargs)
                with torch.no_grad():
                    for parameter in source._actor_encoder.parameters():
                        parameter.add_(.2)
                source._update_target_critic()
                source._update_target_critic()
                with tempfile.TemporaryDirectory() as directory:
                    Checkpointer(directory, algorithm=source).save(2)
                    restored = self._make_alg(**kwargs)
                    Checkpointer(directory, algorithm=restored).load(2, strict=True)
                self.assertEqual(restored._update_target_critic._counter, 2)
                for _ in range(7):
                    with torch.no_grad():
                        for alg in (source, restored):
                            for network in (alg._actor_encoder, alg._critic_networks):
                                for parameter in network.parameters():
                                    parameter.add_(.1)
                            alg._update_target_critic()
                    self.assertEqual(source._update_target_critic._counter,
                                     restored._update_target_critic._counter)
                    source_state, restored_state = (source.state_dict(),
                                                     restored.state_dict())
                    self.assertEqual(set(source_state), set(restored_state))
                    for a, b in zip(alf.nest.flatten(source_state),
                                    alf.nest.flatten(restored_state)):
                        if isinstance(a, torch.Tensor):
                            self.assertTensorEqual(a, b)
                        else:
                            self.assertEqual(a, b)
                self.assertTrue(all(not parameter.requires_grad for parameter in
                                    restored._target_actor_encoder.parameters()))


if __name__ == '__main__':
    alf.test.main()
