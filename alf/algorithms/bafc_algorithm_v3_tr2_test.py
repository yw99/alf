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
from functools import partial
from types import SimpleNamespace
from unittest import mock

import torch

import alf
from alf.algorithms.bafc_algorithm_v3_tr2 import (BafcActorInfo,
                                                  BafcAlgorithmV3TR2,
                                                  BafcCriticInfo, BafcInfo,
                                                  BafcState)
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import LossInfo, StepType, TimeStep
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.nest import utils as nest_utils
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.schedulers import update_progress


class _DummyProcess:

    def memory_info(self):
        return SimpleNamespace(rss=0)

    def children(self, recursive=True):
        del recursive
        return []


class _LinearParallelCritic(torch.nn.Module):

    def __init__(self, slopes, encoding_slopes=None):
        super().__init__()
        self.register_buffer("_slopes", torch.as_tensor(slopes))
        if encoding_slopes is None:
            encoding_slopes = torch.zeros(self._slopes.shape[0])
        self.register_buffer("_encoding_slopes",
                             torch.as_tensor(encoding_slopes))

    def forward(self, inputs, state=()):
        actor_encoding, (_, action) = inputs
        q_value = (action * self._slopes.unsqueeze(0)).sum(dim=-1)
        q_value = q_value + (actor_encoding[..., 0] *
                             self._encoding_slopes.unsqueeze(0))
        return q_value, state


class _FirstTokenEncoder(torch.nn.Module):

    def forward(self, actor_tokens, state=()):
        return actor_tokens[:, 0, :1], state


class BafcAlgorithmV3TR2Test(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._psutil_patcher = mock.patch(
            "alf.algorithms.algorithm.psutil.Process",
            return_value=_DummyProcess())
        self._psutil_patcher.start()
        self.addCleanup(self._psutil_patcher.stop)
        update_progress("env_steps", 0)
        self.addCleanup(update_progress, "env_steps", 0)

    def _make_alg(self, **kwargs):
        num_actor_critic = kwargs.pop("num_actor_critic", 3)
        reward_spec = kwargs.pop("reward_spec", TensorSpec(()))
        reward_weights = kwargs.pop("reward_weights", None)
        num_updates_per_train_iter = kwargs.pop("num_updates_per_train_iter",
                                                3)
        num_env_steps = kwargs.pop("num_env_steps", 0)
        actor_network_cls = kwargs.pop(
            "actor_network_cls",
            partial(ActorFCNetwork, fc_layer_params=(32, 32)))
        critic_network_cls = kwargs.pop(
            "critic_network_cls",
            partial(
                FuncCriticNetwork,
                obs_action_joint_fc_layer_params=(32, 32),
                actor_obs_action_joint_fc_layer_params=(32, 32)))
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_tr2_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_env_steps=num_env_steps,
            num_updates_per_train_iter=num_updates_per_train_iter)
        kwargs.setdefault("trust_metric_num_obs", 8)
        return BafcAlgorithmV3TR2(
            observation_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec((2, ), minimum=-1.0, maximum=1.0),
            reward_spec=reward_spec,
            reward_weights=reward_weights,
            config=config,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            actor_encoder_cls=partial(
                TransformerEncoder, num_layers=2, num_attention_heads=1),
            num_actor_critic=num_actor_critic,
            num_actor_eval_samples=16,
            **kwargs)

    def _make_train_time_step(self, observation=None):
        if observation is None:
            observation = torch.randn(4, 4)
        batch_size = observation.shape[0]
        return TimeStep(
            step_type=torch.full((batch_size, ), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(batch_size),
            discount=torch.ones(batch_size),
            observation=observation,
            prev_action=(),
            env_id=())

    def _set_env_steps(self, alg, env_steps):
        del alg
        update_progress("env_steps", env_steps)

    def _clone_state_dict(self, module):
        return {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }

    def _assert_state_dict_equal(self, module, expected_state):
        actual_state = module.state_dict()
        self.assertEqual(set(actual_state.keys()), set(expected_state.keys()))
        for name, value in actual_state.items():
            self.assertTensorClose(value, expected_state[name])

    def _fill_module(self, module, fill_value):
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(fill_value)

    def test_initialization_smoke_and_config_name(self):
        alg = self._make_alg()
        self.assertIsInstance(alg, BafcAlgorithmV3TR2)
        self.assertEqual(BafcAlgorithmV3TR2.__name__, "BafcAlgorithmV3TR2")
        self.assertEqual(alg._num_actor_critic, 3)
        self.assertEqual(alg._trust_metric_target_obs_cache_size, 32)
        self.assertFalse(alg._enable_critic_reweighting)
        self.assertEqual(alg._critic_reweighting_solver_iters, 5)

    def test_num_sampled_critics_validation(self):
        with self.assertRaisesRegex(AssertionError, "between 1"):
            self._make_alg(
                actor_critic_pairing=False,
                num_sampled_critics_for_actor=0)
        with self.assertRaisesRegex(AssertionError, "between 1"):
            self._make_alg(
                actor_critic_pairing=False,
                num_sampled_critics_for_actor=4)
        with self.assertRaisesRegex(AssertionError, "pairing is True"):
            self._make_alg(num_sampled_critics_for_actor=2)

    def test_num_sampled_critic_targets_validation(self):
        with self.assertRaisesRegex(AssertionError, "between 1"):
            self._make_alg(num_sampled_critic_targets=0)
        with self.assertRaisesRegex(AssertionError, "between 1"):
            self._make_alg(num_sampled_critic_targets=4)

    def test_disabled_random_critic_targets_preserve_targets_and_rng(self):
        alg = self._make_alg()
        targets = torch.randn(2, 3, 3)
        with mock.patch("torch.randperm") as randperm:
            selected = alg._select_critic_targets(targets)
        self.assertIs(selected, targets)
        randperm.assert_not_called()

    def test_random_single_critic_target_is_shared_by_all_losses(self):
        alg = self._make_alg(
            use_random_critic_targets=True,
            num_sampled_critic_targets=1)
        targets = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        with mock.patch(
                "torch.randperm", return_value=torch.tensor([2, 0, 1])):
            selected = alg._select_critic_targets(targets)
        self.assertTensorEqual(selected, targets[:, :, 2])

        captured_targets = []

        def _loss(**kwargs):
            captured_targets.append(kwargs["target_value"])
            return SimpleNamespace(loss=torch.zeros(1, 1, 3))

        alg._critic_losses = [mock.Mock(side_effect=_loss) for _ in range(3)]
        shared_target = selected[:1].unsqueeze(0)
        info = BafcInfo(
            critic=BafcCriticInfo(
                critic=torch.zeros(1, 1, 3, 3),
                target_critic=shared_target))
        alg._calc_critic_loss(info)
        self.assertEqual(len(captured_targets), 3)
        for target in captured_targets:
            self.assertIs(target, shared_target)

    def test_shared_random_target_keeps_critic_sample_weighting(self):
        alg = self._make_alg(use_random_critic_targets=True)
        t, b, n = 2, 3, alg._num_actor_critic
        sample_weight = torch.tensor([[0.5, 1.0, 1.5],
                                      [2.0, 0.25, 0.75]])
        shared_target = torch.zeros(t, b, n)
        captured_targets = []

        class _UnitLoss:

            def __call__(self, info, value, target_value):
                del info, value
                captured_targets.append(target_value)
                return LossInfo(loss=torch.ones(t, b, n))

        alg._critic_losses = [_UnitLoss() for _ in range(n)]
        info = BafcInfo(
            critic=BafcCriticInfo(
                critic=torch.zeros(t, b, n, n),
                target_critic=shared_target,
                critic_sample_weight=sample_weight),
            bootstrap_mask=torch.ones(t, b, n))

        loss = alg._calc_critic_loss(info)

        self.assertEqual(len(captured_targets), n)
        for target in captured_targets:
            self.assertIs(target, shared_target)
        self.assertTensorClose(
            loss.loss,
            sample_weight.unsqueeze(-1).expand(t, b, n) * float(n))

    def test_multiple_random_critic_targets_use_subset_min(self):
        alg = self._make_alg(
            use_random_critic_targets=True,
            num_sampled_critic_targets=2)
        targets = torch.tensor([[[3., 1., 2.], [6., 5., 4.]]])
        with mock.patch(
                "torch.randperm", return_value=torch.tensor([0, 2, 1])):
            selected = alg._select_critic_targets(targets)
        self.assertTensorEqual(selected, torch.tensor([[2., 4.]]))

    def test_all_random_critic_targets_use_full_min(self):
        alg = self._make_alg(
            use_random_critic_targets=True,
            num_sampled_critic_targets=3)
        targets = torch.tensor([[[3., 1., 2.], [6., 5., 4.]]])
        with mock.patch("torch.randperm") as randperm:
            selected = alg._select_critic_targets(targets)
        self.assertTensorEqual(selected, torch.tensor([[1., 4.]]))
        randperm.assert_not_called()

    def test_random_critic_targets_preserve_signed_reward_dimensions(self):
        alg = self._make_alg(
            reward_spec=TensorSpec((2, )),
            reward_weights=[1., -1.],
            use_random_critic_targets=True,
            num_sampled_critic_targets=2)
        targets = torch.tensor([[[[1., 10.], [2., 5.], [0., 7.]]]])
        with mock.patch(
                "torch.randperm", return_value=torch.tensor([0, 1, 2])):
            selected = alg._select_critic_targets(targets)
        self.assertEqual(selected.shape, (1, 1, 2))
        self.assertTensorEqual(selected, torch.tensor([[[1., 10.]]]))

    def test_balanced_actor_critic_matchings(self):
        alg = self._make_alg(
            actor_critic_pairing=False,
            num_sampled_critics_for_actor=2)
        matching = alg._sample_actor_critic_matchings()
        self.assertEqual(matching.shape, (2, 3))
        for row in matching:
            self.assertTensorEqual(row.sort().values, torch.arange(3))
        for actor_id in range(3):
            critic_ids = torch.nonzero(
                matching == actor_id, as_tuple=False)[:, 1]
            self.assertEqual(torch.unique(critic_ids).numel(), 2)

        matched_value = torch.arange(6).reshape(2, 1, 3)
        actor_order_value = alg._restore_actor_order(matched_value, matching)
        for k in range(2):
            for critic_id in range(3):
                self.assertEqual(
                    actor_order_value[k, 0, matching[k, critic_id]],
                    matched_value[k, 0, critic_id])

        all_alg = self._make_alg(
            actor_critic_pairing=False,
            num_sampled_critics_for_actor=3)
        all_matching = all_alg._sample_actor_critic_matchings()
        pairs = {(int(all_matching[k, critic_id]), critic_id)
                 for k in range(3) for critic_id in range(3)}
        self.assertEqual(len(pairs), 9)

        one_alg = self._make_alg(actor_critic_pairing=False)
        torch.manual_seed(7)
        expected = torch.randperm(3)
        torch.manual_seed(7)
        self.assertTensorEqual(
            one_alg._sample_actor_critic_matchings(), expected.unsqueeze(0))

        fixed_alg = self._make_alg()
        with mock.patch("torch.randperm") as randperm:
            fixed_matching = fixed_alg._sample_actor_critic_matchings()
        self.assertTensorEqual(fixed_matching, torch.arange(3).unsqueeze(0))
        randperm.assert_not_called()

    def test_scattered_gradient_matches_direct_gradient(self):
        alg = self._make_alg(
            actor_critic_pairing=False,
            num_sampled_critics_for_actor=2)
        action = torch.randn(2, 3, 2, requires_grad=True)
        matching = torch.tensor([[2, 0, 1], [1, 2, 0]])
        matched_action = torch.gather(
            action.unsqueeze(0).expand(2, *action.shape), 2,
            matching[:, None, :, None].expand(2, 2, 3, 2))
        slopes = torch.tensor([[1., 2.], [3., 5.], [7., 11.]])
        objective = (matched_action * slopes[None, None]).sum() / 2

        direct_dqda = torch.autograd.grad(
            objective, action, retain_graph=True)[0]
        matched_dqda = torch.autograd.grad(objective, matched_action)[0]
        scattered_dqda = alg._aggregate_matched_action_gradients(
            matched_dqda, matching, action)
        self.assertTensorClose(scattered_dqda, direct_dqda)

    def test_all_critics_produce_exact_mean_actor_gradient(self):
        slopes = torch.tensor([[1., 2.], [3., 5.], [8., 13.]])
        alg = self._make_alg(
            actor_critic_pairing=False,
            num_sampled_critics_for_actor=3,
            actor_eval_type='output')
        alg._critic_networks = _LinearParallelCritic(slopes)
        action = torch.randn(2, 3, 2, requires_grad=True)
        _, actor_info = alg._actor_train_step(
            torch.randn(2, 4), action, torch.zeros(2, 2), torch.ones(2, 3), ())

        actor_loss_grad = torch.autograd.grad(actor_info.loss.sum(), action)[0]
        expected = -slopes.mean(dim=0).reshape(1, 1, 2).expand_as(action)
        self.assertTensorClose(actor_loss_grad, expected)
        self.assertTensorClose(actor_info.extra.grad_trust_metric,
                               torch.ones(2))

    def test_all_critics_produce_exact_mean_functional_gradient(self):
        encoding_slopes = torch.tensor([2., 5., 11.])
        alg = self._make_alg(
            actor_critic_pairing=False,
            num_sampled_critics_for_actor=3,
            actor_eval_type='output')
        alg._actor_encoder = _FirstTokenEncoder()
        alg._critic_networks = _LinearParallelCritic(
            torch.zeros(3, 2), encoding_slopes=encoding_slopes)
        observation = torch.randn(2, 4)
        action = alg._actor_networks(observation)[0]
        original_grad = nest_utils.grad
        captured = {}

        def _capture_grad(*args, **kwargs):
            result = original_grad(*args, **kwargs)
            captured['dqde'] = result[1]
            return result

        with mock.patch(
                "alf.algorithms.bafc_algorithm_v3_tr2.nest_utils.grad",
                side_effect=_capture_grad):
            alg._actor_train_step(observation, action, torch.zeros(2, 2),
                                  torch.ones(2, 3), ())

        expected = torch.zeros_like(captured['dqde'])
        expected[0, :, 0] = observation.shape[0] * encoding_slopes.mean()
        self.assertTensorClose(captured['dqde'], expected)

    def test_k1_random_pairing_preserves_gradient_and_loss(self):
        slopes = torch.tensor([[1., 2.], [3., 5.], [8., 13.]])
        alg = self._make_alg(
            actor_critic_pairing=False, actor_eval_type='output')
        alg._critic_networks = _LinearParallelCritic(slopes)
        matching = torch.tensor([[2, 0, 1]])
        action = torch.randn(2, 3, 2, requires_grad=True)
        with mock.patch.object(
                alg,
                "_sample_actor_critic_matchings",
                return_value=matching):
            _, actor_info = alg._actor_train_step(
                torch.randn(2, 4), action, torch.zeros(2, 2),
                torch.ones(2, 3), ())

        actor_slopes = torch.empty_like(slopes)
        for critic_id, actor_id in enumerate(matching[0]):
            actor_slopes[actor_id] = slopes[critic_id]
        actor_loss_grad = torch.autograd.grad(actor_info.loss.sum(), action)[0]
        self.assertTensorClose(actor_loss_grad,
                               -actor_slopes.unsqueeze(0).expand_as(action))
        expected_loss = 0.5 * actor_slopes.square().sum()
        self.assertTensorClose(actor_info.loss,
                               expected_loss.expand(action.shape[0]))
        self.assertTensorClose(actor_info.extra.eval_action_loss,
                               torch.zeros(action.shape[0]))

    def test_random_pairing_keeps_bootstrap_mask_in_actor_order(self):
        slopes = torch.tensor([[1., 2.], [3., 5.], [8., 13.]])
        alg = self._make_alg(
            actor_critic_pairing=False,
            use_bootstrap_actors=True,
            actor_eval_type='output')
        alg._critic_networks = _LinearParallelCritic(slopes)
        matching = torch.tensor([[2, 0, 1]])
        action = torch.randn(2, 3, 2, requires_grad=True)
        mask = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
        with mock.patch.object(
                alg,
                "_sample_actor_critic_matchings",
                return_value=matching):
            _, actor_info = alg._actor_train_step(
                torch.randn(2, 4), action, torch.zeros(2, 2), mask, ())

        actor_loss_grad = torch.autograd.grad(actor_info.loss.sum(), action)[0]
        actor_slopes = torch.empty_like(slopes)
        for critic_id, actor_id in enumerate(matching[0]):
            actor_slopes[actor_id] = slopes[critic_id]
        expected = (-actor_slopes.unsqueeze(0) * mask.unsqueeze(-1) /
                    alg._bootstrap_mask_prob)
        self.assertTensorClose(actor_loss_grad, expected)

    def test_multi_critic_actor_step_uses_one_forward_and_vjp(self):
        alg = self._make_alg(
            actor_critic_pairing=False,
            num_sampled_critics_for_actor=2,
            actor_eval_type='last_two',
            dqda_clipping=0.1,
            debug_summaries=True)
        observation = torch.randn(2, 4)
        action = alg._actor_networks(observation)[0]
        with mock.patch.object(
                alf.summary, "should_record_summaries", return_value=True), \
             mock.patch.object(alf.summary, "scalar") as scalar_mock, \
             mock.patch.object(alf.summary, "histogram"), \
             mock.patch(
                 "alf.algorithms.bafc_algorithm_v3_tr2.safe_mean_hist_summary"
             ) as summary_mock, \
             mock.patch(
                 "alf.algorithms.bafc_algorithm_v3_tr2.nest_utils.grad",
                 wraps=nest_utils.grad) as grad_mock, \
             mock.patch.object(
                 alg._critic_networks,
                 "forward",
                 wraps=alg._critic_networks.forward) as critic_forward_mock:
            _, actor_info = alg._actor_train_step(
                observation, action, torch.zeros(2, 2), torch.ones(2, 3), ())

        self.assertEqual(grad_mock.call_count, 1)
        self.assertEqual(critic_forward_mock.call_count, 1)
        summary_names = {call.args[0] for call in summary_mock.call_args_list}
        self.assertIn('actor_gradients/dqda', summary_names)
        self.assertIn('actor_gradients/clipped_dqda', summary_names)
        self.assertIn('actor_critic_aggregation/q_mean', summary_names)
        self.assertIn('actor_critic_aggregation/dqda_pairwise_cosine',
                      summary_names)
        scalar_names = {
            call.args[0] if call.args else call.kwargs['name']
            for call in scalar_mock.call_args_list
        }
        self.assertIn('actor_gradients/dqda_clip_fraction', scalar_names)
        self.assertTensorClose(actor_info.extra.grad_trust_metric,
                               torch.ones(2))

        total_loss = (actor_info.loss.mean() +
                      actor_info.extra.eval_action_loss.mean())
        total_loss.backward()

    def test_agreement_summaries_follow_debug_and_k(self):
        observation = torch.randn(2, 4)
        with mock.patch.object(
                alf.summary, "should_record_summaries", return_value=True), \
             mock.patch.object(alf.summary, "scalar"), \
             mock.patch.object(alf.summary, "histogram"), \
             mock.patch(
                 "alf.algorithms.bafc_algorithm_v3_tr2.safe_mean_hist_summary"
             ) as summary_mock:
            no_debug_alg = self._make_alg(
                actor_critic_pairing=False,
                num_sampled_critics_for_actor=2,
                actor_eval_type='output')
            no_debug_action = no_debug_alg._actor_networks(observation)[0]
            no_debug_alg._actor_train_step(
                observation, no_debug_action, torch.zeros(2, 2),
                torch.ones(2, 3), ())
            summary_mock.assert_not_called()

            k1_alg = self._make_alg(
                actor_critic_pairing=False,
                actor_eval_type='output',
                debug_summaries=True)
            k1_action = k1_alg._actor_networks(observation)[0]
            k1_alg._actor_train_step(
                observation, k1_action, torch.zeros(2, 2), torch.ones(2, 3), ())
            summary_names = {
                call.args[0] for call in summary_mock.call_args_list
            }
            self.assertIn('actor_gradients/dqda', summary_names)
            self.assertFalse(
                any(name.startswith('actor_critic_aggregation/')
                    for name in summary_names))

    def test_linear_memory_pairwise_cosine(self):
        individual_dqda = torch.randn(4, 2, 3, 5)
        normalized = individual_dqda / individual_dqda.norm(
            dim=-1, keepdim=True).clamp_min(1e-12)
        expected = torch.zeros(2, 3)
        for i in range(4):
            for j in range(4):
                if i != j:
                    expected += (normalized[i] * normalized[j]).sum(dim=-1)
        expected /= 4 * 3
        actual = BafcAlgorithmV3TR2._mean_pairwise_cosine(individual_dqda)
        self.assertTensorClose(actual, expected)

    def test_aggregated_train_modes_keep_tr2_info_structures_valid(self):
        alg = self._make_alg(
            actor_critic_pairing=False,
            num_sampled_critics_for_actor=2,
            use_random_critic_targets=True,
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3)
        num_samples = 4
        n = alg._num_actor_critic
        inputs = self._make_train_time_step(torch.randn(num_samples, 4))
        current_action = torch.zeros(num_samples, n, 2)
        rollout_info = BafcInfo(
            action=torch.zeros(num_samples, 2),
            bootstrap_mask=torch.ones(num_samples, n),
            discounted_return=torch.zeros(num_samples))
        actor_info = LossInfo(
            loss=torch.ones(num_samples),
            extra=BafcActorInfo(
                eval_action_loss=torch.ones(num_samples),
                grad_trust_metric=torch.ones(num_samples)))
        critic_info = BafcCriticInfo(
            critic=torch.ones(num_samples, n, n),
            target_critic=torch.ones(num_samples, n),
            eval_trust_metric=torch.ones(num_samples),
            critic_sample_weight=torch.ones(num_samples))

        def _run_train_step(mode, initial=False):
            alg._train_mode = mode
            alg._actor_update_counter = 0 if initial else 1
            alg._critic_update_counter = 0 if initial else 1
            with mock.patch.object(
                    alg,
                    "_predict_action",
                    return_value=(current_action, ())), mock.patch.object(
                        alg,
                        "_actor_train_step",
                        return_value=((), actor_info)), mock.patch.object(
                            alg,
                            "_critic_train_step",
                            return_value=((), critic_info)):
                return alg.train_step(inputs, BafcState(), rollout_info).info

        combined_info = _run_train_step(TrainMode.critic, initial=True)
        actor_only_info = _run_train_step(TrainMode.actor)
        critic_only_info = _run_train_step(TrainMode.critic)

        for info in (combined_info, actor_only_info, critic_only_info):
            self.assertIsInstance(info, BafcInfo)
            alf.nest.map_structure(
                lambda value: value.reshape(num_samples, *value.shape[1:])
                if isinstance(value, torch.Tensor) else value, info)
        self.assertTensorClose(combined_info.actor.extra.grad_trust_metric,
                               torch.ones(num_samples))
        self.assertEqual(combined_info.critic.target_critic.shape,
                         (num_samples, n))
        self.assertEqual(actor_only_info.critic.critic, ())
        self.assertEqual(actor_only_info.critic.target_critic, ())
        self.assertEqual(critic_only_info.actor.loss, ())
        self.assertEqual(critic_only_info.actor.extra.eval_action_loss, ())
        self.assertTensorClose(critic_only_info.critic.critic_sample_weight,
                               torch.ones(num_samples))
        self.assertIsInstance(actor_only_info.critic, BafcCriticInfo)
        self.assertIsInstance(critic_only_info.actor.extra, BafcActorInfo)

    def test_critic_reweighting_disabled_returns_empty_weight(self):
        alg = self._make_alg()
        weight = alg._compute_critic_sample_weights(
            torch.randn(4, 4), torch.randn(4, 2))

        self.assertEqual(weight, ())

    def test_critic_reweighting_weights_are_normalized(self):
        alg = self._make_alg(
            enable_critic_reweighting=True,
            critic_reweighting_solver_iters=2,
            trust_metric_num_feature_coords=3)
        obs = torch.randn(2, 3, 4)
        action = torch.randn(2, 3, 2)
        phi_target = torch.randn(5, alg._num_actor_critic, 4)
        phi_behavior = torch.randn(6, alg._num_actor_critic, 4)

        with mock.patch.object(
                alg,
                "_compute_reference_metric_feature_maps",
                return_value=(phi_target, phi_behavior)) as feature_mock:
            weight = alg._compute_critic_sample_weights(obs, action)

        feature_mock.assert_called_once()
        self.assertEqual(tuple(weight.shape), (2, 3))
        self.assertTrue(torch.isfinite(weight).all().item())
        self.assertTrue((weight >= 0).all().item())
        self.assertAlmostEqual(weight.mean().item(), 1.0, places=5)

    def test_critic_reweighting_degenerate_features_fall_back_to_uniform(self):
        alg = self._make_alg(enable_critic_reweighting=True)
        obs = torch.randn(4, 4)
        action = torch.randn(4, 2)
        phi_target = torch.full((4, alg._num_actor_critic, 3), float('nan'))
        phi_behavior = torch.full((4, alg._num_actor_critic, 3), float('nan'))

        with mock.patch.object(
                alg,
                "_compute_reference_metric_feature_maps",
                return_value=(phi_target, phi_behavior)):
            weight = alg._compute_critic_sample_weights(obs, action)

        self.assertTensorClose(weight, torch.ones(4))

    def test_critic_reweighting_solver_returns_simplex_distribution(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, critic_reweighting_solver_iters=2)
        projected = alg._project_simplex(torch.tensor([-1.0, 2.0, 0.5]))
        self.assertTrue((projected >= 0).all().item())
        self.assertAlmostEqual(projected.sum().item(), 1.0, places=6)

        features = torch.randn(5, alg._num_actor_critic, 3)
        target = torch.randn(4, alg._num_actor_critic, 3)
        target_cov = alg._feature_covariance(target)
        p = alg._solve_critic_reweighting_distribution(
            features, target_cov, torch.tensor(0.5), torch.tensor(1e-3))

        self.assertEqual(tuple(p.shape), (5, ))
        self.assertTrue(torch.isfinite(p).all().item())
        self.assertTrue((p >= 0).all().item())
        self.assertAlmostEqual(p.sum().item(), 1.0, places=5)

    def test_critic_loss_applies_sample_weight(self):
        alg = self._make_alg()
        t, b, n = 2, 3, alg._num_actor_critic
        sample_weight = torch.tensor([[0.5, 1.0, 1.5], [2.0, 0.25, 0.75]])

        class _UnitLoss:

            def __call__(self, info, value, target_value):
                del info, value, target_value
                return LossInfo(loss=torch.ones(t, b))

        alg._critic_losses = [_UnitLoss() for _ in range(n)]
        info = BafcInfo(
            critic=BafcCriticInfo(
                critic=torch.zeros(t, b, n, n),
                target_critic=torch.zeros(t, b, n, n),
                critic_sample_weight=sample_weight),
            bootstrap_mask=torch.ones(t, b, n))

        loss = alg._calc_critic_loss(info)

        self.assertTensorClose(loss.loss, sample_weight * float(n))

    def test_eval_and_reweighting_share_reference_feature_helper(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, critic_reweighting_solver_iters=1)
        obs = torch.randn(4, 4)
        action = torch.randn(4, 2)
        phi_target = torch.randn(4, alg._num_actor_critic, 3)
        phi_behavior = torch.randn(4, alg._num_actor_critic, 3)

        with mock.patch.object(
                alg,
                "_compute_reference_metric_feature_maps",
                return_value=(phi_target, phi_behavior)) as feature_mock:
            alg._compute_eval_trust_metric(obs, action)
        feature_mock.assert_called_once()

        with mock.patch.object(
                alg,
                "_compute_reference_metric_feature_maps",
                return_value=(phi_target, phi_behavior)) as feature_mock:
            alg._compute_critic_sample_weights(obs, action)
        feature_mock.assert_called_once()

    def test_eval_trust_uses_reference_encoding_cache_and_replay_action(self):
        alg = self._make_alg(trust_metric_num_obs=4)
        replay_obs = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        replay_action = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        target_obs = torch.full((2, 4), 9.0)
        alg._target_metric_observation_cache = target_obs.clone()

        target_action = torch.full((2, alg._num_actor_critic, 2), 3.0)
        reference_encoding = torch.full((alg._num_actor_critic, 5), 4.0)
        phi_target = torch.full((2, alg._num_actor_critic, 3), 5.0)
        phi_behavior = torch.full((4, alg._num_actor_critic, 3), 6.0)
        calls = []

        def _feature_map(observation, encoding, action, critic_network=None):
            del critic_network
            calls.append((observation.clone(), encoding.clone(), action.clone()))
            if len(calls) == 1:
                self.assertTensorClose(observation, target_obs)
                self.assertTensorClose(encoding, reference_encoding)
                self.assertTensorClose(action, target_action)
                return phi_target
            self.assertTensorClose(observation, replay_obs)
            self.assertTensorClose(encoding, reference_encoding)
            self.assertTensorClose(action, replay_action.unsqueeze(1))
            return phi_behavior

        with mock.patch.object(
                alg._reference_actor_networks,
                "forward",
                return_value=(target_action, ())) as actor_mock, mock.patch.object(
                    alg,
                    "_compute_actor_encoding",
                    return_value=reference_encoding) as encoding_mock, mock.patch.object(
                        alg,
                        "_compute_snapshot_feature_map",
                        side_effect=_feature_map), mock.patch.object(
                            alg,
                            "_compute_eval_trust_from_features",
                            return_value=torch.tensor(7.0)) as trust_mock:
            trust = alg._compute_eval_trust_metric(replay_obs, replay_action)

        self.assertTensorClose(trust, torch.tensor(7.0))
        actor_mock.assert_called_once()
        encoding_mock.assert_called_once_with(alg._reference_actor_networks)
        trust_mock.assert_called_once()
        trust_args, _ = trust_mock.call_args
        self.assertTensorClose(trust_args[0], phi_target)
        self.assertTensorClose(trust_args[1], phi_behavior)
        self.assertEqual(len(calls), 2)

    def test_eval_trust_falls_back_to_replay_obs_without_target_cache(self):
        alg = self._make_alg(trust_metric_num_obs=4)
        replay_obs = torch.randn(3, 4)
        replay_action = torch.randn(3, 2)
        target_action = torch.randn(3, alg._num_actor_critic, 2)
        actor_encoding = torch.randn(alg._num_actor_critic, 5)
        seen_target_obs = []

        def _feature_map(observation, encoding, action, critic_network=None):
            del encoding, action, critic_network
            if not seen_target_obs:
                seen_target_obs.append(observation.clone())
            return torch.ones(observation.shape[0], alg._num_actor_critic, 3)

        with mock.patch.object(
                alg._reference_actor_networks,
                "forward",
                return_value=(target_action, ())), mock.patch.object(
                    alg,
                    "_compute_actor_encoding",
                    return_value=actor_encoding), mock.patch.object(
                        alg,
                        "_compute_snapshot_feature_map",
                        side_effect=_feature_map):
            trust = alg._compute_eval_trust_metric(replay_obs, replay_action)

        self.assertTrue(torch.isfinite(trust).item())
        self.assertTensorClose(seen_target_obs[0], replay_obs)

    def test_after_update_passes_replay_action_to_eval_metric(self):
        alg = self._make_alg(enable_eval_rollout_skip_gate=True)
        inputs = self._make_train_time_step()
        info = BafcInfo(action=torch.randn(inputs.observation.shape[0], 2))
        alg._last_update_had_actor_step = True

        with mock.patch.object(
                alg,
                "_compute_eval_trust_metric",
                return_value=torch.tensor(2.5)) as metric_mock:
            alg.after_update(inputs, info)

        metric_mock.assert_called_once()
        args, _ = metric_mock.call_args
        self.assertIs(args[0], inputs.observation)
        self.assertIs(args[1], info.action)
        self.assertTensorClose(alg._last_eval_trust, torch.tensor(2.5))

    def test_after_update_pre_syncs_reference_before_eval_metric_when_grad_gate_disabled(self):
        alg = self._make_alg(
            enable_eval_rollout_skip_gate=True,
            enable_grad_actor_extend_gate=False)
        inputs = self._make_train_time_step()
        info = BafcInfo(action=torch.randn(inputs.observation.shape[0], 2))
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._last_update_had_actor_step = True

        def _metric(observation, behavior_action):
            self._assert_state_dict_equal(
                alg._reference_actor_networks,
                self._clone_state_dict(alg._actor_networks))
            return torch.tensor(3.5)

        with mock.patch.object(
                alg,
                "_compute_eval_trust_metric",
                side_effect=_metric) as metric_mock:
            alg.after_update(inputs, info)

        metric_mock.assert_called_once()
        args, _ = metric_mock.call_args
        self.assertIs(args[0], inputs.observation)
        self.assertIs(args[1], info.action)
        self.assertTensorClose(alg._last_eval_trust, torch.tensor(3.5))

    def test_eval_metric_uses_post_transition_reference_after_grad_violation(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_eval_rollout_skip_gate=True,
            enable_grad_actor_extend_gate=True,
            delta_trust_max=1.0)
        inputs = self._make_train_time_step()
        info = BafcInfo(action=torch.randn(inputs.observation.shape[0], 2))
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True

        def _metric(observation, behavior_action):
            self._assert_state_dict_equal(
                alg._reference_actor_networks,
                self._clone_state_dict(alg._actor_networks))
            return torch.tensor(4.5)

        with mock.patch.object(
                alg,
                "_compute_grad_generalization_trust_metric",
                return_value=torch.tensor(2.0)) as grad_mock, mock.patch.object(
                    alg,
                    "_compute_eval_trust_metric",
                    side_effect=_metric) as eval_mock:
            alg.after_update(inputs, info)

        grad_mock.assert_called_once_with(inputs.observation)
        eval_mock.assert_called_once()
        args, _ = eval_mock.call_args
        self.assertIs(args[0], inputs.observation)
        self.assertIs(args[1], info.action)
        self.assertEqual(alg._train_mode, TrainMode.critic)
        self.assertTensorClose(alg._last_grad_trust, torch.tensor(2.0))
        self.assertTensorClose(alg._last_eval_trust, torch.tensor(4.5))

    def test_rollout_updates_target_observation_cache(self):
        alg = self._make_alg(trust_metric_target_obs_cache_size=3)
        state = alg.get_initial_rollout_state(batch_size=1)
        inputs = TimeStep(
            step_type=torch.full((1, ), StepType.FIRST, dtype=torch.int64),
            reward=torch.zeros(1),
            discount=torch.ones(1),
            observation=torch.arange(4, dtype=torch.float32).reshape(1, 4),
            prev_action=(),
            env_id=())

        alg.rollout_step(inputs, state)
        self.assertTensorClose(alg._target_metric_observation_cache,
                               inputs.observation)

        for start in (4, 8, 12):
            inputs = inputs._replace(
                observation=torch.arange(
                    start, start + 4, dtype=torch.float32).reshape(1, 4))
            alg.rollout_step(inputs, state)
        self.assertEqual(alg._target_metric_observation_cache.shape[0], 3)
        expected = torch.stack([
            torch.arange(4, 8, dtype=torch.float32),
            torch.arange(8, 12, dtype=torch.float32),
            torch.arange(12, 16, dtype=torch.float32),
        ])
        self.assertTensorClose(alg._target_metric_observation_cache, expected)

    def test_reference_syncs_after_actor_update_when_grad_gate_disabled(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=False,
            monitor_trust_metrics=False)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_reference_syncs_without_actor_update_when_grad_gate_disabled(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_eval_rollout_skip_gate=True,
            enable_grad_actor_extend_gate=False,
            monitor_trust_metrics=True)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.critic
        alg._critic_update_counter = 1
        alg._last_update_had_actor_step = False

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_reference_syncs_when_grad_gated_actor_epoch_starts(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=True,
            monitor_trust_metrics=False)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.critic
        alg._critic_update_counter = 2
        alg._last_update_had_actor_step = False

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self.assertEqual(alg._train_mode, TrainMode.actor)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_reference_stays_fixed_during_safe_grad_gate_extension(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=True,
            monitor_trust_metrics=False)
        self._fill_module(alg._reference_actor_networks, 0.0)
        ref_before = self._clone_state_dict(alg._reference_actor_networks)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True
        alg._last_grad_trust = torch.tensor(0.0)

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self.assertEqual(alg._train_mode, TrainMode.actor)
        self._assert_state_dict_equal(alg._reference_actor_networks, ref_before)

    def test_reference_syncs_when_grad_trust_violates_threshold(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=True,
            monitor_trust_metrics=False,
            delta_trust_max=1.0)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True
        alg._last_grad_trust = torch.tensor(2.0)

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self.assertEqual(alg._train_mode, TrainMode.critic)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_reference_syncs_when_grad_extension_cap_breaks_epoch(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=True,
            grad_gate_max_consecutive_actor_extensions=1,
            monitor_trust_metrics=False,
            delta_trust_max=10.0)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True
        alg._last_grad_trust = torch.tensor(0.0)
        alg._grad_gate_consecutive_actor_extensions = 1

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self.assertEqual(alg._train_mode, TrainMode.critic)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_eval_trust_max_decay_disabled_keeps_fixed_threshold(self):
        alg = self._make_alg(eval_trust_max=10.0, num_env_steps=100)
        self._set_env_steps(alg, 50)

        self.assertEqual(alg._current_eval_trust_max(), 10.0)

    def test_eval_trust_max_decay_uses_env_step_progress(self):
        alg = self._make_alg(
            eval_trust_max=10.0,
            enable_eval_trust_max_decay=True,
            num_env_steps=100)

        self._set_env_steps(alg, 0)
        self.assertAlmostEqual(alg._current_eval_trust_max(), 10.0)
        self._set_env_steps(alg, 50)
        self.assertAlmostEqual(alg._current_eval_trust_max(), 5.0)
        self._set_env_steps(alg, 100)
        self.assertAlmostEqual(alg._current_eval_trust_max(), 0.0)

    def test_eval_gate_uses_decayed_eval_trust_max(self):
        alg = self._make_alg(
            eval_trust_max=10.0,
            enable_eval_rollout_skip_gate=True,
            enable_eval_trust_max_decay=True,
            num_env_steps=100)
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._completed_cycles_since_rollout = alg._rollout_cycles_per_collect
        self._set_env_steps(alg, 50)

        alg._last_eval_trust = torch.tensor(6.0)
        self.assertFalse(alg._should_skip_unroll_iter_off_policy())

        alg._last_eval_trust = torch.tensor(4.0)
        self.assertTrue(alg._should_skip_unroll_iter_off_policy())

    def _without_runtime_state(self, state_dict):
        state_dict = state_dict.copy()
        for key in list(state_dict.keys()):
            if "_bafc_runtime." in key:
                del state_dict[key]
        return state_dict

    def test_runtime_checkpoint_round_trip_and_legacy_fallback(self):
        alg = self._make_alg()
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._rollout_actor_id = torch.tensor(2)
        alg._actor_update_counter = 5
        alg._critic_update_counter = 7
        alg._completed_cycles_since_rollout = 3
        alg._real_rollouts_since_reference_sync = 4
        alg._last_eval_trust = torch.tensor(0.25)
        alg._last_grad_trust = torch.tensor(0.5)
        alg._trust_metric_update_counter = 6
        alg._eval_gate_consecutive_rollout_skips = 2
        alg._rollout_skip_due_eval_gate_count = 8
        alg._rollout_opportunity_count = 9
        alg._grad_gate_actor_extension_count = 10
        alg._grad_gate_consecutive_actor_extensions = 11
        alg._target_metric_observation_cache = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        alg._apply_train_mode_grad_flags()

        state = alg.state_dict()
        self.assertIn("_bafc_runtime.training_started", state)
        self.assertIn("_bafc_runtime.target_metric_observation_cache", state)

        restored = self._make_alg()
        restored.load_state_dict(state)

        self.assertTrue(restored._training_started)
        self.assertEqual(restored._train_mode, TrainMode.critic)
        self.assertEqual(restored._rollout_actor_id, 2)
        self.assertEqual(restored._actor_update_counter, 5)
        self.assertEqual(restored._critic_update_counter, 7)
        self.assertEqual(restored._completed_cycles_since_rollout, 3)
        self.assertEqual(restored._real_rollouts_since_reference_sync, 4)
        self.assertTensorClose(restored._last_eval_trust, torch.tensor(0.25))
        self.assertTensorClose(restored._last_grad_trust, torch.tensor(0.5))
        self.assertEqual(restored._trust_metric_update_counter, 6)
        self.assertEqual(restored._eval_gate_consecutive_rollout_skips, 2)
        self.assertEqual(restored._rollout_skip_due_eval_gate_count, 8)
        self.assertEqual(restored._rollout_opportunity_count, 9)
        self.assertEqual(restored._grad_gate_actor_extension_count, 10)
        self.assertEqual(restored._grad_gate_consecutive_actor_extensions, 11)
        self.assertTensorClose(restored._target_metric_observation_cache,
                               torch.arange(12, dtype=torch.float32).reshape(3, 4))
        self.assertTrue(all(not p.requires_grad
                            for p in restored._actor_networks.parameters()))
        self.assertTrue(restored._actor_eval_samples.requires_grad)

        legacy_restored = self._make_alg()
        legacy_restored.load_state_dict(self._without_runtime_state(state))
        self.assertTrue(legacy_restored._training_started)



if __name__ == "__main__":
    alf.test.main()
