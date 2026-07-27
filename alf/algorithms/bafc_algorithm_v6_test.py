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
from alf.algorithms.agent import Agent
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3
from alf.algorithms.bafc_algorithm_v6 import (BafcActorInfo,
                                              BafcAlgorithmV6,
                                              BafcCriticInfo,
                                              BafcCriticReweightingInfo,
                                              BafcInfo, BafcState)
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import LossInfo, StepType, TimeStep
from alf.experience_replayers.replay_buffer import BatchInfo, ReplayBuffer
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.nest import utils as nest_utils
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.checkpoint_utils import Checkpointer


class _DummyProcess:

    def memory_info(self):
        return SimpleNamespace(rss=0)

    def children(self, recursive=True):
        del recursive
        return []


class _LinearParallelCritic(torch.nn.Module):

    def __init__(self, slopes):
        super().__init__()
        self.register_buffer("_slopes", torch.as_tensor(slopes))

    def forward(self, inputs, state=()):
        actor_encoding, (_, action) = inputs
        q_value = (action * self._slopes.unsqueeze(0)).sum(dim=-1)
        # Preserve the functional-policy input in the graph so dQ/de is defined.
        q_value = q_value + 0. * actor_encoding.sum(dim=-1)
        return q_value, state


class _DummyReplayBuffer:

    def __init__(self, current_pos):
        self.device = torch.device("cpu")
        self._current_pos = torch.as_tensor(current_pos, dtype=torch.int64)


class BafcAlgorithmV6Test(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._psutil_patcher = mock.patch(
            "alf.algorithms.algorithm.psutil.Process",
            return_value=_DummyProcess())
        self._psutil_patcher.start()
        self.addCleanup(self._psutil_patcher.stop)

    def _make_alg(self, **kwargs):
        num_actor_critic = kwargs.pop("num_actor_critic", 3)
        reward_spec = kwargs.pop("reward_spec", TensorSpec(()))
        reward_weights = kwargs.pop("reward_weights", None)
        num_updates_per_train_iter = kwargs.pop("num_updates_per_train_iter",
                                                3)
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
            root_dir=tempfile.mkdtemp(prefix="bafc_v6_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=num_updates_per_train_iter)
        return BafcAlgorithmV6(
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

    def _make_v3_alg(self, **kwargs):
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_for_v6_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=3)
        return BafcAlgorithmV3(
            observation_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec((2, ), minimum=-1.0, maximum=1.0),
            config=config,
            actor_network_cls=partial(ActorFCNetwork, fc_layer_params=(32, 32)),
            critic_network_cls=partial(
                FuncCriticNetwork,
                obs_action_joint_fc_layer_params=(32, 32),
                actor_obs_action_joint_fc_layer_params=(32, 32)),
            actor_encoder_cls=partial(
                TransformerEncoder, num_layers=2, num_attention_heads=1),
            num_actor_critic=3,
            num_actor_eval_samples=16,
            **kwargs)

    def _make_agent(self, rl_algorithm_cls, **kwargs):
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_agent_v6_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=3)
        return Agent(
            observation_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec((2, ), minimum=-1.0, maximum=1.0),
            config=config,
            rl_algorithm_cls=partial(
                rl_algorithm_cls,
                actor_network_cls=partial(
                    ActorFCNetwork, fc_layer_params=(32, 32)),
                critic_network_cls=partial(
                    FuncCriticNetwork,
                    obs_action_joint_fc_layer_params=(32, 32),
                    actor_obs_action_joint_fc_layer_params=(32, 32)),
                actor_encoder_cls=partial(
                    TransformerEncoder, num_layers=2, num_attention_heads=1),
                num_actor_critic=3,
                num_actor_eval_samples=16,
                **kwargs))

    def _assert_module_state_equal(self, left, right):
        left_state = left.state_dict()
        right_state = right.state_dict()
        self.assertEqual(set(left_state.keys()), set(right_state.keys()))
        for key, left_value in left_state.items():
            self.assertTrue(
                torch.equal(left_value.cpu(), right_state[key].cpu()),
                msg=key)

    def _attach_replay_buffer(self, alg, num_items=0):
        replay_buffer = ReplayBuffer(
            data_spec=TimeStep(
                step_type=TensorSpec((), dtype=torch.int64),
                reward=TensorSpec(()),
                discount=TensorSpec(()),
                observation=TensorSpec((4, )),
                prev_action=TensorSpec((2, )),
                env_id=TensorSpec((), dtype=torch.int64)),
            num_environments=1,
            max_length=8)
        for value in range(num_items):
            replay_buffer.add_batch(
                TimeStep(
                    step_type=torch.tensor([StepType.MID], dtype=torch.int64),
                    reward=torch.tensor([float(value)], dtype=torch.float32),
                    discount=torch.ones(1),
                    observation=torch.full((1, 4), float(value)),
                    prev_action=torch.zeros(1, 2),
                    env_id=torch.tensor([0], dtype=torch.int64)),
                env_ids=torch.tensor([0]))
        alg._replay_buffer = replay_buffer
        return replay_buffer

    def _make_rollout_time_step(self, observation):
        batch_size = observation.shape[0]
        return TimeStep(
            step_type=torch.full((batch_size, ), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(batch_size),
            discount=torch.ones(batch_size),
            observation=observation,
            prev_action=(),
            env_id=())

    def test_initialization_smoke_and_config_name(self):
        alg = self._make_alg()

        self.assertIsInstance(alg, BafcAlgorithmV6)
        self.assertEqual(BafcAlgorithmV6.__name__, "BafcAlgorithmV6")
        self.assertEqual(alg._num_actor_critic, 3)
        self.assertFalse(alg._enable_critic_reweighting)
        self.assertEqual(alg._critic_reweighting_solver, "lbfgs_logits")
        self.assertEqual(alg._critic_reweighting_solver_iters, 5)
        self.assertEqual(alg._critic_reweighting_target_obs_cache_size, 512)

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
                torch.randn(2, 4), action, torch.zeros(2, 2), torch.ones(2, 3),
                ())

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
                 "alf.algorithms.bafc_algorithm_v6.safe_mean_hist_summary"
             ) as summary_mock, \
             mock.patch(
                 "alf.algorithms.bafc_algorithm_v6.nest_utils.grad",
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

        total_loss = actor_info.loss.mean() + actor_info.extra.eval_action_loss.mean()
        total_loss.backward()

    def test_agreement_summaries_follow_debug_and_k(self):
        observation = torch.randn(2, 4)
        with mock.patch.object(
                alf.summary, "should_record_summaries", return_value=True), \
             mock.patch.object(alf.summary, "scalar"), \
             mock.patch.object(alf.summary, "histogram"), \
             mock.patch(
                 "alf.algorithms.bafc_algorithm_v6.safe_mean_hist_summary"
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
        actual = BafcAlgorithmV6._mean_pairwise_cosine(individual_dqda)
        self.assertTensorClose(actual, expected)



    def test_invalid_critic_reweighting_solver_raises(self):
        with self.assertRaises(AssertionError):
            self._make_alg(critic_reweighting_solver="bogus")

    def test_info_tuples_do_not_include_trust_fields(self):
        for fields in (BafcInfo._fields, BafcActorInfo._fields,
                       BafcCriticInfo._fields):
            self.assertNotIn("eval_trust_metric", fields)
            self.assertNotIn("grad_trust_metric", fields)

        self.assertIn("critic_sample_weight", BafcCriticInfo._fields)
        self.assertIn("critic_reweighting_info", BafcCriticInfo._fields)
        self.assertIn("sample_age", BafcInfo._fields)

    def test_preprocess_experience_computes_sample_age(self):
        alg = self._make_alg()
        root_inputs = TimeStep(
            step_type=torch.full((2, 3), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(2, 3),
            discount=torch.ones(2, 3),
            observation=torch.zeros(2, 3, 4),
            prev_action=(),
            env_id=())
        batch_info = BatchInfo(
            env_ids=torch.tensor([0, 1], dtype=torch.int64),
            positions=torch.tensor([7, 15], dtype=torch.int64),
            replay_buffer=_DummyReplayBuffer([10, 20]))

        processed_inputs, processed_info = alg.preprocess_experience(
            root_inputs, BafcInfo(), batch_info)

        self.assertIs(processed_inputs, root_inputs)
        self.assertTensorClose(
            processed_info.sample_age,
            torch.tensor([[2, 1, 0], [4, 3, 2]], dtype=torch.float32))

    def test_preprocess_experience_missing_batch_info_leaves_info_unchanged(self):
        alg = self._make_alg()
        root_inputs = TimeStep(
            step_type=torch.full((2, 3), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(2, 3),
            discount=torch.ones(2, 3),
            observation=torch.zeros(2, 3, 4),
            prev_action=(),
            env_id=())
        rollout_info = BafcInfo()

        _, processed_info = alg.preprocess_experience(root_inputs, rollout_info, ())

        self.assertEqual(processed_info.sample_age, ())

    def test_critic_reweighting_disabled_returns_empty_weight(self):
        alg = self._make_alg()
        with mock.patch.object(
                alg, "_record_critic_reweighting_summaries") as summary_mock:
            weight, reweighting_info = alg._compute_critic_sample_weights(
                torch.randn(4, 4), torch.randn(4, 2))

        self.assertEqual(weight, ())
        self.assertEqual(reweighting_info, ())
        summary_mock.assert_not_called()


    def test_critic_reweighting_summary_metrics(self):
        alg = self._make_alg(
            enable_critic_reweighting=True,
            debug_summaries=True,
            critic_reweighting_max_weight=2.0)
        raw_weights = torch.tensor([0.5, 2.0, 3.0])
        clipped_weights = raw_weights.clamp(max=alg._critic_reweighting_max_weight)
        final_weights = clipped_weights / clipped_weights.mean()

        with mock.patch(
                "alf.summary.should_record_summaries",
                return_value=True), mock.patch(
                    "alf.summary.scalar") as scalar_mock, mock.patch(
                        "alf.summary.histogram") as histogram_mock:
            alg._record_critic_reweighting_summaries(
                final_weights,
                raw_weights=raw_weights,
                clipped_weights=clipped_weights,
                fallback_to_uniform=False,
                solver_objective_initial=torch.tensor(5.0),
                solver_objective_final=torch.tensor(3.0),
                solver_distribution_half_final_tv=torch.tensor(0.25),
                solver_objective_half=torch.tensor(4.0),
                solver_objective_half_to_final_improvement=torch.tensor(1.0),
                solver_distribution_uniform_tv=torch.tensor(0.4),
                solver_distribution_ess_ratio=torch.tensor(0.7),
                solver_objective_initial_to_half_improvement=torch.tensor(1.0),
                frac_clipped_at_max_half=torch.tensor(1.0 / 3.0),
                frac_clipped_at_max_half_to_final_delta=torch.tensor(1.0 / 3.0))

        histogram_names = [call.args[0] for call in histogram_mock.call_args_list]
        self.assertIn("critic_reweighting/raw_weight/value", histogram_names)
        self.assertIn("critic_reweighting/clipped_weight/value", histogram_names)
        self.assertIn("critic_reweighting/final_weight/value", histogram_names)
        self.assertNotIn("critic_reweighting/sample_age/value", histogram_names)

        scalars = {call.args[0]: call.args[1] for call in scalar_mock.call_args_list}
        self.assertNotIn("critic_reweighting/final_weight_recency_corr", scalars)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/num_clipped_at_max"]),
            2.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/frac_clipped_at_max"]),
            2.0 / 3.0,
            places=6)
        expected_ess = final_weights.sum().pow(2) / final_weights.pow(2).sum()
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/ess"]),
            float(expected_ess),
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/ess_ratio"]),
            float(expected_ess / final_weights.numel()),
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/fallback_to_uniform"]),
            0.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_objective_initial"]),
            5.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_objective_final"]),
            3.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_objective_improvement"]),
            2.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/solver_distribution_half_final_tv"]),
            0.25,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_objective_half"]),
            4.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/solver_objective_half_to_final_improvement"]),
            1.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_distribution_uniform_tv"]),
            0.4,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_distribution_ess_ratio"]),
            0.7,
            places=6)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/solver_objective_initial_to_half_improvement"]),
            1.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/frac_clipped_at_max_half"]),
            1.0 / 3.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/frac_clipped_at_max_half_to_final_delta"]),
            1.0 / 3.0,
            places=6)

    def test_critic_reweighting_recency_summary_metrics(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, debug_summaries=True)
        raw_weights = torch.tensor([4.0, 3.0, 2.0, 1.0])
        final_weights = raw_weights / raw_weights.mean()
        sample_age = torch.tensor([0.0, 1.0, 2.0, 3.0])

        with mock.patch(
                "alf.summary.should_record_summaries",
                return_value=True), mock.patch(
                    "alf.summary.scalar") as scalar_mock, mock.patch(
                        "alf.summary.histogram") as histogram_mock:
            alg._record_critic_reweighting_summaries(
                final_weights,
                raw_weights=raw_weights,
                clipped_weights=raw_weights,
                sample_age=sample_age,
                fallback_to_uniform=False)

        histogram_names = [call.args[0] for call in histogram_mock.call_args_list]
        self.assertIn("critic_reweighting/sample_age/value", histogram_names)

        scalars = {call.args[0]: call.args[1] for call in scalar_mock.call_args_list}
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/sample_age_min"]),
            0.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/sample_age_max"]),
            3.0,
            places=6)
        self.assertGreater(
            float(scalars["critic_reweighting/final_weight_recency_corr"]),
            0.999)
        self.assertLess(
            float(scalars["critic_reweighting/final_weight_age_corr"]),
            -0.999)
        self.assertGreater(
            float(scalars["critic_reweighting/raw_weight_recency_corr"]),
            0.999)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/newest_quartile_final_weight_mean"]),
            float(final_weights[0]),
            places=6)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/oldest_quartile_final_weight_mean"]),
            float(final_weights[-1]),
            places=6)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/newest_over_oldest_weight_ratio"]),
            float(final_weights[0] / final_weights[-1]),
            places=6)

    def test_critic_reweighting_fallback_records_uniform_summary(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, debug_summaries=True)
        obs = torch.randn(4, 4)
        action = torch.randn(3, 2)

        with mock.patch(
                "alf.summary.should_record_summaries",
                return_value=True), mock.patch.object(
                    alg,
                    "_record_critic_reweighting_summaries") as record_mock:
            weight, reweighting_info = alg._compute_critic_sample_weights(
                obs, action)

        self.assertTensorClose(weight, torch.ones(4))
        self.assertIsInstance(reweighting_info, BafcCriticReweightingInfo)
        record_mock.assert_not_called()

        with mock.patch(
                "alf.summary.should_record_summaries",
                return_value=True), mock.patch(
                    "alf.summary.scalar") as scalar_mock, mock.patch(
                        "alf.summary.histogram") as histogram_mock:
            alg._record_critic_reweighting_info_summaries(reweighting_info)

        histogram_values = {
            call.args[0]: call.args[1]
            for call in histogram_mock.call_args_list
        }
        self.assertTensorClose(
            histogram_values["critic_reweighting/final_weight/value"],
            torch.ones(4))
        scalars = {call.args[0]: call.args[1] for call in scalar_mock.call_args_list}
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/fallback_to_uniform"]),
            1.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/ess_ratio"]),
            1.0,
            places=6)
        for name in (
                "critic_reweighting/solver_distribution_half_final_tv",
                "critic_reweighting/solver_objective_half_to_final_improvement",
                "critic_reweighting/solver_distribution_uniform_tv",
                "critic_reweighting/solver_objective_initial_to_half_improvement",
                "critic_reweighting/frac_clipped_at_max_half_to_final_delta"):
            self.assertAlmostEqual(float(scalars[name]), 0.0, places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_distribution_ess_ratio"]),
            1.0,
            places=6)

    def test_critic_reweighting_weights_are_normalized(self):
        alg = self._make_alg(
            enable_critic_reweighting=True,
            critic_reweighting_beta=0.5,
            critic_reweighting_solver_iters=2,
            critic_reweighting_num_feature_coords=3)
        obs = torch.randn(2, 3, 4)
        action = torch.randn(2, 3, 2)
        phi_target = torch.randn(5, alg._num_actor_critic, 4)
        phi_behavior = torch.randn(6, alg._num_actor_critic, 4)

        with mock.patch.object(
                alg,
                "_compute_reweighting_feature_maps",
                return_value=(phi_target, phi_behavior)) as feature_mock:
            weight, reweighting_info = alg._compute_critic_sample_weights(
                obs, action)

        feature_mock.assert_called_once()
        self.assertEqual(tuple(weight.shape), (2, 3))
        self.assertTrue(torch.isfinite(weight).all().item())
        self.assertTrue((weight >= 0).all().item())
        self.assertAlmostEqual(weight.mean().item(), 1.0, places=5)
        self.assertIsInstance(reweighting_info, BafcCriticReweightingInfo)
        self.assertTensorClose(reweighting_info.final_weight, weight)
        self.assertTrue(torch.isfinite(reweighting_info.raw_weight).all().item())
        self.assertTrue(
            torch.isfinite(reweighting_info.clipped_weight).all().item())
        self.assertAlmostEqual(
            float(reweighting_info.fallback_to_uniform), 0.0, places=6)
        for value in (
                reweighting_info.solver_distribution_half_final_tv,
                reweighting_info.solver_objective_half,
                reweighting_info.solver_objective_half_to_final_improvement,
                reweighting_info.solver_distribution_uniform_tv,
                reweighting_info.solver_distribution_ess_ratio,
                reweighting_info.solver_objective_initial_to_half_improvement,
                reweighting_info.frac_clipped_at_max_half,
                reweighting_info.frac_clipped_at_max_half_to_final_delta):
            self.assertTrue(torch.isfinite(value).all().item())
        self.assertGreaterEqual(
            float(reweighting_info.solver_distribution_half_final_tv), 0.0)
        self.assertLessEqual(
            float(reweighting_info.solver_distribution_half_final_tv), 1.0)
        self.assertGreaterEqual(
            float(reweighting_info.solver_distribution_uniform_tv), 0.0)
        self.assertLessEqual(
            float(reweighting_info.solver_distribution_uniform_tv), 1.0)
        self.assertGreater(
            float(reweighting_info.solver_distribution_ess_ratio), 0.0)
        self.assertLessEqual(
            float(reweighting_info.solver_distribution_ess_ratio), 1.0)
        self.assertGreaterEqual(
            float(reweighting_info.frac_clipped_at_max_half), 0.0)
        self.assertLessEqual(
            float(reweighting_info.frac_clipped_at_max_half), 1.0)

    def test_critic_reweighting_degenerate_features_fall_back_to_uniform(self):
        alg = self._make_alg(enable_critic_reweighting=True)
        obs = torch.randn(4, 4)
        action = torch.randn(4, 2)
        phi_target = torch.full((4, alg._num_actor_critic, 3), float('nan'))
        phi_behavior = torch.full((4, alg._num_actor_critic, 3), float('nan'))

        with mock.patch.object(
                alg,
                "_compute_reweighting_feature_maps",
                return_value=(phi_target, phi_behavior)):
            weight, reweighting_info = alg._compute_critic_sample_weights(
                obs, action)

        self.assertTensorClose(weight, torch.ones(4))
        self.assertIsInstance(reweighting_info, BafcCriticReweightingInfo)
        self.assertAlmostEqual(
            float(reweighting_info.fallback_to_uniform), 1.0, places=6)
        self.assertAlmostEqual(
            float(reweighting_info.solver_distribution_half_final_tv),
            0.0,
            places=6)
        self.assertAlmostEqual(
            float(reweighting_info.solver_objective_half_to_final_improvement),
            0.0,
            places=6)
        self.assertAlmostEqual(
            float(reweighting_info.solver_distribution_uniform_tv),
            0.0,
            places=6)
        self.assertAlmostEqual(
            float(reweighting_info.solver_distribution_ess_ratio),
            1.0,
            places=6)
        self.assertAlmostEqual(
            float(reweighting_info.solver_objective_initial_to_half_improvement),
            0.0,
            places=6)
        self.assertAlmostEqual(
            float(reweighting_info.frac_clipped_at_max_half_to_final_delta),
            0.0,
            places=6)

    def test_critic_reweighting_solver_returns_simplex_distribution(self):
        torch.manual_seed(1234)
        projected_alg = self._make_alg()
        projected = projected_alg._project_simplex(
            torch.tensor([-1.0, 2.0, 0.5]))
        self.assertTrue((projected >= 0).all().item())
        self.assertAlmostEqual(projected.sum().item(), 1.0, places=6)

        features = torch.randn(5, projected_alg._num_actor_critic, 3)
        target = torch.randn(4, projected_alg._num_actor_critic, 3)
        target_cov = projected_alg._feature_covariance(target)
        beta = torch.tensor(0.5)
        ridge = torch.tensor(1e-3)

        for solver in ("lbfgs_logits", "projected_gradient_fw"):
            alg = self._make_alg(
                enable_critic_reweighting=True,
                critic_reweighting_solver=solver,
                critic_reweighting_solver_iters=2)
            p, p_half = alg._solve_critic_reweighting_distribution(
                features, target_cov, beta, ridge)

            for dist in (p, p_half):
                self.assertEqual(tuple(dist.shape), (5, ))
                self.assertTrue(torch.isfinite(dist).all().item())
                self.assertTrue((dist >= 0).all().item())
                self.assertAlmostEqual(dist.sum().item(), 1.0, places=5)

            tv = 0.5 * (p_half - p).abs().sum()
            self.assertGreaterEqual(float(tv), 0.0)
            self.assertLessEqual(float(tv), 1.0)

            uniform = torch.full_like(p, 1.0 / p.numel())
            uniform_obj = alg._critic_reweighting_objective(
                uniform, features, target_cov, beta, ridge)
            final_obj = alg._critic_reweighting_objective(
                p, features, target_cov, beta, ridge)
            self.assertLessEqual(
                float(final_obj), float(uniform_obj) + 1e-3)

    def test_random_target_critic_loss_applies_sample_weight(self):
        alg = self._make_alg(use_random_critic_targets=True)
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
                target_critic=torch.zeros(t, b, n),
                critic_sample_weight=sample_weight),
            bootstrap_mask=torch.ones(t, b, n))

        loss = alg._calc_critic_loss(info)

        self.assertTensorClose(loss.loss, sample_weight * float(n))

    def test_random_target_empty_critic_info_uses_shared_target_shape(self):
        alg = self._make_alg(
            reward_spec=TensorSpec((2, )),
            reward_weights=[1., -1.],
            use_random_critic_targets=True,
            enable_critic_reweighting=True)
        info = alg._empty_critic_info_for_train_info(torch.zeros(5, 2))

        self.assertEqual(info.critic.shape, (5, 3, 3, 2))
        self.assertEqual(info.target_critic.shape, (5, 3, 2))
        self.assertEqual(info.critic_sample_weight.shape, (5, ))

    def test_train_step_records_reweighting_summary_in_summary_phase(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, debug_summaries=True)
        length, batch_size = 2, 64
        num_samples = length * batch_size
        obs = torch.randn(num_samples, 4)
        inputs = self._make_rollout_time_step(obs)
        action = torch.zeros(num_samples, alg._num_actor_critic, 2)
        actor_info = LossInfo(
            loss=torch.zeros(num_samples),
            extra=BafcActorInfo(eval_action_loss=torch.zeros(num_samples)))
        reweighting_info = BafcCriticReweightingInfo(
            final_weight=torch.ones(num_samples),
            raw_weight=torch.ones(num_samples),
            clipped_weight=torch.ones(num_samples),
            sample_age=torch.zeros(num_samples),
            fallback_to_uniform=torch.tensor(0.0),
            solver_objective_initial=torch.tensor(1.0),
            solver_objective_final=torch.tensor(0.5))
        critic_info = BafcCriticInfo(
            critic_reweighting_info=reweighting_info)
        rollout_info = BafcInfo(
            action=action,
            bootstrap_mask=torch.ones(num_samples, alg._num_actor_critic),
            discounted_return=torch.zeros(num_samples),
            sample_age=torch.zeros(num_samples))

        with mock.patch.object(
                alg,
                "_predict_action",
                return_value=(action, ())), mock.patch.object(
                    alg,
                    "_actor_train_step",
                    return_value=((), actor_info)), mock.patch.object(
                        alg,
                        "_critic_train_step",
                        return_value=((), critic_info)), mock.patch.object(
                            alg,
                            "_record_critic_reweighting_summaries"
                        ) as record_mock, mock.patch(
                            "alf.summary.should_record_summaries",
                            return_value=True):
            alg_step = alg.train_step(inputs, BafcState(), rollout_info)

        record_mock.assert_called_once()
        self.assertIs(record_mock.call_args.args[0],
                      reweighting_info.final_weight)
        self.assertIs(record_mock.call_args.kwargs["fallback_to_uniform"],
                      reweighting_info.fallback_to_uniform)
        self.assertIs(record_mock.call_args.kwargs["solver_objective_initial"],
                      reweighting_info.solver_objective_initial)
        self.assertIs(record_mock.call_args.kwargs["solver_objective_final"],
                      reweighting_info.solver_objective_final)

        returned_info = alg_step.info.critic.critic_reweighting_info
        self.assertEqual(returned_info, ())

        alf.nest.map_structure(
            lambda x: x.reshape(length, batch_size, *x.shape[1:])
            if isinstance(x, torch.Tensor) else x,
            alg_step.info)

    def test_actor_only_summary_uses_cached_reweighting_info(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, debug_summaries=True)
        num_samples = 8
        obs = torch.randn(num_samples, 4)
        inputs = self._make_rollout_time_step(obs)
        action = torch.zeros(num_samples, alg._num_actor_critic, 2)
        actor_info = LossInfo(
            loss=torch.zeros(num_samples),
            extra=BafcActorInfo(eval_action_loss=torch.zeros(num_samples)))
        cached_info = BafcCriticReweightingInfo(
            final_weight=torch.ones(num_samples),
            raw_weight=torch.ones(num_samples),
            clipped_weight=torch.ones(num_samples),
            sample_age=torch.zeros(num_samples),
            fallback_to_uniform=torch.tensor(0.0),
            solver_objective_initial=torch.tensor(1.0),
            solver_objective_final=torch.tensor(0.5))
        rollout_info = BafcInfo(
            action=action,
            bootstrap_mask=torch.ones(num_samples, alg._num_actor_critic),
            discounted_return=torch.zeros(num_samples),
            sample_age=torch.zeros(num_samples))
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._critic_update_counter = 1
        alg._last_critic_reweighting_info = cached_info

        with mock.patch.object(
                alg,
                "_predict_action",
                return_value=(action, ())), mock.patch.object(
                    alg,
                    "_actor_train_step",
                    return_value=((), actor_info)), mock.patch.object(
                        alg,
                        "_record_critic_reweighting_summaries"
                    ) as record_mock, mock.patch(
                        "alf.summary.should_record_summaries",
                        return_value=True):
            alg_step = alg.train_step(inputs, BafcState(), rollout_info)

        record_mock.assert_called_once()
        self.assertIs(record_mock.call_args.args[0], cached_info.final_weight)
        self.assertEqual(alg_step.info.critic.critic_reweighting_info, ())

    def test_actor_only_and_critic_only_train_info_structures_match(self):
        alg = self._make_alg(enable_critic_reweighting=True)
        length, batch_size = 2, 64
        num_samples = length * batch_size
        n = alg._num_actor_critic
        obs = torch.randn(num_samples, 4)
        inputs = self._make_rollout_time_step(obs)
        action = torch.zeros(num_samples, n, 2)
        rollout_info = BafcInfo(
            action=action,
            bootstrap_mask=torch.ones(num_samples, n),
            discounted_return=torch.zeros(num_samples),
            sample_age=torch.zeros(num_samples))
        actor_info = LossInfo(
            loss=torch.ones(num_samples),
            extra=BafcActorInfo(eval_action_loss=torch.ones(num_samples)))
        reweighting_info = BafcCriticReweightingInfo(
            final_weight=torch.ones(num_samples),
            raw_weight=torch.ones(num_samples),
            clipped_weight=torch.ones(num_samples),
            sample_age=torch.zeros(num_samples),
            fallback_to_uniform=torch.tensor(0.0),
            solver_objective_initial=torch.tensor(1.0),
            solver_objective_final=torch.tensor(0.5))
        critic_info = BafcCriticInfo(
            critic=torch.ones(num_samples, n, n),
            target_critic=torch.ones(num_samples, n, n),
            critic_sample_weight=torch.ones(num_samples),
            critic_reweighting_info=reweighting_info)

        def _run_train_step(mode):
            alg._train_mode = mode
            alg._actor_update_counter = 1
            alg._critic_update_counter = 1
            with mock.patch.object(
                    alg,
                    "_predict_action",
                    return_value=(action, ())), mock.patch.object(
                        alg,
                        "_actor_train_step",
                        return_value=((), actor_info)), mock.patch.object(
                            alg,
                            "_critic_train_step",
                            return_value=((), critic_info)):
                return alg.train_step(inputs, BafcState(), rollout_info).info

        actor_only_info = _run_train_step(TrainMode.actor)
        critic_only_info = _run_train_step(TrainMode.critic)

        alf.nest.assert_same_structure(actor_only_info, critic_only_info)
        self.assertEqual(actor_only_info.critic.critic_reweighting_info, ())
        self.assertEqual(critic_only_info.critic.critic_reweighting_info, ())
        self.assertTensorClose(actor_only_info.critic.critic_sample_weight,
                               torch.ones(num_samples))
        self.assertTensorClose(critic_only_info.actor.loss,
                               torch.zeros(num_samples))

        for info in (actor_only_info, critic_only_info):
            alf.nest.map_structure(
                lambda x: x.reshape(length, batch_size, *x.shape[1:])
                if isinstance(x, torch.Tensor) else x,
                info)

    def test_rollout_updates_reweighting_target_observation_cache(self):
        alg = self._make_alg(
            enable_critic_reweighting=True,
            critic_reweighting_target_obs_cache_size=3)
        state = alg.get_initial_rollout_state(batch_size=1)
        inputs = self._make_rollout_time_step(
            torch.arange(4, dtype=torch.float32).reshape(1, 4))

        alg.rollout_step(inputs, state)
        self.assertTensorClose(alg._reweighting_target_observation_cache,
                               inputs.observation)

        for start in (4, 8, 12):
            inputs = inputs._replace(
                observation=torch.arange(
                    start, start + 4, dtype=torch.float32).reshape(1, 4))
            alg.rollout_step(inputs, state)

        self.assertEqual(alg._reweighting_target_observation_cache.shape[0], 3)
        expected = torch.stack([
            torch.arange(4, 8, dtype=torch.float32),
            torch.arange(8, 12, dtype=torch.float32),
            torch.arange(12, 16, dtype=torch.float32),
        ])
        self.assertTensorClose(alg._reweighting_target_observation_cache,
                               expected)

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
        alg._reweighting_target_observation_cache = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        alg._apply_train_mode_grad_flags()

        state = alg.state_dict()
        self.assertIn("_bafc_runtime.training_started", state)
        self.assertIn("_bafc_runtime.reweighting_target_observation_cache",
                      state)

        restored = self._make_alg()
        restored.load_state_dict(state)

        self.assertTrue(restored._training_started)
        self.assertEqual(restored._train_mode, TrainMode.critic)
        self.assertEqual(restored._rollout_actor_id, 2)
        self.assertEqual(restored._actor_update_counter, 5)
        self.assertEqual(restored._critic_update_counter, 7)
        self.assertTensorClose(restored._reweighting_target_observation_cache,
                               torch.arange(12, dtype=torch.float32).reshape(3, 4))
        self.assertTrue(all(not p.requires_grad
                            for p in restored._actor_networks.parameters()))
        self.assertTrue(restored._actor_eval_samples.requires_grad)

        legacy_restored = self._make_alg()
        legacy_restored.load_state_dict(self._without_runtime_state(state))
        self.assertTrue(legacy_restored._training_started)

    def test_load_v3_checkpoint_state_strictly(self):
        v3_alg = self._make_v3_alg(
            track_reweighting_target_observation_cache=True)
        v3_alg._training_started = True
        v3_alg._train_mode = TrainMode.actor
        v3_alg._rollout_actor_id = 1
        v3_alg._actor_update_counter = 2
        v3_alg._critic_update_counter = 3
        cache = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        v3_alg._reweighting_target_observation_cache = cache
        v3_state = v3_alg.state_dict()

        self.assertFalse(
            any(key.startswith("_reference_actor_networks.")
                for key in v3_state.keys()))
        self.assertFalse(
            any(key.startswith("_snapshot_critic_networks.")
                for key in v3_state.keys()))
        self.assertIn("_bafc_runtime.reweighting_target_observation_cache",
                      v3_state)

        v6_alg = self._make_alg()
        v6_alg.load_state_dict(v3_state, strict=True)

        self.assertTrue(v6_alg._training_started)
        self.assertEqual(v6_alg._train_mode, TrainMode.actor)
        self.assertEqual(v6_alg._rollout_actor_id, 1)
        self.assertEqual(v6_alg._actor_update_counter, 2)
        self.assertEqual(v6_alg._critic_update_counter, 3)
        self.assertTensorClose(v6_alg._reweighting_target_observation_cache,
                               cache)
        self._assert_module_state_equal(v6_alg._reference_actor_networks,
                                        v6_alg._actor_networks)
        self._assert_module_state_equal(v6_alg._snapshot_critic_networks,
                                        v6_alg._critic_networks)

    def test_checkpointer_loads_v3_agent_checkpoint_with_rank_replay_into_v6(self):
        cache = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        v3_agent = self._make_agent(
            BafcAlgorithmV3,
            checkpoint_replay_buffer=True,
            track_reweighting_target_observation_cache=True)
        v3_alg = v3_agent._rl_algorithm
        v3_alg._training_started = True
        v3_alg._train_mode = TrainMode.actor
        v3_alg._rollout_actor_id = 1
        v3_alg._actor_update_counter = 2
        v3_alg._critic_update_counter = 3
        v3_alg._reweighting_target_observation_cache = cache
        self._attach_replay_buffer(v3_agent, num_items=2)

        with tempfile.TemporaryDirectory() as ckpt_dir:
            v3_checkpointer = Checkpointer(ckpt_dir, algorithm=v3_agent)
            v3_checkpointer.save(123, ddp_rank=0)
            rank0_replay = torch.load(
                f"{ckpt_dir}/ckpt-123-replay_buffer-rank0")["algorithm"]
            self.assertTrue(
                any(key.startswith("_replay_buffer.")
                    for key in rank0_replay.keys()))

            self._attach_replay_buffer(v3_agent, num_items=5)
            v3_checkpointer.save(123, ddp_rank=1)

            v6_agent = self._make_agent(
                BafcAlgorithmV6, checkpoint_replay_buffer=True)
            self._attach_replay_buffer(v6_agent, num_items=0)
            self.assertFalse(
                any("_replay_buffer." in key
                    for key in v6_agent.state_dict().keys()))
            v6_checkpointer = Checkpointer(ckpt_dir, algorithm=v6_agent)
            self.assertEqual(v6_checkpointer.load(123, ddp_rank=1), 123)

        v6_alg = v6_agent._rl_algorithm
        self.assertTrue(v6_alg._training_started)
        self.assertEqual(v6_alg._train_mode, TrainMode.actor)
        self.assertEqual(v6_alg._rollout_actor_id, 1)
        self.assertEqual(v6_alg._actor_update_counter, 2)
        self.assertEqual(v6_alg._critic_update_counter, 3)
        self.assertTensorClose(v6_alg._reweighting_target_observation_cache,
                               cache)
        self.assertTensorEqual(v6_agent._replay_buffer._current_pos,
                               torch.tensor([5]))
        self.assertFalse(
            any("_replay_buffer." in key
                for key in v6_agent.state_dict().keys()))
        self._assert_module_state_equal(v6_alg._reference_actor_networks,
                                        v6_alg._actor_networks)
        self._assert_module_state_equal(v6_alg._snapshot_critic_networks,
                                        v6_alg._critic_networks)


if __name__ == "__main__":
    alf.test.main()
