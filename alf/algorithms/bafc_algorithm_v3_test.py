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
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import StepType, TimeStep
from alf.experience_replayers.replay_buffer import ReplayBuffer
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


class BafcAlgorithmV3CheckpointTest(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._psutil_patcher = mock.patch(
            "alf.algorithms.algorithm.psutil.Process",
            return_value=_DummyProcess())
        self._psutil_patcher.start()
        self.addCleanup(self._psutil_patcher.stop)

    def _make_alg(self, **kwargs):
        num_actor_critic = kwargs.pop("num_actor_critic", 3)
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_test_"),
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
            num_actor_critic=num_actor_critic,
            num_actor_eval_samples=16,
            **kwargs)

    def _make_agent(self, **kwargs):
        num_actor_critic = kwargs.pop("num_actor_critic", 3)
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_agent_test_"),
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
                BafcAlgorithmV3,
                actor_network_cls=partial(
                    ActorFCNetwork, fc_layer_params=(32, 32)),
                critic_network_cls=partial(
                    FuncCriticNetwork,
                    obs_action_joint_fc_layer_params=(32, 32),
                    actor_obs_action_joint_fc_layer_params=(32, 32)),
                actor_encoder_cls=partial(
                    TransformerEncoder, num_layers=2, num_attention_heads=1),
                num_actor_critic=num_actor_critic,
                num_actor_eval_samples=16,
                **kwargs))

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
        alg._actor_update_counter = 3
        alg._critic_update_counter = 4
        alg._apply_train_mode_grad_flags()

        state = alg.state_dict()
        self.assertIn("_bafc_runtime.training_started", state)

        restored = self._make_alg()
        restored.load_state_dict(state)

        self.assertTrue(restored._training_started)
        self.assertEqual(restored._train_mode, TrainMode.critic)
        self.assertEqual(restored._rollout_actor_id, 2)
        self.assertEqual(restored._actor_update_counter, 3)
        self.assertEqual(restored._critic_update_counter, 4)
        self.assertTrue(all(not p.requires_grad
                            for p in restored._actor_networks.parameters()))
        self.assertTrue(restored._actor_eval_samples.requires_grad)

        legacy_restored = self._make_alg()
        legacy_restored.load_state_dict(self._without_runtime_state(state))
        self.assertTrue(legacy_restored._training_started)

    def test_reweighting_target_cache_round_trip(self):
        alg = self._make_alg(
            track_reweighting_target_observation_cache=True,
            critic_reweighting_target_obs_cache_size=3)
        alg._append_reweighting_target_observations(
            torch.arange(8, dtype=torch.float32).reshape(2, 4))
        alg._append_reweighting_target_observations(
            torch.arange(8, 20, dtype=torch.float32).reshape(3, 4))

        expected = torch.stack([
            torch.arange(8, 12, dtype=torch.float32),
            torch.arange(12, 16, dtype=torch.float32),
            torch.arange(16, 20, dtype=torch.float32),
        ])
        self.assertTensorClose(alg._reweighting_target_observation_cache,
                               expected)

        restored = self._make_alg(
            track_reweighting_target_observation_cache=True,
            critic_reweighting_target_obs_cache_size=3)
        restored.load_state_dict(alg.state_dict())
        self.assertTensorClose(restored._reweighting_target_observation_cache,
                               expected)

    def test_replay_checkpoint_is_save_context_only_and_ranked(self):
        alg = self._make_alg(checkpoint_replay_buffer=True)
        self._attach_replay_buffer(alg, num_items=1)
        self.assertFalse(
            any("_replay_buffer." in key for key in alg.state_dict().keys()))

        with tempfile.TemporaryDirectory() as ckpt_dir:
            checkpointer = Checkpointer(ckpt_dir, algorithm=alg)
            checkpointer.save(10, ddp_rank=0)
            self.assertTrue(
                any("_replay_buffer." in key
                    for key in torch.load(
                        f"{ckpt_dir}/ckpt-10-replay_buffer-rank0")[
                            "algorithm"].keys()))
            self.assertTrue(
                any("_replay_buffer." in key
                    for key in torch.load(f"{ckpt_dir}/ckpt-10-replay_buffer")[
                        "algorithm"].keys()))

            self._attach_replay_buffer(alg, num_items=3)
            checkpointer.save(10, ddp_rank=1)
            self.assertTrue(
                torch.load(f"{ckpt_dir}/ckpt-10-replay_buffer-rank1")[
                    "algorithm"])

            restored = self._make_alg(checkpoint_replay_buffer=True)
            self._attach_replay_buffer(restored, num_items=0)
            restored_checkpointer = Checkpointer(ckpt_dir, algorithm=restored)
            restored_checkpointer.load(10, ddp_rank=1)
            self.assertTensorEqual(restored._replay_buffer._current_pos,
                                   torch.tensor([3]))

            self.assertFalse(
                any("_replay_buffer." in key
                    for key in restored.state_dict().keys()))

    def test_agent_replay_checkpoint_is_save_context_only_and_ranked(self):
        agent = self._make_agent(checkpoint_replay_buffer=True)
        self._attach_replay_buffer(agent, num_items=1)
        self.assertFalse(
            any("_replay_buffer." in key
                for key in agent.state_dict().keys()))

        with tempfile.TemporaryDirectory() as ckpt_dir:
            checkpointer = Checkpointer(ckpt_dir, algorithm=agent)
            checkpointer.save(10, ddp_rank=0)
            rank0_state = torch.load(
                f"{ckpt_dir}/ckpt-10-replay_buffer-rank0")["algorithm"]
            self.assertTrue(
                any(key.startswith("_replay_buffer.")
                    for key in rank0_state.keys()))
            self.assertFalse(
                any(key.startswith("_rl_algorithm._replay_buffer.")
                    for key in rank0_state.keys()))

            self._attach_replay_buffer(agent, num_items=3)
            checkpointer.save(10, ddp_rank=1)

            restored = self._make_agent(checkpoint_replay_buffer=True)
            self._attach_replay_buffer(restored, num_items=0)
            restored_checkpointer = Checkpointer(ckpt_dir, algorithm=restored)
            restored_checkpointer.load(10, ddp_rank=1)
            self.assertTensorEqual(restored._replay_buffer._current_pos,
                                   torch.tensor([3]))

            self.assertFalse(
                any("_replay_buffer." in key
                    for key in restored.state_dict().keys()))

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
                 "alf.algorithms.bafc_algorithm_v3.safe_mean_hist_summary"
             ) as summary_mock, \
             mock.patch(
                 "alf.algorithms.bafc_algorithm_v3.nest_utils.grad",
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
                 "alf.algorithms.bafc_algorithm_v3.safe_mean_hist_summary"
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
        actual = BafcAlgorithmV3._mean_pairwise_cosine(individual_dqda)
        self.assertTensorClose(actual, expected)


if __name__ == "__main__":
    alf.test.main()
