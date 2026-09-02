# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.

from functools import partial
import tempfile
from unittest import mock

import torch

import alf
from alf.algorithms.bafc_algorithm_v7 import (BafcAlgorithmV7, BafcV7Info)
from alf.algorithms.config import TrainerConfig
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import StepType, TimeStep
from alf.networks import FuncCriticNetwork, TransformerEncoder
from alf.networks.bafc_v7_actor_network import BafcV7ActorNetwork
from alf.networks.projection_networks import NormalProjectionNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class _ActionOnlyCritic(torch.nn.Module):

    def forward(self, inputs, state=()):
        actor_encoding, (_, action) = inputs
        value = action.sum(dim=-1) + 0. * actor_encoding.sum(dim=-1)
        return value, state


class _EncodingOnlyCritic(torch.nn.Module):

    def forward(self, inputs, state=()):
        actor_encoding, (_, action) = inputs
        value = actor_encoding.sum(dim=-1) + 0. * action.sum(dim=-1)
        return value, state


class BafcAlgorithmV7Test(alf.test.TestCase):

    def _make_alg(self,
                  num_actors=1,
                  num_critics=3,
                  training_policy="seeded",
                  actor_update_mode="min_all",
                  temporal_noise_mix=0.9,
                  **kwargs):
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafcv7_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=4)
        projection = partial(
            NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=True)
        actor_cls = partial(
            BafcV7ActorNetwork,
            fc_layer_params=(16, 12),
            continuous_projection_net_ctor=projection)
        critic_cls = partial(
            FuncCriticNetwork,
            obs_action_joint_fc_layer_params=(16, 12),
            actor_obs_action_joint_fc_layer_params=(16, 12))
        encoder_cls = partial(
            TransformerEncoder,
            num_layers=1,
            num_attention_heads=1,
            dropout=0.)
        return BafcAlgorithmV7(
            observation_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec(
                (2, ), minimum=-1., maximum=1.),
            config=config,
            actor_network_cls=actor_cls,
            critic_network_cls=critic_cls,
            actor_encoder_cls=encoder_cls,
            num_actor_eval_samples=8,
            actor_utd=1,
            critic_utd=3,
            num_actors=num_actors,
            num_critics=num_critics,
            training_policy=training_policy,
            actor_update_mode=actor_update_mode,
            temporal_noise_mix=temporal_noise_mix,
            **kwargs)

    def _state(self, alg, batch_size):
        return alf.nest.map_structure(
            lambda spec: spec.zeros((batch_size, )), alg.rollout_state_spec)

    def _time_step(self, step_type):
        batch_size = step_type.shape[0]
        return TimeStep(
            step_type=step_type,
            reward=torch.zeros(batch_size),
            discount=torch.ones(batch_size),
            observation=torch.randn(batch_size, 4),
            prev_action=torch.zeros(batch_size, 2),
            env_id=torch.arange(batch_size))

    def test_per_environment_seed_and_actor_id_reset_only_on_first(self):
        alg = self._make_alg(
            num_actors=3,
            num_critics=3,
            training_policy="base",
            actor_update_mode="paired")
        state = self._state(alg, 4).action._replace(
            episode_seed=torch.full((4, 2), -3.),
            rollout_actor_id=torch.tensor([2, 2, 2, 2]))
        step_type = torch.tensor(
            [StepType.FIRST, StepType.MID, StepType.FIRST, StepType.LAST])
        sampled_seed = torch.tensor([[0., 1.], [4., 5.]])
        sampled_id = torch.tensor([0, 1])
        with mock.patch(
                "alf.algorithms.bafc_algorithm_v7.torch.randn_like",
                return_value=sampled_seed), mock.patch(
                    "alf.algorithms.bafc_algorithm_v7.torch.randint",
                    return_value=sampled_id):
            updated = alg._resample_episode_state(step_type, state)
        self.assertTensorEqual(updated.episode_seed[0], sampled_seed[0])
        self.assertTensorEqual(updated.episode_seed[2], sampled_seed[1])
        self.assertTensorEqual(updated.episode_seed[1], state.episode_seed[1])
        self.assertTensorEqual(updated.episode_seed[3], state.episode_seed[3])
        self.assertTensorEqual(updated.rollout_actor_id,
                               torch.tensor([0, 2, 1, 2]))

    def test_initial_collection_is_uniform_and_seed_independent(self):
        alg = self._make_alg()
        state = self._state(alg, 32)
        time_step = self._time_step(
            torch.full((32, ), StepType.MID, dtype=torch.int64))
        state_a = state._replace(
            action=state.action._replace(
                episode_seed=torch.full((32, 2), -10.)))
        state_b = state._replace(
            action=state.action._replace(
                episode_seed=torch.full((32, 2), 10.)))
        with mock.patch.object(
                alg._actor_networks,
                "sample_seeded",
                side_effect=AssertionError(
                    "initial collection must not call the seeded actor")):
            torch.manual_seed(77)
            action_a = alg.rollout_step(time_step, state_a).output
            torch.manual_seed(77)
            action_b = alg.rollout_step(time_step, state_b).output
        self.assertTensorEqual(action_a, action_b)
        self.assertTrue(torch.all(action_a >= -1.))
        self.assertTrue(torch.all(action_a <= 1.))

    def test_predict_uses_deterministic_actor_zero_and_ignores_seed(self):
        alg = self._make_alg(
            num_actors=3,
            num_critics=3,
            training_policy="base",
            actor_update_mode="paired")
        state = self._state(alg, 4).action
        observation = torch.randn(4, 4)
        time_step = self._time_step(
            torch.full((4, ), StepType.MID, dtype=torch.int64))._replace(
                observation=observation)
        state_a = state._replace(
            episode_seed=torch.full((4, 2), -9.),
            rollout_actor_id=torch.tensor([0, 1, 2, 1]))
        state_b = state._replace(
            episode_seed=torch.full((4, 2), 9.),
            rollout_actor_id=torch.tensor([2, 2, 1, 0]))
        action_a = alg.predict_step(time_step, state_a).output
        action_b = alg.predict_step(time_step, state_b).output
        expected = alg._actor_networks.mode(
            alg._actor_networks(observation))[:, 0, :]
        self.assertTensorEqual(action_a, action_b)
        self.assertTensorClose(action_a, expected)

    def test_base_training_ignores_seed_but_seeded_training_uses_it(self):
        observation = torch.randn(5, 4)
        seed_a = torch.zeros(5, 2)
        seed_b = torch.full((5, 2), 2.)

        base = self._make_alg(
            num_actors=3,
            num_critics=3,
            training_policy="base",
            actor_update_mode="paired")
        torch.manual_seed(12)
        action_a, _ = base._training_action(observation, seed_a)
        torch.manual_seed(12)
        action_b, _ = base._training_action(observation, seed_b)
        encoding_a, _ = base._training_encoding(base._actor_eval_samples,
                                                seed_a)
        encoding_b, _ = base._training_encoding(base._actor_eval_samples,
                                                seed_b)
        self.assertTensorEqual(action_a, action_b)
        self.assertTensorClose(encoding_a, encoding_b)
        self.assertEqual(encoding_a.shape, (3, 8))

        seeded = self._make_alg()
        torch.manual_seed(12)
        action_a, _ = seeded._training_action(observation, seed_a)
        torch.manual_seed(12)
        action_b, _ = seeded._training_action(observation, seed_b)
        encoding_a, _ = seeded._training_encoding(
            seeded._actor_eval_samples, seed_a)
        encoding_b, _ = seeded._training_encoding(
            seeded._actor_eval_samples, seed_b)
        self.assertFalse(torch.allclose(action_a, action_b))
        self.assertFalse(torch.allclose(encoding_a, encoding_b))
        self.assertEqual(encoding_a.shape, (5, 1, 8))

    def test_fixed_pairing_and_minimum_critic_gradients(self):
        paired = self._make_alg(
            num_actors=3,
            num_critics=3,
            training_policy="base",
            actor_update_mode="paired")
        q_value = torch.arange(
            18, dtype=torch.float32).reshape(2, 3, 3).requires_grad_()
        selected = paired._actor_objective_values(q_value)
        self.assertTensorEqual(selected,
                               torch.stack([q_value[:, i, i]
                                            for i in range(3)], dim=1))
        selected.sum().backward()
        expected_grad = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
        self.assertTensorEqual(q_value.grad, expected_grad)

        minimum = self._make_alg(num_actors=1, num_critics=10)
        q_value = torch.tensor(
            [[[4., 3., -2., 8., 1., 7., 9., 0., 5., 6.]]],
            requires_grad=True)
        selected = minimum._actor_objective_values(q_value)
        self.assertTensorEqual(selected, torch.tensor([[-2.]]))
        selected.sum().backward()
        expected_grad = torch.zeros_like(q_value)
        expected_grad[..., 2] = 1.
        self.assertTensorEqual(q_value.grad, expected_grad)

    def test_one_actor_ten_critic_product_and_random_target(self):
        alg = self._make_alg(num_actors=1, num_critics=10)
        observation = torch.randn(6, 4)
        seed = torch.randn(6, 2)
        action, _ = alg._training_action(observation, seed)
        encoding, _ = alg._training_encoding(alg._actor_eval_samples, seed)
        values, _ = alg._all_critic_values(
            alg._critic_networks, encoding, observation, action, ())
        self.assertEqual(values.shape, (6, 1, 10))

        targets = torch.arange(60, dtype=torch.float32).reshape(6, 1, 10)
        with mock.patch(
                "alf.algorithms.bafc_algorithm_v7.torch.randperm",
                return_value=torch.tensor([7, 0, 1, 2, 3, 4, 5, 6, 8, 9])):
            selected = alg._select_critic_targets(targets)
        self.assertTensorEqual(selected, targets[..., 7])

    def test_both_actor_gradient_paths_reach_gaussian_heads(self):
        alg = self._make_alg(
            num_actors=1,
            num_critics=2,
            actor_update_mode="mean_all")
        observation = torch.randn(5, 4)
        seed = torch.randn(5, 2)

        def _run_path(critic):
            alg.zero_grad()
            action, _ = alg._training_action(observation, seed)
            encoding, features = alg._training_encoding(
                alg._actor_eval_samples, seed)
            alg._critic_networks = critic
            _, actor_info = alg._actor_train_step(
                observation, action, encoding, features,
                torch.zeros(5, 2), ())
            (actor_info.loss.mean() +
             actor_info.extra.eval_action_loss.mean()).backward()
            projection = alg._actor_networks._projection_net
            return (projection._means_projection_layer.weight.grad,
                    projection._std_projection_layer.weight.grad)

        action_grads = _run_path(_ActionOnlyCritic())
        self.assertGreater(action_grads[0].abs().sum().item(), 0.)
        self.assertGreater(action_grads[1].abs().sum().item(), 0.)
        encoding_grads = _run_path(_EncodingOnlyCritic())
        self.assertGreater(encoding_grads[0].abs().sum().item(), 0.)
        self.assertGreater(encoding_grads[1].abs().sum().item(), 0.)

    def test_td_target_uses_next_value_and_terminal_discount(self):
        loss = OneStepTDLoss(gamma=0.99)
        info = BafcV7Info(
            reward=torch.zeros(3, 1),
            step_type=torch.tensor([[StepType.FIRST], [StepType.MID],
                                    [StepType.LAST]]),
            discount=torch.tensor([[1.], [1.], [0.]]))
        target_value = torch.tensor([[[10.]], [[20.]], [[30.]]])
        target = loss.compute_td_target(info, target_value)
        self.assertTensorClose(target[0], torch.tensor([[19.8]]))
        self.assertTensorEqual(target[1], torch.tensor([[0.]]))

    def test_runtime_checkpoint_state_is_isolated_and_restored(self):
        alg = self._make_alg()
        alg._training_started = True
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 5
        alg._critic_update_counter = 7
        alg._apply_train_mode_grad_flags()
        state_dict = alg.state_dict()
        self.assertIn("_bafcv7_runtime.training_started", state_dict)

        restored = self._make_alg()
        restored.load_state_dict(state_dict)
        self.assertTrue(restored._training_started)
        self.assertEqual(restored._train_mode, TrainMode.actor)
        self.assertEqual(restored._actor_update_counter, 5)
        self.assertEqual(restored._critic_update_counter, 7)
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in restored._actor_networks.parameters()))
        self.assertFalse(restored._actor_eval_samples.requires_grad)

    def test_configuration_validation(self):
        with self.assertRaisesRegex(ValueError, "temporal_noise_mix"):
            self._make_alg(temporal_noise_mix=0.)
        with self.assertRaisesRegex(ValueError, "exactly one actor"):
            self._make_alg(num_actors=2, num_critics=2)
        with self.assertRaisesRegex(ValueError, "num_actors == num_critics"):
            self._make_alg(
                num_actors=2,
                num_critics=3,
                training_policy="base",
                actor_update_mode="paired")


if __name__ == "__main__":
    alf.test.main()
