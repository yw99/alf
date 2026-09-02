# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.

from functools import partial
import math

import torch

import alf
from alf.networks.bafc_v7_actor_network import BafcV7ActorNetwork
from alf.networks.projection_networks import NormalProjectionNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class BafcV7ActorNetworkTest(alf.test.TestCase):

    def _make_actor(self, num_actors=3, minimum=-1., maximum=1.):
        projection = partial(
            NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=True)
        return BafcV7ActorNetwork(
            input_tensor_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec(
                (2, ), minimum=minimum, maximum=maximum),
            fc_layer_params=(16, 12),
            n_groups=num_actors,
            continuous_projection_net_ctor=projection)

    def test_gaussian_parameters_features_and_selected_actor_shapes(self):
        actor = self._make_actor(num_actors=4)
        output = actor(torch.randn(7, 4), full_neurons=True)
        self.assertEqual(output.mean.shape, (7, 4, 2))
        self.assertEqual(output.std.shape, (7, 4, 2))
        self.assertEqual(output.policy_features.shape, (7, 4, 4))
        self.assertEqual(output.neurons[-2].shape, (7, 4, 12))
        self.assertTensorEqual(output.neurons[-1], output.policy_features)
        self.assertEqual(output.distribution.batch_shape, (7, 4))
        self.assertEqual(output.distribution.event_shape, (2, ))

        seed = torch.randn(7, 1, 2)
        actions = actor.sample_seeded(output, seed, 0.3)
        actor_id = torch.tensor([0, 1, 2, 3, 0, 1, 2])
        selected = torch.gather(
            actions, 1,
            actor_id[:, None, None].expand(-1, 1, 2)).squeeze(1)
        self.assertEqual(actions.shape, (7, 4, 2))
        self.assertEqual(selected.shape, (7, 2))
        self.assertTrue(torch.all(output.std > 0.))
        self.assertTensorClose(output.policy_features[..., :2], output.mean)
        self.assertTensorClose(output.policy_features[..., 2:],
                               output.std.log())

    def test_seeded_action_has_mean_and_std_head_gradients(self):
        actor = self._make_actor(num_actors=2)
        output = actor(torch.randn(9, 4))
        action = actor.sample_seeded(
            output,
            seed=torch.full((9, 1, 2), 0.4),
            temporal_noise_mix=0.6,
            fresh_noise=torch.full_like(output.mean, 0.7))
        action.square().mean().backward()

        projection = actor._projection_net
        mean_grad = projection._means_projection_layer.weight.grad
        std_grad = projection._std_projection_layer.weight.grad
        self.assertIsNotNone(mean_grad)
        self.assertIsNotNone(std_grad)
        self.assertGreater(mean_grad.abs().sum().item(), 0.)
        self.assertGreater(std_grad.abs().sum().item(), 0.)

    def test_conditional_and_marginal_seeded_statistics(self):
        torch.manual_seed(1234)
        mix = 0.35
        persistent = math.sqrt(1. - mix**2)
        num_samples = 50000

        fixed_seed = torch.tensor(1.25)
        conditional_noise = persistent * fixed_seed + mix * torch.randn(
            num_samples)
        self.assertAlmostEqual(
            conditional_noise.mean().item(),
            persistent * fixed_seed.item(),
            delta=0.01)
        self.assertAlmostEqual(
            conditional_noise.std(unbiased=False).item(), mix, delta=0.01)

        episode_seed = torch.randn(num_samples)
        noise_t0 = persistent * episode_seed + mix * torch.randn(num_samples)
        noise_t1 = persistent * episode_seed + mix * torch.randn(num_samples)
        self.assertAlmostEqual(
            noise_t0.mean().item(), 0., delta=0.015)
        self.assertAlmostEqual(
            noise_t0.var(unbiased=False).item(), 1., delta=0.025)
        correlation = torch.corrcoef(torch.stack([noise_t0, noise_t1]))[0, 1]
        self.assertAlmostEqual(
            correlation.item(), 1. - mix**2, delta=0.02)

    def test_exact_tanh_and_action_range_transform(self):
        actor = self._make_actor(
            num_actors=1, minimum=[-2., 1.], maximum=[4., 5.])
        output = actor(torch.zeros(3, 4))
        pre_squash = torch.tensor([[[0.0, -0.5]], [[1.0, 0.25]],
                                   [[-2.0, 2.0]]])
        actual = actor.apply_transforms(output.distribution, pre_squash)
        midpoint = torch.tensor([1., 3.])
        magnitude = torch.tensor([3., 2.])
        expected = midpoint + magnitude * torch.tanh(pre_squash)
        self.assertTensorClose(actual, expected)

    def test_requires_transformed_normal_projection(self):
        projection = partial(
            NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=False)
        with self.assertRaisesRegex(ValueError, "scale_distribution=True"):
            BafcV7ActorNetwork(
                TensorSpec((4, )),
                BoundedTensorSpec((2, ), minimum=-1., maximum=1.),
                fc_layer_params=(8, 8),
                continuous_projection_net_ctor=projection)


if __name__ == "__main__":
    alf.test.main()
