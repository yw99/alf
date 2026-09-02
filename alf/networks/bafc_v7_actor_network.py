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
"""BAFCv7-only parallel Gaussian actor network.

This module intentionally does not alter ``ActorProjectionFCNetwork``.  It
provides the extra distribution-parameter interface needed by BAFCv7 while
reusing ALF's existing ``NormalProjectionNetwork`` and action transforms.
"""

import functools
import math
from typing import Optional

import torch
import torch.distributions as td

import alf
import alf.layers as layers
from alf.data_structures import namedtuple
from alf.initializers import variance_scaling_init
from alf.networks.network import Network
from alf.networks.projection_networks import NormalProjectionNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils import dist_utils


BafcV7ActorOutput = namedtuple(
    "BafcV7ActorOutput", [
        "distribution", "mean", "std", "policy_features", "neurons",
        "state"
    ],
    default_value=())


@alf.configurable
class BafcV7ActorNetwork(Network):
    """Parallel fully-connected actor exposing a squashed Normal policy.

    The actor group dimension is part of the distribution batch shape.  The
    final policy feature for each actor is ``concat(mean, log(std))`` in the
    pre-squash Normal coordinates.
    """

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 fc_layer_params=(256, 256),
                 n_groups: int = 1,
                 use_bias: bool = True,
                 use_ln: bool = False,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 continuous_projection_net_ctor=None,
                 name="BafcV7ActorNetwork"):
        super().__init__(input_tensor_spec, name=name)
        if not isinstance(action_spec, BoundedTensorSpec):
            raise TypeError("BafcV7ActorNetwork requires a bounded action spec")
        if action_spec.ndim != 1:
            raise ValueError("BafcV7ActorNetwork only supports 1-D action specs")
        if not isinstance(n_groups, int) or n_groups < 1:
            raise ValueError("n_groups must be a positive integer")
        if not isinstance(fc_layer_params, tuple) or not fc_layer_params:
            raise ValueError("fc_layer_params must be a non-empty tuple")

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=math.sqrt(1.0 / 3),
                mode="fan_in",
                distribution="uniform")

        input_size = input_tensor_spec.shape[0]
        self._fc_layers = torch.nn.ModuleList()
        for size in fc_layer_params:
            self._fc_layers.append(
                layers.ParallelFC(
                    input_size,
                    size,
                    n=n_groups,
                    activation=activation,
                    use_bias=use_bias,
                    use_ln=use_ln,
                    kernel_initializer=kernel_initializer))
            input_size = size

        if continuous_projection_net_ctor is None:
            continuous_projection_net_ctor = NormalProjectionNetwork
        self._projection_net = continuous_projection_net_ctor(
            input_size=input_size,
            action_spec=action_spec,
            parallelism=n_groups)
        if not isinstance(self._projection_net, NormalProjectionNetwork):
            raise TypeError(
                "BAFCv7 initially supports only NormalProjectionNetwork")
        if not self._projection_net._scale_distribution:
            raise ValueError(
                "BAFCv7 requires scale_distribution=True so actions use the "
                "exact tanh and action-range transforms")

        self._n_groups = n_groups
        self._action_spec = action_spec
        action_dim = action_spec.shape[0]

        # BAFC uses these properties only to infer actor-token sizes.  The
        # projection feature is [mean, log_std], hence width 2 * action_dim.
        self.register_buffer(
            "_projection_weight_shape",
            torch.zeros(n_groups, 2 * action_dim, input_size))
        self.register_buffer("_projection_bias_shape",
                             torch.zeros(n_groups, 2 * action_dim))
        self._weight_params = [m.weight for m in self._fc_layers] + [
            self._projection_weight_shape
        ]
        self._bias_params = [m.bias for m in self._fc_layers] + [
            self._projection_bias_shape
        ]

    @property
    def weight_params(self):
        return self._weight_params

    @property
    def bias_params(self):
        return self._bias_params

    @property
    def num_actors(self):
        return self._n_groups

    @property
    def action_spec(self):
        return self._action_spec

    @staticmethod
    def seeded_parameters(mean, std, seed, temporal_noise_mix: float):
        """Return the Normal parameters conditional on ``seed``.

        ``seed`` must already be reshaped to broadcast against ``mean`` and
        ``std``.  This makes the helper work for both rollout tensors
        ``[B, A, D]`` and probe-state tensors ``[B, P, A, D]``.
        """
        mix = float(temporal_noise_mix)
        if not 0. < mix <= 1.:
            raise ValueError("temporal_noise_mix must be in (0, 1]")
        persistent_scale = math.sqrt(1. - mix * mix)
        conditional_mean = mean + persistent_scale * std * seed
        conditional_std = (std * mix).expand_as(conditional_mean)
        return conditional_mean, conditional_std

    @staticmethod
    def apply_transforms(distribution, pre_squash_action):
        """Apply exactly the transforms owned by ``distribution``."""
        if not isinstance(distribution, td.TransformedDistribution):
            raise TypeError(
                "BAFCv7 requires a transformed Normal action distribution")
        action = pre_squash_action
        for transform in distribution.transforms:
            action = transform(action)
        return action

    def sample_base(self, output: BafcV7ActorOutput):
        return output.distribution.rsample()

    def sample_seeded(self,
                      output: BafcV7ActorOutput,
                      seed,
                      temporal_noise_mix: float,
                      fresh_noise: Optional[torch.Tensor] = None):
        conditional_mean, conditional_std = self.seeded_parameters(
            output.mean, output.std, seed, temporal_noise_mix)
        if fresh_noise is None:
            fresh_noise = torch.randn_like(conditional_mean)
        pre_squash_action = conditional_mean + conditional_std * fresh_noise
        return self.apply_transforms(output.distribution, pre_squash_action)

    def mode(self, output: BafcV7ActorOutput):
        return dist_utils.get_rmode(output.distribution)

    def forward(self, inputs, full_neurons=False, state=()):
        x = inputs
        hidden = []
        for layer in self._fc_layers:
            x = layer(x)
            hidden.append(x)

        distribution, projection_state = self._projection_net(x)
        base_normal = dist_utils.get_base_dist(distribution)
        if not isinstance(base_normal, td.Normal):
            raise TypeError("BAFCv7 requires a Normal base distribution")
        mean = base_normal.loc
        std = base_normal.scale
        log_std = torch.log(std.clamp_min(torch.finfo(std.dtype).tiny))
        policy_features = torch.cat([mean, log_std], dim=-1)
        # Match ActorFCNetwork's full_neurons convention by appending the
        # policy output after all hidden activations. Thus BAFC's "last_two"
        # means [final_hidden, policy_output]; here policy_output is the
        # deterministic [mean, log_std] fingerprint rather than an action.
        neurons = hidden + [policy_features] if full_neurons else ()
        return BafcV7ActorOutput(
            distribution=distribution,
            mean=mean,
            std=std,
            policy_features=policy_features,
            neurons=neurons,
            state=projection_state)
