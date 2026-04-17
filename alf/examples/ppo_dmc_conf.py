# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from functools import partial

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.examples.benchmarks.dm_control import dmc_conf
from alf.networks import ActorDistributionNetwork, BetaProjectionNetwork
from alf.networks import ValueNetwork
from alf.optimizers import Adam
from alf.utils.math_ops import clipped_exp

alf.define_config('debug_mode', False)
debug_mode = alf.get_config_value('debug_mode')
alf.define_config('projection_type', 'normal')
projection_type = alf.get_config_value('projection_type')

optimizer = Adam(lr=3e-4)
use_obs_normalizer = True

if debug_mode:
    hidden_layers = (64, 64)
    unroll_length = 1
    mini_batch_size = 128
else:
    hidden_layers = dmc_conf.hidden_layers
    unroll_length = 1
    mini_batch_size = 256

# PPO in ALF trains by repeatedly replaying one collected rollout batch.
replay_buffer_length = unroll_length
num_epochs = 4

if use_obs_normalizer:
    data_transformer_ctor = partial(
        ObservationNormalizer, update_mode='rollout', clipping=5.)
else:
    data_transformer_ctor = None

if projection_type == 'beta':
    projection_net_ctor = partial(
        BetaProjectionNetwork, min_concentration=1.)
elif projection_type == 'normal':
    projection_net_ctor = partial(
        alf.networks.NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp)
else:
    raise ValueError(
        "projection_type must be either 'normal' or 'beta'. "
        f"Got: {projection_type}")

actor_network_ctor = partial(
    ActorDistributionNetwork,
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=projection_net_ctor)
value_network_ctor = partial(ValueNetwork, fc_layer_params=hidden_layers)

alf.config(
    'ActorCriticAlgorithm',
    actor_network_ctor=actor_network_ctor,
    value_network_ctor=value_network_ctor,
    optimizer=optimizer)

alf.config(
    'PPOLoss',
    entropy_regularization=1e-4,
    gamma=0.99,
    td_lambda=0.95,
    normalize_advantages=True)

alf.config(
    'Agent',
    rl_algorithm_cls=partial(PPOAlgorithm, loss_class=PPOLoss),
    enforce_entropy_target=False)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    data_transformer_ctor=data_transformer_ctor,
    temporally_independent_train_step=True,
    use_rollout_state=False,
    initial_collect_steps=0,
    whole_replay_buffer_training=True,
    clear_replay_buffer=True,
    replay_buffer_length=replay_buffer_length,
    unroll_length=unroll_length,
    mini_batch_length=1,
    mini_batch_size=mini_batch_size,
    num_updates_per_train_iter=num_epochs,
    num_iterations=0,
    num_env_steps=int(1e6),
    evaluate=False,
    debug_summaries=False,
    summary_interval=1000,
    summarize_grads_and_vars=False,
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    summarize_train_every_mini_batch=True,
    random_seed=2)
