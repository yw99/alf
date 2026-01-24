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
import torch

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.td3_algorithm import Td3Algorithm
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.examples.benchmarks.dm_control import dmc_conf
from alf.optimizers import Adam

alf.define_config('debug_mode', False)
debug_mode = alf.get_config_value('debug_mode')

# Enable find_unused_parameters for DDP (needed for multi-GPU training)
alf.config('make_ddp_performer', find_unused_parameters=True)

optimizer = Adam(lr=3e-4)
use_obs_normalizer = True
obs_normalizer_clipping = False

if debug_mode:
    hidden_layers = (32, 32)
    initial_collect_steps = 1000
else:
    hidden_layers = (256, 256)
    initial_collect_steps = None  # use default from dmc_conf

if use_obs_normalizer:
    data_transformer_ctor = ObservationNormalizer
else:
    data_transformer_ctor = None
if obs_normalizer_clipping:
    alf.config('ObservationNormalizer', clipping=1.)

# TD3 uses deterministic actor (ActorNetwork), not stochastic
actor_network_cls = partial(
    alf.networks.ActorNetwork,
    fc_layer_params=hidden_layers)

critic_network_cls = partial(
    alf.networks.CriticNetwork,
    joint_fc_layer_params=hidden_layers)

alf.config('Agent',
           optimizer=optimizer,
           rl_algorithm_cls=Td3Algorithm)

alf.config(
    'Td3Algorithm',
    actor_network_ctor=actor_network_cls,
    critic_network_ctor=critic_network_cls,
    num_critic_replicas=2,
    # Target policy smoothing (TD3 key feature)
    target_noise_stddev=0.2,
    target_noise_clip=0.5,
    # Delayed actor updates (similar to BAFCv3)
    actor_utd=1,
    critic_utd=3,
    # Target network update
    target_update_tau=0.005,
    target_update_period=1,
    # Exploration noise (OU process)
    ou_stddev=0.2,
    ou_damping=0.15)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    data_transformer_ctor=data_transformer_ctor,
    enable_amp=False,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_updates_per_train_iter=12,
    num_env_steps=int(1e6),
    mini_batch_size=256,
    evaluate=False,
    debug_summaries=False,
    summary_interval=1000,
    summarize_grads_and_vars=False,
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    summarize_train_every_mini_batch=True,
    random_seed=0)

if initial_collect_steps is not None:
    alf.config('TrainerConfig', initial_collect_steps=initial_collect_steps)
