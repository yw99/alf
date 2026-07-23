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
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.examples.benchmarks.dm_control import dmc_conf
from alf.optimizers import Adam

alf.define_config('debug_mode', False)
debug_mode = alf.get_config_value('debug_mode')
alf.define_config('bafcv3_actor_use_ln', False)
actor_use_ln = alf.get_config_value('bafcv3_actor_use_ln')
alf.define_config('bafcv3_learning_rate', 3e-4)
learning_rate = alf.get_config_value('bafcv3_learning_rate')
alf.define_config('bafcv3_actor_critic_pairing', True)
actor_critic_pairing = alf.get_config_value('bafcv3_actor_critic_pairing')
alf.define_config('bafcv3_num_sampled_critics_for_actor', 1)
num_sampled_critics_for_actor = alf.get_config_value(
    'bafcv3_num_sampled_critics_for_actor')
alf.define_config('bafcv3_use_random_critic_targets', False)
use_random_critic_targets = alf.get_config_value(
    'bafcv3_use_random_critic_targets')
alf.define_config('bafcv3_num_sampled_critic_targets', 1)
num_sampled_critic_targets = alf.get_config_value(
    'bafcv3_num_sampled_critic_targets')
alf.define_config('bafcv3_num_actor_critic', 10)
num_actor_critic = alf.get_config_value('bafcv3_num_actor_critic')
alf.define_config('bafcv3_use_bootstrap_actors', False)
use_bootstrap_actors = alf.get_config_value('bafcv3_use_bootstrap_actors')
alf.define_config('bafcv3_use_bootstrap_critics', False)
use_bootstrap_critics = alf.get_config_value('bafcv3_use_bootstrap_critics')
alf.define_config('bafcv3_num_attention_heads', 1)
num_attention_heads = alf.get_config_value('bafcv3_num_attention_heads')

# Enable find_unused_parameters for DDP (needed for multi-GPU training)
alf.config('make_ddp_performer', find_unused_parameters=True)

optimizer = Adam(lr=learning_rate)
use_obs_normalizer = True
obs_normalizer_clipping = False

if debug_mode:
    actor_hidden_layers = (32, 32)
    joint_hidden_layers = (32, 32)
    num_actor_eval_samples = 64
    initial_collect_steps = 1000
else:
    actor_hidden_layers = (256, 256)
    joint_hidden_layers = (256, 256)
    num_actor_eval_samples = 512
    initial_collect_steps = None  # use default from dmc_conf

if use_obs_normalizer:
    data_transformer_ctor = ObservationNormalizer
else:
    data_transformer_ctor = None
if obs_normalizer_clipping:
    alf.config('ObservationNormalizer', clipping=1.)

actor_network_cls = partial(
    alf.networks.ActorFCNetwork,
    fc_layer_params=actor_hidden_layers,
    use_ln=actor_use_ln)

critic_network_cls = partial(
    alf.networks.FuncCriticNetwork,
    obs_action_joint_fc_layer_params=dmc_conf.hidden_layers,
    actor_obs_action_joint_fc_layer_params=joint_hidden_layers,
    use_fc_ln=True)  # turning on critic layernorm is crucial for high utd

alf.config('Agent',
           optimizer=optimizer,
           rl_algorithm_cls=BafcAlgorithmV3)

alf.config(
    'BafcAlgorithmV3',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    num_actor_critic=num_actor_critic,
    actor_critic_pairing=actor_critic_pairing,
    num_sampled_critics_for_actor=num_sampled_critics_for_actor,
    use_random_critic_targets=use_random_critic_targets,
    num_sampled_critic_targets=num_sampled_critic_targets,
    use_bootstrap_actors=use_bootstrap_actors,
    use_bootstrap_critics=use_bootstrap_critics,
    actor_use_ln=actor_use_ln,
    bootstrap_mask_prob=0.9,
    bootstrap_mask_type='episode',
    num_actor_eval_samples=num_actor_eval_samples,
    eval_samples_init_method='normal',
    eval_samples_clipping=obs_normalizer_clipping,
    actor_eval_type='last_two',
    actor_encoding_dim=None,
    obs_action_encoding_dim=128,
    checkpoint_replay_buffer=True,
    track_reweighting_target_observation_cache=True,
    actor_utd=1,
    critic_utd=3,
    # actor_encoder_optimizer=Adam(lr=4e-5),
    # eval_samples_optimizer=Adam(lr=4e-5),
    target_critic_tau=0.005,
    target_critic_period=1,
    target_critic_use_ema=False)

alf.config(
    'TransformerEncoder',
    num_layers=4,
    num_attention_heads=num_attention_heads,
    dropout=0.0)

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
    random_seed=2)

if initial_collect_steps is not None:
    alf.config('TrainerConfig', initial_collect_steps=initial_collect_steps)
