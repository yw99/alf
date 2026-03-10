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
from alf.algorithms.bafc_algorithm_TR import BafcAlgorithmTR
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.examples.benchmarks.dm_control import dmc_conf
from alf.optimizers import Adam

alf.define_config('debug_mode', False)
debug_mode = alf.get_config_value('debug_mode')

optimizer = Adam(lr=3e-4)
use_obs_normalizer = True
obs_normalizer_clipping = False
actor_use_ln = False
policy_eval_updates_per_epoch = 2
max_improve_steps_per_epoch = 8

if debug_mode:
    actor_hidden_layers = (32, 32)
    joint_hidden_layers = (32, 32)
    num_actor_eval_samples = 64
else:
    actor_hidden_layers = (256, 256)
    joint_hidden_layers = (256, 256)
    num_actor_eval_samples = 512

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
    use_fc_ln=True)

alf.config('Agent', optimizer=optimizer, rl_algorithm_cls=BafcAlgorithmTR)

alf.config(
    'BafcAlgorithmTR',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    num_actor_critic=10,
    actor_critic_pairing=True,
    use_bootstrap_actors=False,
    use_bootstrap_critics=False,
    actor_use_ln=actor_use_ln,
    bootstrap_mask_prob=0.9,
    bootstrap_mask_type='episode',
    num_actor_eval_samples=num_actor_eval_samples,
    eval_samples_init_method='normal',
    eval_samples_clipping=obs_normalizer_clipping,
    actor_eval_type='last_two',
    actor_encoding_dim=None,
    obs_action_encoding_dim=128,
    eval_trust_max=2.0,
    delta_trust_max=2.0,
    policy_eval_updates_per_epoch=policy_eval_updates_per_epoch,
    max_improve_steps_per_epoch=max_improve_steps_per_epoch,
    trust_cov_reg=1e-4,
    on_policy_adaptation=True,
    update_critic_in_improve=False,
    target_critic_tau=0.005,
    target_critic_period=1,
    target_critic_use_ema=False)

alf.config(
    'TransformerEncoder',
    num_layers=4,
    num_attention_heads=1,
    dropout=0.0)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    data_transformer_ctor=data_transformer_ctor,
    enable_amp=False,
    num_env_steps=int(1e6),
    unroll_length=32,
    mini_batch_length=32,
    mini_batch_size=256,
    # BAFC-TR runs one full epoch per train_iter: all critic eval updates
    # followed by all actor improve updates on the same rollout.
    num_updates_per_train_iter=(policy_eval_updates_per_epoch +
                                max_improve_steps_per_epoch),
    evaluate=False,
    debug_summaries=False,
    summary_interval=1000,
    summarize_grads_and_vars=False,
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    summarize_train_every_mini_batch=True,
    random_seed=2)
