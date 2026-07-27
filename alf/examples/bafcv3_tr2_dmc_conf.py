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
from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.examples.benchmarks.dm_control import dmc_conf
from alf.optimizers import Adam

alf.define_config('debug_mode', False)
debug_mode = alf.get_config_value('debug_mode')
alf.define_config('bafcv3_tr2_actor_use_ln', True)
actor_use_ln = alf.get_config_value('bafcv3_tr2_actor_use_ln')

# Enable find_unused_parameters for DDP (needed for multi-GPU training)
alf.config('make_ddp_performer', find_unused_parameters=True)

optimizer = Adam(lr=3e-4)
use_obs_normalizer = True
obs_normalizer_clipping = False
trust_cov_reg = 1e-4
monitor_trust_metrics = True
eval_trust_max = 2.0
eval_trust_max_decay = False
delta_trust_max = 2.0
eval_gate_max_consecutive_rollout_skips = 5
grad_gate_max_consecutive_actor_extensions = 5
rollout_cycles_per_collect = 3
# Rollout cadence (cycle-based) should not change replay update budget.
num_updates_per_train_iter = 12

if debug_mode:
    trust_metric_num_obs = 64
    trust_metric_num_feature_coords = 32

    actor_hidden_layers = (32, 32)
    joint_hidden_layers = (32, 32)
    num_actor_eval_samples = 64
    initial_collect_steps = 1000
else:
    trust_metric_num_obs = 128
    trust_metric_num_feature_coords = 64

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
           rl_algorithm_cls=BafcAlgorithmV3TR2)

alf.config(
    'BafcAlgorithmV3TR2',
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
    freeze_eval_samples=False,
    actor_eval_type='last_two',
    actor_encoding_dim=None,
    obs_action_encoding_dim=128,
    trust_cov_reg=trust_cov_reg,
    trust_metric_num_obs=trust_metric_num_obs,
    trust_metric_num_feature_coords=trust_metric_num_feature_coords,
    monitor_trust_metrics=monitor_trust_metrics,
    eval_trust_max=eval_trust_max,
    delta_trust_max=delta_trust_max,
    trust_metric_update_interval=1,
    enable_eval_rollout_skip_gate=False,
    enable_eval_trust_max_decay=eval_trust_max_decay,
    enable_grad_actor_extend_gate=True,
    eval_gate_max_consecutive_rollout_skips=eval_gate_max_consecutive_rollout_skips,
    grad_gate_max_consecutive_actor_extensions=grad_gate_max_consecutive_actor_extensions,
    rollout_cycles_per_collect=rollout_cycles_per_collect,
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
    num_attention_heads=1,
    dropout=0.0)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    data_transformer_ctor=data_transformer_ctor,
    enable_amp=False,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_updates_per_train_iter=num_updates_per_train_iter,
    num_env_steps=int(1e6),
    mini_batch_size=256,
    evaluate=False,
    grad_gate_eval=False,
    rollout_skip_eval_interval=100,
    grad_gate_eval_interval=100,
    debug_summaries=False,
    summary_interval=1000,
    summarize_grads_and_vars=False,
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    summarize_train_every_mini_batch=True,
    random_seed=2)

if initial_collect_steps is not None:
    alf.config('TrainerConfig', initial_collect_steps=initial_collect_steps)
