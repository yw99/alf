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
"""DMC configuration for replay-refill resumes with BAFCv3 TR3."""

from functools import partial

import alf
from alf.algorithms.bafc_algorithm_v3_tr3 import (
    BafcAlgorithmV3TR3, BafcAlgorithmV3TR3Agent)
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.examples.benchmarks.dm_control import dmc_conf
from alf.optimizers import Adam


alf.config("make_ddp_performer", find_unused_parameters=True)

optimizer = Adam(lr=3e-4)
actor_network_cls = partial(
    alf.networks.ActorFCNetwork, fc_layer_params=(256, 256), use_ln=False)
critic_network_cls = partial(
    alf.networks.FuncCriticNetwork,
    obs_action_joint_fc_layer_params=dmc_conf.hidden_layers,
    actor_obs_action_joint_fc_layer_params=(256, 256),
    use_fc_ln=True)

alf.config(
    "BafcAlgorithmV3TR3Agent",
    optimizer=optimizer,
    rl_algorithm_cls=BafcAlgorithmV3TR3)

alf.config(
    "BafcAlgorithmV3TR3",
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    num_actor_critic=10,
    actor_critic_pairing=False,
    num_sampled_critics_for_actor=8,
    use_random_critic_targets=True,
    num_sampled_critic_targets=1,
    use_bootstrap_actors=False,
    use_bootstrap_critics=False,
    actor_use_ln=False,
    bootstrap_mask_prob=0.9,
    bootstrap_mask_type="episode",
    num_actor_eval_samples=512,
    eval_samples_init_method="normal",
    eval_samples_clipping=False,
    freeze_eval_samples=False,
    actor_eval_type="last_two",
    actor_encoding_dim=None,
    obs_action_encoding_dim=128,
    trust_cov_reg=1e-4,
    trust_metric_num_obs=128,
    trust_metric_num_feature_coords=4,
    trust_metric_update_interval=8,
    monitor_trust_metrics=True,
    checkpoint_replay_buffer=True,
    eval_trust_max=30.0,
    delta_trust_max=2.0,
    enable_eval_rollout_skip_gate=False,
    enable_eval_trust_max_decay=False,
    enable_grad_actor_extend_gate=False,
    enable_critic_reweighting=False,
    critic_reweighting_beta=None,
    critic_reweighting_ridge=None,
    critic_reweighting_solver_iters=1,
    critic_reweighting_max_weight=10.0,
    eval_gate_max_consecutive_rollout_skips=3,
    actor_utd=1,
    critic_utd=11,
    rollout_cycles_per_collect=1,
    target_critic_tau=0.005,
    target_critic_period=1,
    target_critic_use_ema=False)

alf.config(
    "TransformerEncoder",
    num_layers=4,
    num_attention_heads=1,
    dropout=0.0)

alf.config(
    "TrainerConfig",
    algorithm_ctor=BafcAlgorithmV3TR3Agent,
    data_transformer_ctor=ObservationNormalizer,
    enable_amp=False,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_updates_per_train_iter=12,
    num_env_steps=600000,
    mini_batch_size=256,
    evaluate=False,
    rollout_skip_eval=False,
    grad_gate_eval=False,
    debug_summaries=True,
    summary_interval=1000,
    summarize_grads_and_vars=False,
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    summarize_train_every_mini_batch=True,
    random_seed=0)
