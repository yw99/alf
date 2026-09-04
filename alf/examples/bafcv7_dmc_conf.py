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
"""BAFCv7 DMC experiment with base-ensemble and seeded-single presets."""

from functools import partial

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.bafc_algorithm_v7 import BafcAlgorithmV7
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.environments import suite_dmc
from alf.environments.gym_wrappers import FrameSkip
from alf.networks.bafc_v7_actor_network import BafcV7ActorNetwork
from alf.optimizers import Adam
from alf.utils.math_ops import clipped_exp


alf.define_config("bafcv7_env_name", None)
env_name = alf.get_config_value("bafcv7_env_name")
if env_name is None:
    raise ValueError(
        "BAFCv7 does not define a default environment. Set "
        "bafcv7_env_name from a launcher or --conf_param.")

alf.define_config("bafcv7_variant", "ensemble_base")
variant = alf.get_config_value("bafcv7_variant")
if variant == "ensemble_base":
    variant_args = dict(
        num_actors=10,
        num_critics=10,
        temporal_noise_mix=0.10,
        training_policy="base",
        actor_update_mode="paired")
elif variant == "single_seeded":
    variant_args = dict(
        num_actors=1,
        num_critics=10,
        temporal_noise_mix=0.90,
        training_policy="seeded",
        actor_update_mode="min_all")
else:
    raise ValueError(
        "bafcv7_variant must be 'ensemble_base' or 'single_seeded'; got "
        f"{variant!r}")

hidden_layers = (256, 256)

alf.config(
    "create_environment",
    env_name=env_name,
    num_parallel_environments=1,
    env_load_fn=suite_dmc.load)
alf.config(
    "suite_dmc.load",
    from_pixels=False,
    gym_env_wrappers=(partial(FrameSkip, skip=1), ),
    max_episode_steps=1000)
alf.config(
    "NormalProjectionNetwork",
    state_dependent_std=True,
    scale_distribution=True,
    std_transform=partial(
        clipped_exp, clip_value_min=-20, clip_value_max=2))
alf.config("make_ddp_performer", find_unused_parameters=True)

optimizer = Adam(lr=3e-4)
actor_network_cls = partial(
    BafcV7ActorNetwork,
    fc_layer_params=(256, 256),
    use_ln=False)
critic_network_cls = partial(
    alf.networks.FuncCriticNetwork,
    obs_action_joint_fc_layer_params=hidden_layers,
    actor_obs_action_joint_fc_layer_params=(256, 256),
    use_fc_ln=True)

alf.config("Agent", optimizer=optimizer, rl_algorithm_cls=BafcAlgorithmV7)
alf.config(
    "BafcAlgorithmV7",
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    num_sampled_critics_for_actor=1,
    num_actor_eval_samples=512,
    policy_feature_mode="mean_log_std",
    eval_samples_init_method="normal",
    eval_samples_clipping=False,
    actor_eval_type="last_two",
    actor_encoding_dim=None,
    obs_action_encoding_dim=128,
    actor_utd=1,
    critic_utd=3,
    use_random_critic_targets=True,
    num_sampled_critic_targets=1,
    target_critic_tau=0.005,
    target_critic_period=1,
    target_critic_use_ema=False,
    checkpoint_replay_buffer=True,
    **variant_args)

alf.config(
    "TransformerEncoder",
    num_layers=4,
    num_attention_heads=1,
    dropout=0.0)

alf.config(
    "TrainerConfig",
    algorithm_ctor=Agent,
    data_transformer_ctor=ObservationNormalizer,
    temporally_independent_train_step=True,
    use_rollout_state=False,
    async_eval=True,
    initial_collect_steps=10000,
    unroll_length=1,
    mini_batch_length=2,
    enable_amp=False,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_updates_per_train_iter=12,
    num_env_steps=800000,
    mini_batch_size=256,
    num_iterations=0,
    num_checkpoints=1,
    eval_interval=5000,
    num_eval_episodes=5,
    replay_buffer_length=100000,
    evaluate=False,
    debug_summaries=False,
    summary_interval=1000,
    summarize_grads_and_vars=False,
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    summarize_train_every_mini_batch=True,
    random_seed=0)
