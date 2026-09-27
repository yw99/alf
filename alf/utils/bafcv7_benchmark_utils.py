# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
"""Synthetic V7 workloads shared by equivalence tests and benchmarks.

The large workload matches dog:run V7 dimensions and optimization settings.
It does not create environments, touch experiment results, or start trainers.
"""
from functools import partial

import torch
import alf
from alf.algorithms.bafc_algorithm_v7 import BafcAlgorithmV7, BafcV7Info
from alf.algorithms.config import TrainerConfig
from alf.data_structures import StepType, TimeStep
from alf.networks import FuncCriticNetwork, TransformerEncoder
from alf.networks.bafc_v7_actor_network import BafcV7ActorNetwork
from alf.networks.bafc_v7_critic_network import BafcV7FuncCriticNetwork
from alf.networks.projection_networks import NormalProjectionNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils.math_ops import clipped_exp


FLAGS = ('cache_frozen_probe_outputs', 'deduplicate_critic_episode_seeds',
         'selective_critic_evaluation', 'share_critic_observation_encoding')


def make_algorithm(optimized, variant='single_seeded', mode='mean_log_std',
                   source='frozen', dropout=0., large=False, flags=None):
    """Identical architectures and training settings for both execution paths."""
    obs_dim, act_dim = (223, 38) if large else (4, 2)
    widths = (256, 256) if large else (16, 12)
    config = TrainerConfig(root_dir='/tmp/bafcv7_validation', unroll_length=1,
                           mini_batch_length=2, mini_batch_size=64 if large else 4,
                           initial_collect_steps=0, num_updates_per_train_iter=12)
    critic = BafcV7FuncCriticNetwork if optimized else FuncCriticNetwork
    projection_options = {}
    if large:
        projection_options['std_transform'] = partial(
            clipped_exp, clip_value_min=-20, clip_value_max=2)
    projection = partial(
        NormalProjectionNetwork, state_dependent_std=True,
        scale_distribution=True, **projection_options)
    enabled_flags = {
        flag: optimized and (flags is None or flag in flags) for flag in FLAGS
    }
    return BafcAlgorithmV7(
        TensorSpec((obs_dim,)), BoundedTensorSpec((act_dim,), minimum=-1., maximum=1.),
        config=config,
        actor_network_cls=partial(
            BafcV7ActorNetwork, fc_layer_params=widths,
            continuous_projection_net_ctor=projection),
        critic_network_cls=partial(critic, obs_action_joint_fc_layer_params=widths,
                                  actor_obs_action_joint_fc_layer_params=widths,
                                  use_fc_ln=True),
        actor_encoder_cls=partial(TransformerEncoder, num_layers=4 if large else 1,
                                  num_attention_heads=1, dropout=dropout),
        num_actors=(10 if large else 3) if variant == 'ensemble_base' else 1,
        num_critics=10 if large else 3, num_actor_eval_samples=512 if large else 8,
        obs_action_encoding_dim=128 if large else 8, actor_encoding_dim=None,
        actor_eval_type='last_two', policy_feature_mode=mode,
        eval_samples_source=source, temporal_noise_mix=.1 if variant == 'ensemble_base' else .9,
        training_policy='base' if variant == 'ensemble_base' else 'seeded',
        actor_update_mode='paired' if variant == 'ensemble_base' else 'min_all',
        actor_utd=1, critic_utd=3, num_sampled_critic_targets=1,
        dqda_clipping=None if large else 1., **enabled_flags)


def make_batch(alg, batch_size=8, unique=2):
    obs = alg._observation_spec.randn((batch_size,))
    action = alg._action_spec.zeros((batch_size,)) + torch.rand(
        batch_size, *alg._action_spec.shape, device=obs.device) * 2 - 1
    seeds = torch.randn(unique, *alg._action_spec.shape, device=obs.device)
    seeds = seeds[torch.arange(batch_size, device=obs.device) % unique]
    dtype = next(alg.parameters()).dtype
    obs, action, seeds = obs.to(dtype), action.to(dtype), seeds.to(dtype)
    time = TimeStep(step_type=torch.full((batch_size,), StepType.MID,
                                        dtype=torch.int32, device=obs.device),
                    reward=torch.randn(batch_size, device=obs.device),
                    discount=torch.ones(batch_size, device=obs.device),
                    observation=obs, prev_action=action,
                    env_id=torch.arange(batch_size, device=obs.device))
    info = BafcV7Info(action=action, episode_seed=seeds,
                     rollout_actor_id=torch.zeros(batch_size, dtype=torch.int64,
                                                 device=obs.device))
    return time, info


def compute_loss(alg, batch):
    time, rollout = batch
    state = alf.nest.map_structure(lambda s: s.zeros((time.observation.shape[0],)),
                                   alg.train_state_spec)
    result = alg.train_step(time, state, rollout)
    info = alf.nest.map_structure(
        lambda x: x.reshape(2, -1, *x.shape[1:]), result.info)
    loss = alg.calc_loss(info)
    total = loss.loss.mean()
    if isinstance(loss.scalar_loss, torch.Tensor):
        total = total + loss.scalar_loss
    return total, result.info

