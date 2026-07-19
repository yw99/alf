# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Soft Actor Critic Algorithm."""

from absl import logging
import numpy as np
import functools
from enum import Enum

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, checkpoint_utils
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.schedulers import Scheduler
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.networks.neural_graphs.actor_graph import ActorGraph
from alf.networks.neural_graphs.graph_network import GraphNetwork

BafcActionState = namedtuple(
    "BafcActionState", ["actor_network"], default_value=())

BafcCriticState = namedtuple("BafcCriticState", ["critic", "target_critic"])

BafcState = namedtuple(
    "BafcState", ["action", "actor", "critic"],
    default_value=())

BafcCriticInfo = namedtuple(
    "BafcCriticInfo", ["critic", "target_critic"], default_value=())

BafcActorInfo = namedtuple(
    "BafcActorInfo", ["eval_action_loss"], default_value=())

BafcInfo = namedtuple(
    "BafcInfo", [
        "reward", "step_type", "discount", "action", "actor", "critic", 
        "discounted_return", "bootstrap_mask"
    ],
    default_value=())

BafcLossInfo = namedtuple(
    'BafcLossInfo', ('actor', 'critic'), default_value=())


@alf.configurable
class BafcAlgorithmV3(OffPolicyAlgorithm):
    r"""Boostrapped Actor and Functional Critic algorithm, 

    ::

        Bai et al "Bootstrapped Actors and Functional Critic", arXiv, 2025

    V3 implements model-free posterior sampling style exploration scheme over V2.
    In particular, it has multiple functional critics, each paired with an actor.
    Each functional critic is trained with all actors, while each actor is only
    trained with its paired functional critic.

    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorFCNetwork,
                 critic_network_cls=FuncCriticNetwork,
                 reward_weights=None,
                 calculate_priority=False,
                 num_actor_critic=10,
                 actor_critic_pairing=True,
                 use_bootstrap_actors=False,
                 use_bootstrap_critics=False,
                 actor_use_ln=False,
                 bootstrap_mask_prob=0.8,
                 bootstrap_mask_type='episode',
                 num_actor_eval_samples=256,
                 eval_samples_init_method='normal',
                 eval_samples_clipping=False,
                 actor_eval_type='full',
                 actor_encoder_cls=TransformerEncoder,
                 actor_encoding_dim=128,
                 obs_action_encoding_dim=64,
                 actor_utd: Optional[int] = None,
                 critic_utd: Optional[int] = None,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_critic_tau: Union[float, Scheduler] = 0.05,
                 target_critic_period: Union[int, Scheduler] = 1,
                 target_critic_use_ema=False,
                 parameter_reset_period: Union[int, Scheduler] = -1,
                 dqda_clipping=None,
                 checkpoint_replay_buffer=False,
                 track_reweighting_target_observation_cache=False,
                 critic_reweighting_num_target_obs: int = 128,
                 critic_reweighting_target_obs_cache_size:
                 Optional[int] = None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 actor_encoder_optimizer=None,
                 eval_samples_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="BafcAlgorithm",
                 num_sampled_critics_for_actor=1):
        """
        Args:

            actor_critic_pairing (bool): whether or not fix the 1-1 pairing of actors 
                and critics during actor_train_step (there are the same number of 
                actors and critics, we pair each actor with a unique and different 
                critic during actor training). If True, such a actor-critic pairing is 
                fixed throughout the training. Otherwise, it is randomized at each
                actor_train_step.
            num_sampled_critics_for_actor (int): the number of distinct critics
                used to update each actor when ``actor_critic_pairing`` is False.
                Their gradients are averaged. It must be 1 when pairing is fixed.
            bootstrap_mask_type (str): the type of sampling the bootstrap_mask for
                bootstrapped training of actors and/or critics. There are two types, 
                ``episode`` and ``step``. ``episode`` means a same bootstrap_mask for
                every step of an episode. ``step`` means resampled bootstrap_mask for
                every step of an episode.
        """
        assert actor_eval_type in ['full', 'exclude_input', 'last_two', 'output'], (
            r"{actor_eval_type} in not supported.")
        assert eval_samples_init_method in ['normal', 'uniform'], (
            r"init method {eval_samples_init_method} is not supported.")
        assert bootstrap_mask_type in ['episode', 'step'], (
            r"bootstrap mask type {bootstrap_mask_type} is not supported.")
        assert 1 <= num_sampled_critics_for_actor <= num_actor_critic, (
            "num_sampled_critics_for_actor must be between 1 and "
            f"num_actor_critic ({num_actor_critic}), got "
            f"{num_sampled_critics_for_actor}.")
        assert not actor_critic_pairing or num_sampled_critics_for_actor == 1, (
            "num_sampled_critics_for_actor must be 1 when "
            "actor_critic_pairing is True.")
        assert critic_reweighting_num_target_obs >= 1, (
            "critic_reweighting_num_target_obs must be >= 1")
        if critic_reweighting_target_obs_cache_size is None:
            critic_reweighting_target_obs_cache_size = (
                4 * critic_reweighting_num_target_obs)
        assert critic_reweighting_target_obs_cache_size >= 1, (
            "critic_reweighting_target_obs_cache_size must be >= 1")
        if actor_utd is None and critic_utd is None:
            self._train_mode = TrainMode.standard
        else:
            total_utd = config.num_updates_per_train_iter
            if actor_utd is None:
                assert critic_utd < total_utd, (
                    "critic_utd should be less than num_updates_per_train_iter "
                    "if actor_utd is not provided.")
                actor_utd = total_utd - critic_utd
            elif critic_utd is None:
                assert actor_utd < total_utd, (
                    "actor_utd should be less than num_updates_per_train_iter "
                    "if critic_utd is not provided.")
                critic_utd = total_utd - actor_utd
            assert actor_utd <= critic_utd, (
                f"actor_utd {actor_utd} should not be greater than critic_utd {critic_utd}"
            )
            self._train_mode = TrainMode.critic
            self._actor_utd = actor_utd
            self._critic_utd = critic_utd

        self._num_actor_critic = num_actor_critic
        self._actor_critic_pairing = actor_critic_pairing
        self._num_sampled_critics_for_actor = num_sampled_critics_for_actor
        self._use_bootstrap_actors = use_bootstrap_actors
        self._use_bootstrap_critics = use_bootstrap_critics
        self._bootstrap_mask_prob = bootstrap_mask_prob
        self._bootstrap_mask_type = bootstrap_mask_type
        self._checkpoint_replay_buffer = checkpoint_replay_buffer
        self._track_reweighting_target_observation_cache = (
            track_reweighting_target_observation_cache)
        self._critic_reweighting_target_obs_cache_size = (
            critic_reweighting_target_obs_cache_size)
        self._reweighting_target_observation_cache = ()
        self._bootstrap_mask = ()
        actor_networks = actor_network_cls(
            input_tensor_spec=observation_spec,
            action_spec=action_spec,
            n_groups=num_actor_critic)
        if eval_samples_init_method == 'normal':
            actor_eval_samples = 2 * torch.randn(
                num_actor_eval_samples, observation_spec.shape[0])
            if eval_samples_clipping:
                actor_eval_samples.clip_(min=-1.0, max=1.0)
        else:
            actor_eval_samples = 2 * torch.rand(
                num_actor_eval_samples, observation_spec.shape[0]) - 1

        # extract actor token length from actor_encoder 
        if actor_eval_type == 'full':
            actor_token_length = observation_spec.shape[0] + sum( 
                t.shape[1] for t in actor_networks.bias_params)
        elif actor_eval_type == 'exclude_input':
            actor_token_length = sum( 
                t.shape[1] for t in actor_networks.bias_params)
        elif actor_eval_type == 'last_two':
            actor_token_length = sum( 
                t.shape[1] for t in actor_networks.bias_params[-2:])
        else:
            actor_token_length = action_spec.shape[0]

        actor_token_spec = TensorSpec(
            shape=(actor_token_length, num_actor_eval_samples))
        actor_encoder = actor_encoder_cls(
            actor_token_spec, core_embedding_dim=actor_encoding_dim)

        # functional critic
        if actor_encoding_dim is None:
            actor_encoding_dim = num_actor_eval_samples
        actor_spec = TensorSpec(shape=(actor_encoding_dim,))
        obs_action_spec = (observation_spec, action_spec)
        critic_network = critic_network_cls(
            input_tensor_spec=(actor_spec, obs_action_spec), 
            obs_action_encoding_dim=obs_action_encoding_dim,
            actor_obs_action_combiner=alf.layers.NestConcat(dim=-1))
        critic_networks = critic_network.make_parallel(num_actor_critic)

        action_state_spec = BafcActionState(
            actor_network=actor_networks.state_spec)
        train_state_spec = BafcState(
            action=action_state_spec,
            actor=critic_network.state_spec,
            critic=BafcCriticState(
                critic=critic_networks.state_spec,
                target_critic=critic_networks.state_spec))

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=train_state_spec,
            predict_state_spec=action_state_spec,
            reward_weights=reward_weights,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None and actor_networks is not None:
            self.add_optimizer(actor_optimizer, [actor_networks])
        if critic_optimizer is not None and critic_networks is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if actor_encoder_optimizer is not None:
            self.add_optimizer(actor_encoder_optimizer, [actor_encoder])
        self._actor_eval_samples = nn.Parameter(actor_eval_samples)
        if eval_samples_optimizer is not None:
            self.add_optimizer(eval_samples_optimizer, [self._actor_eval_samples])

        self._actor_networks = actor_networks
        self._actor_use_ln = actor_use_ln
        self._actor_eval_type = actor_eval_type
        self._actor_encoder = actor_encoder
        self._critic_networks = critic_networks
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')
        # self._target_critic_network.set_obs_action_batch_dominate(True)

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(critic_loss_ctor,
                                             debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_actor_critic):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._rollout_actor_id = 0
        self._actor_update_counter = 0
        self._critic_update_counter = 0
        self._dqda_clipping = dqda_clipping
        self._training_started = False
        self._do_critic_summary = False

        def _filter(x):
            return list(filter(lambda x: x is not None, x))

        def _create_target_updater(model_list, target_model_list,
                                   tau, period, use_ema):
            return common.TargetUpdater(
                models=_filter(model_list),
                target_models=_filter(target_model_list),
                tau=tau,
                period=period,
                delayed_update=use_ema)

        self._update_target_critic = _create_target_updater(
            [self._critic_networks], [self._target_critic_networks],
            target_critic_tau, target_critic_period, target_critic_use_ema)

    def _bafc_runtime_key(self, prefix, name):
        return prefix + "_bafc_runtime." + name

    def _bafc_scalar_tensor(self, value, dtype=torch.int64):
        if isinstance(value, torch.Tensor):
            return value.detach().reshape(()).to(dtype=dtype).clone()
        return torch.tensor(value, dtype=dtype)

    def _bafc_scalar_int(self, value):
        return int(torch.as_tensor(value).reshape(()).item())

    def _bafc_runtime_tensor(self, value):
        return torch.as_tensor(value).detach().clone()

    def _save_bafc_runtime_state(self, destination, prefix):
        destination[self._bafc_runtime_key(
            prefix, "training_started")] = self._bafc_scalar_tensor(
                self._training_started, dtype=torch.bool)
        destination[self._bafc_runtime_key(
            prefix, "train_mode")] = self._bafc_scalar_tensor(
                self._train_mode.value)
        destination[self._bafc_runtime_key(
            prefix, "rollout_actor_id")] = self._bafc_scalar_tensor(
                self._rollout_actor_id)
        destination[self._bafc_runtime_key(
            prefix, "actor_update_counter")] = self._bafc_scalar_tensor(
                self._actor_update_counter)
        destination[self._bafc_runtime_key(
            prefix, "critic_update_counter")] = self._bafc_scalar_tensor(
                self._critic_update_counter)
        if isinstance(self._reweighting_target_observation_cache, torch.Tensor):
            destination[self._bafc_runtime_key(
                prefix, "reweighting_target_observation_cache")] = (
                    self._reweighting_target_observation_cache.detach().clone())

    def _pop_bafc_runtime_state(self, state_dict, prefix):
        runtime_prefix = self._bafc_runtime_key(prefix, "")
        runtime_state = {}
        for key in list(state_dict.keys()):
            if key.startswith(runtime_prefix):
                runtime_state[key[len(runtime_prefix):]] = state_dict.pop(key)
        return runtime_state

    def _has_legacy_actor_checkpoint(self, state_dict, prefix):
        return any(key.startswith(prefix + "_actor_networks.")
                   for key in state_dict.keys())

    def _restore_bafc_runtime_state(self, runtime_state):
        if "training_started" in runtime_state:
            self._training_started = bool(
                torch.as_tensor(
                    runtime_state["training_started"]).reshape(()).item())
        if "train_mode" in runtime_state:
            self._train_mode = TrainMode(
                self._bafc_scalar_int(runtime_state["train_mode"]))
        if "rollout_actor_id" in runtime_state:
            self._rollout_actor_id = self._bafc_scalar_int(
                runtime_state["rollout_actor_id"])
        if "actor_update_counter" in runtime_state:
            self._actor_update_counter = self._bafc_scalar_int(
                runtime_state["actor_update_counter"])
        if "critic_update_counter" in runtime_state:
            self._critic_update_counter = self._bafc_scalar_int(
                runtime_state["critic_update_counter"])
        if "reweighting_target_observation_cache" in runtime_state:
            self._reweighting_target_observation_cache = (
                self._bafc_runtime_tensor(
                    runtime_state["reweighting_target_observation_cache"]))
        self._apply_train_mode_grad_flags()

    def _apply_train_mode_grad_flags(self):
        standard_or_initial = (
            self._train_mode == TrainMode.standard or
            (self._actor_update_counter == 0
             and self._critic_update_counter == 0))
        actor_requires_grad = (standard_or_initial
                               or self._train_mode == TrainMode.actor)
        eval_samples_requires_grad = (
            standard_or_initial or self._train_mode == TrainMode.critic)
        for p in self._actor_networks.parameters():
            p.requires_grad_(actor_requires_grad)
        self._actor_eval_samples.requires_grad_(eval_samples_requires_grad)

    def checkpoint_replay_buffer_enabled(self):
        return self._checkpoint_replay_buffer

    def _set_replay_buffer_checkpoint_enabled(self, enabled):
        if not self._checkpoint_replay_buffer or self._replay_buffer is None:
            return None
        old_enabled = checkpoint_utils.is_checkpoint_enabled(self._replay_buffer)
        checkpoint_utils.enable_checkpoint(self._replay_buffer, enabled)

        def _restore():
            checkpoint_utils.enable_checkpoint(self._replay_buffer, old_enabled)

        return _restore

    def _has_replay_buffer_checkpoint(self, state_dict):
        return any(key.startswith("_replay_buffer.")
                   or "._replay_buffer." in key for key in state_dict.keys())

    def _alf_prepare_checkpoint_save(self):
        return self._set_replay_buffer_checkpoint_enabled(True)

    def _alf_prepare_checkpoint_load(self, state_dict):
        return self._set_replay_buffer_checkpoint_enabled(
            self._has_replay_buffer_checkpoint(state_dict))

    def _save_to_state_dict(self, destination, prefix, visited=None):
        super()._save_to_state_dict(destination, prefix, visited)
        self._save_bafc_runtime_state(destination, prefix)

    def _load_from_state_dict(self,
                              state_dict,
                              prefix,
                              local_metadata,
                              strict,
                              missing_keys,
                              unexpected_keys,
                              error_msgs,
                              visited=None):
        runtime_state = self._pop_bafc_runtime_state(state_dict, prefix)
        legacy_actor_checkpoint = (
            not runtime_state and self._has_legacy_actor_checkpoint(
                state_dict, prefix))
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs, visited)
        if runtime_state:
            self._restore_bafc_runtime_state(runtime_state)
        elif legacy_actor_checkpoint:
            self._training_started = True
            self._apply_train_mode_grad_flags()

    def _flatten_reweighting_observations(self, observation):
        if not isinstance(observation, torch.Tensor):
            return ()
        obs_dim = len(self._observation_spec.shape)
        obs = observation.reshape(-1, *observation.shape[-obs_dim:])
        if obs.shape[0] == 0:
            return ()
        return obs

    def _append_reweighting_target_observations(self, observation):
        obs = self._flatten_reweighting_observations(observation)
        if not isinstance(obs, torch.Tensor):
            return
        obs = obs.detach()
        cache = self._reweighting_target_observation_cache
        if isinstance(cache, torch.Tensor):
            if cache.device != obs.device:
                cache = cache.to(obs.device)
            obs = torch.cat([cache, obs], dim=0)
        if obs.shape[0] > self._critic_reweighting_target_obs_cache_size:
            obs = obs[-self._critic_reweighting_target_obs_cache_size:]
        self._reweighting_target_observation_cache = obs

    def _predict_action(self,
                        actor_net,
                        observation,
                        state: BafcActionState,
                        train=False):
        if not self._training_started:
            # get batch size with ``get_outer_rank`` and ``get_nest_shape``
            # since the observation can be a nest in the general case
            outer_rank = nest_utils.get_outer_rank(observation,
                                                   self._observation_spec)
            outer_dims = alf.nest.get_nest_shape(observation)[:outer_rank]
            # This uniform sampling seems important because for a squashed Gaussian,
            # even with a large scale, a random policy is not nearly uniform.
            action = alf.nest.map_structure(
                lambda spec: spec.sample(outer_dims=outer_dims),
                self._action_spec)
            return action, state

        if train:
            action, state = actor_net(
                observation, state=state.actor_network)
        else:
            if self._actor_use_ln:
                action, state = actor_net(
                    observation, state=state.actor_network)
                # [n_env, n_actor, d_a] --> [n_env, d_a]
                action = action[:, self._rollout_actor_id, :]
            else:
                action, state = actor_net(
                    observation,
                    id=self._rollout_actor_id,
                    state=state.actor_network)
        new_state = BafcActionState(actor_network=state)

        return action, new_state

    def predict_step(self, inputs: TimeStep, state: BafcActionState):
        action, action_state = self._predict_action(
            self._actor_networks,
            inputs.observation,
            state=state)
        return AlgStep(
            output=action,
            state=action_state,
            info=BafcInfo(action=action))

    def rollout_step(self, inputs: TimeStep, state: BafcState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        assert not self._is_eval
        if self._track_reweighting_target_observation_cache:
            self._append_reweighting_target_observations(inputs.observation)
        if inputs.step_type == StepType.FIRST or self._bootstrap_mask_type == 'step':
            if inputs.step_type == StepType.FIRST:
                # commitment: only resample rollout actor at the beginning of an episode
                self._rollout_actor_id = torch.randint(self._num_actor_critic, ())
            if self._use_bootstrap_actors or self._use_bootstrap_critics:
                # [n_env, n_actors] masks for bootstrap actors
                prob_t = torch.full(
                    (inputs.step_type.shape[0], self._num_actor_critic),
                    self._bootstrap_mask_prob)
                self._bootstrap_mask = torch.bernoulli(prob_t)

        action, action_state = self._predict_action(
            self._actor_networks,
            inputs.observation,
            state=state.action)
        return AlgStep(
            output=action,
            state=state._replace(action=action_state),
            info=BafcInfo(action=action, bootstrap_mask=self._bootstrap_mask))

    def _tokenize_actor_out(self, eval_out):
        # To make actor eval_out an input sequence to the transformer, we set
        # n_actor as the batch_size, \sum_d as the length of the sequence, 
        # and num_eval_samples B as the dimension of embedding
        if self._actor_eval_type == 'output':
            # [bs, n_actor, d_a] -> [n_actor, d_a, bs]
            eval_out_seq = eval_out.permute(1, 2, 0)
        else:
            # list of [bs, n_actor, di] --> [n_actor, \sum_di, bs]
            eval_out_seq = torch.cat(eval_out, dim=-1).permute(1, 2, 0)

        return eval_out_seq

    def _sample_actor_critic_matchings(self, device=None):
        """Sample balanced mappings from critic slots to actor identities."""
        n = self._num_actor_critic
        if self._actor_critic_pairing:
            matching = torch.arange(n).unsqueeze(0)
        else:
            actor_permutation = torch.randperm(n)
            if self._num_sampled_critics_for_actor == 1:
                # Keep the random-number usage of the existing K=1 path.
                matching = actor_permutation.unsqueeze(0)
            else:
                offsets = torch.randperm(n)[:self._num_sampled_critics_for_actor]
                critic_slots = torch.arange(n).unsqueeze(0)
                matching = actor_permutation[
                    (critic_slots + offsets.unsqueeze(1)) % n]
        if device is not None:
            matching = matching.to(device=device)
        return matching

    def _restore_actor_order(self, matched_value, matched_actor_ids):
        """Reorder the critic-slot dimension of every matching by actor id."""
        trailing_shape = matched_value.shape[3:]
        index = matched_actor_ids[:, None, :, *([None] * len(trailing_shape))]
        index = index.expand_as(matched_value)
        return torch.zeros_like(matched_value).scatter(2, index, matched_value)

    def _aggregate_matched_action_gradients(self, matched_dqda,
                                            matched_actor_ids, action):
        """Scatter matched critic gradients back into actor identity order."""
        del action
        return self._restore_actor_order(matched_dqda,
                                         matched_actor_ids).sum(dim=0)

    @staticmethod
    def _mean_pairwise_cosine(individual_dqda):
        """Mean off-diagonal cosine without constructing a K by K Gram matrix."""
        k = individual_dqda.shape[0]
        flat_grad = individual_dqda.reshape(*individual_dqda.shape[:3], -1)
        norm = flat_grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit_grad = flat_grad / norm
        summed_sq_norm = unit_grad.sum(dim=0).square().sum(dim=-1)
        self_sq_norm = unit_grad.square().sum(dim=-1).sum(dim=0)
        return (summed_sq_norm - self_sq_norm) / (k * (k - 1))

    def _summarize_actor_gradients(self, dqda, clipped_dqda, dqde,
                                   clipped_dqde, current_action, replay_action,
                                   q_value, individual_dqda):
        dqda = dqda.detach()
        safe_mean_hist_summary('actor_gradients/dqda', dqda)
        safe_mean_hist_summary('actor_gradients/dqda_abs', dqda.abs())
        safe_mean_hist_summary('actor_gradients/dqda_l2_norm',
                               dqda.flatten(start_dim=2).norm(dim=-1))
        for i in range(dqda.shape[-1]):
            alf.summary.scalar(
                f'actor_gradients/dqda_abs_component_{i}',
                dqda[..., i].abs().mean())

        if self._dqda_clipping:
            clipped_dqda = clipped_dqda.detach()
            safe_mean_hist_summary('actor_gradients/clipped_dqda', clipped_dqda)
            safe_mean_hist_summary('actor_gradients/clipped_dqda_abs',
                                   clipped_dqda.abs())
            safe_mean_hist_summary(
                'actor_gradients/clipped_dqda_l2_norm',
                clipped_dqda.flatten(start_dim=2).norm(dim=-1))
            for i in range(clipped_dqda.shape[-1]):
                alf.summary.scalar(
                    f'actor_gradients/clipped_dqda_abs_component_{i}',
                    clipped_dqda[..., i].abs().mean())
            alf.summary.scalar(
                'actor_gradients/dqda_clip_fraction',
                dqda.abs().gt(self._dqda_clipping).to(torch.float32).mean())

        for i, (raw, clipped) in enumerate(
                zip(nest.flatten(dqde), nest.flatten(clipped_dqde))):
            raw = raw.detach()
            safe_mean_hist_summary(f'actor_gradients/dqde_leaf_{i}_abs',
                                   raw.abs())
            safe_mean_hist_summary(
                f'actor_gradients/dqde_leaf_{i}_l2_norm',
                raw.flatten(start_dim=max(0, raw.ndim - 1)).norm(dim=-1))
            if self._dqda_clipping:
                clipped = clipped.detach()
                safe_mean_hist_summary(
                    f'actor_gradients/clipped_dqde_leaf_{i}_abs',
                    clipped.abs())
                safe_mean_hist_summary(
                    f'actor_gradients/clipped_dqde_leaf_{i}_l2_norm',
                    clipped.flatten(start_dim=max(0, clipped.ndim - 1)).norm(
                        dim=-1))

        current_action = current_action.detach()
        replay_action = replay_action.detach().unsqueeze(1)
        safe_mean_hist_summary(
            'actor_actions/current_vs_replay_l2',
            (current_action - replay_action).flatten(start_dim=2).norm(dim=-1))
        safe_mean_hist_summary('actor_actions/current_abs',
                               current_action.abs())
        alf.summary.scalar(
            'actor_actions/current_fraction_abs_gt_0_95',
            current_action.abs().gt(0.95).to(torch.float32).mean())

        if individual_dqda is not None:
            q_value = q_value.detach()
            individual_dqda = individual_dqda.detach()
            safe_mean_hist_summary('actor_critic_aggregation/q_mean',
                                   q_value.mean(dim=0))
            safe_mean_hist_summary(
                'actor_critic_aggregation/q_std',
                q_value.std(dim=0, unbiased=False))
            safe_mean_hist_summary(
                'actor_critic_aggregation/individual_dqda_l2_norm',
                individual_dqda.flatten(start_dim=3).norm(dim=-1))
            safe_mean_hist_summary(
                'actor_critic_aggregation/dqda_pairwise_cosine',
                self._mean_pairwise_cosine(individual_dqda))

    def _actor_train_step(self, observation, action, replay_action, mask, state):
        """Compute the exact off-policy policy gradient from the functional critic,
        which consists of two terms, 

        1. the gradient w.r.t. input action, as in standard actor-critic algorithms.

        2. the gradient w.r.t. eval action, i.e., actor_networks' outputs for 
           self._actor_eval_samples.
        """
        ## Step 1: encode all actors from actor_eval_samples
        ####################################################
        eval_action = self._actor_networks(
            self._actor_eval_samples, full_neurons=self._actor_eval_type != 'output')[0]
        if self._actor_eval_type == 'exclude_input':
            eval_action = eval_action[1:]
        elif self._actor_eval_type == 'last_two':
            eval_action = eval_action[-2:]

        actor_tokens = self._tokenize_actor_out(eval_action)
        actor_encoding = self._actor_encoder(actor_tokens)[0]

        k = self._num_sampled_critics_for_actor
        batch_size = observation.shape[0]
        matched_actor_ids = self._sample_actor_critic_matchings(action.device)
        matched_actor_encoding = actor_encoding[matched_actor_ids]
        matched_action = torch.gather(
            action.unsqueeze(0).expand(k, *action.shape),
            dim=2,
            index=matched_actor_ids[:, None, :, None].expand(
                k, batch_size, self._num_actor_critic, action.shape[-1]))

        critic_actor_encoding = matched_actor_encoding[:, None, :, :].expand(
            k, batch_size, self._num_actor_critic,
            matched_actor_encoding.shape[-1]).reshape(
                k * batch_size, self._num_actor_critic,
                matched_actor_encoding.shape[-1])

        ## Step 2: compute critic values for all actors
        ###############################################
        # # [T*B * n_actor, d_s]
        # critic_observation = observation.repeat_interleave(
        #     self._num_actor_critic, dim=0)
        # [K * T*B, n_critic, d_s]
        critic_observation = observation.unsqueeze(0).expand(
            k, *observation.shape).reshape(k * batch_size,
                                           *observation.shape[1:])
        critic_observation = critic_observation.unsqueeze(1).expand(
            k * batch_size, self._num_actor_critic,
            *observation.shape[1:])
        critic_action = matched_action.reshape(
            k * batch_size, self._num_actor_critic, action.shape[-1])
        q_value, critic_state = self._critic_networks(
            (critic_actor_encoding, (critic_observation, critic_action)), state)
        q_value = q_value.reshape(k, batch_size, *q_value.shape[1:])

        ## Step 3: exact off-policy policy gradient (OPG)
        #################################################
        # need to exclude the input actor_eval_samples, since they don't requires_grad
        # for actor TrainMode
        if self._actor_eval_type == 'full':
            eval_action_in_graph = eval_action[1:]
        else:
            eval_action_in_graph = eval_action

        record_debug = (self._debug_summaries
                        and alf.summary.should_record_summaries())
        need_individual_dqda = record_debug and k > 1
        dqda_input = matched_action if need_individual_dqda else action
        dqda, dqde = nest_utils.grad(
            (dqda_input, eval_action_in_graph),
            q_value.sum() / k,
            retain_graph=self._actor_eval_type != 'output')

        individual_dqda = None
        if need_individual_dqda:
            # dqda is scaled by 1/K through the mean-Q objective. Scatter-summing
            # therefore produces the mean selected-critic gradient in actor order.
            individual_dqda = self._restore_actor_order(
                dqda * k, matched_actor_ids)
            dqda = individual_dqda.mean(dim=0)
            q_value = self._restore_actor_order(q_value, matched_actor_ids)

        if self._dqda_clipping:
            clipped_dqda = torch.clamp(dqda, -self._dqda_clipping,
                                       self._dqda_clipping)
            clipped_dqde = nest.map_structure(
                lambda x: torch.clamp(x, -self._dqda_clipping,
                                      self._dqda_clipping), dqde)
        else:
            clipped_dqda = dqda
            clipped_dqde = dqde

        if record_debug:
            self._summarize_actor_gradients(
                dqda, clipped_dqda, dqde, clipped_dqde, action, replay_action,
                q_value, individual_dqda)

        def action_loss_fn(gradient, a_in):
            loss = 0.5 * losses.element_wise_squared_loss(
                (gradient + a_in).detach(), a_in)
            return loss.sum(list(range(2, loss.ndim)))

        # 1st term of OPG: loss corresponding to input action
        action_loss = nest.map_structure(action_loss_fn, clipped_dqda, action)
        if self._use_bootstrap_actors:
            action_loss = action_loss * mask / self._bootstrap_mask_prob
        action_loss = action_loss.sum(-1)

        # 2nd term of OPG: loss corresponding to input eval_action
        eval_action_loss = nest.map_structure(
            action_loss_fn, clipped_dqde, eval_action_in_graph)
        # ALF workaround: reduce to scalar_loss and repeat to [T*B]
        # Will be averaged to a scalar_loss in calc_loss
        eval_action_loss = math_ops.add_n(eval_action_loss).mean().repeat(
            action_loss.shape[0])

        actor_info = LossInfo(
            loss=action_loss,
            extra=BafcActorInfo(eval_action_loss=eval_action_loss)) 
        return critic_state, actor_info

    def _critic_train_step(self, observation, state: BafcCriticState, 
                           rollout_info: BafcInfo, action): 
        ## Step 1: encode all actors from actor_eval_samples
        ####################################################
        eval_action = self._actor_networks(
            self._actor_eval_samples, 
            full_neurons=self._actor_eval_type != 'output')[0]
        if self._actor_eval_type == 'exclude_input':
            eval_action = eval_action[1:]
        elif self._actor_eval_type == 'last_two':
            eval_action = eval_action[-2:]

        actor_tokens = self._tokenize_actor_out(eval_action)
        actor_encoding = self._actor_encoder(actor_tokens)[0]  # [n_actor, d_enc]

        ## Step 2: compute critics and target critics for training actor batch
        ##
        ## use all actors times all (s, a) samples: a [T*B * n_actor] batch   
        ## - critic network gets [n_actor, d_enc] & [T*B, d_sa]
        ##   it performs a cross-prod to form the desired batch
        ## - target_critic network gets [n_actor, d_enc] & [T*B * n_actor, d_sa]
        ##   with "obs_action_batch_dominate", it forms the desired batch
        ######################################################################
        batch_size = observation.shape[0]
        # repeat the entirety of actor_encoding T*S times -> [n_actor * T*S, d_enc]
        actor_encoding = actor_encoding.repeat(batch_size, 1)
        # repeat each row of rollout obs & action n_actor times -> [n_actor * T*S, d_sa]
        critic_observation = observation.repeat_interleave(
            self._num_actor_critic, dim=0)
        critic_action = rollout_info.action.repeat_interleave(
            self._num_actor_critic, dim=0)

        # [n_actor * T*S, n_critic]
        critics, critic_state = self._critic_networks(
            (actor_encoding, (critic_observation, critic_action)), state.critic)

        with torch.no_grad():
            # [T*B, d_s] --> [T*B * n_actor, d_s], same batch size as action
            target_observation = observation.repeat_interleave(
                self._num_actor_critic, dim=0)
            target_critics, target_critic_state = self._target_critic_networks(
                (actor_encoding, (target_observation, action)), state.target_critic)

        # [T*B*n_actor, n_critic] -> [T*S, n_actor, n_critic]
        critics = critics.reshape(-1, self._num_actor_critic, *critics.shape[1:])
        target_critics = target_critics.reshape(
            -1, self._num_actor_critic, *target_critics.shape[1:])
        target_critics = target_critics.detach()

        state = BafcCriticState(
            critic=critic_state, target_critic=target_critic_state)
        info = BafcCriticInfo(critic=critics, target_critic=target_critics)

        return state, info

    def _update_train_mode(self):
        if self._train_mode == TrainMode.actor:
            if self._actor_update_counter % self._actor_utd == 0:
                self._train_mode = TrainMode.critic
                # self._critic_network.set_obs_action_batch_dominate(False)
                for p in self._actor_networks.parameters():
                    p.requires_grad_(False)
                self._actor_eval_samples.requires_grad_(True)
        elif self._train_mode == TrainMode.critic:
            if self._critic_update_counter % self._critic_utd == 0:
                self._train_mode = TrainMode.actor
                # self._critic_network.set_obs_action_batch_dominate(True)
                for p in self._actor_networks.parameters():
                    p.requires_grad_(True)
                self._actor_eval_samples.requires_grad_(False)

    def train_step(self, inputs: TimeStep, state: BafcState,
                   rollout_info: BafcInfo):
        assert not self._is_eval
        self._training_started = True

        # [T*B, n_actor, d_a]
        action, action_state = self._predict_action(
            self._actor_networks, inputs.observation, 
            state=state.action, train=True)

        if self._train_mode == TrainMode.standard or (
                self._critic_update_counter == 0
                and self._actor_update_counter == 0):
            actor_action = action  # [T*B, n_actor, d_a]
            actor_state, actor_info = self._actor_train_step(
                inputs.observation, actor_action, rollout_info.action,
                rollout_info.bootstrap_mask, state.actor)
            critic_action = action.reshape(-1, action.shape[-1])  # [T*B * n_actor, d_a]
            critic_state, critic_info = self._critic_train_step(
                inputs.observation, state.critic, rollout_info, critic_action)
            new_state = BafcState(action=action_state,
                                  actor=actor_state,
                                  critic=critic_state)
            self._critic_update_counter += 1
        else:
            if self._train_mode == TrainMode.actor:
                actor_state, actor_info = self._actor_train_step(
                    inputs.observation, action, rollout_info.action,
                    rollout_info.bootstrap_mask, state.actor)
                critic_info = BafcCriticInfo()
                new_state = BafcState(action=action_state,
                                      actor=actor_state,
                                      critic=state.critic)
                self._actor_update_counter += 1
            else:
                action = action.reshape(-1, action.shape[-1])  # [T*B * n_actor, d_a]
                critic_state, critic_info = self._critic_train_step(
                    inputs.observation, state.critic, rollout_info, action)
                actor_info = LossInfo(extra=BafcActorInfo())
                new_state = BafcState(action=action_state,
                                      actor=state.actor,
                                      critic=critic_state)
                self._critic_update_counter += 1

        if self._debug_summaries and alf.summary.should_record_summaries():
            self._do_critic_summary = True
            safe_mean_hist_summary('eval_samples', self._actor_eval_samples)
            safe_mean_hist_summary('eval_samples/per_dim_mean',
                                   self._actor_eval_samples.mean(dim=0))
            safe_mean_hist_summary(
                'eval_samples/per_dim_std',
                self._actor_eval_samples.std(dim=0, unbiased=False))
            safe_mean_hist_summary('eval_samples/per_sample_l2_norm',
                                   self._actor_eval_samples.norm(dim=-1))

        info = BafcInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            actor=actor_info,
            critic=critic_info,
            discounted_return=rollout_info.discounted_return,
            bootstrap_mask=rollout_info.bootstrap_mask)
        return AlgStep(action, new_state, info)

    def calc_loss(self, info: BafcInfo):
        assert not self._is_eval
        actor_loss = info.actor
        eval_action_loss = actor_loss.extra.eval_action_loss
        if isinstance(eval_action_loss, torch.Tensor):
            eval_action_loss = eval_action_loss.mean()
        if self._train_mode == TrainMode.actor:
            critic_loss = LossInfo()
        else:
            critic_loss = self._calc_critic_loss(info)

        loss = math_ops.add_ignore_empty(actor_loss.loss, critic_loss.loss)

        return LossInfo(
            loss=loss,
            scalar_loss=eval_action_loss,
            extra=BafcLossInfo(
                actor=actor_loss.extra, critic=critic_loss.extra))

    def _calc_critic_loss(self, info: BafcInfo):
        with alf.summary.record_if(lambda: self._do_critic_summary):
            critic_info = info.critic
            critic_losses = []
            for i, l in enumerate(self._critic_losses):
                # critics & target_critics has shape [T, S, n_actor, n_critic]
                critic_loss = l(
                    info=info,
                    value=critic_info.critic[:, :, :, i, ...],
                    target_value=critic_info.target_critic[:, :, :, i, ...]).loss
                if self._use_bootstrap_critics:
                    bootstrap_mask = info.bootstrap_mask[:, :,
                                                         i] / self._bootstrap_mask_prob
                    critic_loss = critic_loss * bootstrap_mask
                critic_losses.append(critic_loss)

        self._do_critic_summary = False
        critic_loss = math_ops.add_n(critic_losses)

        return LossInfo(
            loss=critic_loss,
            extra=critic_loss)

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']

    def after_update(self, root_inputs, info: BafcInfo):
        self._update_train_mode()
        self._update_target_critic()
