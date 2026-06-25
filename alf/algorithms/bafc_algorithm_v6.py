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
from alf.networks.network import Network
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

BafcCriticReweightingInfo = namedtuple(
    "BafcCriticReweightingInfo", [
        "final_weight", "raw_weight", "clipped_weight", "sample_age",
        "fallback_to_uniform", "solver_objective_initial",
        "solver_objective_final"
    ],
    default_value=())

BafcCriticInfo = namedtuple(
    "BafcCriticInfo", [
        "critic", "target_critic", "critic_sample_weight",
        "critic_reweighting_info"
    ],
    default_value=())

BafcActorInfo = namedtuple(
    "BafcActorInfo", ["eval_action_loss"], default_value=())

BafcInfo = namedtuple(
    "BafcInfo", [
        "reward", "step_type", "discount", "action", "actor", "critic", 
        "discounted_return", "bootstrap_mask", "sample_age"
    ],
    default_value=())

BafcLossInfo = namedtuple(
    'BafcLossInfo', ('actor', 'critic'), default_value=())


@alf.configurable
class BafcAlgorithmV6(OffPolicyAlgorithm):
    r"""Boostrapped Actor and Functional Critic algorithm, 

    ::

        Bai et al "Bootstrapped Actors and Functional Critic", arXiv, 2025

    V6 implements model-free posterior sampling style exploration scheme over V2.
    In particular, it has multiple functional critics, each paired with an actor.
    Each functional critic is trained with all actors, while each actor is only
    trained with its paired functional critic.

    V6 adds critic sample reweighting based on snapshot critic features. It does
    not include the trust-metric rollout or actor-extension gates.

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
                 enable_critic_reweighting: bool = False,
                 critic_reweighting_beta: Optional[float] = None,
                 critic_reweighting_ridge: float = 1e-4,
                 critic_reweighting_solver: str = "lbfgs_logits",
                 critic_reweighting_solver_iters: int = 5,
                 critic_reweighting_max_weight: float = 10.0,
                 critic_reweighting_num_feature_coords: int = 64,
                 critic_reweighting_num_target_obs: int = 128,
                 critic_reweighting_target_obs_cache_size:
                 Optional[int] = None,
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
                 actor_optimizer=None,
                 critic_optimizer=None,
                 actor_encoder_optimizer=None,
                 eval_samples_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="BafcAlgorithmV6"):
        """
        Args:

            actor_critic_pairing (bool): whether or not fix the 1-1 pairing of actors 
                and critics during actor_train_step (there are the same number of 
                actors and critics, we pair each actor with a unique and different 
                critic during actor training). If True, such a actor-critic pairing is 
                fixed throughout the training. Otherwise, it is randomized at each
                actor_train_step.
            bootstrap_mask_type (str): the type of sampling the bootstrap_mask for
                bootstrapped training of actors and/or critics. There are two types, 
                ``episode`` and ``step``. ``episode`` means a same bootstrap_mask for
                every step of an episode. ``step`` means resampled bootstrap_mask for
                every step of an episode.
            enable_critic_reweighting (bool): If True, compute per-sample critic
                loss weights from snapshot critic features.
            critic_reweighting_beta (None|float): Eq. C.10 coverage trade-off.
                When None, use ``1 / (N * (1 - gamma))`` for each batch.
            critic_reweighting_ridge (float): positive ridge added to feature
                covariance matrices for the reweighting solver.
            critic_reweighting_solver (str): solver used for the simplex
                distribution in Eq. C.10. Supported values are
                ``lbfgs_logits`` and ``projected_gradient_fw``.
        """
        assert actor_eval_type in ['full', 'exclude_input', 'last_two', 'output'], (
            r"{actor_eval_type} in not supported.")
        assert eval_samples_init_method in ['normal', 'uniform'], (
            r"init method {eval_samples_init_method} is not supported.")
        assert bootstrap_mask_type in ['episode', 'step'], (
            r"bootstrap mask type {bootstrap_mask_type} is not supported.")
        if critic_reweighting_beta is not None:
            assert critic_reweighting_beta >= 0, (
                "critic_reweighting_beta must be nonnegative when set")
        assert critic_reweighting_ridge > 0, (
            "critic_reweighting_ridge must be > 0")
        assert critic_reweighting_solver in (
            "lbfgs_logits", "projected_gradient_fw"), (
                "critic_reweighting_solver must be one of "
                "['lbfgs_logits', 'projected_gradient_fw']")
        assert critic_reweighting_solver_iters >= 1, (
            "critic_reweighting_solver_iters must be >= 1")
        assert critic_reweighting_max_weight > 0, (
            "critic_reweighting_max_weight must be > 0")
        assert critic_reweighting_num_feature_coords >= 1, (
            "critic_reweighting_num_feature_coords must be >= 1")
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
        self._use_bootstrap_actors = use_bootstrap_actors
        self._use_bootstrap_critics = use_bootstrap_critics
        self._bootstrap_mask_prob = bootstrap_mask_prob
        self._bootstrap_mask_type = bootstrap_mask_type
        self._checkpoint_replay_buffer = checkpoint_replay_buffer
        self._enable_critic_reweighting = enable_critic_reweighting
        self._critic_reweighting_beta = critic_reweighting_beta
        self._critic_reweighting_ridge = critic_reweighting_ridge
        self._critic_reweighting_solver = critic_reweighting_solver
        self._critic_reweighting_solver_iters = critic_reweighting_solver_iters
        self._critic_reweighting_max_weight = critic_reweighting_max_weight
        self._critic_reweighting_num_feature_coords = (
            critic_reweighting_num_feature_coords)
        self._critic_reweighting_num_target_obs = (
            critic_reweighting_num_target_obs)
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
        self._reference_actor_networks = actor_networks.copy(
            name='reference_actor_networks')
        for p in self._reference_actor_networks.parameters():
            p.requires_grad_(False)
        self._actor_use_ln = actor_use_ln
        self._actor_eval_type = actor_eval_type
        self._actor_encoder = actor_encoder
        self._critic_networks = critic_networks
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')
        self._snapshot_critic_networks = critic_networks.copy(
            name='snapshot_critic_networks')
        for p in self._snapshot_critic_networks.parameters():
            p.requires_grad_(False)
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
        self._last_critic_reweighting_info = ()

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
        self._sync_reference_actor_from_current()
        self._sync_snapshot_critic_from_current()

    def _sync_reference_actor_from_current(self):
        self._reference_actor_networks.load_state_dict(
            self._actor_networks.state_dict())

    def _sync_snapshot_critic_from_current(self):
        self._snapshot_critic_networks.load_state_dict(
            self._critic_networks.state_dict())

    def _bafc_runtime_key(self, prefix, name):
        return prefix + "_bafc_runtime." + name

    def _bafc_scalar_tensor(self, value, dtype=torch.int64):
        if isinstance(value, torch.Tensor):
            return value.detach().reshape(()).to(dtype=dtype).clone()
        return torch.tensor(value, dtype=dtype)

    def _bafc_runtime_tensor(self, value):
        return torch.as_tensor(value).detach().clone()

    def _bafc_scalar_int(self, value):
        return int(torch.as_tensor(value).reshape(()).item())

    def _save_bafc_runtime_state(self, destination, prefix):
        scalar_fields = dict(
            training_started=(self._training_started, torch.bool),
            train_mode=(self._train_mode.value, torch.int64),
            rollout_actor_id=(self._rollout_actor_id, torch.int64),
            actor_update_counter=(self._actor_update_counter, torch.int64),
            critic_update_counter=(self._critic_update_counter, torch.int64))
        for name, (value, dtype) in scalar_fields.items():
            destination[self._bafc_runtime_key(
                prefix, name)] = self._bafc_scalar_tensor(value, dtype=dtype)
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

    def _copy_state_prefix_if_missing(self, state_dict, prefix, source, target):
        source_prefix = prefix + source + "."
        target_prefix = prefix + target + "."
        if any(key.startswith(target_prefix) for key in state_dict.keys()):
            return
        for key, value in list(state_dict.items()):
            if key.startswith(source_prefix):
                state_dict[target_prefix + key[len(source_prefix):]] = value

    def _synthesize_v3_missing_network_state(self, state_dict, prefix):
        self._copy_state_prefix_if_missing(
            state_dict, prefix, "_actor_networks", "_reference_actor_networks")
        self._copy_state_prefix_if_missing(
            state_dict, prefix, "_critic_networks", "_snapshot_critic_networks")

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
        self._synthesize_v3_missing_network_state(state_dict, prefix)
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

    def _flatten_reweighting_actions(self, action):
        if not isinstance(action, torch.Tensor):
            return ()
        action_dim = len(self._action_spec.shape)
        action = action.reshape(-1, *action.shape[-action_dim:])
        if action.shape[0] == 0:
            return ()
        return action

    def _sample_reweighting_observations(self, observation):
        obs = self._flatten_reweighting_observations(observation)
        if not isinstance(obs, torch.Tensor):
            return ()
        if obs.shape[0] > self._critic_reweighting_num_target_obs:
            idx = torch.randperm(obs.shape[0], device=obs.device)[
                :self._critic_reweighting_num_target_obs]
            obs = obs[idx]
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

    def _sample_reweighting_target_observations(self, fallback_observation):
        cache = self._reweighting_target_observation_cache
        obs = cache if isinstance(cache, torch.Tensor) else fallback_observation
        return self._sample_reweighting_observations(obs)

    def _ensure_group_action(self, action):
        if not isinstance(action, torch.Tensor):
            return action
        if action.ndim == 2:
            return action.unsqueeze(1)
        return action

    def _extract_eval_action(self, actor_network):
        eval_action = actor_network(
            self._actor_eval_samples,
            full_neurons=self._actor_eval_type != 'output')[0]
        if self._actor_eval_type == 'exclude_input':
            eval_action = eval_action[1:]
        elif self._actor_eval_type == 'last_two':
            eval_action = eval_action[-2:]
        return eval_action

    def _compute_actor_encoding(self, actor_network):
        eval_action = self._extract_eval_action(actor_network)
        actor_tokens = self._tokenize_actor_out(eval_action)
        return self._actor_encoder(actor_tokens)[0]

    def _sample_reweighting_feature_coords(self, feature_dim, device):
        if feature_dim <= self._critic_reweighting_num_feature_coords:
            return torch.arange(feature_dim, device=device)
        return torch.randperm(feature_dim, device=device)[
            :self._critic_reweighting_num_feature_coords]

    def _critic_feature_head_index(self, critic_network):
        critic_core = getattr(critic_network, '_pnet', critic_network)
        modules = getattr(critic_core, '_networks', None)
        if modules is None:
            raise RuntimeError(
                "Snapshot critic does not expose the sequential modules needed "
                "for reweighting feature extraction.")

        head_indices = []
        for idx, module in enumerate(modules):
            output_size = getattr(module, '_output_size', None)
            if output_size is not None and int(output_size) == 1:
                head_indices.append(idx)
                continue
            weight = getattr(module, 'weight', None)
            if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
                if weight.shape[-2] == 1:
                    head_indices.append(idx)
        if not head_indices:
            raise RuntimeError(
                "Failed to locate the snapshot critic's final scalar head.")
        return head_indices[-1]

    def _compute_snapshot_feature_map(self,
                                      observation,
                                      actor_encoding,
                                      action,
                                      critic_network=None):
        """Return normalized snapshot critic features before the scalar head."""
        critic_network = (self._snapshot_critic_networks
                          if critic_network is None else critic_network)
        critic_core = getattr(critic_network, '_pnet', critic_network)
        modules = critic_core._networks
        head_idx = self._critic_feature_head_index(critic_network)

        action = self._ensure_group_action(action)
        if (isinstance(action, torch.Tensor) and action.ndim == 3
                and action.shape[1] == 1
                and self._num_actor_critic != 1):
            action = action.expand(-1, self._num_actor_critic, -1)
        if actor_encoding.ndim == 2:
            actor_encoding = actor_encoding.unsqueeze(0).expand(
                observation.shape[0], -1, -1)
        critic_observation = observation.unsqueeze(1).expand(
            -1, self._num_actor_critic, -1)
        x = (actor_encoding, (critic_observation, action))

        for idx, module in enumerate(modules):
            if idx == head_idx:
                if not isinstance(x, torch.Tensor) or x.ndim != 3:
                    raise RuntimeError(
                        "Unexpected snapshot critic feature-map shape %s" %
                        (type(x), ))
                norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                return x / norm
            if isinstance(module, Network):
                x = module(x)[0]
            else:
                x = module(x)

        raise RuntimeError(
            "Failed to extract snapshot critic feature map before the scalar "
            "head.")

    def _feature_covariance(self, feature_map):
        return (feature_map.permute(1, 2, 0) @
                feature_map.permute(1, 0, 2) / feature_map.shape[0])

    @torch.no_grad()
    def _compute_reweighting_feature_maps(self, replay_obs, replay_action):
        if not isinstance(replay_obs, torch.Tensor) or not isinstance(
                replay_action, torch.Tensor):
            return (), ()

        target_obs = self._sample_reweighting_target_observations(replay_obs)
        if not isinstance(target_obs, torch.Tensor):
            target_obs = replay_obs
        if target_obs.device != replay_obs.device:
            target_obs = target_obs.to(replay_obs.device)
        if replay_action.device != replay_obs.device:
            replay_action = replay_action.to(replay_obs.device)

        reference_encoding = self._compute_actor_encoding(
            self._reference_actor_networks).detach()
        target_action = self._ensure_group_action(
            self._reference_actor_networks(target_obs)[0]).detach()
        behavior_action = self._ensure_group_action(replay_action).detach()

        phi_target = self._compute_snapshot_feature_map(
            target_obs, reference_encoding, target_action).detach()
        phi_behavior = self._compute_snapshot_feature_map(
            replay_obs, reference_encoding, behavior_action).detach()
        return phi_target, phi_behavior

    def _project_simplex(self, values):
        values = torch.as_tensor(values)
        if values.ndim != 1 or values.numel() == 0:
            return values
        sorted_values = torch.sort(values, descending=True).values
        cssv = torch.cumsum(sorted_values, dim=0) - 1.0
        ind = torch.arange(
            1, values.numel() + 1, device=values.device, dtype=values.dtype)
        active = sorted_values - cssv / ind > 0
        if not active.any():
            return torch.full_like(values, 1.0 / values.numel())
        rho = torch.nonzero(active, as_tuple=False)[-1, 0]
        theta = cssv[rho] / ind[rho]
        projected = torch.clamp(values - theta, min=0.)
        total = projected.sum()
        if not torch.isfinite(total) or total <= 0:
            return torch.full_like(values, 1.0 / values.numel())
        return projected / total

    def _critic_reweighting_beta_value(self, num_samples, device, dtype):
        if self._critic_reweighting_beta is not None:
            return torch.as_tensor(
                self._critic_reweighting_beta, device=device, dtype=dtype)
        gamma = torch.as_tensor(
            self._critic_losses[0].gamma, device=device, dtype=dtype).reshape(-1)[0]
        denom = (1.0 - gamma).clamp_min(1e-6)
        return 1.0 / (float(num_samples) * denom)

    def _critic_reweighting_objective(self, p, features, target_cov, beta,
                                      ridge):
        num_features = features.shape[-1]
        eye = torch.eye(
            num_features, dtype=features.dtype, device=features.device).unsqueeze(0)
        sigma = torch.einsum('n,ngd,nge->gde', p, features, features)
        sigma = sigma + ridge * eye
        inv_sigma = torch.linalg.pinv(sigma)
        m_mat = torch.einsum('n,ngd,nge->gde', p.pow(2), features,
                             features)

        h_obj = torch.einsum('gij,gji->g', target_cov, inv_sigma)
        g_obj = torch.einsum('gij,gjk,gkl,gli->g', target_cov,
                             inv_sigma, m_mat, inv_sigma)
        return (g_obj + beta * h_obj).mean()

    def _critic_reweighting_objective_and_grad(self, p, features, target_cov,
                                               beta, ridge):
        num_features = features.shape[-1]
        eye = torch.eye(
            num_features, dtype=features.dtype, device=features.device).unsqueeze(0)
        sigma = torch.einsum('n,ngd,nge->gde', p, features, features)
        sigma = sigma + ridge * eye
        inv_sigma = torch.linalg.pinv(sigma)
        m_mat = torch.einsum('n,ngd,nge->gde', p.pow(2), features, features)

        h_obj = torch.einsum('gij,gji->g', target_cov, inv_sigma)
        g_obj = torch.einsum('gij,gjk,gkl,gli->g', target_cov, inv_sigma,
                             m_mat, inv_sigma)
        objective = (g_obj + beta * h_obj).mean()

        c_ell = inv_sigma @ target_cov @ inv_sigma
        c_h = c_ell @ m_mat @ inv_sigma
        ell = torch.einsum('ngd,gde,nge->ng', features, c_ell, features)
        h = torch.einsum('ngd,gde,nge->ng', features, c_h, features)
        grad = ((2.0 * p).unsqueeze(-1) - beta) * ell - 2.0 * h
        grad = grad.mean(dim=1)
        fw_gap = -beta * h_obj.mean() - grad.min()
        return objective, grad, fw_gap

    def _solve_critic_reweighting_distribution_lbfgs_logits(
            self, features, target_cov, beta, ridge):
        num_samples = features.shape[0]
        if num_samples <= 1:
            return torch.full(
                (num_samples, ),
                1.0 / max(1, num_samples),
                dtype=features.dtype,
                device=features.device)

        features = features.detach()
        target_cov = target_cov.detach()
        beta = torch.as_tensor(beta, device=features.device, dtype=features.dtype)
        ridge = torch.as_tensor(
            ridge, device=features.device, dtype=features.dtype)
        logits = torch.zeros(
            num_samples, dtype=features.dtype, device=features.device,
            requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [logits],
            lr=1.0,
            max_iter=self._critic_reweighting_solver_iters,
            max_eval=2 * self._critic_reweighting_solver_iters,
            history_size=10,
            line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad(set_to_none=True)
            p = torch.softmax(logits, dim=0)
            objective = self._critic_reweighting_objective(
                p, features, target_cov, beta, ridge)
            if not torch.isfinite(objective):
                raise RuntimeError(
                    "non-finite critic reweighting LBFGS objective")
            objective.backward()
            return objective

        with torch.enable_grad():
            optimizer.step(closure)

        p = torch.softmax(logits.detach(), dim=0)
        if not torch.isfinite(p).all():
            raise RuntimeError("non-finite critic reweighting distribution")
        return p

    def _solve_critic_reweighting_distribution_projected_gradient_fw(
            self, features, target_cov, beta, ridge):
        num_samples = features.shape[0]
        if num_samples <= 1:
            return torch.full(
                (num_samples, ),
                1.0 / max(1, num_samples),
                dtype=features.dtype,
                device=features.device)

        uniform = torch.full(
            (num_samples, ),
            1.0 / num_samples,
            dtype=features.dtype,
            device=features.device)
        starts = [uniform]
        try:
            _, uniform_grad, _ = self._critic_reweighting_objective_and_grad(
                uniform, features, target_cov, beta, ridge)
            vertex = torch.zeros_like(uniform)
            vertex[torch.argmin(uniform_grad)] = 1.0
            starts.append(0.5 * uniform + 0.5 * vertex)
        except RuntimeError:
            starts.append(uniform)

        best_p = uniform
        best_obj = None
        for start in starts:
            p = self._project_simplex(start)
            try:
                obj, grad, fw_gap = self._critic_reweighting_objective_and_grad(
                    p, features, target_cov, beta, ridge)
                if not torch.isfinite(obj):
                    continue
                for _ in range(self._critic_reweighting_solver_iters):
                    candidates = []
                    step = 1.0
                    for _ in range(8):
                        cand = self._project_simplex(p - step * grad)
                        cand_obj, _, _ = self._critic_reweighting_objective_and_grad(
                            cand, features, target_cov, beta, ridge)
                        if torch.isfinite(cand_obj) and cand_obj <= obj + 1e-12:
                            candidates.append((cand_obj, cand))
                            break
                        step *= 0.5

                    if torch.isfinite(fw_gap) and fw_gap > 1e-8:
                        vertex = torch.zeros_like(p)
                        vertex[torch.argmin(grad)] = 1.0
                        direction = vertex - p
                        alpha = 1.0
                        for _ in range(8):
                            cand = self._project_simplex(p + alpha * direction)
                            cand_obj, _, _ = self._critic_reweighting_objective_and_grad(
                                cand, features, target_cov, beta, ridge)
                            if torch.isfinite(cand_obj) and cand_obj <= obj + 1e-12:
                                candidates.append((cand_obj, cand))
                                break
                            alpha *= 0.5

                    if not candidates:
                        break
                    obj, p = min(candidates, key=lambda item: float(item[0]))
                    obj, grad, fw_gap = self._critic_reweighting_objective_and_grad(
                        p, features, target_cov, beta, ridge)

                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_p = p
            except RuntimeError:
                continue

        if best_obj is None or not torch.isfinite(best_p).all():
            return uniform
        return self._project_simplex(best_p)

    def _solve_critic_reweighting_distribution(self, features, target_cov, beta,
                                               ridge):
        if self._critic_reweighting_solver == "lbfgs_logits":
            return self._solve_critic_reweighting_distribution_lbfgs_logits(
                features, target_cov, beta, ridge)
        return (
            self._solve_critic_reweighting_distribution_projected_gradient_fw(
                features, target_cov, beta, ridge))

    def _record_critic_reweighting_summaries(self,
                                             final_weights,
                                             raw_weights=None,
                                             clipped_weights=None,
                                             sample_age=(),
                                             fallback_to_uniform=False,
                                             solver_objective_initial=None,
                                             solver_objective_final=None):
        if not (self._debug_summaries and self._enable_critic_reweighting
                and alf.summary.should_record_summaries()):
            return
        if not isinstance(final_weights, torch.Tensor) or final_weights.numel() == 0:
            return

        final_weights = final_weights.detach().reshape(-1)
        if raw_weights is None:
            raw_weights = final_weights
        else:
            raw_weights = raw_weights.detach().reshape(-1)
        if clipped_weights is None:
            clipped_weights = final_weights
        else:
            clipped_weights = clipped_weights.detach().reshape(-1)

        safe_mean_hist_summary('critic_reweighting/raw_weight', raw_weights)
        safe_mean_hist_summary('critic_reweighting/clipped_weight',
                               clipped_weights)
        safe_mean_hist_summary('critic_reweighting/final_weight',
                               final_weights)

        max_weight = torch.as_tensor(
            self._critic_reweighting_max_weight,
            dtype=raw_weights.dtype,
            device=raw_weights.device)
        clipped_mask = raw_weights >= max_weight
        num_clipped = clipped_mask.to(torch.float32).sum()
        num_samples = torch.as_tensor(
            raw_weights.numel(), dtype=final_weights.dtype,
            device=final_weights.device)
        ess = final_weights.sum().pow(2) / final_weights.pow(2).sum().clamp_min(
            1e-12)

        alf.summary.scalar('critic_reweighting/raw_weight_max',
                           raw_weights.max())
        alf.summary.scalar('critic_reweighting/final_weight_max',
                           final_weights.max())
        alf.summary.scalar('critic_reweighting/num_clipped_at_max',
                           num_clipped)
        alf.summary.scalar('critic_reweighting/frac_clipped_at_max',
                           num_clipped / num_samples.clamp_min(1.0))
        alf.summary.scalar('critic_reweighting/ess', ess)
        alf.summary.scalar('critic_reweighting/ess_ratio',
                           ess / num_samples.clamp_min(1.0))
        alf.summary.scalar('critic_reweighting/num_samples', num_samples)
        alf.summary.scalar('critic_reweighting/fallback_to_uniform',
                           float(fallback_to_uniform))
        if solver_objective_initial is not None and solver_objective_final is not None:
            solver_objective_initial = torch.as_tensor(
                solver_objective_initial,
                dtype=final_weights.dtype,
                device=final_weights.device)
            solver_objective_final = torch.as_tensor(
                solver_objective_final,
                dtype=final_weights.dtype,
                device=final_weights.device)
            if (solver_objective_initial.numel() > 0
                    and solver_objective_final.numel() > 0
                    and torch.isfinite(solver_objective_initial).all().item()
                    and torch.isfinite(solver_objective_final).all().item()):
                solver_objective_initial = solver_objective_initial.reshape(())
                solver_objective_final = solver_objective_final.reshape(())
                alf.summary.scalar(
                    'critic_reweighting/solver_objective_initial',
                    solver_objective_initial)
                alf.summary.scalar(
                    'critic_reweighting/solver_objective_final',
                    solver_objective_final)
                alf.summary.scalar(
                    'critic_reweighting/solver_objective_improvement',
                    solver_objective_initial - solver_objective_final)

        if isinstance(sample_age, torch.Tensor):
            sample_age = sample_age.detach().reshape(-1).to(
                device=final_weights.device, dtype=final_weights.dtype)
            if sample_age.numel() == final_weights.numel():
                safe_mean_hist_summary('critic_reweighting/sample_age',
                                       sample_age)
                recency = -sample_age

                def _corr(x, y):
                    if x.numel() <= 1:
                        return torch.zeros((), device=y.device, dtype=y.dtype)
                    x = x - x.mean()
                    y = y - y.mean()
                    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
                    return (x * y).sum() / denom.clamp_min(1e-12)

                final_age_corr = _corr(sample_age, final_weights)
                final_recency_corr = _corr(recency, final_weights)
                raw_recency_corr = _corr(recency, raw_weights)
                sorted_age_indices = torch.argsort(sample_age)
                quartile_size = max(1, sample_age.numel() // 4)
                newest_indices = sorted_age_indices[:quartile_size]
                oldest_indices = sorted_age_indices[-quartile_size:]
                newest_mean = final_weights[newest_indices].mean()
                oldest_mean = final_weights[oldest_indices].mean()

                alf.summary.scalar('critic_reweighting/sample_age_min',
                                   sample_age.min())
                alf.summary.scalar('critic_reweighting/sample_age_max',
                                   sample_age.max())
                alf.summary.scalar('critic_reweighting/final_weight_age_corr',
                                   final_age_corr)
                alf.summary.scalar(
                    'critic_reweighting/final_weight_recency_corr',
                    final_recency_corr)
                alf.summary.scalar('critic_reweighting/raw_weight_recency_corr',
                                   raw_recency_corr)
                alf.summary.scalar(
                    'critic_reweighting/newest_quartile_final_weight_mean',
                    newest_mean)
                alf.summary.scalar(
                    'critic_reweighting/oldest_quartile_final_weight_mean',
                    oldest_mean)
                alf.summary.scalar(
                    'critic_reweighting/newest_over_oldest_weight_ratio',
                    newest_mean / oldest_mean.clamp_min(1e-12))

    def _make_critic_reweighting_info(self,
                                      final_weight,
                                      raw_weight=None,
                                      clipped_weight=None,
                                      sample_age=(),
                                      fallback_to_uniform=False,
                                      solver_objective_initial=(),
                                      solver_objective_final=()):
        if raw_weight is None:
            raw_weight = final_weight
        if clipped_weight is None:
            clipped_weight = final_weight
        if isinstance(final_weight, torch.Tensor):
            fallback_to_uniform = torch.as_tensor(
                float(fallback_to_uniform),
                dtype=final_weight.dtype,
                device=final_weight.device)
        return BafcCriticReweightingInfo(
            final_weight=final_weight,
            raw_weight=raw_weight,
            clipped_weight=clipped_weight,
            sample_age=sample_age,
            fallback_to_uniform=fallback_to_uniform,
            solver_objective_initial=solver_objective_initial,
            solver_objective_final=solver_objective_final)

    def _record_critic_reweighting_info_summaries(self, reweighting_info):
        if not isinstance(reweighting_info, BafcCriticReweightingInfo):
            return
        self._record_critic_reweighting_summaries(
            reweighting_info.final_weight,
            raw_weights=reweighting_info.raw_weight,
            clipped_weights=reweighting_info.clipped_weight,
            sample_age=reweighting_info.sample_age,
            fallback_to_uniform=reweighting_info.fallback_to_uniform,
            solver_objective_initial=reweighting_info.solver_objective_initial,
            solver_objective_final=reweighting_info.solver_objective_final)

    def _sanitize_critic_reweighting_info_for_train_info(self, reweighting_info):
        del reweighting_info
        return ()

    def _sanitize_critic_info_for_train_info(self, critic_info):
        if not isinstance(critic_info, BafcCriticInfo):
            return critic_info
        reweighting_info = self._sanitize_critic_reweighting_info_for_train_info(
            critic_info.critic_reweighting_info)
        if reweighting_info == critic_info.critic_reweighting_info:
            return critic_info
        return critic_info._replace(critic_reweighting_info=reweighting_info)

    def _empty_actor_info_for_train_info(self, reward):
        if not isinstance(reward, torch.Tensor):
            return LossInfo(extra=BafcActorInfo())
        zero = torch.zeros_like(reward)
        return LossInfo(
            loss=zero,
            extra=BafcActorInfo(eval_action_loss=torch.zeros_like(reward)))

    def _empty_critic_info_for_train_info(self, reward):
        if not isinstance(reward, torch.Tensor):
            return BafcCriticInfo()
        critic_shape = reward.shape + (
            self._num_actor_critic,
            self._num_actor_critic,
        )
        critic = reward.new_zeros(critic_shape)
        critic_sample_weight = ()
        if self._enable_critic_reweighting:
            critic_sample_weight = torch.ones_like(reward)
        return BafcCriticInfo(
            critic=critic,
            target_critic=torch.zeros_like(critic),
            critic_sample_weight=critic_sample_weight,
            critic_reweighting_info=())

    @torch.no_grad()
    def _compute_critic_sample_weights(self, observation, action, sample_age=()):
        if not self._enable_critic_reweighting:
            return (), ()
        if not isinstance(observation, torch.Tensor):
            return (), ()
        outer_shape = observation.shape[:-len(self._observation_spec.shape)]
        if not outer_shape:
            outer_shape = (observation.shape[0], )
        obs = self._flatten_reweighting_observations(observation)
        act = self._flatten_reweighting_actions(action)
        ones = torch.ones(
            outer_shape, dtype=observation.dtype, device=observation.device)
        fallback_info = self._make_critic_reweighting_info(
            ones, sample_age=sample_age, fallback_to_uniform=True)
        if not isinstance(obs, torch.Tensor) or not isinstance(act, torch.Tensor):
            return ones, fallback_info
        num_samples = obs.shape[0]
        if num_samples == 0 or act.shape[0] != num_samples:
            return ones, fallback_info

        try:
            phi_target, phi_behavior = self._compute_reweighting_feature_maps(
                obs, act)
            if not isinstance(phi_target, torch.Tensor) or not isinstance(
                    phi_behavior, torch.Tensor):
                raise RuntimeError("empty critic reweighting feature map")
            feature_dim = phi_behavior.shape[-1]
            feature_coords = self._sample_reweighting_feature_coords(
                feature_dim, phi_behavior.device)
            phi_behavior = phi_behavior[:, :, feature_coords]
            phi_target = phi_target[:, :, feature_coords]
            target_cov = self._feature_covariance(phi_target)
            beta = self._critic_reweighting_beta_value(
                num_samples, phi_behavior.device, phi_behavior.dtype)
            ridge = torch.as_tensor(
                self._critic_reweighting_ridge,
                device=phi_behavior.device,
                dtype=phi_behavior.dtype)
            uniform = torch.full(
                (num_samples, ),
                1.0 / num_samples,
                dtype=phi_behavior.dtype,
                device=phi_behavior.device)
            solver_objective_initial = self._critic_reweighting_objective(
                uniform, phi_behavior, target_cov, beta, ridge).detach()
            if not torch.isfinite(solver_objective_initial):
                raise RuntimeError(
                    "non-finite initial critic reweighting objective")
            p = self._solve_critic_reweighting_distribution(
                phi_behavior, target_cov, beta, ridge)
            solver_objective_final = self._critic_reweighting_objective(
                p, phi_behavior, target_cov, beta, ridge).detach()
            if not torch.isfinite(solver_objective_final):
                raise RuntimeError(
                    "non-finite final critic reweighting objective")
            raw_weights = p * float(num_samples)
            clipped_weights = raw_weights.clamp(
                min=0., max=float(self._critic_reweighting_max_weight))
            mean = clipped_weights.mean().clamp_min(1e-12)
            weights = clipped_weights / mean
            if not torch.isfinite(weights).all():
                raise RuntimeError("non-finite critic reweighting weights")
            weights = weights.to(
                device=observation.device,
                dtype=observation.dtype).reshape(outer_shape)
            reweighting_info = self._make_critic_reweighting_info(
                weights,
                raw_weight=raw_weights,
                clipped_weight=clipped_weights,
                sample_age=sample_age,
                fallback_to_uniform=False,
                solver_objective_initial=solver_objective_initial,
                solver_objective_final=solver_objective_final)
            return weights, reweighting_info
        except RuntimeError:
            return ones, fallback_info

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        def _is_empty(value):
            return isinstance(value, tuple) and len(value) == 0

        if _is_empty(batch_info) or _is_empty(rollout_info):
            return root_inputs, rollout_info
        if not hasattr(rollout_info, 'sample_age'):
            return root_inputs, rollout_info

        replay_buffer = getattr(batch_info, 'replay_buffer', ())
        env_ids = getattr(batch_info, 'env_ids', ())
        positions = getattr(batch_info, 'positions', ())
        if _is_empty(replay_buffer) or _is_empty(env_ids) or _is_empty(positions):
            return root_inputs, rollout_info
        if not hasattr(replay_buffer, '_current_pos') or not hasattr(
                replay_buffer, 'device'):
            return root_inputs, rollout_info
        if not isinstance(root_inputs.step_type, torch.Tensor):
            return root_inputs, rollout_info
        if root_inputs.step_type.ndim < 2:
            return root_inputs, rollout_info

        mini_batch_length = root_inputs.step_type.shape[1]
        replay_device = replay_buffer.device
        env_ids = env_ids.to(device=replay_device)
        positions = positions.to(device=replay_device)
        current_pos = replay_buffer._current_pos.to(device=replay_device)[env_ids]
        sample_positions = positions.unsqueeze(-1) + torch.arange(
            mini_batch_length, device=replay_device, dtype=positions.dtype)
        sample_age = current_pos.unsqueeze(-1) - sample_positions - 1
        sample_age = sample_age.clamp_min(0)
        sample_age = sample_age.to(
            device=root_inputs.step_type.device, dtype=torch.float32)
        rollout_info = rollout_info._replace(sample_age=sample_age)
        return root_inputs, rollout_info

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
        if self._enable_critic_reweighting:
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

    def _actor_train_step(self, observation, action, mask, state):
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
        if not self._actor_critic_pairing:
            perm = torch.randperm(self._num_actor_critic)
            actor_encoding = actor_encoding[perm, :]
            action = action[:, perm, :]
        actor_encoding = actor_encoding.unsqueeze(0).repeat(
            observation.shape[0], 1, 1)  # [T*B, n_actor, d_enc]

        ## Step 2: compute critic values for all actors
        ###############################################
        # # [T*B * n_actor, d_s]
        # critic_observation = observation.repeat_interleave(
        #     self._num_actor_critic, dim=0)
        # [T*B, n_critic, d_s]
        critic_observation = observation.unsqueeze(1).repeat(
            1, self._num_actor_critic, 1)
        # [T*B, n_critic]
        q_value, critic_state = self._critic_networks(
            (actor_encoding, (critic_observation, action)), state) 

        ## Step 3: exact off-policy policy gradient (OPG)
        #################################################
        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum(), retain_graph=True)
        # need to exclude the input actor_eval_samples, since they don't requires_grad
        # for actor TrainMode
        if self._actor_eval_type == 'full':
            eval_action_in_graph = eval_action[1:]
        else:
            eval_action_in_graph = eval_action
        dqde = nest_utils.grad(eval_action_in_graph, q_value.sum(), 
                               retain_graph=self._actor_eval_type != 'output')

        def action_loss_fn(dqda, a_in):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + a_in).detach(), a_in)
            return loss.sum(list(range(2, loss.ndim)))

        # 1st term of OPG: loss corresponding to input action
        action_loss = nest.map_structure(action_loss_fn, dqda, action)
        if self._use_bootstrap_actors:
            action_loss = action_loss * mask / self._bootstrap_mask_prob
        action_loss = action_loss.sum(-1)

        # 2nd term of OPG: loss corresponding to input eval_action
        eval_action_loss = nest.map_structure(
            action_loss_fn, dqde, eval_action_in_graph)
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
        critic_sample_weight, critic_reweighting_info = (
            self._compute_critic_sample_weights(
                observation, rollout_info.action, rollout_info.sample_age))
        info = BafcCriticInfo(
            critic=critics,
            target_critic=target_critics,
            critic_sample_weight=critic_sample_weight,
            critic_reweighting_info=critic_reweighting_info)

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
                inputs.observation, actor_action, 
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
                    inputs.observation, action, 
                    rollout_info.bootstrap_mask, state.actor)
                critic_info = self._empty_critic_info_for_train_info(
                    inputs.reward)
                new_state = BafcState(action=action_state,
                                      actor=actor_state,
                                      critic=state.critic)
                self._actor_update_counter += 1
            else:
                action = action.reshape(-1, action.shape[-1])  # [T*B * n_actor, d_a]
                critic_state, critic_info = self._critic_train_step(
                    inputs.observation, state.critic, rollout_info, action)
                actor_info = self._empty_actor_info_for_train_info(
                    inputs.reward)
                new_state = BafcState(action=action_state,
                                      actor=state.actor,
                                      critic=critic_state)
                self._critic_update_counter += 1

        reweighting_info_for_summary = critic_info.critic_reweighting_info
        if isinstance(reweighting_info_for_summary, BafcCriticReweightingInfo):
            self._last_critic_reweighting_info = reweighting_info_for_summary
        else:
            reweighting_info_for_summary = self._last_critic_reweighting_info

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
            self._record_critic_reweighting_info_summaries(
                reweighting_info_for_summary)

        critic_info = self._sanitize_critic_info_for_train_info(critic_info)

        info = BafcInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            actor=actor_info,
            critic=critic_info,
            discounted_return=rollout_info.discounted_return,
            bootstrap_mask=rollout_info.bootstrap_mask,
            sample_age=rollout_info.sample_age)
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
                sample_weight = critic_info.critic_sample_weight
                if isinstance(sample_weight, torch.Tensor):
                    while sample_weight.ndim < critic_loss.ndim:
                        sample_weight = sample_weight.unsqueeze(-1)
                    critic_loss = critic_loss * sample_weight
                critic_losses.append(critic_loss)

        self._do_critic_summary = False
        critic_loss = math_ops.add_n(critic_losses)

        return LossInfo(
            loss=critic_loss,
            extra=critic_loss)

    def _trainable_attributes_to_ignore(self):
        return [
            '_target_critic_networks', '_reference_actor_networks',
            '_snapshot_critic_networks'
        ]

    def after_update(self, root_inputs, info: BafcInfo):
        self._update_train_mode()
        self._update_target_critic()
        self._sync_reference_actor_from_current()
        self._sync_snapshot_critic_from_current()
