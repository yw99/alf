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
from alf.utils import losses, common, dist_utils, math_ops
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.schedulers import Scheduler
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.networks.network import Network
from alf.networks.neural_graphs.actor_graph import ActorGraph
from alf.networks.neural_graphs.graph_network import GraphNetwork

BafcActionState = namedtuple(
    "BafcActionState", ["actor_network"], default_value=())

BafcCriticState = namedtuple("BafcCriticState", ["critic", "target_critic"])

BafcState = namedtuple(
    "BafcState", ["action", "actor", "critic"],
    default_value=())

BafcCriticInfo = namedtuple(
    "BafcCriticInfo", ["critic", "target_critic", "eval_trust_metric"],
    default_value=())

BafcActorInfo = namedtuple(
    "BafcActorInfo", ["eval_action_loss", "grad_trust_metric"],
    default_value=())

BafcInfo = namedtuple(
    "BafcInfo", [
        "reward", "step_type", "discount", "action", "actor", "critic", 
        "discounted_return", "bootstrap_mask", "eval_trust_metric",
        "grad_trust_metric"
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
                 freeze_eval_samples: bool = False,
                 actor_eval_type='full',
                 actor_encoder_cls=TransformerEncoder,
                 actor_encoding_dim=128,
                 obs_action_encoding_dim=64,
                 trust_cov_reg: float = 1e-4,
                 trust_metric_num_obs: int = 128,
                 trust_metric_num_feature_coords: int = 64,
                 trust_metric_update_interval: int = 1,
                 eval_trust_max: float = 2.0,
                 delta_trust_max: float = 2.0,
                 monitor_trust_metrics: bool = True,
                 enable_eval_rollout_skip_gate: bool = False,
                 enable_grad_actor_extend_gate: bool = False,
                 eval_gate_max_consecutive_rollout_skips: int = 5,
                 grad_gate_max_consecutive_actor_extensions: Optional[int] = None,
                 actor_utd: Optional[int] = None,
                 critic_utd: Optional[int] = None,
                 rollout_cycles_per_collect: int = 3,
                 reference_actor_sync_interval: Optional[int] = None,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_critic_tau: Union[float, Scheduler] = 0.05,
                 target_critic_period: Union[int, Scheduler] = 1,
                 target_critic_use_ema=False,
                 parameter_reset_period: Union[int, Scheduler] = -1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 actor_encoder_optimizer=None,
                 eval_samples_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="BafcAlgorithm"):
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
            enable_eval_rollout_skip_gate (bool): whether to skip rollout
                collection when eval trust is below threshold. When False,
                eval trust is not computed (to save compute).
            freeze_eval_samples (bool): If True, keep actor eval samples fixed
                throughout training instead of optimizing them.
            rollout_cycles_per_collect (int): number of completed
                critic-actor cycles to train on replay data after each unroll.
                A cycle is counted when train mode switches from ``actor`` back
                to ``critic``.
            reference_actor_sync_interval (None|int): sync the reference actor
                from the current actor once every this many real rollout
                collections. If None, defaults to half the replay buffer length
                in rollout collections, based on ``TrainerConfig``.
        """
        assert actor_eval_type in ['full', 'exclude_input', 'last_two', 'output'], (
            r"{actor_eval_type} in not supported.")
        assert eval_samples_init_method in ['normal', 'uniform'], (
            r"init method {eval_samples_init_method} is not supported.")
        assert bootstrap_mask_type in ['episode', 'step'], (
            r"bootstrap mask type {bootstrap_mask_type} is not supported.")
        assert trust_cov_reg > 0, "trust_cov_reg must be > 0"
        assert trust_metric_num_obs >= 1, "trust_metric_num_obs must be >= 1"
        assert trust_metric_num_feature_coords >= 1, (
            "trust_metric_num_feature_coords must be >= 1")
        assert trust_metric_update_interval >= 1, (
            "trust_metric_update_interval must be >= 1")
        assert eval_gate_max_consecutive_rollout_skips >= 1, (
            "eval_gate_max_consecutive_rollout_skips must be >= 1")
        assert not (freeze_eval_samples and eval_samples_optimizer is not None), (
            "eval_samples_optimizer cannot be set when freeze_eval_samples=True.")
        if grad_gate_max_consecutive_actor_extensions is not None:
            assert grad_gate_max_consecutive_actor_extensions >= 1, (
                "grad_gate_max_consecutive_actor_extensions must be >= 1 when set")
        assert rollout_cycles_per_collect >= 1, (
            "rollout_cycles_per_collect must be >= 1")
        if reference_actor_sync_interval is None:
            reference_actor_sync_interval = 1
            if config is not None:
                unroll_length = config.unroll_length
                if unroll_length == 0:
                    unroll_length = config.max_unroll_length
                unroll_length = max(1, unroll_length)
                reference_actor_sync_interval = max(
                    1,
                    int(np.ceil(config.replay_buffer_length /
                                (2 * unroll_length))))
        assert reference_actor_sync_interval >= 1, (
            "reference_actor_sync_interval must be >= 1")
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
        self._trust_cov_reg = trust_cov_reg
        self._trust_metric_num_obs = trust_metric_num_obs
        self._trust_metric_num_feature_coords = trust_metric_num_feature_coords
        self._trust_metric_update_interval = trust_metric_update_interval
        self._eval_trust_max = eval_trust_max
        self._delta_trust_max = delta_trust_max
        self._monitor_trust_metrics = monitor_trust_metrics
        self._enable_eval_rollout_skip_gate = enable_eval_rollout_skip_gate
        self._enable_grad_actor_extend_gate = enable_grad_actor_extend_gate
        self._eval_gate_max_consecutive_rollout_skips = (
            eval_gate_max_consecutive_rollout_skips)
        self._freeze_eval_samples = freeze_eval_samples
        self._grad_gate_max_consecutive_actor_extensions = (
            grad_gate_max_consecutive_actor_extensions)
        self._rollout_cycles_per_collect = rollout_cycles_per_collect
        self._reference_actor_sync_interval = reference_actor_sync_interval
        self._real_rollouts_since_reference_sync = 0
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
        if freeze_eval_samples:
            self.register_buffer('_actor_eval_samples', actor_eval_samples)
        else:
            self._actor_eval_samples = nn.Parameter(actor_eval_samples)
            if eval_samples_optimizer is not None:
                self.add_optimizer(eval_samples_optimizer,
                                   [self._actor_eval_samples])

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
        self._completed_cycles_since_rollout = 0
        self._dqda_clipping = dqda_clipping
        self._training_started = False
        self._do_critic_summary = False
        self._last_eval_trust = torch.tensor(1.0)
        self._last_grad_trust = torch.tensor(1.0)
        # Trust metrics are observability-only. This counter only schedules
        # when we refresh the cached logging values.
        self._trust_metric_update_counter = 0
        self._eval_gate_consecutive_rollout_skips = 0
        self._rollout_skip_due_eval_gate_count = 0
        self._rollout_opportunity_count = 0
        self._active_rollout_skip_start_opportunity = None
        self._pending_rollout_skip_event = None
        self._grad_gate_actor_extension_count = 0
        self._grad_gate_consecutive_actor_extensions = 0
        self._active_grad_extension_start_step = None
        self._pending_grad_extension_event = None
        self._last_rollout_skipped_due_eval_gate = False
        self._last_grad_gate_actor_extended = False
        self._last_update_had_actor_step = False

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
        self._sync_reference_from_current()
        self._sync_snapshot_critic_from_current()

    def _sync_reference_from_current(self):
        self._reference_actor_networks.load_state_dict(
            self._actor_networks.state_dict())

    def _sync_snapshot_critic_from_current(self):
        self._snapshot_critic_networks.load_state_dict(
            self._critic_networks.state_dict())

    def _record_debug_scalar(self, name: str, value):
        if not (self._debug_summaries and alf.summary.should_record_summaries()):
            return
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() != 1:
                value = value.mean()
            value = float(value.item())
        else:
            value = float(value)
        with alf.summary.scope(self._name):
            alf.summary.scalar(name, value)

    def _group_grad_sq_norm(self, grads, group_idx: int):
        sq_norm = None
        for grad in grads:
            if grad is None:
                continue
            if grad.ndim > 0 and grad.shape[0] == self._num_actor_critic:
                grad = grad[group_idx]
            if sq_norm is None:
                sq_norm = grad.new_zeros(())
            sq_norm = sq_norm + grad.pow(2).sum()
        if sq_norm is None:
            sq_norm = self._actor_eval_samples.new_zeros(())
        return sq_norm

    def _scalar_grad_sq_norm(self,
                             scalar,
                             params,
                             group_idx: int,
                             retain_graph: bool):
        if not isinstance(scalar, torch.Tensor) or not scalar.requires_grad:
            if isinstance(scalar, torch.Tensor):
                return scalar.new_zeros(())
            return self._actor_eval_samples.new_zeros(())
        grads = torch.autograd.grad(
            scalar,
            params,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True)
        return self._group_grad_sq_norm(grads, group_idx)

    def _batched_output_grad_sq_norm(self,
                                     output_means: torch.Tensor,
                                     params,
                                     retain_graph: bool,
                                     max_chunk_size: Optional[int] = None):
        """Compute per-output grad squared norms with chunked batched VJPs.

        Args:
            output_means: Tensor shaped [G, D], where each scalar output is
                differentiated w.r.t. ``params``.
            params: Iterable of actor parameters.
            retain_graph: Forwarded retain policy for the final successful
                chunk so callers can chain further gradient queries.
            max_chunk_size: Optional internal override used by tests.

        Returns:
            Tensor shaped [G, D] with per-output gradient squared norms,
            matching the semantics of looping over scalars and calling
            ``_scalar_grad_sq_norm``.
        """
        if not isinstance(output_means, torch.Tensor):
            return self._actor_eval_samples.new_zeros((0, 0))
        if output_means.ndim != 2:
            raise ValueError(
                f"Expected output_means with shape [G, D], got "
                f"{tuple(output_means.shape)}")
        if not output_means.requires_grad:
            return output_means.new_zeros(output_means.shape)

        num_groups, output_dim = output_means.shape
        flat_outputs = output_means.reshape(-1)
        num_outputs = flat_outputs.shape[0]
        if num_outputs == 0:
            return output_means.new_zeros(output_means.shape)

        start_chunk = min(128, num_outputs)
        if max_chunk_size is not None:
            start_chunk = min(max(1, int(max_chunk_size)), num_outputs)

        flat_group_idx = torch.arange(
            num_groups, device=output_means.device).repeat_interleave(output_dim)
        grad_sq = output_means.new_zeros(num_outputs)

        chunk_size = start_chunk
        while True:
            try:
                start = 0
                while start < num_outputs:
                    end = min(start + chunk_size, num_outputs)
                    chunk_len = end - start
                    row_idx = torch.arange(chunk_len, device=output_means.device)
                    col_idx = torch.arange(start, end, device=output_means.device)
                    basis = flat_outputs.new_zeros((chunk_len, num_outputs))
                    basis[row_idx, col_idx] = 1.0

                    keep_graph = retain_graph or (end < num_outputs)
                    batched_grads = torch.autograd.grad(
                        flat_outputs,
                        params,
                        grad_outputs=basis,
                        retain_graph=keep_graph,
                        create_graph=False,
                        allow_unused=True,
                        is_grads_batched=True)

                    chunk_grad_sq = flat_outputs.new_zeros(chunk_len)
                    chunk_group_idx = flat_group_idx[start:end]
                    for grad, param in zip(batched_grads, params):
                        if grad is None:
                            continue
                        if param.ndim > 0 and param.shape[0] == num_groups:
                            selected = grad[row_idx, chunk_group_idx]
                        else:
                            selected = grad
                        chunk_grad_sq = chunk_grad_sq + selected.reshape(
                            chunk_len, -1).pow(2).sum(dim=1)

                    grad_sq[start:end] = chunk_grad_sq
                    start = end
                return grad_sq.view(num_groups, output_dim)
            except RuntimeError:
                if chunk_size == 1:
                    raise
                next_chunk = max(1, chunk_size // 2)
                logging.warning(
                    "Batched VJP for grad-trust failed at chunk_size=%d; "
                    "retrying with chunk_size=%d.", chunk_size, next_chunk)
                chunk_size = next_chunk

    def _sample_metric_observations(self, observation):
        if not isinstance(observation, torch.Tensor):
            return ()
        obs_dim = len(self._observation_spec.shape)
        obs = observation.reshape(-1, *observation.shape[-obs_dim:])
        if obs.shape[0] == 0:
            return ()
        if obs.shape[0] > self._trust_metric_num_obs:
            idx = torch.randperm(obs.shape[0], device=obs.device)[
                :self._trust_metric_num_obs]
            obs = obs[idx]
        return obs

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

    def _sample_feature_coords(self, feature_dim, device):
        if feature_dim <= self._trust_metric_num_feature_coords:
            return torch.arange(feature_dim, device=device)
        return torch.randperm(feature_dim, device=device)[
            :self._trust_metric_num_feature_coords]

    def _critic_feature_head_index(self, critic_network):
        critic_core = getattr(critic_network, '_pnet', critic_network)
        modules = getattr(critic_core, '_networks', None)
        if modules is None:
            raise RuntimeError(
                "Snapshot critic does not expose the sequential modules needed "
                "for trust-feature extraction.")

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
        """Return the normalized snapshot critic feature map before the scalar head.

        The feature map is the pre-activation input to the last scalar critic
        head, not the raw action and not the final scalar Q value. Each
        feature vector is L2-normalized along the feature dimension.
        """
        critic_network = (self._snapshot_critic_networks
                          if critic_network is None else critic_network)
        critic_core = getattr(critic_network, '_pnet', critic_network)
        modules = critic_core._networks
        head_idx = self._critic_feature_head_index(critic_network)

        action = self._ensure_group_action(action)
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

    def _compute_feature_inv_cov(self, feature_map):
        feature_by_group = feature_map.permute(1, 0, 2)  # [G, N, F]
        cov = (feature_by_group.transpose(1, 2) @ feature_by_group /
               feature_by_group.shape[1])
        feature_dim = cov.shape[-1]
        eye = torch.eye(
            feature_dim, dtype=cov.dtype, device=cov.device).unsqueeze(0)
        cov = cov + self._trust_cov_reg * eye
        return torch.linalg.pinv(cov)

    def _compute_weighted_feature_norm(self, feature_map, inv_cov):
        feature_by_group = feature_map.permute(1, 0, 2)  # [G, N, F]
        weighted = torch.matmul(feature_by_group, inv_cov)
        squared_norm = torch.clamp(
            (weighted * feature_by_group).sum(-1), min=0.)
        return torch.sqrt(squared_norm + 1e-12).permute(1, 0)  # [N, G]

    def _compute_eval_trust_from_features(self, phi_ref, phi_beh):
        inv_cov = self._compute_feature_inv_cov(phi_beh)
        weighted_norm = self._compute_weighted_feature_norm(phi_ref, inv_cov)
        return torch.clamp(weighted_norm.pow(2).mean(), min=0.)

    def _should_refresh_trust_metrics(self):
        return (self._monitor_trust_metrics and self._last_update_had_actor_step
                and self._trust_metric_update_counter %
                self._trust_metric_update_interval == 0)

    @torch.no_grad()
    def _compute_eval_trust_metric(self, observation):
        obs = self._sample_metric_observations(observation)
        if not isinstance(obs, torch.Tensor):
            return torch.ones_like(self._last_eval_trust)

        ref_action = self._ensure_group_action(
            self._reference_actor_networks(obs)[0]).detach()
        cur_action = self._ensure_group_action(self._actor_networks(obs)[0]).detach()
        ref_encoding = self._compute_actor_encoding(
            self._reference_actor_networks).detach()

        phi_ref = self._compute_snapshot_feature_map(
            obs, ref_encoding, ref_action).detach()
        # Use the same reference conditioning while swapping in the current
        # actor actions to measure drift relative to rollout-aligned reference.
        phi_beh = self._compute_snapshot_feature_map(
            obs, ref_encoding, cur_action).detach()

        return self._compute_eval_trust_from_features(phi_ref, phi_beh)

    def _compute_grad_generalization_trust_components(self, observation):
        obs = self._sample_metric_observations(observation)
        if not isinstance(obs, torch.Tensor):
            one = torch.ones_like(self._last_eval_trust)
            return one, one

        actor_params = list(self._actor_networks.parameters())
        if len(actor_params) == 0:
            one = torch.ones_like(self._last_eval_trust)
            return one, one
        param_requires_grad = [p.requires_grad for p in actor_params]
        for p in actor_params:
            if not p.requires_grad:
                p.requires_grad_(True)

        try:
            obs = obs.detach()
            cur_action = self._ensure_group_action(self._actor_networks(obs)[0])
            with torch.no_grad():
                ref_action = self._ensure_group_action(
                    self._reference_actor_networks(obs.detach())[0])
                ref_encoding = self._compute_actor_encoding(
                    self._reference_actor_networks).detach()
            cur_encoding = self._compute_actor_encoding(self._actor_networks)

            # Keep the snapshot critic frozen and measure only how actor
            # changes move the anchored feature map.
            phi_ref = self._compute_snapshot_feature_map(
                obs, ref_encoding, ref_action).detach()
            phi_t = self._compute_snapshot_feature_map(
                obs, cur_encoding, cur_action)
            a_inv = self._compute_feature_inv_cov(phi_ref)
            feature_norm = self._compute_weighted_feature_norm(phi_t, a_inv).mean(
                dim=0)  # [G]

            action_mean = cur_action.mean(dim=0)  # [G, A]
            grad_sq_mu = self._batched_output_grad_sq_norm(
                action_mean, actor_params, retain_graph=True)
            feature_dim = phi_t.shape[-1]
            feature_coords = self._sample_feature_coords(feature_dim, phi_t.device)

            jacobian_norm = torch.sqrt(
                torch.clamp(grad_sq_mu.sum(-1), min=0.) + 1e-12)
            c1 = torch.mean(jacobian_norm * feature_norm)

            a_inv_diag = torch.clamp(
                torch.diagonal(a_inv, dim1=-2, dim2=-1), min=0.)
            sampled_inv_diag = a_inv_diag[:, feature_coords]
            # C2 is the deterministic proxy for D_theta phi_t: the frozen
            # feature map changes only through the current actor.
            phi_coord_mean = phi_t[:, :, feature_coords].mean(dim=0)  # [G, F']
            grad_sq_phi = self._batched_output_grad_sq_norm(
                phi_coord_mean, actor_params, retain_graph=False)

            weighted_grad_sq = torch.clamp(sampled_inv_diag * grad_sq_phi,
                                           min=0.)
            coord_scale = float(feature_dim) / float(feature_coords.shape[0])
            c2 = torch.mean(
                torch.sqrt(
                    torch.clamp(coord_scale * weighted_grad_sq.sum(-1),
                                min=0.) + 1e-12))

            return torch.clamp(c1, min=0.), torch.clamp(c2, min=0.)
        finally:
            for p, req in zip(actor_params, param_requires_grad):
                p.requires_grad_(req)

    def _compute_grad_generalization_trust_metric(self, observation):
        c1, c2 = self._compute_grad_generalization_trust_components(observation)
        return torch.maximum(c1, c2)

    def _broadcast_trust_metric(self, metric, batch_shape, device, dtype):
        metric = torch.as_tensor(metric, device=device, dtype=dtype)
        batch_shape = tuple(batch_shape)
        if not batch_shape:
            return metric
        if metric.shape == batch_shape:
            return metric
        if metric.numel() == 1:
            return torch.full(
                batch_shape,
                float(metric.reshape(()).item()),
                dtype=dtype,
                device=device)
        return torch.broadcast_to(metric, batch_shape).clone()

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
            info=BafcInfo(
                action=action,
                bootstrap_mask=self._bootstrap_mask,
                eval_trust_metric=self._broadcast_trust_metric(
                    self._last_eval_trust,
                    inputs.reward.shape,
                    inputs.reward.device,
                    inputs.reward.dtype),
                grad_trust_metric=self._broadcast_trust_metric(
                    self._last_grad_trust,
                    inputs.reward.shape,
                    inputs.reward.device,
                    inputs.reward.dtype)))

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
            extra=BafcActorInfo(
                eval_action_loss=eval_action_loss,
                grad_trust_metric=self._broadcast_trust_metric(
                    self._last_grad_trust,
                    action_loss.shape,
                    action_loss.device,
                    action_loss.dtype)))
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
        info = BafcCriticInfo(
            critic=critics,
            target_critic=target_critics,
            eval_trust_metric=self._broadcast_trust_metric(
                self._last_eval_trust,
                critics.shape[:-2],
                critics.device,
                critics.dtype))

        return state, info

    def _update_train_mode(self):
        self._last_grad_gate_actor_extended = False
        if self._train_mode == TrainMode.actor:
            if self._actor_update_counter % self._actor_utd == 0:
                should_extend_actor = False
                previous_consecutive_extensions = (
                    self._grad_gate_consecutive_actor_extensions)
                if self._enable_grad_actor_extend_gate:
                    grad_trust = float(
                        torch.as_tensor(self._last_grad_trust).reshape(
                            ()).item())
                    # Algorithm 3: continue actor-improvement steps while the
                    # anchored critic remains trustworthy, and refresh critic
                    # when grad trust violates the threshold.
                    should_extend_actor = grad_trust <= self._delta_trust_max
                    if (should_extend_actor and
                            self._grad_gate_max_consecutive_actor_extensions
                            is not None and
                            self._grad_gate_consecutive_actor_extensions >=
                            self._grad_gate_max_consecutive_actor_extensions):
                        should_extend_actor = False
                if should_extend_actor:
                    if previous_consecutive_extensions == 0:
                        self._active_grad_extension_start_step = int(
                            alf.summary.get_global_counter())
                    self._last_grad_gate_actor_extended = True
                    self._grad_gate_actor_extension_count += 1
                    self._grad_gate_consecutive_actor_extensions += 1
                    if previous_consecutive_extensions == 0:
                        self._set_grad_extension_event(
                            "grad_extension_start",
                            self._grad_gate_consecutive_actor_extensions)
                else:
                    if previous_consecutive_extensions > 0:
                        self._set_grad_extension_event(
                            "grad_extension_end",
                            previous_consecutive_extensions)
                        self._active_grad_extension_start_step = None
                    self._grad_gate_consecutive_actor_extensions = 0
                    self._train_mode = TrainMode.critic
                    self._completed_cycles_since_rollout += 1
                    # self._critic_network.set_obs_action_batch_dominate(False)
                    for p in self._actor_networks.parameters():
                        p.requires_grad_(False)
                    if not self._freeze_eval_samples:
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
        self._last_update_had_actor_step = False

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
            self._last_update_had_actor_step = True
            self._critic_update_counter += 1
        else:
            if self._train_mode == TrainMode.actor:
                actor_state, actor_info = self._actor_train_step(
                    inputs.observation, action, 
                    rollout_info.bootstrap_mask, state.actor)
                critic_info = BafcCriticInfo()
                new_state = BafcState(action=action_state,
                                      actor=actor_state,
                                      critic=state.critic)
                self._last_update_had_actor_step = True
                self._actor_update_counter += 1
            else:
                action = action.reshape(-1, action.shape[-1])  # [T*B * n_actor, d_a]
                critic_state, critic_info = self._critic_train_step(
                    inputs.observation, state.critic, rollout_info, action)
                actor_info = LossInfo(
                    extra=BafcActorInfo(
                        grad_trust_metric=self._broadcast_trust_metric(
                            self._last_grad_trust,
                            inputs.reward.shape,
                            inputs.reward.device,
                            inputs.reward.dtype)))
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
            bootstrap_mask=rollout_info.bootstrap_mask,
            eval_trust_metric=self._broadcast_trust_metric(
                self._last_eval_trust,
                inputs.reward.shape,
                inputs.reward.device,
                inputs.reward.dtype),
            grad_trust_metric=self._broadcast_trust_metric(
                self._last_grad_trust,
                inputs.reward.shape,
                inputs.reward.device,
                inputs.reward.dtype))
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
        if not isinstance(info.critic, BafcCriticInfo) or not isinstance(
                info.critic.critic, torch.Tensor):
            return LossInfo()
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
        return [
            '_target_critic_networks', '_reference_actor_networks',
            '_snapshot_critic_networks'
        ]

    def get_policy_boundary_eval_state(self):
        """Return transient policy state needed by boundary evaluators."""
        rollout_actor_id = torch.as_tensor(self._rollout_actor_id).reshape(())
        return dict(
            training_started=bool(self._training_started),
            rollout_actor_id=int(rollout_actor_id.item()))

    def set_policy_boundary_eval_state(self, state):
        """Restore transient policy state needed by ``predict_step()``."""
        self._training_started = bool(state["training_started"])
        self._rollout_actor_id = int(state["rollout_actor_id"])

    def _pop_rollout_skip_event(self):
        """Return and clear the pending rollout-skip boundary event."""
        event = self._pending_rollout_skip_event
        self._pending_rollout_skip_event = None
        return event

    def _pop_grad_extension_event(self):
        """Return and clear the pending grad-gate extension boundary event."""
        event = self._pending_grad_extension_event
        self._pending_grad_extension_event = None
        return event

    def _set_grad_extension_event(self, event_type: str,
                                  extension_length: int):
        start_step = self._active_grad_extension_start_step
        current_step = int(alf.summary.get_global_counter())
        if start_step is None:
            start_step = current_step
        self._pending_grad_extension_event = dict(
            type=event_type,
            start_step=int(start_step),
            end_step=current_step,
            extension_length=int(extension_length))

    def _set_rollout_skip_event(self, event_type: str, skip_length: int):
        start_opportunity = self._active_rollout_skip_start_opportunity
        if start_opportunity is None:
            start_opportunity = max(
                1, self._rollout_opportunity_count - skip_length)
        self._pending_rollout_skip_event = dict(
            type=event_type,
            start_rollout_opportunity=int(start_opportunity),
            end_rollout_opportunity=int(self._rollout_opportunity_count),
            skip_length=int(skip_length))

    def _should_skip_unroll_iter_off_policy(self):
        """Return whether the next off-policy unroll should be skipped."""
        self._last_rollout_skipped_due_eval_gate = False
        self._pending_rollout_skip_event = None

        # Preserve the default behavior during warmup/initial collection and
        # for standard (non-alternating) training mode.
        if not self._training_started or self._train_mode == TrainMode.standard:
            return False

        # Skip rollout until enough actor->critic cycles have completed since
        # the previous rollout. Once the threshold is reached, this iter becomes
        # a rollout opportunity.
        if self._completed_cycles_since_rollout < self._rollout_cycles_per_collect:
            return True

        self._rollout_opportunity_count += 1
        previous_consecutive_skips = self._eval_gate_consecutive_rollout_skips

        # Real rollout skip: when eval trust remains below threshold, skip this
        # rollout and keep training from replay only.
        if self._enable_eval_rollout_skip_gate:
            eval_trust = float(
                torch.as_tensor(self._last_eval_trust).reshape(()).item())
            if (eval_trust <= self._eval_trust_max
                    and previous_consecutive_skips <
                    self._eval_gate_max_consecutive_rollout_skips):
                if previous_consecutive_skips == 0:
                    self._active_rollout_skip_start_opportunity = (
                        self._rollout_opportunity_count)
                self._eval_gate_consecutive_rollout_skips += 1
                self._rollout_skip_due_eval_gate_count += 1
                self._last_rollout_skipped_due_eval_gate = True
                if previous_consecutive_skips == 0:
                    self._set_rollout_skip_event(
                        "skip_start",
                        self._eval_gate_consecutive_rollout_skips)
                return True

        if previous_consecutive_skips > 0:
            self._set_rollout_skip_event("skip_end",
                                         previous_consecutive_skips)

        return False

    def _after_unroll_iter_off_policy(self, unrolled):
        """Update BAFC rollout-gate state after a real parent unroll."""
        if not unrolled:
            return
        self._real_rollouts_since_reference_sync += 1
        if (self._real_rollouts_since_reference_sync >=
                self._reference_actor_sync_interval):
            self._sync_reference_from_current()
            self._real_rollouts_since_reference_sync = 0
        if self._training_started and self._train_mode != TrainMode.standard:
            self._completed_cycles_since_rollout = 0
        self._eval_gate_consecutive_rollout_skips = 0
        self._active_rollout_skip_start_opportunity = None

    def _unroll_iter_off_policy(self):
        """Gate rollout cadence by completed actor->critic cycles."""
        if self._should_skip_unroll_iter_off_policy():
            # Keep summary-step progression aligned with the parent off-policy
            # path when BAFC bypasses ``super()._unroll_iter_off_policy()``.
            if not self._config.update_counter_every_mini_batch:
                alf.summary.increment_global_counter()
            return False, None, None

        unrolled, root_inputs, rollout_info = super()._unroll_iter_off_policy()
        self._after_unroll_iter_off_policy(unrolled)
        return unrolled, root_inputs, rollout_info

    def after_update(self, root_inputs, info: BafcInfo):
        del info
        should_compute_trust_metrics = self._should_refresh_trust_metrics()
        if should_compute_trust_metrics:
            observation = ()
            if hasattr(root_inputs, "observation"):
                observation = root_inputs.observation
            if isinstance(observation, torch.Tensor):
                if self._enable_eval_rollout_skip_gate:
                    self._last_eval_trust = self._compute_eval_trust_metric(
                        observation).detach()
                else:
                    # Keep eval-trust outputs interface-stable while skipping
                    # expensive eval-trust computation when eval gating is off.
                    self._last_eval_trust = torch.ones_like(
                        self._last_eval_trust)
                if self._enable_grad_actor_extend_gate:
                    self._last_grad_trust = self._compute_grad_generalization_trust_metric(
                        observation).detach()
                else:
                    # Keep grad-trust outputs interface-stable while skipping
                    # expensive grad-trust computation when grad gating is off.
                    self._last_grad_trust = torch.ones_like(
                        self._last_grad_trust)
            else:
                self._last_eval_trust = torch.ones_like(self._last_eval_trust)
                self._last_grad_trust = torch.ones_like(self._last_grad_trust)
        if self._last_update_had_actor_step:
            self._trust_metric_update_counter += 1

        self._update_train_mode()
        self._update_target_critic()
        self._sync_snapshot_critic_from_current()
        self._record_debug_scalar('eval_trust_metric', self._last_eval_trust)
        self._record_debug_scalar('grad_trust_metric', self._last_grad_trust)
        self._record_debug_scalar('eval_trust_max', self._eval_trust_max)
        self._record_debug_scalar('delta_trust_max', self._delta_trust_max)
        self._record_debug_scalar('eval_trust_over_max',
                                  self._last_eval_trust /
                                  max(self._eval_trust_max, 1e-6))
        self._record_debug_scalar('grad_trust_over_max',
                                  self._last_grad_trust /
                                  max(self._delta_trust_max, 1e-6))
        self._record_debug_scalar('eval_gate_enabled',
                                  float(self._enable_eval_rollout_skip_gate))
        self._record_debug_scalar('grad_gate_enabled',
                                  float(self._enable_grad_actor_extend_gate))
        self._record_debug_scalar(
            'eval_gate_consecutive_rollout_skips',
            float(self._eval_gate_consecutive_rollout_skips))
        self._record_debug_scalar(
            'rollout_skipped_due_eval_gate',
            float(self._last_rollout_skipped_due_eval_gate))
        self._record_debug_scalar(
            'rollout_skip_due_eval_gate_count',
            float(self._rollout_skip_due_eval_gate_count))
        self._record_debug_scalar('rollout_opportunity_count',
                                  float(self._rollout_opportunity_count))
        self._record_debug_scalar('actor_extended_due_grad_gate',
                                  float(self._last_grad_gate_actor_extended))
        self._record_debug_scalar('actor_extension_due_grad_gate_count',
                                  float(self._grad_gate_actor_extension_count))
        self._record_debug_scalar(
            'actor_extension_consecutive_count',
            float(self._grad_gate_consecutive_actor_extensions))
        cap = self._grad_gate_max_consecutive_actor_extensions
        if cap is None:
            cap = -1
        self._record_debug_scalar('actor_extension_consecutive_cap',
                                  float(cap))
