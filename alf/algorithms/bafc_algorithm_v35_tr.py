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
class BafcAlgorithmV35(OffPolicyAlgorithm):
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
                 trust_cov_reg: float = 1e-4,
                 trust_metric_num_obs: int = 128,
                 trust_metric_num_feature_coords: int = 64,
                 trust_metric_update_interval: int = 1,
                 eval_trust_max: float = 2.0,
                 delta_trust_max: float = 2.0,
                 monitor_trust_metrics: bool = True,
                 enable_eval_rollout_skip_gate: bool = False,
                 enable_grad_actor_extend_gate: bool = False,
                 preupdate_grad_metric_anchor: str = 'reference',
                 eval_gate_max_consecutive_rollout_actor_holds:
                 Optional[int] = None,
                 eval_gate_max_consecutive_rollout_skips:
                 Optional[int] = None,
                 grad_gate_max_consecutive_actor_extensions: Optional[int] = None,
                 actor_utd: Optional[int] = None,
                 critic_utd: Optional[int] = None,
                 rollout_cycles_per_collect: int = 3,
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
            enable_eval_rollout_skip_gate (bool): whether to gate rollout actor
                refreshes using the eval trust metric. When False, eval trust is
                not computed (to save compute), and rollout always refreshes from
                the latest reference actor.
            rollout_cycles_per_collect (int): number of completed
                critic-actor cycles to train on replay data after each unroll.
                A cycle is counted when train mode switches from ``actor`` back
                to ``critic``.
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
        if eval_gate_max_consecutive_rollout_actor_holds is None:
            if eval_gate_max_consecutive_rollout_skips is None:
                eval_gate_max_consecutive_rollout_actor_holds = 5
            else:
                logging.warning(
                    "`eval_gate_max_consecutive_rollout_skips` is deprecated. "
                    "Use `eval_gate_max_consecutive_rollout_actor_holds` "
                    "instead.")
                eval_gate_max_consecutive_rollout_actor_holds = (
                    eval_gate_max_consecutive_rollout_skips)
        elif (eval_gate_max_consecutive_rollout_skips is not None
              and eval_gate_max_consecutive_rollout_skips !=
              eval_gate_max_consecutive_rollout_actor_holds):
            raise ValueError(
                "`eval_gate_max_consecutive_rollout_actor_holds` and "
                "`eval_gate_max_consecutive_rollout_skips` must match when "
                "both are provided.")
        elif eval_gate_max_consecutive_rollout_skips is not None:
            logging.warning(
                "`eval_gate_max_consecutive_rollout_skips` is deprecated. "
                "Use `eval_gate_max_consecutive_rollout_actor_holds` instead.")
        assert eval_gate_max_consecutive_rollout_actor_holds >= 1, (
            "eval_gate_max_consecutive_rollout_actor_holds must be >= 1")
        if grad_gate_max_consecutive_actor_extensions is not None:
            assert grad_gate_max_consecutive_actor_extensions >= 1, (
                "grad_gate_max_consecutive_actor_extensions must be >= 1 when set")
        assert rollout_cycles_per_collect >= 1, (
            "rollout_cycles_per_collect must be >= 1")
        assert preupdate_grad_metric_anchor in ('reference', 'self'), (
            "preupdate_grad_metric_anchor must be 'reference' or 'self'")
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
        self._preupdate_grad_metric_anchor = preupdate_grad_metric_anchor
        self._eval_gate_max_consecutive_rollout_actor_holds = (
            eval_gate_max_consecutive_rollout_actor_holds)
        self._eval_gate_max_consecutive_rollout_skips = (
            eval_gate_max_consecutive_rollout_actor_holds)
        self._grad_gate_max_consecutive_actor_extensions = (
            grad_gate_max_consecutive_actor_extensions)
        self._rollout_cycles_per_collect = rollout_cycles_per_collect
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
        critic_saved_args = getattr(critic_network, 'saved_args', {})
        self._manual_critic_saved_args = dict(critic_saved_args)
        self._manual_critic_obs_action_hidden = tuple(
            critic_saved_args.get('obs_action_joint_fc_layer_params') or ())
        self._manual_critic_joint_hidden = tuple(
            critic_saved_args.get('actor_obs_action_joint_fc_layer_params') or ())
        self._manual_critic_obs_action_num_layers = (
            len(self._manual_critic_obs_action_hidden) + 1)
        self._manual_critic_joint_num_layers = (
            len(self._manual_critic_joint_hidden) + 1)

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
        self._behavior_actor_networks = actor_networks.copy(
            name='behavior_actor_networks')
        for p in self._reference_actor_networks.parameters():
            p.requires_grad_(False)
        for p in self._behavior_actor_networks.parameters():
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
        # Pre-update grad trust computed inside the actor train step.  When
        # present, after_update() consumes this value instead of recomputing the
        # expensive post-update metric.
        self._pending_preupdate_grad_trust = None
        # Trust metrics are observability-only. This counter only schedules
        # when we refresh the cached logging values.
        self._trust_metric_update_counter = 0
        self._eval_gate_consecutive_rollout_actor_holds = 0
        self._rollout_actor_hold_due_eval_gate_count = 0
        self._grad_gate_actor_extension_count = 0
        self._grad_gate_consecutive_actor_extensions = 0
        self._last_rollout_actor_held_due_eval_gate = False
        self._last_rollout_actor_refreshed_from_reference = False
        self._last_rollout_actor_refresh_forced_by_eval_gate_cap = False
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
        self._sync_behavior_from_reference()
        self._sync_snapshot_critic_from_current()

    def _sync_reference_from_current(self):
        self._reference_actor_networks.load_state_dict(
            self._actor_networks.state_dict())

    def _sync_behavior_from_reference(self):
        self._behavior_actor_networks.load_state_dict(
            self._reference_actor_networks.state_dict())

    def _update_rollout_actor_from_eval_gate(self):
        self._last_rollout_actor_held_due_eval_gate = False
        self._last_rollout_actor_refreshed_from_reference = False
        self._last_rollout_actor_refresh_forced_by_eval_gate_cap = False

        if not self._enable_eval_rollout_skip_gate:
            self._sync_behavior_from_reference()
            self._last_rollout_actor_refreshed_from_reference = True
            self._eval_gate_consecutive_rollout_actor_holds = 0
            return

        eval_trust = float(
            torch.as_tensor(self._last_eval_trust).reshape(()).item())
        if eval_trust > self._eval_trust_max:
            self._sync_behavior_from_reference()
            self._last_rollout_actor_refreshed_from_reference = True
            self._eval_gate_consecutive_rollout_actor_holds = 0
            return

        if (self._eval_gate_consecutive_rollout_actor_holds >=
                self._eval_gate_max_consecutive_rollout_actor_holds):
            self._sync_behavior_from_reference()
            self._last_rollout_actor_refreshed_from_reference = True
            self._last_rollout_actor_refresh_forced_by_eval_gate_cap = True
            self._eval_gate_consecutive_rollout_actor_holds = 0
            return

        self._eval_gate_consecutive_rollout_actor_holds += 1
        self._rollout_actor_hold_due_eval_gate_count += 1
        self._last_rollout_actor_held_due_eval_gate = True

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
        """Return the frozen snapshot critic feature map before the scalar head.

        The feature map is the pre-activation input to the last scalar critic
        head, not the raw action and not the final scalar Q value.
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
                return x
            if isinstance(module, Network):
                x = module(x)[0]
            else:
                x = module(x)

        raise RuntimeError(
            "Failed to extract snapshot critic feature map before the scalar "
            "head.")

    def _compute_critic_value_and_feature_map(self, observation, actor_encoding,
                                           action, state=(), critic_network=None):
        """Run a parallel functional critic and also return its penultimate feature.

        This mirrors ``_compute_snapshot_feature_map()`` but continues through the
        scalar head. It is used by the actor-step fast grad-trust path so that
        the penultimate feature map ``phi`` is captured from the same critic
        forward that produces ``q_value``.
        """
        critic_network = self._critic_networks if critic_network is None else critic_network
        critic_core = getattr(critic_network, '_pnet', critic_network)
        modules = getattr(critic_core, '_networks', None)
        action = self._ensure_group_action(action)
        if actor_encoding.ndim == 2:
            actor_encoding = actor_encoding.unsqueeze(0).expand(
                observation.shape[0], -1, -1)
        critic_observation = observation.unsqueeze(1).expand(
            -1, self._num_actor_critic, -1)
        if modules is None:
            q_value, critic_state = critic_network(
                (actor_encoding, (critic_observation, action)), state)
            return q_value, critic_state, None

        head_idx = self._critic_feature_head_index(critic_network)
        x = (actor_encoding, (critic_observation, action))
        phi = None
        for idx, module in enumerate(modules):
            if idx == head_idx:
                if not isinstance(x, torch.Tensor) or x.ndim != 3:
                    raise RuntimeError(
                        "Unexpected critic feature-map shape %s" % (type(x), ))
                phi = x
            if isinstance(module, Network):
                x = module(x)[0]
            else:
                x = module(x)
        return x, state, phi

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

    @torch.no_grad()
    def _compute_eval_trust_metric(self, observation):
        obs = self._sample_metric_observations(observation)
        if not isinstance(obs, torch.Tensor):
            return torch.ones_like(self._last_eval_trust)

        ref_action = self._ensure_group_action(
            self._reference_actor_networks(obs)[0]).detach()
        beh_action = self._ensure_group_action(
            self._behavior_actor_networks(obs)[0]).detach()
        ref_encoding = self._compute_actor_encoding(
            self._reference_actor_networks).detach()

        phi_ref = self._compute_snapshot_feature_map(
            obs, ref_encoding, ref_action).detach()
        # Use the same reference conditioning while swapping in the behavior
        # actions, which mirrors the target-conditioned coverage view in Eq. (3.2).
        phi_beh = self._compute_snapshot_feature_map(
            obs, ref_encoding, beh_action).detach()

        return self._compute_eval_trust_from_features(phi_ref, phi_beh)

    def _is_relu_activation(self, activation) -> bool:
        if activation in (torch.relu, torch.relu_):
            return True
        if isinstance(activation, nn.ReLU):
            return True
        name = getattr(activation, '__name__', '')
        return name in ('relu', 'relu_')

    def _is_tanh_squashing(self, squashing_func) -> bool:
        if squashing_func is torch.tanh:
            return True
        name = getattr(squashing_func, '__name__', '')
        return name in ('tanh', 'tanh_')

    def _supports_fast_grad_metric_last_two(self) -> bool:
        if self._actor_eval_type != 'last_two':
            return False
        actor_net = self._actor_networks
        if not isinstance(actor_net, ActorFCNetwork):
            return False
        if len(getattr(actor_net, '_fc_layers', ())) < 1:
            return False
        if getattr(actor_net, '_first_layer_modulated', False):
            return False
        if getattr(actor_net, '_action_layer_modulated', False):
            return False
        if not self._is_tanh_squashing(getattr(actor_net, '_squashing_func', None)):
            return False
        action_layer = getattr(actor_net, '_action_layer', None)
        if action_layer is None:
            return False
        if getattr(action_layer, '_use_bn', False) or getattr(action_layer,
                                                              '_use_ln', False):
            return False
        for layer in actor_net._fc_layers:
            if getattr(layer, '_use_bn', False):
                return False
            if not self._is_relu_activation(getattr(layer, '_activation', None)):
                return False
        return True

    def _is_identity_activation(self, activation) -> bool:
        if activation in (None, math_ops.identity):
            return True
        if isinstance(activation, nn.Identity):
            return True
        name = getattr(activation, '__name__', '')
        return name == 'identity'

    def _is_nest_concat_combiner(self, combiner) -> bool:
        if combiner is None:
            return True
        return combiner.__class__.__name__ == 'NestConcat'

    def _supports_manual_critic_row_vjp(self) -> bool:
        """Whether the current critic layout is supported by the manual row VJP.

        The previous prototype relied on ``saved_args['activation']``. Some
        construction paths do not preserve that key even though the instantiated
        layers are ReLU/identity. The actual ``ParallelFC`` modules are the
        source of truth.
        """
        saved = getattr(self, '_manual_critic_saved_args', {}) or {}

        activation = saved.get('activation', torch.relu_)
        if activation is not None and not self._is_relu_activation(activation):
            return False
        last_activation = saved.get('last_layer_activation', math_ops.identity)
        if last_activation is not None and not self._is_identity_activation(
                last_activation):
            return False
        if saved.get('use_fc_bn', False) or saved.get('last_use_fc_bn', False):
            return False
        if saved.get('joint_use_residual_fc_block', False):
            return False
        if saved.get('use_naive_parallel_network', False):
            return False
        if saved.get('observation_input_processors', None) is not None:
            return False
        if saved.get('observation_input_processors_ctor', None) is not None:
            return False
        if saved.get('action_input_processors', None) is not None:
            return False
        if saved.get('action_input_processors_ctor', None) is not None:
            return False
        if saved.get('observation_preprocessing_combiner', None) is not None:
            return False
        if saved.get('action_preprocessing_combiner', None) is not None:
            return False
        if saved.get('observation_conv_layer_params', None) is not None:
            return False
        if saved.get('observation_fc_layer_params', None) not in (None, (), []):
            return False
        if saved.get('action_fc_layer_params', None) not in (None, (), []):
            return False
        if not self._is_nest_concat_combiner(
                saved.get('observation_action_combiner', None)):
            return False
        if not self._is_nest_concat_combiner(
                saved.get('actor_obs_action_combiner', None)):
            return False

        try:
            obs_action_layers, joint_layers = (
                self._get_manual_critic_parallel_fc_layers(self._critic_networks))
        except RuntimeError:
            return False

        # Hidden layers must be ReLU; the scalar head must be identity. BatchNorm
        # is intentionally excluded because its train/eval-state semantics make
        # a manual row VJP brittle.
        for layer in list(obs_action_layers) + list(joint_layers[:-1]):
            if not isinstance(layer, alf.layers.ParallelFC):
                return False
            if getattr(layer, '_use_bn', False):
                return False
            if not self._is_relu_activation(getattr(layer, '_activation', None)):
                return False

        head = joint_layers[-1]
        if not isinstance(head, alf.layers.ParallelFC):
            return False
        if getattr(head, '_use_bn', False):
            return False
        if not self._is_identity_activation(getattr(head, '_activation', None)):
            return False
        return True

    def _supports_manual_actor_step_rows(self) -> bool:
        # The manual actor-step cache assumes actor group g is paired with
        # critic group g, and that the current critic uses the plain
        # (obs, action) -> obs_action_encoder -> joint_encoder -> scalar head
        # ParallelFC + GroupNorm/ReLU layout of the BAFC DMC configs.
        return (self._actor_critic_pairing and
                self._supports_fast_grad_metric_last_two() and
                self._supports_manual_critic_row_vjp() and
                getattr(self._critic_networks, 'state_spec', ()) == ())

    def _supports_actor_step_preupdate_grad_metric(self) -> bool:
        return self._supports_manual_actor_step_rows()

    def _should_compute_actor_step_preupdate_grad_metric(self) -> bool:
        return (self._monitor_trust_metrics and
                self._trust_metric_update_counter %
                self._trust_metric_update_interval == 0 and
                self._supports_actor_step_preupdate_grad_metric())

    def _should_refresh_trust_metrics(self):
        return (self._monitor_trust_metrics and self._last_update_had_actor_step
                and self._trust_metric_update_counter %
                self._trust_metric_update_interval == 0)

    def _get_manual_critic_parallel_fc_layers(self, critic_network=None):
        critic_network = self._critic_networks if critic_network is None else critic_network
        critic_core = getattr(critic_network, '_pnet', critic_network)
        fc_layers = [
            module for module in critic_core.modules()
            if isinstance(module, alf.layers.ParallelFC)
        ]
        expected = (self._manual_critic_obs_action_num_layers +
                    self._manual_critic_joint_num_layers)
        if len(fc_layers) != expected:
            raise RuntimeError(
                'Manual critic row-VJP expected %d ParallelFC layers but '
                'found %d.' % (expected, len(fc_layers)))
        obs_action_layers = fc_layers[:self._manual_critic_obs_action_num_layers]
        joint_layers = fc_layers[self._manual_critic_obs_action_num_layers:]
        if len(joint_layers) < 1:
            raise RuntimeError('Manual critic row-VJP requires at least one '
                               'joint layer plus a scalar head.')
        return obs_action_layers, joint_layers

    def _forward_parallel_fc_group(self, layer, group_idx: int,
                                   inputs: torch.Tensor):
        weight = layer.weight[group_idx].detach()
        bias = None if layer.bias is None else layer.bias[group_idx].detach()
        linear = inputs.matmul(weight.t())
        if bias is not None:
            linear = linear + bias
        cache = dict(
            input=inputs,
            weight=weight,
            bias=bias,
            use_ln=bool(getattr(layer, '_use_ln', False)))
        if cache['use_ln']:
            ln = layer._ln
            out_dim = weight.shape[0]
            gamma = ln.weight.detach().view(self._num_actor_critic,
                                            out_dim)[group_idx]
            if bias is None:
                beta = torch.zeros_like(gamma)
            else:
                beta = ln.bias.detach().view(self._num_actor_critic,
                                             out_dim)[group_idx]
            xhat, rstd = self._group_norm_stats(linear, ln.eps)
            pre_act = gamma.unsqueeze(0) * xhat + beta.unsqueeze(0)
            cache.update(gamma=gamma, xhat=xhat, rstd=rstd)
        else:
            pre_act = linear

        activation = getattr(layer, '_activation', None)
        if self._is_relu_activation(activation):
            outputs = torch.relu(pre_act)
            cache.update(is_relu=True, mask=pre_act.gt(0).to(pre_act.dtype))
        elif self._is_identity_activation(activation):
            outputs = pre_act
            cache.update(is_relu=False, mask=None)
        else:
            raise RuntimeError(
                'Manual critic row-VJP only supports ReLU/identity '
                'activations, got %s.' % (activation, ))
        return outputs, cache

    def _vjp_parallel_fc_group(self, layer_cache, grad_out: torch.Tensor):
        grad = grad_out
        if layer_cache['is_relu']:
            grad = grad * layer_cache['mask'].unsqueeze(0)
        if layer_cache['use_ln']:
            grad, _, _ = self._group_norm_vjp(grad, layer_cache['xhat'],
                                              layer_cache['rstd'],
                                              layer_cache['gamma'])
        grad_in = torch.einsum('kbo,oi->kbi', grad, layer_cache['weight'])
        return grad_in

    def _group_norm_vjp_all_groups(self, grad_out: torch.Tensor,
                                   xhat: torch.Tensor, rstd: torch.Tensor,
                                   gamma: torch.Tensor):
        """VJP for ParallelFC GroupNorm over all groups.

        Args:
            grad_out: Tensor [K, B, G, O].
            xhat: Tensor [B, G, O].
            rstd: Tensor [B, G, 1].
            gamma: Tensor [G, O].

        Returns:
            grad_in: Tensor [K, B, G, O].
            grad_gamma: Tensor [K, G, O].
            grad_beta: Tensor [K, G, O].
        """
        u = grad_out * gamma.unsqueeze(0).unsqueeze(0)
        xhat_view = xhat.unsqueeze(0)
        rstd_view = rstd.unsqueeze(0)
        mean_u = u.mean(dim=-1, keepdim=True)
        mean_uxhat = (u * xhat_view).mean(dim=-1, keepdim=True)
        grad_in = rstd_view * (u - mean_u - xhat_view * mean_uxhat)
        grad_gamma = (grad_out * xhat_view).sum(dim=1)
        grad_beta = grad_out.sum(dim=1)
        return grad_in, grad_gamma, grad_beta

    def _forward_parallel_fc_all_groups(self, layer, inputs: torch.Tensor):
        """Forward one ParallelFC layer and cache data for all groups.

        This mirrors ``alf.layers.ParallelFC.forward`` for tensor inputs with
        shape [B, G, I], but stores the affine/LN/ReLU intermediates required by
        the manual row-VJP. It removes the Python loop over groups from the
        previous implementation.
        """
        weight = layer.weight.detach()  # [G, O, I]
        bias = None if layer.bias is None else layer.bias.detach()  # [G, O]
        linear = torch.einsum('bgi,goi->bgo', inputs, weight)
        if bias is not None:
            linear = linear + bias.unsqueeze(0)

        cache = dict(
            input=inputs,
            weight=weight,
            bias=bias,
            use_ln=bool(getattr(layer, '_use_ln', False)))

        if cache['use_ln']:
            ln = layer._ln
            group_count = weight.shape[0]
            out_dim = weight.shape[1]
            gamma = ln.weight.detach().view(group_count, out_dim)
            if bias is None:
                beta = torch.zeros_like(gamma)
            else:
                beta = ln.bias.detach().view(group_count, out_dim)
            xhat, rstd = self._group_norm_stats(linear, ln.eps)
            pre_act = gamma.unsqueeze(0) * xhat + beta.unsqueeze(0)
            cache.update(gamma=gamma, xhat=xhat, rstd=rstd)
        else:
            pre_act = linear

        activation = getattr(layer, '_activation', None)
        if self._is_relu_activation(activation):
            outputs = torch.relu(pre_act)
            cache.update(is_relu=True, mask=pre_act.gt(0).to(pre_act.dtype))
        elif self._is_identity_activation(activation):
            outputs = pre_act
            cache.update(is_relu=False, mask=None)
        else:
            raise RuntimeError(
                'Manual critic row-VJP only supports ReLU/identity '
                'activations, got %s.' % (activation, ))
        return outputs, cache

    def _vjp_parallel_fc_all_groups(self, layer_cache, grad_out: torch.Tensor):
        """Manual VJP for one cached all-group ParallelFC layer.

        Args:
            grad_out: Tensor [K, B, G, O].

        Returns:
            Tensor [K, B, G, I].
        """
        grad = grad_out
        if layer_cache['is_relu']:
            grad = grad * layer_cache['mask'].unsqueeze(0)
        if layer_cache['use_ln']:
            grad, _, _ = self._group_norm_vjp_all_groups(
                grad, layer_cache['xhat'], layer_cache['rstd'],
                layer_cache['gamma'])
        return torch.einsum('kbgo,goi->kbgi', grad, layer_cache['weight'])

    def _manual_critic_all_group_rows(self,
                                      observation,
                                      actor_encoding_base,
                                      action,
                                      feature_coords=None,
                                      critic_network=None,
                                      obs_action_layers=None,
                                      joint_layers=None):
        """Manual multi-row critic VJP for all groups in one vectorized pass.

        The row set is:
            row 0: ``sum_n Q_{n,g}``;
            rows 1..R: ``mean_n phi_{n,g,j_r}``.

        It returns row upstreams to the actor action and actor encoding
        cutpoints. Critic weights are constants during the actor update.
        """
        if obs_action_layers is None or joint_layers is None:
            obs_action_layers, joint_layers = (
                self._get_manual_critic_parallel_fc_layers(critic_network))

        obs = observation.detach()
        action_detached = action.detach()
        batch_size = obs.shape[0]
        group_count = action_detached.shape[1]
        obs_g = obs.unsqueeze(1).expand(-1, group_count, -1)
        actor_u = actor_encoding_base.detach().unsqueeze(0).expand(
            batch_size, -1, -1)

        x = torch.cat([obs_g, action_detached], dim=-1)
        obs_action_caches = []
        for layer in obs_action_layers:
            x, cache = self._forward_parallel_fc_all_groups(layer, x)
            obs_action_caches.append(cache)
        obs_action_encoding = x

        x = torch.cat([actor_u, obs_action_encoding], dim=-1)
        joint_hidden_caches = []
        for layer in joint_layers[:-1]:
            x, cache = self._forward_parallel_fc_all_groups(layer, x)
            joint_hidden_caches.append(cache)
        phi = x
        feature_dim = phi.shape[-1]

        q, head_cache = self._forward_parallel_fc_all_groups(joint_layers[-1], phi)
        q = q.squeeze(-1)

        num_feature_rows = 0 if feature_coords is None else int(
            feature_coords.shape[0])
        num_rows = 1 + num_feature_rows
        grad_phi_rows = phi.new_zeros(
            (num_rows, batch_size, group_count, feature_dim))

        q_rows = phi.new_ones((1, batch_size, group_count, 1))
        grad_phi_rows[0:1] = self._vjp_parallel_fc_all_groups(head_cache, q_rows)

        if feature_coords is not None:
            scale = 1.0 / float(max(batch_size, 1))
            for row_offset, coord in enumerate(feature_coords.tolist(), start=1):
                grad_phi_rows[row_offset, :, :, int(coord)] = scale

        grad_joint_rows = grad_phi_rows
        for cache in reversed(joint_hidden_caches):
            grad_joint_rows = self._vjp_parallel_fc_all_groups(
                cache, grad_joint_rows)

        actor_encoding_dim = actor_encoding_base.shape[-1]
        grad_u_rows = grad_joint_rows[:, :, :, :actor_encoding_dim].sum(dim=1)
        grad_obs_action_rows = grad_joint_rows[:, :, :, actor_encoding_dim:]

        grad_input_rows = grad_obs_action_rows
        for cache in reversed(obs_action_caches):
            grad_input_rows = self._vjp_parallel_fc_all_groups(
                cache, grad_input_rows)

        obs_dim = observation.shape[-1]
        grad_action_rows = grad_input_rows[:, :, :, obs_dim:]

        result = dict(
            q_value=q,
            phi=phi,
            dqda=grad_action_rows[0],
            dqu=grad_u_rows[0])
        if feature_coords is not None:
            result.update(
                feature_action_rows=grad_action_rows[1:],
                feature_du_rows=grad_u_rows[1:])
        else:
            result.update(feature_action_rows=None, feature_du_rows=None)
        return result

    def _manual_critic_group_rows(self,
                                  observation,
                                  actor_encoding_base,
                                  action,
                                  group_idx: int,
                                  feature_coords=None,
                                  critic_network=None,
                                  obs_action_layers=None,
                                  joint_layers=None):
        if obs_action_layers is None or joint_layers is None:
            obs_action_layers, joint_layers = (
                self._get_manual_critic_parallel_fc_layers(critic_network))

        obs = observation.detach()
        action_g = action[:, group_idx, :].detach()
        actor_u = actor_encoding_base[group_idx].detach()
        actor_u = actor_u.unsqueeze(0).expand(obs.shape[0], -1)

        x = torch.cat([obs, action_g], dim=-1)
        obs_action_caches = []
        for layer in obs_action_layers:
            x, cache = self._forward_parallel_fc_group(layer, group_idx, x)
            obs_action_caches.append(cache)
        obs_action_encoding = x

        x = torch.cat([actor_u, obs_action_encoding], dim=-1)
        joint_hidden_caches = []
        for layer in joint_layers[:-1]:
            x, cache = self._forward_parallel_fc_group(layer, group_idx, x)
            joint_hidden_caches.append(cache)
        phi = x

        q, head_cache = self._forward_parallel_fc_group(
            joint_layers[-1], group_idx, phi)
        q = q.squeeze(-1)

        q_rows = torch.ones(
            1, q.shape[0], 1, dtype=q.dtype, device=q.device)
        grad_phi_rows = self._vjp_parallel_fc_group(head_cache, q_rows)

        if feature_coords is not None:
            num_rows = int(feature_coords.shape[0])
            feature_rows = phi.new_zeros((num_rows, phi.shape[0], phi.shape[1]))
            scale = 1.0 / float(max(phi.shape[0], 1))
            row_ids = torch.arange(num_rows, device=phi.device)
            feature_rows[row_ids, :, feature_coords] = scale
            grad_phi_rows = torch.cat([grad_phi_rows, feature_rows], dim=0)

        grad_joint_rows = grad_phi_rows
        for cache in reversed(joint_hidden_caches):
            grad_joint_rows = self._vjp_parallel_fc_group(cache, grad_joint_rows)

        actor_encoding_dim = actor_encoding_base.shape[-1]
        grad_u_rows = grad_joint_rows[:, :, :actor_encoding_dim].sum(dim=1)
        grad_obs_action_rows = grad_joint_rows[:, :, actor_encoding_dim:]

        grad_input_rows = grad_obs_action_rows
        for cache in reversed(obs_action_caches):
            grad_input_rows = self._vjp_parallel_fc_group(cache, grad_input_rows)

        obs_dim = observation.shape[-1]
        grad_action_rows = grad_input_rows[:, :, obs_dim:]

        result = dict(
            q_value=q,
            phi=phi,
            dqda=grad_action_rows[0],
            dqu=grad_u_rows[0])
        if feature_coords is not None:
            result.update(
                feature_action_rows=grad_action_rows[1:],
                feature_du_rows=grad_u_rows[1:])
        else:
            result.update(feature_action_rows=None, feature_du_rows=None)
        return result

    def _compute_manual_actor_step_rows(self,
                                        observation,
                                        action,
                                        obs_full,
                                        eval_full,
                                        compute_preupdate_grad_metric=False):
        eval_action = eval_full[-2:]
        eval_hidden = eval_action[0]
        eval_action_tensor = eval_action[1]
        actor_tokens = self._tokenize_actor_out(eval_action)
        actor_encoding_base = self._actor_encoder(actor_tokens)[0]

        obs_action_layers, joint_layers = self._get_manual_critic_parallel_fc_layers(
            self._critic_networks)
        feature_dim = joint_layers[-1].weight.shape[-1]
        feature_coords = None
        actor_cache = None
        if compute_preupdate_grad_metric:
            feature_coords = self._sample_feature_coords(feature_dim,
                                                        action.device)
            actor_cache = self._build_fast_actor_last_two_cache(obs_full,
                                                                eval_full)

        # One vectorized critic row pass for all groups. This replaces the
        # previous group-by-group manual critic VJP.
        rows = self._manual_critic_all_group_rows(
            observation,
            actor_encoding_base,
            action,
            feature_coords=feature_coords,
            obs_action_layers=obs_action_layers,
            joint_layers=joint_layers)

        dqda = rows['dqda']
        phi_t = rows['phi'] if compute_preupdate_grad_metric else None
        grad_sq_phi = None
        if compute_preupdate_grad_metric:
            grad_sq_phi = action.new_zeros(
                (self._num_actor_critic, feature_coords.shape[0]))

        up_u_rows_all = rows['dqu'].unsqueeze(0)
        if compute_preupdate_grad_metric:
            up_u_rows_all = torch.cat([up_u_rows_all, rows['feature_du_rows']],
                                      dim=0)

        dqde_hidden = eval_hidden.new_zeros(eval_hidden.shape)
        dqde_action = eval_action_tensor.new_zeros(eval_action_tensor.shape)

        # The actor encoder is a Transformer in the TR config. We still use a
        # batched VJP for this non-MLP component, but all critic feature rows are
        # supplied together for each group; there is no per-coordinate autograd
        # loop. Keeping the group loop avoids allocating a huge off-diagonal
        # [rows, eval_batch, all_groups, hidden] gradient tensor.
        for group_idx in range(self._num_actor_critic):
            up_u_rows = up_u_rows_all[:, group_idx, :]
            # Keep the graph for the final optimizer backward on actor_loss.
            # We only need first-order row values here (create_graph=False),
            # but freeing the graph in the last group would break the outer
            # ``loss.backward()`` with "backward through the graph a second time".
            keep_graph = True
            encoder_grads = torch.autograd.grad(
                actor_encoding_base[group_idx],
                [eval_hidden, eval_action_tensor],
                grad_outputs=up_u_rows,
                retain_graph=keep_graph,
                create_graph=False,
                allow_unused=True,
                is_grads_batched=True)

            hidden_rows = encoder_grads[0]
            if hidden_rows is None:
                hidden_rows = eval_hidden.new_zeros(
                    (up_u_rows.shape[0], ) + eval_hidden.shape)
            action_rows = encoder_grads[1]
            if action_rows is None:
                action_rows = eval_action_tensor.new_zeros(
                    (up_u_rows.shape[0], ) + eval_action_tensor.shape)

            dqde_hidden[:, group_idx, :] = hidden_rows[0, :, group_idx, :]
            dqde_action[:, group_idx, :] = action_rows[0, :, group_idx, :]

            if compute_preupdate_grad_metric:
                grad_sq_phi[group_idx] = (
                    self._explicit_actor_grad_sq_last_two_group(
                        actor_cache[group_idx],
                        rows['feature_action_rows'][:, :, group_idx, :],
                        hidden_rows[1:, :, group_idx, :],
                        action_rows[1:, :, group_idx, :]))

        preupdate_grad_trust = None
        if compute_preupdate_grad_metric and phi_t is not None:
            with torch.no_grad():
                if self._preupdate_grad_metric_anchor == 'reference':
                    ref_action = self._ensure_group_action(
                        self._reference_actor_networks(observation.detach())[0])
                    ref_encoding = self._compute_actor_encoding(
                        self._reference_actor_networks).detach()
                    phi_anchor = self._compute_snapshot_feature_map(
                        observation.detach(), ref_encoding, ref_action).detach()
                else:
                    phi_anchor = phi_t.detach()
                a_inv = self._compute_feature_inv_cov(phi_anchor)
                feature_norm = self._compute_weighted_feature_norm(
                    phi_t.detach(), a_inv).mean(dim=0)

            num_obs = observation.shape[0]
            action_dim = action.shape[-1]
            obs_action_basis = torch.eye(
                action_dim,
                device=action.device,
                dtype=action.dtype).unsqueeze(1).expand(action_dim, num_obs,
                                                        action_dim)
            obs_action_basis = obs_action_basis / float(max(num_obs, 1))
            grad_sq_mu = action.new_zeros((self._num_actor_critic, action_dim))
            for group_idx in range(self._num_actor_critic):
                grad_sq_mu[group_idx] = (
                    self._explicit_actor_grad_sq_last_two_group(
                        actor_cache[group_idx], obs_action_basis, None, None))

            jacobian_norm = torch.sqrt(
                torch.clamp(grad_sq_mu.sum(-1), min=0.) + 1e-12)
            c1 = torch.mean(jacobian_norm * feature_norm)

            a_inv_diag = torch.clamp(
                torch.diagonal(a_inv, dim1=-2, dim2=-1), min=0.)
            sampled_inv_diag = a_inv_diag[:, feature_coords]
            weighted_grad_sq = torch.clamp(sampled_inv_diag * grad_sq_phi,
                                           min=0.)
            coord_scale = float(feature_dim) / float(feature_coords.shape[0])
            c2 = torch.mean(
                torch.sqrt(
                    torch.clamp(coord_scale * weighted_grad_sq.sum(-1),
                                min=0.) + 1e-12))
            preupdate_grad_trust = torch.maximum(torch.clamp(c1, min=0.),
                                                 torch.clamp(c2, min=0.))

        # Return a list, not a tuple, because eval_action_in_graph is a list for
        # actor_eval_type='last_two' and alf.nest.map_structure requires matching
        # sequence types.
        return dqda, [dqde_hidden, dqde_action], preupdate_grad_trust

    def _action_scale_tensor(self, device, dtype):
        return 0.5 * (
            torch.as_tensor(self._action_spec.maximum, device=device, dtype=dtype)
            - torch.as_tensor(self._action_spec.minimum,
                              device=device,
                              dtype=dtype))

    def _group_norm_stats(self, linear_out: torch.Tensor, eps: float):
        mean = linear_out.mean(dim=-1, keepdim=True)
        var = (linear_out - mean).pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        xhat = (linear_out - mean) * rstd
        return xhat, rstd

    def _group_norm_vjp(self, grad_out: torch.Tensor, xhat: torch.Tensor,
                        rstd: torch.Tensor, gamma: torch.Tensor):
        gamma_view = gamma.view(1, 1, -1)
        xhat_view = xhat.unsqueeze(0)
        rstd_view = rstd.unsqueeze(0)
        u = grad_out * gamma_view
        mean_u = u.mean(dim=-1, keepdim=True)
        mean_uxhat = (u * xhat_view).mean(dim=-1, keepdim=True)
        grad_in = rstd_view * (u - mean_u - xhat_view * mean_uxhat)
        grad_gamma = (grad_out * xhat_view).sum(dim=1)
        grad_beta = grad_out.sum(dim=1)
        return grad_in, grad_gamma, grad_beta

    def _build_fast_actor_last_two_cache(self, obs_full, eval_full):
        actor_net = self._actor_networks
        hidden_layers = list(actor_net._fc_layers)
        num_hidden_layers = len(hidden_layers)
        expected_num_tensors = num_hidden_layers + 2
        if (len(obs_full) != expected_num_tensors
                or len(eval_full) != expected_num_tensors):
            raise RuntimeError(
                'Unexpected full_neurons layout for ActorFCNetwork: '
                f'expected {expected_num_tensors} tensors, got '
                f'{len(obs_full)} and {len(eval_full)}.')

        action_scale = self._action_scale_tensor(obs_full[-1].device,
                                                 obs_full[-1].dtype)
        group_caches = []
        action_layer = actor_net._action_layer
        for group_idx in range(self._num_actor_critic):
            hidden_caches = []
            for layer_idx, layer in enumerate(hidden_layers):
                weight = layer.weight[group_idx].detach()
                bias = (None if layer.bias is None else
                        layer.bias[group_idx].detach())
                obs_input = obs_full[layer_idx][:, group_idx, :].detach()
                eval_input = eval_full[layer_idx][:, group_idx, :].detach()
                obs_hidden = obs_full[layer_idx + 1][:, group_idx, :].detach()
                eval_hidden = eval_full[layer_idx + 1][:, group_idx, :].detach()
                obs_linear = obs_input.matmul(weight.t())
                eval_linear = eval_input.matmul(weight.t())
                if bias is not None:
                    obs_linear = obs_linear + bias
                    eval_linear = eval_linear + bias

                layer_cache = dict(
                    obs_input=obs_input,
                    eval_input=eval_input,
                    weight=weight,
                    bias=bias,
                    use_ln=bool(getattr(layer, '_use_ln', False)),
                    mask_obs=obs_hidden.gt(0).to(obs_hidden.dtype),
                    mask_eval=eval_hidden.gt(0).to(eval_hidden.dtype))
                if layer_cache['use_ln']:
                    out_dim = weight.shape[0]
                    ln = layer._ln
                    gamma = ln.weight.detach().view(self._num_actor_critic,
                                                    out_dim)[group_idx]
                    xhat_obs, rstd_obs = self._group_norm_stats(obs_linear,
                                                                ln.eps)
                    xhat_eval, rstd_eval = self._group_norm_stats(eval_linear,
                                                                  ln.eps)
                    layer_cache.update(
                        gamma=gamma,
                        xhat_obs=xhat_obs,
                        xhat_eval=xhat_eval,
                        rstd_obs=rstd_obs,
                        rstd_eval=rstd_eval)
                hidden_caches.append(layer_cache)

            action_weight = action_layer.weight[group_idx].detach()
            action_bias = (None if action_layer.bias is None else
                           action_layer.bias[group_idx].detach())
            obs_action_input = obs_full[-2][:, group_idx, :].detach()
            eval_action_input = eval_full[-2][:, group_idx, :].detach()
            obs_pre_action = obs_action_input.matmul(action_weight.t())
            eval_pre_action = eval_action_input.matmul(action_weight.t())
            if action_bias is not None:
                obs_pre_action = obs_pre_action + action_bias
                eval_pre_action = eval_pre_action + action_bias
            action_cache = dict(
                obs_input=obs_action_input,
                eval_input=eval_action_input,
                weight=action_weight,
                bias=action_bias,
                local_deriv_obs=action_scale *
                (1. - torch.tanh(obs_pre_action).pow(2)),
                local_deriv_eval=action_scale *
                (1. - torch.tanh(eval_pre_action).pow(2)))
            group_caches.append(
                dict(hidden_layers=hidden_caches, action=action_cache))
        return group_caches

    def _explicit_actor_grad_sq_last_two_group(self,
                                               group_cache,
                                               up_obs_action: Optional[
                                                   torch.Tensor],
                                               up_eval_hidden: Optional[
                                                   torch.Tensor],
                                               up_eval_action: Optional[
                                                   torch.Tensor]):
        action_cache = group_cache['action']
        if up_obs_action is not None:
            output_dim = up_obs_action.shape[0]
            device = up_obs_action.device
            dtype = up_obs_action.dtype
        elif up_eval_hidden is not None:
            output_dim = up_eval_hidden.shape[0]
            device = up_eval_hidden.device
            dtype = up_eval_hidden.dtype
        elif up_eval_action is not None:
            output_dim = up_eval_action.shape[0]
            device = up_eval_action.device
            dtype = up_eval_action.dtype
        else:
            return action_cache['weight'].new_zeros((0, ))

        obs_count = action_cache['obs_input'].shape[0]
        eval_count = action_cache['eval_input'].shape[0]
        action_dim, hidden_dim = action_cache['weight'].shape
        if up_obs_action is None:
            up_obs_action = torch.zeros(
                output_dim,
                obs_count,
                action_dim,
                device=device,
                dtype=dtype)
        if up_eval_action is None:
            up_eval_action = torch.zeros(
                output_dim,
                eval_count,
                action_dim,
                device=device,
                dtype=dtype)
        if up_eval_hidden is None:
            up_eval_hidden = torch.zeros(
                output_dim,
                eval_count,
                hidden_dim,
                device=device,
                dtype=dtype)

        grad_sq = torch.zeros(output_dim, device=device, dtype=dtype)
        delta_obs = up_obs_action * action_cache['local_deriv_obs'].unsqueeze(0)
        delta_eval = (up_eval_action *
                      action_cache['local_deriv_eval'].unsqueeze(0))
        grad_w = torch.einsum('dno,ni->doi', delta_obs,
                              action_cache['obs_input'])
        grad_w = grad_w + torch.einsum('dmo,mi->doi', delta_eval,
                                       action_cache['eval_input'])
        grad_sq = grad_sq + grad_w.pow(2).sum(dim=(-1, -2))
        if action_cache['bias'] is not None:
            grad_b = delta_obs.sum(dim=1) + delta_eval.sum(dim=1)
            grad_sq = grad_sq + grad_b.pow(2).sum(dim=-1)

        up_obs = torch.einsum('dno,oi->dni', delta_obs, action_cache['weight'])
        up_eval = torch.einsum('dmo,oi->dmi', delta_eval,
                               action_cache['weight']) + up_eval_hidden

        for layer_cache in reversed(group_cache['hidden_layers']):
            grad_ln_out_obs = up_obs * layer_cache['mask_obs'].unsqueeze(0)
            grad_ln_out_eval = up_eval * layer_cache['mask_eval'].unsqueeze(0)
            if layer_cache['use_ln']:
                delta_obs, grad_gamma_obs, grad_beta_obs = self._group_norm_vjp(
                    grad_ln_out_obs, layer_cache['xhat_obs'],
                    layer_cache['rstd_obs'], layer_cache['gamma'])
                delta_eval, grad_gamma_eval, grad_beta_eval = self._group_norm_vjp(
                    grad_ln_out_eval, layer_cache['xhat_eval'],
                    layer_cache['rstd_eval'], layer_cache['gamma'])
                grad_gamma = grad_gamma_obs + grad_gamma_eval
                grad_beta = grad_beta_obs + grad_beta_eval
                grad_sq = grad_sq + grad_gamma.pow(2).sum(dim=-1)
                grad_sq = grad_sq + grad_beta.pow(2).sum(dim=-1)
            else:
                delta_obs = grad_ln_out_obs
                delta_eval = grad_ln_out_eval

            grad_w = torch.einsum('dnh,ni->dhi', delta_obs,
                                  layer_cache['obs_input'])
            grad_w = grad_w + torch.einsum('dmh,mi->dhi', delta_eval,
                                           layer_cache['eval_input'])
            grad_sq = grad_sq + grad_w.pow(2).sum(dim=(-1, -2))
            if layer_cache['bias'] is not None:
                grad_b = delta_obs.sum(dim=1) + delta_eval.sum(dim=1)
                grad_sq = grad_sq + grad_b.pow(2).sum(dim=-1)

            up_obs = torch.einsum('dnh,hi->dni', delta_obs,
                                  layer_cache['weight'])
            up_eval = torch.einsum('dmh,hi->dmi', delta_eval,
                                   layer_cache['weight'])

        return grad_sq

    def _batched_cutpoint_output_grad_sq_norm_fast(self,
                                                   output_means: torch.Tensor,
                                                   cutpoint_inputs,
                                                   actor_cache,
                                                   retain_graph: bool,
                                                   max_chunk_size:
                                                   Optional[int] = None):
        if not isinstance(output_means, torch.Tensor):
            return self._actor_eval_samples.new_zeros((0, 0))
        if output_means.ndim != 2:
            raise ValueError(
                f'Expected output_means with shape [G, D], got '
                f'{tuple(output_means.shape)}')
        if not output_means.requires_grad:
            return output_means.new_zeros(output_means.shape)

        num_groups, output_dim = output_means.shape
        flat_outputs = output_means.reshape(-1)
        num_outputs = flat_outputs.shape[0]
        if num_outputs == 0:
            return output_means.new_zeros(output_means.shape)

        start_chunk = min(64, num_outputs)
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
                        cutpoint_inputs,
                        grad_outputs=basis,
                        retain_graph=keep_graph,
                        create_graph=False,
                        allow_unused=True,
                        is_grads_batched=True)

                    chunk_group_idx = flat_group_idx[start:end]
                    chunk_grad_sq = flat_outputs.new_zeros(chunk_len)
                    for group_idx in range(num_groups):
                        rows = torch.nonzero(
                            chunk_group_idx == group_idx,
                            as_tuple=False).squeeze(-1)
                        if rows.numel() == 0:
                            continue

                        up_obs_action = None
                        if batched_grads[0] is not None:
                            up_obs_action = batched_grads[0].index_select(
                                0, rows)[:, :, group_idx, :]

                        up_eval_hidden = None
                        if batched_grads[1] is not None:
                            up_eval_hidden = batched_grads[1].index_select(
                                0, rows)[:, :, group_idx, :]

                        up_eval_action = None
                        if batched_grads[2] is not None:
                            up_eval_action = batched_grads[2].index_select(
                                0, rows)[:, :, group_idx, :]

                        group_grad_sq = self._explicit_actor_grad_sq_last_two_group(
                            actor_cache[group_idx], up_obs_action,
                            up_eval_hidden, up_eval_action)
                        chunk_grad_sq.index_copy_(0, rows, group_grad_sq)

                    grad_sq[start:end] = chunk_grad_sq
                    start = end
                return grad_sq.view(num_groups, output_dim)
            except RuntimeError:
                if chunk_size == 1:
                    raise
                next_chunk = max(1, chunk_size // 2)
                logging.warning(
                    'Fast cutpoint batched VJP for grad-trust failed at '
                    'chunk_size=%d; retrying with chunk_size=%d.', chunk_size,
                    next_chunk)
                chunk_size = next_chunk

    def _compute_grad_generalization_trust_components_fast(self, observation):
        obs = self._sample_metric_observations(observation)
        if not isinstance(obs, torch.Tensor):
            one = torch.ones_like(self._last_eval_trust)
            return one, one

        obs = obs.detach()
        with torch.no_grad():
            obs_full = self._actor_networks(obs, full_neurons=True)[0]
            eval_full = self._actor_networks(
                self._actor_eval_samples, full_neurons=True)[0]
            ref_action = self._ensure_group_action(
                self._reference_actor_networks(obs)[0]).detach()
            ref_encoding = self._compute_actor_encoding(
                self._reference_actor_networks).detach()
            actor_cache = self._build_fast_actor_last_two_cache(obs_full,
                                                                eval_full)

        cur_action = obs_full[-1].detach().requires_grad_(True)
        eval_hidden = eval_full[-2].detach().requires_grad_(True)
        eval_action = eval_full[-1].detach().requires_grad_(True)
        actor_tokens = self._tokenize_actor_out([eval_hidden, eval_action])
        cur_encoding = self._actor_encoder(actor_tokens)[0]

        phi_ref = self._compute_snapshot_feature_map(obs, ref_encoding,
                                                     ref_action).detach()
        phi_t = self._compute_snapshot_feature_map(obs, cur_encoding, cur_action)
        a_inv = self._compute_feature_inv_cov(phi_ref)
        feature_norm = self._compute_weighted_feature_norm(phi_t, a_inv).mean(
            dim=0)

        num_obs = obs.shape[0]
        num_groups = cur_action.shape[1]
        action_dim = cur_action.shape[-1]
        obs_action_basis = torch.eye(
            action_dim,
            device=cur_action.device,
            dtype=cur_action.dtype).unsqueeze(1).expand(action_dim, num_obs,
                                                        action_dim)
        obs_action_basis = obs_action_basis / float(max(num_obs, 1))
        grad_sq_mu = cur_action.new_zeros((num_groups, action_dim))
        for group_idx in range(num_groups):
            grad_sq_mu[group_idx] = self._explicit_actor_grad_sq_last_two_group(
                actor_cache[group_idx], obs_action_basis, None, None)

        jacobian_norm = torch.sqrt(
            torch.clamp(grad_sq_mu.sum(-1), min=0.) + 1e-12)
        c1 = torch.mean(jacobian_norm * feature_norm)

        feature_dim = phi_t.shape[-1]
        feature_coords = self._sample_feature_coords(feature_dim, phi_t.device)
        a_inv_diag = torch.clamp(
            torch.diagonal(a_inv, dim1=-2, dim2=-1), min=0.)
        sampled_inv_diag = a_inv_diag[:, feature_coords]
        phi_coord_mean = phi_t[:, :, feature_coords].mean(dim=0)
        grad_sq_phi = self._batched_cutpoint_output_grad_sq_norm_fast(
            phi_coord_mean, [cur_action, eval_hidden, eval_action], actor_cache,
            retain_graph=False)

        weighted_grad_sq = torch.clamp(sampled_inv_diag * grad_sq_phi, min=0.)
        coord_scale = float(feature_dim) / float(feature_coords.shape[0])
        c2 = torch.mean(
            torch.sqrt(
                torch.clamp(coord_scale * weighted_grad_sq.sum(-1), min=0.) +
                1e-12))
        return torch.clamp(c1, min=0.), torch.clamp(c2, min=0.)

    def _compute_preupdate_grad_generalization_trust_from_actor_step(
            self, observation, cur_action, eval_hidden, eval_action,
            actor_cache, phi_t):
        """Compute the pre-update grad trust metric from actor-step caches.

        The covariance anchor is the pre-update critic feature map itself. This
        intentionally changes the semantics from the post-update/reference-actor
        metric: it is a cheaper, one-step-stale proxy used for gating/logging.
        The expensive actor-parameter VJPs are replaced by the explicit actor
        Jacobian-norm kernel, while autograd is used only for VJPs from sampled
        critic feature coordinates to the small cutpoints
        ``(cur_action, eval_hidden, eval_action)``.
        """
        if phi_t is None:
            return None
        with torch.no_grad():
            if self._preupdate_grad_metric_anchor == 'reference':
                ref_action = self._ensure_group_action(
                    self._reference_actor_networks(observation.detach())[0])
                ref_encoding = self._compute_actor_encoding(
                    self._reference_actor_networks).detach()
                phi_anchor = self._compute_snapshot_feature_map(
                    observation.detach(), ref_encoding, ref_action).detach()
            else:
                phi_anchor = phi_t.detach()
            a_inv = self._compute_feature_inv_cov(phi_anchor)
            feature_norm = self._compute_weighted_feature_norm(
                phi_t.detach(), a_inv).mean(dim=0)

        num_obs = observation.shape[0]
        num_groups = cur_action.shape[1]
        action_dim = cur_action.shape[-1]
        obs_action_basis = torch.eye(
            action_dim,
            device=cur_action.device,
            dtype=cur_action.dtype).unsqueeze(1).expand(action_dim, num_obs,
                                                        action_dim)
        obs_action_basis = obs_action_basis / float(max(num_obs, 1))
        grad_sq_mu = cur_action.new_zeros((num_groups, action_dim))
        for group_idx in range(num_groups):
            grad_sq_mu[group_idx] = self._explicit_actor_grad_sq_last_two_group(
                actor_cache[group_idx], obs_action_basis, None, None)

        jacobian_norm = torch.sqrt(
            torch.clamp(grad_sq_mu.sum(-1), min=0.) + 1e-12)
        c1 = torch.mean(jacobian_norm * feature_norm)

        feature_dim = phi_t.shape[-1]
        feature_coords = self._sample_feature_coords(feature_dim, phi_t.device)
        a_inv_diag = torch.clamp(
            torch.diagonal(a_inv, dim1=-2, dim2=-1), min=0.)
        sampled_inv_diag = a_inv_diag[:, feature_coords]
        phi_coord_mean = phi_t[:, :, feature_coords].mean(dim=0)
        grad_sq_phi = self._batched_cutpoint_output_grad_sq_norm_fast(
            phi_coord_mean, [cur_action, eval_hidden, eval_action], actor_cache,
            # Preserve the actor graph for the outer training backward.
            retain_graph=True)
        weighted_grad_sq = torch.clamp(sampled_inv_diag * grad_sq_phi, min=0.)
        coord_scale = float(feature_dim) / float(feature_coords.shape[0])
        c2 = torch.mean(
            torch.sqrt(
                torch.clamp(coord_scale * weighted_grad_sq.sum(-1), min=0.) +
                1e-12))
        return torch.maximum(torch.clamp(c1, min=0.), torch.clamp(c2, min=0.))

    def _compute_grad_generalization_trust_components(self, observation):
        if self._supports_fast_grad_metric_last_two():
            return self._compute_grad_generalization_trust_components_fast(
                observation)

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

            action_dim = cur_action.shape[-1]
            num_groups = cur_action.shape[1]
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

        rollout_actor_networks = self._actor_networks
        if self._enable_eval_rollout_skip_gate:
            rollout_actor_networks = self._behavior_actor_networks
        action, action_state = self._predict_action(
            rollout_actor_networks,
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

    def _actor_train_step(self,
                          observation,
                          action,
                          mask,
                          state,
                          obs_full=None,
                          compute_preupdate_grad_metric=False):
        """Compute the exact off-policy policy gradient from the functional critic.

        Fast path:
            - manual multi-row backward through the current critic MLP to obtain
              ``dQ/da`` and ``dQ/du`` together with selected critic-penultimate
              feature-coordinate rows in one pass;
            - one grouped VJP through the actor encoder from ``u`` back to the
              eval cutpoints ``(eval_hidden, eval_action)``;
            - explicit actor Jacobian-norm contractions for the pre-update
              grad-trust cache.

        Fallback:
            - original autograd implementation.
        """
        use_preupdate_metric = (compute_preupdate_grad_metric and
                                self._supports_actor_step_preupdate_grad_metric())
        use_manual_rows = self._supports_manual_actor_step_rows()
        preupdate_grad_trust = None

        ## Step 0: optionally reuse a full-neuron actor forward for observations.
        if use_preupdate_metric:
            if obs_full is None:
                obs_full = self._actor_networks(
                    observation, full_neurons=True)[0]
            action = obs_full[-1]

        ## Step 1: encode all actors from actor_eval_samples
        ####################################################
        eval_full = self._actor_networks(
            self._actor_eval_samples,
            full_neurons=self._actor_eval_type != 'output')[0]
        eval_action = eval_full
        if self._actor_eval_type == 'exclude_input':
            eval_action = eval_action[1:]
        elif self._actor_eval_type == 'last_two':
            eval_action = eval_action[-2:]

        # Fast manual critic-row path. This path is only enabled for the
        # supported BAFC DMC architecture and when actor/critic pairing is fixed.
        if use_manual_rows and self._actor_eval_type == 'last_two':
            try:
                dqda, dqde, preupdate_grad_trust = (
                    self._compute_manual_actor_step_rows(
                        observation,
                        action,
                        obs_full,
                        eval_full,
                        compute_preupdate_grad_metric=use_preupdate_metric))
                critic_state = state
                eval_action_in_graph = eval_action
                if preupdate_grad_trust is not None:
                    self._pending_preupdate_grad_trust = (
                        preupdate_grad_trust.detach())

                def action_loss_fn(dqda, a_in):
                    if self._dqda_clipping:
                        dqda = torch.clamp(dqda, -self._dqda_clipping,
                                           self._dqda_clipping)
                    loss = 0.5 * losses.element_wise_squared_loss(
                        (dqda + a_in).detach(), a_in)
                    return loss.sum(list(range(2, loss.ndim)))

                action_loss = nest.map_structure(action_loss_fn, dqda, action)
                if self._use_bootstrap_actors:
                    action_loss = action_loss * mask / self._bootstrap_mask_prob
                action_loss = action_loss.sum(-1)

                eval_action_loss = nest.map_structure(
                    action_loss_fn, dqde, eval_action_in_graph)
                eval_action_loss = math_ops.add_n(eval_action_loss).mean().repeat(
                    action_loss.shape[0])

                grad_metric_for_info = (preupdate_grad_trust
                                        if preupdate_grad_trust is not None else
                                        self._last_grad_trust)
                actor_info = LossInfo(
                    loss=action_loss,
                    extra=BafcActorInfo(
                        eval_action_loss=eval_action_loss,
                        grad_trust_metric=self._broadcast_trust_metric(
                            grad_metric_for_info,
                            action_loss.shape,
                            action_loss.device,
                            action_loss.dtype)))
                return critic_state, actor_info
            except RuntimeError as err:
                logging.warning(
                    'Manual actor-step critic row-VJP failed; recomputing a '
                    'fresh graph for the original autograd actor step. '
                    'The pre-update cache is disabled for this update. '
                    'Error: %s', err)
                # A failed manual attempt may already have consumed part of the
                # actor-encoder graph. Do not reuse the same action/eval tensors
                # for fallback; rebuild a clean graph and let after_update() use
                # the original post-update metric path if a metric is due.
                use_preupdate_metric = False
                preupdate_grad_trust = None
                self._pending_preupdate_grad_trust = None
                obs_full = None
                action = self._actor_networks(observation)[0]
                eval_full = self._actor_networks(
                    self._actor_eval_samples,
                    full_neurons=self._actor_eval_type != 'output')[0]
                eval_action = eval_full
                if self._actor_eval_type == 'exclude_input':
                    eval_action = eval_action[1:]
                elif self._actor_eval_type == 'last_two':
                    eval_action = eval_action[-2:]

        ## Fallback: original autograd-based actor step
        ##############################################
        actor_tokens = self._tokenize_actor_out(eval_action)
        actor_encoding = self._actor_encoder(actor_tokens)[0]
        if not self._actor_critic_pairing:
            perm = torch.randperm(self._num_actor_critic)
            actor_encoding = actor_encoding[perm, :]
            action = action[:, perm, :]
        actor_encoding = actor_encoding.unsqueeze(0).repeat(
            observation.shape[0], 1, 1)  # [T*B, n_actor, d_enc]

        critic_observation = observation.unsqueeze(1).repeat(
            1, self._num_actor_critic, 1)
        if use_preupdate_metric:
            actor_cache = self._build_fast_actor_last_two_cache(obs_full,
                                                                eval_full)
            q_value, critic_state, phi_t = self._compute_critic_value_and_feature_map(
                observation, actor_encoding, action, state,
                critic_network=self._critic_networks)
            if phi_t is None:
                q_value, critic_state = self._critic_networks(
                    (actor_encoding, (critic_observation, action)), state)
        else:
            q_value, critic_state = self._critic_networks(
                (actor_encoding, (critic_observation, action)), state)

        dqda = nest_utils.grad(action, q_value.sum(), retain_graph=True)
        if self._actor_eval_type == 'full':
            eval_action_in_graph = eval_action[1:]
        else:
            eval_action_in_graph = eval_action
        dqde = nest_utils.grad(
            eval_action_in_graph,
            q_value.sum(),
            retain_graph=(use_preupdate_metric or
                          self._actor_eval_type != 'output'))

        if use_preupdate_metric and phi_t is not None:
            try:
                preupdate_grad_trust = (
                    self._compute_preupdate_grad_generalization_trust_from_actor_step(
                        observation, action, eval_action_in_graph[0],
                        eval_action_in_graph[1], actor_cache, phi_t))
            except RuntimeError as err:
                logging.warning(
                    'Pre-update fast grad-trust computation failed; falling '
                    'back to the after_update metric. Error: %s', err)
                preupdate_grad_trust = None
            if preupdate_grad_trust is not None:
                self._pending_preupdate_grad_trust = preupdate_grad_trust.detach()

        def action_loss_fn(dqda, a_in):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + a_in).detach(), a_in)
            return loss.sum(list(range(2, loss.ndim)))

        action_loss = nest.map_structure(action_loss_fn, dqda, action)
        if self._use_bootstrap_actors:
            action_loss = action_loss * mask / self._bootstrap_mask_prob
        action_loss = action_loss.sum(-1)

        eval_action_loss = nest.map_structure(
            action_loss_fn, dqde, eval_action_in_graph)
        eval_action_loss = math_ops.add_n(eval_action_loss).mean().repeat(
            action_loss.shape[0])

        grad_metric_for_info = (preupdate_grad_trust if preupdate_grad_trust
                                is not None else self._last_grad_trust)
        actor_info = LossInfo(
            loss=action_loss,
            extra=BafcActorInfo(
                eval_action_loss=eval_action_loss,
                grad_trust_metric=self._broadcast_trust_metric(
                    grad_metric_for_info,
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
                    self._last_grad_gate_actor_extended = True
                    self._grad_gate_actor_extension_count += 1
                    self._grad_gate_consecutive_actor_extensions += 1
                else:
                    self._grad_gate_consecutive_actor_extensions = 0
                    self._train_mode = TrainMode.critic
                    self._completed_cycles_since_rollout += 1
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
        self._last_update_had_actor_step = False

        compute_preupdate_grad_metric = (
            (self._train_mode == TrainMode.actor or
             self._train_mode == TrainMode.standard or
             (self._critic_update_counter == 0 and
              self._actor_update_counter == 0)) and
            self._should_compute_actor_step_preupdate_grad_metric())
        obs_full = None
        if compute_preupdate_grad_metric:
            obs_full, actor_state_raw = self._actor_networks(
                inputs.observation,
                full_neurons=True,
                state=state.action.actor_network)
            action = obs_full[-1]
            action_state = BafcActionState(actor_network=actor_state_raw)
        else:
            # [T*B, n_actor, d_a]
            action, action_state = self._predict_action(
                self._actor_networks, inputs.observation,
                state=state.action, train=True)

        if self._train_mode == TrainMode.standard or (
                self._critic_update_counter == 0
                and self._actor_update_counter == 0):
            actor_action = action  # [T*B, n_actor, d_a]
            actor_state, actor_info = self._actor_train_step(
                inputs.observation,
                actor_action,
                rollout_info.bootstrap_mask,
                state.actor,
                obs_full=obs_full,
                compute_preupdate_grad_metric=compute_preupdate_grad_metric)
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
                    inputs.observation,
                    action,
                    rollout_info.bootstrap_mask,
                    state.actor,
                    obs_full=obs_full,
                    compute_preupdate_grad_metric=compute_preupdate_grad_metric)
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
            '_behavior_actor_networks', '_snapshot_critic_networks'
        ]

    def _unroll_iter_off_policy(self):
        """Gate rollout cadence by completed actor->critic cycles.

        Keep replay update budget unchanged by using the base
        ``_train_iter_off_policy()`` and only controlling whether unroll is
        performed for a given outer iteration.
        """
        # Preserve the default behavior during warmup/initial collection and
        # for standard (non-alternating) training mode.
        if not self._training_started or self._train_mode == TrainMode.standard:
            return super()._unroll_iter_off_policy()

        # Skip rollout until enough completed actor->critic cycles have
        # accumulated since the previous actual rollout.
        if self._completed_cycles_since_rollout < self._rollout_cycles_per_collect:
            return False, None, None

        unrolled, root_inputs, rollout_info = super()._unroll_iter_off_policy()
        if unrolled:
            self._completed_cycles_since_rollout = 0
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
                if self._pending_preupdate_grad_trust is not None:
                    self._last_grad_trust = (
                        self._pending_preupdate_grad_trust.detach())
                else:
                    self._last_grad_trust = self._compute_grad_generalization_trust_metric(
                        observation).detach()
            else:
                self._last_eval_trust = torch.ones_like(self._last_eval_trust)
                self._last_grad_trust = torch.ones_like(self._last_grad_trust)
        self._pending_preupdate_grad_trust = None
        if self._last_update_had_actor_step:
            self._trust_metric_update_counter += 1

        self._sync_reference_from_current()
        self._update_rollout_actor_from_eval_gate()
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
            'eval_gate_consecutive_rollout_actor_holds',
            float(self._eval_gate_consecutive_rollout_actor_holds))
        self._record_debug_scalar(
            'rollout_actor_held_due_eval_gate',
            float(self._last_rollout_actor_held_due_eval_gate))
        self._record_debug_scalar(
            'rollout_actor_hold_due_eval_gate_count',
            float(self._rollout_actor_hold_due_eval_gate_count))
        self._record_debug_scalar(
            'rollout_actor_refreshed_from_reference',
            float(self._last_rollout_actor_refreshed_from_reference))
        self._record_debug_scalar(
            'rollout_actor_refresh_forced_by_eval_gate_cap',
            float(self._last_rollout_actor_refresh_forced_by_eval_gate_cap))
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
