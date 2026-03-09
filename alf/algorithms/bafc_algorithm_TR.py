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
"""Trust-regulated Bootstrapped Actor and Functional Critic algorithm."""

import functools
from enum import Enum

import torch
import torch.nn as nn
from typing import Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, math_ops
from alf.utils.schedulers import Scheduler
from alf.utils.summary_utils import safe_mean_hist_summary, record_time


class _Phase(Enum):
    EVAL = "eval"
    IMPROVE = "improve"


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
    "BafcInfo",
    [
        "reward", "step_type", "discount", "observation", "action", "actor",
        "critic", "bootstrap_mask", "phase", "epoch_idx", "epoch_reset_flag",
        "eval_trust_metric", "grad_trust_metric"
    ],
    default_value=())

BafcLossInfo = namedtuple(
    "BafcLossInfo", ("actor", "critic"), default_value=())


@alf.configurable
class BafcAlgorithmTR(OnPolicyAlgorithm):
    r"""Algorithm 3-style BAFC with trust-regulated epoch control.

    This implementation adds explicit policy-evaluation and policy-improvement
    phases, plus the evaluation trust metric and gradient generalization trust
    metric gates described in Algorithm 3 of the paper.
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
                 eval_trust_max: float = 2.0,
                 delta_trust_max: float = 2.0,
                 policy_eval_updates_per_epoch: int = 1,
                 max_improve_steps_per_epoch: int = 8,
                 trust_cov_reg: float = 1e-4,
                 trust_metric_num_obs: int = 128,
                 on_policy_adaptation: bool = True,
                 update_critic_in_improve: bool = False,
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
                 name="BafcAlgorithmTR"):
        """
        Args:
            actor_critic_pairing (bool): whether or not fix the 1-1 pairing of
                actors and critics during actor training.
            bootstrap_mask_type (str): bootstrap mask sampling granularity.
            eval_trust_max (float): threshold C_max for evaluation trust metric.
            delta_trust_max (float): threshold for gradient trust metric Delta.
            policy_eval_updates_per_epoch (int): critic-only updates per epoch.
            max_improve_steps_per_epoch (int): max actor updates per epoch.
            trust_cov_reg (float): covariance regularization for trust metrics.
            trust_metric_num_obs (int): max number of observations used to
                estimate the Eq. (3.4)-style gradient trust metric each update.
            on_policy_adaptation (bool): if True, set behavior policy equal to
                reference policy at each epoch start so that the rollout policy
                stays fixed for the full epoch.
            update_critic_in_improve (bool): if True, also update critic during
                improvement phase. This is kept only as a non-Remark-3 variant;
                the on-policy adaptation path expects actor-only updates during
                improvement.
        """
        del calculate_priority, parameter_reset_period, reproduce_locomotion

        assert actor_eval_type in ['full', 'exclude_input', 'last_two', 'output'], (
            r"{actor_eval_type} in not supported.")
        assert eval_samples_init_method in ['normal', 'uniform'], (
            r"init method {eval_samples_init_method} is not supported.")
        assert bootstrap_mask_type in ['episode', 'step'], (
            r"bootstrap mask type {bootstrap_mask_type} is not supported.")
        assert policy_eval_updates_per_epoch >= 1, (
            "policy_eval_updates_per_epoch must be >= 1")
        assert max_improve_steps_per_epoch >= 1, (
            "max_improve_steps_per_epoch must be >= 1")
        assert trust_cov_reg > 0, "trust_cov_reg must be > 0"
        assert trust_metric_num_obs >= 1, "trust_metric_num_obs must be >= 1"
        assert not (on_policy_adaptation and update_critic_in_improve), (
            "update_critic_in_improve=True deviates from the Remark-3 "
            "on-policy adaptation path; disable it or turn off "
            "on_policy_adaptation.")

        self._num_actor_critic = num_actor_critic
        self._actor_critic_pairing = actor_critic_pairing
        self._use_bootstrap_actors = use_bootstrap_actors
        self._use_bootstrap_critics = use_bootstrap_critics
        self._bootstrap_mask_prob = bootstrap_mask_prob
        self._bootstrap_mask_type = bootstrap_mask_type
        self._bootstrap_mask = ()
        self._actor_use_ln = actor_use_ln
        self._dqda_clipping = dqda_clipping

        self._eval_trust_max = eval_trust_max
        self._delta_trust_max = delta_trust_max
        self._policy_eval_updates_per_epoch = policy_eval_updates_per_epoch
        self._max_improve_steps_per_epoch = max_improve_steps_per_epoch
        self._trust_cov_reg = trust_cov_reg
        self._trust_metric_num_obs = trust_metric_num_obs
        self._on_policy_adaptation = on_policy_adaptation
        self._update_critic_in_improve = update_critic_in_improve

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

        if actor_eval_type == 'full':
            actor_token_length = observation_spec.shape[0] + sum(
                t.shape[1] for t in actor_networks.bias_params)
        elif actor_eval_type == 'exclude_input':
            actor_token_length = sum(t.shape[1] for t in actor_networks.bias_params)
        elif actor_eval_type == 'last_two':
            actor_token_length = sum(t.shape[1] for t in actor_networks.bias_params[-2:])
        else:
            actor_token_length = action_spec.shape[0]

        actor_token_spec = TensorSpec(
            shape=(actor_token_length, num_actor_eval_samples))
        actor_encoder = actor_encoder_cls(
            actor_token_spec, core_embedding_dim=actor_encoding_dim)

        if actor_encoding_dim is None:
            actor_encoding_dim = num_actor_eval_samples
        actor_spec = TensorSpec(shape=(actor_encoding_dim, ))
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
        self._behavior_actor_networks = actor_networks.copy(
            name='behavior_actor_networks')
        for p in self._reference_actor_networks.parameters():
            p.requires_grad_(False)
        for p in self._behavior_actor_networks.parameters():
            p.requires_grad_(False)

        self._actor_eval_type = actor_eval_type
        self._actor_encoder = actor_encoder
        self._critic_networks = critic_networks
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        self._critic_losses = []
        for i in range(num_actor_critic):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._rollout_actor_id = 0
        self._training_started = False
        self._do_critic_summary = False

        self._phase = _Phase.EVAL
        self._epoch_idx = 0
        self._eval_step_idx = 0
        self._improve_step_idx = 0
        self._pending_epoch_refresh = False
        self._epoch_reset_flag = False
        self._last_eval_trust = torch.tensor(0.0)
        self._last_grad_trust = torch.tensor(1.0)

        def _filter(x):
            return list(filter(lambda y: y is not None, x))

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

        self._start_new_epoch(increment_epoch=False)

    def _phase_id(self):
        return 0 if self._phase == _Phase.EVAL else 1

    def _set_phase(self, phase: _Phase):
        self._phase = phase
        if phase == _Phase.EVAL:
            for p in self._actor_networks.parameters():
                p.requires_grad_(False)
            for p in self._critic_networks.parameters():
                p.requires_grad_(True)
            for p in self._actor_encoder.parameters():
                p.requires_grad_(True)
            self._actor_eval_samples.requires_grad_(True)
        else:
            for p in self._actor_networks.parameters():
                p.requires_grad_(True)
            critic_grad = self._update_critic_in_improve
            for p in self._critic_networks.parameters():
                p.requires_grad_(critic_grad)
            for p in self._actor_encoder.parameters():
                p.requires_grad_(critic_grad)
            self._actor_eval_samples.requires_grad_(critic_grad)

    def _sync_reference_from_current(self):
        self._reference_actor_networks.load_state_dict(
            self._actor_networks.state_dict())

    def _sync_behavior_from_reference(self):
        self._behavior_actor_networks.load_state_dict(
            self._reference_actor_networks.state_dict())

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

    def _quadratic_form(self, diff, cov):
        try:
            solve = torch.linalg.solve(cov, diff.t())
        except RuntimeError:
            solve = torch.linalg.pinv(cov) @ diff.t()
        return (diff @ solve).squeeze()

    def _group_grad_sq_norm(self, grads, group_idx: int):
        sq_norm = None
        for grad in grads:
            if grad is None:
                continue
            # Parallel actor params have group dimension first.
            if grad.ndim > 0 and grad.shape[0] == self._num_actor_critic:
                grad = grad[group_idx]
            if sq_norm is None:
                sq_norm = grad.new_zeros(())
            sq_norm = sq_norm + grad.pow(2).sum()
        if sq_norm is None:
            sq_norm = torch.zeros(())
        return sq_norm

    @torch.no_grad()
    def _compute_eval_trust_metric(self):
        ref_enc = self._compute_actor_encoding(self._reference_actor_networks)
        beh_enc = self._compute_actor_encoding(self._behavior_actor_networks)
        diff = (ref_enc - beh_enc).mean(dim=0, keepdim=True)
        dim = beh_enc.shape[-1]
        eye = torch.eye(dim, dtype=beh_enc.dtype, device=beh_enc.device)
        cov = beh_enc.transpose(0, 1) @ beh_enc / beh_enc.shape[0]
        cov = cov + self._trust_cov_reg * eye
        trust = self._quadratic_form(diff, cov)
        return torch.clamp(trust, min=0.)

    # Previous heuristic implementation retained for reference:
    # @torch.no_grad()
    # def _compute_grad_generalization_trust_metric(self, observation):
    #     ref_enc = self._compute_actor_encoding(self._reference_actor_networks)
    #     cur_enc = self._compute_actor_encoding(self._actor_networks)
    #     diff = (cur_enc - ref_enc).mean(dim=0, keepdim=True)
    #     dim = ref_enc.shape[-1]
    #     eye = torch.eye(dim, dtype=ref_enc.dtype, device=ref_enc.device)
    #     cov = ref_enc.transpose(0, 1) @ ref_enc / ref_enc.shape[0]
    #     cov = cov + self._trust_cov_reg * eye
    #     c1 = 1. + torch.sqrt(torch.clamp(self._quadratic_form(diff, cov), min=0.))
    #
    #     if observation is None:
    #         return c1
    #
    #     obs = observation.reshape(-1, *observation.shape[2:])
    #     cur_action = self._actor_networks(obs)[0]
    #     ref_action = self._reference_actor_networks(obs)[0]
    #     action_delta = (cur_action - ref_action).pow(2).sum(-1).sqrt().mean()
    #     ref_scale = ref_action.pow(2).sum(-1).sqrt().mean()
    #     c2 = 1. + action_delta / (ref_scale + 1e-6)
    #
    #     return torch.maximum(c1, c2)

    def _compute_grad_generalization_trust_metric(self, observation):
        r"""Estimate the Eq. (3.4)-style trust metric Delta_{k,t}.

        The implementation uses a sample-based estimator adapted to this
        deterministic actor setting:
        - A_k is estimated from reference-policy action features a_ref(s).
        - C1 uses policy-Jacobian magnitude times feature norm
          ||phi_t(s)||_{A_k^{-1}}.
        - C2 uses A_k^{-1}-weighted Jacobian magnitude as a proxy for
          ||D_t(s)||_{A_k^{-1},1}.
        """
        if observation is None:
            return torch.ones_like(self._last_eval_trust)

        obs_dim = len(self._observation_spec.shape)
        obs = observation.reshape(-1, *observation.shape[-obs_dim:])
        if obs.shape[0] == 0:
            return torch.ones_like(self._last_eval_trust)

        if obs.shape[0] > self._trust_metric_num_obs:
            idx = torch.randperm(obs.shape[0], device=obs.device)[
                :self._trust_metric_num_obs]
            obs = obs[idx]

        actor_params = [
            p for p in self._actor_networks.parameters() if p.requires_grad
        ]
        if len(actor_params) == 0:
            return torch.ones_like(self._last_eval_trust)

        obs = obs.detach().clone().requires_grad_(True)
        cur_action = self._actor_networks(obs)[0]  # [N, G, A]
        with torch.no_grad():
            ref_action = self._reference_actor_networks(obs.detach())[0]

        # Estimate A_k = E[a_ref a_ref^T] for each actor group.
        ref_by_group = ref_action.permute(1, 0, 2)  # [G, N, A]
        cov = ref_by_group.transpose(1, 2) @ ref_by_group / ref_by_group.shape[1]
        action_dim = cov.shape[-1]
        eye = torch.eye(
            action_dim, dtype=cov.dtype, device=cov.device).unsqueeze(0)
        cov = cov + self._trust_cov_reg * eye
        a_inv = torch.linalg.pinv(cov)

        # Feature proxy phi_t(s,a): deterministic policy action a_t(s).
        cur_by_group = cur_action.permute(1, 0, 2)  # [G, N, A]
        weighted_cur = torch.matmul(cur_by_group, a_inv)
        feature_norm = torch.sqrt(
            torch.clamp((weighted_cur * cur_by_group).sum(-1), min=0.) +
            1e-12).mean(dim=1)  # [G]

        # Estimate policy Jacobian norms wrt actor parameters.
        num_groups = cur_action.shape[1]
        grad_sq = torch.zeros(
            (num_groups, action_dim),
            dtype=cur_action.dtype,
            device=cur_action.device)
        num_terms = num_groups * action_dim
        term_idx = 0
        for group_idx in range(num_groups):
            for action_idx in range(action_dim):
                term_idx += 1
                retain_graph = term_idx < num_terms
                scalar = cur_action[:, group_idx, action_idx].mean()
                grads = torch.autograd.grad(
                    scalar,
                    actor_params,
                    retain_graph=retain_graph,
                    create_graph=False,
                    allow_unused=True)
                grad_sq[group_idx, action_idx] = self._group_grad_sq_norm(
                    grads, group_idx)

        # C1: policy-gradient-weighted feature norm.
        jacobian_norm = torch.sqrt(torch.clamp(grad_sq.sum(-1), min=0.) + 1e-12)
        c1 = torch.mean(jacobian_norm * feature_norm)

        # C2: A_k^{-1}-weighted Jacobian norm proxy for ||D_t||_{A_k^{-1},1}.
        a_inv_diag = torch.clamp(torch.diagonal(a_inv, dim1=-2, dim2=-1), min=0.)
        weighted_jac_norm = torch.sqrt(
            torch.clamp((grad_sq * a_inv_diag).sum(-1), min=0.) + 1e-12)
        c2 = torch.mean(weighted_jac_norm)

        # # Keep the metric in Algorithm-3 style coefficient regime (>= 1).
        # one = torch.ones_like(c1)
        # c1 = torch.maximum(c1, one)
        # c2 = torch.maximum(c2, one)
        return torch.maximum(c1, c2)

    def _start_new_epoch(self, increment_epoch=True):
        if increment_epoch:
            self._epoch_idx += 1

        self._sync_reference_from_current()
        if self._on_policy_adaptation:
            # Remark 3's on-policy adaptation keeps behavior equal to the
            # epoch reference policy for the entire epoch.
            self._sync_behavior_from_reference()

        # eval_trust = self._compute_eval_trust_metric()
        self._epoch_reset_flag = False

        # TODO: Re-enable eval-trust gate (C > C_max) for full Algorithm-3
        # trust-regulated epoch reset. It is intentionally disabled during
        # current on-policy adaptation exploration.
        # if eval_trust.item() > self._eval_trust_max:
        #     self._sync_behavior_from_reference()
        #     self._epoch_reset_flag = True
        #     eval_trust = self._compute_eval_trust_metric()

        self._last_eval_trust = torch.ones_like(self._last_eval_trust) # eval_trust.detach()
        self._last_grad_trust = torch.ones_like(self._last_eval_trust)
        self._eval_step_idx = 0
        self._improve_step_idx = 0
        self._pending_epoch_refresh = False
        self._set_phase(_Phase.EVAL)

    def _predict_action(self,
                        actor_net,
                        observation,
                        state: BafcActionState,
                        train=False):
        # Intentional exploration: use random actions before first training pass.
        if not self._training_started:
            outer_rank = nest_utils.get_outer_rank(observation,
                                                   self._observation_spec)
            outer_dims = alf.nest.get_nest_shape(observation)[:outer_rank]
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
        assert not self._is_eval
        # Start a new epoch only at rollout boundary (before collecting data).
        if self._pending_epoch_refresh:
            self._start_new_epoch(increment_epoch=True)
        first_mask = (inputs.step_type == StepType.FIRST)
        has_first = bool(torch.any(first_mask).item())
        if has_first or self._bootstrap_mask_type == 'step':
            if has_first:
                self._rollout_actor_id = torch.randint(self._num_actor_critic, ())
            if self._use_bootstrap_actors or self._use_bootstrap_critics:
                prob_t = torch.full(
                    (inputs.step_type.shape[0], self._num_actor_critic),
                    self._bootstrap_mask_prob,
                    device=inputs.step_type.device)
                self._bootstrap_mask = torch.bernoulli(prob_t)

        action, action_state = self._predict_action(
            self._behavior_actor_networks,
            inputs.observation,
            state=state.action)

        batch_size = inputs.step_type.shape[0]
        device = inputs.step_type.device
        eval_trust = torch.full(
            (batch_size, ),
            float(self._last_eval_trust.item()),
            dtype=torch.float32,
            device=device)
        grad_trust = torch.full(
            (batch_size, ),
            float(self._last_grad_trust.item()),
            dtype=torch.float32,
            device=device)

        info = BafcInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            observation=inputs.observation,
            action=action,
            bootstrap_mask=self._bootstrap_mask,
            phase=torch.full((batch_size, ), self._phase_id(),
                             dtype=torch.int64,
                             device=device),
            epoch_idx=torch.full((batch_size, ), self._epoch_idx,
                                 dtype=torch.int64,
                                 device=device),
            epoch_reset_flag=torch.full((batch_size, ),
                                        float(self._epoch_reset_flag),
                                        dtype=torch.float32,
                                        device=device),
            eval_trust_metric=eval_trust,
            grad_trust_metric=grad_trust)

        return AlgStep(
            output=action,
            state=state._replace(action=action_state),
            info=info)

    # Keep compatibility with OffPolicyAlgorithm-style invocation.
    def train_step(self, inputs: TimeStep, state: BafcState, rollout_info: BafcInfo):
        del rollout_info
        return self.rollout_step(inputs, state)

    def _tokenize_actor_out(self, eval_out):
        if self._actor_eval_type == 'output':
            eval_out_seq = eval_out.permute(1, 2, 0)
        else:
            eval_out_seq = torch.cat(eval_out, dim=-1).permute(1, 2, 0)

        return eval_out_seq

    def _actor_train_step(self, observation, action, mask, state):
        del state
        eval_action = self._extract_eval_action(self._actor_networks)

        actor_tokens = self._tokenize_actor_out(eval_action)
        actor_encoding = self._actor_encoder(actor_tokens)[0]
        if not self._actor_critic_pairing:
            perm = torch.randperm(self._num_actor_critic, device=action.device)
            actor_encoding = actor_encoding[perm, :]
            action = action[:, perm, :]
        actor_encoding = actor_encoding.unsqueeze(0).repeat(
            observation.shape[0], 1, 1)

        critic_observation = observation.unsqueeze(1).repeat(
            1, self._num_actor_critic, 1)
        q_value, critic_state = self._critic_networks(
            (actor_encoding, (critic_observation, action)), ())

        dqda = nest_utils.grad(action, q_value.sum(), retain_graph=True)
        if self._actor_eval_type == 'full':
            eval_action_in_graph = eval_action[1:]
        else:
            eval_action_in_graph = eval_action
        dqde = nest_utils.grad(
            eval_action_in_graph,
            q_value.sum(),
            retain_graph=self._actor_eval_type != 'output')

        def action_loss_fn(dqda, a_in):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + a_in).detach(), a_in)
            return loss.sum(list(range(2, loss.ndim)))

        action_loss = nest.map_structure(action_loss_fn, dqda, action)
        if self._use_bootstrap_actors and isinstance(mask, torch.Tensor):
            action_loss = action_loss * mask / self._bootstrap_mask_prob
        action_loss = action_loss.sum(-1)

        eval_action_loss = nest.map_structure(
            action_loss_fn, dqde, eval_action_in_graph)
        eval_action_loss = math_ops.add_n(eval_action_loss).mean().repeat(
            action_loss.shape[0])

        actor_info = LossInfo(
            loss=action_loss,
            extra=BafcActorInfo(eval_action_loss=eval_action_loss))
        return critic_state, actor_info

    def _critic_train_step(self,
                           observation,
                           state: BafcCriticState,
                           rollout_info: BafcInfo,
                           action,
                           actor_source_network=None):
        if actor_source_network is None:
            actor_source_network = self._actor_networks

        eval_action = self._extract_eval_action(actor_source_network)
        actor_tokens = self._tokenize_actor_out(eval_action)
        actor_encoding = self._actor_encoder(actor_tokens)[0]

        batch_size = observation.shape[0]
        actor_encoding = actor_encoding.repeat(batch_size, 1)
        critic_observation = observation.repeat_interleave(
            self._num_actor_critic, dim=0)
        critic_action = rollout_info.action.repeat_interleave(
            self._num_actor_critic, dim=0)

        critics, critic_state = self._critic_networks(
            (actor_encoding, (critic_observation, critic_action)), state.critic)

        with torch.no_grad():
            target_observation = observation.repeat_interleave(
                self._num_actor_critic, dim=0)
            target_critics, target_critic_state = self._target_critic_networks(
                (actor_encoding, (target_observation, action)), state.target_critic)

        critics = critics.reshape(-1, self._num_actor_critic, *critics.shape[1:])
        target_critics = target_critics.reshape(
            -1, self._num_actor_critic, *target_critics.shape[1:])
        target_critics = target_critics.detach()

        state = BafcCriticState(
            critic=critic_state, target_critic=target_critic_state)
        info = BafcCriticInfo(
            critic=critics,
            target_critic=target_critics,
            eval_trust_metric=self._last_eval_trust)

        return state, info

    def _prepare_eval_or_improve_batches(self, info: BafcInfo):
        observation = info.observation
        batch_shape = observation.shape[:2]
        obs_flat = observation.reshape(-1, *observation.shape[2:])
        action_flat = info.action.reshape(-1, *info.action.shape[2:])
        if isinstance(info.bootstrap_mask, torch.Tensor):
            mask_flat = info.bootstrap_mask.reshape(-1, *info.bootstrap_mask.shape[2:])
        else:
            mask_flat = info.bootstrap_mask
        return batch_shape, obs_flat, action_flat, mask_flat

    def _build_phase_metrics(self, batch_shape, device):
        eval_trust = torch.full(
            batch_shape,
            float(self._last_eval_trust.item()),
            dtype=torch.float32,
            device=device)
        grad_trust = torch.full(
            batch_shape,
            float(self._last_grad_trust.item()),
            dtype=torch.float32,
            device=device)
        return eval_trust, grad_trust

    def _compute_reference_critic_info(self, batch_shape, obs_flat,
                                       rollout_action_flat, device):
        ref_action = self._reference_actor_networks(obs_flat)[0]
        critic_action = ref_action.reshape(-1, ref_action.shape[-1])

        state, critic_info_flat = self._critic_train_step(
            obs_flat,
            BafcCriticState(critic=(), target_critic=()),
            BafcInfo(action=rollout_action_flat),
            critic_action,
            actor_source_network=self._reference_actor_networks)
        del state

        critics = critic_info_flat.critic.reshape(
            *batch_shape, *critic_info_flat.critic.shape[1:])
        target_critics = critic_info_flat.target_critic.reshape(
            *batch_shape, *critic_info_flat.target_critic.shape[1:])
        eval_trust, _ = self._build_phase_metrics(batch_shape, device)

        return BafcCriticInfo(
            critic=critics,
            target_critic=target_critics,
            eval_trust_metric=eval_trust)

    def _calc_eval_phase_loss(self, info: BafcInfo, batch_shape, obs_flat,
                              rollout_action_flat, device):
        eval_trust, grad_trust = self._build_phase_metrics(batch_shape, device)
        actor_info = LossInfo(extra=BafcActorInfo(
            eval_action_loss=torch.zeros(batch_shape, device=device),
            grad_trust_metric=grad_trust))
        critic_info = self._compute_reference_critic_info(
            batch_shape, obs_flat, rollout_action_flat, device)
        merged_info = info._replace(
            actor=actor_info,
            critic=critic_info,
            eval_trust_metric=eval_trust,
            grad_trust_metric=grad_trust)
        critic_loss = self._calc_critic_loss(merged_info)

        return LossInfo(
            loss=critic_loss.loss,
            scalar_loss=torch.zeros((), dtype=torch.float32, device=device),
            extra=BafcLossInfo(
                actor=actor_info.extra, critic=critic_loss.extra))

    def _calc_improve_phase_loss(self, info: BafcInfo, batch_shape, obs_flat,
                                 rollout_action_flat, mask_flat, device):
        _, grad_trust = self._build_phase_metrics(batch_shape, device)
        actor_action = self._actor_networks(obs_flat)[0]
        _, actor_info_flat = self._actor_train_step(
            obs_flat, actor_action, mask_flat, ())

        action_loss = actor_info_flat.loss.reshape(batch_shape)
        eval_action_loss = actor_info_flat.extra.eval_action_loss.reshape(
            batch_shape)
        actor_info = LossInfo(
            loss=action_loss,
            extra=BafcActorInfo(
                eval_action_loss=eval_action_loss,
                grad_trust_metric=grad_trust))

        if self._update_critic_in_improve:
            critic_info = self._compute_reference_critic_info(
                batch_shape, obs_flat, rollout_action_flat, device)
            critic_loss = self._calc_critic_loss(
                info._replace(actor=actor_info, critic=critic_info))
        else:
            critic_loss = LossInfo()

        return LossInfo(
            loss=math_ops.add_ignore_empty(actor_info.loss, critic_loss.loss),
            scalar_loss=eval_action_loss.mean(),
            extra=BafcLossInfo(
                actor=actor_info.extra, critic=critic_loss.extra))

    def calc_loss(self, info: BafcInfo):
        """Build the per-phase loss for one ALF optimizer update.

        The current on-policy adaptation maps the epoch logic from Remark 3 to
        ALF as:
        - EVAL phase: refresh the anchored critic against the epoch reference
          policy.
        - IMPROVE phase: keep behavior fixed to the epoch reference policy and
          take actor-only improvement steps with that anchored critic.

        The explicit trust-metric gates remain disabled for now; epoch
        termination is still controlled by ``max_improve_steps_per_epoch`` in
        ``after_update()``.
        """
        assert not self._is_eval
        self._training_started = True

        batch_shape, obs_flat, rollout_action_flat, mask_flat = (
            self._prepare_eval_or_improve_batches(info))
        device = obs_flat.device

        if self._phase == _Phase.EVAL:
            return self._calc_eval_phase_loss(
                info, batch_shape, obs_flat, rollout_action_flat, device)

        return self._calc_improve_phase_loss(
            info, batch_shape, obs_flat, rollout_action_flat, mask_flat,
            device)

    def _train_iter_on_policy(self):
        """Run multiple updates per rollout for on-policy BAFC-TR.

        This keeps rollout collection at the train-iter boundary while allowing
        ``num_updates_per_train_iter`` optimization steps on the same rollout.
        Epoch refresh is deferred to the next rollout boundary.
        """
        alf.summary.increment_global_counter()

        train_info, loss_info, experience = self._compute_train_info_and_loss_info_on_policy(
            self._config.unroll_length)

        with record_time("time/train"):
            valid_masks = (experience.step_type != StepType.LAST).to(torch.float32)
            num_updates = max(1, int(self._config.num_updates_per_train_iter))
            params = None
            for update_idx in range(num_updates):
                # Recompute loss against the current model for additional updates.
                if update_idx > 0:
                    loss_info = self.calc_loss(train_info)
                loss_info, params = self.update_with_gradient(
                    loss_info, valid_masks)
                self.after_update(experience.time_step, train_info)
                # Stop updates for this rollout once the epoch budget is reached.
                if self._pending_epoch_refresh:
                    break

            self.summarize_train(experience, train_info, loss_info, params)
            shape = alf.nest.get_nest_shape(experience)
            steps = shape[0] * shape[1]

        with record_time("time/after_train_iter"):
            root_inputs = (
                experience.time_step
                if self._config.use_root_inputs_for_after_train_iter else None)
            self.after_train_iter(root_inputs, train_info)

        return steps

    def _calc_critic_loss(self, info: BafcInfo):
        if not isinstance(info.critic, BafcCriticInfo) or not isinstance(
                info.critic.critic, torch.Tensor):
            return LossInfo()

        with alf.summary.record_if(lambda: self._do_critic_summary):
            critic_info = info.critic
            critic_losses = []
            for i, loss_module in enumerate(self._critic_losses):
                critic_loss = loss_module(
                    info=info,
                    value=critic_info.critic[:, :, :, i, ...],
                    target_value=critic_info.target_critic[:, :, :, i, ...]).loss
                if self._use_bootstrap_critics and isinstance(
                        info.bootstrap_mask, torch.Tensor):
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
            '_behavior_actor_networks'
        ]

    def after_update(self, root_inputs, info: BafcInfo):
        del info
        self._update_target_critic()

        if self._phase == _Phase.EVAL:
            # ``policy_eval_updates_per_epoch`` approximates the epoch-local
            # policy-evaluation subroutine before improvement begins.
            self._eval_step_idx += 1
            if self._eval_step_idx >= self._policy_eval_updates_per_epoch:
                self._set_phase(_Phase.IMPROVE)
                self._improve_step_idx = 0
        else:
            self._improve_step_idx += 1
            observation = ()
            if hasattr(root_inputs, "observation"):
                observation = root_inputs.observation
            ## Commented out for now: Sanity-check other components.
            # if isinstance(observation, torch.Tensor):
            #     self._last_grad_trust = self._compute_grad_generalization_trust_metric(
            #         observation).detach()
            # else:
            #     self._last_grad_trust = torch.ones_like(self._last_eval_trust)

            # if (self._last_grad_trust.item() > self._delta_trust_max
            #         or self._improve_step_idx >= self._max_improve_steps_per_epoch):
            # Until the explicit trust trigger is re-enabled, the current
            # variant uses ``max_improve_steps_per_epoch`` as a fixed epoch
            # budget before refreshing the anchored critic.
            if (self._improve_step_idx >= self._max_improve_steps_per_epoch):
                self._pending_epoch_refresh = True

        if self._debug_summaries and alf.summary.should_record_summaries():
            self._do_critic_summary = True
            safe_mean_hist_summary('eval_samples', self._actor_eval_samples)
            with alf.summary.scope(self._name):
                alf.summary.scalar('epoch_idx', float(self._epoch_idx))
                alf.summary.scalar('phase_id', float(self._phase_id()))
                alf.summary.scalar('eval_trust_metric', float(self._last_eval_trust.item()))
                alf.summary.scalar('grad_trust_metric', float(self._last_grad_trust.item()))
                alf.summary.scalar('epoch_reset_flag', float(self._epoch_reset_flag))
