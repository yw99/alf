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
"""BAFCv7: BAFC with an episode-seeded squashed-Gaussian actor.

BAFCv7 is deliberately implemented in a new module.  It does not change the
BAFCv3 algorithm, ``ActorFCNetwork``, or ``NormalProjectionNetwork``.
"""

import functools
from typing import Optional, Union

import torch
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import AlgStep, LossInfo, StepType, TimeStep, namedtuple
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import FuncCriticNetwork, TransformerEncoder
from alf.networks.bafc_v7_actor_network import BafcV7ActorNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils import checkpoint_utils, common, losses, math_ops
from alf.utils.schedulers import Scheduler


BafcV7ActionState = namedtuple(
    "BafcV7ActionState",
    ["actor_network", "episode_seed", "rollout_actor_id"],
    default_value=())

BafcV7CriticState = namedtuple("BafcV7CriticState",
                                ["critic", "target_critic"])

BafcV7State = namedtuple(
    "BafcV7State", ["action", "actor", "critic"], default_value=())

BafcV7CriticInfo = namedtuple(
    "BafcV7CriticInfo", ["critic", "target_critic"], default_value=())

BafcV7ActorInfo = namedtuple(
    "BafcV7ActorInfo", ["eval_action_loss"], default_value=())

BafcV7Info = namedtuple(
    "BafcV7Info", [
        "reward", "step_type", "discount", "action", "episode_seed",
        "rollout_actor_id", "actor", "critic", "discounted_return"
    ],
    default_value=())

BafcV7LossInfo = namedtuple(
    "BafcV7LossInfo", ["actor", "critic"], default_value=())


@alf.configurable
class BafcAlgorithmV7(OffPolicyAlgorithm):
    r"""Functional critic with an episode-seeded Gaussian actor.

    For an episode seed :math:`e \sim N(0,I)`, fresh step noise
    :math:`z_t \sim N(0,I)`, and ``temporal_noise_mix`` :math:`\lambda`, the
    behavior policy samples

    .. math::

        u_t = \mu(s_t) + \sigma(s_t) \odot
              (\sqrt{1-\lambda^2}e + \lambda z_t),\qquad a_t=T(u_t).

    ``training_policy='base'`` uses the marginal, unconditioned actor for all
    training computations. ``training_policy='seeded'`` conditions training
    actions and actor fingerprints on the seed stored in replay.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=BafcV7ActorNetwork,
                 critic_network_cls=FuncCriticNetwork,
                 reward_weights=None,
                 calculate_priority=False,
                 num_actors=10,
                 num_critics=10,
                 temporal_noise_mix=0.1,
                 training_policy="base",
                 actor_update_mode="paired",
                 num_sampled_critics_for_actor=1,
                 num_actor_eval_samples=512,
                 eval_samples_init_method="normal",
                 eval_samples_clipping=False,
                 actor_eval_type="last_two",
                 actor_encoder_cls=TransformerEncoder,
                 actor_encoding_dim=None,
                 obs_action_encoding_dim=128,
                 actor_utd: Optional[int] = None,
                 critic_utd: Optional[int] = None,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_critic_tau: Union[float, Scheduler] = 0.005,
                 target_critic_period: Union[int, Scheduler] = 1,
                 target_critic_use_ema=False,
                 dqda_clipping=None,
                 checkpoint_replay_buffer=False,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 actor_encoder_optimizer=None,
                 eval_samples_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="BafcAlgorithmV7",
                 use_random_critic_targets=True,
                 num_sampled_critic_targets=1):
        del calculate_priority
        if not isinstance(action_spec, BoundedTensorSpec):
            raise TypeError("BAFCv7 requires a bounded continuous action spec")
        if action_spec.ndim != 1 or not action_spec.is_continuous:
            raise ValueError(
                "BAFCv7 supports only bounded vector-valued continuous actions")
        if not isinstance(num_actors, int) or num_actors < 1:
            raise ValueError("num_actors must be a positive integer")
        if not isinstance(num_critics, int) or num_critics < 1:
            raise ValueError("num_critics must be a positive integer")
        if not 0. < temporal_noise_mix <= 1.:
            raise ValueError("temporal_noise_mix must be in (0, 1]")
        if training_policy not in ("base", "seeded"):
            raise ValueError("training_policy must be 'base' or 'seeded'")
        if training_policy == "seeded" and num_actors != 1:
            raise ValueError(
                "seeded training initially supports exactly one actor")
        valid_update_modes = ("paired", "random_subset_mean", "mean_all",
                              "min_all")
        if actor_update_mode not in valid_update_modes:
            raise ValueError(
                f"actor_update_mode must be one of {valid_update_modes}")
        if actor_update_mode == "paired" and num_actors != num_critics:
            raise ValueError(
                "paired actor updates require num_actors == num_critics")
        if not 1 <= num_sampled_critics_for_actor <= num_critics:
            raise ValueError(
                "num_sampled_critics_for_actor must be in [1, num_critics]")
        if not 1 <= num_sampled_critic_targets <= num_critics:
            raise ValueError(
                "num_sampled_critic_targets must be in [1, num_critics]")
        if actor_eval_type not in ("last_two", "output"):
            raise ValueError(
                "BAFCv7 supports actor_eval_type 'last_two' or 'output'")
        if eval_samples_init_method not in ("normal", "uniform"):
            raise ValueError(
                "eval_samples_init_method must be 'normal' or 'uniform'")

        if actor_utd is None and critic_utd is None:
            self._train_mode = TrainMode.standard
        else:
            if config is None:
                raise ValueError(
                    "config is required when actor_utd or critic_utd is set")
            total_utd = config.num_updates_per_train_iter
            if actor_utd is None:
                actor_utd = total_utd - critic_utd
            elif critic_utd is None:
                critic_utd = total_utd - actor_utd
            if actor_utd < 1 or critic_utd < 1:
                raise ValueError("actor_utd and critic_utd must be positive")
            if actor_utd > critic_utd:
                raise ValueError("actor_utd must not exceed critic_utd")
            self._train_mode = TrainMode.critic
            self._actor_utd = actor_utd
            self._critic_utd = critic_utd

        self._num_actors = num_actors
        self._num_critics = num_critics
        self._temporal_noise_mix = float(temporal_noise_mix)
        self._training_policy = training_policy
        self._actor_update_mode = actor_update_mode
        self._num_sampled_critics_for_actor = (
            num_sampled_critics_for_actor)
        self._use_random_critic_targets = use_random_critic_targets
        self._num_sampled_critic_targets = num_sampled_critic_targets
        self._actor_eval_type = actor_eval_type
        self._num_actor_eval_samples = num_actor_eval_samples
        self._checkpoint_replay_buffer = checkpoint_replay_buffer
        self._dqda_clipping = dqda_clipping

        actor_networks = actor_network_cls(
            input_tensor_spec=observation_spec,
            action_spec=action_spec,
            n_groups=num_actors)
        if not isinstance(actor_networks, BafcV7ActorNetwork):
            raise TypeError(
                "actor_network_cls must construct BafcV7ActorNetwork")

        if eval_samples_init_method == "normal":
            actor_eval_samples = 2 * torch.randn(
                num_actor_eval_samples, observation_spec.shape[0])
            if eval_samples_clipping:
                actor_eval_samples.clamp_(-1., 1.)
        else:
            actor_eval_samples = 2 * torch.rand(
                num_actor_eval_samples, observation_spec.shape[0]) - 1

        policy_feature_size = 2 * action_spec.shape[0]
        if actor_eval_type == "last_two":
            actor_token_length = (
                actor_networks.bias_params[-2].shape[1] + policy_feature_size)
        else:
            actor_token_length = policy_feature_size
        actor_token_spec = TensorSpec(
            (actor_token_length, num_actor_eval_samples))
        actor_encoder = actor_encoder_cls(
            actor_token_spec, core_embedding_dim=actor_encoding_dim)
        if actor_encoding_dim is None:
            actor_encoding_dim = num_actor_eval_samples

        actor_spec = TensorSpec((actor_encoding_dim, ))
        critic_network = critic_network_cls(
            input_tensor_spec=(actor_spec, (observation_spec, action_spec)),
            obs_action_encoding_dim=obs_action_encoding_dim,
            actor_obs_action_combiner=alf.layers.NestConcat(dim=-1))
        critic_networks = critic_network.make_parallel(num_critics)

        action_state_spec = BafcV7ActionState(
            actor_network=actor_networks.state_spec,
            episode_seed=TensorSpec(action_spec.shape),
            rollout_actor_id=TensorSpec((), dtype=torch.int64))
        train_state_spec = BafcV7State(
            action=action_state_spec,
            actor=critic_networks.state_spec,
            critic=BafcV7CriticState(
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

        self._actor_networks = actor_networks
        self._actor_encoder = actor_encoder
        self._critic_networks = critic_networks
        self._target_critic_networks = critic_networks.copy(
            name="target_critic_networks")
        self._actor_eval_samples = nn.Parameter(actor_eval_samples)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_networks])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if actor_encoder_optimizer is not None:
            self.add_optimizer(actor_encoder_optimizer, [actor_encoder])
        if eval_samples_optimizer is not None:
            self.add_optimizer(eval_samples_optimizer,
                               [self._actor_eval_samples])

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        self._critic_losses = [
            critic_loss_ctor(name=f"critic_loss{i + 1}")
            for i in range(num_critics)
        ]

        self._actor_update_counter = 0
        self._critic_update_counter = 0
        self._training_started = False
        self._update_target_critic = common.TargetUpdater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_critic_tau,
            period=target_critic_period,
            delayed_update=target_critic_use_ema)

    @property
    def num_actors(self):
        return self._num_actors

    @property
    def num_critics(self):
        return self._num_critics

    def _runtime_key(self, prefix, name):
        return prefix + "_bafcv7_runtime." + name

    def _save_runtime_state(self, destination, prefix):
        destination[self._runtime_key(
            prefix, "training_started")] = torch.tensor(
                self._training_started, dtype=torch.bool)
        destination[self._runtime_key(
            prefix, "train_mode")] = torch.tensor(
                self._train_mode.value, dtype=torch.int64)
        destination[self._runtime_key(
            prefix, "actor_update_counter")] = torch.tensor(
                self._actor_update_counter, dtype=torch.int64)
        destination[self._runtime_key(
            prefix, "critic_update_counter")] = torch.tensor(
                self._critic_update_counter, dtype=torch.int64)

    def _pop_runtime_state(self, state_dict, prefix):
        runtime_prefix = self._runtime_key(prefix, "")
        runtime_state = {}
        for key in list(state_dict.keys()):
            if key.startswith(runtime_prefix):
                runtime_state[key[len(runtime_prefix):]] = state_dict.pop(key)
        return runtime_state

    def _save_to_state_dict(self, destination, prefix, visited=None):
        super()._save_to_state_dict(destination, prefix, visited)
        self._save_runtime_state(destination, prefix)

    def _load_from_state_dict(self,
                              state_dict,
                              prefix,
                              local_metadata,
                              strict,
                              missing_keys,
                              unexpected_keys,
                              error_msgs,
                              visited=None):
        runtime_state = self._pop_runtime_state(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs, visited)
        if runtime_state:
            self._training_started = bool(
                runtime_state["training_started"].reshape(()).item())
            self._train_mode = TrainMode(
                int(runtime_state["train_mode"].reshape(()).item()))
            self._actor_update_counter = int(
                runtime_state["actor_update_counter"].reshape(()).item())
            self._critic_update_counter = int(
                runtime_state["critic_update_counter"].reshape(()).item())
            self._apply_train_mode_grad_flags()

    def checkpoint_replay_buffer_enabled(self):
        return self._checkpoint_replay_buffer

    def _set_replay_buffer_checkpoint_enabled(self, enabled):
        if not self._checkpoint_replay_buffer or self._replay_buffer is None:
            return None
        old_enabled = checkpoint_utils.is_checkpoint_enabled(
            self._replay_buffer)
        checkpoint_utils.enable_checkpoint(self._replay_buffer, enabled)

        def _restore():
            checkpoint_utils.enable_checkpoint(self._replay_buffer,
                                               old_enabled)

        return _restore

    def _alf_prepare_checkpoint_save(self):
        return self._set_replay_buffer_checkpoint_enabled(True)

    def _alf_prepare_checkpoint_load(self, state_dict):
        has_replay = any(
            key.startswith("_replay_buffer.") or "._replay_buffer." in key
            for key in state_dict)
        return self._set_replay_buffer_checkpoint_enabled(has_replay)

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        return root_inputs, rollout_info

    def _resample_episode_state(self, step_type, state):
        """Resample seed and actor id independently for each FIRST element."""
        is_first = step_type == StepType.FIRST
        episode_seed = state.episode_seed.clone()
        rollout_actor_id = state.rollout_actor_id.clone()
        num_first = int(is_first.sum().item())
        if num_first:
            episode_seed[is_first] = torch.randn_like(
                episode_seed[is_first])
            rollout_actor_id[is_first] = torch.randint(
                self._num_actors,
                (num_first, ),
                device=step_type.device,
                dtype=torch.int64)
        return state._replace(
            episode_seed=episode_seed, rollout_actor_id=rollout_actor_id)

    def _initial_random_action(self, observation):
        outer_rank = nest_utils.get_outer_rank(observation,
                                               self._observation_spec)
        outer_dims = nest.get_nest_shape(observation)[:outer_rank]
        return nest.map_structure(
            lambda spec: spec.sample(outer_dims=outer_dims), self._action_spec)

    def _behavior_action(self, observation, state):
        output = self._actor_networks(
            observation, state=state.actor_network)
        seed = state.episode_seed.unsqueeze(1)
        all_actions = self._actor_networks.sample_seeded(
            output, seed, self._temporal_noise_mix)
        action_dim = all_actions.shape[-1]
        selected = torch.gather(
            all_actions,
            dim=1,
            index=state.rollout_actor_id[:, None, None].expand(
                -1, 1, action_dim)).squeeze(1)
        return selected, state._replace(actor_network=output.state)

    def predict_step(self, inputs: TimeStep, state: BafcV7ActionState):
        """Use deterministic actor 0; evaluation never consumes a seed."""
        output = self._actor_networks(
            inputs.observation, state=state.actor_network)
        action = self._actor_networks.mode(output)[:, 0, :]
        new_state = state._replace(actor_network=output.state)
        return AlgStep(
            output=action,
            state=new_state,
            info=BafcV7Info(
                action=action,
                episode_seed=state.episode_seed,
                rollout_actor_id=torch.zeros_like(state.rollout_actor_id)))

    def rollout_step(self, inputs: TimeStep, state: BafcV7State):
        assert not self._is_eval
        action_state = self._resample_episode_state(inputs.step_type,
                                                    state.action)
        if self._training_started:
            action, action_state = self._behavior_action(
                inputs.observation, action_state)
        else:
            # Keep initial collection independent and uniform at every step.
            action = self._initial_random_action(inputs.observation)
        info = BafcV7Info(
            action=action,
            episode_seed=action_state.episode_seed,
            rollout_actor_id=action_state.rollout_actor_id)
        return AlgStep(
            output=action,
            state=state._replace(action=action_state),
            info=info)

    def _policy_encoding(self, actor_eval_samples, episode_seed=None):
        """Encode either every base actor or every seed-conditioned actor."""
        output = self._actor_networks(
            actor_eval_samples,
            full_neurons=self._actor_eval_type == "last_two")
        if episode_seed is None:
            policy_features = output.policy_features
            if self._actor_eval_type == "last_two":
                feature_tensors = [output.neurons[-2], policy_features]
            else:
                feature_tensors = [policy_features]
            tokens = torch.cat(feature_tensors, dim=-1).permute(1, 2, 0)
            encoding = self._actor_encoder(tokens)[0]
            return encoding, feature_tensors

        # Probe parameters are [P,A,D]. Broadcast one replay seed per batch
        # item to [B,P,A,D], then flatten [B,A] for the transformer batch.
        seed = episode_seed[:, None, None, :]
        conditional_mean, conditional_std = (
            self._actor_networks.seeded_parameters(
                output.mean[None], output.std[None], seed,
                self._temporal_noise_mix))
        policy_features = torch.cat([
            conditional_mean,
            torch.log(conditional_std.clamp_min(
                torch.finfo(conditional_std.dtype).tiny))
        ],
                                    dim=-1)
        if self._actor_eval_type == "last_two":
            hidden = output.neurons[-2].unsqueeze(0).expand(
                episode_seed.shape[0], *output.neurons[-2].shape)
            feature_tensors = [hidden, policy_features]
        else:
            feature_tensors = [policy_features]
        tokens = torch.cat(feature_tensors, dim=-1).permute(0, 2, 3, 1)
        batch_size, num_actors = tokens.shape[:2]
        encoding = self._actor_encoder(
            tokens.reshape(batch_size * num_actors,
                           *tokens.shape[2:]))[0]
        encoding = encoding.reshape(batch_size, num_actors, -1)
        return encoding, feature_tensors

    def _training_action(self, observation, episode_seed):
        output = self._actor_networks(observation)
        if self._training_policy == "base":
            action = self._actor_networks.sample_base(output)
        else:
            action = self._actor_networks.sample_seeded(
                output, episode_seed[:, None, :], self._temporal_noise_mix)
        return action, output.state

    def _training_encoding(self, actor_eval_samples, episode_seed):
        if self._training_policy == "base":
            return self._policy_encoding(actor_eval_samples)
        return self._policy_encoding(actor_eval_samples, episode_seed)

    def _expand_encoding(self, actor_encoding, batch_size):
        if actor_encoding.ndim == 2:
            return actor_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        return actor_encoding

    def _all_critic_values(self, critic_networks, actor_encoding,
                           observation, action, state):
        """Evaluate every critic on every actor-aligned policy input."""
        batch_size = observation.shape[0]
        actor_encoding = self._expand_encoding(actor_encoding, batch_size)
        observation = observation.unsqueeze(1).expand(
            batch_size, self._num_actors, *observation.shape[1:])
        if action.ndim == len(self._action_spec.shape) + 1:
            action = action.unsqueeze(1).expand(batch_size, self._num_actors,
                                                *action.shape[1:])

        flat_size = batch_size * self._num_actors
        actor_encoding = actor_encoding.reshape(flat_size, -1)
        observation = observation.reshape(flat_size, *observation.shape[2:])
        action = action.reshape(flat_size, *action.shape[2:])

        actor_encoding = actor_encoding.unsqueeze(1).expand(
            flat_size, self._num_critics, actor_encoding.shape[-1])
        observation = observation.unsqueeze(1).expand(
            flat_size, self._num_critics, *observation.shape[1:])
        action = action.unsqueeze(1).expand(flat_size, self._num_critics,
                                           *action.shape[1:])
        values, new_state = critic_networks(
            (actor_encoding, (observation, action)), state)
        values = values.reshape(batch_size, self._num_actors,
                                self._num_critics, *values.shape[2:])
        return values, new_state

    def _minimum_over_critics(self, q_values):
        if self.has_multidim_reward():
            sign = self.reward_weights.sign()
            return (q_values * sign).min(dim=2)[0] * sign
        return q_values.min(dim=2)[0]

    def _actor_objective_values(self, q_values):
        if self._actor_update_mode == "paired":
            ids = torch.arange(self._num_actors, device=q_values.device)
            return q_values[:, ids, ids, ...]
        if self._actor_update_mode == "mean_all":
            return q_values.mean(dim=2)
        if self._actor_update_mode == "min_all":
            return self._minimum_over_critics(q_values)

        k = self._num_sampled_critics_for_actor
        critic_ids = torch.stack([
            torch.randperm(self._num_critics, device=q_values.device)[:k]
            for _ in range(self._num_actors)
        ])
        gather_shape = [1, self._num_actors, k] + [1] * (
            q_values.ndim - 3)
        expand_shape = list(q_values.shape)
        expand_shape[2] = k
        selected = torch.gather(
            q_values, 2,
            critic_ids.reshape(gather_shape).expand(expand_shape))
        return selected.mean(dim=2)

    def _surrogate_loss(self, gradient, value):
        if self._dqda_clipping is not None:
            gradient = gradient.clamp(-self._dqda_clipping,
                                      self._dqda_clipping)
        loss = 0.5 * losses.element_wise_squared_loss(
            (gradient + value).detach(), value)
        return loss.sum(tuple(range(2, loss.ndim)))

    def _actor_train_step(self, observation, action, actor_encoding,
                          actor_features, replay_action, state):
        q_values, critic_state = self._all_critic_values(
            self._critic_networks, actor_encoding, observation, action, state)
        objective = self._actor_objective_values(q_values).sum()
        dqda, dqde = nest_utils.grad((action, actor_features), objective)

        action_loss = self._surrogate_loss(dqda, action).sum(dim=-1)
        feature_losses = nest.map_structure(self._surrogate_loss, dqde,
                                            actor_features)
        eval_action_loss = math_ops.add_n(feature_losses).mean().repeat(
            action_loss.shape[0])
        actor_info = LossInfo(
            loss=action_loss,
            extra=BafcV7ActorInfo(eval_action_loss=eval_action_loss))
        return critic_state, actor_info

    def _select_critic_targets(self, target_critics):
        if not self._use_random_critic_targets:
            return target_critics
        if self._num_sampled_critic_targets < self._num_critics:
            ids = torch.randperm(
                self._num_critics,
                device=target_critics.device)[:self._num_sampled_critic_targets]
            target_critics = target_critics.index_select(2, ids)
        return self._minimum_over_critics(target_critics)

    def _critic_train_step(self, observation, state, rollout_info,
                           target_action, actor_encoding):
        critics, critic_state = self._all_critic_values(
            self._critic_networks, actor_encoding, observation,
            rollout_info.action, state.critic)
        with torch.no_grad():
            target_critics, target_critic_state = self._all_critic_values(
                self._target_critic_networks, actor_encoding.detach(),
                observation, target_action.detach(), state.target_critic)
            target_critics = self._select_critic_targets(target_critics)
        new_state = BafcV7CriticState(
            critic=critic_state, target_critic=target_critic_state)
        return new_state, BafcV7CriticInfo(
            critic=critics, target_critic=target_critics)

    def _apply_train_mode_grad_flags(self):
        standard_or_initial = (
            self._train_mode == TrainMode.standard or
            (self._actor_update_counter == 0
             and self._critic_update_counter == 0))
        actor_requires_grad = (standard_or_initial
                               or self._train_mode == TrainMode.actor)
        for parameter in self._actor_networks.parameters():
            parameter.requires_grad_(actor_requires_grad)
        self._actor_eval_samples.requires_grad_(
            standard_or_initial or self._train_mode == TrainMode.critic)

    def _update_train_mode(self):
        if self._train_mode == TrainMode.actor:
            if self._actor_update_counter % self._actor_utd == 0:
                self._train_mode = TrainMode.critic
        elif self._train_mode == TrainMode.critic:
            if self._critic_update_counter % self._critic_utd == 0:
                self._train_mode = TrainMode.actor
        self._apply_train_mode_grad_flags()

    def train_step(self, inputs: TimeStep, state: BafcV7State,
                   rollout_info: BafcV7Info):
        assert not self._is_eval
        self._training_started = True
        episode_seed = rollout_info.episode_seed
        action, action_state = self._training_action(inputs.observation,
                                                     episode_seed)
        actor_encoding, actor_features = self._training_encoding(
            self._actor_eval_samples, episode_seed)

        standard_or_initial = (
            self._train_mode == TrainMode.standard or
            (self._actor_update_counter == 0
             and self._critic_update_counter == 0))
        if standard_or_initial:
            actor_state, actor_info = self._actor_train_step(
                inputs.observation, action, actor_encoding, actor_features,
                rollout_info.action, state.actor)
            critic_state, critic_info = self._critic_train_step(
                inputs.observation, state.critic, rollout_info, action,
                actor_encoding)
            new_state = BafcV7State(
                action=state.action._replace(actor_network=action_state),
                actor=actor_state,
                critic=critic_state)
            self._critic_update_counter += 1
        elif self._train_mode == TrainMode.actor:
            actor_state, actor_info = self._actor_train_step(
                inputs.observation, action, actor_encoding, actor_features,
                rollout_info.action, state.actor)
            critic_info = BafcV7CriticInfo()
            new_state = BafcV7State(
                action=state.action._replace(actor_network=action_state),
                actor=actor_state,
                critic=state.critic)
            self._actor_update_counter += 1
        else:
            critic_state, critic_info = self._critic_train_step(
                inputs.observation, state.critic, rollout_info, action,
                actor_encoding)
            actor_info = LossInfo(extra=BafcV7ActorInfo())
            new_state = BafcV7State(
                action=state.action._replace(actor_network=action_state),
                actor=state.actor,
                critic=critic_state)
            self._critic_update_counter += 1

        info = BafcV7Info(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            episode_seed=episode_seed,
            rollout_actor_id=rollout_info.rollout_actor_id,
            actor=actor_info,
            critic=critic_info,
            discounted_return=rollout_info.discounted_return)
        return AlgStep(action, new_state, info)

    def _calc_critic_loss(self, info):
        critic_losses = []
        for critic_id, loss_fn in enumerate(self._critic_losses):
            if self._use_random_critic_targets:
                target_value = info.critic.target_critic
            else:
                target_value = info.critic.target_critic[:, :, :, critic_id,
                                                         ...]
            critic_losses.append(
                loss_fn(
                    info=info,
                    value=info.critic.critic[:, :, :, critic_id, ...],
                    target_value=target_value).loss)
        critic_loss = math_ops.add_n(critic_losses)
        return LossInfo(loss=critic_loss, extra=critic_loss)

    def calc_loss(self, info: BafcV7Info):
        actor_loss = info.actor
        eval_action_loss = actor_loss.extra.eval_action_loss
        if isinstance(eval_action_loss, torch.Tensor):
            eval_action_loss = eval_action_loss.mean()
        if self._train_mode == TrainMode.actor:
            critic_loss = LossInfo()
        else:
            critic_loss = self._calc_critic_loss(info)
        return LossInfo(
            loss=math_ops.add_ignore_empty(actor_loss.loss, critic_loss.loss),
            scalar_loss=eval_action_loss,
            extra=BafcV7LossInfo(
                actor=actor_loss.extra, critic=critic_loss.extra))

    def _trainable_attributes_to_ignore(self):
        return ["_target_critic_networks"]

    def after_update(self, root_inputs, info: BafcV7Info):
        self._update_train_mode()
        self._update_target_critic()
