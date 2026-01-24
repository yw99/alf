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
"""Twin Delayed Deep Deterministic Policy Gradient (TD3).

Reference:
Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods"
https://arxiv.org/abs/1802.09477

TD3 extends DDPG with three key improvements:
1. Twin Critics: Use two Q-networks and take the minimum to reduce overestimation
2. Delayed Policy Updates: Update actor less frequently than critics
3. Target Policy Smoothing: Add noise to target actions to smooth Q-value estimates
"""

import functools
from enum import Enum
from typing import Callable, Optional

import torch
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, spec_utils

Td3CriticState = namedtuple("Td3CriticState",
                            ['critics', 'target_actor', 'target_critics'],
                            default_value=())
Td3CriticInfo = namedtuple("Td3CriticInfo", ["q_values", "target_q_values"],
                           default_value=())
Td3ActorState = namedtuple("Td3ActorState", ['actor', 'critics'],
                           default_value=())
Td3State = namedtuple("Td3State", ['actor', 'critics', 'noise'],
                      default_value=())
Td3ActorInfo = namedtuple("Td3ActorInfo", ["actor_loss"], default_value=())
Td3Info = namedtuple("Td3Info", [
    "reward", "step_type", "discount", "action", "action_distribution",
    "actor", "critic", "discounted_return"
],
                     default_value=())
Td3LossInfo = namedtuple('Td3LossInfo', ('actor', 'critic'))

TrainMode = Enum('TrainMode', ('standard', 'critic', 'actor'))


@alf.configurable
class Td3Algorithm(OffPolicyAlgorithm):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3).

    TD3 extends DDPG with three key improvements:
    1. Twin Critics: Use two Q-networks and take the minimum to reduce overestimation
    2. Delayed Policy Updates: Update actor less frequently than critics
    3. Target Policy Smoothing: Add noise to target actions to smooth Q-value estimates

    Reference:
    Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods"
    https://arxiv.org/abs/1802.09477
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_ctor=ActorNetwork,
                 critic_network_ctor=CriticNetwork,
                 reward_weights=None,
                 epsilon_greedy=None,
                 calculate_priority=False,
                 env=None,
                 config: TrainerConfig = None,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 critic_loss_ctor=None,
                 num_critic_replicas=2,
                 target_update_tau=0.005,
                 target_update_period=1,
                 rollout_random_action=0.,
                 dqda_clipping=None,
                 action_l2=0,
                 target_noise_stddev=0.2,
                 target_noise_clip=0.5,
                 actor_utd: Optional[int] = None,
                 critic_utd: Optional[int] = None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="Td3Algorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            critic_network_ctor (Callable): Function to construct the critic
                network. ``critic_netwrok_ctor`` needs to accept ``input_tensor_spec``
                which is a tuple of ``(observation_spec, action_spec)``. The
                constructed network will be called with
                ``forward((observation, action), state)``.
            reward_weights (list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            num_critic_replicas (int): number of critics to be used. Default is 2
                for TD3's twin critics.
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. ``env`` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            ou_stddev (float): Standard deviation for the Ornstein-Uhlenbeck
                (OU) noise added in the default collect policy.
            ou_damping (float): Damping factor for the OU noise added in the
                default collect policy.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            rollout_random_action (float): the probability of taking a uniform
                random action during a ``rollout_step()``. 0 means always directly
                taking actions added with OU noises and 1 means always sample
                uniformly random actions. A bigger value results in more
                exploration during rollout.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between ``[-dqda_clipping, dqda_clipping]``.
                Does not perform clipping if ``dqda_clipping == 0``.
            action_l2 (float): weight of squared action l2-norm on actor loss.
            target_noise_stddev (float): Standard deviation of Gaussian noise
                added to target actions for target policy smoothing.
            target_noise_clip (float): Clipping range for the target noise.
                The noise is clipped to [-target_noise_clip, target_noise_clip].
            actor_utd (int): Number of actor updates per train iteration when
                using alternating actor/critic updates. If both actor_utd and
                critic_utd are None, standard training mode is used where both
                actor and critic are updated together every step.
            critic_utd (int): Number of critic updates per train iteration when
                using alternating actor/critic updates.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        self._calculate_priority = calculate_priority
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        critic_network = critic_network_ctor(
            input_tensor_spec=(observation_spec, action_spec),
            output_tensor_spec=reward_spec)
        actor_network = actor_network_ctor(input_tensor_spec=observation_spec,
                                           action_spec=action_spec)

        critic_networks = critic_network.make_parallel(num_critic_replicas)

        self._action_l2 = action_l2
        self._target_noise_stddev = target_noise_stddev
        self._target_noise_clip = target_noise_clip

        noise_process = alf.networks.OUProcess(state_spec=action_spec,
                                               damping=ou_damping,
                                               stddev=ou_stddev)
        noise_state = noise_process.state_spec

        predict_state_spec = Td3State(noise=noise_state,
                                      actor=Td3ActorState(
                                          actor=actor_network.state_spec,
                                          critics=critic_networks.state_spec),
                                      critics=Td3CriticState())

        train_state_spec = Td3State(
            noise=noise_state,
            actor=Td3ActorState(actor=actor_network.state_spec,
                                critics=critic_networks.state_spec),
            critics=Td3CriticState(critics=critic_networks.state_spec,
                                   target_actor=actor_network.state_spec,
                                   target_critics=critic_networks.state_spec))
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         reward_spec=reward_spec,
                         predict_state_spec=predict_state_spec,
                         train_state_spec=train_state_spec,
                         reward_weights=reward_weights,
                         env=env,
                         config=config,
                         checkpoint=checkpoint,
                         debug_summaries=debug_summaries,
                         name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])

        self._actor_network = actor_network
        self._num_critic_replicas = num_critic_replicas
        self._critic_networks = critic_networks

        self._target_actor_network = actor_network.copy(
            name='target_actor_networks')
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')

        self._rollout_random_action = float(rollout_random_action)

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(critic_loss_ctor,
                                             debug_summaries=debug_summaries)
        self._critic_losses = [None] * num_critic_replicas
        for i in range(num_critic_replicas):
            self._critic_losses[i] = critic_loss_ctor(name=("critic_loss" +
                                                            str(i)))

        self._noise_process = noise_process

        self._update_target = common.TargetUpdater(
            models=[self._actor_network, self._critic_networks],
            target_models=[
                self._target_actor_network, self._target_critic_networks
            ],
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

        # Setup train mode for delayed actor updates
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

        self._actor_update_counter = 0
        self._critic_update_counter = 0

        # Store action spec bounds for target noise clipping (convert to tensors)
        self._action_low = torch.as_tensor(action_spec.minimum)
        self._action_high = torch.as_tensor(action_spec.maximum)

    def predict_step(self, inputs: TimeStep, state):
        return self._predict_step(inputs, state, self._epsilon_greedy)

    def _predict_step(self, time_step: TimeStep, state, epsilon_greedy=1.):
        action, actor_state = self._actor_network(time_step.observation,
                                                  state=state.actor.actor)
        empty_state = nest.map_structure(lambda x: (), self.rollout_state_spec)

        def _sample(a, noise):
            if epsilon_greedy == 0:
                return a
            elif epsilon_greedy >= 1.0:
                return a + noise
            else:
                choose_noisy_action = torch.rand(a.shape[0]) < epsilon_greedy
                a[choose_noisy_action] += noise[choose_noisy_action]
                return a

        noise, noise_state = self._noise_process(state.noise)
        noisy_action = nest.map_structure(_sample, action, noise)
        noisy_action = nest.map_structure(spec_utils.clip_to_spec,
                                          noisy_action, self._action_spec)
        state = empty_state._replace(noise=noise_state,
                                     actor=Td3ActorState(actor=actor_state,
                                                         critics=()))

        return AlgStep(output=noisy_action,
                       state=state,
                       info=Td3Info(action=noisy_action,
                                    action_distribution=action))

    def rollout_step(self, time_step: TimeStep, state: Td3State = None):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by Td3Algorithm")

        def _update_random_action(spec, noisy_action):
            random_action = spec_utils.scale_to_spec(
                torch.rand_like(noisy_action) * 2 - 1, spec)
            ind = torch.where(
                torch.rand(noisy_action.shape[:1]) <
                self._rollout_random_action)
            noisy_action[ind[0], :] = random_action[ind[0], :]

        pred_step = self._predict_step(time_step, state, epsilon_greedy=1.0)
        if self._rollout_random_action > 0:
            nest.map_structure(_update_random_action, self._action_spec,
                               pred_step.output)
        return pred_step

    def _critic_train_step(self, inputs: TimeStep, state: Td3CriticState,
                           rollout_info: Td3Info):
        target_action, target_actor_state = self._target_actor_network(
            inputs.observation, state=state.target_actor)

        # Target Policy Smoothing: add clipped Gaussian noise to target actions
        def _add_target_noise(action):
            noise = torch.randn_like(action) * self._target_noise_stddev
            noise = torch.clamp(noise, -self._target_noise_clip,
                                self._target_noise_clip)
            noisy_action = action + noise
            # Clip to action bounds
            noisy_action = torch.clamp(
                noisy_action,
                self._action_low.to(action.device),
                self._action_high.to(action.device))
            return noisy_action

        target_action = nest.map_structure(_add_target_noise, target_action)

        target_q_values, target_critic_states = self._target_critic_networks(
            (inputs.observation, target_action), state=state.target_critics)

        if self.has_multidim_reward():
            sign = self.reward_weights.sign()
            target_q_values = (target_q_values * sign).min(dim=1)[0] * sign
        else:
            target_q_values = target_q_values.min(dim=1)[0]

        q_values, critic_states = self._critic_networks(
            (inputs.observation, rollout_info.action), state=state.critics)

        state = Td3CriticState(critics=critic_states,
                               target_actor=target_actor_state,
                               target_critics=target_critic_states)

        info = Td3CriticInfo(q_values=q_values,
                             target_q_values=target_q_values)

        return state, info

    def _actor_train_step(self, inputs: TimeStep, state: Td3ActorState):
        action, actor_state = self._actor_network(inputs.observation,
                                                  state=state.actor)

        q_values, critic_states = self._critic_networks(
            (inputs.observation, action), state=state.critics)
        if self.has_multidim_reward():
            # Multidimensional reward: [B, replicas, reward_dim]
            q_values = q_values * self.reward_weights
        # min over replicas
        q_value = q_values.min(dim=1)[0]

        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            if self._action_l2 > 0:
                assert action.requires_grad
                loss += self._action_l2 * (action**2)
            loss = loss.sum(list(range(1, loss.ndim)))
            return loss

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        state = Td3ActorState(actor=actor_state, critics=critic_states)
        info = LossInfo(
            loss=sum(nest.flatten(actor_loss)),
            extra=Td3ActorInfo(actor_loss=actor_loss))
        return AlgStep(output=action, state=state, info=info)

    def _update_train_mode(self):
        """Update train mode based on actor/critic update counters."""
        if self._train_mode == TrainMode.actor:
            if self._actor_update_counter % self._actor_utd == 0:
                self._train_mode = TrainMode.critic
        elif self._train_mode == TrainMode.critic:
            if self._critic_update_counter % self._critic_utd == 0:
                self._train_mode = TrainMode.actor

    def train_step(self, inputs: TimeStep, state: Td3State,
                   rollout_info: Td3Info):
        if self._train_mode == TrainMode.standard or (
                self._critic_update_counter == 0
                and self._actor_update_counter == 0):
            # Standard mode: update both actor and critic
            critic_states, critic_info = self._critic_train_step(
                inputs=inputs, state=state.critics, rollout_info=rollout_info)
            policy_step = self._actor_train_step(inputs=inputs,
                                                 state=state.actor)
            new_state = state._replace(actor=policy_step.state,
                                       critics=critic_states)
            self._critic_update_counter += 1
            return policy_step._replace(
                state=new_state,
                info=Td3Info(reward=inputs.reward,
                             step_type=inputs.step_type,
                             discount=inputs.discount,
                             action_distribution=policy_step.output,
                             critic=critic_info,
                             actor=policy_step.info,
                             discounted_return=rollout_info.discounted_return))
        else:
            # Alternating mode
            if self._train_mode == TrainMode.actor:
                # Actor mode: only train actor, skip critic
                policy_step = self._actor_train_step(inputs=inputs,
                                                     state=state.actor)
                critic_info = Td3CriticInfo()
                new_state = state._replace(actor=policy_step.state)
                self._actor_update_counter += 1
                return policy_step._replace(
                    state=new_state,
                    info=Td3Info(
                        reward=inputs.reward,
                        step_type=inputs.step_type,
                        discount=inputs.discount,
                        action_distribution=policy_step.output,
                        critic=critic_info,
                        actor=policy_step.info,
                        discounted_return=rollout_info.discounted_return))
            else:
                # Critic mode: only train critic, skip actor
                critic_states, critic_info = self._critic_train_step(
                    inputs=inputs, state=state.critics, rollout_info=rollout_info)
                # Create zero tensors with proper batch size for consistent structure
                batch_size = inputs.reward.shape[0]
                device = inputs.reward.device
                zero_loss = torch.zeros(batch_size, device=device)
                dummy_actor_loss = LossInfo(
                    loss=zero_loss,
                    extra=Td3ActorInfo(actor_loss=zero_loss))
                new_state = state._replace(critics=critic_states)
                self._critic_update_counter += 1
                return AlgStep(
                    output=rollout_info.action,
                    state=new_state,
                    info=Td3Info(
                        reward=inputs.reward,
                        step_type=inputs.step_type,
                        discount=inputs.discount,
                        action_distribution=rollout_info.action,
                        critic=critic_info,
                        actor=dummy_actor_loss,
                        discounted_return=rollout_info.discounted_return))

    def calc_loss(self, info: Td3Info):
        if self._train_mode == TrainMode.actor:
            # In actor mode, only compute actor loss
            actor_loss = info.actor
            return LossInfo(loss=actor_loss.loss,
                            priority=(),
                            extra=Td3LossInfo(critic=torch.tensor(0.),
                                              actor=actor_loss.extra))

        critic_losses = [None] * self._num_critic_replicas
        for i in range(self._num_critic_replicas):
            critic_losses[i] = self._critic_losses[i](
                info=info,
                value=info.critic.q_values[:, :, i, ...],
                target_value=info.critic.target_q_values).loss

        critic_loss = math_ops.add_n(critic_losses)

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = ((critic_loss * valid_masks).sum(dim=0) /
                        valid_n).sqrt()
        else:
            priority = ()

        actor_loss = info.actor

        if self._train_mode == TrainMode.critic:
            # In critic mode, only compute critic loss
            return LossInfo(loss=critic_loss,
                            priority=priority,
                            extra=Td3LossInfo(critic=critic_loss,
                                              actor=actor_loss.extra))
        else:
            # Standard mode: compute both
            return LossInfo(loss=math_ops.add_ignore_empty(
                                critic_loss, actor_loss.loss),
                            priority=priority,
                            extra=Td3LossInfo(critic=critic_loss,
                                              actor=actor_loss.extra))

    def after_update(self, root_inputs, info: Td3Info):
        self._update_target()
        if self._train_mode != TrainMode.standard:
            self._update_train_mode()

    def _trainable_attributes_to_ignore(self):
        return ['_target_actor_network', '_target_critic_networks']
