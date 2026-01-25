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
"""Twin Delayed Deep Deterministic Policy Gradient (TD3) - Original Paper Version.

This implementation follows the original TD3 paper more closely:
- Critics are updated every gradient step
- Actor is updated only every `policy_delay` steps (default: 2)
- Target networks are updated only when actor is updated

Reference:
Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods"
https://arxiv.org/abs/1802.09477
"""

import functools
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


@alf.configurable
class Td3Algorithm2(OffPolicyAlgorithm):
    """TD3 Algorithm following the original paper's delayed policy update.

    This version follows the original TD3 paper more closely:
    - Critics are updated every gradient step
    - Actor is updated only every `policy_delay` steps (default: 2)
    - Target networks are updated only when actor is updated

    This is different from Td3Algorithm which uses BAFCv3-style alternating
    actor/critic training phases.

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
                 policy_delay=2,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="Td3Algorithm2"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_ctor (Callable): Function to construct the actor network.
            critic_network_ctor (Callable): Function to construct the critic network.
            reward_weights (list[float]): this is only used when the reward is
                multidimensional.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax.
            calculate_priority (bool): whether to calculate priority.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            env (Environment): The environment to interact with.
            config (TrainerConfig): config for training.
            ou_stddev (float): Standard deviation for the Ornstein-Uhlenbeck noise.
            ou_damping (float): Damping factor for the OU noise.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss constructor.
            target_update_tau (float): Factor for soft update of target networks.
            target_update_period (int): Period for soft update of target networks.
                Note: In original TD3, target update happens only when actor is updated.
            rollout_random_action (float): probability of taking random action during rollout.
            dqda_clipping (float): clips the gradient dqda element-wise.
            action_l2 (float): weight of squared action l2-norm on actor loss.
            target_noise_stddev (float): Standard deviation of Gaussian noise
                added to target actions for target policy smoothing.
            target_noise_clip (float): Clipping range for the target noise.
            policy_delay (int): Number of critic updates per actor update.
                Default is 2 following the original TD3 paper.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            checkpoint (None|str): checkpoint to load.
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
        self._policy_delay = policy_delay

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

        # Target updater - will be called manually only when actor is updated
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._update_target = common.TargetUpdater(
            models=[self._actor_network, self._critic_networks],
            target_models=[
                self._target_actor_network, self._target_critic_networks
            ],
            tau=target_update_tau,
            period=1)  # We control the period manually

        self._dqda_clipping = dqda_clipping

        # Counter for delayed policy updates
        self._update_counter = 0
        self._update_actor_this_step = False

        # Store action spec bounds for target noise clipping
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
                                      "is not supported by Td3Algorithm2")

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
            q_values = q_values * self.reward_weights
        q_value = q_values.min(dim=1)[0]

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

    def train_step(self, inputs: TimeStep, state: Td3State,
                   rollout_info: Td3Info):
        """Train step following original TD3 paper.

        - Always update critics
        - Update actor only every `policy_delay` steps
        """
        self._update_counter += 1
        self._update_actor_this_step = (self._update_counter % self._policy_delay == 0)

        # Always update critics
        critic_states, critic_info = self._critic_train_step(
            inputs=inputs, state=state.critics, rollout_info=rollout_info)

        if self._update_actor_this_step:
            # Update actor
            policy_step = self._actor_train_step(inputs=inputs,
                                                 state=state.actor)
            new_state = state._replace(actor=policy_step.state,
                                       critics=critic_states)
            actor_info = policy_step.info
            action_output = policy_step.output
        else:
            # Skip actor update - use dummy loss
            batch_size = inputs.reward.shape[0]
            device = inputs.reward.device
            zero_loss = torch.zeros(batch_size, device=device)
            actor_info = LossInfo(
                loss=zero_loss,
                extra=Td3ActorInfo(actor_loss=zero_loss))
            new_state = state._replace(critics=critic_states)
            action_output = rollout_info.action

        return AlgStep(
            output=action_output,
            state=new_state,
            info=Td3Info(reward=inputs.reward,
                         step_type=inputs.step_type,
                         discount=inputs.discount,
                         action_distribution=action_output,
                         critic=critic_info,
                         actor=actor_info,
                         discounted_return=rollout_info.discounted_return))

    def calc_loss(self, info: Td3Info):
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

        # Always include both losses, but actor loss is zero when not updating
        return LossInfo(loss=math_ops.add_ignore_empty(critic_loss, actor_loss.loss),
                        priority=priority,
                        extra=Td3LossInfo(critic=critic_loss,
                                          actor=actor_loss.extra))

    def after_update(self, root_inputs, info: Td3Info):
        # Update target networks only when actor was updated (following original TD3)
        if self._update_actor_this_step:
            self._update_target()

    def _trainable_attributes_to_ignore(self):
        return ['_target_actor_network', '_target_critic_networks']
