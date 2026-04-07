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

import os
import functools
import tempfile
import torch
from tensorboard.backend.event_processing import event_file_loader

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.agent import Agent
from alf.algorithms.config import TrainerConfig
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import AlgStep, LossInfo, StepType, TimeStep
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils import common


def _find_event_file(root_dir):
    event_file = None
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if "events" in file_name and 'profile' not in file_name:
                event_file = os.path.join(root, file_name)
                break
    return event_file


class _SummaryTestEnv(object):

    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._observation_spec = TensorSpec((2, ), dtype='float32')
        self._action_spec = BoundedTensorSpec(shape=(),
                                              dtype='int64',
                                              minimum=0,
                                              maximum=1)
        self.reset()

    @property
    def is_tensor_based(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return TensorSpec(())

    def reset(self):
        self._prev_action = torch.zeros(self._batch_size, dtype=torch.int64)
        self._current_time_step = TimeStep(
            observation=self._observation_spec.randn([self._batch_size]),
            step_type=torch.full([self._batch_size],
                                 StepType.FIRST,
                                 dtype=torch.int32),
            reward=torch.zeros(self._batch_size),
            discount=torch.zeros(self._batch_size),
            prev_action=self._prev_action,
            env_id=torch.arange(self._batch_size, dtype=torch.int32))
        return self._current_time_step

    def step(self, action):
        step_type = torch.where(
            self._current_time_step.step_type == StepType.LAST,
            torch.full([self._batch_size], StepType.FIRST, dtype=torch.int32),
            torch.full([self._batch_size], StepType.MID, dtype=torch.int32))
        step_type = torch.where(
            self._current_time_step.step_type == StepType.MID,
            torch.full([self._batch_size], StepType.LAST, dtype=torch.int32),
            step_type)

        reward = action.to(torch.float32)
        self._current_time_step = TimeStep(
            observation=self._observation_spec.randn([self._batch_size]),
            step_type=step_type,
            reward=reward,
            discount=torch.zeros(self._batch_size),
            prev_action=self._prev_action,
            env_id=torch.arange(self._batch_size, dtype=torch.int32))
        self._prev_action = action
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step

    def close(self):
        pass


class _AlwaysSkipAfterWarmupAlg(RLAlgorithm):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config=None,
                 debug_summaries=False):
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         reward_spec=reward_spec,
                         train_state_spec=(),
                         env=env,
                         is_on_policy=False,
                         config=config,
                         optimizer=alf.optimizers.Adam(lr=1e-2),
                         debug_summaries=debug_summaries,
                         name="AlwaysSkipAfterWarmupAlg")
        self._proj_net = alf.networks.CategoricalProjectionNetwork(
            input_size=2, action_spec=action_spec)
        self._training_started = False

    def predict_step(self, time_step: TimeStep, state):
        dist, _ = self._proj_net(time_step.observation)
        return AlgStep(output=dist.sample(), state=(), info=())

    def rollout_step(self, time_step: TimeStep, state):
        dist, _ = self._proj_net(time_step.observation)
        action = dist.sample()
        return AlgStep(output=action,
                       state=(),
                       info=dict(action=action, dist=dist))

    def train_step(self, time_step: TimeStep, state, rollout_info):
        self._training_started = True
        dist, _ = self._proj_net(time_step.observation)
        return AlgStep(output=dist.sample(),
                       state=(),
                       info=dict(action=rollout_info['action'], dist=dist))

    def calc_loss(self, info):
        return LossInfo(loss=-info['dist'].log_prob(info['action']))

    def request_skip_rollout_iter(self):
        return self._training_started


class AgentTest(alf.test.TestCase):

    def test_agent_steps(self):
        batch_size = 1
        observation_spec = TensorSpec((10, ))
        action_spec = BoundedTensorSpec((), dtype='int64')
        time_step = TimeStep(
            reward=torch.ones((batch_size, )),
            observation=observation_spec.zeros(outer_dims=(batch_size, )),
            prev_action=action_spec.zeros(outer_dims=(batch_size, )))

        actor_net = functools.partial(ActorDistributionNetwork,
                                      fc_layer_params=(100, ))
        value_net = functools.partial(ValueNetwork, fc_layer_params=(100, ))

        # TODO: add a goal generator and an entropy target algorithm once they
        # are implemented.
        agent = Agent(observation_spec=observation_spec,
                      action_spec=action_spec,
                      rl_algorithm_cls=functools.partial(
                          ActorCriticAlgorithm,
                          actor_network_ctor=actor_net,
                          value_network_ctor=value_net),
                      intrinsic_reward_module=ICMAlgorithm(
                          action_spec=action_spec,
                          observation_spec=observation_spec))

        predict_state = agent.get_initial_predict_state(batch_size)
        rollout_state = agent.get_initial_rollout_state(batch_size)
        train_state = agent.get_initial_train_state(batch_size)

        pred_step = agent.predict_step(time_step, predict_state)
        self.assertEqual(pred_step.state.irm, ())

        rollout_step = agent.rollout_step(time_step, rollout_state)
        self.assertFalse(rollout_step.state.irm == ())

        train_step = agent.train_step(time_step, train_state,
                                      rollout_step.info)
        self.assertFalse(train_step.state.irm == ())

        self.assertTensorEqual(rollout_step.state.irm, train_step.state.irm)

    def test_off_policy_metrics_keep_logging_when_rollout_is_skipped(self):
        with tempfile.TemporaryDirectory() as root_dir:
            alf.summary.reset_global_counter()
            alf.summary.enable_summary()

            config = TrainerConfig(root_dir=root_dir,
                                   unroll_length=1,
                                   mini_batch_length=1,
                                   mini_batch_size=1,
                                   initial_collect_steps=1,
                                   replay_buffer_length=20,
                                   num_updates_per_train_iter=1)
            env = _SummaryTestEnv(batch_size=1)
            agent = Agent(observation_spec=env.observation_spec(),
                          action_spec=env.action_spec(),
                          env=env,
                          config=config,
                          rl_algorithm_cls=_AlwaysSkipAfterWarmupAlg)

            common.run_under_record_context(
                lambda: [agent.train_iter() for _ in range(15)],
                summary_dir=root_dir,
                summary_interval=5,
                flush_secs=1)

            self.assertTrue(agent._rl_algorithm._training_started)
            self.assertLess(agent.get_step_metrics()[1].result().item(), 15)

            event_file = _find_event_file(root_dir)
            self.assertIsNotNone(event_file)

            metrics_steps = []
            metrics_vs_env_count = 0
            for event in event_file_loader.EventFileLoader(event_file).Load():
                if not event.summary.value:
                    continue
                for item in event.summary.value:
                    if item.tag == 'Metrics/EnvironmentSteps':
                        metrics_steps.append(event.step)
                    elif item.tag == 'Metrics_vs_EnvironmentSteps/AverageReturn':
                        metrics_vs_env_count += 1

            self.assertGreater(max(metrics_steps), 5)
            self.assertGreater(metrics_vs_env_count, 1)


if __name__ == "__main__":
    alf.test.main()
