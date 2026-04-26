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

import functools
from unittest import mock
import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.agent import Agent
from alf.algorithms.icm_algorithm import ICMAlgorithm
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


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

    def test_off_policy_unroll_hooks_are_delegated_to_rl_algorithm(self):
        observation_spec = TensorSpec((10, ))
        action_spec = BoundedTensorSpec((), dtype='int64')
        actor_net = functools.partial(ActorDistributionNetwork,
                                      fc_layer_params=(100, ))
        value_net = functools.partial(ValueNetwork, fc_layer_params=(100, ))
        agent = Agent(observation_spec=observation_spec,
                      action_spec=action_spec,
                      rl_algorithm_cls=functools.partial(
                          ActorCriticAlgorithm,
                          actor_network_ctor=actor_net,
                          value_network_ctor=value_net))

        should_skip = mock.Mock(return_value=True)
        after_unroll = mock.Mock()
        agent._rl_algorithm._should_skip_unroll_iter_off_policy = should_skip
        agent._rl_algorithm._after_unroll_iter_off_policy = after_unroll

        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                return_value=(True, "root", "info")) as parent_unroll:
            unrolled, root_inputs, rollout_info = agent._unroll_iter_off_policy()

        self.assertFalse(unrolled)
        self.assertIsNone(root_inputs)
        self.assertIsNone(rollout_info)
        should_skip.assert_called_once()
        parent_unroll.assert_not_called()
        after_unroll.assert_not_called()

        should_skip.reset_mock()
        should_skip.return_value = False
        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                return_value=(True, "root", "info")) as parent_unroll:
            unrolled, root_inputs, rollout_info = agent._unroll_iter_off_policy()

        self.assertTrue(unrolled)
        self.assertEqual(root_inputs, "root")
        self.assertEqual(rollout_info, "info")
        should_skip.assert_called_once()
        parent_unroll.assert_called_once()
        after_unroll.assert_called_once_with(True)


if __name__ == "__main__":
    alf.test.main()
