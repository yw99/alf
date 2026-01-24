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
"""Tests for td3_algorithm.py."""

from absl.testing import parameterized
import functools
import os
import torch

# Force CPU-only for tests to avoid device mismatch issues in test framework
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import alf
from alf.algorithms.td3_algorithm import Td3Algorithm
from alf.algorithms.config import TrainerConfig
from alf.environments.suite_unittest import PolicyUnittestEnv, ActionType
from alf.networks import ActorNetwork, CriticNetwork


class Td3AlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(
        dict(num_critic_replicas=1),
        dict(num_critic_replicas=2),
    )
    def test_td3_algorithm(self, num_critic_replicas):
        """Test TD3 algorithm with different number of critic replicas."""
        num_env = 128
        steps_per_episode = 13
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=steps_per_episode,
            mini_batch_length=2,
            mini_batch_size=128,
            initial_collect_steps=steps_per_episode,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
        )

        env = PolicyUnittestEnv(
            num_env, steps_per_episode, action_type=ActionType.Continuous)

        obs_spec = env._observation_spec
        action_spec = env._action_spec

        fc_layer_params = (16, 16)

        actor_network = functools.partial(
            ActorNetwork, fc_layer_params=fc_layer_params)

        critic_network = functools.partial(
            CriticNetwork, joint_fc_layer_params=fc_layer_params)

        alg = Td3Algorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            actor_network_ctor=actor_network,
            critic_network_ctor=critic_network,
            env=env,
            config=config,
            num_critic_replicas=num_critic_replicas,
            target_noise_stddev=0.2,
            target_noise_clip=0.5,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2))

        for _ in range(5):
            alg.train_iter()

    def test_td3_delayed_updates(self):
        """Test TD3 with delayed actor updates using actor_utd/critic_utd."""
        num_env = 128
        steps_per_episode = 13
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=steps_per_episode,
            mini_batch_length=2,
            mini_batch_size=128,
            initial_collect_steps=steps_per_episode,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
            num_updates_per_train_iter=4,
        )

        env = PolicyUnittestEnv(
            num_env, steps_per_episode, action_type=ActionType.Continuous)

        obs_spec = env._observation_spec
        action_spec = env._action_spec

        fc_layer_params = (16, 16)

        actor_network = functools.partial(
            ActorNetwork, fc_layer_params=fc_layer_params)

        critic_network = functools.partial(
            CriticNetwork, joint_fc_layer_params=fc_layer_params)

        # Test with critic_utd=3, actor_utd=1 (1 actor update per 3 critic updates)
        alg = Td3Algorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            actor_network_ctor=actor_network,
            critic_network_ctor=critic_network,
            env=env,
            config=config,
            num_critic_replicas=2,
            actor_utd=1,
            critic_utd=3,
            target_noise_stddev=0.2,
            target_noise_clip=0.5,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2))

        for _ in range(5):
            alg.train_iter()


if __name__ == '__main__':
    alf.test.main()
