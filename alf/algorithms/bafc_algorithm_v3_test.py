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

import tempfile
from functools import partial
from types import SimpleNamespace
from unittest import mock

import torch

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.bafc_algorithm_v3 import BafcAlgorithmV3
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import StepType, TimeStep
from alf.experience_replayers.replay_buffer import ReplayBuffer
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.checkpoint_utils import Checkpointer


class _DummyProcess:

    def memory_info(self):
        return SimpleNamespace(rss=0)

    def children(self, recursive=True):
        del recursive
        return []


class BafcAlgorithmV3CheckpointTest(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._psutil_patcher = mock.patch(
            "alf.algorithms.algorithm.psutil.Process",
            return_value=_DummyProcess())
        self._psutil_patcher.start()
        self.addCleanup(self._psutil_patcher.stop)

    def _make_alg(self, **kwargs):
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=3)
        return BafcAlgorithmV3(
            observation_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec((2, ), minimum=-1.0, maximum=1.0),
            config=config,
            actor_network_cls=partial(ActorFCNetwork, fc_layer_params=(32, 32)),
            critic_network_cls=partial(
                FuncCriticNetwork,
                obs_action_joint_fc_layer_params=(32, 32),
                actor_obs_action_joint_fc_layer_params=(32, 32)),
            actor_encoder_cls=partial(
                TransformerEncoder, num_layers=2, num_attention_heads=1),
            num_actor_critic=3,
            num_actor_eval_samples=16,
            **kwargs)

    def _make_agent(self, **kwargs):
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_agent_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=3)
        return Agent(
            observation_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec((2, ), minimum=-1.0, maximum=1.0),
            config=config,
            rl_algorithm_cls=partial(
                BafcAlgorithmV3,
                actor_network_cls=partial(
                    ActorFCNetwork, fc_layer_params=(32, 32)),
                critic_network_cls=partial(
                    FuncCriticNetwork,
                    obs_action_joint_fc_layer_params=(32, 32),
                    actor_obs_action_joint_fc_layer_params=(32, 32)),
                actor_encoder_cls=partial(
                    TransformerEncoder, num_layers=2, num_attention_heads=1),
                num_actor_critic=3,
                num_actor_eval_samples=16,
                **kwargs))

    def _attach_replay_buffer(self, alg, num_items=0):
        replay_buffer = ReplayBuffer(
            data_spec=TimeStep(
                step_type=TensorSpec((), dtype=torch.int64),
                reward=TensorSpec(()),
                discount=TensorSpec(()),
                observation=TensorSpec((4, )),
                prev_action=TensorSpec((2, )),
                env_id=TensorSpec((), dtype=torch.int64)),
            num_environments=1,
            max_length=8)
        for value in range(num_items):
            replay_buffer.add_batch(
                TimeStep(
                    step_type=torch.tensor([StepType.MID], dtype=torch.int64),
                    reward=torch.tensor([float(value)], dtype=torch.float32),
                    discount=torch.ones(1),
                    observation=torch.full((1, 4), float(value)),
                    prev_action=torch.zeros(1, 2),
                    env_id=torch.tensor([0], dtype=torch.int64)),
                env_ids=torch.tensor([0]))
        alg._replay_buffer = replay_buffer
        return replay_buffer

    def _without_runtime_state(self, state_dict):
        state_dict = state_dict.copy()
        for key in list(state_dict.keys()):
            if "_bafc_runtime." in key:
                del state_dict[key]
        return state_dict

    def test_runtime_checkpoint_round_trip_and_legacy_fallback(self):
        alg = self._make_alg()
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._rollout_actor_id = torch.tensor(2)
        alg._actor_update_counter = 3
        alg._critic_update_counter = 4
        alg._apply_train_mode_grad_flags()

        state = alg.state_dict()
        self.assertIn("_bafc_runtime.training_started", state)

        restored = self._make_alg()
        restored.load_state_dict(state)

        self.assertTrue(restored._training_started)
        self.assertEqual(restored._train_mode, TrainMode.critic)
        self.assertEqual(restored._rollout_actor_id, 2)
        self.assertEqual(restored._actor_update_counter, 3)
        self.assertEqual(restored._critic_update_counter, 4)
        self.assertTrue(all(not p.requires_grad
                            for p in restored._actor_networks.parameters()))
        self.assertTrue(restored._actor_eval_samples.requires_grad)

        legacy_restored = self._make_alg()
        legacy_restored.load_state_dict(self._without_runtime_state(state))
        self.assertTrue(legacy_restored._training_started)

    def test_reweighting_target_cache_round_trip(self):
        alg = self._make_alg(
            track_reweighting_target_observation_cache=True,
            critic_reweighting_target_obs_cache_size=3)
        alg._append_reweighting_target_observations(
            torch.arange(8, dtype=torch.float32).reshape(2, 4))
        alg._append_reweighting_target_observations(
            torch.arange(8, 20, dtype=torch.float32).reshape(3, 4))

        expected = torch.stack([
            torch.arange(8, 12, dtype=torch.float32),
            torch.arange(12, 16, dtype=torch.float32),
            torch.arange(16, 20, dtype=torch.float32),
        ])
        self.assertTensorClose(alg._reweighting_target_observation_cache,
                               expected)

        restored = self._make_alg(
            track_reweighting_target_observation_cache=True,
            critic_reweighting_target_obs_cache_size=3)
        restored.load_state_dict(alg.state_dict())
        self.assertTensorClose(restored._reweighting_target_observation_cache,
                               expected)

    def test_replay_checkpoint_is_save_context_only_and_ranked(self):
        alg = self._make_alg(checkpoint_replay_buffer=True)
        self._attach_replay_buffer(alg, num_items=1)
        self.assertFalse(
            any("_replay_buffer." in key for key in alg.state_dict().keys()))

        with tempfile.TemporaryDirectory() as ckpt_dir:
            checkpointer = Checkpointer(ckpt_dir, algorithm=alg)
            checkpointer.save(10, ddp_rank=0)
            self.assertTrue(
                any("_replay_buffer." in key
                    for key in torch.load(
                        f"{ckpt_dir}/ckpt-10-replay_buffer-rank0")[
                            "algorithm"].keys()))
            self.assertTrue(
                any("_replay_buffer." in key
                    for key in torch.load(f"{ckpt_dir}/ckpt-10-replay_buffer")[
                        "algorithm"].keys()))

            self._attach_replay_buffer(alg, num_items=3)
            checkpointer.save(10, ddp_rank=1)
            self.assertTrue(
                torch.load(f"{ckpt_dir}/ckpt-10-replay_buffer-rank1")[
                    "algorithm"])

            restored = self._make_alg(checkpoint_replay_buffer=True)
            self._attach_replay_buffer(restored, num_items=0)
            restored_checkpointer = Checkpointer(ckpt_dir, algorithm=restored)
            restored_checkpointer.load(10, ddp_rank=1)
            self.assertTensorEqual(restored._replay_buffer._current_pos,
                                   torch.tensor([3]))

            self.assertFalse(
                any("_replay_buffer." in key
                    for key in restored.state_dict().keys()))

    def test_agent_replay_checkpoint_is_save_context_only_and_ranked(self):
        agent = self._make_agent(checkpoint_replay_buffer=True)
        self._attach_replay_buffer(agent, num_items=1)
        self.assertFalse(
            any("_replay_buffer." in key
                for key in agent.state_dict().keys()))

        with tempfile.TemporaryDirectory() as ckpt_dir:
            checkpointer = Checkpointer(ckpt_dir, algorithm=agent)
            checkpointer.save(10, ddp_rank=0)
            rank0_state = torch.load(
                f"{ckpt_dir}/ckpt-10-replay_buffer-rank0")["algorithm"]
            self.assertTrue(
                any(key.startswith("_replay_buffer.")
                    for key in rank0_state.keys()))
            self.assertFalse(
                any(key.startswith("_rl_algorithm._replay_buffer.")
                    for key in rank0_state.keys()))

            self._attach_replay_buffer(agent, num_items=3)
            checkpointer.save(10, ddp_rank=1)

            restored = self._make_agent(checkpoint_replay_buffer=True)
            self._attach_replay_buffer(restored, num_items=0)
            restored_checkpointer = Checkpointer(ckpt_dir, algorithm=restored)
            restored_checkpointer.load(10, ddp_rank=1)
            self.assertTensorEqual(restored._replay_buffer._current_pos,
                                   torch.tensor([3]))

            self.assertFalse(
                any("_replay_buffer." in key
                    for key in restored.state_dict().keys()))


if __name__ == "__main__":
    alf.test.main()
