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

import copy
import random
import tempfile
from functools import partial
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2
from alf.algorithms.bafc_algorithm_v3_tr3 import (
    BafcAlgorithmV3TR3, BafcAlgorithmV3TR3Agent)
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import StepType, TimeStep
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class _DummyProcess:

    def memory_info(self):
        return SimpleNamespace(rss=0)

    def children(self, recursive=True):
        del recursive
        return []


class _TinyContinuousEnv:

    def __init__(self, batch_size=1):
        self._batch_size = batch_size
        self._observation_spec = TensorSpec((4, ), dtype="float32")
        self._action_spec = BoundedTensorSpec(
            (2, ), dtype="float32", minimum=-1.0, maximum=1.0)
        self.step_count = 0
        self.reset_count = 0
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
        self.reset_count += 1
        self._prev_action = self._action_spec.zeros(
            outer_dims=(self._batch_size, ))
        self._current_time_step = TimeStep(
            observation=self._observation_spec.zeros([self._batch_size]),
            step_type=torch.full(
                [self._batch_size], StepType.FIRST, dtype=torch.int32),
            reward=torch.zeros(self._batch_size),
            discount=torch.zeros(self._batch_size),
            prev_action=self._prev_action,
            env_id=torch.arange(self._batch_size, dtype=torch.int32))
        return self._current_time_step

    def step(self, action):
        self.step_count += 1
        self._current_time_step = TimeStep(
            observation=torch.full((self._batch_size, 4),
                                   float(self.step_count)),
            step_type=torch.full(
                [self._batch_size], StepType.MID, dtype=torch.int32),
            reward=action.mean(dim=-1),
            discount=torch.ones(self._batch_size),
            prev_action=action,
            env_id=torch.arange(self._batch_size, dtype=torch.int32))
        self._prev_action = action
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step

    def close(self):
        pass


class BafcAlgorithmV3TR3Test(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        default_device = alf.get_default_device()
        alf.set_default_device("cpu")
        self.addCleanup(alf.set_default_device, default_device)
        patcher = mock.patch(
            "alf.algorithms.algorithm.psutil.Process",
            return_value=_DummyProcess())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _algorithm_ctor(self, algorithm_cls, config, **kwargs):
        return algorithm_cls(
            observation_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec(
                (2, ), minimum=-1.0, maximum=1.0),
            config=config,
            actor_network_cls=partial(
                ActorFCNetwork, fc_layer_params=(16, 16)),
            critic_network_cls=partial(
                FuncCriticNetwork,
                obs_action_joint_fc_layer_params=(16, 16),
                actor_obs_action_joint_fc_layer_params=(16, 16)),
            actor_encoder_cls=partial(
                TransformerEncoder, num_layers=2, num_attention_heads=1),
            num_actor_critic=3,
            num_actor_eval_samples=8,
            trust_metric_num_obs=4,
            **kwargs)

    def _make_alg(self, algorithm_cls=BafcAlgorithmV3TR3, initial_steps=3,
                  **kwargs):
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_tr3_alg_test_"),
            unroll_length=1,
            mini_batch_length=2,
            mini_batch_size=2,
            initial_collect_steps=initial_steps,
            replay_buffer_length=16,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
            num_updates_per_train_iter=2)
        return self._algorithm_ctor(algorithm_cls, config, **kwargs)

    def _make_agent(self,
                    initial_steps=3,
                    use_normalizer=True,
                    actor_utd=1,
                    critic_utd=1,
                    num_updates_per_train_iter=2):
        env = _TinyContinuousEnv()
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_tr3_agent_test_"),
            unroll_length=1,
            mini_batch_length=2,
            mini_batch_size=2,
            initial_collect_steps=initial_steps,
            replay_buffer_length=16,
            data_transformer_ctor=(ObservationNormalizer
                                   if use_normalizer else None),
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
            num_updates_per_train_iter=num_updates_per_train_iter)
        agent = BafcAlgorithmV3TR3Agent(
            observation_spec=env.observation_spec(),
            action_spec=env.action_spec(),
            env=env,
            config=config,
            optimizer=alf.optimizers.Adam(lr=1e-3),
            rl_algorithm_cls=partial(
                BafcAlgorithmV3TR3,
                actor_network_cls=partial(
                    ActorFCNetwork, fc_layer_params=(16, 16)),
                critic_network_cls=partial(
                    FuncCriticNetwork,
                    obs_action_joint_fc_layer_params=(16, 16),
                    actor_obs_action_joint_fc_layer_params=(16, 16)),
                actor_encoder_cls=partial(
                    TransformerEncoder, num_layers=2,
                    num_attention_heads=1),
                num_actor_critic=3,
                num_actor_eval_samples=8,
                trust_metric_num_obs=4,
                monitor_trust_metrics=False,
                checkpoint_replay_buffer=True,
                actor_utd=actor_utd,
                critic_utd=critic_utd,
                rollout_cycles_per_collect=1))
        return agent, env

    def _clone_tensor_state(self, module):
        return copy.deepcopy(module.state_dict())

    def _assert_nested_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTensorEqual(expected, actual)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self._assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self._assert_nested_equal(expected_item, actual_item)
        else:
            self.assertEqual(expected, actual)

    def _assert_tensor_state_equal(self, expected, module):
        actual = module.state_dict()
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            self._assert_nested_equal(expected[key], actual[key])

    def test_tr2_state_dict_strict_loads_into_tr3(self):
        torch.manual_seed(123)
        tr2 = self._make_alg(
            algorithm_cls=BafcAlgorithmV3TR2,
            actor_utd=1,
            critic_utd=1,
            rollout_cycles_per_collect=1)
        torch.manual_seed(456)
        tr3 = self._make_alg(
            actor_utd=1,
            critic_utd=1,
            rollout_cycles_per_collect=1)

        tr2_state = tr2.state_dict()
        self.assertEqual(set(tr2_state), set(tr3.state_dict()))
        for key, value in tr2_state.items():
            self.assertEqual(value.shape, tr3.state_dict()[key].shape)
            self.assertEqual(value.dtype, tr3.state_dict()[key].dtype)
        tr3.load_state_dict(tr2_state, strict=True)
        self._assert_tensor_state_equal(tr2_state, tr3)

        tr3._alf_prepare_checkpoint_load({})
        self.assertEqual(set(tr2_state), set(tr3.state_dict()))

    def test_checkpoint_status_absent_partial_and_sufficient(self):
        tr3 = self._make_alg(initial_steps=3)

        tr3._alf_prepare_checkpoint_load({})
        self.assertTrue(tr3._tr3_is_replay_refill_decision_pending())
        self.assertTrue(tr3._tr3_is_train_info_spec_priming_pending())
        self.assertEqual(
            tr3._tr3_replay_refill_plan(),
            dict(clear_probe_replay=True, checkpoint_size=None))

        tr3._alf_prepare_checkpoint_load(
            {"_replay_buffer._current_size": torch.tensor([2])})
        self.assertTrue(tr3._tr3_is_replay_refill_decision_pending())
        self.assertEqual(
            tr3._tr3_replay_refill_plan(),
            dict(clear_probe_replay=False, checkpoint_size=2))

        tr3._alf_prepare_checkpoint_load(
            {"_replay_buffer._current_size": torch.tensor([3])})
        self.assertTrue(tr3._tr3_is_replay_refill_decision_pending())
        self.assertIsNone(tr3._tr3_replay_refill_plan())

        tr3._tr3_mark_replay_refill_complete()
        self.assertFalse(tr3._tr3_is_replay_refill_decision_pending())
        self.assertTrue(tr3._tr3_is_train_info_spec_priming_pending())
        tr3._tr3_mark_train_info_spec_primed()
        self.assertFalse(tr3._tr3_is_train_info_spec_priming_pending())
        tr3._alf_prepare_checkpoint_load({})
        self.assertTrue(tr3._tr3_is_replay_refill_decision_pending())
        self.assertTrue(tr3._tr3_is_train_info_spec_priming_pending())

    def test_refill_decision_collective_runs_once_after_restore(self):
        agent, _ = self._make_agent(initial_steps=3, use_normalizer=False)
        controller = agent._rl_algorithm

        with mock.patch.object(
                Agent, "train_iter", return_value=1), mock.patch.object(
                    controller, "_all_reduce_control") as all_reduce:
            # Training from scratch, including Trainer's lazy pre-load probe,
            # must not perform a TR3 refill-decision collective.
            self.assertEqual(agent.train_iter(), 1)
            all_reduce.assert_not_called()
            self.assertFalse(
                controller._tr3_is_train_info_spec_priming_pending())

            # A sufficient replay checkpoint still arms one common decision
            # across ranks. Once resolved, later iterations bypass it.
            controller._alf_prepare_checkpoint_load(
                {"_replay_buffer._current_size": torch.tensor([3])})
            all_reduce.return_value = torch.tensor([0.0])
            with mock.patch.object(
                    torch.distributed, "is_available", return_value=True), \
                    mock.patch.object(
                        torch.distributed, "is_initialized", return_value=True), \
                    mock.patch.object(
                        torch.distributed, "get_world_size", return_value=4):
                self.assertEqual(agent.train_iter(), 1)
                self.assertEqual(agent.train_iter(), 1)

            all_reduce.assert_called_once_with(
                [False], op=torch.distributed.ReduceOp.MAX)
            self.assertFalse(
                controller._tr3_is_replay_refill_decision_pending())

    def test_resume_primes_spec_across_real_critic_actor_cycle(self):
        agent, _ = self._make_agent(
            initial_steps=3,
            use_normalizer=False,
            actor_utd=1,
            critic_utd=11,
            num_updates_per_train_iter=12)
        controller = agent._rl_algorithm

        # Mirror Trainer's lazy construction pass before restoring an old
        # checkpoint without replay.
        agent.train_iter()
        self.assertIsNone(agent._train_info_spec)
        controller._alf_prepare_checkpoint_load({})
        controller._training_started = True
        controller._train_mode = TrainMode.critic
        controller._actor_update_counter = 7
        controller._critic_update_counter = 77
        controller._completed_cycles_since_rollout = 1
        controller._apply_train_mode_grad_flags()

        with mock.patch.object(
                controller, "train_step", wraps=controller.train_step) as step:
            trained_steps = agent.train_iter()

        self.assertGreater(trained_steps, 0)
        # Spec priming reuses the first normal update; it does not add a probe
        # forward to the configured 12-update critic/actor cycle.
        self.assertEqual(step.call_count, 12)
        self.assertEqual(controller._critic_update_counter, 88)
        self.assertEqual(controller._actor_update_counter, 8)
        self.assertEqual(controller._train_mode, TrainMode.critic)
        self.assertEqual(controller._completed_cycles_since_rollout, 1)
        self.assertFalse(
            controller._tr3_is_train_info_spec_priming_pending())

        spec = agent.train_info_spec.rl
        for leaf in (spec.actor.loss,
                     spec.actor.extra.eval_action_loss,
                     spec.critic.critic,
                     spec.critic.target_critic,
                     spec.critic.eval_trust_metric,
                     spec.critic.critic_sample_weight):
            self.assertIsInstance(leaf, TensorSpec)

    def test_resume_spec_priming_handles_actor_and_standard_modes(self):
        cases = ((TrainMode.actor, 1, 1), (TrainMode.standard, None, None))
        for mode, actor_utd, critic_utd in cases:
            with self.subTest(mode=mode):
                agent, _ = self._make_agent(
                    initial_steps=3,
                    use_normalizer=False,
                    actor_utd=actor_utd,
                    critic_utd=critic_utd,
                    num_updates_per_train_iter=2)
                controller = agent._rl_algorithm
                agent.train_iter()
                controller._alf_prepare_checkpoint_load({})
                controller._training_started = True
                controller._train_mode = mode
                controller._actor_update_counter = 5
                controller._critic_update_counter = 5
                controller._completed_cycles_since_rollout = 1
                controller._apply_train_mode_grad_flags()

                with mock.patch.object(
                        controller,
                        "train_step",
                        wraps=controller.train_step) as step:
                    self.assertGreater(agent.train_iter(), 0)

                self.assertEqual(step.call_count, 2)
                self.assertFalse(
                    controller._tr3_is_train_info_spec_priming_pending())
                spec = agent.train_info_spec.rl
                self.assertIsInstance(spec.actor.loss, TensorSpec)
                self.assertIsInstance(spec.critic.critic, TensorSpec)

    def test_train_info_spec_priming_failure_stays_pending(self):
        agent, _ = self._make_agent(initial_steps=3, use_normalizer=False)
        controller = agent._rl_algorithm
        controller._alf_prepare_checkpoint_load({})

        with mock.patch.object(
                Agent,
                "train_step",
                side_effect=RuntimeError("intentional train-step failure")):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                agent.train_step(None, None, None)

        self.assertIsNone(agent._train_info_spec)
        self.assertTrue(
            controller._tr3_is_train_info_spec_priming_pending())

    def test_refill_is_pretraining_only_and_clears_probe_sample(self):
        agent, env = self._make_agent(initial_steps=3)

        # This mirrors Trainer._restore_checkpoint(): the one lazy iteration
        # constructs replay before an old checkpoint without replay is loaded.
        agent.train_iter()
        self.assertEqual(int(agent._replay_buffer.total_size), 1)
        agent._rl_algorithm._alf_prepare_checkpoint_load({})
        agent._rl_algorithm._training_started = True

        controller = agent._rl_algorithm
        controller._target_metric_observation_cache = torch.arange(
            8, dtype=torch.float32).reshape(2, 4)
        model_state = self._clone_tensor_state(agent)
        metric_states = [
            copy.deepcopy(metric.state_dict()) for metric in agent.get_metrics()
        ]
        global_counter = int(alf.summary.get_global_counter())
        cached_observations = controller._target_metric_observation_cache.clone()
        controller_fields = (
            controller._rollout_actor_id,
            controller._actor_update_counter,
            controller._critic_update_counter,
            controller._completed_cycles_since_rollout,
            controller._trust_metric_update_counter)
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state().clone()

        action_calls = []

        def restored_policy(actor_net, observation, state, train=False):
            del actor_net, train
            self.assertTrue(controller._training_started)
            action_calls.append(observation.detach().clone())
            return torch.full((observation.shape[0], 2), 0.25), state

        with mock.patch.object(
                controller, "_predict_action", side_effect=restored_policy):
            agent._tr3_refill_replay_buffer(
                controller._tr3_replay_refill_plan())

        self.assertEqual(len(action_calls), 3)
        self.assertEqual(int(agent._replay_buffer.total_size), 3)
        self.assertIsNone(controller._tr3_replay_refill_plan())
        self.assertFalse(
            controller._tr3_is_replay_refill_decision_pending())
        self._assert_tensor_state_equal(model_state, agent)
        for expected, metric in zip(metric_states, agent.get_metrics()):
            actual = metric.state_dict()
            self.assertEqual(set(expected), set(actual))
            for key, value in expected.items():
                self.assertTensorEqual(value, actual[key])
        self.assertEqual(int(alf.summary.get_global_counter()), global_counter)
        self.assertTensorEqual(controller._target_metric_observation_cache,
                               cached_observations)
        self.assertEqual(
            (controller._rollout_actor_id,
             controller._actor_update_counter,
             controller._critic_update_counter,
             controller._completed_cycles_since_rollout,
             controller._trust_metric_update_counter), controller_fields)
        self.assertEqual(random.getstate(), python_rng_state)
        actual_numpy_rng = np.random.get_state()
        self.assertEqual(actual_numpy_rng[0], numpy_rng_state[0])
        self.assertTrue(np.array_equal(actual_numpy_rng[1], numpy_rng_state[1]))
        self.assertEqual(actual_numpy_rng[2:], numpy_rng_state[2:])
        self.assertTensorEqual(torch.get_rng_state(), torch_rng_state)
        self.assertGreaterEqual(env.reset_count, 2)

    def test_partial_replay_is_preserved_and_only_remainder_is_collected(self):
        agent, _ = self._make_agent(initial_steps=3, use_normalizer=False)
        agent.train_iter()
        self.assertEqual(int(agent._replay_buffer.total_size), 1)
        controller = agent._rl_algorithm
        controller._alf_prepare_checkpoint_load(
            {"_replay_buffer._current_size": torch.tensor([1])})
        controller._training_started = True
        action_calls = []

        def restored_policy(actor_net, observation, state, train=False):
            del actor_net, train
            action_calls.append(observation)
            return torch.zeros((observation.shape[0], 2)), state

        with mock.patch.object(
                controller, "_predict_action", side_effect=restored_policy):
            agent._tr3_refill_replay_buffer(
                controller._tr3_replay_refill_plan())

        self.assertEqual(len(action_calls), 2)
        self.assertEqual(int(agent._replay_buffer.total_size), 3)
    def test_train_iter_refills_then_runs_one_normal_rollout(self):
        agent, env = self._make_agent(initial_steps=3, use_normalizer=False)
        agent.train_iter()
        self.assertEqual(int(agent._replay_buffer.total_size), 1)

        agent._rl_algorithm._alf_prepare_checkpoint_load({})
        agent._rl_algorithm._training_started = True
        agent._rl_algorithm._train_mode = TrainMode.critic
        agent._rl_algorithm._completed_cycles_since_rollout = 1
        steps_before = env.step_count

        with mock.patch.object(
                agent, "train_from_replay_buffer", return_value=1):
            trained_steps = agent.train_iter()

        # Three hidden refill transitions are cleared/rebuilt, followed by one
        # ordinary trainer-visible rollout before replay training.
        self.assertGreater(trained_steps, 0)
        self.assertEqual(int(agent._replay_buffer.total_size), 4)
        self.assertEqual(env.step_count - steps_before, 4)
        self.assertIsNone(agent._rl_algorithm._tr3_replay_refill_plan())

    def test_refill_capacity_guard(self):
        agent, _ = self._make_agent(initial_steps=17)
        agent.train_iter()
        agent._rl_algorithm._alf_prepare_checkpoint_load({})
        with self.assertRaisesRegex(RuntimeError, "replay capacity"):
            agent._tr3_refill_replay_buffer(
                agent._rl_algorithm._tr3_replay_refill_plan())
        self.assertTrue(
            agent._rl_algorithm._tr3_is_replay_refill_decision_pending())

    def test_ddp_refill_plan_includes_full_ranks(self):
        agent, _ = self._make_agent(initial_steps=3, use_normalizer=False)
        agent.train_iter()
        full_rank_plan = None
        with mock.patch.object(
                torch.distributed, "is_available", return_value=True), \
                mock.patch.object(
                    torch.distributed, "is_initialized", return_value=True), \
                mock.patch.object(
                    torch.distributed, "get_world_size", return_value=4), \
                mock.patch.object(
                    agent._rl_algorithm,
                    "_all_reduce_control",
                    return_value=torch.tensor([1.0])):
            full_rank_plan = agent._tr3_synchronize_refill_plan(None)

        self.assertEqual(
            full_rank_plan,
            dict(
                clear_probe_replay=False,
                checkpoint_size=int(agent._replay_buffer.total_size)))

    def test_one_rollout_is_eligible_each_completed_cycle(self):
        tr3 = self._make_alg(
            actor_utd=1,
            critic_utd=1,
            rollout_cycles_per_collect=1)
        tr3._training_started = True
        tr3._train_mode = TrainMode.critic
        tr3._completed_cycles_since_rollout = 1
        self.assertFalse(tr3._local_rollout_skip_proposal()["should_skip"])
        tr3._after_unroll_iter_off_policy(True)
        self.assertEqual(tr3._completed_cycles_since_rollout, 0)


if __name__ == "__main__":
    alf.test.main()
