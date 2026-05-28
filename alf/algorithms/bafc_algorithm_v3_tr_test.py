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
from alf.algorithms.bafc_algorithm_v3_tr import (
    BafcActorInfo,
    BafcAlgorithmV3,
    BafcCriticInfo,
    BafcInfo,
)
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import LossInfo, StepType, TimeStep
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils import dist_utils


class _DummyProcess:

    def memory_info(self):
        return SimpleNamespace(rss=0)

    def children(self, recursive=True):
        del recursive
        return []


class _ScaleActor(torch.nn.Module):

    def __init__(self, num_groups, action_dim):
        super().__init__()
        self._num_groups = num_groups
        self._action_dim = action_dim
        self.scale = torch.nn.Parameter(torch.zeros(num_groups, action_dim))

    def forward(self, observation, state=(), id=None, full_neurons=False):
        del full_neurons
        action = torch.exp(self.scale).unsqueeze(0) * observation[
            :, :self._action_dim].unsqueeze(1)
        if id is not None:
            action = action[:, id, :]
        return action, state


class _TinyContinuousEnv(object):

    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._observation_spec = TensorSpec((4, ), dtype='float32')
        self._action_spec = BoundedTensorSpec(
            shape=(2, ), dtype='float32', minimum=-1.0, maximum=1.0)
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
        self._prev_action = self._action_spec.zeros(outer_dims=(self._batch_size, ))
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

        self._current_time_step = TimeStep(
            observation=self._observation_spec.randn([self._batch_size]),
            step_type=step_type,
            reward=action.mean(dim=-1),
            discount=torch.zeros(self._batch_size),
            prev_action=self._prev_action,
            env_id=torch.arange(self._batch_size, dtype=torch.int32))
        self._prev_action = action
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step

    def close(self):
        pass


class BafcAlgorithmV3TRTest(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        # Some restricted environments cannot resolve the current PID via
        # psutil.Process(). Mocking keeps these unit tests deterministic.
        self._psutil_patcher = mock.patch(
            "alf.algorithms.algorithm.psutil.Process",
            return_value=_DummyProcess())
        self._psutil_patcher.start()
        self.addCleanup(self._psutil_patcher.stop)

    def _make_alg(self, **kwargs):
        num_updates_per_train_iter = kwargs.pop("num_updates_per_train_iter",
                                                3)
        actor_network_cls = kwargs.pop(
            "actor_network_cls",
            partial(ActorFCNetwork, fc_layer_params=(32, 32)))
        critic_network_cls = kwargs.pop(
            "critic_network_cls",
            partial(
                FuncCriticNetwork,
                obs_action_joint_fc_layer_params=(32, 32),
                actor_obs_action_joint_fc_layer_params=(32, 32)))
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_tr_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=num_updates_per_train_iter)
        return BafcAlgorithmV3(
            observation_spec=TensorSpec((4, )),
            action_spec=BoundedTensorSpec((2, ), minimum=-1.0, maximum=1.0),
            config=config,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            actor_encoder_cls=partial(
                TransformerEncoder, num_layers=2, num_attention_heads=1),
            num_actor_critic=3,
            num_actor_eval_samples=16,
            **kwargs)

    def _make_rollout_time_step(self, batch_size=1):
        return TimeStep(
            step_type=torch.full((batch_size, ),
                                 StepType.FIRST,
                                 dtype=torch.int64),
            reward=torch.zeros(batch_size),
            discount=torch.ones(batch_size),
            observation=torch.randn(batch_size, 4),
            prev_action=(),
            env_id=())

    def _make_train_time_step(self, batch_size=4):
        return TimeStep(
            step_type=torch.full((batch_size, ),
                                 StepType.MID,
                                 dtype=torch.int64),
            reward=torch.zeros(batch_size),
            discount=torch.ones(batch_size),
            observation=torch.randn(batch_size, 4),
            prev_action=(),
            env_id=())

    def _make_rollout_info(self, batch_size=4, num_actor_critic=3):
        return BafcInfo(
            action=torch.randn(batch_size, 2),
            discounted_return=torch.zeros(batch_size),
            bootstrap_mask=torch.ones(batch_size, num_actor_critic))

    def _clone_state_dict(self, module):
        return {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }

    def _assert_state_dict_equal(self, module, expected_state):
        actual_state = module.state_dict()
        self.assertEqual(set(actual_state.keys()), set(expected_state.keys()))
        for name, value in actual_state.items():
            self.assertTensorEqual(value, expected_state[name])

    def _fill_module(self, module, fill_value):
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(fill_value)

    def test_initialization_smoke(self):
        alg = self._make_alg()
        self.assertIsInstance(alg, BafcAlgorithmV3)
        self.assertEqual(alg._num_actor_critic, 3)
        self.assertEqual(alg._train_mode, TrainMode.standard)

    def test_eval_samples_default_to_trainable_parameter(self):
        alg = self._make_alg()

        self.assertIn("_actor_eval_samples", dict(alg.named_parameters()))
        self.assertNotIn("_actor_eval_samples", dict(alg.named_buffers()))

    def test_frozen_eval_samples_are_checkpointed_buffers(self):
        alg = self._make_alg(freeze_eval_samples=True)

        self.assertNotIn("_actor_eval_samples", dict(alg.named_parameters()))
        self.assertIn("_actor_eval_samples", dict(alg.named_buffers()))
        self.assertIn("_actor_eval_samples", alg.state_dict())
        self.assertFalse(alg._actor_eval_samples.requires_grad)

    def test_frozen_eval_samples_stay_frozen_across_mode_switches(self):
        alg = self._make_alg(
            freeze_eval_samples=True,
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            enable_grad_actor_extend_gate=False,
            monitor_trust_metrics=False)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_grad_trust = torch.tensor(0.0)

        alg._update_train_mode()
        self.assertEqual(alg._train_mode, TrainMode.critic)
        self.assertFalse(alg._actor_eval_samples.requires_grad)

        alg._critic_update_counter = 2
        alg._update_train_mode()
        self.assertEqual(alg._train_mode, TrainMode.actor)
        self.assertFalse(alg._actor_eval_samples.requires_grad)

    def test_freeze_eval_samples_rejects_eval_samples_optimizer(self):
        with self.assertRaisesRegex(AssertionError, "eval_samples_optimizer"):
            self._make_alg(
                freeze_eval_samples=True,
                eval_samples_optimizer=alf.optimizers.Adam(lr=1e-3))

    def test_policy_boundary_eval_state_round_trip(self):
        alg = self._make_alg()
        alg._training_started = True
        alg._rollout_actor_id = torch.tensor(2)

        state = alg.get_policy_boundary_eval_state()

        self.assertEqual(
            state, dict(training_started=True, rollout_actor_id=2))

        alg._training_started = False
        alg._rollout_actor_id = 0
        alg.set_policy_boundary_eval_state(state)

        self.assertTrue(alg._training_started)
        self.assertEqual(int(torch.as_tensor(alg._rollout_actor_id).item()), 2)

    def test_predict_step_selects_single_actor_with_layer_norm(self):
        alg = self._make_alg(
            actor_network_cls=partial(
                ActorFCNetwork, fc_layer_params=(32, 32), use_ln=True),
            actor_use_ln=True)
        alg._training_started = True
        alg._rollout_actor_id = torch.tensor(1)
        time_step = self._make_rollout_time_step(batch_size=3)
        state = alg.get_initial_predict_state(batch_size=3)

        with mock.patch.object(
                alg._actor_networks,
                "forward",
                wraps=alg._actor_networks.forward) as forward:
            pred = alg.predict_step(time_step, state)

        self.assertEqual(
            int(torch.as_tensor(forward.call_args.kwargs["id"]).item()), 1)
        full_action, _ = alg._actor_networks(
            time_step.observation, state=state.actor_network)
        self.assertTensorClose(pred.output, full_action[:, 1, :])

    def test_trust_metric_update_interval_must_be_positive(self):
        with self.assertRaises(AssertionError):
            self._make_alg(trust_metric_update_interval=0)

    def test_reference_actor_sync_interval_must_be_positive(self):
        with self.assertRaises(AssertionError):
            self._make_alg(reference_actor_sync_interval=0)

    def test_reference_actor_sync_interval_defaults_to_half_buffer(self):
        alg = self._make_alg()

        self.assertEqual(alg._reference_actor_sync_interval, 256)

    def test_custom_feature_coord_config_is_applied(self):
        alg = self._make_alg(trust_metric_num_feature_coords=5)

        self.assertEqual(alg._trust_metric_num_feature_coords, 5)
        coords = alg._sample_feature_coords(9, torch.device("cpu"))
        self.assertEqual(tuple(coords.shape), (5, ))

    def test_predict_and_rollout_shapes_with_bootstrap_mask(self):
        alg = self._make_alg(
            use_bootstrap_actors=True,
            use_bootstrap_critics=True,
            bootstrap_mask_prob=1.0)
        rollout_state = alg.get_initial_rollout_state(batch_size=1)
        time_step = self._make_rollout_time_step(batch_size=1)

        predict_step = alg.predict_step(time_step, rollout_state.action)
        self.assertEqual(tuple(predict_step.output.shape), (1, 2))

        rollout_step = alg.rollout_step(time_step, rollout_state)
        self.assertEqual(tuple(rollout_step.output.shape), (1, 2))
        self.assertEqual(
            tuple(rollout_step.info.bootstrap_mask.shape),
            (1, alg._num_actor_critic))
        self.assertEqual(tuple(rollout_step.info.eval_trust_metric.shape), (1, ))
        self.assertEqual(tuple(rollout_step.info.grad_trust_metric.shape), (1, ))
        self.assertTrue(
            torch.all((rollout_step.info.bootstrap_mask == 0)
                      | (rollout_step.info.bootstrap_mask == 1)))

    def test_train_step_shapes(self):
        alg = self._make_alg()
        batch_size = 4
        train_state = alg.get_initial_train_state(batch_size=batch_size)
        inputs = self._make_train_time_step(batch_size=batch_size)
        rollout_info = self._make_rollout_info(
            batch_size=batch_size, num_actor_critic=alg._num_actor_critic)

        step = alg.train_step(inputs, train_state, rollout_info)

        self.assertEqual(tuple(step.output.shape),
                         (batch_size, alg._num_actor_critic, 2))
        self.assertEqual(tuple(step.info.actor.loss.shape), (batch_size, ))
        self.assertEqual(
            tuple(step.info.actor.extra.eval_action_loss.shape), (batch_size, ))
        self.assertEqual(
            tuple(step.info.actor.extra.grad_trust_metric.shape), (batch_size, ))
        self.assertEqual(
            tuple(step.info.critic.critic.shape),
            (batch_size, alg._num_actor_critic, alg._num_actor_critic))
        self.assertEqual(tuple(step.info.critic.eval_trust_metric.shape),
                         (batch_size, ))
        self.assertEqual(tuple(step.info.eval_trust_metric.shape),
                         (batch_size, ))
        self.assertEqual(tuple(step.info.grad_trust_metric.shape),
                         (batch_size, ))
        self.assertTrue(
            torch.isfinite(step.info.eval_trust_metric).all().item())
        self.assertTrue(
            torch.isfinite(step.info.grad_trust_metric).all().item())
        self.assertTrue(
            torch.isfinite(step.info.critic.eval_trust_metric).all().item())

    def test_train_step_info_reshapes_like_distributed_collector(self):
        alg = self._make_alg()
        t, b = 2, 3
        batch_size = t * b
        train_state = alg.get_initial_train_state(batch_size=batch_size)
        inputs = self._make_train_time_step(batch_size=batch_size)
        rollout_info = self._make_rollout_info(
            batch_size=batch_size, num_actor_critic=alg._num_actor_critic)

        step = alg.train_step(inputs, train_state, rollout_info)
        info_params = dist_utils.distributions_to_params(step.info)
        reshaped = alf.nest.map_structure(
            lambda x: x.reshape(t, b, *x.shape[1:]), info_params)

        self.assertEqual(tuple(reshaped.eval_trust_metric.shape), (t, b))
        self.assertEqual(tuple(reshaped.grad_trust_metric.shape), (t, b))
        self.assertEqual(tuple(reshaped.actor.loss.shape), (t, b))
        self.assertEqual(tuple(reshaped.actor.extra.eval_action_loss.shape),
                         (t, b))
        self.assertEqual(tuple(reshaped.actor.extra.grad_trust_metric.shape),
                         (t, b))
        self.assertEqual(tuple(reshaped.critic.eval_trust_metric.shape), (t, b))

    def test_calc_loss_with_synthetic_critic_info(self):
        alg = self._make_alg(use_bootstrap_critics=True, bootstrap_mask_prob=0.5)
        t, b, n = 2, 3, alg._num_actor_critic
        info = BafcInfo(
            reward=torch.zeros(t, b),
            step_type=torch.full((t, b), StepType.MID, dtype=torch.int64),
            discount=torch.ones(t, b),
            action=torch.zeros(t, b, 2),
            actor=LossInfo(
                loss=torch.ones(t, b),
                extra=BafcActorInfo(
                    eval_action_loss=torch.ones(t, b),
                    grad_trust_metric=torch.full((t, b), 1.5))),
            critic=BafcCriticInfo(
                critic=torch.zeros(t, b, n, n),
                target_critic=torch.zeros(t, b, n, n),
                eval_trust_metric=torch.full((t, b), 1.2)),
            discounted_return=torch.zeros(t, b),
            bootstrap_mask=torch.ones(t, b, n),
            eval_trust_metric=torch.full((t, b), 1.2),
            grad_trust_metric=torch.full((t, b), 1.5))

        loss = alg.calc_loss(info)

        self.assertEqual(tuple(loss.loss.shape), (t, b))
        self.assertTrue(isinstance(loss.scalar_loss, torch.Tensor))
        self.assertEqual(tuple(loss.extra.actor.eval_action_loss.shape), (t, b))
        self.assertEqual(tuple(loss.extra.actor.grad_trust_metric.shape), (t, b))

    def test_after_update_computes_finite_trust_metrics(self):
        alg = self._make_alg(trust_metric_num_obs=2, monitor_trust_metrics=True)
        batch_size = 8
        inputs = self._make_train_time_step(batch_size=batch_size)
        rollout_info = self._make_rollout_info(
            batch_size=batch_size, num_actor_critic=alg._num_actor_critic)
        state = alg.get_initial_train_state(batch_size=batch_size)
        step = alg.train_step(inputs, state, rollout_info)

        alg.after_update(inputs, step.info)

        self.assertTrue(torch.isfinite(alg._last_eval_trust).item())
        self.assertTrue(torch.isfinite(alg._last_grad_trust).item())
        self.assertEqual(alg._last_eval_trust.ndim, 0)
        self.assertEqual(alg._last_grad_trust.ndim, 0)

    def test_after_update_respects_trust_metric_interval(self):
        alg = self._make_alg(
            trust_metric_update_interval=2,
            monitor_trust_metrics=True,
            enable_eval_rollout_skip_gate=True,
            enable_grad_actor_extend_gate=True)
        inputs = self._make_train_time_step(batch_size=4)
        alg._last_update_had_actor_step = True

        with mock.patch.object(
                alg,
                "_compute_eval_trust_metric",
                side_effect=[torch.tensor(1.25),
                             torch.tensor(2.5)]) as eval_mock, mock.patch.object(
                                 alg,
                                 "_compute_grad_generalization_trust_metric",
                                 side_effect=[torch.tensor(1.75),
                                              torch.tensor(3.5)]) as grad_mock:
            alg.after_update(inputs, BafcInfo())
            self.assertEqual(eval_mock.call_count, 1)
            self.assertEqual(grad_mock.call_count, 1)
            self.assertTensorClose(alg._last_eval_trust, torch.tensor(1.25))
            self.assertTensorClose(alg._last_grad_trust, torch.tensor(1.75))

            alg.after_update(inputs, BafcInfo())
            self.assertEqual(eval_mock.call_count, 1)
            self.assertEqual(grad_mock.call_count, 1)
            self.assertTensorClose(alg._last_eval_trust, torch.tensor(1.25))
            self.assertTensorClose(alg._last_grad_trust, torch.tensor(1.75))

            alg.after_update(inputs, BafcInfo())
            self.assertEqual(eval_mock.call_count, 2)
            self.assertEqual(grad_mock.call_count, 2)
            self.assertTensorClose(alg._last_eval_trust, torch.tensor(2.5))
            self.assertTensorClose(alg._last_grad_trust, torch.tensor(3.5))

    def test_after_update_eval_only_mode_skips_grad_metric_compute(self):
        alg = self._make_alg(
            trust_metric_update_interval=1,
            monitor_trust_metrics=True,
            enable_eval_rollout_skip_gate=True,
            enable_grad_actor_extend_gate=False)
        inputs = self._make_train_time_step(batch_size=4)
        alg._last_update_had_actor_step = True

        with mock.patch.object(
                alg,
                "_compute_eval_trust_metric",
                return_value=torch.tensor(1.25)) as eval_mock, mock.patch.object(
                    alg,
                    "_compute_grad_generalization_trust_metric",
                    side_effect=AssertionError(
                        "grad trust should not be computed when grad gate is disabled"
                    )) as grad_mock:
            alg.after_update(inputs, BafcInfo())
            self.assertEqual(eval_mock.call_count, 1)
            self.assertEqual(grad_mock.call_count, 0)
            self.assertTensorClose(alg._last_eval_trust, torch.tensor(1.25))
            self.assertTensorClose(alg._last_grad_trust, torch.tensor(1.0))

    def test_after_update_skips_eval_metric_when_eval_gate_disabled(self):
        alg = self._make_alg(
            trust_metric_update_interval=2,
            monitor_trust_metrics=True,
            enable_eval_rollout_skip_gate=False)
        inputs = self._make_train_time_step(batch_size=4)
        alg._last_update_had_actor_step = True

        with mock.patch.object(
                alg,
                "_compute_eval_trust_metric",
                side_effect=AssertionError(
                    "eval trust should not be computed when eval gate is disabled"
                )) as eval_mock, mock.patch.object(
                    alg,
                    "_compute_grad_generalization_trust_metric",
                    side_effect=AssertionError(
                        "grad trust should not be computed when grad gate is disabled"
                    )) as grad_mock:
            alg.after_update(inputs, BafcInfo())
            self.assertEqual(eval_mock.call_count, 0)
            self.assertEqual(grad_mock.call_count, 0)
            self.assertTensorClose(alg._last_eval_trust, torch.tensor(1.0))
            self.assertTensorClose(alg._last_grad_trust, torch.tensor(1.0))

            alg.after_update(inputs, BafcInfo())
            self.assertEqual(eval_mock.call_count, 0)
            self.assertEqual(grad_mock.call_count, 0)
            self.assertTensorClose(alg._last_eval_trust, torch.tensor(1.0))
            self.assertTensorClose(alg._last_grad_trust, torch.tensor(1.0))

            alg.after_update(inputs, BafcInfo())
            self.assertEqual(eval_mock.call_count, 0)
            self.assertEqual(grad_mock.call_count, 0)
            self.assertTensorClose(alg._last_eval_trust, torch.tensor(1.0))
            self.assertTensorClose(alg._last_grad_trust, torch.tensor(1.0))

    def test_eval_metric_info_stays_finite_when_eval_compute_is_skipped(self):
        alg = self._make_alg(
            monitor_trust_metrics=True, enable_eval_rollout_skip_gate=False)
        batch_size = 4
        inputs = self._make_train_time_step(batch_size=batch_size)
        rollout_info = self._make_rollout_info(
            batch_size=batch_size, num_actor_critic=alg._num_actor_critic)
        state = alg.get_initial_train_state(batch_size=batch_size)

        step1 = alg.train_step(inputs, state, rollout_info)
        alg.after_update(inputs, step1.info)
        step2 = alg.train_step(inputs, step1.state, rollout_info)

        self.assertEqual(tuple(step2.info.eval_trust_metric.shape), (batch_size, ))
        self.assertEqual(tuple(step2.info.critic.eval_trust_metric.shape),
                         (batch_size, ))
        self.assertTrue(torch.isfinite(step2.info.eval_trust_metric).all().item())
        self.assertTrue(
            torch.isfinite(step2.info.critic.eval_trust_metric).all().item())
        self.assertTensorClose(step2.info.eval_trust_metric,
                               torch.ones_like(step2.info.eval_trust_metric))
        self.assertTensorClose(
            step2.info.critic.eval_trust_metric,
            torch.ones_like(step2.info.critic.eval_trust_metric))

        rollout_state = alg.get_initial_rollout_state(batch_size=1)
        rollout_step = alg.rollout_step(
            self._make_rollout_time_step(batch_size=1), rollout_state)
        self.assertEqual(tuple(rollout_step.info.eval_trust_metric.shape), (1, ))
        self.assertTrue(
            torch.isfinite(rollout_step.info.eval_trust_metric).all().item())

    def test_calc_loss_is_independent_of_trust_metrics(self):
        alg = self._make_alg(use_bootstrap_critics=True, bootstrap_mask_prob=0.5)
        t, b, n = 2, 3, alg._num_actor_critic
        base_info = BafcInfo(
            reward=torch.zeros(t, b),
            step_type=torch.full((t, b), StepType.MID, dtype=torch.int64),
            discount=torch.ones(t, b),
            action=torch.zeros(t, b, 2),
            actor=LossInfo(
                loss=torch.ones(t, b),
                extra=BafcActorInfo(
                    eval_action_loss=torch.ones(t, b),
                    grad_trust_metric=torch.full((t, b), 1.5))),
            critic=BafcCriticInfo(
                critic=torch.zeros(t, b, n, n),
                target_critic=torch.zeros(t, b, n, n),
                eval_trust_metric=torch.full((t, b), 1.2)),
            discounted_return=torch.zeros(t, b),
            bootstrap_mask=torch.ones(t, b, n),
            eval_trust_metric=torch.full((t, b), 1.2),
            grad_trust_metric=torch.full((t, b), 1.5))
        changed_metrics = base_info._replace(
            actor=base_info.actor._replace(
                extra=base_info.actor.extra._replace(
                    grad_trust_metric=torch.full((t, b), 9.0))),
            critic=base_info.critic._replace(
                eval_trust_metric=torch.full((t, b), 8.0)),
            eval_trust_metric=torch.full((t, b), 7.0),
            grad_trust_metric=torch.full((t, b), 6.0))

        base_loss = alg.calc_loss(base_info)
        changed_loss = alg.calc_loss(changed_metrics)

        self.assertTensorClose(base_loss.loss, changed_loss.loss)
        self.assertTensorClose(base_loss.scalar_loss, changed_loss.scalar_loss)

    def test_eval_trust_metric_matches_feature_formula(self):
        alg = self._make_alg(trust_cov_reg=0.25)
        obs = torch.randn(3, 4)
        ref_action = torch.full((3, alg._num_actor_critic, 2), 0.5)
        cur_action = torch.full((3, alg._num_actor_critic, 2), -0.25)
        phi_ref = torch.tensor(
            [
                [[1.0, 0.0, 2.0], [0.5, 1.0, 0.0], [1.5, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [1.0, 0.5, 0.5], [0.5, 1.5, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.5, 1.5], [1.0, 0.5, 1.0]],
            ])
        phi_beh = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.5, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.5, 0.5], [0.0, 1.0, 0.5]],
                [[1.0, 1.0, 0.0], [0.5, 0.0, 1.0], [0.5, 0.5, 1.0]],
            ])

        with mock.patch.object(
                alg._reference_actor_networks,
                "forward",
                return_value=(ref_action, ())), mock.patch.object(
                    alg._actor_networks,
                    "forward",
                    return_value=(cur_action, ())), mock.patch.object(
                        alg,
                        "_compute_actor_encoding",
                        return_value=torch.zeros(
                            alg._num_actor_critic, 1)):

            def _feature_map(_obs, _actor_encoding, action, critic_network=None):
                del _obs, _actor_encoding, critic_network
                if torch.equal(action, ref_action):
                    return phi_ref.clone()
                if torch.equal(action, cur_action):
                    return phi_beh.clone()
                raise AssertionError("Unexpected action tensor passed to feature map")

            with mock.patch.object(
                    alg,
                    "_compute_snapshot_feature_map",
                    side_effect=_feature_map):
                trust = alg._compute_eval_trust_metric(obs)

        beh_by_group = phi_beh.permute(1, 0, 2)
        cov = beh_by_group.transpose(1, 2) @ beh_by_group / beh_by_group.shape[1]
        cov = cov + 0.25 * torch.eye(3).unsqueeze(0)
        cov_inv = torch.linalg.pinv(cov)
        ref_by_group = phi_ref.permute(1, 0, 2)
        manual = ((ref_by_group @ cov_inv) * ref_by_group).sum(-1).mean()
        self.assertTensorClose(trust, manual)

    def test_grad_trust_c2_zero_for_constant_frozen_features(self):
        alg = self._make_alg(
            trust_metric_num_obs=3, trust_metric_num_feature_coords=3)
        obs = torch.randn(5, 4)

        def _constant_feature_map(observation,
                                  actor_encoding,
                                  action,
                                  critic_network=None):
            del actor_encoding, action, critic_network
            return torch.ones(
                observation.shape[0],
                alg._num_actor_critic,
                4,
                dtype=observation.dtype,
                device=observation.device)

        with mock.patch.object(
                alg,
                "_compute_snapshot_feature_map",
                side_effect=_constant_feature_map):
            c1, c2 = alg._compute_grad_generalization_trust_components(obs)

        self.assertTrue(torch.isfinite(c1).item())
        self.assertTrue(torch.isfinite(c2).item())
        self.assertLess(c2.item(), 1e-5)

    def test_grad_trust_c1_tracks_policy_jacobian(self):
        alg = self._make_alg(
            trust_metric_num_obs=4, trust_metric_num_feature_coords=2)
        obs = torch.randn(6, 4)
        alg._actor_networks = _ScaleActor(alg._num_actor_critic, 2)
        alg._reference_actor_networks = _ScaleActor(alg._num_actor_critic, 2)

        def _constant_feature_map(observation,
                                  actor_encoding,
                                  action,
                                  critic_network=None):
            del actor_encoding, action, critic_network
            return torch.ones(
                observation.shape[0],
                alg._num_actor_critic,
                3,
                dtype=observation.dtype,
                device=observation.device)

        with mock.patch.object(
                alg,
                "_compute_actor_encoding",
                return_value=torch.zeros(alg._num_actor_critic, 1)), mock.patch.object(
                    alg,
                    "_compute_snapshot_feature_map",
                    side_effect=_constant_feature_map):
            c1_before, c2_before = alg._compute_grad_generalization_trust_components(
                obs)
            with torch.no_grad():
                alg._actor_networks.scale.fill_(1.5)
            c1_after, c2_after = alg._compute_grad_generalization_trust_components(
                obs)

        self.assertLess(c2_before.item(), 1e-5)
        self.assertLess(c2_after.item(), 1e-5)
        self.assertGreater(c1_after.item(), c1_before.item())

    def test_grad_trust_c2_increases_when_features_depend_on_actor(self):
        alg = self._make_alg(
            trust_metric_num_obs=4, trust_metric_num_feature_coords=3)
        obs = torch.randn(6, 4)

        def _constant_feature_map(observation,
                                  actor_encoding,
                                  action,
                                  critic_network=None):
            del actor_encoding, action, critic_network
            return torch.ones(
                observation.shape[0],
                alg._num_actor_critic,
                3,
                dtype=observation.dtype,
                device=observation.device)

        def _action_feature_map(observation,
                                actor_encoding,
                                action,
                                critic_network=None):
            del actor_encoding, critic_network
            action = alg._ensure_group_action(action)
            ones = torch.ones(
                observation.shape[0], alg._num_actor_critic, 1,
                dtype=action.dtype,
                device=action.device)
            return torch.cat([action, ones], dim=-1)

        with mock.patch.object(
                alg,
                "_compute_snapshot_feature_map",
                side_effect=_constant_feature_map):
            _, c2_constant = alg._compute_grad_generalization_trust_components(
                obs)
        with mock.patch.object(
                alg,
                "_compute_snapshot_feature_map",
                side_effect=_action_feature_map):
            _, c2_action = alg._compute_grad_generalization_trust_components(obs)

        self.assertLess(c2_constant.item(), 1e-5)
        self.assertGreater(c2_action.item(), c2_constant.item() + 1e-6)

    def test_batched_vjp_matches_scalar_reference(self):
        alg = self._make_alg(trust_metric_num_obs=4)
        obs = torch.randn(6, 4)
        actor_params = list(alg._actor_networks.parameters())
        param_requires_grad = [p.requires_grad for p in actor_params]
        for p in actor_params:
            if not p.requires_grad:
                p.requires_grad_(True)

        try:
            cur_action = alg._ensure_group_action(alg._actor_networks(obs)[0])
            output_means = cur_action.mean(dim=0)
            batched = alg._batched_output_grad_sq_norm(
                output_means, actor_params, retain_graph=True)

            scalar = torch.zeros_like(output_means)
            num_groups, output_dim = output_means.shape
            num_terms = num_groups * output_dim
            term_idx = 0
            for group_idx in range(num_groups):
                for output_idx in range(output_dim):
                    term_idx += 1
                    retain_graph = term_idx < num_terms
                    scalar[group_idx, output_idx] = alg._scalar_grad_sq_norm(
                        output_means[group_idx, output_idx], actor_params,
                        group_idx, retain_graph)
        finally:
            for p, req in zip(actor_params, param_requires_grad):
                p.requires_grad_(req)

        self.assertTensorClose(batched, scalar)

    def test_batched_vjp_chunk_size_consistency(self):
        alg = self._make_alg(trust_metric_num_obs=4)
        obs = torch.randn(6, 4)
        actor_params = list(alg._actor_networks.parameters())
        param_requires_grad = [p.requires_grad for p in actor_params]
        for p in actor_params:
            if not p.requires_grad:
                p.requires_grad_(True)

        try:
            output_means_full = alg._ensure_group_action(
                alg._actor_networks(obs)[0]).mean(dim=0)
            full_chunk = alg._batched_output_grad_sq_norm(
                output_means_full,
                actor_params,
                retain_graph=False,
                max_chunk_size=output_means_full.numel())

            output_means_small = alg._ensure_group_action(
                alg._actor_networks(obs)[0]).mean(dim=0)
            small_chunk = alg._batched_output_grad_sq_norm(
                output_means_small,
                actor_params,
                retain_graph=False,
                max_chunk_size=2)
        finally:
            for p, req in zip(actor_params, param_requires_grad):
                p.requires_grad_(req)

        self.assertTensorClose(full_chunk, small_chunk)

    def test_grad_trust_c2_matches_weighted_rms_formula(self):
        alg = self._make_alg(
            trust_metric_num_obs=2, trust_metric_num_feature_coords=2)
        obs = torch.randn(2, 4)
        ref_action = torch.ones(2, alg._num_actor_critic, 2)
        cur_action = torch.full((2, alg._num_actor_critic, 2), 2.0)
        phi_ref = torch.ones(2, alg._num_actor_critic, 3)
        phi_t = torch.full((2, alg._num_actor_critic, 3), 3.0)
        feature_coords = torch.tensor([0, 2], dtype=torch.int64)
        inv_cov = torch.tensor(
            [
                [[2.0, 0.0, 0.0], [0.0, 99.0, 0.0], [0.0, 0.0, 8.0]],
                [[3.0, 0.0, 0.0], [0.0, 99.0, 0.0], [0.0, 0.0, 12.0]],
                [[5.0, 0.0, 0.0], [0.0, 99.0, 0.0], [0.0, 0.0, 20.0]],
            ])
        grad_sq_phi = torch.tensor(
            [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]])

        def _batched_grad_sq(output_means,
                             params,
                             retain_graph,
                             max_chunk_size=None):
            del params, retain_graph, max_chunk_size
            if torch.allclose(output_means, torch.full_like(output_means, 2.0)):
                return torch.zeros_like(output_means)
            if torch.allclose(output_means, torch.full_like(output_means, 3.0)):
                return grad_sq_phi.clone()
            raise AssertionError(
                f"Unexpected output means passed to batched grad helper: "
                f"shape={tuple(output_means.shape)}")

        with mock.patch.object(
                alg._actor_networks,
                "forward",
                return_value=(cur_action, ())), mock.patch.object(
                    alg._reference_actor_networks,
                    "forward",
                    return_value=(ref_action, ())), mock.patch.object(
                        alg,
                        "_compute_actor_encoding",
                        return_value=torch.zeros(alg._num_actor_critic, 1)), mock.patch.object(
                            alg,
                            "_sample_feature_coords",
                            return_value=feature_coords), mock.patch.object(
                                alg,
                                "_compute_feature_inv_cov",
                                return_value=inv_cov), mock.patch.object(
                                    alg,
                                    "_batched_output_grad_sq_norm",
                                    side_effect=_batched_grad_sq):

            def _feature_map(_obs, _actor_encoding, action, critic_network=None):
                del _obs, _actor_encoding, critic_network
                if torch.equal(action, ref_action):
                    return phi_ref.clone()
                if torch.equal(action, cur_action):
                    return phi_t.clone()
                raise AssertionError("Unexpected action tensor passed to feature map")

            with mock.patch.object(
                    alg,
                    "_compute_snapshot_feature_map",
                    side_effect=_feature_map):
                _, c2 = alg._compute_grad_generalization_trust_components(obs)

        sampled_inv_diag = torch.tensor(
            [[2.0, 8.0], [3.0, 12.0], [5.0, 20.0]])
        coord_scale = 3.0 / 2.0
        manual = torch.sqrt(coord_scale * (sampled_inv_diag * grad_sq_phi).sum(-1)).mean()
        old_formula = (torch.sqrt(sampled_inv_diag * grad_sq_phi).sum(-1)
                       * coord_scale).mean()

        self.assertTensorClose(c2, manual)
        self.assertGreater(torch.abs(c2 - old_formula).item(), 1e-5)

    def test_grad_trust_does_not_touch_critic_parameter_grads(self):
        alg = self._make_alg(trust_metric_num_obs=3)
        obs = torch.randn(5, 4)

        for param in alg._critic_networks.parameters():
            param.grad = None
        for param in alg._snapshot_critic_networks.parameters():
            param.grad = None

        c1, c2 = alg._compute_grad_generalization_trust_components(obs)

        self.assertTrue(torch.isfinite(c1).item())
        self.assertTrue(torch.isfinite(c2).item())
        self.assertTrue(
            all(not param.requires_grad
                for param in alg._snapshot_critic_networks.parameters()))
        self.assertTrue(
            all(param.grad is None for param in alg._critic_networks.parameters()))
        self.assertTrue(
            all(param.grad is None
                for param in alg._snapshot_critic_networks.parameters()))

    def test_snapshot_feature_map_is_normalized_penultimate_critic_layer(self):
        alg = self._make_alg(trust_metric_num_obs=3)
        obs = torch.randn(5, 4)
        action = alg._ensure_group_action(
            alg._reference_actor_networks(obs)[0]).detach()
        actor_encoding = alg._compute_actor_encoding(
            alg._reference_actor_networks).detach()

        phi = alg._compute_snapshot_feature_map(obs, actor_encoding, action)
        feature_norm = phi.norm(p=2, dim=-1)
        nonzero = feature_norm > 1e-6

        self.assertEqual(tuple(phi.shape[:2]),
                         (obs.shape[0], alg._num_actor_critic))
        self.assertGreater(phi.shape[-1], 1)
        self.assertNotEqual(phi.shape[-1], action.shape[-1])
        self.assertTrue(torch.isfinite(phi).all().item())
        self.assertTrue(nonzero.any().item())
        self.assertLess(
            (feature_norm[nonzero] - 1.).abs().max().item(), 1e-5)

    def test_utd_mode_switch_via_after_update(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            trust_metric_update_interval=2,
            eval_trust_max=1e-6,
            delta_trust_max=1e-6,
            monitor_trust_metrics=True)
        batch_size = 4
        inputs = self._make_train_time_step(batch_size=batch_size)
        rollout_info = self._make_rollout_info(
            batch_size=batch_size, num_actor_critic=alg._num_actor_critic)
        state = alg.get_initial_train_state(batch_size=batch_size)

        self.assertEqual(alg._train_mode, TrainMode.critic)

        step1 = alg.train_step(inputs, state, rollout_info)
        alg.after_update(inputs, step1.info)
        self.assertEqual(alg._train_mode, TrainMode.critic)

        step2 = alg.train_step(inputs, step1.state, rollout_info)
        alg.after_update(inputs, step2.info)
        self.assertEqual(alg._train_mode, TrainMode.actor)

        step3 = alg.train_step(inputs, step2.state, rollout_info)
        self.assertEqual(step3.info.critic, BafcCriticInfo())
        alg.after_update(inputs, step3.info)
        self.assertEqual(alg._train_mode, TrainMode.critic)

    def test_constructor_allows_non_one_updates_per_train_iter(self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=3)

        self.assertIsInstance(alg, BafcAlgorithmV3)
        self.assertEqual(alg._train_mode, TrainMode.critic)
        self.assertEqual(alg._actor_utd, 1)
        self.assertEqual(alg._critic_utd, 3)

    def test_unroll_iter_off_policy_gates_on_completed_cycles(self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=2,
            rollout_cycles_per_collect=3,
            reference_actor_sync_interval=1)
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._completed_cycles_since_rollout = 2

        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                return_value=(True, "root", "info")) as parent_unroll, mock.patch.object(
                    alg, "_sync_reference_from_current") as sync_mock:
            unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()

        self.assertFalse(unrolled)
        self.assertIsNone(root_inputs)
        self.assertIsNone(rollout_info)
        parent_unroll.assert_not_called()
        sync_mock.assert_not_called()

        # Above-threshold cycle counts should also be eligible for rollout.
        alg._completed_cycles_since_rollout = 5
        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                return_value=(True, "root", "info")) as parent_unroll, mock.patch.object(
                    alg, "_sync_reference_from_current") as sync_mock:
            unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()

        self.assertTrue(unrolled)
        self.assertEqual(root_inputs, "root")
        self.assertEqual(rollout_info, "info")
        self.assertEqual(alg._completed_cycles_since_rollout, 0)
        parent_unroll.assert_called_once()
        sync_mock.assert_called_once()

    def test_reference_actor_sync_interval_delays_sync(self):
        alg = self._make_alg(reference_actor_sync_interval=3)

        with mock.patch.object(alg, "_sync_reference_from_current") as sync_mock:
            alg._after_unroll_iter_off_policy(False)
            self.assertEqual(alg._real_rollouts_since_reference_sync, 0)
            sync_mock.assert_not_called()

            alg._after_unroll_iter_off_policy(True)
            self.assertEqual(alg._real_rollouts_since_reference_sync, 1)
            sync_mock.assert_not_called()

            alg._after_unroll_iter_off_policy(True)
            self.assertEqual(alg._real_rollouts_since_reference_sync, 2)
            sync_mock.assert_not_called()

            alg._after_unroll_iter_off_policy(True)
            self.assertEqual(alg._real_rollouts_since_reference_sync, 0)
            sync_mock.assert_called_once()

    def test_unroll_iter_off_policy_skips_rollout_when_eval_gate_blocks(self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=2,
            rollout_cycles_per_collect=3,
            enable_eval_rollout_skip_gate=True,
            monitor_trust_metrics=False)
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._completed_cycles_since_rollout = 5
        alg._last_eval_trust = torch.tensor(1.0)

        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                return_value=(True, "root", "info")) as parent_unroll, mock.patch.object(
                    alg, "_sync_reference_from_current") as sync_mock:
            unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()

        self.assertFalse(unrolled)
        self.assertIsNone(root_inputs)
        self.assertIsNone(rollout_info)
        self.assertEqual(alg._completed_cycles_since_rollout, 5)
        self.assertEqual(alg._rollout_opportunity_count, 1)
        self.assertEqual(alg._eval_gate_consecutive_rollout_skips, 1)
        self.assertEqual(alg._rollout_skip_due_eval_gate_count, 1)
        self.assertTrue(alg._last_rollout_skipped_due_eval_gate)
        parent_unroll.assert_not_called()
        sync_mock.assert_not_called()

    def test_unroll_iter_off_policy_eval_skip_cap_allows_rollout(self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=2,
            rollout_cycles_per_collect=3,
            enable_eval_rollout_skip_gate=True,
            monitor_trust_metrics=False,
            eval_gate_max_consecutive_rollout_skips=2,
            reference_actor_sync_interval=1)
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._completed_cycles_since_rollout = 3
        alg._last_eval_trust = torch.tensor(1.0)
        alg._eval_gate_consecutive_rollout_skips = 2

        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                return_value=(True, "root", "info")) as parent_unroll, mock.patch.object(
                    alg, "_sync_reference_from_current") as sync_mock:
            unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()

        self.assertTrue(unrolled)
        self.assertEqual(root_inputs, "root")
        self.assertEqual(rollout_info, "info")
        self.assertEqual(alg._completed_cycles_since_rollout, 0)
        self.assertEqual(alg._eval_gate_consecutive_rollout_skips, 0)
        self.assertFalse(alg._last_rollout_skipped_due_eval_gate)
        parent_unroll.assert_called_once()
        sync_mock.assert_called_once()

    def test_rollout_skip_events_track_start_continue_and_end(self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=2,
            rollout_cycles_per_collect=3,
            enable_eval_rollout_skip_gate=True,
            monitor_trust_metrics=False)
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._last_eval_trust = torch.tensor(1.0)

        alg._completed_cycles_since_rollout = 2
        self.assertTrue(alg._should_skip_unroll_iter_off_policy())
        self.assertEqual(alg._rollout_opportunity_count, 0)
        self.assertIsNone(alg._pop_rollout_skip_event())

        alg._completed_cycles_since_rollout = 3
        self.assertTrue(alg._should_skip_unroll_iter_off_policy())
        self.assertEqual(alg._rollout_opportunity_count, 1)
        self.assertEqual(
            alg._pop_rollout_skip_event(),
            dict(
                type="skip_start",
                start_rollout_opportunity=1,
                end_rollout_opportunity=1,
                skip_length=1))

        alg._completed_cycles_since_rollout = 3
        self.assertTrue(alg._should_skip_unroll_iter_off_policy())
        self.assertEqual(alg._rollout_opportunity_count, 2)
        self.assertIsNone(alg._pop_rollout_skip_event())

        alg._last_eval_trust = torch.tensor(10.0)
        alg._completed_cycles_since_rollout = 3
        self.assertFalse(alg._should_skip_unroll_iter_off_policy())
        self.assertEqual(alg._rollout_opportunity_count, 3)
        self.assertEqual(
            alg._pop_rollout_skip_event(),
            dict(
                type="skip_end",
                start_rollout_opportunity=1,
                end_rollout_opportunity=3,
                skip_length=2))

    def test_rollout_skip_cap_emits_end_event(self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=2,
            rollout_cycles_per_collect=3,
            enable_eval_rollout_skip_gate=True,
            monitor_trust_metrics=False,
            eval_gate_max_consecutive_rollout_skips=2)
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._last_eval_trust = torch.tensor(1.0)

        for _ in range(2):
            alg._completed_cycles_since_rollout = 3
            self.assertTrue(alg._should_skip_unroll_iter_off_policy())
            alg._pop_rollout_skip_event()

        alg._completed_cycles_since_rollout = 3
        self.assertFalse(alg._should_skip_unroll_iter_off_policy())
        self.assertEqual(
            alg._pop_rollout_skip_event(),
            dict(
                type="skip_end",
                start_rollout_opportunity=1,
                end_rollout_opportunity=3,
                skip_length=2))

    def test_unroll_iter_off_policy_delegates_during_warmup_or_standard_mode(
            self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=2,
            rollout_cycles_per_collect=3,
            reference_actor_sync_interval=1)
        alg._completed_cycles_since_rollout = 0

        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                return_value=(False, "warmup", "none")) as parent_unroll, mock.patch.object(
                    alg, "_sync_reference_from_current") as sync_mock:
            alg._training_started = False
            alg._train_mode = TrainMode.critic
            unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()
        self.assertFalse(unrolled)
        self.assertEqual(root_inputs, "warmup")
        self.assertEqual(rollout_info, "none")
        parent_unroll.assert_called_once()
        sync_mock.assert_not_called()

        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                return_value=(True, "std", "info")) as parent_unroll, mock.patch.object(
                    alg, "_sync_reference_from_current") as sync_mock:
            alg._training_started = True
            alg._train_mode = TrainMode.standard
            unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()
        self.assertTrue(unrolled)
        self.assertEqual(root_inputs, "std")
        self.assertEqual(rollout_info, "info")
        parent_unroll.assert_called_once()
        sync_mock.assert_called_once()

    def test_train_iter_off_policy_keeps_single_replay_pass_per_outer_iter(self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=2,
            rollout_cycles_per_collect=3)
        alg._replay_buffer = object()
        alg._training_started = True
        alg._train_mode = TrainMode.critic

        with mock.patch.object(
                alg,
                "_unroll_iter_off_policy",
                return_value=(False, None, None)) as unroll_mock, mock.patch.object(
                    alg,
                    "train_from_replay_buffer",
                    return_value=6144) as replay_mock, mock.patch.object(
                        alg, "after_train_iter") as after_iter_mock:
            steps = alg._train_iter_off_policy()

        self.assertEqual(steps, 6144)
        unroll_mock.assert_called_once()
        replay_mock.assert_called_once_with(update_global_counter=True)
        after_iter_mock.assert_not_called()

    def test_unroll_skip_increments_counter_when_not_per_minibatch(self):
        alg = self._make_alg(
            num_updates_per_train_iter=12,
            actor_utd=1,
            critic_utd=2,
            rollout_cycles_per_collect=3)
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._completed_cycles_since_rollout = 2
        self.assertFalse(alg._config.update_counter_every_mini_batch)

        old_counter = int(alf.summary.get_global_counter())
        try:
            alf.summary.set_global_counter(100)
            with mock.patch.object(RLAlgorithm, "_unroll_iter_off_policy") as parent_unroll:
                unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()

            self.assertFalse(unrolled)
            self.assertIsNone(root_inputs)
            self.assertIsNone(rollout_info)
            self.assertEqual(int(alf.summary.get_global_counter()), 101)
            parent_unroll.assert_not_called()
        finally:
            alf.summary.set_global_counter(old_counter)

    def test_after_update_does_not_sync_reference_from_current(self):
        alg = self._make_alg(
            enable_eval_rollout_skip_gate=True,
            monitor_trust_metrics=False)
        inputs = self._make_train_time_step(batch_size=4)

        with mock.patch.object(alg, "_sync_reference_from_current") as sync_mock:
            alg.after_update(inputs, BafcInfo())

        sync_mock.assert_not_called()

    def test_rollout_skip_cap_config_works(self):
        alg = self._make_alg(eval_gate_max_consecutive_rollout_skips=7)
        self.assertEqual(alg._eval_gate_max_consecutive_rollout_skips, 7)

    def test_constructor_rejects_removed_rollout_hold_cap_arg(self):
        with self.assertRaisesRegex(
                TypeError, "eval_gate_max_consecutive_rollout_actor_holds"):
            self._make_alg(eval_gate_max_consecutive_rollout_actor_holds=6)

    def test_grad_gate_extends_actor_block_without_counter_bump(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            enable_grad_actor_extend_gate=True,
            monitor_trust_metrics=False)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1

        old_counter = int(alf.summary.get_global_counter())
        try:
            alf.summary.set_global_counter(10)
            alg._last_grad_trust = torch.tensor(1.0)
            before = alg._trust_metric_update_counter
            alg._update_train_mode()
            self.assertEqual(alg._train_mode, TrainMode.actor)
            self.assertTrue(alg._last_grad_gate_actor_extended)
            self.assertEqual(alg._grad_gate_actor_extension_count, 1)
            self.assertEqual(alg._trust_metric_update_counter, before)
            self.assertEqual(
                alg._pop_grad_extension_event(),
                dict(
                    type="grad_extension_start",
                    start_step=10,
                    end_step=10,
                    extension_length=1))

            alf.summary.set_global_counter(11)
            alg._last_grad_trust = torch.tensor(1.0)
            before = alg._trust_metric_update_counter
            alg._update_train_mode()
            self.assertEqual(alg._train_mode, TrainMode.actor)
            self.assertTrue(alg._last_grad_gate_actor_extended)
            self.assertEqual(alg._grad_gate_actor_extension_count, 2)
            self.assertEqual(alg._trust_metric_update_counter, before)
            self.assertIsNone(alg._pop_grad_extension_event())

            alf.summary.set_global_counter(12)
            alg._last_grad_trust = torch.tensor(3.0)
            before = alg._trust_metric_update_counter
            alg._update_train_mode()
            self.assertEqual(alg._train_mode, TrainMode.critic)
            self.assertFalse(alg._last_grad_gate_actor_extended)
            self.assertEqual(alg._trust_metric_update_counter, before)
            self.assertEqual(
                alg._pop_grad_extension_event(),
                dict(
                    type="grad_extension_end",
                    start_step=10,
                    end_step=12,
                    extension_length=2))
        finally:
            alf.summary.set_global_counter(old_counter)

    def test_grad_gate_extension_cap_forces_critic_switch_without_counter_bump(
            self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            enable_grad_actor_extend_gate=True,
            grad_gate_max_consecutive_actor_extensions=1,
            monitor_trust_metrics=False)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_grad_trust = torch.tensor(0.0)

        old_counter = int(alf.summary.get_global_counter())
        try:
            alf.summary.set_global_counter(20)
            before = alg._trust_metric_update_counter
            alg._update_train_mode()
            self.assertEqual(alg._train_mode, TrainMode.actor)
            self.assertTrue(alg._last_grad_gate_actor_extended)
            self.assertEqual(alg._grad_gate_consecutive_actor_extensions, 1)
            self.assertEqual(alg._grad_gate_actor_extension_count, 1)
            self.assertEqual(alg._trust_metric_update_counter, before)
            self.assertEqual(
                alg._pop_grad_extension_event(),
                dict(
                    type="grad_extension_start",
                    start_step=20,
                    end_step=20,
                    extension_length=1))

            alf.summary.set_global_counter(21)
            before = alg._trust_metric_update_counter
            alg._update_train_mode()
            self.assertEqual(alg._train_mode, TrainMode.critic)
            self.assertFalse(alg._last_grad_gate_actor_extended)
            self.assertEqual(alg._grad_gate_consecutive_actor_extensions, 0)
            self.assertEqual(alg._grad_gate_actor_extension_count, 1)
            self.assertEqual(alg._trust_metric_update_counter, before)
            self.assertEqual(
                alg._pop_grad_extension_event(),
                dict(
                    type="grad_extension_end",
                    start_step=20,
                    end_step=21,
                    extension_length=1))
        finally:
            alf.summary.set_global_counter(old_counter)

    def test_grad_gate_disabled_keeps_default_mode_switch(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            enable_grad_actor_extend_gate=False,
            monitor_trust_metrics=False,
            delta_trust_max=1e6)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_grad_trust = torch.tensor(0.0)

        before = alg._trust_metric_update_counter
        alg._update_train_mode()

        self.assertEqual(alg._train_mode, TrainMode.critic)
        self.assertFalse(alg._last_grad_gate_actor_extended)
        self.assertEqual(alg._trust_metric_update_counter, before)
        self.assertIsNone(alg._pop_grad_extension_event())

    def test_critic_boundary_switches_to_actor_without_extension(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            enable_grad_actor_extend_gate=True,
            monitor_trust_metrics=False,
            delta_trust_max=1e6)
        alg._train_mode = TrainMode.critic
        alg._critic_update_counter = 2
        alg._last_grad_trust = torch.tensor(0.0)

        before = alg._trust_metric_update_counter
        alg._update_train_mode()

        self.assertEqual(alg._train_mode, TrainMode.actor)
        self.assertFalse(alg._last_grad_gate_actor_extended)
        self.assertEqual(alg._grad_gate_actor_extension_count, 0)
        self.assertEqual(alg._trust_metric_update_counter, before)

    def test_rollout_uses_latest_actor_when_eval_gate_enabled(self):
        alg = self._make_alg(enable_eval_rollout_skip_gate=True)
        rollout_state = alg.get_initial_rollout_state(batch_size=1)
        time_step = self._make_rollout_time_step(batch_size=1)
        called = {}

        def _fake_predict(actor_net, observation, state, train=False):
            del observation, train
            called['actor_net'] = actor_net
            return torch.zeros(1, 2), state

        with mock.patch.object(alg, "_predict_action", side_effect=_fake_predict):
            alg.rollout_step(time_step, rollout_state)

        self.assertIs(called['actor_net'], alg._actor_networks)

    def test_rollout_uses_train_actor_when_only_grad_gate_enabled(self):
        alg = self._make_alg(enable_grad_actor_extend_gate=True)
        rollout_state = alg.get_initial_rollout_state(batch_size=1)
        time_step = self._make_rollout_time_step(batch_size=1)
        called = {}

        def _fake_predict(actor_net, observation, state, train=False):
            del observation, train
            called['actor_net'] = actor_net
            return torch.zeros(1, 2), state

        with mock.patch.object(alg, "_predict_action", side_effect=_fake_predict):
            alg.rollout_step(time_step, rollout_state)

        self.assertIs(called['actor_net'], alg._actor_networks)

    def test_agent_rollout_continues_after_training_starts(self):
        env = _TinyContinuousEnv(batch_size=1)
        with tempfile.TemporaryDirectory() as root_dir:
            config = TrainerConfig(
                root_dir=root_dir,
                unroll_length=2,
                mini_batch_length=2,
                mini_batch_size=1,
                initial_collect_steps=2,
                replay_buffer_length=32,
                num_updates_per_train_iter=3)
            agent = Agent(
                observation_spec=env.observation_spec(),
                action_spec=env.action_spec(),
                env=env,
                config=config,
                optimizer=alf.optimizers.Adam(lr=1e-3),
                rl_algorithm_cls=partial(
                    BafcAlgorithmV3,
                    actor_network_cls=partial(ActorFCNetwork, fc_layer_params=(16, 16)),
                    critic_network_cls=partial(
                        FuncCriticNetwork,
                        obs_action_joint_fc_layer_params=(16, 16),
                        actor_obs_action_joint_fc_layer_params=(16, 16)),
                    actor_encoder_cls=partial(
                        TransformerEncoder, num_layers=2, num_attention_heads=1),
                    num_actor_critic=3,
                    num_actor_eval_samples=8,
                    actor_utd=1,
                    critic_utd=2,
                    enable_eval_rollout_skip_gate=True,
                    enable_grad_actor_extend_gate=True,
                    monitor_trust_metrics=False))

            for _ in range(8):
                agent.train_iter()

            self.assertTrue(agent._rl_algorithm._training_started)
            self.assertGreater(agent.get_step_metrics()[1].result().item(), 1)


if __name__ == "__main__":
    alf.test.main()
