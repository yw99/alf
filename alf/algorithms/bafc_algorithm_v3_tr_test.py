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
from alf.algorithms.bafc_algorithm_v3_tr import (
    BafcActorInfo,
    BafcAlgorithmV3,
    BafcCriticInfo,
    BafcInfo,
)
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import LossInfo, StepType, TimeStep
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class _DummyProcess:

    def memory_info(self):
        return SimpleNamespace(rss=0)

    def children(self, recursive=True):
        del recursive
        return []


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

    def test_initialization_smoke(self):
        alg = self._make_alg()
        self.assertIsInstance(alg, BafcAlgorithmV3)
        self.assertEqual(alg._num_actor_critic, 3)
        self.assertEqual(alg._train_mode, TrainMode.standard)

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
        self.assertTrue(isinstance(step.info.eval_trust_metric, torch.Tensor))
        self.assertTrue(isinstance(step.info.grad_trust_metric, torch.Tensor))
        self.assertTrue(
            torch.isfinite(step.info.eval_trust_metric).item())
        self.assertTrue(
            torch.isfinite(step.info.grad_trust_metric).item())
        self.assertTrue(
            torch.isfinite(step.info.critic.eval_trust_metric).item())

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
                eval_trust_metric=torch.tensor(1.2)),
            discounted_return=torch.zeros(t, b),
            bootstrap_mask=torch.ones(t, b, n),
            eval_trust_metric=torch.tensor(1.2),
            grad_trust_metric=torch.tensor(1.5))

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

    def test_utd_mode_switch_via_after_update(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
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


if __name__ == "__main__":
    alf.test.main()
