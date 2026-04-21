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
from alf.algorithms.bafc_algorithm_v36_tr import BafcAlgorithmV36, BafcInfo
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
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


class BafcAlgorithmV36TRTest(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._psutil_patcher = mock.patch(
            "alf.algorithms.algorithm.psutil.Process",
            return_value=_DummyProcess())
        self._psutil_patcher.start()
        self.addCleanup(self._psutil_patcher.stop)

    def _make_alg(self, **kwargs):
        num_updates_per_train_iter = kwargs.pop("num_updates_per_train_iter", 1)
        actor_eval_type = kwargs.pop("actor_eval_type", "last_two")
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_v36_tr_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=num_updates_per_train_iter)
        return BafcAlgorithmV36(
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
            actor_eval_type=actor_eval_type,
            num_actor_critic=3,
            num_actor_eval_samples=16,
            **kwargs)

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

    def test_manual_critic_all_group_rows_matches_row_vjp_reference(self):
        alg = self._make_alg()
        if not alg._supports_manual_critic_row_vjp():
            self.skipTest("Current critic layout does not support manual row VJP.")

        observation = torch.randn(5, 4)
        action = self._ensure_group_action(alg, observation)
        eval_full = alg._actor_networks(
            alg._actor_eval_samples, full_neurons=True)[0]
        actor_tokens = alg._tokenize_actor_out(eval_full[-2:])
        actor_encoding_base = alg._actor_encoder(actor_tokens)[0]
        obs_action_layers, joint_layers = (
            alg._get_manual_critic_parallel_fc_layers(alg._critic_networks))

        all_rows = alg._manual_critic_all_group_rows(
            observation,
            actor_encoding_base,
            action,
            feature_coords=None,
            include_feature_rows=True,
            obs_action_layers=obs_action_layers,
            joint_layers=joint_layers)
        feature_dim = all_rows["phi"].shape[-1]
        feature_coords = torch.arange(feature_dim, device=observation.device)

        for group_idx in range(alg._num_actor_critic):
            ref = alg._manual_critic_group_rows(
                observation,
                actor_encoding_base,
                action,
                group_idx=group_idx,
                feature_coords=feature_coords,
                obs_action_layers=obs_action_layers,
                joint_layers=joint_layers)
            self.assertTensorClose(all_rows["q_value"][:, group_idx], ref["q_value"])
            self.assertTensorClose(all_rows["dqda"][:, group_idx, :], ref["dqda"])
            self.assertTensorClose(all_rows["dqu"][group_idx], ref["dqu"])
            self.assertTensorClose(
                all_rows["feature_action_rows"][:, :, group_idx, :],
                ref["feature_action_rows"])
            self.assertTensorClose(
                all_rows["feature_du_rows"][:, group_idx, :], ref["feature_du_rows"])

    def test_manual_actor_step_backward_works_without_preupdate_metric(self):
        alg = self._make_alg()
        if not alg._supports_manual_actor_step_rows():
            self.skipTest("Current setup does not use the manual actor-step row path.")

        train_state = alg.get_initial_train_state(batch_size=4)
        inputs = self._make_train_time_step(batch_size=4)
        rollout_info = self._make_rollout_info(
            batch_size=4, num_actor_critic=alg._num_actor_critic)

        with mock.patch.object(
                alg,
                "_should_compute_actor_step_preupdate_grad_metric",
                return_value=False):
            step = alg.train_step(inputs, train_state, rollout_info)
            loss = step.info.actor.loss.mean()
            loss.backward()

        actor_grads = [
            p.grad for p in alg._actor_networks.parameters() if p.requires_grad
        ]
        self.assertTrue(any(g is not None for g in actor_grads))

    def test_manual_actor_step_backward_works_with_preupdate_metric(self):
        alg = self._make_alg()
        if not alg._supports_actor_step_preupdate_grad_metric():
            self.skipTest(
                "Current setup does not support actor-step preupdate grad metric.")

        train_state = alg.get_initial_train_state(batch_size=4)
        inputs = self._make_train_time_step(batch_size=4)
        rollout_info = self._make_rollout_info(
            batch_size=4, num_actor_critic=alg._num_actor_critic)

        with mock.patch.object(
                alg,
                "_should_compute_actor_step_preupdate_grad_metric",
                return_value=True):
            step = alg.train_step(inputs, train_state, rollout_info)
            loss = step.info.actor.loss.mean()
            loss.backward()

        self.assertIsNotNone(alg._pending_preupdate_grad_trust)
        actor_grads = [
            p.grad for p in alg._actor_networks.parameters() if p.requires_grad
        ]
        self.assertTrue(any(g is not None for g in actor_grads))

    def test_unroll_iter_off_policy_gates_on_completed_cycles(self):
        alg = self._make_alg(actor_utd=1, critic_utd=1, rollout_cycles_per_collect=2)
        alg._training_started = True
        alg._train_mode = TrainMode.critic

        alg._completed_cycles_since_rollout = 1
        with mock.patch.object(
                RLAlgorithm, "_unroll_iter_off_policy", autospec=True) as super_unroll:
            unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()
            self.assertFalse(unrolled)
            self.assertIsNone(root_inputs)
            self.assertIsNone(rollout_info)
            super_unroll.assert_not_called()

        expected_inputs = object()
        expected_info = object()
        alg._completed_cycles_since_rollout = 2
        with mock.patch.object(
                RLAlgorithm,
                "_unroll_iter_off_policy",
                autospec=True,
                return_value=(True, expected_inputs, expected_info)) as super_unroll:
            unrolled, root_inputs, rollout_info = alg._unroll_iter_off_policy()
            self.assertTrue(unrolled)
            self.assertIs(root_inputs, expected_inputs)
            self.assertIs(rollout_info, expected_info)
            super_unroll.assert_called_once_with(alg)
            self.assertEqual(alg._completed_cycles_since_rollout, 0)

    def test_after_update_refresh_cadence_follows_last_actor_step_flag(self):
        alg = self._make_alg(
            monitor_trust_metrics=True, trust_metric_update_interval=2)
        inputs = self._make_train_time_step(batch_size=4)

        with mock.patch.object(
                alg,
                "_compute_grad_generalization_trust_metric",
                return_value=torch.tensor(2.5)) as grad_metric_mock:
            alg._last_update_had_actor_step = False
            alg.after_update(inputs, BafcInfo())
            self.assertEqual(grad_metric_mock.call_count, 0)
            self.assertEqual(alg._trust_metric_update_counter, 0)

            alg._last_update_had_actor_step = True
            alg.after_update(inputs, BafcInfo())
            self.assertEqual(grad_metric_mock.call_count, 1)
            self.assertEqual(alg._trust_metric_update_counter, 1)

            alg._last_update_had_actor_step = True
            alg.after_update(inputs, BafcInfo())
            self.assertEqual(grad_metric_mock.call_count, 1)
            self.assertEqual(alg._trust_metric_update_counter, 2)

    def test_after_update_consumes_pending_preupdate_grad_metric(self):
        alg = self._make_alg(
            monitor_trust_metrics=True, trust_metric_update_interval=1)
        inputs = self._make_train_time_step(batch_size=4)
        pending_metric = torch.tensor(3.25)
        alg._pending_preupdate_grad_trust = pending_metric
        alg._last_update_had_actor_step = True

        with mock.patch.object(
                alg,
                "_compute_grad_generalization_trust_metric",
                side_effect=AssertionError(
                    "Post-update grad metric should not run when pending metric exists.")):
            alg.after_update(inputs, BafcInfo())

        self.assertTensorClose(alg._last_grad_trust, pending_metric)
        self.assertIsNone(alg._pending_preupdate_grad_trust)

    @staticmethod
    def _ensure_group_action(alg, observation):
        action = alg._actor_networks(observation)[0]
        return alg._ensure_group_action(action).detach()
