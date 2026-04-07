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

    def test_trust_metric_update_interval_must_be_positive(self):
        with self.assertRaises(AssertionError):
            self._make_alg(trust_metric_update_interval=0)

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
            trust_metric_update_interval=2, monitor_trust_metrics=True)
        inputs = self._make_train_time_step(batch_size=4)

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
        beh_action = torch.full((3, alg._num_actor_critic, 2), -0.25)
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
                    alg._behavior_actor_networks,
                    "forward",
                    return_value=(beh_action, ())), mock.patch.object(
                        alg,
                        "_compute_actor_encoding",
                        return_value=torch.zeros(
                            alg._num_actor_critic, 1)):

            def _feature_map(_obs, _actor_encoding, action, critic_network=None):
                del _obs, _actor_encoding, critic_network
                if torch.equal(action, ref_action):
                    return phi_ref.clone()
                if torch.equal(action, beh_action):
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
        self.assertLess(c2.item(), 1e-6)

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

        self.assertLess(c2_before.item(), 1e-6)
        self.assertLess(c2_after.item(), 1e-6)
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

        self.assertLess(c2_constant.item(), 1e-6)
        self.assertGreater(c2_action.item(), c2_constant.item() + 1e-6)

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

    def test_snapshot_feature_map_matches_penultimate_critic_layer(self):
        alg = self._make_alg(trust_metric_num_obs=3)
        obs = torch.randn(5, 4)
        action = alg._ensure_group_action(
            alg._reference_actor_networks(obs)[0]).detach()
        actor_encoding = alg._compute_actor_encoding(
            alg._reference_actor_networks).detach()

        phi = alg._compute_snapshot_feature_map(obs, actor_encoding, action)
        critic = alg._snapshot_critic_networks
        critic_core = getattr(critic, "_pnet", critic)
        head_idx = alg._critic_feature_head_index(critic)
        head = critic_core._networks[head_idx]
        q_from_phi = head(phi)
        if isinstance(q_from_phi, tuple):
            q_from_phi = q_from_phi[0]
        critic_obs = obs.unsqueeze(1).expand(-1, alg._num_actor_critic, -1)
        q_direct = critic(
            (actor_encoding.unsqueeze(0).expand(obs.shape[0], -1, -1),
             (critic_obs, action)))[0]
        if q_from_phi.ndim == q_direct.ndim + 1 and q_from_phi.shape[-1] == 1:
            q_from_phi = q_from_phi.squeeze(-1)

        self.assertEqual(tuple(phi.shape[:2]), tuple(q_direct.shape[:2]))
        self.assertGreater(phi.shape[-1], 1)
        self.assertNotEqual(phi.shape[-1], action.shape[-1])
        self.assertEqual(tuple(q_from_phi.shape), tuple(q_direct.shape))
        self.assertTensorClose(q_from_phi, q_direct)

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


    def test_eval_gate_low_trust_no_longer_skips_rollout(self):
        alg = self._make_alg(
            enable_eval_rollout_skip_gate=True,
            eval_gate_max_consecutive_rollout_actor_holds=2)
        alg._last_eval_trust = torch.tensor(1.0)

        self.assertFalse(alg.request_skip_rollout_iter())
        self.assertFalse(alg._last_eval_gate_skip_decision)
        self.assertFalse(alg._last_rollout_skip_due_eval_gate)
        self.assertEqual(alg._eval_gate_skip_count, 0)
        self.assertEqual(alg._rollout_skip_due_eval_gate_count, 0)

    def test_critic_mode_skip_not_counted_as_eval_gate_skip(self):
        alg = self._make_alg(enable_eval_rollout_skip_gate=True)
        alg._training_started = True
        alg._train_mode = TrainMode.critic
        alg._last_eval_trust = torch.tensor(0.1)

        self.assertTrue(alg.request_skip_rollout_iter())
        self.assertFalse(alg._last_eval_gate_skip_decision)
        self.assertFalse(alg._last_rollout_skip_due_eval_gate)
        self.assertEqual(alg._eval_gate_skip_count, 0)
        self.assertEqual(alg._rollout_skip_due_eval_gate_count, 0)

    def test_low_eval_trust_holds_behavior_actor_and_advances_reference(self):
        alg = self._make_alg(
            enable_eval_rollout_skip_gate=True,
            monitor_trust_metrics=False)
        inputs = self._make_train_time_step(batch_size=4)
        behavior_before = self._clone_state_dict(alg._behavior_actor_networks)

        self._fill_module(alg._actor_networks, 0.5)
        actor_after = self._clone_state_dict(alg._actor_networks)
        alg._last_eval_trust = torch.tensor(1.0)

        alg.after_update(inputs, BafcInfo())

        self._assert_state_dict_equal(alg._behavior_actor_networks,
                                      behavior_before)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      actor_after)
        self.assertEqual(alg._eval_gate_consecutive_rollout_actor_holds, 1)
        self.assertEqual(alg._rollout_actor_hold_due_eval_gate_count, 1)
        self.assertTrue(alg._last_rollout_actor_held_due_eval_gate)
        self.assertFalse(alg._last_rollout_actor_refreshed_from_reference)
        self.assertFalse(
            alg._last_rollout_actor_refresh_forced_by_eval_gate_cap)

    def test_high_eval_trust_refreshes_behavior_actor_and_resets_hold_counter(self):
        alg = self._make_alg(
            enable_eval_rollout_skip_gate=True,
            monitor_trust_metrics=False)
        inputs = self._make_train_time_step(batch_size=4)

        self._fill_module(alg._behavior_actor_networks, -1.0)
        self._fill_module(alg._reference_actor_networks, 1.0)
        self._fill_module(alg._actor_networks, 2.0)
        reference_before = self._clone_state_dict(alg._reference_actor_networks)
        actor_after = self._clone_state_dict(alg._actor_networks)
        alg._eval_gate_consecutive_rollout_actor_holds = 2
        alg._last_eval_trust = torch.tensor(10.0)

        alg.after_update(inputs, BafcInfo())

        self._assert_state_dict_equal(alg._behavior_actor_networks,
                                      reference_before)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      actor_after)
        self.assertEqual(alg._eval_gate_consecutive_rollout_actor_holds, 0)
        self.assertFalse(alg._last_rollout_actor_held_due_eval_gate)
        self.assertTrue(alg._last_rollout_actor_refreshed_from_reference)
        self.assertFalse(
            alg._last_rollout_actor_refresh_forced_by_eval_gate_cap)

    def test_eval_gate_hold_cap_forces_rollout_actor_refresh(self):
        alg = self._make_alg(
            enable_eval_rollout_skip_gate=True,
            monitor_trust_metrics=False,
            eval_gate_max_consecutive_rollout_actor_holds=2)
        inputs = self._make_train_time_step(batch_size=4)

        self._fill_module(alg._behavior_actor_networks, -1.0)
        self._fill_module(alg._reference_actor_networks, 1.0)
        self._fill_module(alg._actor_networks, 2.0)
        reference_before = self._clone_state_dict(alg._reference_actor_networks)
        actor_after = self._clone_state_dict(alg._actor_networks)
        alg._eval_gate_consecutive_rollout_actor_holds = 2
        alg._last_eval_trust = torch.tensor(1.0)

        alg.after_update(inputs, BafcInfo())

        self._assert_state_dict_equal(alg._behavior_actor_networks,
                                      reference_before)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      actor_after)
        self.assertEqual(alg._eval_gate_consecutive_rollout_actor_holds, 0)
        self.assertEqual(alg._rollout_actor_hold_due_eval_gate_count, 0)
        self.assertFalse(alg._last_rollout_actor_held_due_eval_gate)
        self.assertTrue(alg._last_rollout_actor_refreshed_from_reference)
        self.assertTrue(
            alg._last_rollout_actor_refresh_forced_by_eval_gate_cap)

    def test_rollout_hold_cap_alias_old_name_only_works(self):
        alg = self._make_alg(eval_gate_max_consecutive_rollout_skips=7)

        self.assertEqual(alg._eval_gate_max_consecutive_rollout_actor_holds, 7)

    def test_rollout_hold_cap_new_name_only_works(self):
        alg = self._make_alg(
            eval_gate_max_consecutive_rollout_actor_holds=6)

        self.assertEqual(alg._eval_gate_max_consecutive_rollout_actor_holds, 6)

    def test_rollout_hold_cap_conflicting_aliases_fail(self):
        with self.assertRaisesRegex(
                ValueError,
                "must match when both are provided"):
            self._make_alg(
                eval_gate_max_consecutive_rollout_actor_holds=6,
                eval_gate_max_consecutive_rollout_skips=5)

    def test_grad_gate_extends_critic_block_and_bumps_counter(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            enable_grad_critic_extend_gate=True,
            monitor_trust_metrics=False)
        alg._train_mode = TrainMode.critic
        alg._critic_update_counter = 2

        alg._last_grad_trust = torch.tensor(1.0)
        before = alg._trust_metric_update_counter
        alg._update_train_mode()
        self.assertEqual(alg._train_mode, TrainMode.critic)
        self.assertTrue(alg._last_grad_gate_extended)
        self.assertEqual(alg._grad_gate_extension_count, 1)
        self.assertEqual(alg._trust_metric_update_counter, before + 1)

        alg._last_grad_trust = torch.tensor(3.0)
        before = alg._trust_metric_update_counter
        alg._update_train_mode()
        self.assertEqual(alg._train_mode, TrainMode.actor)
        self.assertFalse(alg._last_grad_gate_extended)
        self.assertEqual(alg._trust_metric_update_counter, before)

    def test_grad_gate_extension_cap_forces_actor_switch(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            enable_grad_critic_extend_gate=True,
            grad_gate_max_consecutive_critic_extensions=1,
            monitor_trust_metrics=False)
        alg._train_mode = TrainMode.critic
        alg._critic_update_counter = 2
        alg._last_grad_trust = torch.tensor(0.0)

        before = alg._trust_metric_update_counter
        alg._update_train_mode()
        self.assertEqual(alg._train_mode, TrainMode.critic)
        self.assertTrue(alg._last_grad_gate_extended)
        self.assertEqual(alg._grad_gate_consecutive_extensions, 1)
        self.assertEqual(alg._grad_gate_extension_count, 1)
        self.assertEqual(alg._trust_metric_update_counter, before + 1)

        before = alg._trust_metric_update_counter
        alg._update_train_mode()
        self.assertEqual(alg._train_mode, TrainMode.actor)
        self.assertFalse(alg._last_grad_gate_extended)
        self.assertEqual(alg._grad_gate_consecutive_extensions, 0)
        self.assertEqual(alg._grad_gate_extension_count, 1)
        self.assertEqual(alg._trust_metric_update_counter, before)

    def test_grad_gate_disabled_keeps_default_mode_switch(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            num_updates_per_train_iter=3,
            enable_grad_critic_extend_gate=False,
            monitor_trust_metrics=False,
            delta_trust_max=1e6)
        alg._train_mode = TrainMode.critic
        alg._critic_update_counter = 2
        alg._last_grad_trust = torch.tensor(0.0)

        before = alg._trust_metric_update_counter
        alg._update_train_mode()

        self.assertEqual(alg._train_mode, TrainMode.actor)
        self.assertFalse(alg._last_grad_gate_extended)
        self.assertEqual(alg._trust_metric_update_counter, before)

    def test_rollout_uses_behavior_actor_when_gates_enabled(self):
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

        self.assertIs(called['actor_net'], alg._behavior_actor_networks)


if __name__ == "__main__":
    alf.test.main()
