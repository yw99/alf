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
from alf.algorithms.bafc_algorithm_v3_tr2 import (BafcAlgorithmV3TR2,
                                                  BafcCriticInfo, BafcInfo)
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


class BafcAlgorithmV3TR2Test(alf.test.TestCase):

    def setUp(self):
        super().setUp()
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
            root_dir=tempfile.mkdtemp(prefix="bafc_v3_tr2_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=num_updates_per_train_iter)
        kwargs.setdefault("trust_metric_num_obs", 8)
        return BafcAlgorithmV3TR2(
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

    def _make_train_time_step(self, observation=None):
        if observation is None:
            observation = torch.randn(4, 4)
        batch_size = observation.shape[0]
        return TimeStep(
            step_type=torch.full((batch_size, ), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(batch_size),
            discount=torch.ones(batch_size),
            observation=observation,
            prev_action=(),
            env_id=())

    def _clone_state_dict(self, module):
        return {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }

    def _assert_state_dict_equal(self, module, expected_state):
        actual_state = module.state_dict()
        self.assertEqual(set(actual_state.keys()), set(expected_state.keys()))
        for name, value in actual_state.items():
            self.assertTensorClose(value, expected_state[name])

    def _fill_module(self, module, fill_value):
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(fill_value)

    def test_initialization_smoke_and_config_name(self):
        alg = self._make_alg()
        self.assertIsInstance(alg, BafcAlgorithmV3TR2)
        self.assertEqual(BafcAlgorithmV3TR2.__name__, "BafcAlgorithmV3TR2")
        self.assertEqual(alg._num_actor_critic, 3)
        self.assertEqual(alg._trust_metric_target_obs_cache_size, 32)
        self.assertFalse(alg._enable_critic_reweighting)
        self.assertEqual(alg._critic_reweighting_solver_iters, 5)

    def test_critic_reweighting_disabled_returns_empty_weight(self):
        alg = self._make_alg()
        weight = alg._compute_critic_sample_weights(
            torch.randn(4, 4), torch.randn(4, 2))

        self.assertEqual(weight, ())

    def test_critic_reweighting_weights_are_normalized(self):
        alg = self._make_alg(
            enable_critic_reweighting=True,
            critic_reweighting_solver_iters=2,
            trust_metric_num_feature_coords=3)
        obs = torch.randn(2, 3, 4)
        action = torch.randn(2, 3, 2)
        phi_target = torch.randn(5, alg._num_actor_critic, 4)
        phi_behavior = torch.randn(6, alg._num_actor_critic, 4)

        with mock.patch.object(
                alg,
                "_compute_reference_metric_feature_maps",
                return_value=(phi_target, phi_behavior)) as feature_mock:
            weight = alg._compute_critic_sample_weights(obs, action)

        feature_mock.assert_called_once()
        self.assertEqual(tuple(weight.shape), (2, 3))
        self.assertTrue(torch.isfinite(weight).all().item())
        self.assertTrue((weight >= 0).all().item())
        self.assertAlmostEqual(weight.mean().item(), 1.0, places=5)

    def test_critic_reweighting_degenerate_features_fall_back_to_uniform(self):
        alg = self._make_alg(enable_critic_reweighting=True)
        obs = torch.randn(4, 4)
        action = torch.randn(4, 2)
        phi_target = torch.full((4, alg._num_actor_critic, 3), float('nan'))
        phi_behavior = torch.full((4, alg._num_actor_critic, 3), float('nan'))

        with mock.patch.object(
                alg,
                "_compute_reference_metric_feature_maps",
                return_value=(phi_target, phi_behavior)):
            weight = alg._compute_critic_sample_weights(obs, action)

        self.assertTensorClose(weight, torch.ones(4))

    def test_critic_reweighting_solver_returns_simplex_distribution(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, critic_reweighting_solver_iters=2)
        projected = alg._project_simplex(torch.tensor([-1.0, 2.0, 0.5]))
        self.assertTrue((projected >= 0).all().item())
        self.assertAlmostEqual(projected.sum().item(), 1.0, places=6)

        features = torch.randn(5, alg._num_actor_critic, 3)
        target = torch.randn(4, alg._num_actor_critic, 3)
        target_cov = alg._feature_covariance(target)
        p = alg._solve_critic_reweighting_distribution(
            features, target_cov, torch.tensor(0.5), torch.tensor(1e-3))

        self.assertEqual(tuple(p.shape), (5, ))
        self.assertTrue(torch.isfinite(p).all().item())
        self.assertTrue((p >= 0).all().item())
        self.assertAlmostEqual(p.sum().item(), 1.0, places=5)

    def test_critic_loss_applies_sample_weight(self):
        alg = self._make_alg()
        t, b, n = 2, 3, alg._num_actor_critic
        sample_weight = torch.tensor([[0.5, 1.0, 1.5], [2.0, 0.25, 0.75]])

        class _UnitLoss:

            def __call__(self, info, value, target_value):
                del info, value, target_value
                return LossInfo(loss=torch.ones(t, b))

        alg._critic_losses = [_UnitLoss() for _ in range(n)]
        info = BafcInfo(
            critic=BafcCriticInfo(
                critic=torch.zeros(t, b, n, n),
                target_critic=torch.zeros(t, b, n, n),
                critic_sample_weight=sample_weight),
            bootstrap_mask=torch.ones(t, b, n))

        loss = alg._calc_critic_loss(info)

        self.assertTensorClose(loss.loss, sample_weight * float(n))

    def test_eval_and_reweighting_share_reference_feature_helper(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, critic_reweighting_solver_iters=1)
        obs = torch.randn(4, 4)
        action = torch.randn(4, 2)
        phi_target = torch.randn(4, alg._num_actor_critic, 3)
        phi_behavior = torch.randn(4, alg._num_actor_critic, 3)

        with mock.patch.object(
                alg,
                "_compute_reference_metric_feature_maps",
                return_value=(phi_target, phi_behavior)) as feature_mock:
            alg._compute_eval_trust_metric(obs, action)
        feature_mock.assert_called_once()

        with mock.patch.object(
                alg,
                "_compute_reference_metric_feature_maps",
                return_value=(phi_target, phi_behavior)) as feature_mock:
            alg._compute_critic_sample_weights(obs, action)
        feature_mock.assert_called_once()

    def test_eval_trust_uses_reference_encoding_cache_and_replay_action(self):
        alg = self._make_alg(trust_metric_num_obs=4)
        replay_obs = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        replay_action = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        target_obs = torch.full((2, 4), 9.0)
        alg._target_metric_observation_cache = target_obs.clone()

        target_action = torch.full((2, alg._num_actor_critic, 2), 3.0)
        reference_encoding = torch.full((alg._num_actor_critic, 5), 4.0)
        phi_target = torch.full((2, alg._num_actor_critic, 3), 5.0)
        phi_behavior = torch.full((4, alg._num_actor_critic, 3), 6.0)
        calls = []

        def _feature_map(observation, encoding, action, critic_network=None):
            del critic_network
            calls.append((observation.clone(), encoding.clone(), action.clone()))
            if len(calls) == 1:
                self.assertTensorClose(observation, target_obs)
                self.assertTensorClose(encoding, reference_encoding)
                self.assertTensorClose(action, target_action)
                return phi_target
            self.assertTensorClose(observation, replay_obs)
            self.assertTensorClose(encoding, reference_encoding)
            self.assertTensorClose(action, replay_action.unsqueeze(1))
            return phi_behavior

        with mock.patch.object(
                alg._reference_actor_networks,
                "forward",
                return_value=(target_action, ())) as actor_mock, mock.patch.object(
                    alg,
                    "_compute_actor_encoding",
                    return_value=reference_encoding) as encoding_mock, mock.patch.object(
                        alg,
                        "_compute_snapshot_feature_map",
                        side_effect=_feature_map), mock.patch.object(
                            alg,
                            "_compute_eval_trust_from_features",
                            return_value=torch.tensor(7.0)) as trust_mock:
            trust = alg._compute_eval_trust_metric(replay_obs, replay_action)

        self.assertTensorClose(trust, torch.tensor(7.0))
        actor_mock.assert_called_once()
        encoding_mock.assert_called_once_with(alg._reference_actor_networks)
        trust_mock.assert_called_once()
        trust_args, _ = trust_mock.call_args
        self.assertTensorClose(trust_args[0], phi_target)
        self.assertTensorClose(trust_args[1], phi_behavior)
        self.assertEqual(len(calls), 2)

    def test_eval_trust_falls_back_to_replay_obs_without_target_cache(self):
        alg = self._make_alg(trust_metric_num_obs=4)
        replay_obs = torch.randn(3, 4)
        replay_action = torch.randn(3, 2)
        target_action = torch.randn(3, alg._num_actor_critic, 2)
        actor_encoding = torch.randn(alg._num_actor_critic, 5)
        seen_target_obs = []

        def _feature_map(observation, encoding, action, critic_network=None):
            del encoding, action, critic_network
            if not seen_target_obs:
                seen_target_obs.append(observation.clone())
            return torch.ones(observation.shape[0], alg._num_actor_critic, 3)

        with mock.patch.object(
                alg._reference_actor_networks,
                "forward",
                return_value=(target_action, ())), mock.patch.object(
                    alg,
                    "_compute_actor_encoding",
                    return_value=actor_encoding), mock.patch.object(
                        alg,
                        "_compute_snapshot_feature_map",
                        side_effect=_feature_map):
            trust = alg._compute_eval_trust_metric(replay_obs, replay_action)

        self.assertTrue(torch.isfinite(trust).item())
        self.assertTensorClose(seen_target_obs[0], replay_obs)

    def test_after_update_passes_replay_action_to_eval_metric(self):
        alg = self._make_alg(enable_eval_rollout_skip_gate=True)
        inputs = self._make_train_time_step()
        info = BafcInfo(action=torch.randn(inputs.observation.shape[0], 2))
        alg._last_update_had_actor_step = True

        with mock.patch.object(
                alg,
                "_compute_eval_trust_metric",
                return_value=torch.tensor(2.5)) as metric_mock:
            alg.after_update(inputs, info)

        metric_mock.assert_called_once()
        args, _ = metric_mock.call_args
        self.assertIs(args[0], inputs.observation)
        self.assertIs(args[1], info.action)
        self.assertTensorClose(alg._last_eval_trust, torch.tensor(2.5))

    def test_after_update_pre_syncs_reference_before_eval_metric_when_grad_gate_disabled(self):
        alg = self._make_alg(
            enable_eval_rollout_skip_gate=True,
            enable_grad_actor_extend_gate=False)
        inputs = self._make_train_time_step()
        info = BafcInfo(action=torch.randn(inputs.observation.shape[0], 2))
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._last_update_had_actor_step = True

        def _metric(observation, behavior_action):
            self._assert_state_dict_equal(
                alg._reference_actor_networks,
                self._clone_state_dict(alg._actor_networks))
            return torch.tensor(3.5)

        with mock.patch.object(
                alg,
                "_compute_eval_trust_metric",
                side_effect=_metric) as metric_mock:
            alg.after_update(inputs, info)

        metric_mock.assert_called_once()
        args, _ = metric_mock.call_args
        self.assertIs(args[0], inputs.observation)
        self.assertIs(args[1], info.action)
        self.assertTensorClose(alg._last_eval_trust, torch.tensor(3.5))

    def test_eval_metric_uses_post_transition_reference_after_grad_violation(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_eval_rollout_skip_gate=True,
            enable_grad_actor_extend_gate=True,
            delta_trust_max=1.0)
        inputs = self._make_train_time_step()
        info = BafcInfo(action=torch.randn(inputs.observation.shape[0], 2))
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True

        def _metric(observation, behavior_action):
            self._assert_state_dict_equal(
                alg._reference_actor_networks,
                self._clone_state_dict(alg._actor_networks))
            return torch.tensor(4.5)

        with mock.patch.object(
                alg,
                "_compute_grad_generalization_trust_metric",
                return_value=torch.tensor(2.0)) as grad_mock, mock.patch.object(
                    alg,
                    "_compute_eval_trust_metric",
                    side_effect=_metric) as eval_mock:
            alg.after_update(inputs, info)

        grad_mock.assert_called_once_with(inputs.observation)
        eval_mock.assert_called_once()
        args, _ = eval_mock.call_args
        self.assertIs(args[0], inputs.observation)
        self.assertIs(args[1], info.action)
        self.assertEqual(alg._train_mode, TrainMode.critic)
        self.assertTensorClose(alg._last_grad_trust, torch.tensor(2.0))
        self.assertTensorClose(alg._last_eval_trust, torch.tensor(4.5))

    def test_rollout_updates_target_observation_cache(self):
        alg = self._make_alg(trust_metric_target_obs_cache_size=3)
        state = alg.get_initial_rollout_state(batch_size=1)
        inputs = TimeStep(
            step_type=torch.full((1, ), StepType.FIRST, dtype=torch.int64),
            reward=torch.zeros(1),
            discount=torch.ones(1),
            observation=torch.arange(4, dtype=torch.float32).reshape(1, 4),
            prev_action=(),
            env_id=())

        alg.rollout_step(inputs, state)
        self.assertTensorClose(alg._target_metric_observation_cache,
                               inputs.observation)

        for start in (4, 8, 12):
            inputs = inputs._replace(
                observation=torch.arange(
                    start, start + 4, dtype=torch.float32).reshape(1, 4))
            alg.rollout_step(inputs, state)
        self.assertEqual(alg._target_metric_observation_cache.shape[0], 3)
        expected = torch.stack([
            torch.arange(4, 8, dtype=torch.float32),
            torch.arange(8, 12, dtype=torch.float32),
            torch.arange(12, 16, dtype=torch.float32),
        ])
        self.assertTensorClose(alg._target_metric_observation_cache, expected)

    def test_reference_syncs_after_actor_update_when_grad_gate_disabled(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=False,
            monitor_trust_metrics=False)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_reference_syncs_without_actor_update_when_grad_gate_disabled(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_eval_rollout_skip_gate=True,
            enable_grad_actor_extend_gate=False,
            monitor_trust_metrics=True)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.critic
        alg._critic_update_counter = 1
        alg._last_update_had_actor_step = False

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_reference_syncs_when_grad_gated_actor_epoch_starts(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=True,
            monitor_trust_metrics=False)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.critic
        alg._critic_update_counter = 2
        alg._last_update_had_actor_step = False

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self.assertEqual(alg._train_mode, TrainMode.actor)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_reference_stays_fixed_during_safe_grad_gate_extension(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=True,
            monitor_trust_metrics=False)
        self._fill_module(alg._reference_actor_networks, 0.0)
        ref_before = self._clone_state_dict(alg._reference_actor_networks)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True
        alg._last_grad_trust = torch.tensor(0.0)

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self.assertEqual(alg._train_mode, TrainMode.actor)
        self._assert_state_dict_equal(alg._reference_actor_networks, ref_before)

    def test_reference_syncs_when_grad_trust_violates_threshold(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=True,
            monitor_trust_metrics=False,
            delta_trust_max=1.0)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True
        alg._last_grad_trust = torch.tensor(2.0)

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self.assertEqual(alg._train_mode, TrainMode.critic)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

    def test_reference_syncs_when_grad_extension_cap_breaks_epoch(self):
        alg = self._make_alg(
            actor_utd=1,
            critic_utd=2,
            enable_grad_actor_extend_gate=True,
            grad_gate_max_consecutive_actor_extensions=1,
            monitor_trust_metrics=False,
            delta_trust_max=10.0)
        self._fill_module(alg._reference_actor_networks, 0.0)
        self._fill_module(alg._actor_networks, 1.0)
        alg._train_mode = TrainMode.actor
        alg._actor_update_counter = 1
        alg._last_update_had_actor_step = True
        alg._last_grad_trust = torch.tensor(0.0)
        alg._grad_gate_consecutive_actor_extensions = 1

        alg.after_update(self._make_train_time_step(), BafcInfo())

        self.assertEqual(alg._train_mode, TrainMode.critic)
        self._assert_state_dict_equal(alg._reference_actor_networks,
                                      self._clone_state_dict(alg._actor_networks))

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
        alg._actor_update_counter = 5
        alg._critic_update_counter = 7
        alg._completed_cycles_since_rollout = 3
        alg._real_rollouts_since_reference_sync = 4
        alg._last_eval_trust = torch.tensor(0.25)
        alg._last_grad_trust = torch.tensor(0.5)
        alg._trust_metric_update_counter = 6
        alg._eval_gate_consecutive_rollout_skips = 2
        alg._rollout_skip_due_eval_gate_count = 8
        alg._rollout_opportunity_count = 9
        alg._grad_gate_actor_extension_count = 10
        alg._grad_gate_consecutive_actor_extensions = 11
        alg._target_metric_observation_cache = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        alg._apply_train_mode_grad_flags()

        state = alg.state_dict()
        self.assertIn("_bafc_runtime.training_started", state)
        self.assertIn("_bafc_runtime.target_metric_observation_cache", state)

        restored = self._make_alg()
        restored.load_state_dict(state)

        self.assertTrue(restored._training_started)
        self.assertEqual(restored._train_mode, TrainMode.critic)
        self.assertEqual(restored._rollout_actor_id, 2)
        self.assertEqual(restored._actor_update_counter, 5)
        self.assertEqual(restored._critic_update_counter, 7)
        self.assertEqual(restored._completed_cycles_since_rollout, 3)
        self.assertEqual(restored._real_rollouts_since_reference_sync, 4)
        self.assertTensorClose(restored._last_eval_trust, torch.tensor(0.25))
        self.assertTensorClose(restored._last_grad_trust, torch.tensor(0.5))
        self.assertEqual(restored._trust_metric_update_counter, 6)
        self.assertEqual(restored._eval_gate_consecutive_rollout_skips, 2)
        self.assertEqual(restored._rollout_skip_due_eval_gate_count, 8)
        self.assertEqual(restored._rollout_opportunity_count, 9)
        self.assertEqual(restored._grad_gate_actor_extension_count, 10)
        self.assertEqual(restored._grad_gate_consecutive_actor_extensions, 11)
        self.assertTensorClose(restored._target_metric_observation_cache,
                               torch.arange(12, dtype=torch.float32).reshape(3, 4))
        self.assertTrue(all(not p.requires_grad
                            for p in restored._actor_networks.parameters()))
        self.assertTrue(restored._actor_eval_samples.requires_grad)

        legacy_restored = self._make_alg()
        legacy_restored.load_state_dict(self._without_runtime_state(state))
        self.assertTrue(legacy_restored._training_started)



if __name__ == "__main__":
    alf.test.main()
