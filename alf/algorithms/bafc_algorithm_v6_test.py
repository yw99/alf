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
from alf.algorithms.bafc_algorithm_v6 import (BafcActorInfo,
                                              BafcAlgorithmV6,
                                              BafcCriticInfo, BafcInfo)
from alf.algorithms.config import TrainerConfig
from alf.data_structures import LossInfo, StepType, TimeStep
from alf.experience_replayers.replay_buffer import BatchInfo
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class _DummyProcess:

    def memory_info(self):
        return SimpleNamespace(rss=0)

    def children(self, recursive=True):
        del recursive
        return []


class _DummyReplayBuffer:

    def __init__(self, current_pos):
        self.device = torch.device("cpu")
        self._current_pos = torch.as_tensor(current_pos, dtype=torch.int64)


class BafcAlgorithmV6Test(alf.test.TestCase):

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
            root_dir=tempfile.mkdtemp(prefix="bafc_v6_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            initial_collect_steps=0,
            num_updates_per_train_iter=num_updates_per_train_iter)
        return BafcAlgorithmV6(
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

    def _make_rollout_time_step(self, observation):
        batch_size = observation.shape[0]
        return TimeStep(
            step_type=torch.full((batch_size, ), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(batch_size),
            discount=torch.ones(batch_size),
            observation=observation,
            prev_action=(),
            env_id=())

    def test_initialization_smoke_and_config_name(self):
        alg = self._make_alg()

        self.assertIsInstance(alg, BafcAlgorithmV6)
        self.assertEqual(BafcAlgorithmV6.__name__, "BafcAlgorithmV6")
        self.assertEqual(alg._num_actor_critic, 3)
        self.assertFalse(alg._enable_critic_reweighting)
        self.assertEqual(alg._critic_reweighting_solver, "lbfgs_logits")
        self.assertEqual(alg._critic_reweighting_solver_iters, 5)
        self.assertEqual(alg._critic_reweighting_target_obs_cache_size, 512)

    def test_invalid_critic_reweighting_solver_raises(self):
        with self.assertRaises(AssertionError):
            self._make_alg(critic_reweighting_solver="bogus")

    def test_info_tuples_do_not_include_trust_fields(self):
        for fields in (BafcInfo._fields, BafcActorInfo._fields,
                       BafcCriticInfo._fields):
            self.assertNotIn("eval_trust_metric", fields)
            self.assertNotIn("grad_trust_metric", fields)

        self.assertIn("critic_sample_weight", BafcCriticInfo._fields)
        self.assertIn("sample_age", BafcInfo._fields)

    def test_preprocess_experience_computes_sample_age(self):
        alg = self._make_alg()
        root_inputs = TimeStep(
            step_type=torch.full((2, 3), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(2, 3),
            discount=torch.ones(2, 3),
            observation=torch.zeros(2, 3, 4),
            prev_action=(),
            env_id=())
        batch_info = BatchInfo(
            env_ids=torch.tensor([0, 1], dtype=torch.int64),
            positions=torch.tensor([7, 15], dtype=torch.int64),
            replay_buffer=_DummyReplayBuffer([10, 20]))

        processed_inputs, processed_info = alg.preprocess_experience(
            root_inputs, BafcInfo(), batch_info)

        self.assertIs(processed_inputs, root_inputs)
        self.assertTensorClose(
            processed_info.sample_age,
            torch.tensor([[2, 1, 0], [4, 3, 2]], dtype=torch.float32))

    def test_preprocess_experience_missing_batch_info_leaves_info_unchanged(self):
        alg = self._make_alg()
        root_inputs = TimeStep(
            step_type=torch.full((2, 3), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(2, 3),
            discount=torch.ones(2, 3),
            observation=torch.zeros(2, 3, 4),
            prev_action=(),
            env_id=())
        rollout_info = BafcInfo()

        _, processed_info = alg.preprocess_experience(root_inputs, rollout_info, ())

        self.assertEqual(processed_info.sample_age, ())

    def test_critic_reweighting_disabled_returns_empty_weight(self):
        alg = self._make_alg()
        with mock.patch.object(
                alg, "_record_critic_reweighting_summaries") as summary_mock:
            weight = alg._compute_critic_sample_weights(
                torch.randn(4, 4), torch.randn(4, 2))

        self.assertEqual(weight, ())
        summary_mock.assert_not_called()


    def test_critic_reweighting_summary_metrics(self):
        alg = self._make_alg(
            enable_critic_reweighting=True,
            debug_summaries=True,
            critic_reweighting_max_weight=2.0)
        raw_weights = torch.tensor([0.5, 2.0, 3.0])
        clipped_weights = raw_weights.clamp(max=alg._critic_reweighting_max_weight)
        final_weights = clipped_weights / clipped_weights.mean()

        with mock.patch(
                "alf.summary.should_record_summaries",
                return_value=True), mock.patch(
                    "alf.summary.scalar") as scalar_mock, mock.patch(
                        "alf.summary.histogram") as histogram_mock:
            alg._record_critic_reweighting_summaries(
                final_weights,
                raw_weights=raw_weights,
                clipped_weights=clipped_weights,
                fallback_to_uniform=False,
                solver_objective_initial=torch.tensor(5.0),
                solver_objective_final=torch.tensor(3.0))

        histogram_names = [call.args[0] for call in histogram_mock.call_args_list]
        self.assertIn("critic_reweighting/raw_weight/value", histogram_names)
        self.assertIn("critic_reweighting/clipped_weight/value", histogram_names)
        self.assertIn("critic_reweighting/final_weight/value", histogram_names)
        self.assertNotIn("critic_reweighting/sample_age/value", histogram_names)

        scalars = {call.args[0]: call.args[1] for call in scalar_mock.call_args_list}
        self.assertNotIn("critic_reweighting/final_weight_recency_corr", scalars)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/num_clipped_at_max"]),
            2.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/frac_clipped_at_max"]),
            2.0 / 3.0,
            places=6)
        expected_ess = final_weights.sum().pow(2) / final_weights.pow(2).sum()
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/ess"]),
            float(expected_ess),
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/ess_ratio"]),
            float(expected_ess / final_weights.numel()),
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/fallback_to_uniform"]),
            0.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_objective_initial"]),
            5.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_objective_final"]),
            3.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/solver_objective_improvement"]),
            2.0,
            places=6)

    def test_critic_reweighting_recency_summary_metrics(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, debug_summaries=True)
        raw_weights = torch.tensor([4.0, 3.0, 2.0, 1.0])
        final_weights = raw_weights / raw_weights.mean()
        sample_age = torch.tensor([0.0, 1.0, 2.0, 3.0])

        with mock.patch(
                "alf.summary.should_record_summaries",
                return_value=True), mock.patch(
                    "alf.summary.scalar") as scalar_mock, mock.patch(
                        "alf.summary.histogram") as histogram_mock:
            alg._record_critic_reweighting_summaries(
                final_weights,
                raw_weights=raw_weights,
                clipped_weights=raw_weights,
                sample_age=sample_age,
                fallback_to_uniform=False)

        histogram_names = [call.args[0] for call in histogram_mock.call_args_list]
        self.assertIn("critic_reweighting/sample_age/value", histogram_names)

        scalars = {call.args[0]: call.args[1] for call in scalar_mock.call_args_list}
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/sample_age_min"]),
            0.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/sample_age_max"]),
            3.0,
            places=6)
        self.assertGreater(
            float(scalars["critic_reweighting/final_weight_recency_corr"]),
            0.999)
        self.assertLess(
            float(scalars["critic_reweighting/final_weight_age_corr"]),
            -0.999)
        self.assertGreater(
            float(scalars["critic_reweighting/raw_weight_recency_corr"]),
            0.999)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/newest_quartile_final_weight_mean"]),
            float(final_weights[0]),
            places=6)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/oldest_quartile_final_weight_mean"]),
            float(final_weights[-1]),
            places=6)
        self.assertAlmostEqual(
            float(scalars[
                "critic_reweighting/newest_over_oldest_weight_ratio"]),
            float(final_weights[0] / final_weights[-1]),
            places=6)

    def test_critic_reweighting_fallback_records_uniform_summary(self):
        alg = self._make_alg(
            enable_critic_reweighting=True, debug_summaries=True)
        obs = torch.randn(4, 4)
        action = torch.randn(3, 2)

        with mock.patch(
                "alf.summary.should_record_summaries",
                return_value=True), mock.patch(
                    "alf.summary.scalar") as scalar_mock, mock.patch(
                        "alf.summary.histogram") as histogram_mock:
            weight = alg._compute_critic_sample_weights(obs, action)

        self.assertTensorClose(weight, torch.ones(4))
        histogram_values = {
            call.args[0]: call.args[1]
            for call in histogram_mock.call_args_list
        }
        self.assertTensorClose(
            histogram_values["critic_reweighting/final_weight/value"],
            torch.ones(4))
        scalars = {call.args[0]: call.args[1] for call in scalar_mock.call_args_list}
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/fallback_to_uniform"]),
            1.0,
            places=6)
        self.assertAlmostEqual(
            float(scalars["critic_reweighting/ess_ratio"]),
            1.0,
            places=6)

    def test_critic_reweighting_weights_are_normalized(self):
        alg = self._make_alg(
            enable_critic_reweighting=True,
            critic_reweighting_beta=0.5,
            critic_reweighting_solver_iters=2,
            critic_reweighting_num_feature_coords=3)
        obs = torch.randn(2, 3, 4)
        action = torch.randn(2, 3, 2)
        phi_target = torch.randn(5, alg._num_actor_critic, 4)
        phi_behavior = torch.randn(6, alg._num_actor_critic, 4)

        with mock.patch.object(
                alg,
                "_compute_reweighting_feature_maps",
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
                "_compute_reweighting_feature_maps",
                return_value=(phi_target, phi_behavior)):
            weight = alg._compute_critic_sample_weights(obs, action)

        self.assertTensorClose(weight, torch.ones(4))

    def test_critic_reweighting_solver_returns_simplex_distribution(self):
        torch.manual_seed(1234)
        projected_alg = self._make_alg()
        projected = projected_alg._project_simplex(
            torch.tensor([-1.0, 2.0, 0.5]))
        self.assertTrue((projected >= 0).all().item())
        self.assertAlmostEqual(projected.sum().item(), 1.0, places=6)

        features = torch.randn(5, projected_alg._num_actor_critic, 3)
        target = torch.randn(4, projected_alg._num_actor_critic, 3)
        target_cov = projected_alg._feature_covariance(target)
        beta = torch.tensor(0.5)
        ridge = torch.tensor(1e-3)

        for solver in ("lbfgs_logits", "projected_gradient_fw"):
            alg = self._make_alg(
                enable_critic_reweighting=True,
                critic_reweighting_solver=solver,
                critic_reweighting_solver_iters=2)
            p = alg._solve_critic_reweighting_distribution(
                features, target_cov, beta, ridge)

            self.assertEqual(tuple(p.shape), (5, ))
            self.assertTrue(torch.isfinite(p).all().item())
            self.assertTrue((p >= 0).all().item())
            self.assertAlmostEqual(p.sum().item(), 1.0, places=5)

            uniform = torch.full_like(p, 1.0 / p.numel())
            uniform_obj = alg._critic_reweighting_objective(
                uniform, features, target_cov, beta, ridge)
            final_obj = alg._critic_reweighting_objective(
                p, features, target_cov, beta, ridge)
            self.assertLessEqual(
                float(final_obj), float(uniform_obj) + 1e-3)

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

    def test_rollout_updates_reweighting_target_observation_cache(self):
        alg = self._make_alg(
            enable_critic_reweighting=True,
            critic_reweighting_target_obs_cache_size=3)
        state = alg.get_initial_rollout_state(batch_size=1)
        inputs = self._make_rollout_time_step(
            torch.arange(4, dtype=torch.float32).reshape(1, 4))

        alg.rollout_step(inputs, state)
        self.assertTensorClose(alg._reweighting_target_observation_cache,
                               inputs.observation)

        for start in (4, 8, 12):
            inputs = inputs._replace(
                observation=torch.arange(
                    start, start + 4, dtype=torch.float32).reshape(1, 4))
            alg.rollout_step(inputs, state)

        self.assertEqual(alg._reweighting_target_observation_cache.shape[0], 3)
        expected = torch.stack([
            torch.arange(4, 8, dtype=torch.float32),
            torch.arange(8, 12, dtype=torch.float32),
            torch.arange(12, 16, dtype=torch.float32),
        ])
        self.assertTensorClose(alg._reweighting_target_observation_cache,
                               expected)


if __name__ == "__main__":
    alf.test.main()
