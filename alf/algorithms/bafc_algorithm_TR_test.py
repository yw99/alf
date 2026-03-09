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

import torch
import tempfile
from functools import partial
from types import MethodType

import alf
from alf.algorithms.bafc_algorithm_TR import BafcAlgorithmTR, BafcInfo
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, StepType, LossInfo, namedtuple
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder


class BafcAlgorithmTRTest(alf.test.TestCase):

    def _make_alg(self, **kwargs):
        num_updates_per_train_iter = kwargs.pop("num_updates_per_train_iter", 1)
        obs_spec = TensorSpec((4, ))
        action_spec = BoundedTensorSpec((2, ), minimum=-1.0, maximum=1.0)
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_tr_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            num_updates_per_train_iter=num_updates_per_train_iter)
        return BafcAlgorithmTR(
            observation_spec=obs_spec,
            action_spec=action_spec,
            config=config,
            actor_network_cls=partial(ActorFCNetwork, fc_layer_params=(64, 64)),
            critic_network_cls=partial(
                FuncCriticNetwork,
                obs_action_joint_fc_layer_params=(64, 64),
                actor_obs_action_joint_fc_layer_params=(64, 64)),
            actor_encoder_cls=partial(
                TransformerEncoder, num_layers=2, num_attention_heads=1),
            num_actor_critic=3,
            num_actor_eval_samples=32,
            policy_eval_updates_per_epoch=1,
            max_improve_steps_per_epoch=1,
            **kwargs)

    def _make_update_time_step(self, t=2, b=3):
        return TimeStep(
            step_type=torch.full((t, b), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(t, b),
            discount=torch.ones(t, b),
            observation=torch.randn(t, b, 4),
            prev_action=(),
            env_id=())

    def _make_rollout_time_step(self, b=3):
        return TimeStep(
            step_type=torch.full((b, ), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(b),
            discount=torch.ones(b),
            observation=torch.randn(b, 4),
            prev_action=(),
            env_id=())

    def _make_rollout_info(self, t=2, b=3):
        return BafcInfo(
            observation=torch.randn(t, b, 4),
            action=torch.randn(t, b, 2),
            bootstrap_mask=())

    def _flatten_params(self, module):
        return torch.cat([param.reshape(-1) for param in module.parameters()])

    def test_initialization(self):
        alg = self._make_alg()
        self.assertTrue(alg.on_policy)
        self.assertEqual(alg._phase.value, "eval")
        self.assertEqual(alg._epoch_idx, 0)
        self.assertEqual(alg._eval_step_idx, 0)
        self.assertEqual(alg._improve_step_idx, 0)

    def test_eval_trust_metric_updates_without_behavior_reset(self):
        alg = self._make_alg(eval_trust_max=-1.0)
        # Eval trust gate is intentionally disabled for on-policy adaptation
        # exploration, so the reset flag should stay false.
        self.assertFalse(alg._epoch_reset_flag)
        self.assertGreaterEqual(float(alg._last_eval_trust), 0.0)

    def test_phase_transition_and_epoch_refresh(self):
        alg = self._make_alg(eval_trust_max=10.0)
        ts_update = self._make_update_time_step()

        # Eval phase consumes one update and enters improvement.
        alg.after_update(ts_update, BafcInfo())
        self.assertEqual(alg._phase.value, "improve")

        # Improvement phase consumes one update and marks epoch refresh pending.
        alg.after_update(ts_update, BafcInfo())
        self.assertEqual(alg._phase.value, "improve")
        self.assertEqual(alg._epoch_idx, 0)
        self.assertTrue(alg._pending_epoch_refresh)

        # Epoch refresh is applied at the next rollout boundary.
        rollout_ts = self._make_rollout_time_step()
        rollout_state = alg.get_initial_rollout_state(batch_size=3)
        alg.rollout_step(rollout_ts, rollout_state)

        self.assertEqual(alg._phase.value, "eval")
        self.assertEqual(alg._epoch_idx, 1)
        self.assertFalse(alg._pending_epoch_refresh)

    def test_calc_loss_improve_phase_is_actor_only(self):
        alg = self._make_alg()
        alg.after_update(self._make_update_time_step(), BafcInfo())

        self.assertEqual(alg._phase.value, "improve")
        self.assertTrue(all(param.requires_grad
                            for param in alg._actor_networks.parameters()))
        self.assertTrue(all(not param.requires_grad
                            for param in alg._critic_networks.parameters()))

        loss_info = alg.calc_loss(self._make_rollout_info())

        self.assertEqual(tuple(loss_info.loss.shape), (2, 3))
        self.assertIsInstance(loss_info.scalar_loss, torch.Tensor)
        self.assertEqual(loss_info.extra.critic, ())
        self.assertEqual(
            tuple(loss_info.extra.actor.grad_trust_metric.shape), (2, 3))

    def test_behavior_policy_stays_fixed_until_epoch_boundary(self):
        alg = self._make_alg()

        ref_before = self._flatten_params(alg._reference_actor_networks).clone()
        beh_before = self._flatten_params(alg._behavior_actor_networks).clone()
        self.assertTrue(torch.allclose(ref_before, beh_before))

        with torch.no_grad():
            for param in alg._actor_networks.parameters():
                param.add_(0.1)

        actor_mid = self._flatten_params(alg._actor_networks)
        beh_mid = self._flatten_params(alg._behavior_actor_networks)
        self.assertFalse(torch.allclose(actor_mid, beh_mid))
        self.assertTrue(torch.allclose(ref_before, beh_mid))

        alg._pending_epoch_refresh = True
        alg.rollout_step(
            self._make_rollout_time_step(),
            alg.get_initial_rollout_state(batch_size=3))

        actor_after = self._flatten_params(alg._actor_networks)
        ref_after = self._flatten_params(alg._reference_actor_networks)
        beh_after = self._flatten_params(alg._behavior_actor_networks)
        self.assertTrue(torch.allclose(actor_after, ref_after))
        self.assertTrue(torch.allclose(actor_after, beh_after))

    def test_update_critic_in_improve_rejected_for_on_policy_adaptation(self):
        with self.assertRaisesRegex(
                AssertionError, "update_critic_in_improve=True deviates"):
            self._make_alg(update_critic_in_improve=True)

    def test_train_iter_on_policy_reuses_rollout_for_multiple_updates(self):
        alg = self._make_alg(num_updates_per_train_iter=3)
        Experience = namedtuple("Experience", ["step_type", "time_step"])
        time_step = self._make_update_time_step()
        experience = Experience(
            step_type=time_step.step_type, time_step=time_step)
        call_counts = {
            "compute": 0,
            "calc": 0,
            "update": 0,
            "summarize": 0,
            "after_iter": 0,
        }
        calc_phases = []

        def fake_compute(self, unroll_length):
            del unroll_length
            call_counts["compute"] += 1
            return BafcInfo(), LossInfo(loss=torch.ones(2, 3)), experience

        def fake_calc(self, info):
            del info
            call_counts["calc"] += 1
            calc_phases.append(self._phase.value)
            return LossInfo(loss=torch.full((2, 3), float(call_counts["calc"])))

        def fake_update(self, loss_info, valid_masks):
            del valid_masks
            call_counts["update"] += 1
            return loss_info, ["param"]

        def fake_summarize(self, experience, train_info, loss_info, params):
            del experience, train_info, loss_info, params
            call_counts["summarize"] += 1

        def fake_after_iter(self, root_inputs, train_info):
            del root_inputs, train_info
            call_counts["after_iter"] += 1

        alg._compute_train_info_and_loss_info_on_policy = MethodType(
            fake_compute, alg)
        alg.calc_loss = MethodType(fake_calc, alg)
        alg.update_with_gradient = MethodType(fake_update, alg)
        alg.summarize_train = MethodType(fake_summarize, alg)
        alg.after_train_iter = MethodType(fake_after_iter, alg)

        alg.train_iter()

        self.assertEqual(call_counts["compute"], 1)
        self.assertEqual(call_counts["update"], 2)
        self.assertEqual(call_counts["calc"], 1)
        self.assertEqual(calc_phases, ["improve"])
        self.assertEqual(call_counts["summarize"], 1)
        self.assertEqual(call_counts["after_iter"], 1)
        self.assertEqual(alg._phase.value, "improve")
        self.assertTrue(alg._pending_epoch_refresh)


if __name__ == "__main__":
    alf.test.main()
