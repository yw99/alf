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

import alf
from alf.algorithms.bafc_algorithm_TR import BafcAlgorithmTR, BafcInfo
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, StepType
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder


class BafcAlgorithmTRTest(alf.test.TestCase):

    def _make_alg(self, **kwargs):
        obs_spec = TensorSpec((4, ))
        action_spec = BoundedTensorSpec((2, ), minimum=-1.0, maximum=1.0)
        config = TrainerConfig(
            root_dir=tempfile.mkdtemp(prefix="bafc_tr_test_"),
            unroll_length=2,
            mini_batch_length=2,
            mini_batch_size=4,
            num_updates_per_train_iter=1)
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
        obs = torch.randn(2, 3, 4)
        ts = TimeStep(
            step_type=torch.full((2, 3), StepType.MID, dtype=torch.int64),
            reward=torch.zeros(2, 3),
            discount=torch.ones(2, 3),
            observation=obs,
            prev_action=(),
            env_id=())

        # Eval phase consumes one update and enters improvement.
        alg.after_update(ts, BafcInfo())
        self.assertEqual(alg._phase.value, "improve")

        # Improvement phase consumes one update and refreshes epoch.
        alg.after_update(ts, BafcInfo())
        self.assertEqual(alg._phase.value, "eval")
        self.assertEqual(alg._epoch_idx, 1)


if __name__ == "__main__":
    alf.test.main()
