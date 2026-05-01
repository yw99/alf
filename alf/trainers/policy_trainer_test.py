# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import functools
import tempfile
from unittest import mock
import torch

import alf
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.algorithms.rl_algorithm_test import MyEnv, MyAlg
from alf.trainers.policy_trainer import RLTrainer, TrainerConfig, play
from alf.trainers.policy_trainer import SLTrainer
from alf.trainers import evaluator
from alf.utils import common, datagen


def env_load(env_name, batch_size):
    return MyEnv(3)


env_load.batched = True


class TrainerTest(alf.test.TestCase):

    def _make_policy_boundary_eval_trainer(self,
                                           rollout_interval=100,
                                           grad_interval=100):
        trainer = object.__new__(RLTrainer)
        trainer._rollout_skip_evaluator = mock.Mock()
        trainer._rollout_skip_eval = True
        trainer._grad_gate_eval = True
        trainer._rollout_skip_eval_interval = rollout_interval
        trainer._grad_gate_eval_interval = grad_interval
        trainer._next_rollout_skip_eval_start = rollout_interval
        trainer._next_grad_gate_eval_start = grad_interval
        trainer._sampled_rollout_skip_eval_starts = set()
        trainer._sampled_grad_gate_eval_starts = set()
        metric = mock.Mock()
        metric.name = "EnvironmentSteps"
        metric.result.return_value = 123
        trainer._algorithm = mock.Mock()
        trainer._algorithm.get_step_metrics.return_value = [metric]
        return trainer

    def test_rl_trainer(self):
        with tempfile.TemporaryDirectory() as root_dir:
            alf.config("create_environment", env_load_fn=env_load)
            conf = TrainerConfig(algorithm_ctor=MyAlg,
                                 root_dir=root_dir,
                                 unroll_length=5,
                                 num_iterations=100)

            # test train
            trainer = RLTrainer(conf)
            self.assertEqual(RLTrainer.progress(), 0)
            trainer.train()
            self.assertEqual(RLTrainer.progress(), 1)

            alg = trainer._algorithm
            env = common.get_env()
            time_step = common.get_initial_time_step(env)
            state = alg.get_initial_predict_state(env.batch_size)
            policy_step = alg.rollout_step(time_step, state)
            logits = policy_step.info['dist'].logits
            print("logits: ", logits)
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))

            # test checkpoint
            conf.num_iterations = 200
            new_trainer = RLTrainer(conf)
            new_trainer._restore_checkpoint()
            self.assertEqual(RLTrainer.progress(), 0.5)
            time_step = common.get_initial_time_step(env)
            state = alg.get_initial_predict_state(env.batch_size)
            policy_step = alg.rollout_step(time_step, state)
            logits = policy_step.info['dist'].logits
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 0]))
            self.assertTrue(torch.all(logits[:, 1] > logits[:, 2]))

            new_trainer.train()
            self.assertEqual(RLTrainer.progress(), 1)

            # TODO: test play. Need real env to test.

    def test_sl_trainer(self):
        with tempfile.TemporaryDirectory() as root_dir:
            conf = TrainerConfig(algorithm_ctor=functools.partial(
                HyperNetwork,
                data_creator=datagen.load_test,
                hidden_layers=None,
                loss_type='regression',
                num_train_classes=1,
                optimizer=alf.optimizers.Adam(lr=1e-4, weight_decay=1e-4)),
                                 root_dir=root_dir,
                                 num_checkpoints=1,
                                 evaluate=True,
                                 eval_interval=1,
                                 num_iterations=1)

            # test train
            trainer = SLTrainer(conf)
            self.assertEqual(SLTrainer.progress(), 0)
            trainer.train()
            self.assertEqual(SLTrainer.progress(), 1)

            # test checkpoint
            conf2 = TrainerConfig(algorithm_ctor=functools.partial(
                HyperNetwork,
                data_creator=datagen.load_test,
                hidden_layers=None,
                loss_type='regression',
                num_train_classes=1,
                optimizer=alf.optimizers.Adam(lr=1e-4, weight_decay=1e-4)),
                                  root_dir=root_dir,
                                  num_checkpoints=1,
                                  evaluate=True,
                                  eval_interval=1,
                                  num_iterations=2)

            new_trainer = SLTrainer(conf2)
            new_trainer._restore_checkpoint()
            self.assertEqual(SLTrainer.progress(), 0.5)
            new_trainer.train()
            self.assertEqual(SLTrainer.progress(), 1)

    def test_rollout_skip_eval_summary_steps_and_relative_change(self):
        event = dict(
            type="skip_end",
            start_rollout_opportunity=4,
            end_rollout_opportunity=7,
            skip_length=3)

        with mock.patch.object(alf.summary, "scalar") as scalar:
            evaluator._write_rollout_skip_start_summary(2.0, event)
            evaluator._write_rollout_skip_result_summaries(2.0, 3.0, event)

        scalar.assert_any_call("rollout_skip_eval/start_average_return",
                               2.0,
                               step=4)
        scalar.assert_any_call("rollout_skip_eval/average_return",
                               2.0,
                               step=4)
        scalar.assert_any_call("rollout_skip_eval/end_average_return",
                               3.0,
                               step=7)
        scalar.assert_any_call("rollout_skip_eval/average_return",
                               3.0,
                               step=7)
        scalar.assert_any_call(
            "rollout_skip_eval/relative_average_return_change",
            0.5,
            step=7)
        scalar.assert_any_call("rollout_skip_eval/skip_length", 3, step=7)
        self.assertEqual(scalar.call_count, 6)
        self.assertAlmostEqual(
            evaluator._relative_return_change(-2.0, -1.0), 0.5)

    def test_grad_gate_eval_summary_steps_and_relative_change(self):
        event = dict(
            type="grad_extension_end",
            start_step=11,
            end_step=15,
            extension_length=4)

        with mock.patch.object(alf.summary, "scalar") as scalar:
            evaluator._write_grad_gate_start_summary(4.0, event)
            evaluator._write_grad_gate_result_summaries(4.0, 3.0, event)

        scalar.assert_any_call("grad_gate_eval/start_average_return",
                               4.0,
                               step=11)
        scalar.assert_any_call("grad_gate_eval/average_return", 4.0, step=11)
        scalar.assert_any_call("grad_gate_eval/end_average_return",
                               3.0,
                               step=15)
        scalar.assert_any_call("grad_gate_eval/average_return", 3.0, step=15)
        scalar.assert_any_call(
            "grad_gate_eval/relative_average_return_change",
            -0.25,
            step=15)
        scalar.assert_any_call("grad_gate_eval/extension_length", 4, step=15)
        self.assertEqual(scalar.call_count, 6)

    def test_policy_boundary_eval_state_restore(self):
        algorithm = mock.Mock()
        state = dict(training_started=True, rollout_actor_id=2)
        event = dict(type="skip_start", policy_eval_state=state)

        evaluator._restore_policy_boundary_eval_state(algorithm, event)

        algorithm.set_policy_boundary_eval_state.assert_called_once_with(state)

    def test_policy_boundary_eval_state_restore_ignores_missing_hook(self):
        algorithm = object()
        state = dict(training_started=True, rollout_actor_id=2)
        event = dict(type="skip_start", policy_eval_state=state)

        evaluator._restore_policy_boundary_eval_state(algorithm, event)

    def test_rollout_skip_eval_interval_samples_complete_windows(self):
        trainer = self._make_policy_boundary_eval_trainer(
            rollout_interval=100)
        state_dict = {"weights": 1}

        with mock.patch.object(alf.summary, "get_global_counter", return_value=7):
            trainer._handle_rollout_skip_eval_event(
                dict(
                    type="skip_start",
                    start_rollout_opportunity=50,
                    end_rollout_opportunity=50,
                    skip_length=1), state_dict)
            trainer._handle_rollout_skip_eval_event(
                dict(
                    type="skip_end",
                    start_rollout_opportunity=50,
                    end_rollout_opportunity=60,
                    skip_length=10), state_dict)
            self.assertEqual(trainer._rollout_skip_evaluator.eval.call_count, 0)
            trainer._algorithm.get_step_metrics.assert_not_called()

            start_100 = dict(
                type="skip_start",
                start_rollout_opportunity=100,
                end_rollout_opportunity=100,
                skip_length=1)
            end_100 = dict(
                type="skip_end",
                start_rollout_opportunity=100,
                end_rollout_opportunity=101,
                skip_length=1)
            trainer._handle_rollout_skip_eval_event(start_100, state_dict)
            trainer._handle_rollout_skip_eval_event(end_100, state_dict)
            self.assertEqual(trainer._rollout_skip_evaluator.eval.call_count, 2)
            trainer._rollout_skip_evaluator.eval.assert_any_call(
                event=start_100,
                state_dict=state_dict,
                global_counter=7,
                step_metric_values={"EnvironmentSteps": 123})
            trainer._rollout_skip_evaluator.eval.assert_any_call(
                event=end_100,
                state_dict=state_dict,
                global_counter=7,
                step_metric_values={"EnvironmentSteps": 123})
            self.assertEqual(trainer._next_rollout_skip_eval_start, 200)
            self.assertEqual(trainer._sampled_rollout_skip_eval_starts, set())

            trainer._handle_rollout_skip_eval_event(
                dict(
                    type="skip_start",
                    start_rollout_opportunity=150,
                    end_rollout_opportunity=150,
                    skip_length=1), state_dict)
            self.assertEqual(trainer._rollout_skip_evaluator.eval.call_count, 2)

    def test_grad_gate_eval_interval_samples_complete_windows_independently(
            self):
        trainer = self._make_policy_boundary_eval_trainer(
            rollout_interval=100, grad_interval=100)
        state_dict = {"weights": 1}

        with mock.patch.object(alf.summary, "get_global_counter", return_value=9):
            rollout_start = dict(
                type="skip_start",
                start_rollout_opportunity=100,
                end_rollout_opportunity=100,
                skip_length=1)
            trainer._handle_rollout_skip_eval_event(rollout_start, state_dict)
            self.assertEqual(trainer._rollout_skip_evaluator.eval.call_count, 1)
            self.assertEqual(trainer._next_rollout_skip_eval_start, 200)
            self.assertEqual(trainer._next_grad_gate_eval_start, 100)

            trainer._handle_rollout_skip_eval_event(
                dict(
                    type="grad_extension_start",
                    start_step=50,
                    end_step=50,
                    extension_length=1), state_dict)
            trainer._handle_rollout_skip_eval_event(
                dict(
                    type="grad_extension_end",
                    start_step=50,
                    end_step=60,
                    extension_length=3), state_dict)
            self.assertEqual(trainer._rollout_skip_evaluator.eval.call_count, 1)

            grad_start = dict(
                type="grad_extension_start",
                start_step=100,
                end_step=100,
                extension_length=1)
            grad_end = dict(
                type="grad_extension_end",
                start_step=100,
                end_step=101,
                extension_length=1)
            trainer._handle_rollout_skip_eval_event(grad_start, state_dict)
            trainer._handle_rollout_skip_eval_event(grad_end, state_dict)
            self.assertEqual(trainer._rollout_skip_evaluator.eval.call_count, 3)
            trainer._rollout_skip_evaluator.eval.assert_any_call(
                event=grad_start,
                state_dict=state_dict,
                global_counter=9,
                step_metric_values={"EnvironmentSteps": 123})
            trainer._rollout_skip_evaluator.eval.assert_any_call(
                event=grad_end,
                state_dict=state_dict,
                global_counter=9,
                step_metric_values={"EnvironmentSteps": 123})
            self.assertEqual(trainer._next_grad_gate_eval_start, 200)
            self.assertEqual(trainer._sampled_grad_gate_eval_starts, set())


if __name__ == "__main__":
    alf.test.main()
