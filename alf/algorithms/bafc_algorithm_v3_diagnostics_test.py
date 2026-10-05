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

"""Integration tests for BAFCv3 gradient diagnostics.

Run the optional real two-GPU DDP smoke test with::

    BAFCV3_DDP_SMOKE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
        python -m unittest alf.algorithms.bafc_algorithm_v3_diagnostics_test

The smoke test uses CUDA models with the production Gloo backend and performs
three optimizer updates on synthetic replay batches. It does not create environments, launch experiments, or write summaries.
"""

from contextlib import contextmanager
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import time
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import alf
from alf.algorithms.bafc_algorithm_v3 import (
    BafcAlgorithmV3, BafcCriticInfo, BafcInfo)
from alf.algorithms.bafc_algorithm_v3_test import _DummyProcess
from alf.algorithms.config import TrainerConfig
from alf.data_structures import Experience, StepType, TimeStep
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils import dist_utils


def _make_algorithm(root_dir, diagnostics, normalize_qk=False,
                    debug_summaries=True, dropout=0., **kwargs):
    config = TrainerConfig(
        root_dir=str(root_dir), unroll_length=2, mini_batch_length=2,
        mini_batch_size=2, initial_collect_steps=0,
        num_updates_per_train_iter=2, summary_interval=1000,
        summarize_first_interval=False)
    config.data_transformer = None
    return BafcAlgorithmV3(
        observation_spec=TensorSpec((4,)),
        action_spec=BoundedTensorSpec((2,), minimum=-1., maximum=1.),
        config=config,
        actor_network_cls=partial(ActorFCNetwork, fc_layer_params=(8, 8)),
        critic_network_cls=partial(
            FuncCriticNetwork, obs_action_joint_fc_layer_params=(8,),
            actor_obs_action_joint_fc_layer_params=(8,)),
        actor_encoder_cls=partial(
            TransformerEncoder, num_layers=1, num_attention_heads=1,
            dropout=dropout, norm_first=True, final_norm=True,
            normalize_qk=normalize_qk),
        num_actor_critic=2, num_actor_eval_samples=8,
        actor_eval_type='last_two', actor_utd=1, critic_utd=1,
        actor_critic_pairing=False, num_sampled_critics_for_actor=2,
        use_random_critic_targets=True, num_sampled_critic_targets=1,
        use_target_actor_encoder=True,
        debug_summaries=debug_summaries,
        debug_gradient_chain=diagnostics,
        debug_gradient_chain_compare_backends=diagnostics,
        **kwargs)


def _experience(algorithm):
    inputs = TimeStep(
        step_type=torch.tensor([[StepType.MID, StepType.LAST],
                                [StepType.MID, StepType.MID]], dtype=torch.int32),
        reward=torch.tensor([[.1, -.2], [.3, .5]]),
        discount=torch.tensor([[1., 0.], [1., 1.]]),
        observation=torch.randn(2, 2, 4),
        prev_action=torch.zeros(2, 2, 2),
        env_id=torch.zeros(2, 2, dtype=torch.int64))
    experience = Experience(
        time_step=inputs, action=torch.zeros(2, 2, 2),
        rollout_info=BafcInfo(action=torch.zeros(2, 2, 2),
                              bootstrap_mask=torch.ones(2, 2, 2)))
    algorithm._processed_experience_spec = dist_utils.extract_spec(
        experience, from_dim=2)
    algorithm._exp_contains_step_type = True
    return experience


@contextmanager
def _summary_capture(record=True):
    values = {}

    def scalar(name, data, *args, **kwargs):
        values[name] = (data.detach().clone() if isinstance(data, torch.Tensor)
                        else data)

    with mock.patch.object(alf.summary, 'should_record_summaries',
                           return_value=record), \
         mock.patch.object(alf.summary, 'is_summary_enabled',
                           return_value=record), \
         mock.patch.object(alf.summary, 'get_global_counter',
                           return_value=1000), \
         mock.patch.object(alf.summary, 'scalar', side_effect=scalar), \
         mock.patch.object(alf.summary, 'histogram'):
        yield values


def _total_loss(algorithm, experience, loss):
    valid = (experience.step_type != StepType.LAST).to(torch.float32)
    return algorithm._aggregate_loss(loss, valid_masks=valid).loss.mean()


def _ddp_worker(rank, world_size, rendezvous, output_dir, normalize_qk):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    alf.set_default_device('cuda')
    dist.init_process_group(
        'gloo', init_method='file://' + rendezvous, rank=rank,
        world_size=world_size, timeout=timedelta(seconds=90))
    try:
        # Match alf.bin.train: CUDA tensors with a Gloo process group.
        torch.manual_seed(103)
        alf.config('make_ddp_performer', find_unused_parameters=True)
        with mock.patch('alf.algorithms.algorithm.psutil.Process',
                        return_value=_DummyProcess()):
            algorithm = _make_algorithm(
                output_dir, diagnostics=True, normalize_qk=normalize_qk)
        algorithm._temporally_independent_train_step = True
        algorithm.activate_ddp(rank)
        # Different replay observations on each rank require real gradient
        # averaging; equal post-update parameters cannot pass by coincidence.
        torch.manual_seed(500 + rank)
        experience = _experience(algorithm)
        optimizer = torch.optim.Adam(algorithm.parameters(), lr=1e-4)
        counters = []
        with _summary_capture(record=(rank == 0)) as summaries:
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                info, loss = algorithm._compute_train_info_and_loss_info(experience)
                total = _total_loss(algorithm, experience, loss)
                assert torch.isfinite(total), total
                total.backward()
                gradients = [p.grad for p in algorithm.parameters()
                             if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                optimizer.step()
                algorithm.after_update(experience.time_step, info)
                counters.append((algorithm._actor_update_counter,
                                 algorithm._critic_update_counter))
            parameters = torch.cat([p.detach().flatten()
                                    for p in algorithm.parameters()])
            reference = parameters.clone()
            dist.broadcast(reference, src=0)
            assert torch.equal(parameters, reference), (parameters - reference).abs().max()
            assert algorithm._actor_update_counter > 0
            assert algorithm._critic_update_counter > 0
            assert algorithm._gradient_chain_pending is None
            assert not any(p.requires_grad for p in
                           algorithm._target_actor_encoder.parameters())
            if rank == 0:
                assert any('backend_comparison' in name for name in summaries)
            else:
                assert not any(name.startswith('gradient_chain/rank_local/')
                               for name in summaries)
        result = dict(rank=rank, normalize_qk=normalize_qk, updates=counters,
                      gradient_tensors=len(gradients), finite=True,
                      ranks_identical=True, summary_scalars=len(summaries))
        Path(output_dir, f'rank_{rank}.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class BafcV3GradientDiagnosticsTest(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._temporary = tempfile.TemporaryDirectory(prefix='bafcv3_diag_test_')
        self.addCleanup(self._temporary.cleanup)
        self._root = Path(self._temporary.name)
        patcher = mock.patch('alf.algorithms.algorithm.psutil.Process',
                             return_value=_DummyProcess())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _algorithm(self, **kwargs):
        return _make_algorithm(self._root, **kwargs)

    def test_diagnostics_preserve_updates_and_rng(self):
        for normalize_qk, dropout in ((False, 0.), (True, 0.),
                                      (False, .1), (True, .1)):
            with self.subTest(normalize_qk=normalize_qk, dropout=dropout):
                torch.manual_seed(111)
                baseline = self._algorithm(diagnostics=False,
                                           normalize_qk=normalize_qk, dropout=dropout)
                initialization_rng = torch.get_rng_state()
                torch.manual_seed(111)
                diagnostic = self._algorithm(diagnostics=True,
                                             normalize_qk=normalize_qk, dropout=dropout)
                self.assertTensorEqual(initialization_rng, torch.get_rng_state())
                experience = _experience(baseline)
                diagnostic._processed_experience_spec = baseline._processed_experience_spec
                diagnostic._exp_contains_step_type = True
                backend_state = (
                    torch.backends.cuda.flash_sdp_enabled(),
                    torch.backends.cuda.mem_efficient_sdp_enabled(),
                    torch.backends.cuda.math_sdp_enabled())
                optimizers = [torch.optim.Adam(alg.parameters(), lr=1e-4)
                              for alg in (baseline, diagnostic)]
                all_names = set()
                for update in range(3):
                    results = []
                    for algorithm, optimizer in zip((baseline, diagnostic), optimizers):
                        torch.manual_seed(900 + update)
                        optimizer.zero_grad(set_to_none=True)
                        with _summary_capture() as summaries:
                            info = algorithm._collect_train_info_parallelly(experience)
                            loss = algorithm.calc_loss(info)
                            total = _total_loss(algorithm, experience, loss)
                            self.assertTrue(torch.isfinite(total))
                            total.backward()
                            gradients = {name: (parameter.grad.detach().clone()
                                                if parameter.grad is not None else None)
                                         for name, parameter in algorithm.named_parameters()}
                            optimizer.step()
                            algorithm.after_update(experience.time_step, info)
                        results.append((total.detach(), gradients,
                                        torch.get_rng_state().clone(),
                                        [x.detach().clone() for x in alf.nest.flatten(info)
                                         if isinstance(x, torch.Tensor)]))
                        if algorithm is diagnostic:
                            all_names.update(summaries)
                            if not dropout:
                                for name, value in summaries.items():
                                    if ('backend_comparison/' in name
                                            and name.endswith('/relative_l2_error')):
                                        self.assertLess(float(value), 1e-4, name)
                            if update == 1:
                                snapshot = algorithm._gradient_chain_intervals.snapshot(1)
                                self.assertEqual(snapshot[
                                    'optimizer_gradients/critic/latest/age_updates'], 1)
                                self.assertEqual(snapshot[
                                    'optimizer_gradients/encoder/latest/age_updates'], 1)
                        self.assertIsNone(algorithm._gradient_chain_pending)
                        self.assertTrue(all(
                            layer._gradient_chain_capture is None for layer in
                            algorithm._actor_encoder._transformer.layers))
                    torch.testing.assert_close(results[0][0], results[1][0], atol=2e-6, rtol=2e-5)
                    self.assertTensorEqual(results[0][2], results[1][2])
                    self.assertEqual(backend_state, (
                        torch.backends.cuda.flash_sdp_enabled(),
                        torch.backends.cuda.mem_efficient_sdp_enabled(),
                        torch.backends.cuda.math_sdp_enabled()))
                    for left, right in zip(results[0][3], results[1][3]):
                        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
                    for name, gradient in results[0][1].items():
                        other = results[1][1][name]
                        with self.subTest(update=update, parameter=name):
                            if gradient is None:
                                self.assertIsNone(other)
                            else:
                                torch.testing.assert_close(gradient, other, atol=2e-6, rtol=2e-5)
                    for left, right in zip(baseline.parameters(), diagnostic.parameters()):
                        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
                self.assertEqual(set(dict(baseline.named_buffers())),
                                 set(dict(diagnostic.named_buffers())))
                for name, buffer in baseline.named_buffers():
                    torch.testing.assert_close(
                        buffer, dict(diagnostic.named_buffers())[name],
                        atol=2e-6, rtol=2e-5)
                if not dropout:
                    self.assertIn('gradient_chain/backend_comparison/tokens/relative_l2_error',
                                  all_names)
                else:
                    self.assertTrue(any('skipped_dropout' in name for name in all_names))
                self.assertIn('gradient_chain/rank_local/parameters/total/norm', all_names)
                self.assertIn('gradient_chain/rank_local/features/leaf_0/through_descendants/norm',
                              all_names)
                self.assertTrue(any('/cotangent/' in name for name in all_names))
                self.assertTrue(any('gradient_chain/intervals/td/' in name for name in all_names))

    def test_expensive_diagnostics_follow_summary_gate(self):
        for debug_summaries in (False, True):
            for scheduled in (False, True):
                if debug_summaries and scheduled:
                    continue
                with self.subTest(debug_summaries=debug_summaries, scheduled=scheduled):
                    algorithm = self._algorithm(
                        diagnostics=True, debug_summaries=debug_summaries)
                    experience = _experience(algorithm)
                    with _summary_capture(scheduled), \
                         mock.patch.object(algorithm._actor_encoder,
                                           'capture_gradient_chain') as capture, \
                         mock.patch.object(algorithm,
                                           '_compare_actor_attention_backends') as compare:
                        info = algorithm._collect_train_info_parallelly(experience)
                        loss = algorithm.calc_loss(info)
                        _total_loss(algorithm, experience, loss).backward()
                    capture.assert_not_called()
                    compare.assert_not_called()
                    self.assertIsNone(algorithm._gradient_chain_pending)
                    # Cheap interval measurements are retained between summaries.
                    snapshot = algorithm._gradient_chain_intervals.snapshot(0)
                    self.assertTrue(any(name.startswith('dqde/') for name in snapshot))
                    self.assertTrue(any(name.startswith('td/') for name in snapshot))

    def test_capture_cleanup_after_backward_failure(self):
        algorithm = self._algorithm(diagnostics=True)
        experience = _experience(algorithm)
        with _summary_capture(), \
             mock.patch('torch.autograd.grad', side_effect=RuntimeError('test backward failure')):
            with self.assertRaisesRegex(RuntimeError, 'test backward failure'):
                algorithm._collect_train_info_parallelly(experience)
        self.assertTrue(all(layer._gradient_chain_capture is None for layer in
                            algorithm._actor_encoder._transformer.layers))
        self.assertIsNone(algorithm._gradient_chain_pending)

    def test_td_residual_masks_and_interval_ages(self):
        algorithm = self._algorithm(diagnostics=True, use_bootstrap_critics=True)
        algorithm._gradient_chain_step = 7
        step_type = torch.tensor([[StepType.MID, StepType.LAST],
                                  [StepType.MID, StepType.MID],
                                  [StepType.MID, StepType.MID]], dtype=torch.int32)
        bootstrap = torch.tensor([[[1., 1.], [1., 1.]],
                                  [[0., 1.], [1., 0.]],
                                  [[1., 1.], [1., 1.]]])
        value = torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2)
        value[-1] = 1e20  # Padded online values must not enter diagnostic statistics.
        targets = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) + 10
        reward = torch.tensor([[0., 0.], [1., 2.], [3., 4.]])
        discount = torch.tensor([[1., 1.], [1., 0.], [.5, 1.]])
        info = BafcInfo(step_type=step_type, reward=reward, discount=discount,
                        bootstrap_mask=bootstrap,
                        critic=BafcCriticInfo(critic=value, target_critic=targets))
        with _summary_capture(False):
            algorithm._calc_critic_loss(info)
        snapshot = algorithm._gradient_chain_intervals.snapshot(9)
        self.assertTrue(snapshot)
        # Assertions for the vectorized [actor, critic] statistic layout.
        gamma = algorithm._critic_losses[0].gamma
        expected_target = reward[1:, :, None] + gamma * discount[1:, :, None] * targets[1:]
        expected_residual = expected_target[..., None] - value[:-1]
        for critic in range(2):
            mask = (step_type[:-1] != StepType.LAST) & bootstrap[:-1, :, critic].bool()
            for actor in range(2):
                selected = expected_residual[:, :, actor, critic][mask].double()
                prefix = 'td/residual/latest/'
                self.assertTensorClose(snapshot[prefix + 'abs_mean'][actor, critic],
                                       selected.abs().mean())
                self.assertTensorClose(snapshot[prefix + 'rms'][actor, critic],
                                       selected.square().mean().sqrt())
                self.assertTensorClose(snapshot[prefix + 'abs_max'][actor, critic],
                                       selected.abs().max())
                self.assertEqual(snapshot[prefix + 'count'][actor, critic], selected.numel())
        self.assertEqual(snapshot['td/residual/latest/age_updates'], 2)
        self.assertEqual(snapshot['td/residual/interval/updates'], 1)
        self.assertEqual(snapshot['td/residual/interval/peak_abs_max_update_id'].unique().tolist(), [7])

    def test_td_target_uses_exact_forward_value_despite_cancellation(self):
        algorithm = self._algorithm(diagnostics=True, debug_summaries=False)
        # The one-step target is exactly one. Float32 cannot recover it from
        # prediction + residual because 1e20 + (1 - 1e20) rounds to zero.
        prediction = torch.full((2, 1, 2, 2), 1e20)
        info = BafcInfo(
            step_type=torch.full((2, 1), StepType.MID, dtype=torch.int32),
            reward=torch.tensor([[0.], [1.]]),
            discount=torch.zeros(2, 1),
            critic=BafcCriticInfo(
                critic=prediction, target_critic=torch.zeros(2, 1, 2)))
        with _summary_capture(False):
            algorithm._calc_critic_loss(info)
        snapshot = algorithm._gradient_chain_intervals.snapshot()
        target = snapshot['td/target/latest/mean']
        value = snapshot['td/value/latest/mean']
        residual = snapshot['td/residual/latest/mean']
        self.assertTensorEqual(target, torch.ones_like(target))
        self.assertTensorEqual(value + residual, torch.zeros_like(value))
        self.assertTensorEqual(snapshot['td/target/latest/count'],
                               torch.ones_like(target, dtype=torch.int64))

    def test_checkpoint_restore_clears_cache_and_resumes_update_ids(self):
        original = self._algorithm(diagnostics=True)
        experience = _experience(original)
        optimizer = torch.optim.Adam(original.parameters(), lr=1e-4)
        with _summary_capture(False):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                info = original._collect_train_info_parallelly(experience)
                _total_loss(original, experience, original.calc_loss(info)).backward()
                optimizer.step()
                original.after_update(experience.time_step, info)
        self.assertTrue(original._gradient_chain_intervals.snapshot())
        restored = self._algorithm(diagnostics=True)
        restored._gradient_chain_intervals.record('stale', torch.ones(2), 100)
        restored._gradient_chain_pending = (torch.ones((), requires_grad=True),) * 3
        restored.load_state_dict(original.state_dict())
        expected_step = original._actor_update_counter + original._critic_update_counter
        self.assertEqual(restored._gradient_chain_step, expected_step)
        self.assertEqual(restored._gradient_chain_intervals.snapshot(), {})
        self.assertIsNone(restored._gradient_chain_pending)
        experience = _experience(restored)
        with _summary_capture(False):
            info = restored._collect_train_info_parallelly(experience)
            _total_loss(restored, experience, restored.calc_loss(info)).backward()
            restored.after_update(experience.time_step, info)
        self.assertEqual(restored._gradient_chain_step, expected_step + 1)
        snapshot = restored._gradient_chain_intervals.snapshot()
        self.assertTrue(snapshot)
        for name, value in snapshot.items():
            if name.endswith('/latest/update_id'):
                self.assertEqual(int(value), expected_step)

    def test_two_gpu_ddp_smoke(self):
        if os.environ.get('BAFCV3_DDP_SMOKE') != '1':
            self.skipTest('Set BAFCV3_DDP_SMOKE=1 to run the two-GPU smoke test')
        if torch.cuda.device_count() < 2:
            self.skipTest('Requires two CUDA devices')
        for normalize_qk in (False, True):
            output = self._root / ('qknorm_' + str(normalize_qk))
            output.mkdir()
            processes = mp.spawn(
                _ddp_worker, args=(2, str(output / 'rendezvous'),
                                  str(output), normalize_qk),
                nprocs=2, join=False)
            deadline = time.monotonic() + 120
            try:
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail('Two-GPU DDP smoke exceeded 120 seconds')
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)
            results = [json.loads((output / f'rank_{rank}.json').read_text())
                       for rank in range(2)]
            self.assertTrue(all(result['finite'] and result['ranks_identical']
                                for result in results))
            self.assertGreater(results[0]['summary_scalars'], results[1]['summary_scalars'])
            if os.environ.get('BAFCV3_DIAGNOSTIC_ARTIFACT_DIR'):
                artifact = Path(os.environ['BAFCV3_DIAGNOSTIC_ARTIFACT_DIR'])
                artifact.mkdir(parents=True, exist_ok=True)
                (artifact / f'ddp_qknorm{normalize_qk}.json').write_text(
                    json.dumps(dict(backend='gloo', devices='cuda', world_size=2,
                                    ranks=results), indent=2) + '\n')


if __name__ == '__main__':
    alf.test.main()
