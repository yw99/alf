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

from unittest import mock

import torch

import alf
from alf.utils import bafcv3_gradient_diagnostics as diagnostics


class GradientDiagnosticsTest(alf.test.TestCase):

    def test_large_cotangents_and_autograd_unchanged(self):
        value = torch.tensor([3e27, -4e27], requires_grad=True)
        random_state = torch.random.get_rng_state().clone()
        stats = diagnostics.tensor_statistics(value)
        self.assertEqual(stats['norm'].dtype, torch.float64)
        self.assertTrue(torch.isfinite(stats['norm']))
        self.assertAlmostEqual(float(stats['norm']) / 1e27, 5, places=5)
        self.assertAlmostEqual(float(stats['rms']) / 1e27,
                               (25 / 2)**0.5, places=5)
        self.assertTrue(all(not entry.requires_grad for entry in stats.values()))
        self.assertIsNone(value.grad)
        self.assertEqual(torch.random.get_rng_state(), random_state)
        value.sum().backward()
        self.assertEqual(value.grad, torch.ones_like(value))

    def test_nonfinite_and_masked_values(self):
        value = torch.tensor([
            [3., float('nan'), float('inf'), -float('inf')],
            [-4., 0., 0., 0.],
            [1e30, 1e30, 1e30, 1e30]
        ])
        stats = diagnostics.tensor_statistics(
            value, torch.tensor([True, True, False]))
        for name, count in dict(count=8, finite_count=5, nonfinite_count=3,
                                nan_count=1, posinf_count=1,
                                neginf_count=1).items():
            self.assertEqual(stats[name], count)
        self.assertEqual(stats['norm'], 5)
        self.assertEqual(stats['abs_max'], 4)
        self.assertAlmostEqual(float(stats['mean']), -.2)
        self.assertAlmostEqual(float(stats['abs_mean']), 1.4)

    def test_empty_and_all_nonfinite(self):
        for value, mask in (
                (torch.empty(0), None),
                (torch.tensor([float('nan'), float('inf')]), None),
                (torch.ones(2, 3), torch.zeros(2, dtype=torch.bool))):
            stats = diagnostics.tensor_statistics(value, mask)
            self.assertEqual(stats['finite_count'], 0)
            for name in ('mean', 'abs_mean', 'rms', 'norm', 'abs_max'):
                self.assertEqual(stats[name], 0)

    def test_batched_reductions_match_individual_actor_critic_statistics(self):
        value = torch.arange(48, dtype=torch.float32).reshape(2, 2, 2, 3, 2)
        value[0, 0, 0, 0, 0] = 1e27
        value[1, 0, 1, 2, 1] = float('nan')
        value[0, 1] = float('inf')
        mask = torch.tensor([[True, False], [True, True]])
        batched = diagnostics.tensor_statistics(value, mask, reduce_dims=(0, 1, -1))
        for actor in range(2):
            for critic in range(3):
                expected = diagnostics.tensor_statistics(value[:, :, actor, critic], mask)
                for key in expected:
                    self.assertEqual(batched[key].shape, (2, 3))
                    torch.testing.assert_close(batched[key][actor, critic], expected[key])

    def test_batched_empty_axes_and_elementwise_statistics(self):
        stats = diagnostics.tensor_statistics(torch.empty(0, 3, 2), reduce_dims=(0,))
        for value in stats.values():
            self.assertEqual(value.shape, (3, 2))
            self.assertEqual(value, torch.zeros_like(value))
        stats = diagnostics.tensor_statistics(torch.empty(2, 0, 3), reduce_dims=(0,))
        for value in stats.values():
            self.assertEqual(value.shape, (0, 3))
        stats = diagnostics.tensor_statistics(
            torch.tensor([-3., float('nan')]), reduce_dims=())
        self.assertEqual(stats['abs_max'], torch.tensor([3., 0.], dtype=torch.float64))
        self.assertEqual(stats['nonfinite_count'], torch.tensor([0, 1]))
        stats = diagnostics.tensor_statistics(torch.tensor(-3.), reduce_dims=())
        self.assertEqual(stats['norm'], 3)
        with self.assertRaises(ValueError):
            diagnostics.tensor_statistics(torch.ones(2, 3), reduce_dims=(0, -2))
        with self.assertRaises(ValueError):
            diagnostics.tensor_statistics(torch.ones(2, 3), reduce_dims=(2,))

    def test_batched_interval_peak_ids_and_scalar_emission(self):
        cache = diagnostics.DiagnosticIntervalAccumulator()
        first = torch.tensor([[[1., 4.], [7., 2.]]])
        second = torch.tensor([[[3., 2.], [7., 5.]]])
        with mock.patch.object(torch.Tensor, 'item',
                               side_effect=AssertionError('host sync')):
            cache.record('td/residual', first, update_id=10, reduce_dims=(0,))
            cache.record('td/residual', second, update_id=11, reduce_dims=(0,))
        snapshot = cache.snapshot()
        self.assertEqual(snapshot['td/residual/interval/peak_abs_max_update_id'],
                         torch.tensor([[11, 10], [10, 11]]))
        events = {}
        with mock.patch.object(alf.summary, 'should_record_summaries',
                               return_value=True), \
             mock.patch.object(alf.summary, 'scalar',
                               side_effect=lambda n, v: events.update({n: v})):
            cache.summarize('chain', current_update_id=12)
        self.assertTrue(all(value.ndim == 0 for value in events.values()))
        self.assertEqual(events['chain/td/residual/latest/abs_max/actor_1/critic_0'], 7)
        self.assertEqual(events['chain/td/residual/interval/peak_abs_max_update_id'
                                '/actor_0/critic_1'], 10)
        self.assertEqual(events['chain/td/residual/latest/age_updates'], 1)

    def test_histogram_filters_nonfinite_after_reporting_counts(self):
        events = []
        with mock.patch.object(alf.summary, 'scalar',
                               side_effect=lambda n, v: events.append((n, v))), \
             mock.patch.object(alf.summary, 'histogram',
                               side_effect=lambda n, v: events.append((n, v))):
            diagnostics.summarize_tensor(
                'chain/dqde', torch.tensor([1., float('nan'), float('inf')]))
        names = [name for name, value in events]
        self.assertLess(names.index('chain/dqde/nonfinite_count'),
                        names.index('chain/dqde/value'))
        self.assertEqual(dict(events)['chain/dqde/nonfinite_count'], 2)
        self.assertEqual(dict(events)['chain/dqde/value'], torch.tensor([1.]))

    def test_parameter_vector_statistics_and_cosines(self):
        left = [torch.tensor([3e27]), None, torch.tensor([-4e27, 0.])]
        right = [torch.tensor([0.]), torch.tensor([0.]), torch.tensor([5e27, 0.])]
        stats = diagnostics.gradient_vector_statistics(left)
        reference = diagnostics.tensor_statistics(
            torch.tensor([3e27, -4e27, 0.]))
        for key in reference:
            self.assertEqual(stats[key], reference[key])
        cosine = diagnostics.gradient_cosine_similarity(left, right)
        self.assertAlmostEqual(float(cosine), -.8, places=6)
        self.assertEqual(diagnostics.gradient_cosine_similarity(
            [None, torch.zeros(2)], [torch.ones(3), None]), 0)
        self.assertEqual(diagnostics.gradient_vector_statistics([None])['count'], 0)
        self.assertTrue(torch.isnan(diagnostics.gradient_cosine_similarity(
            [torch.tensor([float('inf')])], [None])))
        with self.assertRaises(ValueError):
            diagnostics.gradient_cosine_similarity([None], [])

    def test_interval_peaks_age_reset_and_no_host_scalar_conversion(self):
        cache = diagnostics.DiagnosticIntervalAccumulator()
        with mock.patch.object(torch.Tensor, 'item',
                               side_effect=AssertionError('host sync')):
            cache.record('td', torch.tensor([1., -4.]), update_id=10)
            cache.record('td', torch.tensor([3., -3.]), update_id=11)
            cache.record('td', torch.tensor([2., -2.]), update_id=12)
        snapshot = cache.snapshot(current_update_id=13)
        self.assertEqual(snapshot['td/latest/update_id'], 12)
        self.assertEqual(snapshot['td/latest/age_updates'], 1)
        self.assertEqual(snapshot['td/latest/abs_max'], 2)
        self.assertEqual(snapshot['td/interval/updates'], 3)
        self.assertEqual(snapshot['td/interval/peak_abs_max'], 4)
        self.assertEqual(snapshot['td/interval/peak_abs_max_update_id'], 10)
        self.assertEqual(snapshot['td/interval/peak_rms_update_id'], 11)
        with mock.patch.object(alf.summary, 'should_record_summaries',
                               return_value=False):
            self.assertEqual(cache.summarize('chain'), {})
        self.assertIn('td/interval/peak_abs_max', cache.snapshot())
        with mock.patch.object(alf.summary, 'should_record_summaries',
                               return_value=True), \
             mock.patch.object(alf.summary, 'scalar') as scalar:
            emitted = cache.summarize('chain/', current_update_id=13)
        scalar.assert_any_call('chain/td/interval/peak_abs_max_update_id',
                               emitted['td/interval/peak_abs_max_update_id'])
        snapshot = cache.snapshot(current_update_id=20)
        self.assertEqual(snapshot['td/latest/age_updates'], 8)
        self.assertEqual(snapshot['td/interval/updates'], 0)
        self.assertNotIn('td/interval/peak_abs_max', snapshot)
        cache.record('td', torch.tensor([1., -1.]), update_id=21)
        snapshot = cache.snapshot()
        self.assertEqual(snapshot['td/interval/peak_abs_max'], 1)
        self.assertEqual(snapshot['td/interval/peak_abs_max_update_id'], 21)

    def test_peak_nonfinite_counts_and_first_peak_timestamp(self):
        cache = diagnostics.DiagnosticIntervalAccumulator()
        cache.record('td', torch.tensor([1., float('nan')]), update_id=1)
        cache.record('td', torch.tensor([float('inf'), float('nan')]), update_id=2)
        cache.record('td', torch.tensor([float('inf'), float('nan')]), update_id=3)
        snapshot = cache.snapshot()
        self.assertEqual(snapshot['td/interval/peak_nonfinite_count'], 2)
        self.assertEqual(snapshot['td/interval/peak_nonfinite_count_update_id'], 2)


if __name__ == '__main__':
    alf.test.main()
