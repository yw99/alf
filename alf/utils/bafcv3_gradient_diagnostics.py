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
"""Detached, finite-safe measurements for BAFCv3 gradient diagnostics.

These helpers never change gradients or draw random numbers. Statistics describe
the finite, selected elements; the explicit element counts must accompany them
so a nonfinite gradient cannot be mistaken for a small gradient. Per-update
accumulation keeps scalar tensors on their original device without synchronizing
them to the host. Summary emission is intentionally the synchronization boundary.
"""

from itertools import product

import torch

import alf


def _selected_mask(value, mask):
    if mask is None:
        return torch.ones_like(value, dtype=torch.bool)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=value.device)
    # Masks select leading batch/time dimensions unless already fully shaped.
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return torch.broadcast_to(mask, value.shape)


def tensor_statistics(value, mask=None, reduce_dims=None):
    """Return detached statistics, with floating reductions in float64.

    ``mask`` selects leading dimensions (for example ``[T, B]`` for a
    ``[T, B, actors, critics]`` tensor), or can match the complete tensor shape.
    ``reduce_dims=None`` reduces all elements to scalar statistics. Explicit
    reduction dimensions retain the other dimensions in their original order;
    e.g. ``reduce_dims=(0, 1)`` produces actor-by-critic statistic matrices for a
    ``[T, B, actors, critics]`` tensor, without per-actor Python/GPU operations.
    An empty tuple computes elementwise statistics without reducing any axes.

    Nonfinite and masked elements are excluded from all magnitude reductions.
    Empty selections and selections with no finite elements have zero magnitude
    statistics. Counts distinguish these cases. No host synchronization occurs.
    """
    value = value.detach()
    if reduce_dims is None:
        reduce_dims = tuple(range(value.ndim))
    else:
        reduce_dims = tuple(reduce_dims)
        if any(not -value.ndim <= dim < value.ndim for dim in reduce_dims):
            raise ValueError('Reduction dimension is outside the tensor rank')
        reduce_dims = tuple(dim % value.ndim for dim in reduce_dims)
        if len(set(reduce_dims)) != len(reduce_dims):
            raise ValueError('Reduction dimensions must be distinct')

    def reduce_sum(tensor):
        if reduce_dims:
            return tensor.sum(dim=reduce_dims)
        return tensor.to(torch.int64) if tensor.dtype == torch.bool else tensor

    selected = _selected_mask(value, mask)
    finite = selected & torch.isfinite(value)
    count = reduce_sum(selected)
    finite_count = reduce_sum(finite)
    values = torch.where(finite, value, 0).to(torch.float64)
    denominator = finite_count.clamp_min(1)
    # Float32 cotangents as large as 1e27 must not be squared in float32.
    square_sum = reduce_sum(values.square())
    abs_values = values.abs()
    if not reduce_dims:
        abs_max = abs_values
    elif any(value.shape[dim] == 0 for dim in reduce_dims):
        output_shape = [size for dim, size in enumerate(value.shape)
                        if dim not in reduce_dims]
        abs_max = values.new_zeros(output_shape)
    else:
        abs_max = abs_values.amax(dim=reduce_dims)
    return dict(
        count=count,
        finite_count=finite_count,
        nonfinite_count=count - finite_count,
        nan_count=reduce_sum(selected & torch.isnan(value)),
        posinf_count=reduce_sum(selected & torch.isposinf(value)),
        neginf_count=reduce_sum(selected & torch.isneginf(value)),
        mean=reduce_sum(values) / denominator,
        abs_mean=reduce_sum(abs_values) / denominator,
        rms=(square_sum / denominator).sqrt(),
        norm=square_sum.sqrt(),
        abs_max=abs_max)


def _summarize_statistic(name, value):
    """Expand retained axes only when summaries are emitted, never per update."""
    if value.ndim == 0:
        alf.summary.scalar(name, value)
        return
    for index in product(*(range(size) for size in value.shape)):
        if value.ndim == 2:
            suffix = '/actor_%d/critic_%d' % index
        else:
            suffix = ''.join('/dim_%d_%d' % (dimension, position)
                             for dimension, position in enumerate(index))
        alf.summary.scalar(name + suffix, value[index])


def summarize_tensor(name, value, mask=None, histogram=True):
    """Write statistics and a histogram containing only finite selected values.

    Callers gate this helper at the desired summary cadence. Return the same
    detached statistics as :func:`tensor_statistics` for optional reuse.
    """
    statistics = tensor_statistics(value, mask)
    for key, scalar in statistics.items():
        alf.summary.scalar(name + '/' + key, scalar)
    if histogram:
        value = value.detach()
        finite_values = value[_selected_mask(value, mask)
                              & torch.isfinite(value)]
        if finite_values.numel():
            alf.summary.histogram(name + '/value', finite_values)
    return statistics


def gradient_vector_statistics(gradients):
    """Statistics of parameter gradients without concatenating their tensors.

    ``None`` gradients represent unused parameters and contribute no elements.
    If every gradient is ``None``, return empty statistics on the CPU.
    """
    gradients = [gradient for gradient in gradients if gradient is not None]
    if not gradients:
        return tensor_statistics(torch.empty(0))
    chunks = [tensor_statistics(gradient) for gradient in gradients]
    count_keys = ('count', 'finite_count', 'nonfinite_count', 'nan_count',
                  'posinf_count', 'neginf_count')
    result = {key: sum(chunk[key] for chunk in chunks) for key in count_keys}
    denominator = result['finite_count'].clamp_min(1)
    for key in ('mean', 'abs_mean'):
        result[key] = sum(chunk[key] * chunk['finite_count']
                          for chunk in chunks) / denominator
    square_sum = sum(chunk['norm'].square() for chunk in chunks)
    result['norm'] = square_sum.sqrt()
    result['rms'] = (square_sum / denominator).sqrt()
    result['abs_max'] = torch.stack(
        [chunk['abs_max'] for chunk in chunks]).amax()
    return result


def gradient_cosine_similarity(left, right):
    """Cosine of aligned gradient sequences, treating unused entries as zeros.

    Zero vectors have cosine zero. Any nonfinite gradient produces a NaN cosine,
    including nonfinite gradients opposite an unused parameter. This prevents a
    finite-only cosine from hiding numerical failures. Inputs must have matching
    parameter ordering and lengths, but may contain different ``None`` entries.
    """
    left, right = tuple(left), tuple(right)
    if len(left) != len(right):
        raise ValueError('Gradient sequences must have matching lengths')
    device = next((gradient.device for gradient in left + right
                   if gradient is not None), torch.device('cpu'))
    dot = torch.zeros((), device=device, dtype=torch.float64)
    left_squared = dot.clone()
    right_squared = dot.clone()
    for lgrad, rgrad in zip(left, right):
        if lgrad is not None:
            lgrad = lgrad.detach().to(torch.float64)
            left_squared = left_squared + lgrad.square().sum()
        if rgrad is not None:
            rgrad = rgrad.detach().to(torch.float64)
            right_squared = right_squared + rgrad.square().sum()
        if lgrad is not None and rgrad is not None:
            if lgrad.shape != rgrad.shape:
                raise ValueError('Aligned parameter gradients must match shapes')
            dot = dot + (lgrad * rgrad).sum()
    norm_product = left_squared.sqrt() * right_squared.sqrt()
    cosine = dot / norm_product.clamp_min(torch.finfo(torch.float64).tiny)
    valid = torch.isfinite(left_squared) & torch.isfinite(right_squared)
    return torch.where(valid, cosine.clamp(-1, 1), cosine.new_full((), float('nan')))


class DiagnosticIntervalAccumulator:
    """Latest measurements and interval peaks without storing computation graphs.

    Records contain scalar or batched statistics, not history buffers.
    Two-dimensional batched statistics represent actors by critics and are
    expanded to individual scalar summaries only at emission. ``update_id`` is the caller's
    monotonically increasing optimizer-update index. A peak records the *first*
    update attaining it. Emission retains latest observations so an actor summary
    can report the age of a preceding critic observation, while interval peaks
    start fresh after each emission.
    """

    _PEAK_KEYS = ('abs_max', 'rms', 'norm', 'nonfinite_count')

    def __init__(self):
        self._latest = {}
        self._interval = {}

    def record(self, name, value, update_id, mask=None, reduce_dims=None):
        """Compute and retain a tensor's statistics; return them for reuse."""
        statistics = tensor_statistics(value, mask, reduce_dims=reduce_dims)
        self.record_statistics(name, statistics, update_id)
        return statistics

    def record_statistics(self, name, statistics, update_id):
        """Retain existing detached statistics, avoiding duplicate reductions."""
        statistics = {key: value.detach() for key, value in statistics.items()}
        device = statistics['count'].device
        update_id = torch.as_tensor(
            update_id, dtype=torch.int64, device=device).detach()
        self._latest[name] = (statistics, update_id)
        if name not in self._interval:
            self._interval[name] = dict(
                updates=torch.ones((), dtype=torch.int64, device=device),
                peaks={key: (statistics[key],
                             torch.broadcast_to(update_id, statistics[key].shape))
                       for key in self._PEAK_KEYS})
            return
        interval = self._interval[name]
        interval['updates'] = interval['updates'] + 1
        for key in self._PEAK_KEYS:
            old_value, old_update_id = interval['peaks'][key]
            larger = statistics[key] > old_value
            interval['peaks'][key] = (
                torch.where(larger, statistics[key], old_value),
                torch.where(larger, update_id, old_update_id))

    def snapshot(self, current_update_id=None):
        """Return a mapping of metric suffixes to detached statistic tensors."""
        result = {}
        for name, (statistics, update_id) in self._latest.items():
            result.update({name + '/latest/' + key: value
                           for key, value in statistics.items()})
            result[name + '/latest/update_id'] = update_id
            if current_update_id is not None:
                current = torch.as_tensor(
                    current_update_id, dtype=torch.int64,
                    device=update_id.device).detach()
                result[name + '/latest/age_updates'] = current - update_id
            interval = self._interval.get(name)
            if interval is None:
                result[name + '/interval/updates'] = update_id.new_zeros(())
                continue
            result[name + '/interval/updates'] = interval['updates']
            for key, (value, peak_id) in interval['peaks'].items():
                result[name + '/interval/peak_' + key] = value
                result[name + '/interval/peak_' + key + '_update_id'] = peak_id
        return result

    def reset_interval(self):
        """Drop interval peaks while retaining latest observations."""
        self._interval.clear()

    def summarize(self, prefix, current_update_id=None, reset=True):
        """Emit the snapshot at summary cadence; optionally reset interval peaks.

        A call outside summary cadence does not emit or discard pending peaks.
        """
        if not alf.summary.should_record_summaries():
            return {}
        snapshot = self.snapshot(current_update_id)
        for name, value in snapshot.items():
            _summarize_statistic(prefix.rstrip('/') + '/' + name, value)
        if reset:
            self.reset_interval()
        return snapshot
