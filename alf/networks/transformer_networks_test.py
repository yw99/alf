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

from absl.testing import parameterized
import math
from unittest import mock

import torch
import torch.nn as nn
from alf.networks import TransformerNetwork
from alf.networks.transformer_networks import (
    TransformerEncoder, _InspectableTransformerEncoderLayer,
    _normalize_attention_vector)

import alf


class TransformerNetworkTest(parameterized.TestCase, alf.test.TestCase):

    @parameterized.parameters(True, False)
    def test_transformer_network(self, centralized_memory=True):
        d_model = 32
        core_size = 2
        memory_size = 128
        num_memory_layers = 8
        input_tensor_spec = [
            alf.TensorSpec((), dtype=torch.int64),
            alf.TensorSpec((3, 7, 7), dtype=torch.float32)
        ]
        input_preprocessors = [
            nn.Sequential(nn.Embedding(100, d_model),
                          alf.layers.Reshape((1, d_model))),
            nn.Sequential(alf.layers.Conv2D(3, d_model, kernel_size=1),
                          alf.layers.Reshape((d_model, 49)),
                          alf.layers.Transpose())
        ]
        transformer = TransformerNetwork(
            input_tensor_spec,
            memory_size=memory_size,
            core_size=core_size,
            num_prememory_layers=2,
            num_memory_layers=num_memory_layers,
            num_attention_heads=8,
            d_ff=d_model,
            centralized_memory=centralized_memory,
            input_preprocessors=input_preprocessors)

        state_spec = transformer.state_spec
        if centralized_memory:
            self.assertEqual(len(state_spec), 1)
            self.assertEqual(state_spec[0][0].shape, (memory_size, d_model))
        else:
            self.assertEqual(len(state_spec), 8)
            for i in range(num_memory_layers):
                self.assertEqual(state_spec[i][0].shape,
                                 (memory_size, d_model))
        batch_size = 64
        x = [
            torch.randint(100, size=(batch_size, )),
            torch.rand((batch_size, 3, 7, 7))
        ]
        state = alf.utils.spec_utils.zeros_from_spec(transformer.state_spec,
                                                     batch_size)
        y, state = transformer(x, state)

        self.assertEqual(y.shape, (batch_size, core_size * d_model))


class TransformerEncoderTest(parameterized.TestCase, alf.test.TestCase):

    def _encoder(self, **kwargs):
        return TransformerEncoder(
            alf.TensorSpec((5, 8)), num_layers=2, num_attention_heads=2,
            dropout=0., **kwargs)

    def test_default_stock_initialization_and_checkpoint(self):
        torch.manual_seed(912)
        encoder = self._encoder()
        rng = torch.get_rng_state()
        torch.manual_seed(912)
        reference = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                8, 2, dim_feedforward=32, dropout=0., batch_first=True,
                activation='gelu'), 2)
        self.assertEqual(set(encoder._transformer.state_dict()),
                         set(reference.state_dict()))
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, encoder._transformer.state_dict()[name]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        restored = self._encoder()
        restored.load_state_dict(encoder.state_dict(), strict=True)
        x = torch.randn(3, 5, 8)
        self.assertTrue(torch.equal(
            encoder(x)[0], reference(encoder._pos_encoder(x))[:, 0]))
        with mock.patch.object(
                _InspectableTransformerEncoderLayer, '_attention',
                side_effect=AssertionError('default must use stock attention')):
            self.assertTrue(torch.equal(encoder(x)[0], restored(x)[0]))

    @parameterized.parameters(
        (False, False), (True, False), (False, True), (True, True))
    def test_capture_preserves_training_graph_and_rng(self, pre_ln, qk_norm):
        encoder = TransformerEncoder(
            alf.TensorSpec((5, 8)), 2, 2, dropout=.2,
            norm_first=pre_ln, final_norm=pre_ln, normalize_qk=qk_norm)
        x = torch.randn(3, 5, 8, requires_grad=True)
        parameters = tuple(encoder.parameters())
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        expected = encoder(x)[0]
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (x, *parameters))
        expected_rng = torch.get_rng_state()
        expected_cuda_rng = torch.cuda.get_rng_state_all() if cuda_rng else []
        torch.set_rng_state(rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        with encoder.capture_gradient_chain() as capture:
            actual = encoder(x)[0]
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        if cuda_rng:
            for a, b in zip(expected_cuda_rng, torch.cuda.get_rng_state_all()):
                self.assertTrue(torch.equal(a, b))
        actual_grads = torch.autograd.grad(
            actual.square().sum(), (x, *parameters), retain_graph=True)
        self.assertTrue(torch.equal(expected, actual))
        for expected_grad, actual_grad in zip(expected_grads, actual_grads):
            self.assertTrue(torch.equal(expected_grad, actual_grad))
        self.assertEqual(capture['layers/0/q'].shape, (3, 2, 5, 4))
        self.assertIs(capture['output'], actual)
        intermediate = capture['layers/0/attention']
        gradient, = torch.autograd.grad(actual.sum(), intermediate)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertIsNone(encoder._gradient_chain_capture)
        self.assertTrue(all(layer._gradient_chain_capture is None
                            for layer in encoder._transformer.layers))

    def test_capture_cleanup_on_exception_and_nested_capture(self):
        encoder = self._encoder()
        with self.assertRaisesRegex(ValueError, 'test failure'):
            with encoder.capture_gradient_chain():
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    with encoder.capture_gradient_chain():
                        pass
                raise ValueError('test failure')
        self.assertIsNone(encoder._gradient_chain_capture)
        self.assertTrue(all(layer._gradient_chain_capture is None
                            for layer in encoder._transformer.layers))

    def test_final_norm_precedes_core_projection(self):
        encoder = self._encoder(norm_first=True, final_norm=True,
                                core_size=2, core_embedding_dim=3)
        x = torch.randn(2, 5, 8) * 1000
        with encoder.capture_gradient_chain() as capture:
            output = encoder(x)[0]
        normalized = capture['final_norm']
        self.assertTensorClose(normalized.mean(dim=-1), torch.zeros(2, 5), epsilon=1e-6)
        self.assertTensorClose(normalized.square().mean(dim=-1), torch.ones(2, 5),
                            epsilon=1e-5)
        self.assertTrue(torch.equal(
            output, encoder._core_fc(normalized[:, :2].reshape(2, -1))))

    @parameterized.parameters(torch.float32, torch.float64, torch.float16,
                              torch.bfloat16)
    def test_qk_normalization_extremes_and_gradients(self, dtype):
        eps = 1e-6
        maximum = torch.finfo(dtype).max
        x = torch.tensor([[0., 0., 0., 0.],
                          [1e-8, -1e-8, 0., 0.],
                          [1., 2., -3., 4.],
                          [maximum, -maximum, maximum, maximum]],
                         dtype=dtype, requires_grad=True)
        actual = _normalize_attention_vector(x, eps)
        ordinary = x[:3].double()
        expected = torch.cat((
            2 * ordinary / torch.linalg.vector_norm(
                ordinary, dim=-1, keepdim=True).clamp_min(eps),
            x.new_tensor([[1., -1., 1., 1.]], dtype=torch.float64)))
        self.assertTensorClose(actual.double(), expected,
                            epsilon=2e-2 if dtype == torch.bfloat16 else 1e-3)
        self.assertTrue(torch.isfinite(actual).all())
        # Half precision cannot represent the zero-vector derivative 1/eps;
        # production uses float32. Check every representable derivative.
        if dtype in (torch.float32, torch.float64):
            gradient, = torch.autograd.grad(
                (actual * torch.tensor([1., 2., 3., 4.])).sum(), x)
            self.assertTrue(torch.isfinite(gradient).all())
        y = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: _normalize_attention_vector(value, eps), (y,)))
        self.assertTensorClose(_normalize_attention_vector(y, eps),
                            _normalize_attention_vector(y * 1e8, eps))

    def test_qk_normalization_bounds_logits_and_runs_in_eval(self):
        encoder = self._encoder(normalize_qk=True).eval()
        for layer in encoder._transformer.layers:
            with torch.no_grad():
                layer.self_attn.in_proj_weight[:16].mul_(1e10)
        x = torch.randn(3, 5, 8) * 1e8
        with torch.no_grad():
            with mock.patch('torch._transformer_encoder_layer_fwd',
                            side_effect=AssertionError('must not bypass QK norm')):
                expected = encoder(x)[0]
                with encoder.capture_gradient_chain() as capture:
                    actual = encoder(x)[0]
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.isfinite(actual).all())
        for i in range(2):
            q, k = (capture['layers/%d/%s' % (i, name)] for name in ('q', 'k'))
            logits = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            self.assertLessEqual(logits.abs().max().item(), 2.000001)

    @parameterized.parameters((False, False), (False, True), (True, False),
                              (True, True))
    def test_capture_attention_masks_match_stock(self, pre_ln, causal):
        layer = _InspectableTransformerEncoderLayer(
            8, 2, dropout=0., batch_first=True, norm_first=pre_ln)
        x = torch.randn(3, 5, 8, requires_grad=True)
        mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
        padding = torch.zeros(3, 5, dtype=torch.bool)
        padding[:, -1] = True
        for use_padding in (False, True):
            args = dict(src_mask=mask,
                        src_key_padding_mask=padding if use_padding else None,
                        is_causal=causal)
            expected = layer(x, **args)
            expected_grad, = torch.autograd.grad(expected.square().sum(), x)
            layer._gradient_chain_capture = ({}, '')
            try:
                actual = layer(x, **args)
            finally:
                layer._gradient_chain_capture = None
            actual_grad, = torch.autograd.grad(actual.square().sum(), x)
            self.assertTensorClose(expected, actual, epsilon=1e-6)
            self.assertTensorClose(expected_grad, actual_grad, epsilon=1e-6)

    def test_normalized_production_shape_matches_math_backend(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA attention backend comparison')
        from torch.nn.attention import SDPBackend, sdpa_kernel
        shape = (10, 1, 277, 512)
        raw_q = (torch.randn(shape, device='cuda') * 1e5).requires_grad_()
        raw_k = (torch.randn(shape, device='cuda') * 1e5).requires_grad_()
        value = torch.randn(shape, device='cuda', requires_grad=True)
        cotangent = torch.randn(shape, device='cuda')
        # The configured one-layer encoder reads only token zero. Exercise
        # exactly that backward, including gradients into every key/value.
        cotangent[:, :, 1:] = 0

        def evaluate():
            query = _normalize_attention_vector(raw_q, 1e-6)
            key = _normalize_attention_vector(raw_k, 1e-6)
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value)
            gradients = torch.autograd.grad(
                output, (raw_q, raw_k, value), grad_outputs=cotangent)
            return output, gradients

        actual, actual_grads = evaluate()
        with sdpa_kernel(SDPBackend.MATH):
            expected, expected_grads = evaluate()
        for left, right in zip((actual, *actual_grads),
                               (expected, *expected_grads)):
            self.assertTrue(torch.isfinite(left).all())
            relative_error = (left.double() - right.double()).norm() / right.double().norm()
            self.assertLess(relative_error.item(), 2e-5)

    def test_invalid_normalization_epsilon(self):
        for value in (0., -1., float('inf'), float('nan')):
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                self._encoder(qk_norm_eps=value)


if __name__ == '__main__':
    alf.test.main()
