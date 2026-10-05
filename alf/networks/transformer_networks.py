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

import contextlib
import math
import torch
import torch.nn as nn

import alf
from alf.networks import PreprocessorNetwork
from alf.networks.memory import FIFOMemory
from alf.nest.utils import NestConcat

import functools
import torch.nn.functional as F
from alf.initializers import variance_scaling_init
import alf.layers as layers


@alf.configurable
class TransformerNetwork(PreprocessorNetwork):
    """A Network composed of Memory and TransformerBlock.

    The following is the pseudocode for the computation:

    .. code-block:: python

        for i in range(num_prememory_layers):
            core, inputs = T_i([core, inputs], [core, inputs])
        for j in range(num_memory_layers):
            new_core, inputs = TM_j([memory_j, core, inputs], [core, inputs])
            memory_j.write(core)
            core = new_core
        return core, new_memory_state

    where T_i denotes the ``TransformerBlock``  for the i-th prememory layers
    and TM_j denotes the ``TransformerBlock`` for the j-th memory layers. memory_j
    is an ``FIFOMemory`` object (not to be confused with the ``memory`` argument
    of ``TransformerBlock.forward() function``)

    The core embedding serves the same purpose of [CLS] in the BERT model in [1],
    which is to generate a fixed dimensional representation for downstream tasks.
    Different from BERT, which only has one [CLS] embedding, we allow the option
    of having multiple core embeddings. In addition to generating a fixed dimensional
    representation, the core embedding is also used to update the memory.

    [1]. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for
         Language Understanding
    """

    def __init__(self,
                 input_tensor_spec,
                 num_prememory_layers,
                 num_attention_heads,
                 d_ff=None,
                 core_size=1,
                 use_core_embedding=True,
                 memory_size=0,
                 num_memory_layers=0,
                 return_core_only=True,
                 centralized_memory=True,
                 input_preprocessors=None,
                 name="TransformerNetwork"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If ``input_tensor_spec`` is not nested, it should
                represent a rank-2 tensor of shape ``[input_size, d_model]``, where
                ``input_size`` is the length of the input sequence, and ``d_model``
                is the dimension of embedding.
            num_prememory_layers (int): number of TransformerBlock calculation
                without using memory
            num_attention_heads (int): number of attention heads for each
                ``TransformerBlock``
            d_ff (int): the size of the hidden layer of the feedforward network
                in each ``TransformerBlock``. If None, ``TransformerBlock`` will
                calculate it as ``4*d_model``.
            memory_size (int): size of memory.
            num_memory_layers (int): number of TransformerBlock calculation
                using memory
            return_core_only (bool): If True, only return the core embedding.
                Otherwise, return all embeddings
            core_size (int): size of core (i.e. number of embeddings of core)
            use_core_embedding (bool): whether to use learnable core embedding.
                If True, will use additional learnable core embedding to augment
                the input. If False, the first ``core_size`` embeddings of the
                input are treated as core.
            centralized_memory (bool): if False, there will be a separate memory
                for each memory layers. if True, there will be a single memory
                for all the memory layers and it is updated using the last core
                embeddings.
            input_preprocessors (nested Network|nn.Module): a nest of
                stateless preprocessor networks, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with ``input_tensor_spec``. If any element is None, then
                it will be treated as math_ops.identity. This arg is helpful if
                you want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector. The output_spec of each input preprocessor i
                should be [input_size_i, d_model]. The result of all the preprocessors
                will be concatenated as a Tensor of shape ``[batch_size, input_size, d_model]``,
                where ``input_size = sum_i input_size_i``.
        """
        preprocessing_combiner = None
        if input_preprocessors is not None:
            preprocessing_combiner = NestConcat(dim=-2)
        super().__init__(input_tensor_spec,
                         input_preprocessors,
                         preprocessing_combiner=preprocessing_combiner,
                         name=name)

        assert self._processed_input_tensor_spec.ndim == 2

        input_size, d_model = self._processed_input_tensor_spec.shape
        if num_memory_layers > 0:
            assert memory_size > 0, ("memory_size needs to be set if "
                                     "num_memory_layers > 0")
            if centralized_memory:
                self._memories = [FIFOMemory(d_model, memory_size)]
            else:
                self._memories = [
                    FIFOMemory(d_model, memory_size)
                    for _ in range(num_memory_layers)
                ]
        else:
            self._memories = []
        self._centralized_memory = centralized_memory

        self._core_size = core_size
        if use_core_embedding:
            self._core_embedding = nn.Parameter(
                torch.empty(1, core_size, d_model))
            nn.init.uniform_(self._core_embedding, -0.1, 0.1)
        else:
            self._core_embedding = None

        self._state_spec = [mem.state_spec for mem in self._memories]
        self._num_memory_layers = num_memory_layers
        self._num_prememory_layers = num_prememory_layers

        self._transformers = nn.ModuleList()

        for i in range(num_prememory_layers):
            self._transformers.append(
                alf.layers.TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_attention_heads,
                    memory_size=input_size + core_size,
                    positional_encoding='abs' if i == 0 else 'none'))

        for i in range(num_memory_layers):
            self._transformers.append(
                alf.layers.TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_attention_heads,
                    memory_size=memory_size + input_size + core_size,
                    positional_encoding='abs' if i == 0 else 'none'))

        self._return_core_only = return_core_only

    @property
    def state_spec(self):
        return self._state_spec

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (nested Tensor): consistent with ``input_tensor_spec`` provided
                at ``__init__()``
            state (nested Tensor): states
        Returns:
            - Tensor: shape is [B, core_size * d_model] if ``return_core_only``,
                    and [B, core_size + input_size, d_model] if not ``return_core_only``,
                    where ``input_size`` is the number of embeddings from the
                    (processed) input.
            - nested Tensor: network states.
        """
        z, _ = super().forward(inputs, state)
        batch_size = z.shape[0]
        if self._core_embedding is not None:
            core_embedding = self._core_embedding.expand(batch_size, -1, -1)
            query = torch.cat([core_embedding, z], dim=-2)
        else:
            query = z
        for i in range(self._num_prememory_layers):
            query = self._transformers[i].forward(query)

        if self._num_memory_layers > 0 and self._centralized_memory:
            memory = self._memories[0]
            memory.from_states(state[0])
            mem = memory.memory()
            for i in range(self._num_memory_layers):
                transformer = self._transformers[self._num_prememory_layers +
                                                 i]
                query = transformer.forward(memory=torch.cat([mem, query],
                                                             dim=-2),
                                            query=query)
            memory.write(query[:, :self._core_size, :])
        else:
            for i in range(self._num_memory_layers):
                memory = self._memories[i]
                memory.from_states(state[i])
                transformer = self._transformers[self._num_prememory_layers +
                                                 i]
                new_query = transformer.forward(memory=torch.cat(
                    [memory.memory(), query], dim=-2),
                                                query=query)
                memory.write(query[:, :self._core_size, :])
                query = new_query

        new_state = [mem.states for mem in self._memories]

        if self._return_core_only:
            return query[:, :self._core_size, :].reshape(batch_size,
                                                         -1), new_state
        else:
            return query, new_state


@alf.configurable
class SocialAttentionNetwork(PreprocessorNetwork):
    """Simple graph encoding network, which takes as input a set of objects and
        outputs one encoded feature vector.
        Reference:
            Leurent et al "Social Attention for Autonomous Decision-Making in
            Dense Traffic", arXiv:1911.12250
    """

    def __init__(self,
                 input_tensor_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 fc_layer_params=(128, 128),
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 num_of_heads=1,
                 last_layer_size=None,
                 last_activation=None,
                 last_kernel_initializer=None,
                 name="SocialAttentionNetwork"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            input_preprocessors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with ``input_tensor_spec``. This arg is helpful if you
                want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                ``input_tensor_spec`` as the input to compute the processed
                tensor spec. For example, see ``alf.nest.utils.NestConcat``. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            fc_layer_params (tuple[int]): a tuple of integers
                representing FC layer sizes for generating embeddings.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If None, a variance_scaling_initializer will be
                used.
            use_fc_bn (bool): whether use Batch Normalization for fc layers.
            num_of_heads (int): number of heads for the mult-head attention
            last_layer_size (None): nt used; for interface compatibility
            last_activation (None): not used; for interface compatibility
            last_kernel_initializer (None): not used; for interface compatibility
            last_use_fc_bn (None): not used; for interface compatibility
            name (str):
        """
        super().__init__(input_tensor_spec,
                         input_preprocessors,
                         preprocessing_combiner=preprocessing_combiner,
                         name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        embedding_layers = nn.ModuleList()
        assert self._processed_input_tensor_spec.ndim == 2, (
            "expect the "
            "processed spec to have the shape of [entity_num, feature_dim]")
        input_size = self._processed_input_tensor_spec.shape[-1]
        for size in fc_layer_params:
            embedding_layers.append(
                layers.FC(input_size,
                          size,
                          activation=activation,
                          use_bn=use_fc_bn,
                          kernel_initializer=kernel_initializer))
            input_size = size
        self._embedding_layers = embedding_layers

        fea_dim = input_size
        assert fea_dim % num_of_heads == 0, "improper value for num_of_heads"
        self._num_of_heads = num_of_heads
        self._fea_dim_per_head = fea_dim // num_of_heads

        # attention related layers
        self._value_proj = layers.FC(fea_dim,
                                     fea_dim,
                                     use_bias=False,
                                     kernel_initializer=kernel_initializer)
        self._key_proj = layers.FC(fea_dim,
                                   fea_dim,
                                   use_bias=False,
                                   kernel_initializer=kernel_initializer)
        self._query_proj = layers.FC(fea_dim,
                                     fea_dim,
                                     use_bias=False,
                                     kernel_initializer=kernel_initializer)

        self._simple_attention = alf.layers.SimpleAttention()

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):  with the shape of [B, N, d], where
                B denotes batch size, N the number of entities, and d the
                feature dimension
            state (nested Tensor): states
        Returns:
            - Tensor: shape is [B, d'], where d' denotes the output dimension of
            the last layer specified by fc_layer_params (i.e. fc_layer_params[-1])
        """
        x, _ = super().forward(inputs, state)

        B, N, d = x.shape

        x = x.reshape(B * N, -1)

        # forward through embedding layers shared across all entities
        for i, net in enumerate(self._embedding_layers):
            x = net(x)

        # [B, N, d'] (batch, entities, fea_dim)
        X = x.reshape(B, N, -1)

        key = X[:, 0]

        # [B, head * d'] -> [B, 1, head, d']
        query = self._query_proj(key).reshape(B, 1, self._num_of_heads,
                                              self._fea_dim_per_head)

        key = self._key_proj(X).reshape(B, N, self._num_of_heads,
                                        self._fea_dim_per_head)

        value = self._value_proj(X).reshape(B, N, self._num_of_heads,
                                            self._fea_dim_per_head)

        # [B, N, head, d'] -> [B, head, N, d']
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        v, _ = self._simple_attention(query=query, key=key, value=value)
        out = v.reshape(B, -1)
        return out, state


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from "Attention is All You Need".
    Adds positional information to token embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        # register_buffer ensures 'pe' is not a model parameter, but is saved with the model
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of same shape with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def _normalize_attention_vector(x, eps):
    """Return ``sqrt(head_dim) * x / max(norm(x), eps)`` without overflow.

    Scaling before the reduction also handles finite vectors whose norm is
    larger than the largest representable value. Half-precision reductions
    are performed in float32.
    """
    dtype = x.dtype
    work = x.float() if dtype in (torch.float16, torch.bfloat16) else x
    scale = work.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scaled = work / scale
    denominator = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    denominator = torch.maximum(denominator, eps / scale)
    return (scaled / denominator * math.sqrt(x.shape[-1])).to(dtype)


class _InspectableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Stock parameter layout with optional Q/K normalization and capture.

    The default path is PyTorch's implementation. The explicit path uses the
    same packed projection and SDPA layouts, without materializing an attention
    matrix or selecting a different attention backend.
    """

    def __init__(self, *args, normalize_qk=False, qk_norm_eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_qk = normalize_qk
        self.qk_norm_eps = qk_norm_eps
        self._gradient_chain_capture = None

    def _capture(self, name, value):
        if self._gradient_chain_capture is not None:
            values, prefix = self._gradient_chain_capture
            values[prefix + name] = value
        return value

    def _attention(self, x, attn_mask, padding_mask, is_causal):
        self._capture('attention_input', x)
        # Match multi_head_attention_forward's projection and memory layout.
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(1)
        elif self.self_attn.batch_first:
            x = x.transpose(0, 1)
        length, batch, embed_dim = x.shape
        heads = self.self_attn.num_heads
        head_dim = embed_dim // heads
        q, k, v = F._in_projection_packed(
            x, x, x, self.self_attn.in_proj_weight,
            self.self_attn.in_proj_bias)

        def split_heads(value):
            return value.view(length, batch * heads, head_dim).transpose(
                0, 1).view(batch, heads, length, head_dim)

        q, k, v = map(split_heads, (q, k, v))
        self._capture('q_raw', q)
        self._capture('k_raw', k)
        self._capture('v', v)
        if self.normalize_qk:
            q = _normalize_attention_vector(q, self.qk_norm_eps)
            k = _normalize_attention_vector(k, self.qk_norm_eps)
        self._capture('q', q)
        self._capture('k', k)

        if is_causal and attn_mask is None:
            raise RuntimeError('Need attn_mask if specifying the is_causal hint.')
        if is_causal and padding_mask is None:
            attn_mask = None
        else:
            if attn_mask is not None:
                if attn_mask.ndim == 2:
                    if tuple(attn_mask.shape) != (length, length):
                        raise ValueError('Invalid 2D attention mask shape')
                    attn_mask = attn_mask[None, None]
                elif attn_mask.ndim == 3:
                    if tuple(attn_mask.shape) != (batch * heads, length, length):
                        raise ValueError('Invalid 3D attention mask shape')
                    attn_mask = attn_mask.view(batch, heads, length, length)
                else:
                    raise ValueError('Attention mask must have 2 or 3 dimensions')
            if padding_mask is not None:
                if unbatched:
                    padding_mask = padding_mask.unsqueeze(0)
                if tuple(padding_mask.shape) != (batch, length):
                    raise ValueError('Invalid key padding mask shape')
                padding_mask = padding_mask[:, None, None, :]
                attn_mask = (padding_mask if attn_mask is None else
                             attn_mask + padding_mask)
                is_causal = False

        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask,
            self.self_attn.dropout if self.training else 0., is_causal)
        self._capture('attention', output)
        output = output.permute(2, 0, 1, 3).contiguous().view(
            batch * length, embed_dim)
        output = F.linear(output, self.self_attn.out_proj.weight,
                          self.self_attn.out_proj.bias)
        output = output.view(length, batch, embed_dim)
        if unbatched:
            output = output.squeeze(1)
        elif self.self_attn.batch_first:
            output = output.transpose(0, 1)
        self._capture('attention_projected', output)
        return self.dropout1(output)

    def _feedforward(self, x):
        self._capture('ff_input', x)
        x = self._capture('ff_hidden', self.linear1(x))
        x = self._capture('ff_activation', self.activation(x))
        x = self.linear2(self.dropout(x))
        return self._capture('ff_output', self.dropout2(x))

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                is_causal=False):
        if not self.normalize_qk and self._gradient_chain_capture is None:
            return super().forward(src, src_mask, src_key_padding_mask,
                                   is_causal=is_causal)
        # Do not use the fused inference layer: it would bypass normalization.
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask, mask_name='src_key_padding_mask',
            other_type=F._none_or_dtype(src_mask), other_name='src_mask',
            target_type=src.dtype)
        src_mask = F._canonical_mask(
            mask=src_mask, mask_name='src_mask', other_type=None,
            other_name='', target_type=src.dtype, check_other=False)
        x = self._capture('input', src)
        if self.norm_first:
            normalized = self._capture('norm1', self.norm1(x))
            x = self._capture('attention_residual', x + self._attention(
                normalized, src_mask, src_key_padding_mask, is_causal))
            normalized = self._capture('norm2', self.norm2(x))
            x = self._capture('ff_residual', x + self._feedforward(normalized))
        else:
            x = self._capture('attention_residual', x + self._attention(
                x, src_mask, src_key_padding_mask, is_causal))
            x = self._capture('norm1', self.norm1(x))
            x = self._capture('ff_residual', x + self._feedforward(x))
            x = self._capture('norm2', self.norm2(x))
        return self._capture('output', x)


@alf.configurable
class TransformerEncoder(PreprocessorNetwork):
    """A BERT-like transformer encoder.

    The following is the pseudocode for the computation:

    .. code-block:: python

        for i in range(num_prememory_layers):
            core, inputs = T_i([core, inputs], [core, inputs])
        for j in range(num_memory_layers):
            new_core, inputs = TM_j([memory_j, core, inputs], [core, inputs])
            memory_j.write(core)
            core = new_core
        return core, new_memory_state

    where T_i denotes the ``TransformerBlock``  for the i-th prememory layers
    and TM_j denotes the ``TransformerBlock`` for the j-th memory layers. memory_j
    is an ``FIFOMemory`` object (not to be confused with the ``memory`` argument
    of ``TransformerBlock.forward() function``)

    The core embedding serves the same purpose of [CLS] in the BERT model in [1],
    which is to generate a fixed dimensional representation for downstream tasks.
    Different from BERT, which only has one [CLS] embedding, we allow the option
    of having multiple core embeddings. In addition to generating a fixed dimensional
    representation, the core embedding is also used to update the memory.

    [1]. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for
         Language Understanding
    """

    def __init__(self,
                 input_tensor_spec,
                 num_layers,
                 num_attention_heads,
                 d_ff=None,
                 dropout=0.1,
                 batch_first=True,
                 norm_first=False,
                 core_size=1,
                 return_core_only=True,
                 core_embedding_dim=None,
                 input_preprocessors=None,
                 name="TransformerNetwork",
                 final_norm=False,
                 normalize_qk=False,
                 qk_norm_eps=1e-6):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If ``input_tensor_spec`` is not nested, it should
                represent a rank-2 tensor of shape ``[input_size, d_model]``, where
                ``input_size`` is the length of the input sequence, and ``d_model``
                is the dimension of embedding.
            num_prememory_layers (int): number of TransformerBlock calculation
                without using memory
            num_attention_heads (int): number of attention heads for each
                ``TransformerBlock``
            d_ff (int): the size of the hidden layer of the feedforward network
                in each ``TransformerBlock``. If None, ``TransformerBlock`` will
                calculate it as ``4*d_model``.
            core_size (int): size of core (i.e. number of embeddings of core)
            return_core_only (bool): if True, will only return the core embedding
            core_embedding_dim (int): dimension of the output embedding of the core,
                if not None, an extra FC layer is used to project the core embedding.
            final_norm (bool): normalize the stack output before core selection.
                Useful with ``norm_first=True``. Disabled by default to preserve
                existing parameter names and initialization.
            normalize_qk (bool): normalize each attention head's query and key
                to length ``sqrt(head_dim)`` before standard scaled attention.
            qk_norm_eps (float): minimum query/key normalization denominator.
            input_preprocessors (nested Network|nn.Module): a nest of
                stateless preprocessor networks, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with ``input_tensor_spec``. If any element is None, then
                it will be treated as math_ops.identity. This arg is helpful if
                you want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector. The output_spec of each input preprocessor i
                should be [input_size_i, d_model]. The result of all the preprocessors
                will be concatenated as a Tensor of shape ``[batch_size, input_size, d_model]``,
                where ``input_size = sum_i input_size_i``.
        """
        preprocessing_combiner = None
        if input_preprocessors is not None:
            preprocessing_combiner = NestConcat(dim=-2)
        super().__init__(
            input_tensor_spec,
            input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            name=name)

        assert self._processed_input_tensor_spec.ndim == 2
        if not math.isfinite(qk_norm_eps) or qk_norm_eps <= 0:
            raise ValueError('qk_norm_eps must be finite and positive')

        input_length, d_model = self._processed_input_tensor_spec.shape
        if d_ff is None:
            d_ff = 4 * d_model
        self._core_size = core_size
        # self._state_spec = [mem.state_spec for mem in self._memories]
        self._num_layers = num_layers

        # Positional encoding
        self._pos_encoder = PositionalEncoding(d_model, dropout, input_length)

        # Transformer encoder layers
        encoder_layer = _InspectableTransformerEncoderLayer(
            d_model=d_model,
            nhead=num_attention_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
            activation="gelu",
            normalize_qk=normalize_qk,
            qk_norm_eps=qk_norm_eps)

        # Nested inference tensors cannot pass through explicit Q/K projection.
        self._transformer = nn.TransformerEncoder(
            encoder_layer, num_layers,
            norm=nn.LayerNorm(d_model) if final_norm else None,
            enable_nested_tensor=not normalize_qk)
        self._gradient_chain_capture = None
        self._return_core_only = return_core_only
        if return_core_only and core_embedding_dim is not None:
            self._core_fc = layers.FC(core_size * d_model, core_embedding_dim,
                                      use_ln=True)
        else:
            self._core_fc = None

    @contextlib.contextmanager
    def capture_gradient_chain(self):
        """Capture named intermediate tensors from the executed forward graph.

        Yields a dictionary populated by the next forward pass, without
        detaching tensors or changing the SDPA backend. Keys under
        ``layers/{index}/`` include Q/K/V (``[B, heads, tokens, head_dim]``),
        normalization, residual, and feedforward stages. The caller may retain
        this dictionary for vector--Jacobian products after leaving the context.
        References held by the module are removed even if the forward fails.
        """
        if self._gradient_chain_capture is not None:
            raise RuntimeError('Gradient-chain capture is already active')
        values = {}
        self._gradient_chain_capture = values
        for i, layer in enumerate(self._transformer.layers):
            layer._gradient_chain_capture = (values, 'layers/%d/' % i)
        try:
            yield values
        finally:
            self._gradient_chain_capture = None
            for layer in self._transformer.layers:
                layer._gradient_chain_capture = None

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (nested Tensor): consistent with ``input_tensor_spec`` provided
                at ``__init__()``
            state (nested Tensor): states
        Returns:
            - Tensor: shape is [B, core_size * d_model] if ``return_core_only``,
                    and [B, core_size + input_size, d_model] if not ``return_core_only``,
                    where ``input_size`` is the number of embeddings from the
                    (processed) input.
            - nested Tensor: network states.
        """
        z, state = super().forward(inputs, state)
        batch_size = z.shape[0]
        query = self._pos_encoder(z)
        capture = self._gradient_chain_capture
        if capture is None:
            output = self._transformer(query)
        else:
            capture['input'] = z
            capture['positioned_input'] = query
            # Avoid the stack's nested-tensor inference conversion during capture.
            output = query
            for layer in self._transformer.layers:
                output = layer(output)
            capture['stack_output'] = output
            if self._transformer.norm is not None:
                output = self._transformer.norm(output)
                capture['final_norm'] = output

        if self._return_core_only:
            core_embedding = output[:, :self._core_size, :].reshape(
                batch_size, -1)
            if self._core_fc is not None:
                core_embedding = self._core_fc(core_embedding)
            if capture is not None:
                capture['output'] = core_embedding
            return core_embedding, state
        else:
            if capture is not None:
                capture['output'] = output
            return output, state
