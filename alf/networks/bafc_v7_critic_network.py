# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
"""BAFCv7-only functional critic fast paths with the legacy parameter layout."""

import torch
import torch.nn.functional as F

import alf
from alf import layers
from alf.networks.critic_networks import FuncCriticNetwork
from alf.networks.encoding_networks import _ReplicateInputForParallel


@alf.configurable
class BafcV7FuncCriticNetwork(FuncCriticNetwork):
    """Keep ordinary forward/copy semantics; specialize parallel evaluation."""

    def make_parallel(self, n):
        net = super().make_parallel(n)
        if type(net) is not _ReplicateInputForParallel:
            return net
        return BafcV7ParallelCritic(
            self.input_tensor_spec, n, net._pnet, net.name,
            tuple(self.output_spec.shape))


class BafcV7ParallelCritic(_ReplicateInputForParallel):
    """Reuse the original modules, parameters, and state-dict paths.

    Fast paths support the stock stateless vector FC/LayerNorm critic only.
    Unsupported custom networks retain the inherited forward implementation.
    No extra parameter-owning aliases are registered.
    """

    def __init__(self, input_tensor_spec, n, pnet, name, output_shape):
        super().__init__(input_tensor_spec, n, pnet, name)
        self._v7_output_shape = output_shape
        self.supports_v7_fast_paths = self._supported()

    def _supported(self):
        try:
            nets = self._pnet._nets
            obs = nets[0]._nets[1]._pnet._nets
            identity = nets[0]._nets[0]._module
            return (
                self.state_spec == ()
                and all(s.ndim == 1 for s in alf.nest.flatten(self.input_tensor_spec))
                and type(identity) is layers.Identity
                and type(nets[1]) is layers.NestConcat
                and nets[1]._dim == -1 and nets[1]._nest_mask is None
                and type(obs[1]) is layers.NestConcat
                and obs[1]._dim == -1 and obs[1]._nest_mask is None
                and all(len(p._nets) == 0 for p in obs[0]._nets)
                and type(nets[-1]) is layers.Reshape
                and all(type(m) is layers.ParallelFC and m._bn is None
                        and m._activation in (torch.relu_, torch.relu, alf.math.identity)
                        for m in list(obs[2:]) + list(nets[2:-1])))
        except (AttributeError, IndexError):
            return False

    @staticmethod
    def _fc(layer, x, ids):
        if ids is None:
            return layer(x)
        # Gather the critic axis without CPU synchronization or new Parameters.
        weight = layer.weight.index_select(0, ids)
        bias = None if layer.bias is None else layer.bias.index_select(0, ids)
        xt = x.transpose(0, 1)
        if weight.shape[1] == 1:
            y = torch.einsum('kbi,ki->kb', xt, weight[:, 0]).unsqueeze(-1)
            if bias is not None:
                y = y + bias[:, None]
        elif bias is None:
            y = torch.bmm(xt, weight.transpose(1, 2))
        else:
            y = torch.baddbmm(bias[:, None], xt, weight.transpose(1, 2))
        y = y.transpose(0, 1)
        if layer._ln is not None:
            k, width = weight.shape[:2]
            ln = layer._ln
            w = ln.weight.reshape(layer._n, width).index_select(0, ids).flatten()
            b = ln.bias.reshape(layer._n, width).index_select(0, ids).flatten()
            if bias is None:
                b = torch.zeros_like(b)
            y = F.group_norm(y.reshape(-1, k * width), k, w, b, ln.eps)
            y = y.reshape(-1, k, width)
        return layer._activation(y)

    def _encode_observation_action(self, observation, action, ids=None):
        x = torch.cat((observation, action), dim=-1)
        for layer in self._pnet._nets[0]._nets[1]._pnet._nets[2:]:
            x = self._fc(layer, x, ids)
        return x

    def _head(self, encoding, obs_action, ids=None):
        x = torch.cat((encoding, obs_action), dim=-1)
        for layer in self._pnet._nets[2:-1]:
            x = self._fc(layer, x, ids)
        return x

    def actor_critic_product(self, encoding, observation, action,
                             share_observation=False, critic_ids=None):
        """Return [B,A,C,*reward_shape], or [B,A,K,*reward_shape]."""
        b, a = encoding.shape[:2]
        c = self._n if critic_ids is None else critic_ids.numel()
        shared = share_observation and action.ndim == 2
        if shared:
            obs = observation[:, None].expand(-1, c, -1)
            act = action[:, None].expand(-1, c, -1)
            oa = self._encode_observation_action(obs, act, critic_ids)
            oa = oa[:, None].expand(-1, a, -1, -1).reshape(b * a, c, -1)
        else:
            obs = observation[:, None].expand(-1, a, -1).reshape(b * a, -1)
            act = (action[:, None].expand(-1, a, -1)
                   if action.ndim == 2 else action).reshape(b * a, -1)
            oa = self._encode_observation_action(
                obs[:, None].expand(-1, c, -1),
                act[:, None].expand(-1, c, -1), critic_ids)
        enc = encoding.reshape(b * a, -1)[:, None].expand(-1, c, -1)
        return self._head(enc, oa, critic_ids).reshape(
            b, a, c, *self._v7_output_shape)

    def paired_actor_values(self, encoding, observation, action):
        """Evaluate actor i only with critic i; return [B,A,*reward_shape]."""
        b, a = encoding.shape[:2]
        assert a == self._n
        oa = self._encode_observation_action(
            observation[:, None].expand(-1, a, -1), action)
        return self._head(encoding, oa).reshape(b, a, *self._v7_output_shape)

    def selected_target_values(self, encoding, observation, action, critic_ids):
        return self.actor_critic_product(
            encoding, observation, action, critic_ids=critic_ids)
