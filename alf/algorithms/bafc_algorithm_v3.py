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
"""Soft Actor Critic Algorithm."""

from absl import logging
import numpy as np
import functools
import copy
from contextlib import nullcontext
from enum import Enum

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.rlpd_algorithm import TrainMode
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorFCNetwork, FuncCriticNetwork, TransformerEncoder
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, math_ops, checkpoint_utils
from alf.utils import bafcv3_gradient_diagnostics as gradient_diagnostics
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.schedulers import Scheduler
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.networks.neural_graphs.actor_graph import ActorGraph
from alf.networks.neural_graphs.graph_network import GraphNetwork

BafcActionState = namedtuple(
    "BafcActionState", ["actor_network"], default_value=())

BafcCriticState = namedtuple("BafcCriticState", ["critic", "target_critic"])

BafcState = namedtuple(
    "BafcState", ["action", "actor", "critic"],
    default_value=())

BafcCriticInfo = namedtuple(
    "BafcCriticInfo", ["critic", "target_critic"], default_value=())

BafcActorInfo = namedtuple(
    "BafcActorInfo", ["eval_action_loss"], default_value=())

BafcInfo = namedtuple(
    "BafcInfo", [
        "reward", "step_type", "discount", "action", "actor", "critic", 
        "discounted_return", "bootstrap_mask"
    ],
    default_value=())

BafcLossInfo = namedtuple(
    'BafcLossInfo', ('actor', 'critic'), default_value=())


@alf.configurable
class BafcAlgorithmV3(OffPolicyAlgorithm):
    r"""Boostrapped Actor and Functional Critic algorithm, 

    ::

        Bai et al "Bootstrapped Actors and Functional Critic", arXiv, 2025

    V3 implements model-free posterior sampling style exploration scheme over V2.
    In particular, it has multiple functional critics, each paired with an actor.
    Each functional critic is trained with all actors, while each actor is only
    trained with its paired functional critic.

    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorFCNetwork,
                 critic_network_cls=FuncCriticNetwork,
                 reward_weights=None,
                 calculate_priority=False,
                 num_actor_critic=10,
                 actor_critic_pairing=True,
                 use_bootstrap_actors=False,
                 use_bootstrap_critics=False,
                 actor_use_ln=False,
                 bootstrap_mask_prob=0.8,
                 bootstrap_mask_type='episode',
                 num_actor_eval_samples=256,
                 eval_samples_init_method='normal',
                 eval_samples_clipping=False,
                 actor_eval_type='full',
                 actor_encoder_cls=TransformerEncoder,
                 actor_encoding_dim=128,
                 obs_action_encoding_dim=64,
                 actor_utd: Optional[int] = None,
                 critic_utd: Optional[int] = None,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_critic_tau: Union[float, Scheduler] = 0.05,
                 target_critic_period: Union[int, Scheduler] = 1,
                 target_critic_use_ema=False,
                 parameter_reset_period: Union[int, Scheduler] = -1,
                 dqda_clipping=None,
                 checkpoint_replay_buffer=False,
                 track_reweighting_target_observation_cache=False,
                 critic_reweighting_num_target_obs: int = 128,
                 critic_reweighting_target_obs_cache_size:
                 Optional[int] = None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 actor_encoder_optimizer=None,
                 eval_samples_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="BafcAlgorithm",
                 num_sampled_critics_for_actor=1,
                 use_random_critic_targets=False,
                 num_sampled_critic_targets=1,
                 eval_samples_source='trainable',
                 use_actor_id_encoding=False,
                 detach_actor_policy_input=False,
                 use_single_layer_transformer_encoder=False,
                 use_target_actor_encoder=False,
                 use_legacy_actor_gradient=False,
                 debug_gradient_chain=False,
                 debug_gradient_chain_compare_backends=False,
                 dqde_weight=1.0):
        """
        Args:

            use_target_actor_encoder (bool): use a frozen, Polyak-updated encoder
                for TD targets, sharing the critic target updater and clock.
                Requires matching architecture flags when resuming.
            use_legacy_actor_gradient (bool): reproduce the historical connected
                feature surrogates. False injects independent token partials once.
            dqde_weight (float): finite, nonnegative weight for the probe-policy
                gradient contribution, applied after clipping. Scales both hidden
                and output probe contributions, including legacy gradients, while
                leaving the ordinary action gradient unchanged. Defaults to 1.
            debug_gradient_chain (bool): collect gradient-chain and TD diagnostics.
                Expensive measurements use the existing debug-summary cadence.
            debug_gradient_chain_compare_backends (bool): additionally compare
                against an isolated float32 math-SDPA reference at that cadence.
                Training dropout must be zero for a comparable reference;
                otherwise emit a skipped_dropout metric and skip the comparison.
            actor_critic_pairing (bool): whether or not fix the 1-1 pairing of actors 
                and critics during actor_train_step (there are the same number of 
                actors and critics, we pair each actor with a unique and different 
                critic during actor training). If True, such a actor-critic pairing is 
                fixed throughout the training. Otherwise, it is randomized at each
                actor_train_step.
            num_sampled_critics_for_actor (int): the number of distinct critics
                used to update each actor when ``actor_critic_pairing`` is False.
                Their gradients are averaged. It must be 1 when pairing is fixed.
            use_random_critic_targets (bool): whether to use an RLPD-style
                shared TD target sampled from the target critic ensemble. When
                False, critic ``i`` uses target critic ``i`` as before.
            num_sampled_critic_targets (int): the number of target critics
                sampled without replacement when ``use_random_critic_targets``
                is True. Their elementwise minimum is shared by all online
                critics. A value of 1 selects one random target critic.
            eval_samples_source (str): source of the observations used to
                encode actors for the functional critic. ``'trainable'`` keeps
                the existing randomly initialized, trainable evaluation
                samples. ``'frozen'`` keeps those same initialized samples
                fixed throughout training. ``'replay'`` samples without
                replacement from transformed observations in the current
                training iteration.
            use_actor_id_encoding (bool): replace functional policy encodings
                with learned actor-ID embeddings (control A). Embeddings learn
                from critic losses only; the original encoder and evaluation
                samples remain frozen and unused. Resume with the same flag.
            detach_actor_policy_input (bool): stop the policy-encoding gradient
                during actor updates only (control B), retaining the ordinary
                action gradient and unchanged functional critic training. This
                is redundant when ``use_actor_id_encoding`` is True.
            use_single_layer_transformer_encoder (bool): use one transformer
                layer to encode actor tokens, retaining the encoder's positional
                encoding, attention, feedforward network and normalization.
                Overrides ``num_layers`` to 1 when calling ``actor_encoder_cls``;
                other encoder settings and the output dimension are unchanged.
                The constructor must accept ``num_layers``. Cannot be combined
                with ``use_actor_id_encoding``. Resume with the same architecture;
                flattened linear and multilayer checkpoints are not converted.
                False preserves the existing encoder construction exactly.
            bootstrap_mask_type (str): the type of sampling the bootstrap_mask for
                bootstrapped training of actors and/or critics. There are two types, 
                ``episode`` and ``step``. ``episode`` means a same bootstrap_mask for
                every step of an episode. ``step`` means resampled bootstrap_mask for
                every step of an episode.
        """
        if not np.isfinite(dqde_weight) or dqde_weight < 0:
            raise ValueError('dqde_weight must be finite and nonnegative')
        assert not (use_target_actor_encoder and use_actor_id_encoding), (
            "use_target_actor_encoder cannot be combined with use_actor_id_encoding.")
        assert not debug_gradient_chain_compare_backends or debug_gradient_chain, (
            "debug_gradient_chain_compare_backends requires debug_gradient_chain.")
        assert not (use_single_layer_transformer_encoder
                    and use_actor_id_encoding), (
            "use_single_layer_transformer_encoder cannot be combined with "
            "use_actor_id_encoding.")
        assert actor_eval_type in ['full', 'exclude_input', 'last_two', 'output'], (
            r"{actor_eval_type} in not supported.")
        assert eval_samples_init_method in ['normal', 'uniform'], (
            r"init method {eval_samples_init_method} is not supported.")
        assert eval_samples_source in ['trainable', 'frozen', 'replay'], (
            "eval_samples_source must be 'trainable', 'frozen', or 'replay', "
            f"got {eval_samples_source!r}.")
        assert (eval_samples_source == 'trainable'
                or eval_samples_optimizer is None), (
                    "eval_samples_optimizer is only supported when "
                    "eval_samples_source='trainable'.")
        assert not use_actor_id_encoding or eval_samples_optimizer is None, (
            "eval_samples_optimizer is unused with use_actor_id_encoding=True.")
        assert bootstrap_mask_type in ['episode', 'step'], (
            r"bootstrap mask type {bootstrap_mask_type} is not supported.")
        assert 1 <= num_sampled_critics_for_actor <= num_actor_critic, (
            "num_sampled_critics_for_actor must be between 1 and "
            f"num_actor_critic ({num_actor_critic}), got "
            f"{num_sampled_critics_for_actor}.")
        assert not actor_critic_pairing or num_sampled_critics_for_actor == 1, (
            "num_sampled_critics_for_actor must be 1 when "
            "actor_critic_pairing is True.")
        assert 1 <= num_sampled_critic_targets <= num_actor_critic, (
            "num_sampled_critic_targets must be between 1 and "
            f"num_actor_critic ({num_actor_critic}), got "
            f"{num_sampled_critic_targets}.")
        assert critic_reweighting_num_target_obs >= 1, (
            "critic_reweighting_num_target_obs must be >= 1")
        if critic_reweighting_target_obs_cache_size is None:
            critic_reweighting_target_obs_cache_size = (
                4 * critic_reweighting_num_target_obs)
        assert critic_reweighting_target_obs_cache_size >= 1, (
            "critic_reweighting_target_obs_cache_size must be >= 1")
        if actor_utd is None and critic_utd is None:
            self._train_mode = TrainMode.standard
        else:
            total_utd = config.num_updates_per_train_iter
            if actor_utd is None:
                assert critic_utd < total_utd, (
                    "critic_utd should be less than num_updates_per_train_iter "
                    "if actor_utd is not provided.")
                actor_utd = total_utd - critic_utd
            elif critic_utd is None:
                assert actor_utd < total_utd, (
                    "actor_utd should be less than num_updates_per_train_iter "
                    "if critic_utd is not provided.")
                critic_utd = total_utd - actor_utd
            assert actor_utd <= critic_utd, (
                f"actor_utd {actor_utd} should not be greater than critic_utd {critic_utd}"
            )
            self._train_mode = TrainMode.critic
            self._actor_utd = actor_utd
            self._critic_utd = critic_utd

        self._use_actor_id_encoding = use_actor_id_encoding
        self._use_target_actor_encoder = use_target_actor_encoder
        self._use_legacy_actor_gradient = use_legacy_actor_gradient
        self._dqde_weight = dqde_weight
        self._debug_gradient_chain = debug_gradient_chain
        self._debug_gradient_chain_compare_backends = debug_gradient_chain_compare_backends
        self._gradient_chain_pending = None
        self._gradient_chain_step = 0
        self._gradient_chain_intervals = (
            gradient_diagnostics.DiagnosticIntervalAccumulator()
            if debug_gradient_chain else None)
        self._detach_actor_policy_input = detach_actor_policy_input
        self._num_actor_critic = num_actor_critic
        self._actor_critic_pairing = actor_critic_pairing
        self._num_sampled_critics_for_actor = num_sampled_critics_for_actor
        self._use_random_critic_targets = use_random_critic_targets
        self._num_sampled_critic_targets = num_sampled_critic_targets
        self._eval_samples_source = eval_samples_source
        self._num_actor_eval_samples = num_actor_eval_samples
        # An ephemeral, iteration-local cache. Keeping this as a plain
        # attribute deliberately excludes it from parameters, buffers and
        # checkpoints.
        self._replay_eval_observation_pool = None
        self._use_bootstrap_actors = use_bootstrap_actors
        self._use_bootstrap_critics = use_bootstrap_critics
        self._bootstrap_mask_prob = bootstrap_mask_prob
        self._bootstrap_mask_type = bootstrap_mask_type
        self._checkpoint_replay_buffer = checkpoint_replay_buffer
        self._track_reweighting_target_observation_cache = (
            track_reweighting_target_observation_cache)
        self._critic_reweighting_target_obs_cache_size = (
            critic_reweighting_target_obs_cache_size)
        self._reweighting_target_observation_cache = ()
        self._bootstrap_mask = ()
        actor_networks = actor_network_cls(
            input_tensor_spec=observation_spec,
            action_spec=action_spec,
            n_groups=num_actor_critic)
        if eval_samples_init_method == 'normal':
            actor_eval_samples = 2 * torch.randn(
                num_actor_eval_samples, observation_spec.shape[0])
            if eval_samples_clipping:
                actor_eval_samples.clip_(min=-1.0, max=1.0)
        else:
            actor_eval_samples = 2 * torch.rand(
                num_actor_eval_samples, observation_spec.shape[0]) - 1

        # extract actor token length from actor_encoder 
        if actor_eval_type == 'full':
            actor_token_length = observation_spec.shape[0] + sum( 
                t.shape[1] for t in actor_networks.bias_params)
        elif actor_eval_type == 'exclude_input':
            actor_token_length = sum( 
                t.shape[1] for t in actor_networks.bias_params)
        elif actor_eval_type == 'last_two':
            actor_token_length = sum( 
                t.shape[1] for t in actor_networks.bias_params[-2:])
        else:
            actor_token_length = action_spec.shape[0]

        actor_token_spec = TensorSpec(
            shape=(actor_token_length, num_actor_eval_samples))
        if use_single_layer_transformer_encoder:
            actor_encoder = actor_encoder_cls(
                actor_token_spec, core_embedding_dim=actor_encoding_dim,
                num_layers=1)
        else:
            actor_encoder = actor_encoder_cls(
                actor_token_spec, core_embedding_dim=actor_encoding_dim)

        # functional critic
        if actor_encoding_dim is None:
            actor_encoding_dim = num_actor_eval_samples
        actor_spec = TensorSpec(shape=(actor_encoding_dim,))
        obs_action_spec = (observation_spec, action_spec)
        critic_network = critic_network_cls(
            input_tensor_spec=(actor_spec, obs_action_spec), 
            obs_action_encoding_dim=obs_action_encoding_dim,
            actor_obs_action_combiner=alf.layers.NestConcat(dim=-1))
        critic_networks = critic_network.make_parallel(num_actor_critic)

        action_state_spec = BafcActionState(
            actor_network=actor_networks.state_spec)
        train_state_spec = BafcState(
            action=action_state_spec,
            actor=critic_network.state_spec,
            critic=BafcCriticState(
                critic=critic_networks.state_spec,
                target_critic=critic_networks.state_spec))

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=train_state_spec,
            predict_state_spec=action_state_spec,
            reward_weights=reward_weights,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None and actor_networks is not None:
            self.add_optimizer(actor_optimizer, [actor_networks])
        if critic_optimizer is not None and critic_networks is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if actor_encoder_optimizer is not None and not use_actor_id_encoding:
            self.add_optimizer(actor_encoder_optimizer, [actor_encoder])
        # Keep this parameter in replay mode so that initialization RNG,
        # checkpoint keys, and optimizer parameter layouts remain compatible
        # with existing BAFCv3 runs. It is frozen and unused in replay mode.
        self._actor_eval_samples = nn.Parameter(
            actor_eval_samples,
            requires_grad=(eval_samples_source == 'trainable'
                           and not use_actor_id_encoding))
        if eval_samples_optimizer is not None:
            self.add_optimizer(eval_samples_optimizer, [self._actor_eval_samples])

        self._actor_networks = actor_networks
        self._actor_use_ln = actor_use_ln
        self._actor_eval_type = actor_eval_type
        self._actor_encoder = actor_encoder
        self._critic_networks = critic_networks
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')
        if use_target_actor_encoder:
            # Network.copy() reinitializes and consumes RNG; this is a weight copy.
            self._target_actor_encoder = copy.deepcopy(actor_encoder)
            self._target_actor_encoder.requires_grad_(False)
        # self._target_critic_network.set_obs_action_batch_dominate(True)

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(critic_loss_ctor,
                                             debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_actor_critic):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._rollout_actor_id = 0
        self._actor_update_counter = 0
        self._critic_update_counter = 0
        self._dqda_clipping = dqda_clipping
        self._training_started = False
        self._do_critic_summary = False

        def _filter(x):
            return list(filter(lambda x: x is not None, x))

        def _create_target_updater(model_list, target_model_list,
                                   tau, period, use_ema):
            return common.TargetUpdater(
                models=_filter(model_list),
                target_models=_filter(target_model_list),
                tau=tau,
                period=period,
                delayed_update=use_ema)

        target_sources = [self._critic_networks]
        target_destinations = [self._target_critic_networks]
        if use_target_actor_encoder:
            target_sources.append(self._actor_encoder)
            target_destinations.append(self._target_actor_encoder)
        self._update_target_critic = _create_target_updater(
            target_sources, target_destinations,
            target_critic_tau, target_critic_period, target_critic_use_ema)

        if use_actor_id_encoding:
            # Create this only after the existing networks (including targets)
            # so shared network initialization and the default checkpoint keys
            # are unchanged. Keep the unused encoder for checkpoint inspection.
            self._actor_encoder.requires_grad_(False)
            self._actor_id_embedding = nn.Embedding(num_actor_critic,
                                                   actor_encoding_dim)
            if actor_encoder_optimizer is not None:
                self.add_optimizer(actor_encoder_optimizer,
                                   [self._actor_id_embedding])

    @property
    def has_dynamic_train_info(self):
        # Inactive actor/critic branches return empty leaves. This also applies
        # when the first update after a checkpoint is a critic-only update.
        return True

    def _bafc_runtime_key(self, prefix, name):
        return prefix + "_bafc_runtime." + name

    def _bafc_scalar_tensor(self, value, dtype=torch.int64):
        if isinstance(value, torch.Tensor):
            return value.detach().reshape(()).to(dtype=dtype).clone()
        return torch.tensor(value, dtype=dtype)

    def _bafc_scalar_int(self, value):
        return int(torch.as_tensor(value).reshape(()).item())

    def _bafc_runtime_tensor(self, value):
        return torch.as_tensor(value).detach().clone()

    def _save_bafc_runtime_state(self, destination, prefix):
        updater = self._update_target_critic
        destination[self._bafc_runtime_key(
            prefix, "target_updater_counter")] = self._bafc_scalar_tensor(
                updater._counter)
        if updater._delayed_update:
            # TargetUpdater keeps these modules in an ordinary list, so they
            # are not traversed by the normal module state_dict machinery.
            destination[self._bafc_runtime_key(
                prefix, "target_updater_recent_models")] = [
                    {key: value.detach().clone()
                     for key, value in model.state_dict().items()}
                    for model in updater._recent_models
                ]

        destination[self._bafc_runtime_key(
            prefix, "training_started")] = self._bafc_scalar_tensor(
                self._training_started, dtype=torch.bool)
        destination[self._bafc_runtime_key(
            prefix, "train_mode")] = self._bafc_scalar_tensor(
                self._train_mode.value)
        destination[self._bafc_runtime_key(
            prefix, "rollout_actor_id")] = self._bafc_scalar_tensor(
                self._rollout_actor_id)
        destination[self._bafc_runtime_key(
            prefix, "actor_update_counter")] = self._bafc_scalar_tensor(
                self._actor_update_counter)
        destination[self._bafc_runtime_key(
            prefix, "critic_update_counter")] = self._bafc_scalar_tensor(
                self._critic_update_counter)
        if isinstance(self._reweighting_target_observation_cache, torch.Tensor):
            destination[self._bafc_runtime_key(
                prefix, "reweighting_target_observation_cache")] = (
                    self._reweighting_target_observation_cache.detach().clone())

    def _pop_bafc_runtime_state(self, state_dict, prefix):
        runtime_prefix = self._bafc_runtime_key(prefix, "")
        runtime_state = {}
        for key in list(state_dict.keys()):
            if key.startswith(runtime_prefix):
                runtime_state[key[len(runtime_prefix):]] = state_dict.pop(key)
        return runtime_state

    def _has_legacy_actor_checkpoint(self, state_dict, prefix):
        return any(key.startswith(prefix + "_actor_networks.")
                   for key in state_dict.keys())

    def _restore_bafc_runtime_state(self, runtime_state):
        updater = self._update_target_critic
        updater._counter = self._bafc_scalar_int(
            runtime_state.get("target_updater_counter", 0))
        if "target_updater_counter" not in runtime_state and updater._period() > 1:
            logging.warning(
                "BAFCv3 checkpoint has no target-updater counter; resetting "
                "only the target-update phase to zero. Training progress and "
                "actor/critic counters are restored independently.")
        if updater._delayed_update:
            recent = runtime_state.get("target_updater_recent_models")
            if recent is None:
                raise RuntimeError(
                    "BAFCv3 checkpoint lacks intermediate target-updater models "
                    "required by target_critic_use_ema=True. Resume with the "
                    "original non-delayed configuration if applicable, or use "
                    "a checkpoint containing delayed target-updater state.")
            if len(recent) != len(updater._recent_models):
                raise RuntimeError("BAFCv3 target-updater model count mismatch")
            for model, state in zip(updater._recent_models, recent):
                model.load_state_dict(state, strict=True)

        if "training_started" in runtime_state:
            self._training_started = bool(
                torch.as_tensor(
                    runtime_state["training_started"]).reshape(()).item())
        if "train_mode" in runtime_state:
            self._train_mode = TrainMode(
                self._bafc_scalar_int(runtime_state["train_mode"]))
        if "rollout_actor_id" in runtime_state:
            self._rollout_actor_id = self._bafc_scalar_int(
                runtime_state["rollout_actor_id"])
        if "actor_update_counter" in runtime_state:
            self._actor_update_counter = self._bafc_scalar_int(
                runtime_state["actor_update_counter"])
        if "critic_update_counter" in runtime_state:
            self._critic_update_counter = self._bafc_scalar_int(
                runtime_state["critic_update_counter"])
        # Diagnostic caches are ephemeral, but update IDs should stay monotonic
        # across an exact resume using the already-checkpointed training counters.
        self._gradient_chain_step = (self._actor_update_counter
                                     + self._critic_update_counter)
        self._gradient_chain_pending = None
        if self._debug_gradient_chain:
            self._gradient_chain_intervals = gradient_diagnostics.DiagnosticIntervalAccumulator()
        if "reweighting_target_observation_cache" in runtime_state:
            self._reweighting_target_observation_cache = (
                self._bafc_runtime_tensor(
                    runtime_state["reweighting_target_observation_cache"]))
        self._apply_train_mode_grad_flags()

    def _apply_train_mode_grad_flags(self):
        standard_or_initial = (
            self._train_mode == TrainMode.standard or
            (self._actor_update_counter == 0
             and self._critic_update_counter == 0))
        actor_requires_grad = (standard_or_initial
                               or self._train_mode == TrainMode.actor)
        eval_samples_requires_grad = (
            self._eval_samples_source == 'trainable'
            and not self._use_actor_id_encoding
            and (standard_or_initial or self._train_mode == TrainMode.critic))
        for p in self._actor_networks.parameters():
            p.requires_grad_(actor_requires_grad)
        self._actor_eval_samples.requires_grad_(eval_samples_requires_grad)

    def checkpoint_replay_buffer_enabled(self):
        return self._checkpoint_replay_buffer

    def _set_replay_buffer_checkpoint_enabled(self, enabled):
        if not self._checkpoint_replay_buffer or self._replay_buffer is None:
            return None
        old_enabled = checkpoint_utils.is_checkpoint_enabled(self._replay_buffer)
        checkpoint_utils.enable_checkpoint(self._replay_buffer, enabled)

        def _restore():
            checkpoint_utils.enable_checkpoint(self._replay_buffer, old_enabled)

        return _restore

    def _has_replay_buffer_checkpoint(self, state_dict):
        return any(key.startswith("_replay_buffer.")
                   or "._replay_buffer." in key for key in state_dict.keys())

    def _alf_prepare_checkpoint_save(self):
        return self._set_replay_buffer_checkpoint_enabled(True)

    def _alf_prepare_checkpoint_load(self, state_dict):
        return self._set_replay_buffer_checkpoint_enabled(
            self._has_replay_buffer_checkpoint(state_dict))

    def _save_to_state_dict(self, destination, prefix, visited=None):
        super()._save_to_state_dict(destination, prefix, visited)
        self._save_bafc_runtime_state(destination, prefix)

    def _load_from_state_dict(self,
                              state_dict,
                              prefix,
                              local_metadata,
                              strict,
                              missing_keys,
                              unexpected_keys,
                              error_msgs,
                              visited=None):
        self._replay_eval_observation_pool = None
        runtime_state = self._pop_bafc_runtime_state(state_dict, prefix)
        legacy_actor_checkpoint = (
            not runtime_state and self._has_legacy_actor_checkpoint(
                state_dict, prefix))
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs, visited)
        if runtime_state:
            self._restore_bafc_runtime_state(runtime_state)
        elif legacy_actor_checkpoint:
            self._restore_bafc_runtime_state({})
            self._training_started = True
            self._apply_train_mode_grad_flags()

    def _flatten_reweighting_observations(self, observation):
        if not isinstance(observation, torch.Tensor):
            return ()
        obs_dim = len(self._observation_spec.shape)
        obs = observation.reshape(-1, *observation.shape[-obs_dim:])
        if obs.shape[0] == 0:
            return ()
        return obs

    def _append_reweighting_target_observations(self, observation):
        obs = self._flatten_reweighting_observations(observation)
        if not isinstance(obs, torch.Tensor):
            return
        obs = obs.detach()
        cache = self._reweighting_target_observation_cache
        if isinstance(cache, torch.Tensor):
            if cache.device != obs.device:
                cache = cache.to(obs.device)
            obs = torch.cat([cache, obs], dim=0)
        if obs.shape[0] > self._critic_reweighting_target_obs_cache_size:
            obs = obs[-self._critic_reweighting_target_obs_cache_size:]
        self._reweighting_target_observation_cache = obs

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        """Prepare replay evaluation observations from the training batch."""
        if self._use_actor_id_encoding or self._eval_samples_source != 'replay':
            return root_inputs, rollout_info

        # Invalidate first so a validation failure cannot leave a stale pool
        # available to a later train_step().
        self._replay_eval_observation_pool = None
        observation = root_inputs.observation
        observation_shape = tuple(self._observation_spec.shape)
        if (not isinstance(observation, torch.Tensor)
                or observation.ndim != 2 + len(observation_shape)
                or tuple(observation.shape[2:]) != observation_shape):
            actual_shape = (tuple(observation.shape)
                            if isinstance(observation, torch.Tensor) else
                            type(observation).__name__)
            raise RuntimeError(
                "eval_samples_source='replay' requires a tensor observation "
                "with shape [batch, time, *observation_shape], where "
                f"observation_shape={observation_shape}; got {actual_shape}.")

        pool = observation.reshape(-1, *observation_shape).detach()
        actual_size = pool.shape[0]
        required_size = self._num_actor_eval_samples
        if actual_size < required_size:
            raise RuntimeError(
                "The current transformed replay observation pool is too "
                f"small: actual size {actual_size}, required size "
                f"{required_size}. Increase the replay training batch size "
                "or reduce num_actor_eval_samples.")
        self._replay_eval_observation_pool = pool
        return root_inputs, rollout_info

    def _get_actor_eval_samples(self):
        if self._eval_samples_source == 'replay':
            pool = self._replay_eval_observation_pool
            if pool is None:
                raise RuntimeError(
                    "eval_samples_source='replay' requires a valid "
                    "preprocess_experience() call before train_step().")
            indices = torch.randperm(
                pool.shape[0], device=pool.device)[:self._num_actor_eval_samples]
            return torch.index_select(pool, 0, indices)
        if self._eval_samples_source == 'frozen':
            # Keep the checkpointed parameter but never connect it to the
            # training graph, including when training mode changes.
            return self._actor_eval_samples.detach()
        return self._actor_eval_samples

    def _predict_action(self,
                        actor_net,
                        observation,
                        state: BafcActionState,
                        train=False):
        if not self._training_started:
            # get batch size with ``get_outer_rank`` and ``get_nest_shape``
            # since the observation can be a nest in the general case
            outer_rank = nest_utils.get_outer_rank(observation,
                                                   self._observation_spec)
            outer_dims = alf.nest.get_nest_shape(observation)[:outer_rank]
            # This uniform sampling seems important because for a squashed Gaussian,
            # even with a large scale, a random policy is not nearly uniform.
            action = alf.nest.map_structure(
                lambda spec: spec.sample(outer_dims=outer_dims),
                self._action_spec)
            return action, state

        if train:
            action, state = actor_net(
                observation, state=state.actor_network)
        else:
            if self._actor_use_ln:
                action, state = actor_net(
                    observation, state=state.actor_network)
                # [n_env, n_actor, d_a] --> [n_env, d_a]
                action = action[:, self._rollout_actor_id, :]
            else:
                action, state = actor_net(
                    observation,
                    id=self._rollout_actor_id,
                    state=state.actor_network)
        new_state = BafcActionState(actor_network=state)

        return action, new_state

    def predict_step(self, inputs: TimeStep, state: BafcActionState):
        action, action_state = self._predict_action(
            self._actor_networks,
            inputs.observation,
            state=state)
        return AlgStep(
            output=action,
            state=action_state,
            info=BafcInfo(action=action))

    def rollout_step(self, inputs: TimeStep, state: BafcState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        assert not self._is_eval
        if self._track_reweighting_target_observation_cache:
            self._append_reweighting_target_observations(inputs.observation)
        if inputs.step_type == StepType.FIRST or self._bootstrap_mask_type == 'step':
            if inputs.step_type == StepType.FIRST:
                # commitment: only resample rollout actor at the beginning of an episode
                self._rollout_actor_id = torch.randint(self._num_actor_critic, ())
            if self._use_bootstrap_actors or self._use_bootstrap_critics:
                # [n_env, n_actors] masks for bootstrap actors
                prob_t = torch.full(
                    (inputs.step_type.shape[0], self._num_actor_critic),
                    self._bootstrap_mask_prob)
                self._bootstrap_mask = torch.bernoulli(prob_t)

        action, action_state = self._predict_action(
            self._actor_networks,
            inputs.observation,
            state=state.action)
        return AlgStep(
            output=action,
            state=state._replace(action=action_state),
            info=BafcInfo(action=action, bootstrap_mask=self._bootstrap_mask))

    def _tokenize_actor_out(self, eval_out):
        # To make actor eval_out an input sequence to the transformer, we set
        # n_actor as the batch_size, \sum_d as the length of the sequence, 
        # and num_eval_samples B as the dimension of embedding
        if self._actor_eval_type == 'output':
            # [bs, n_actor, d_a] -> [n_actor, d_a, bs]
            eval_out_seq = eval_out.permute(1, 2, 0)
        else:
            # list of [bs, n_actor, di] --> [n_actor, \sum_di, bs]
            eval_out_seq = torch.cat(eval_out, dim=-1).permute(1, 2, 0)

        return eval_out_seq

    def _sample_actor_critic_matchings(self, device=None):
        """Sample balanced mappings from critic slots to actor identities."""
        n = self._num_actor_critic
        if self._actor_critic_pairing:
            matching = torch.arange(n).unsqueeze(0)
        else:
            actor_permutation = torch.randperm(n)
            if self._num_sampled_critics_for_actor == 1:
                # Keep the random-number usage of the existing K=1 path.
                matching = actor_permutation.unsqueeze(0)
            else:
                offsets = torch.randperm(n)[:self._num_sampled_critics_for_actor]
                critic_slots = torch.arange(n).unsqueeze(0)
                matching = actor_permutation[
                    (critic_slots + offsets.unsqueeze(1)) % n]
        if device is not None:
            matching = matching.to(device=device)
        return matching

    def _restore_actor_order(self, matched_value, matched_actor_ids):
        """Reorder the critic-slot dimension of every matching by actor id."""
        trailing_shape = matched_value.shape[3:]
        index = matched_actor_ids[:, None, :, *([None] * len(trailing_shape))]
        index = index.expand_as(matched_value)
        return torch.zeros_like(matched_value).scatter(2, index, matched_value)

    def _aggregate_matched_action_gradients(self, matched_dqda,
                                            matched_actor_ids, action):
        """Scatter matched critic gradients back into actor identity order."""
        del action
        return self._restore_actor_order(matched_dqda,
                                         matched_actor_ids).sum(dim=0)

    @staticmethod
    def _mean_pairwise_cosine(individual_dqda):
        """Mean off-diagonal cosine without constructing a K by K Gram matrix."""
        k = individual_dqda.shape[0]
        flat_grad = individual_dqda.double().reshape(*individual_dqda.shape[:3], -1)
        norm = flat_grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit_grad = flat_grad / norm
        summed_sq_norm = unit_grad.sum(dim=0).square().sum(dim=-1)
        self_sq_norm = unit_grad.square().sum(dim=-1).sum(dim=0)
        return (summed_sq_norm - self_sq_norm) / (k * (k - 1))

    def _summarize_actor_gradients(self, dqda, clipped_dqda, dqde,
                                   clipped_dqde, current_action, replay_action,
                                   q_value, individual_dqda):
        def finite_summary(name, value):
            value = value.detach()
            finite = torch.isfinite(value)
            alf.summary.scalar(name + '/nonfinite_count', (~finite).sum())
            safe_mean_hist_summary(name, value[finite])

        dqda = dqda.detach()
        finite_summary('actor_gradients/dqda', dqda)
        finite_summary('actor_gradients/dqda_abs', dqda.abs())
        finite_summary('actor_gradients/dqda_l2_norm',
                               dqda.double().flatten(start_dim=2).norm(dim=-1))
        for i in range(dqda.shape[-1]):
            alf.summary.scalar(
                f'actor_gradients/dqda_abs_component_{i}',
                dqda[..., i].abs().mean())

        if self._dqda_clipping:
            clipped_dqda = clipped_dqda.detach()
            finite_summary('actor_gradients/clipped_dqda', clipped_dqda)
            finite_summary('actor_gradients/clipped_dqda_abs',
                                   clipped_dqda.abs())
            finite_summary(
                'actor_gradients/clipped_dqda_l2_norm',
                clipped_dqda.double().flatten(start_dim=2).norm(dim=-1))
            for i in range(clipped_dqda.shape[-1]):
                alf.summary.scalar(
                    f'actor_gradients/clipped_dqda_abs_component_{i}',
                    clipped_dqda[..., i].abs().mean())
            alf.summary.scalar(
                'actor_gradients/dqda_clip_fraction',
                dqda.abs().gt(self._dqda_clipping).to(torch.float32).mean())

        for i, raw in enumerate(nest.flatten(dqde)):
            raw = raw.detach()
            finite_summary(f'actor_gradients/dqde_leaf_{i}_abs',
                                   raw.abs())
            finite_summary(
                f'actor_gradients/dqde_leaf_{i}_l2_norm',
                raw.double().flatten(start_dim=max(0, raw.ndim - 1)).norm(dim=-1))
            if self._dqda_clipping and clipped_dqde is not None:
                clipped = nest.flatten(clipped_dqde)[i].detach()
                finite_summary(
                    f'actor_gradients/clipped_dqde_leaf_{i}_abs',
                    clipped.abs())
                finite_summary(
                    f'actor_gradients/clipped_dqde_leaf_{i}_l2_norm',
                    clipped.double().flatten(start_dim=max(0, clipped.ndim - 1)).norm(
                        dim=-1))

        current_action = current_action.detach()
        replay_action = replay_action.detach().unsqueeze(1)
        finite_summary(
            'actor_actions/current_vs_replay_l2',
            (current_action - replay_action).flatten(start_dim=2).norm(dim=-1))
        finite_summary('actor_actions/current_abs',
                               current_action.abs())
        alf.summary.scalar(
            'actor_actions/current_fraction_abs_gt_0_95',
            current_action.abs().gt(0.95).to(torch.float32).mean())

        if individual_dqda is not None:
            q_value = q_value.detach()
            individual_dqda = individual_dqda.detach()
            finite_summary('actor_critic_aggregation/q_mean',
                                   q_value.mean(dim=0))
            finite_summary(
                'actor_critic_aggregation/q_std',
                q_value.std(dim=0, unbiased=False))
            finite_summary(
                'actor_critic_aggregation/individual_dqda_l2_norm',
                individual_dqda.double().flatten(start_dim=3).norm(dim=-1))
            finite_summary(
                'actor_critic_aggregation/dqda_pairwise_cosine',
                self._mean_pairwise_cosine(individual_dqda))

    def _encode_actor_policies(self, actor_eval_samples=None, return_tokens=False,
                               encoder_rng_state=None):
        """Return encodings in stable actor order and functional eval outputs."""
        if self._use_actor_id_encoding:
            result = (self._actor_id_embedding.weight, ())
            return (*result, ()) if return_tokens else result
        if actor_eval_samples is None:
            actor_eval_samples = self._get_actor_eval_samples()
        elif self._eval_samples_source == 'frozen':
            actor_eval_samples = actor_eval_samples.detach()
        eval_action = self._actor_networks(
            actor_eval_samples,
            full_neurons=self._actor_eval_type != 'output')[0]
        if self._actor_eval_type == 'exclude_input':
            eval_action = eval_action[1:]
        elif self._actor_eval_type == 'last_two':
            eval_action = eval_action[-2:]

        actor_tokens = self._tokenize_actor_out(eval_action)
        if encoder_rng_state is not None:
            encoder_rng_state['cpu'] = torch.get_rng_state()
            if actor_tokens.is_cuda:
                encoder_rng_state['cuda'] = torch.cuda.get_rng_state(actor_tokens.device)
        actor_encoding = self._actor_encoder(actor_tokens)[0]

        result = (actor_encoding, eval_action)
        return (*result, actor_tokens) if return_tokens else result

    @staticmethod
    def _linear_gradient_surrogate(gradient, value):
        """Zero-valued surrogate with derivative -gradient, without squaring it."""
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        value = value.to(dtype)
        return -gradient.detach().to(dtype) * (value - value.detach())

    def _actor_train_step(self,
                          observation,
                          action,
                          replay_action,
                          mask,
                          state,
                          actor_eval_samples=None):
        """Compute the exact off-policy policy gradient from the functional critic,
        which consists of two terms, 

        1. the gradient w.r.t. input action, as in standard actor-critic algorithms.

        2. the gradient w.r.t. eval action, i.e., actor_networks' outputs for 
           self._actor_eval_samples.
        """
        ## Step 1: encode all actors from actor_eval_samples
        ####################################################
        record_debug = (self._debug_summaries
                        and alf.summary.should_record_summaries())
        record_chain = self._debug_gradient_chain and record_debug
        capture_context = (self._actor_encoder.capture_gradient_chain()
                           if record_chain and not self._use_actor_id_encoding
                           and hasattr(self._actor_encoder, 'capture_gradient_chain')
                           else nullcontext({}))
        reference_rng = ({} if record_chain and
                         self._debug_gradient_chain_compare_backends else None)
        with capture_context as capture:
            actor_encoding, eval_action, actor_tokens = self._encode_actor_policies(
                actor_eval_samples, return_tokens=True, encoder_rng_state=reference_rng)
        action_only = (self._use_actor_id_encoding
                       or self._detach_actor_policy_input)
        if action_only:
            actor_encoding = actor_encoding.detach()

        k = self._num_sampled_critics_for_actor
        batch_size = observation.shape[0]
        matched_actor_ids = self._sample_actor_critic_matchings(action.device)
        matched_actor_encoding = actor_encoding[matched_actor_ids]
        matched_action = torch.gather(
            action.unsqueeze(0).expand(k, *action.shape),
            dim=2,
            index=matched_actor_ids[:, None, :, None].expand(
                k, batch_size, self._num_actor_critic, action.shape[-1]))

        critic_actor_encoding = matched_actor_encoding[:, None, :, :].expand(
            k, batch_size, self._num_actor_critic,
            matched_actor_encoding.shape[-1]).reshape(
                k * batch_size, self._num_actor_critic,
                matched_actor_encoding.shape[-1])

        ## Step 2: compute critic values for all actors
        ###############################################
        # # [T*B * n_actor, d_s]
        # critic_observation = observation.repeat_interleave(
        #     self._num_actor_critic, dim=0)
        # [K * T*B, n_critic, d_s]
        critic_observation = observation.unsqueeze(0).expand(
            k, *observation.shape).reshape(k * batch_size,
                                           *observation.shape[1:])
        critic_observation = critic_observation.unsqueeze(1).expand(
            k * batch_size, self._num_actor_critic,
            *observation.shape[1:])
        critic_action = matched_action.reshape(
            k * batch_size, self._num_actor_critic, action.shape[-1])
        q_value, critic_state = self._critic_networks(
            (critic_actor_encoding, (critic_observation, critic_action)), state)
        q_value = q_value.reshape(k, batch_size, *q_value.shape[1:])

        ## Step 3: exact off-policy policy gradient (OPG)
        #################################################
        # need to exclude the input actor_eval_samples, since they don't requires_grad
        # for actor TrainMode
        if action_only:
            eval_action_in_graph = ()
        elif self._actor_eval_type == 'full':
            eval_action_in_graph = eval_action[1:]
        else:
            eval_action_in_graph = eval_action

        need_individual_dqda = record_debug and k > 1
        dqda_input = matched_action if need_individual_dqda else action
        # Connected-feature derivatives are useful measurements, but reinjecting
        # them at both a hidden feature and its descendant counts a path twice.
        measure_connected = (self._use_legacy_actor_gradient or record_debug
                             or self._debug_gradient_chain)
        leaves = nest.flatten(eval_action_in_graph) if not action_only else []
        inputs = [dqda_input]
        if not action_only and not self._use_legacy_actor_gradient:
            inputs.append(actor_tokens)
        if measure_connected:
            inputs.extend(x for x in leaves if x.requires_grad)
        if record_chain and not action_only:
            inputs.extend([actor_encoding, actor_tokens])
        inputs = list({id(x): x for x in inputs}.values())
        stage_gradients = {}
        handles = []
        if record_chain:
            for name, tensor in capture.items():
                if tensor.requires_grad:
                    def save_gradient(gradient, name=name):
                        stage_gradients[name] = gradient.detach()
                    handles.append(tensor.register_hook(save_gradient))
        try:
            values = torch.autograd.grad(q_value.sum() / k, inputs,
                                         retain_graph=True, allow_unused=True)
        finally:
            for handle in handles:
                handle.remove()
        gradients = {id(x): (g if g is not None else torch.zeros_like(x))
                     for x, g in zip(inputs, values)}
        dqda = gradients[id(dqda_input)]
        dqde = [gradients.get(id(x), torch.zeros_like(x)) for x in leaves]
        token_gradient = (gradients.get(id(actor_tokens)) if not action_only
                          else None)

        individual_dqda = None
        if need_individual_dqda:
            # Restore mean selected-critic gradients to stable actor order.
            individual_dqda = self._restore_actor_order(
                dqda * k, matched_actor_ids)
            dqda = individual_dqda.mean(dim=0)
            q_value = self._restore_actor_order(q_value, matched_actor_ids)

        def clip(x):
            return (x.clamp(-self._dqda_clipping, self._dqda_clipping)
                    if self._dqda_clipping else x)

        clipped_dqda = clip(dqda)
        clipped_dqde = [clip(g) for g in dqde] if self._use_legacy_actor_gradient else None
        if record_debug:
            self._summarize_actor_gradients(
                dqda, clipped_dqda, dqde, clipped_dqde, action, replay_action,
                q_value, individual_dqda)
            alf.summary.scalar('actor_gradients/critic_objective',
                               q_value.detach().mean())

        def action_loss_fn(gradient, a_in):
            if self._use_legacy_actor_gradient:
                loss = 0.5 * losses.element_wise_squared_loss(
                    (gradient + a_in).detach(), a_in)
            else:
                loss = self._linear_gradient_surrogate(gradient, a_in)
            return loss.sum(list(range(2, loss.ndim)))

        action_loss = action_loss_fn(clipped_dqda, action)
        if self._use_bootstrap_actors:
            action_loss = action_loss * mask / self._bootstrap_mask_prob
        action_loss = action_loss.sum(-1)

        probe_hidden_loss = probe_output_loss = None
        if action_only:
            eval_action_loss = torch.zeros_like(action_loss)
        elif self._use_legacy_actor_gradient:
            leaf_losses = [action_loss_fn(g, x).mean()
                           for g, x in zip(clipped_dqde, leaves)]
            eval_action_loss = sum(leaf_losses).repeat(action_loss.shape[0])
            probe_hidden_loss = sum(leaf_losses[:-1]) if len(leaf_losses) > 1 else None
            probe_output_loss = leaf_losses[-1]
        else:
            clipped_tokens = clip(token_gradient)
            if self._actor_eval_type == 'full':
                # Preserve the old exclusion of the direct raw-probe-input path,
                # including the initial joint update where probes are trainable.
                input_width = eval_action[0].shape[-1]
                clipped_tokens = torch.cat((
                    torch.zeros_like(clipped_tokens[:, :input_width]),
                    clipped_tokens[:, input_width:]), dim=1)
            token_loss = self._linear_gradient_surrogate(
                clipped_tokens, actor_tokens)
            scale = actor_tokens.shape[0] * actor_tokens.shape[-1]
            action_width = action.shape[-1]
            probe_hidden_loss = token_loss[:, :-action_width].sum() / scale
            probe_output_loss = token_loss[:, -action_width:].sum() / scale
            eval_action_loss = (probe_hidden_loss + probe_output_loss).repeat(
                action_loss.shape[0])
            if record_chain:
                gradient_diagnostics.summarize_tensor(
                    'gradient_chain/rank_local/token_applied',
                    clipped_tokens * self._dqde_weight)

        # Weight the full probe contribution after clipping and reduction. Keep
        # the diagnostic branch losses consistent with the optimizer objective.
        eval_action_loss = eval_action_loss * self._dqde_weight
        if probe_hidden_loss is not None:
            probe_hidden_loss = probe_hidden_loss * self._dqde_weight
        if probe_output_loss is not None:
            probe_output_loss = probe_output_loss * self._dqde_weight

        if self._debug_gradient_chain:
            for i, gradient in enumerate(dqde):
                self._gradient_chain_intervals.record(
                    'dqde/leaf_%d' % i, gradient, self._gradient_chain_step)
        if record_chain:
            self._record_actor_gradient_chain(
                dqda, token_gradient, dqde, actor_encoding, actor_tokens,
                gradients, capture, stage_gradients, leaves)
            if self._debug_gradient_chain_compare_backends and not action_only:
                self._compare_actor_attention_backends(
                    actor_tokens, leaves, actor_encoding, token_gradient, dqde,
                    critic_observation, critic_action, matched_actor_ids, k,
                    batch_size, state, reference_rng)
            self._gradient_chain_pending = (
                action_loss, probe_hidden_loss, probe_output_loss)

        actor_info = LossInfo(
            loss=action_loss,
            extra=BafcActorInfo(eval_action_loss=eval_action_loss)) 
        return critic_state, actor_info

    def _record_actor_gradient_chain(self, dqda, token_gradient, dqde,
                                     encoding, tokens, gradients, capture,
                                     stage_gradients, leaves):
        """Observe the executed objective backward, never a reconstructed graph."""
        prefix = 'gradient_chain/rank_local/'
        summarize = gradient_diagnostics.summarize_tensor
        summarize(prefix + 'action', dqda)
        if token_gradient is not None:
            summarize(prefix + 'tokens', token_gradient)
            summarize(prefix + 'encoding', gradients[id(encoding)])
            # Token partials and connected leaf cotangents are different objects.
            # Their difference isolates all paths through descendant features.
            offset = (tokens.shape[1] - sum(x.shape[-1] for x in leaves))
            for i, (leaf, total) in enumerate(zip(leaves, dqde)):
                width = leaf.shape[-1]
                direct = token_gradient[:, offset:offset + width].permute(2, 0, 1)
                summarize(prefix + 'features/leaf_%d/direct' % i, direct)
                summarize(prefix + 'features/leaf_%d/connected' % i, total)
                summarize(prefix + 'features/leaf_%d/through_descendants' % i,
                          total - direct)
                offset += width
        for name, activation in capture.items():
            summarize(prefix + 'stages/' + name + '/activation', activation,
                      histogram=False)
            if name in stage_gradients:
                summarize(prefix + 'stages/' + name + '/cotangent',
                          stage_gradients[name], histogram=False)
        # Only readout queries are materialized: O(L*d), not all L*L scores.
        for name, query in capture.items():
            if not name.endswith('/q'):
                continue
            layer = name[:-2]
            key = capture[layer + '/k']
            logits = (query.detach().double()[:, :, :1] @
                      key.detach().double().transpose(-2, -1)) / query.shape[-1]**0.5
            probability = logits.softmax(-1)
            top = logits.topk(min(2, logits.shape[-1]), dim=-1)
            winner = top.indices[..., 0]
            attention_prefix = prefix + 'attention/' + layer + '/'
            summarize(attention_prefix + 'readout_logits', logits)
            summarize(attention_prefix + 'entropy',
                      -(probability * probability.clamp_min(1e-300).log()).sum(-1))
            summarize(attention_prefix + 'max_probability', probability.amax(-1))
            summarize(attention_prefix + 'winner', winner)
            if top.values.shape[-1] == 2:
                summarize(attention_prefix + 'winner_margin',
                          top.values[..., 0] - top.values[..., 1])
            output_start = tokens.shape[1] - self._action_spec.shape[-1]
            summarize(attention_prefix + 'winner_is_action',
                      (winner >= output_start).float())
            for actor in range(winner.shape[0]):
                for head in range(winner.shape[1]):
                    alf.summary.scalar(
                        attention_prefix + 'actor_%d/head_%d/winner' % (actor, head),
                        winner[actor, head, 0])
            for kind in ('q_raw', 'k_raw', 'q', 'k'):
                summarize(attention_prefix + kind + '_norm',
                          capture[layer + '/' + kind].detach().double().norm(dim=-1))

    def _compare_actor_attention_backends(self, tokens, leaves, encoding,
                                           token_gradient, dqde, observation,
                                           action, matched_ids, k, batch_size,
                                           state, reference_rng):
        """Shadow the same actor objective using math SDPA and isolated state."""
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.func import functional_call

        if not isinstance(self._actor_encoder, TransformerEncoder):
            return
        # Different SDPA backends need not draw identical attention-dropout
        # masks from the same RNG seed. A deterministic comparison requires p=0.
        stochastic = any(
            module.training and (
                (isinstance(module, nn.Dropout) and module.p > 0) or
                (isinstance(module, nn.MultiheadAttention) and module.dropout > 0))
            for network in (self._actor_encoder, self._critic_networks)
            for module in network.modules())
        alf.summary.scalar('gradient_chain/backend_comparison/skipped_dropout',
                           int(stochastic))
        if stochastic:
            return
        # Reuse the actor graph only for the final VJP, so connected dqde includes
        # precisely the same h -> action path and the same probe realization.
        reference_tokens = tokens.detach().float().requires_grad_(True)
        devices = ([tokens.device.index] if tokens.is_cuda else [])
        encoder_state = {
            name: (tensor.detach().float().clone() if tensor.is_floating_point()
                   else tensor.detach().clone()) for name, tensor in
            list(self._actor_encoder.named_parameters()) +
            list(self._actor_encoder.named_buffers())}
        critic_state = {
            name: (tensor.detach().float().clone() if tensor.is_floating_point()
                   else tensor.detach().clone()) for name, tensor in
            list(self._critic_networks.named_parameters()) +
            list(self._critic_networks.named_buffers())}
        # No new critic/probe selection and no persistent RNG/buffer changes.
        with torch.random.fork_rng(devices=devices), sdpa_kernel(SDPBackend.MATH), \
                torch.autocast(device_type=tokens.device.type, enabled=False):
            torch.set_rng_state(reference_rng['cpu'])
            if tokens.is_cuda:
                torch.cuda.set_rng_state(reference_rng['cuda'], tokens.device)
            reference_encoding = functional_call(
                self._actor_encoder, encoder_state, (reference_tokens,))[0]
            matched_encoding = reference_encoding[matched_ids]
            critic_encoding = matched_encoding[:, None].expand(
                k, batch_size, self._num_actor_critic, matched_encoding.shape[-1]
            ).reshape(k * batch_size, self._num_actor_critic, -1)
            reference_q = functional_call(
                self._critic_networks, critic_state,
                ((critic_encoding, (observation.detach().float(), action.detach().float())), state))[0]
            reference_gradient = torch.autograd.grad(
                reference_q.sum() / k, reference_tokens)[0]
        reference_leaves = [x for x in leaves if x.requires_grad]
        connected = torch.autograd.grad(
            tokens, reference_leaves, grad_outputs=reference_gradient.to(tokens.dtype),
            retain_graph=True, allow_unused=True) if reference_leaves else []
        connected = {id(x): (g if g is not None else torch.zeros_like(x))
                     for x, g in zip(reference_leaves, connected)}
        comparisons = [('encoding', encoding, reference_encoding),
                       ('tokens', token_gradient, reference_gradient)]
        comparisons.extend(('dqde_leaf_%d' % i, actual,
                            connected.get(id(leaf), torch.zeros_like(leaf)))
                           for i, (leaf, actual) in enumerate(zip(leaves, dqde)))
        for name, actual, reference in comparisons:
            prefix = 'gradient_chain/backend_comparison/' + name
            actual, reference = actual.detach().double(), reference.detach().double()
            gradient_diagnostics.summarize_tensor(prefix + '/math', reference)
            gradient_diagnostics.summarize_tensor(prefix + '/difference', actual - reference)
            alf.summary.scalar(prefix + '/relative_l2_error',
                               (actual - reference).norm() /
                               reference.norm().clamp_min(1e-30))
            alf.summary.scalar(prefix + '/cosine',
                               gradient_diagnostics.gradient_cosine_similarity(
                                   [actual], [reference]))

    def _finish_actor_gradient_chain(self, info):
        """Parameter VJPs with the actual action/probe loss reductions."""
        pending, self._gradient_chain_pending = self._gradient_chain_pending, None
        if pending is None:
            return
        action_loss, hidden_loss, output_loss = pending
        action_loss = action_loss.reshape(info.step_type.shape)
        if self._config.mask_out_loss_for_last_step:
            action_loss = action_loss * (info.step_type != StepType.LAST)
        params = [p for p in self._actor_networks.parameters() if p.requires_grad]
        losses_to_measure = [('replay_action', action_loss.mean()),
                             ('connected_hidden' if self._use_legacy_actor_gradient
                              else 'direct_hidden', hidden_loss),
                             ('probe_output_path', output_loss)]
        branches = {}
        for name, loss in losses_to_measure:
            branches[name] = (torch.autograd.grad(
                loss, params, retain_graph=True, allow_unused=True)
                if isinstance(loss, torch.Tensor) and loss.requires_grad
                else (None,) * len(params))
        def add(*vectors):
            return tuple(sum(values) if values else None for values in
                         ([g for g in entries if g is not None]
                          for entries in zip(*vectors)))
        hidden_name = losses_to_measure[1][0]
        branches['probe_sum'] = add(branches[hidden_name], branches['probe_output_path'])
        branches['total'] = add(branches['replay_action'], branches['probe_sum'])
        prefix = 'gradient_chain/rank_local/parameters/'
        statistics = {}
        for name, vector in branches.items():
            statistics[name] = gradient_diagnostics.gradient_vector_statistics(vector)
            for key, scalar in statistics[name].items():
                alf.summary.scalar(prefix + name + '/' + key, scalar)
        total_norm = statistics['total']['norm'].clamp_min(1e-30)
        for name in branches:
            alf.summary.scalar(prefix + name + '/norm_relative_to_total',
                               statistics[name]['norm'] / total_norm)
        for left, right in ((hidden_name, 'probe_output_path'),
                            ('replay_action', 'probe_sum')):
            alf.summary.scalar(prefix + left + '_vs_' + right + '/cosine',
                               gradient_diagnostics.gradient_cosine_similarity(
                                   branches[left], branches[right]))
        # These are local objective contributions before an outer Agent's loss
        # weight or replay importance weight. Actual reduced optimizer gradients
        # are separately observed in after_update(). Baseline uses unit weights.
        alf.summary.scalar(prefix + 'excludes_outer_loss_weights', 1)

    def _record_td_gradient_diagnostics(self, info, td_records):
        # Batch reductions across all actors/critics. Per-actor attribution is
        # expanded into scalar tags only at the sparse summary boundary.
        if len(td_records) != self._num_actor_critic:
            return
        values, residuals, targets = [], [], []
        for loss_fn, value, residual, capture in td_records:
            if not isinstance(residual, torch.Tensor) or residual.shape != value.shape:
                return  # A custom/quantile loss may not expose ordinary TD residuals.
            if capture:
                value, residual = capture['value'], capture['residual']
                targets.append(capture['target'])
            else:
                value, residual = value[:-1].detach(), residual[:-1].detach()
                if getattr(loss_fn, '_normalize_target', False):
                    value = loss_fn._target_normalizer.normalize(value)
            values.append(value)
            residuals.append(residual)
        value = torch.stack(values, dim=3)
        residual = torch.stack(residuals, dim=3)
        mask = (info.step_type[:-1] != StepType.LAST)[:, :, None, None]
        if self._use_bootstrap_critics:
            mask = mask & info.bootstrap_mask[:-1, :, None, :].bool()
        dimensions = (0, 1) + tuple(range(4, value.ndim))
        measurements = [('residual', residual), ('value', value)]
        if len(targets) == self._num_actor_critic:
            measurements.append(('target', torch.stack(targets, dim=3)))
        # A custom loss without the observer cannot expose an exact target.
        # Reconstructing value + residual would lose small targets by cancellation.
        for name, tensor in measurements:
            statistics = gradient_diagnostics.tensor_statistics(
                tensor, mask=mask, reduce_dims=dimensions)
            self._gradient_chain_intervals.record_statistics(
                'td/' + name, statistics, self._gradient_chain_step)

    def _record_optimizer_gradient_diagnostics(self):
        # The inline minibatch summary gate selects only the last update. Read
        # preceding critic gradients on the same summary ITERATION before they
        # are cleared by the actor update. No extra backward/collective is used.
        step = int(alf.summary.get_global_counter())
        config = self._config
        scheduled = (step % config.summary_interval == 0 or
                     (config.summarize_first_interval and step < config.summary_interval))
        if not (self._debug_summaries and alf.summary.is_summary_enabled() and scheduled):
            return
        modules = [('actor', self._actor_networks), ('encoder', self._actor_encoder),
                   ('critic', self._critic_networks)]
        for name, module in modules:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients:
                continue
            statistics = gradient_diagnostics.gradient_vector_statistics(gradients)
            self._gradient_chain_intervals.record_statistics(
                'optimizer_gradients/' + name, statistics, self._gradient_chain_step)
        if self._actor_eval_samples.grad is not None:
            self._gradient_chain_intervals.record_statistics(
                'optimizer_gradients/probes',
                gradient_diagnostics.gradient_vector_statistics([self._actor_eval_samples.grad]),
                self._gradient_chain_step)

    def _select_critic_targets(self, target_critics):
        """Optionally construct one RLPD-style target for all critics.

        Args:
            target_critics: target values with shape
                ``[batch, n_actor, n_critic, ...]``.

        Returns:
            The input unchanged when random targets are disabled. Otherwise,
            returns the elementwise minimum of a random subset with shape
            ``[batch, n_actor, ...]``.
        """
        if not self._use_random_critic_targets:
            return target_critics

        if self._num_sampled_critic_targets < self._num_actor_critic:
            critic_ids = torch.randperm(
                self._num_actor_critic,
                device=target_critics.device)[:self._num_sampled_critic_targets]
            sampled_targets = target_critics.index_select(2, critic_ids)
        else:
            sampled_targets = target_critics
        if self.has_multidim_reward():
            sign = self.reward_weights.sign()
            return (sampled_targets * sign).min(dim=2)[0] * sign
        return sampled_targets.min(dim=2)[0]

    def _critic_train_step(self,
                           observation,
                           state: BafcCriticState,
                           rollout_info: BafcInfo,
                           action,
                           actor_eval_samples=None):
        ## Step 1: encode all actors from actor_eval_samples
        ####################################################
        actor_encoding, _, actor_tokens = self._encode_actor_policies(
            actor_eval_samples, return_tokens=True)
        if self._use_target_actor_encoder:
            with torch.no_grad():
                target_actor_encoding = self._target_actor_encoder(
                    actor_tokens.detach())[0]
        else:
            target_actor_encoding = actor_encoding

        ## Step 2: compute critics and target critics for training actor batch
        ##
        ## use all actors times all (s, a) samples: a [T*B * n_actor] batch   
        ## - critic network gets [n_actor, d_enc] & [T*B, d_sa]
        ##   it performs a cross-prod to form the desired batch
        ## - target_critic network gets [n_actor, d_enc] & [T*B * n_actor, d_sa]
        ##   with "obs_action_batch_dominate", it forms the desired batch
        ######################################################################
        batch_size = observation.shape[0]
        # repeat the entirety of actor_encoding T*S times -> [n_actor * T*S, d_enc]
        actor_encoding = actor_encoding.repeat(batch_size, 1)
        target_actor_encoding = target_actor_encoding.repeat(batch_size, 1)
        # repeat each row of rollout obs & action n_actor times -> [n_actor * T*S, d_sa]
        critic_observation = observation.repeat_interleave(
            self._num_actor_critic, dim=0)
        critic_action = rollout_info.action.repeat_interleave(
            self._num_actor_critic, dim=0)

        # [n_actor * T*S, n_critic]
        critics, critic_state = self._critic_networks(
            (actor_encoding, (critic_observation, critic_action)), state.critic)

        with torch.no_grad():
            # [T*B, d_s] --> [T*B * n_actor, d_s], same batch size as action
            target_observation = observation.repeat_interleave(
                self._num_actor_critic, dim=0)
            target_critics, target_critic_state = self._target_critic_networks(
                (target_actor_encoding, (target_observation, action)), state.target_critic)

        # [T*B*n_actor, n_critic] -> [T*S, n_actor, n_critic]
        critics = critics.reshape(-1, self._num_actor_critic, *critics.shape[1:])
        target_critics = target_critics.reshape(
            -1, self._num_actor_critic, *target_critics.shape[1:])
        target_critics = target_critics.detach()
        target_critics = self._select_critic_targets(target_critics)

        state = BafcCriticState(
            critic=critic_state, target_critic=target_critic_state)
        info = BafcCriticInfo(critic=critics, target_critic=target_critics)

        return state, info

    def _update_train_mode(self):
        if self._train_mode == TrainMode.actor:
            if self._actor_update_counter % self._actor_utd == 0:
                self._train_mode = TrainMode.critic
                # self._critic_network.set_obs_action_batch_dominate(False)
                for p in self._actor_networks.parameters():
                    p.requires_grad_(False)
                self._actor_eval_samples.requires_grad_(
                    self._eval_samples_source == 'trainable'
                    and not self._use_actor_id_encoding)
        elif self._train_mode == TrainMode.critic:
            if self._critic_update_counter % self._critic_utd == 0:
                self._train_mode = TrainMode.actor
                # self._critic_network.set_obs_action_batch_dominate(True)
                for p in self._actor_networks.parameters():
                    p.requires_grad_(True)
                self._actor_eval_samples.requires_grad_(False)

    def train_step(self, inputs: TimeStep, state: BafcState,
                   rollout_info: BafcInfo):
        assert not self._is_eval
        self._training_started = True
        actor_eval_samples = (None if self._use_actor_id_encoding else
                              self._get_actor_eval_samples())

        # [T*B, n_actor, d_a]
        action, action_state = self._predict_action(
            self._actor_networks, inputs.observation, 
            state=state.action, train=True)

        if self._train_mode == TrainMode.standard or (
                self._critic_update_counter == 0
                and self._actor_update_counter == 0):
            actor_action = action  # [T*B, n_actor, d_a]
            actor_state, actor_info = self._actor_train_step(
                inputs.observation, actor_action, rollout_info.action,
                rollout_info.bootstrap_mask, state.actor, actor_eval_samples)
            critic_action = action.reshape(-1, action.shape[-1])  # [T*B * n_actor, d_a]
            critic_state, critic_info = self._critic_train_step(
                inputs.observation, state.critic, rollout_info, critic_action,
                actor_eval_samples)
            new_state = BafcState(action=action_state,
                                  actor=actor_state,
                                  critic=critic_state)
            self._critic_update_counter += 1
        else:
            if self._train_mode == TrainMode.actor:
                actor_state, actor_info = self._actor_train_step(
                    inputs.observation, action, rollout_info.action,
                    rollout_info.bootstrap_mask, state.actor,
                    actor_eval_samples)
                critic_info = BafcCriticInfo()
                new_state = BafcState(action=action_state,
                                      actor=actor_state,
                                      critic=state.critic)
                self._actor_update_counter += 1
            else:
                action = action.reshape(-1, action.shape[-1])  # [T*B * n_actor, d_a]
                critic_state, critic_info = self._critic_train_step(
                    inputs.observation, state.critic, rollout_info, action,
                    actor_eval_samples)
                actor_info = LossInfo(extra=BafcActorInfo())
                new_state = BafcState(action=action_state,
                                      actor=state.actor,
                                      critic=critic_state)
                self._critic_update_counter += 1

        if self._debug_summaries and alf.summary.should_record_summaries():
            self._do_critic_summary = True
            if actor_eval_samples is not None:
                safe_mean_hist_summary('eval_samples', actor_eval_samples)
                safe_mean_hist_summary('eval_samples/per_dim_mean',
                                       actor_eval_samples.mean(dim=0))
                safe_mean_hist_summary(
                    'eval_samples/per_dim_std',
                    actor_eval_samples.std(dim=0, unbiased=False))
                safe_mean_hist_summary('eval_samples/per_sample_l2_norm',
                                       actor_eval_samples.norm(dim=-1))

        info = BafcInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            actor=actor_info,
            critic=critic_info,
            discounted_return=rollout_info.discounted_return,
            bootstrap_mask=rollout_info.bootstrap_mask)
        return AlgStep(action, new_state, info)

    def calc_loss(self, info: BafcInfo):
        assert not self._is_eval
        if self._gradient_chain_pending is not None:
            self._finish_actor_gradient_chain(info)
        actor_loss = info.actor
        eval_action_loss = actor_loss.extra.eval_action_loss
        if isinstance(eval_action_loss, torch.Tensor):
            eval_action_loss = eval_action_loss.mean()
        if self._train_mode == TrainMode.actor:
            critic_loss = LossInfo()
        else:
            critic_loss = self._calc_critic_loss(info)

        loss = math_ops.add_ignore_empty(actor_loss.loss, critic_loss.loss)

        return LossInfo(
            loss=loss,
            scalar_loss=eval_action_loss,
            extra=BafcLossInfo(
                actor=actor_loss.extra, critic=critic_loss.extra))

    def _calc_critic_loss(self, info: BafcInfo):
        with alf.summary.record_if(lambda: self._do_critic_summary):
            critic_info = info.critic
            critic_losses = []
            td_records = []
            for i, l in enumerate(self._critic_losses):
                # critics has shape [T, S, n_actor, n_critic, ...].
                # A random shared target has shape [T, S, n_actor, ...];
                # otherwise targets retain the critic dimension.
                if self._use_random_critic_targets:
                    target_value = critic_info.target_critic
                else:
                    target_value = critic_info.target_critic[:, :, :, i, ...]
                value = critic_info.critic[:, :, :, i, ...]
                capture_context = (l.capture_diagnostics()
                                   if self._debug_gradient_chain and
                                   hasattr(l, 'capture_diagnostics') else nullcontext({}))
                with capture_context as capture:
                    td_loss = l(info=info, value=value, target_value=target_value)
                critic_loss = td_loss.loss
                if self._debug_gradient_chain:
                    td_records.append((l, value, td_loss.extra, capture))
                if self._use_bootstrap_critics:
                    bootstrap_mask = info.bootstrap_mask[:, :,
                                                         i] / self._bootstrap_mask_prob
                    critic_loss = critic_loss * bootstrap_mask
                critic_losses.append(critic_loss)

        if self._debug_gradient_chain:
            self._record_td_gradient_diagnostics(info, td_records)
        self._do_critic_summary = False
        critic_loss = math_ops.add_n(critic_losses)

        return LossInfo(
            loss=critic_loss,
            extra=critic_loss)

    def _trainable_attributes_to_ignore(self):
        ignored = ['_target_critic_networks']
        if self._use_target_actor_encoder:
            ignored.append('_target_actor_encoder')
        return ignored

    def after_update(self, root_inputs, info: BafcInfo):
        if self._debug_gradient_chain:
            self._record_optimizer_gradient_diagnostics()
            if self._debug_summaries and alf.summary.should_record_summaries():
                self._gradient_chain_intervals.summarize(
                    'gradient_chain/intervals', self._gradient_chain_step)
                alf.summary.scalar('gradient_chain/optimizer_gradients_are_ddp_reduced',
                                   int(torch.distributed.is_initialized()))
                alf.summary.scalar('gradient_chain/actor_update_counter',
                                   self._actor_update_counter)
                alf.summary.scalar('gradient_chain/critic_update_counter',
                                   self._critic_update_counter)
            self._gradient_chain_step += 1
        self._update_train_mode()
        self._update_target_critic()
