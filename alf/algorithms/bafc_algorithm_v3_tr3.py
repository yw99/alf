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
"""TR2-compatible BAFC variant for rebuilding missing replay state.

TR3 intentionally keeps TR2's registered module layout unchanged. Its
resume-only additions are a transient pre-training replay refill and structural
train-info spec priming coordinated by ``BafcAlgorithmV3TR3Agent``.
"""

import time

from absl import logging
import torch

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2
from alf.tensor_specs import TensorSpec
from alf.utils import dist_utils


@alf.configurable
class BafcAlgorithmV3TR3(BafcAlgorithmV3TR2):
    """A checkpoint-compatible TR2 subclass with transient resume control."""

    def _root_replay_checkpoint_size(self, state_dict):
        replay_keys = [
            key for key in state_dict
            if key.startswith("_replay_buffer.")
        ]
        if not replay_keys:
            return None

        size_keys = [
            key for key in replay_keys if key.endswith("._current_size")
        ]
        if not size_keys:
            logging.warning(
                "TR3 found replay checkpoint fields without current-size "
                "metadata; treating the replay checkpoint as empty.")
            return 0
        return sum(
            int(torch.as_tensor(state_dict[key]).sum().item())
            for key in size_keys)

    def _alf_prepare_checkpoint_load(self, state_dict):
        replay_size = self._root_replay_checkpoint_size(state_dict)
        initial_collect_steps = int(self._config.initial_collect_steps)
        # ``Algorithm._train_info_spec`` is lazy and is not checkpointed. A
        # restored TR2 controller can start in a mode whose train info omits
        # the inactive actor or critic fields, so TR3 must prime a complete
        # root-Agent spec before the first resumed replay update is batched.
        self._tr3_train_info_spec_priming_pending = True
        # Every rank must participate in exactly one refill decision after a
        # checkpoint load, even when its own replay is already sufficient.
        # Keep this transient so TR2 checkpoint keys and shapes stay unchanged.
        self._tr3_replay_refill_decision_pending = True
        self._tr3_replay_checkpoint_size = replay_size
        self._tr3_clear_probe_replay = replay_size is None
        self._tr3_replay_refill_pending = (
            initial_collect_steps > 0
            and (replay_size is None or replay_size < initial_collect_steps))
        return super()._alf_prepare_checkpoint_load(state_dict)

    def _tr3_is_train_info_spec_priming_pending(self):
        return getattr(self, "_tr3_train_info_spec_priming_pending", False)

    def _tr3_mark_train_info_spec_primed(self):
        self._tr3_train_info_spec_priming_pending = False

    def _tr3_is_replay_refill_decision_pending(self):
        return getattr(self, "_tr3_replay_refill_decision_pending", False)

    def _tr3_replay_refill_plan(self):
        """Return transient refill information populated during checkpoint load."""
        if not getattr(self, "_tr3_replay_refill_pending", False):
            return None
        return dict(
            clear_probe_replay=getattr(self, "_tr3_clear_probe_replay", False),
            checkpoint_size=getattr(self, "_tr3_replay_checkpoint_size", None))

    def _tr3_begin_replay_refill(self):
        """Freeze rank-local controller/RNG state around refill rollouts."""
        assert not getattr(self, "_tr3_replay_refill_active", False)
        self._tr3_replay_refill_snapshot = self._rank_local_checkpoint_state()
        self._tr3_replay_refill_active = True

    def _tr3_end_replay_refill(self):
        snapshot = getattr(self, "_tr3_replay_refill_snapshot", None)
        try:
            if snapshot is not None:
                self._load_rank_local_checkpoint_state(snapshot)
        finally:
            self._tr3_replay_refill_active = False
            self._tr3_replay_refill_snapshot = None

    def _tr3_mark_replay_refill_complete(self):
        self._tr3_replay_refill_decision_pending = False
        self._tr3_replay_refill_pending = False
        self._tr3_clear_probe_replay = False

    def _append_target_metric_observations(self, observation):
        # The target metric cache is training state. Refill rollouts are only
        # used to rebuild replay and must not alter it or pay its copy cost.
        if getattr(self, "_tr3_replay_refill_active", False):
            return
        return super()._append_target_metric_observations(observation)


@alf.configurable
class BafcAlgorithmV3TR3Agent(Agent):
    """Agent that prepares replay and train-info state for resumed TR3."""

    @staticmethod
    def _tr3_tensor_leaf_spec(value):
        # In a train-info spec, TensorSpec is also the structural leaf marker
        # used by params_to_distributions(). Its shape is not consulted there.
        return TensorSpec(()) if value == () else value

    def _tr3_complete_train_info_spec(self, info):
        """Make TR2's actor/critic mode-dependent leaves structural leaves."""
        spec = dist_utils.extract_spec(info)
        rl_spec = spec.rl

        actor_spec = rl_spec.actor
        actor_extra_spec = actor_spec.extra._replace(
            eval_action_loss=self._tr3_tensor_leaf_spec(
                actor_spec.extra.eval_action_loss))
        actor_spec = actor_spec._replace(
            loss=self._tr3_tensor_leaf_spec(actor_spec.loss),
            extra=actor_extra_spec)

        critic_spec = rl_spec.critic
        critic_spec = critic_spec._replace(
            critic=self._tr3_tensor_leaf_spec(critic_spec.critic),
            target_critic=self._tr3_tensor_leaf_spec(
                critic_spec.target_critic),
            eval_trust_metric=self._tr3_tensor_leaf_spec(
                critic_spec.eval_trust_metric),
            critic_sample_weight=self._tr3_tensor_leaf_spec(
                critic_spec.critic_sample_weight))

        return spec._replace(
            rl=rl_spec._replace(actor=actor_spec, critic=critic_spec))

    def train_step(self, time_step, state, rollout_info):
        policy_step = super().train_step(time_step, state, rollout_info)
        controller = self._rl_algorithm
        if controller._tr3_is_train_info_spec_priming_pending():
            if self._train_info_spec is None:
                # Set this inside train_step(), before Algorithm's outer
                # collection path tries to infer a sparse spec from the same
                # critic-only or actor-only result.
                self._train_info_spec = self._tr3_complete_train_info_spec(
                    policy_step.info)
            controller._tr3_mark_train_info_spec_primed()
        return policy_step

    def observe_for_metrics(self, time_step):
        if getattr(self, "_tr3_replay_refill_active", False):
            return
        return super().observe_for_metrics(time_step)

    def _tr3_distributed_rank(self):
        if (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            return torch.distributed.get_rank()
        return -1

    def _tr3_refill_replay_buffer(self, plan):
        replay_buffer = self._replay_buffer
        if replay_buffer is None:
            raise RuntimeError(
                "TR3 replay refill was requested before the replay buffer was "
                "created.")
        if self._config.async_unroll:
            raise RuntimeError("TR3 replay refill requires async_unroll=False.")

        target_size = int(self._config.initial_collect_steps)
        capacity = replay_buffer.num_environments * replay_buffer.max_length
        if target_size > capacity:
            raise RuntimeError(
                "TR3 replay refill cannot reach initial_collect_steps="
                f"{target_size}: replay capacity is only {capacity}.")

        if plan["clear_probe_replay"]:
            # RingBuffer.clear() converts environment IDs through its logical
            # storage device. ALF can keep the registered metadata tensors on
            # the current accelerator instead, so clear those tensors directly.
            replay_buffer._current_size.zero_()
            replay_buffer._current_pos.zero_()
            if replay_buffer._dequeued:
                replay_buffer._enqueued.clear()
                replay_buffer._dequeued.set()

        start_size = int(replay_buffer.total_size)
        if start_size >= target_size:
            if (torch.distributed.is_available()
                    and torch.distributed.is_initialized()):
                torch.distributed.barrier()
            self._rl_algorithm._tr3_mark_replay_refill_complete()
            return

        rank = self._tr3_distributed_rank()
        logging.info(
            "TR3 replay refill starting [rank %s]: size=%s target=%s",
            rank, start_size, target_size)
        start_time = time.monotonic()
        was_training = self.training
        refill_succeeded = False
        self._tr3_replay_refill_active = True
        self._rl_algorithm._tr3_begin_replay_refill()
        try:
            self.eval()
            self.reset_state()
            previous_size = start_size
            with torch.no_grad(), alf.summary.record_if(lambda: False):
                while previous_size < target_size:
                    self.unroll(1)
                    current_size = int(replay_buffer.total_size)
                    if current_size <= previous_size:
                        raise RuntimeError(
                            "TR3 replay refill made no progress: replay size "
                            f"remained at {current_size}.")
                    previous_size = current_size
            refill_succeeded = True
        finally:
            try:
                self._rl_algorithm._tr3_end_replay_refill()
            finally:
                self._tr3_replay_refill_active = False
                self.train(was_training)
                # Refill environment and rollout state are deliberately
                # discarded; resumed training starts from a fresh episode.
                self._env.reset()
                self.reset_state()

        if (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            torch.distributed.barrier()

        if refill_succeeded:
            self._rl_algorithm._tr3_mark_replay_refill_complete()
            final_size = int(replay_buffer.total_size)
            elapsed = max(time.monotonic() - start_time, 1e-9)
            logging.info(
                "TR3 replay refill complete [rank %s]: size=%s collected=%s "
                "elapsed=%.1fs throughput=%.1f transitions/s", rank,
                final_size, final_size - start_size, elapsed,
                (final_size - start_size) / elapsed)

    def _tr3_synchronize_refill_plan(self, plan):
        if (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_world_size() <= 1):
            return plan
        any_rank_needs_refill = bool(
            self._rl_algorithm._all_reduce_control(
                [plan is not None],
                op=torch.distributed.ReduceOp.MAX)[0].item())
        if not any_rank_needs_refill:
            return None
        if plan is None:
            return dict(
                clear_probe_replay=False,
                checkpoint_size=int(self._replay_buffer.total_size))
        return plan

    def train_iter(self):
        if self._rl_algorithm._tr3_is_replay_refill_decision_pending():
            plan = self._rl_algorithm._tr3_replay_refill_plan()
            plan = self._tr3_synchronize_refill_plan(plan)
            if plan is not None:
                self._tr3_refill_replay_buffer(plan)
            else:
                self._rl_algorithm._tr3_mark_replay_refill_complete()
        # This is the first trainer-visible iteration after refill and follows
        # the ordinary Agent/TR2 path in full.
        return super().train_iter()
