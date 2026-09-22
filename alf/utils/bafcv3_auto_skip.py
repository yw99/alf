"""Opt-in, checkpointable automatic activation of the TR2 rollout gate.

The controller consumes observations, never TensorBoard files. All time axes are
absolute rank-zero environment steps. Values are held causally on a fixed grid.
"""
import copy
import math
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

DEFAULTS = dict(sample_steps=1000, window_steps=5000, warmup_fraction=.5,
                slowdown_multiplier=1., return_floor=.02,
                max_return_decline=.02, trust_growth_limit=1.10,
                consecutive_checks=2)


def validate_settings(settings):
    if set(settings) != set(DEFAULTS):
        raise ValueError('Unknown or missing automatic skip settings')
    for key in ('sample_steps', 'window_steps', 'consecutive_checks'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if settings['window_steps'] % settings['sample_steps']:
        raise ValueError('window_steps must be divisible by sample_steps')
    for key in set(DEFAULTS) - {'sample_steps', 'window_steps', 'consecutive_checks'}:
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if settings['warmup_fraction'] > 1 or settings['trust_growth_limit'] < 1:
        raise ValueError('Invalid warmup fraction or trust growth limit')


def _finite(value, nonnegative=False):
    try:
        value = float(value)
        return value if math.isfinite(value) and (not nonnegative or value >= 0) else None
    except (TypeError, ValueError):
        return None


class AutoSkipController:
    """Pure deterministic controller; activation calibration is managed outside."""
    def __init__(self, settings, final_steps):
        validate_settings(settings)
        self.settings = dict(settings)
        self.final_steps = int(final_steps)
        self.first_step = None
        self.next_step = None
        self.previous = None
        self.samples = []
        self.streak = 0
        self.activation_step = None
        self.calibration = None

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        if state['settings'] != self.settings or state['final_steps'] != self.final_steps:
            raise ValueError('Automatic skip checkpoint settings mismatch')
        self.__dict__.update(copy.deepcopy(state))

    def observe(self, step, average_return, trust):
        """Return audit records for newly crossed sample boundaries.

        At an exact boundary use the current value; for a crossed boundary use
        the previous observation, never a measurement from the future.
        """
        step = int(step)
        current = (step, _finite(average_return), _finite(trust, True))
        if self.previous is not None and step < self.previous[0]:
            raise ValueError('Environment steps moved backwards')
        if self.activation_step is not None:
            return []
        interval = self.settings['sample_steps']
        if self.first_step is None:
            self.first_step = step
            self.next_step = ((step + interval - 1) // interval) * interval
        records = []
        while self.next_step <= step:
            t = self.next_step
            observation = current if t == step else self.previous
            ret, metric = observation[1:] if observation else (None, None)
            self.samples.append((t, ret, metric))
            n = self.settings['window_steps'] // interval
            self.samples = self.samples[-3 * n:]
            eligible = (t >= self.first_step + 3 * self.settings['window_steps']
                        and t >= self.settings['warmup_fraction'] * self.final_steps
                        and len(self.samples) == 3 * n)
            record = dict(step=t, average_return=ret, trust=metric,
                          history_ready=eligible, previous_gain=None,
                          recent_gain=None, trust_previous_median=None,
                          trust_recent_median=None, return_pass=False,
                          trust_pass=False)
            if eligible and all(r is not None for _, r, _ in self.samples) and all(
                    m is not None for _, _, m in self.samples[-2*n:]):
                returns = [r for _, r, _ in self.samples]
                r0, r1, r2 = [statistics.mean(returns[i*n:(i+1)*n]) for i in range(3)]
                gp = (r1-r0) / max(abs(r0), 1.)
                gr = (r2-r1) / max(abs(r1), 1.)
                old = statistics.median([m for _, _, m in self.samples[-2*n:-n]])
                new = statistics.median([m for _, _, m in self.samples[-n:]])
                record.update(previous_gain=gp, recent_gain=gr,
                              trust_previous_median=old, trust_recent_median=new,
                              return_pass=(-self.settings['max_return_decline'] <= gr <= max(
                                  self.settings['return_floor'], self.settings['slowdown_multiplier']*gp)),
                              trust_pass=new <= self.settings['trust_growth_limit']*old)
            self.streak = self.streak + 1 if record['return_pass'] and record['trust_pass'] else 0
            record['streak'] = self.streak
            if self.streak >= self.settings['consecutive_checks']:
                self.activation_step = t
            record['active'] = self.activation_step is not None
            records.append(record)
            self.next_step += interval
            if self.activation_step is not None:
                break
        self.previous = current
        return records


def after_iteration(agent):
    """Called by every worker, before checkpointing and the next rollout."""
    from alf.bin.evaluate_bafcv3_checkpoints import atomic_json
    from alf.utils.bafcv3_restart import collective_call, measure_calibration
    import json
    alg = agent._rl_algorithm
    controller = alg._auto_skip_controller
    if controller.calibration is not None:
        return
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    def observation():
        if rank != 0:
            return None
        metrics = {m.name: m.result() for m in agent.get_metrics()}
        return (int(metrics['EnvironmentSteps']), _finite(metrics.get('AverageReturn')),
                _finite(getattr(alg, '_last_eval_trust_rank_max', None), True))
    box = [collective_call(observation)]
    if distributed:
        dist.broadcast_object_list(box, src=0)
    records = collective_call(lambda: controller.observe(*box[0]))
    # Rank zero is the authority, even though all ranks replay identical inputs.
    decision = [controller.activation_step if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(decision, src=0)
    if decision[0] is not None:
        options = alg._restart_options
        result = measure_calibration(agent, options, rank,
                                     seed_context=('auto_skip', decision[0]))
        result.update(activation_step=decision[0], trigger=copy.deepcopy(records[-1]))
        controller.calibration = result
        alg._eval_trust_max = result['threshold']
        alg._eval_gate_consecutive_rollout_skips = 0
        alg._enable_eval_rollout_skip_gate = True
        alg._trust_metric_update_counter = 0
        def audit():
            if rank == 0:
                atomic_json(Path(options['root_dir']) / 'auto_skip_activation.json', result)
        collective_call(audit)
    def log():
        if rank == 0 and records:
            path = Path(alg._restart_options['root_dir']) / 'auto_skip_checks.jsonl'
            with path.open('a') as stream:
                for record in records:
                    entry = dict(record, observed_env_step=box[0][0])
                    stream.write(json.dumps(entry, allow_nan=False) + '\n')
    collective_call(log)
