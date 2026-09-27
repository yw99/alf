"""Pure checkpoint compatibility rules for the BAFCv3 -> TR2 experiment.

Only migration calls this module. Ordinary training and TR2 checkpoint resume
keep their native state loading behavior.
"""

from numbers import Integral

import torch


RUNTIME_PREFIX = '_rl_algorithm._bafc_runtime.'
_REQUIRED = ('training_started', 'train_mode', 'rollout_actor_id',
             'actor_update_counter', 'critic_update_counter')
_OPTIONAL = {'reweighting_target_observation_cache', 'target_updater_counter'}


def migrate_runtime_state(source_state, *, target_critic_period,
                          target_critic_use_ema):
    """Return runtime overrides and audit metadata without modifying inputs.

    The observation-cache field is accepted here and migrated by the restart module,
    which supplies the cache reconstruction/normalization protocol. Delayed EMA
    and non-unit update periods require a separate migration implementation.
    """
    if (isinstance(target_critic_period, bool)
            or not isinstance(target_critic_period, Integral)
            or target_critic_period != 1):
        raise ValueError('BAFCv3 -> TR2 restart requires target_critic_period=1')
    if target_critic_use_ema is not False:
        raise ValueError('BAFCv3 -> TR2 restart requires target_critic_use_ema=False')
    runtime = {key[len(RUNTIME_PREFIX):]: value
               for key, value in source_state.items()
               if key.startswith(RUNTIME_PREFIX)}
    if 'target_updater_recent_models' in runtime:
        raise ValueError('BAFCv3 -> TR2 restart does not support intermediate '
                         'target-updater models (delayed EMA)')
    unknown = set(runtime) - set(_REQUIRED) - _OPTIONAL
    if unknown:
        raise ValueError(f'Unrecognized source runtime fields: {sorted(unknown)}')
    for name in _REQUIRED:
        if name not in runtime:
            raise ValueError(f'Missing source runtime state: {name}')

    saved = 'target_updater_counter' in runtime
    counter = runtime.get('target_updater_counter', 0)
    if isinstance(counter, torch.Tensor):
        if counter.ndim != 0 or counter.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('target_updater_counter must be a nonnegative scalar integer')
        counter = counter.item()
    if isinstance(counter, bool) or not isinstance(counter, Integral) or counter < 0:
        raise ValueError('target_updater_counter must be a nonnegative scalar integer')
    overrides = {RUNTIME_PREFIX + name: runtime[name] for name in _REQUIRED}
    overrides[RUNTIME_PREFIX + 'target_updater_counter'] = torch.tensor(
        int(counter), dtype=torch.int64)
    audit = dict(target_updater_counter=int(counter),
                 target_updater_counter_protocol=(
                     'restored' if saved else 'legacy_period_one_zero_fallback'))
    return overrides, audit
