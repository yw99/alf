# BAFCv7 computation reuse

The shared `bafcv7_dmc_conf.py` enables the optimized implementation for all V7
launchers. Pass `--disable-optimizations` to any V7 launcher, or set
`bafcv7_enable_optimizations=False`, to select the reference implementation.
The algorithm constructor itself defaults all optimization flags to `False`.

## Behavior and isolation

- `cache_frozen_probe_outputs`: reuse deterministic actor outputs on frozen
  probes between actor updates. Actor/probe tensor versions, identity, dtype,
  device, and autocast settings validate the cache. Actor updates, checkpoint
  loads, and device/dtype changes invalidate it. No encoder outputs or replay
  batches are cached across updates.
- `deduplicate_critic_episode_seeds`: encode each exact distinct episode seed
  once within a critic-only update, then differentiably gather into replay
  order. This preserves summed gradients to the encoder and trainable probes.
  Actor updates and initial joint updates retain the original surrogate path.
  Unsupported actors/encoders, nonzero dropout, and all-distinct batches use the
  original encoding operation (the latter still pays the uniqueness check).
- `selective_critic_evaluation`: evaluate only paired actor/critic diagonal
  entries on paired actor updates, and only sampled target critics. Online
  critic losses and minimum-over-all actor objectives still use all critics.
- `share_critic_observation_encoding`: encode a replay observation/action pair
  once per critic, then broadcast across actors before the policy-conditioned
  head. The features are never reused across optimizer updates.

`BafcV7FuncCriticNetwork` inherits the existing critic and returns a specialized
parallel wrapper with the original parameter names, shapes, order, and copy
semantics. The fast operations support the stock vector FC/LayerNorm structure;
unsupported custom configurations retain ordinary forward execution. No shared
network/layer implementation or other algorithm is modified. Caches are local
transient Python state and do not appear in checkpoints.

Episode seeds remain Gaussian vectors held constant throughout each episode.
Fresh action noise, the 12-update schedule, precision, architecture, probe count,
result paths, and GPU allocation are unchanged. The summary scalars
`unique_episode_seeds` and `replay_seed_items` expose the actual reuse opportunity.

## Validation

Reference/optimized checks cover both variants, both policy feature modes,
frozen/trainable probes, initial/critic/actor updates, duplicate/distinct seeds,
clipping, normalization, selected gradients, cache invalidation, target updates,
and strict model/optimizer checkpoint loading in both directions. Multi-update
comparisons use float64 to avoid Adam amplifying float32 rounding in nearly-zero
bias gradients; separate float32 tests check losses and gradients.

Full dog-sized float32 GPU checks passed for all 12 variant/feature/update-phase
combinations with `rtol=3e-4, atol=3e-6`. Maximum observed absolute gradient
error was approximately `1.2e-6`. A four-GPU Gloo smoke test (the launcher's
backend) passed gradient/parameter synchronization and checkpoint reload across
all four variant/feature combinations and 12 consecutive updates each.

Run the regression checks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest \
  alf.algorithms.bafc_algorithm_v7_test \
  alf.algorithms.bafc_algorithm_v7_optimization_test \
  alf.networks.bafc_v7_actor_network_test \
  alf.networks.critic_networks_test \
  alf.networks.encoding_networks_test \
  alf.examples.bafcv7_dmc_conf_test
```

## Synthetic performance measurements

Measured on RTX 5090 GPUs, with each benchmark process on a separate GPU:
223 observation coordinates, 38 action coordinates, 512 probes, four Transformer
layers, float32, frozen probes, and 128 flattened replay items per rank.
Each case uses 10 warmup iterations and 30 synchronized measured iterations.
These timings include loss, backward, Adam, and target updates, but exclude
MuJoCo, replay sampling, and DDP communication. Concurrent benchmark processes
can introduce host scheduling noise, especially for small ensemble workloads.

Complete 12-update cycles with 32 distinct seeds per batch:

| Variant | Features | Reference | Optimized | Speed ratio | Peak allocated MiB, reference → optimized |
|---|---|---:|---:|---:|---:|
| ensemble_base | mean_log_std | 0.280 s | 0.283 s | 0.99× | 958 → 855 |
| ensemble_base | action_quantiles | 0.285 s | 0.287 s | 0.99× | 1016 → 906 |
| single_seeded | mean_log_std | 1.482 s | 0.685 s | 2.16× | 6766 → 6772 |
| single_seeded | action_quantiles | 1.642 s | 0.749 s | 2.19× | 7558 → 7559 |

The full individual-flag sweep covered 1, 8, 32, 64, and 128 distinct seeds.
For seeded critic-only updates with 32 seeds, combined median time decreased
from 126.8 to 36.5 ms (`mean_log_std`) and 141.8 to 41.5 ms
(`action_quantiles`). With 128 distinct seeds it was effectively unchanged:
127.2 → 127.1 ms and 142.4 → 142.5 ms, respectively. Cache/selection/sharing
savings are much smaller than deduplication in seeded workloads; ensemble
full-cycle gains were in memory rather than measured speed.

**Seeded full-cycle peak memory remains dominated by actor updates, where
encoding deduplication is intentionally disabled. These optimizations do not
establish that four seeded jobs fit on the same GPU group.** Existing GPU
allocation is unchanged and no experiments were restarted.

No V7 replay checkpoint was present in the stopped experiment directories, so
these seed ratios are synthetic; they are not a measurement of replay contents.

Reproduce benchmarks (JSON output defaults to `/tmp`):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m alf.bin.benchmark_bafcv7
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m alf.bin.benchmark_bafcv7 \
  --cycle --unique 32 --cases reference combined --output /tmp/bafcv7_cycles.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m alf.bin.benchmark_bafcv7 --ddp-smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m alf.bin.benchmark_bafcv7 --check-equivalence
```

Use `--variants`, `--features`, `--unique`, and `--cases` for smaller sweeps;
`--probes trainable` checks the alternate probe configuration. Benchmark files
are synthetic validation artifacts and never touch experiment result directories.
