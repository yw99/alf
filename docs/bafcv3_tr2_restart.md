# Restart BAFCv3 checkpoints with TR2

The restart entry point is `python -m alf.bin.train_bafcv3_tr2_restart`.
It loads an exact BAFCv3 checkpoint and its four replay shards, calibrates a
fixed threshold, and uses ALF's existing four-GPU launcher for continuation.
Source files are read only. Large outputs belong under `artifacts/`, which is
ignored by Git. The restart wrapper disables ALF's generic repository snapshot
with `--nostore_snapshot` because these output roots are inside the repository;
its explicit source/configuration/code snapshots are retained. Future launches
use `--port=0` for an automatically assigned optional HTTP control port per job.

## Start, validate, and resume

Run from `/root/alf` with the repository virtual environment:

```bash
MUJOCO_GL=egl .venv/bin/python -m alf.bin.train_bafcv3_tr2_restart \
  --source-checkpoint /workspace/server2_copy/dog_bafcv3_s0/train/algorithm/ckpt-75075 \
  --root-dir /root/alf/artifacts/my_dog_restart \
  --critic-utd 3 --rollout-skipping on \
  --threshold-quantile 0.33 --calibration-repetitions 100 \
  --final-env-steps-per-rank 150000 --worker-gpus 0,1,2,3
```

Add `--validate-only --validation-device cuda` to restore, calibrate, save an
initial restart checkpoint, and verify a fresh resume without learning or
collection. CPU validation is also available. The GPU path uses one visible GPU
per worker, exactly as the training launcher does. Validation uses ALF's
deterministic Torch settings. The CLI defaults `CUBLAS_WORKSPACE_CONFIG` to
`:4096:8` before starting workers and exposes the Python environment's companion
tools on the training worker `PATH` (including Ninja for native extensions).

For bounded startup validation, additionally supply `--smoke-env-steps 2
--smoke-train-iters 2`. These explicitly enable two real environment steps and
two replay-training iterations **after** saving the pristine initial checkpoint.
Smoke checkpoints are written separately under `validation_smoke/algorithm`;
they verify save/resume after learning and are not selected for full training.
The rollout smoke directly exercises collection, irrespective of gate eligibility.

Add `--resume` with the same source and experiment settings to reuse an existing
run directory. Once a restart checkpoint exists, resume restores its threshold
and controller without recalibration. A directory created using `--prepare-only`
contains configuration only; its first `--resume` completes migration and
calibration. Changed source inputs, configuration snapshots, or experiment
settings fail explicitly. To change an experiment setting, choose a new directory.

| Parameter | Default / constraint |
|---|---|
| `--source-checkpoint` | Required exact `ckpt-N` model file |
| `--root-dir` | Required separate new directory |
| `--critic-utd` | 3; also supports 11; UTD + 1 must divide 12 |
| `--rollout-skipping` | `on` or `off` |
| `--threshold-quantile` | 0.33, finite and within [0, 1] |
| `--calibration-repetitions` | 100, positive |
| `--calibration-seed` | 20260917 |
| `--final-env-steps-per-rank` | Absolute 150000, beyond the source progress |
| `--worker-gpus` | `0,1,2,3`, four distinct logical visible GPU indices |
| `--validation-device` | `cpu`; choose `cuda` for GPU startup checks |

Task, seed, networks, optimizer settings, replay, and observation preprocessing
come from the saved configuration. Supported inputs use a vector observation,
trainable actor evaluation samples, uniform non-recurrent replay, and
`ObservationNormalizer`. Unsupported schemas fail rather than silently resetting
state. Model, optimizer, and exactly replay ranks 0–3 are mandatory.

## Calibration and skipping

Each repetition samples a fresh native replay minibatch and a target-cache subset.
For the selected configurations this is 64 replay sequences of length two,
flattened to 128 observations, plus up to 128 target observations from a cache of
512. Samples can overlap between repetitions. Replay windows crossing an episode
reset or restart boundary are excluded. Raw replay observations are normalized
once with frozen checkpoint statistics; cached normalized observations are not
normalized again.

Let `C[r,j]` be TR2's ensemble-averaged metric for rank `r`, repetition `j`:

```text
M[j] = max_r C[r,j]
threshold = linear_quantile(M, q=0.33)
```

The rank maximum is taken **before** the quantile. Networks, optimizer, replay,
normalizer, progress, and training RNG are frozen during calibration. Sampling
seeds depend on the source model fingerprint, calibration seed, and rank; they do
not depend on UTD or whether skipping is enabled. Gate-disabled controls perform
the same calibration and continue monitoring the real metric.

`rollout_skip_sync_mode="min"` means all four ranks must vote to skip, equivalently
`max_r C[r] <= threshold`, together with cadence eligibility and a cap of three
consecutive skips. Skipping is disabled before calibration. The threshold stays
fixed; threshold decay, actor extension, and critic reweighting are disabled.
A 33% calibration quantile is initial sampled eligibility, not a promised skip
fraction during learning.

Metric settings remain ridge `1e-4`, 128 observations, a 512-observation target
cache, and refresh every eight actor updates. With 12 update slots per iteration:

| Critic UTD | Critic updates | Actor updates | Complete cycles |
|---|---:|---:|---:|
| 3 | 9 | 3 | 3 |
| 11 | 11 | 1 | 1 |

Thus UTD 3 refreshes the metric more frequently per training iteration. All arms
finish at the same **150k logged environment steps per rank**. The generated ALF
configuration uses 600k aggregate steps before ALF's four-rank division. Compute
and wall time must be compared separately; equal environment horizons do not
imply equal learning iterations when skipping is enabled.

## Restoration guarantees and limits

Migration checks tensor names, shapes, and dtypes explicitly. It restores actors,
trainable actor evaluation samples, encoders, current and target critics, saved
normalization, optimizer moments/steps/group settings, replay layout and ring
positions, cumulative update counts, trainer progress, global step, and saved
metric buffers. Optimizer IDs are mapped through verified parameter names in
fresh source and destination models. Reference actors and snapshot critics start
from the restored current networks; target critics retain their separate values.

No preparatory `train_iter()` runs. DDP is materialized with both actor and critic
parameters registered even when the current update phase temporarily freezes one
network. Unchanged UTD preserves the source phase. Changed UTD starts a complete
critic block using a separate phase offset without resetting cumulative counts
or triggering the first-ever joint update.

Rank 0 retains the available normalized observation cache. Other ranks reconstruct
caches from their own latest replay observations using the saved shared normalizer.
Historical rank-local normalization, simulator state, and historical RNG were not
saved and cannot be recovered. New deterministic training RNG streams are used;
future restart checkpoints save rank-local normalizers, caches, and RNG state.
Simulator collection begins a fresh episode. Fresh metric objects omit unavailable
partial-episode accumulators while retaining saved completed-episode buffers and
cumulative counts. Explicit replay boundaries prevent training across the missing
simulator transition. Replay loading preserves saved step types, including terminal
and time-limit discounts, rather than rewriting the newest entry.

All loads use `weights_only=True`. Native rank-local sidecars have a narrow NumPy
allowlist for their saved RNG state. Initial checkpoint publication puts the model
file last, after all rank sidecars are complete. Incomplete resumes fail explicitly.

## Outputs and interpretation

Each run stores `restart_manifest.json`, source/configuration/code snapshots,
`resolved_config.json`, four migration audits, `calibration.json` with every rank
and repetition, `restart_ready.json`, and a durable checkpoint under
`train/algorithm/`. Validation adds per-rank results, CPU/GPU comparisons, and
post-learning resume checks. TensorBoard training summaries include local and
cross-rank trust, threshold, eligibility, actual cumulative skip fraction,
actor/critic update counts, returns, environment progress, and timing.

The selected experiment grid is dog:walk with UTD 11 at 75k and 105k, and
UTD 3 at 105k. Each combination uses seeds 0 and 1 with skipping enabled:
six continuations. Existing BAFCv3 results serve as the baseline. UTD-3/75k
and all skipping-off arms are excluded from the launcher; existing directories
are retained. Relative to the UTD-11 BAFCv3 baseline, UTD-3 runs change both the
update schedule and skipping, so that comparison does not isolate skipping.
Validation findings
and prepared commands are linked in the [validation report](bafcv3_tr2_restart_validation.md). Full
training can be launched from the repository root with:

```bash
bash alf/examples/run_dog_walk_bafcv3_tr2_restart_6jobs-4g.sh
```

The launcher starts all six jobs concurrently on GPUs 0–3 (six training ranks
per GPU). Every invocation starts fresh from the selected original BAFCv3
checkpoints, with fresh calibration, under
`/workspace/alf_results/dog_walk/bafcv3_tr2_restart/<UTC-run-id>/<job>/`.
Each job has its own `out.log`; the batch directory stores its manifest,
preparation logs and process IDs in `launches.tsv`. After preparation, the
launcher starts all six jobs with `nohup` and exits without waiting for training
completion. Each job keeps running independently and writes to its own log;
there is no batch completion-status file. Existing study directories
are rejected. Use `--dry-run` to preview without creating or launching runs,
`--dir` to override the results base, and `--run-id` to set a unique study ID.
CPU thread counts default to one and remain configurable through
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`.
A lock prevents overlapping preparation/launch operations under the same results
base; it is released when the launcher exits. Training processes do not inherit
the lock.
The original study's `commands.sh` forwards to this launcher. Although training
uses `--resume` after preparing each empty directory, it resumes only the new
configuration; it migrates the original BAFCv3 checkpoint, not a stopped TR2 run.

The earlier six runs were checkpointed and stopped on 2026-09-17. They remain
under `/root/alf/artifacts/bafcv3_tr2_restart/20260917T050003Z/experiments/`;
no copied runs are retained under `/workspace`. The new launcher has been
syntax-checked and dry-run verified; the fresh batch has not been launched.
Startup tests establish restoration and execution
correctness; they do not establish that delayed skipping or lower UTD improves
returns.

Run focused tests with:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  alf.utils.bafcv3_restart_test \
  alf.algorithms.bafc_algorithm_v3_tr2_test \
  alf.bin.calibrate_bafcv3_tr2_thresholds_test
```
