# BAFCv3 → BAFCv6 checkpoint restart

This opt-in pipeline resumes the saved BAFCv3 learner and enables BAFCv6 critic
sample reweighting immediately. The first resumed critic update is weighted;
there is no calibration, rollout skipping, or additional warm-up. Existing
BAFCv3, BAFCv6, and TR2 entry points retain their behavior.

## Supported configuration

V1 supports four-rank vector DM Control BAFCv3 Agent checkpoints with trainable
actor evaluation samples, uniform replay, an ObservationNormalizer, a target
update period of 1, and the source schedule of 11 critic updates and 1 actor
update per 12-update iteration. Other schemas fail explicitly. The source model,
optimizer sidecar, saved configuration, and replay shards for ranks 0–3 are
required. No missing replay rank falls back to rank 0.

Networks, target critics, encoder, evaluation samples, optimizer moments and
parameter groups, normalization, metrics, absolute progress, and update counters
are restored and checked. V6 reference actors and snapshot critics are initialized
from the checkpoint's online actors and critics and subsequently follow V6's
normal refresh schedule. Reweighting does not change actor pairing, random
critic targets, learning rates, or UTD.

Default reweighting matches the existing dog:run V6 launcher:

| Setting | Value |
|---|---|
| Solver | `lbfgs_logits` |
| Solver iterations | 1 |
| Feature coordinates | 32 |
| Target observations / cache size | 128 / 512 |
| Maximum weight | 10 |
| Ridge | `1e-4` |
| Beta | `None` (V6's automatic formula) |

The CLI exposes these as `--critic-reweighting-*` options. It deliberately has no
UTD override. `--final-env-steps-per-rank` is an **absolute** budget (default
200000), not an additional number of steps. The config passes 800000 aggregate
steps to ALF, which divides this once across four ranks.

## Prepare, validate, and resume one run

Run commands from the repository root. Choose a fresh destination for each
experiment; validation outputs should be separate from production outputs.

```bash
.venv/bin/python -m alf.bin.train_bafcv6_restart \
  --source-checkpoint /workspace/server2_copy/dog_run_bafcv3_rtT_s0/train/algorithm/ckpt-120120 \
  --root-dir /root/alf/artifacts/my_v6_validation \
  --validate-only --validation-device cpu \
  --smoke-train-iters 1 --smoke-env-steps 2
```

CPU validation uses four Gloo ranks. GPU validation uses the same four isolated
GPU workers as training; use `--validation-device cuda --worker-gpus 0,1,2,3` and
a fresh destination. Smoke options are validation-only and accept 0–2 iterations
and 0–2 real environment steps. One smoke iteration executes 11 critic updates
and 1 actor update. Validation checks finite weights/diagnostics, synchronized
actor/critic parameters, and exact post-learning resume. It saves smoke state
separately from the pristine migrated checkpoint.

`--prepare-only` writes configuration/provenance without constructing or training
the learner. To start a prepared run, repeat its original options with `--resume`
and without `--prepare-only`. To resume a stopped V6 run, use the same command with
`--resume`. Preparation checks input and configuration fingerprints and rejects
changed settings. A new output must be empty and separate from the source run.

## Four-job dog:run launcher

```bash
bash alf/examples/run_dog_run_bafcv6_restart_6jobs-4g.sh --dry-run
bash alf/examples/run_dog_run_bafcv6_restart_6jobs-4g.sh
```

The second command starts four concurrent jobs, each using all four configured
GPUs. It selects seeds 0/1 and checkpoints `ckpt-120120` and `ckpt-140140`, then
verifies their actual environment steps (120k/140k), and ends
at 200k per rank. Sources default to
`/workspace/server2_copy/dog_run_bafcv3_rtT_s{seed}`.

Options: `--dir`, `--run-id`, `--gpus`, `--source-base-dir`, and `--dry-run`.
Results default to `/workspace/alf_results/dog_run/bafcv6_restart/<run-id>/`.
The launcher uses 64 target observations with a 512-observation cache.
All four inputs are preflighted and all manifests prepared before any launch.
Dry-run only prints commands and requires the selected source model paths to
exist; it does not create files or inspect their tensor payloads. Full preflight
also checks optimizer/replay shards and saved task/seed configuration.

Each job has `out.log`; the study has `launches.tsv` and an experiment manifest.
The launcher detaches jobs and returns. Fresh destinations prevent accidental
reuse. Distributed workers obtain their rendezvous ports through ALF's worker
launcher; each job's optional HTTP control port is allocated by the OS.

## Checkpoint contract and limits

Each run stores `restart_manifest.json`, source/configuration/code snapshots,
input hashes, `resolved_config.json`, per-rank migration audits, and
`restart_ready.json`. The initial checkpoint retains the source global step and
is published only after every rank's sidecars are saved. It records the activation
environment step, pipeline version, and settings fingerprint. Native V6 resume
requires both new networks and all optimizer, replay, and rank-state files; it
never synthesizes missing native V6 networks.

Rank-local sidecars preserve runtime/cache, normalization, and Python, NumPy,
Torch, and CUDA RNG state. Native resume restores this state without migration.
The source BAFCv3 checkpoints did not save historical simulator/RNG or every
rank's normalizer/cache. Migration therefore starts fresh episodes and new
reproducible rank-specific RNG streams (`--restart-seed`, default 20260917).
Rank 0 retains its saved normalized cache; other ranks rebuild caches from their
own replay using the shared saved normalizer. These limits are recorded in audits.

Replay step types and contents are preserved. Both migration and later simulator
restarts mark boundaries so replay sampling excludes invented transitions across
the restart. Exact resume means exact saved learner/rank-local state, not a
bitwise continuation of an unsaved simulator episode.

## Validation evidence

Initial implementation validation artifacts are in
`artifacts/bafcv6_restart_validation/` (Git-ignored):

- 71 regression tests passed: 63 V6/TR2/restart tests and all 8 existing trainer tests.
- All six real seed/checkpoint combinations passed four-rank CPU migration and
  initial native-resume equality checks, with source input hashes unchanged.
- `dog_run_s0_120k` additionally passed two real environment steps per rank,
  one complete 11/1 replay-training cycle with 11 reweighting calls per rank,
  finite diagnostics, synchronized actor/critic parameters, and exact
  post-learning learner/rank-local resume.
- `validation.json` and `validation_rank{0,1,2,3}.json` record actual checks;
  `validation_smoke/` holds the separate post-learning checkpoint.
- GPU validation could not run in this session: NVML initialization failed and
  PyTorch reported `cuda.is_available() == False` and zero devices. CPU results
  do not certify CUDA execution. Repeat the GPU command above when CUDA is
  accessible. Full six-job training was not launched.

Regression commands:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  alf.utils.bafcv6_restart_test \
  alf.utils.bafcv3_restart_test \
  alf.algorithms.bafc_algorithm_v6_test \
  alf.trainers.policy_trainer_test
bash -n alf/examples/run_dog_run_bafcv6_restart_6jobs-4g.sh
```

Tests cover V3/V6 equivalence with reweighting disabled through a full 11/1 cycle,
first-update activation, nonuniform weighted loss, optimizer remapping, replay
boundaries, native metadata/RNG, missing-shard collective failures, ordinary
trainer fallback, TR2 compatibility, and the four-command dry-run grid.
