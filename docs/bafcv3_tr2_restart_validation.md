# BAFCv3 → TR2 restart validation

Validation completed on 2026-09-17 UTC using four RTX 5090 GPUs,
PyTorch 2.7.1+cu128, deterministic Torch algorithms,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and evaluator version 2.

**All four selected source checkpoints passed migration, calibration, bounded
GPU startup, and exact save/resume checks.** The implementation and prepared
commands are available. Full training was not launched during validation;
the six selected continuations were subsequently launched concurrently at the
user's request. Current launch information is in `experiment_manifest.json`
and `launch_status.json` in the study directory.

## Calibrated thresholds

All values below use 100 native TR2 repetitions, quantile 0.33, ridge 1e-4,
128 observations, four replay ranks, and the **rank maximum before quantile**.
Task is dog:walk. Times measure calibration across four concurrently active
ranks, excluding checkpoint I/O and validation setup.

| Seed | Source logged environment steps per rank | Fixed threshold | Calibration seconds | Validation |
|---|---:|---:|---:|---|
| 0 | 75,000 | 310.476154 | 9.79 | Passed |
| 1 | 75,000 | 159.029926 | 9.01 | Passed |
| 0 | 105,000 | 336.955246 | 9.99 | Passed |
| 1 | 105,000 | 239.712159 | 9.49 | Passed |

Each calibration has sampled eligibility 0.33. This is not a prediction of the
realized rollout-skip rate. Seed 1 changes substantially between these horizons;
calibrating each restart separately is appropriate for the agreed experiment.
These thresholds use the native online minibatch path and must not be substituted
with earlier offline-evaluator values from different sampling protocols.

The additional seed-0/75k UTD-11, skipping-off control passed with threshold
310.476154. Its entire 100 × 4 calibration measurement array is
**exactly equal** to the UTD-3, skipping-on arm. This verifies that the initial
calibration is independent of these experiment settings on this source.

## What was verified

- Exact restoration of saved tensors and optimizer states, with parameter-name
  mapping; original target critics are retained separately from new snapshots.
- All four replay partitions, positions, saved step types, normalization state,
  evaluation samples, cumulative counters, trainer progress, and global step.
- Zero optimizer updates and zero environment steps during migration/calibration.
  Calibration checks frozen model, optimizer, normalizer, and replay state.
- Fresh native resume preserves the calibrated initial state without recalibration.
- Per rank: two real environment steps, then two bounded replay-learning iterations.
  UTD 3 performs 18 critic and 6 actor updates; UTD 11 performs 22 and 2.
  Both actor and critic parameters stay synchronized across all four ranks.
- A separate post-learning checkpoint restores learner, optimizer, rank-local
  normalizer/cache/controller state, and RNG exactly. A new simulator boundary is
  added to replay as intended. Pristine initial checkpoints remain unchanged.
- Identical-input CPU/GPU metric comparisons passed on all 20 validated ranks.
  Maximum observed relative difference was 2.86e-05; the predetermined
  tolerance was relative 0.005 plus absolute 0.001. Random target sampling is fixed
  explicitly for this comparison because CPU and CUDA RNG algorithms differ.
- Every source fingerprint remained unchanged after validation.

The four-rank startup validations and the additional control took
208.7 seconds in total, including
process startup, migration, checkpoint writes, real rollout, learning, and resume.

An additional production-path probe used `alf.bin.train`'s actual mapped workers,
ALF's native parallel environment, and `RLTrainer.train()`, with only its main
training loop bounded to two iterations. All four workers passed with 18 critic
and 6 actor updates. This check also exercised the all-rank gate through actual
`train_iter()` calls. Its calibrated threshold matched the validation worker.
The probe is recorded in `production_startup.json` and `production_startup.log`;
its separate smoke output directory is named in that JSON and is not an experiment
arm. The CLI now supplies the deterministic CuBLAS workspace and puts the Python
environment's companion tools on the worker PATH, so Ninja is found when ALF
builds its native environment extension.

## Tests

The focused suite passed **84 tests**: migration helpers, optimizer name mapping,
UTD phase handling, complete BAFCv3/TR2 UTD-11 fixed-minibatch update agreement,
replay wraparound/reset boundaries, next-reward and terminal/time-limit preservation,
RNG isolation, gate-disabled metric monitoring, quantile aggregation, configuration
invalidation, atomic initial save, existing TR2 behavior, and calibration utilities.

The extended checkpoint suite added 12 tests: 11 passed; the existing
`TestWithCycle.test_with_cycle` fails its hard-coded optimizer-state expectation.
The identical failure was reproduced with the unchanged `HEAD` version of
`checkpoint_utils.py`. Logs are preserved; this is not counted as a passing test.
The installed Adam defaults include `decoupled_weight_decay`, which that fixture
omits. The fixture was left unchanged.

## Durable outputs and experiment commands

The launch grid was reduced at the user's request by excluding UTD 3 at 75k
and all skipping-off continuations. Existing BAFCv3 results serve as the baseline.
The original 16-arm manifest is archived as `experiment_manifest.original16.json`;
existing validation results and experiment directories are retained.

Study directory: `/root/alf/artifacts/bafcv3_tr2_restart/20260917T050003Z`.

- [Experiment manifest](/root/alf/artifacts/bafcv3_tr2_restart/20260917T050003Z/experiment_manifest.json): six selected arms, resolved
  settings fingerprints, output directories, and argument arrays.
- [Prepared commands](/root/alf/artifacts/bafcv3_tr2_restart/20260917T050003Z/commands.sh): UTD 11 at 75k and 105k, and UTD 3 at 105k,
  each with seeds 0 and 1 and skipping enabled (six jobs). Executing the script launches full continuations
  concurrently, each with four ranks on GPUs 0–3. The original batch was launched, then checkpointed and stopped on 2026-09-17;
  its archived logs remain under `experiments/<name>/out.log`. The command now
  forwards to the fresh launcher described in the usage instructions, writing
  new timestamped runs under `/workspace/alf_results` without copying old runs.
- [Validation summary](/root/alf/artifacts/bafcv3_tr2_restart/20260917T050003Z/validation_summary.csv) and per-run JSON files.
- Each validation run contains source/configuration/code snapshots, hashes,
  migration audits, repetition-level calibration, timings, initial checkpoints,
  separate smoke checkpoints, and rank-level resume/CPU-GPU validation results.
- Prepared experiment directories are under `experiments/`. Commands use
  `--resume` to continue their prepared configuration; initial calibration occurs
  on first launch. All finish at absolute 150k per-rank environment steps.

The archived validation and stopped-run files are under Git-ignored `/artifacts/`.
The fresh batch has not been launched. Stop evidence is in
`stop_checkpoint_requests.json` and `stop_summary.json` in the original study directory. The usage instructions
are in [bafcv3_tr2_restart.md](/root/alf/docs/bafcv3_tr2_restart.md).

## Findings and limits

Four-rank continuation is operational for the selected inputs. Key fixes required
explicit optimizer materialization, full actor/critic registration with DDP,
phase-specific training-info handling, preserving replay step types and restart
boundaries, and stripping new live trust telemetry from the legacy replay schema.
Gate-disabled controls compute and aggregate the same real trust metric.

Faithfulness is limited to saved state. Historical simulator and RNG state were
not saved. Rank 0's normalized cache is retained; other caches are reconstructed
from each rank's replay using the checkpoint's shared normalizer. Historical
rank-local normalization cannot be recovered. Fresh episodes and new deterministic
training RNG streams are recorded in migration audits. Future TR2 checkpoints
save rank-local normalizer and RNG state explicitly.

The smoke tests deliberately do not estimate return improvement or realized skip
fractions. Whether delayed skipping helps, whether skips previously concentrated
early, and whether UTD 3 improves the actor remain questions for the six selected continuations. Against the existing UTD-11 BAFCv3
baseline, UTD-3 runs change both the update schedule and rollout skipping; that
comparison cannot isolate the effect of skipping alone. Compare returns at matched environment steps and report updates,
skip fractions, and wall time separately.
