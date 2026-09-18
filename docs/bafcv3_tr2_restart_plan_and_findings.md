# BAFCv3 → TR2 restart implementation plan and findings

Date: 2026-09-17 UTC.

> Historical reboot handoff. Implementation subsequently resumed with GPUs
> available. See [current implementation and usage](bafcv3_tr2_restart.md).
> The rollback status below describes the earlier interruption only.

## Shutdown and rollback status

Implementation was stopped at the user's request for a server reboot. The active
validation parent and all four workers were terminated; a process check found no
remaining restart validation or test workers. No full training continuations were
launched.

All active implementation changes from this turn were reverted: `agent.py`,
`bafc_algorithm_v3_tr2.py`, `policy_trainer.py`, and `checkpoint_utils.py` were
restored to their original repository versions. The new restart CLI, helper,
tests, temporary usage document, and this turn's `.gitignore` addition were
removed. **The implementation described below is not installed.**

The pre-existing offline evaluator, threshold calibration utility, their tests,
offline evaluation documentation, prior `.gitignore` change, and earlier study
results were preserved. Local diagnostic artifacts from this attempted
implementation were also retained. Any experimental code snapshots inside those
artifacts are historical evidence, not accepted or active implementation.

## Agreed experiment and public interface

Implement `python -m alf.bin.train_bafcv3_tr2_restart` using ALF's existing
four-GPU launcher. Required inputs are `--source-checkpoint` (an exact model file)
and `--root-dir` (a separate output directory). Parameters:

| Input | Default / behavior |
|---|---|
| `--critic-utd` | 3; support 11 controls |
| `--threshold-quantile` | 0.33; finite, within [0,1] |
| `--calibration-repetitions` | 100, as selected by user |
| `--calibration-seed` | Fixed recorded seed; prototype used 20260917 |
| `--rollout-skipping` | on; also support off controls |
| `--final-env-steps-per-rank` | Absolute final horizon 150000 |
| `--worker-gpus` | 0,1,2,3 |
| `--validate-only` | Restore, calibrate, save and check resume without learning |
| `--resume` | Continue matching saved restart; never recalibrate |
| `--prepare-only` | Save configuration/provenance without launching |
| `--validation-device` | CPU or CUDA; CPU permits validation without GPUs |
| `--smoke-train-iters` | Optional bounded replay-only validation, at most 2 |

Infer task, seed, architecture, optimizer settings and preprocessing from saved
configuration. Preserve distinct variants. Reject unsupported configurations and
missing state rather than silently falling back.

Selected inputs are dog:walk, seeds 0 and 1:

| Seed | Logged per-rank environment step | Model file |
|---|---:|---|
| 0 | 75000 | `/workspace/server2_copy/dog_bafcv3_s0/train/algorithm/ckpt-75075` |
| 1 | 75000 | `/workspace/server2_copy/dog_bafcv3_s1/train/algorithm/ckpt-75075` |
| 0 | 105000 | `/workspace/server2_copy/dog_bafcv3_s0/train/algorithm/ckpt-105105` |
| 1 | 105000 | `/workspace/server2_copy/dog_bafcv3_s1/train/algorithm/ckpt-105105` |

All four selected checkpoints have model, optimizer and all four replay shards.
Do not restart at 30k and wait until the chosen horizon; load the selected
checkpoint directly. Prepare commands/manifest for these four inputs × UTD
{3,11} × skipping {off,on}: 16 continuations. Full launches require a subsequent
instruction.

Keep actor UTD 1 and 12 update slots per iteration. UTD 3 gives nine critic and
three actor updates, grouped into three cycles. UTD 11 gives eleven critic and
one actor update. Require complete cycles rather than silently rounding down.

ALF's `adjust_config_by_multi_process_divider()` divides the configured
`num_env_steps` by world size. Thus 150k per rank means 600k in the source-style
configuration, divided exactly once. Preserve checkpoint/global-step and logged
environment-step units separately.

## Restoration design and code findings

1. **No preparatory training iteration.** `Trainer._restore_checkpoint()` currently
   calls `train_iter()` to materialize lazy objects before loading. Add an opt-in
   restart dispatch that creates replay schema/optimizers without stepping the
   environment or optimizer. Use `torch.load(weights_only=True)` explicitly,
   including native restart model, optimizer and replay loads.
2. **Strict migration.** Restore actor networks, trainable actor evaluation
   samples, encoder, current and target critics, normalizer, replay tensors and
   ring positions, optimizer state, saved runtime counters, metrics and trainer
   progress. Validate names, shapes and dtypes. Only explicitly identified TR2
   additions may be initialized. Do not broadly load with `strict=False`.
3. **Optimizer mapping.** Source Adam state contains parameter IDs, not names:
   the inspected source has an empty first group and 77 parameters/states in the
   second. Construct the source BAFC Agent/optimizer using its saved config,
   recover the ID-to-name mapping from actual groups, then map names into TR2.
   Preserve moments, step counters and group settings; verify exact equality.
   `copy.deepcopy(optimizer)` loses ALF wrapper attributes through PyTorch's
   optimizer serialization hooks. The prototype cloned a pristine optimizer's
   full `__dict__` instead; this was tested. Do not share mutable optimizer
   instances between source, destination or repeated validation constructions.
4. **New networks.** Initialize TR2 reference actors and snapshot critics from
   the saved current networks. Preserve the separately saved target critic.
5. **Phase and counters.** Preserve the source phase for unchanged UTD. For a
   changed UTD, introduce a separately checkpointed critic-phase offset and begin
   a full critic block without resetting cumulative update counters. Avoid the
   special first-ever joint actor/critic branch triggered by zero counters.
6. **Lazy train-info structure.** A warm start beginning in critic-only mode
   initially produces empty actor-info leaves. ALF caches that structure and
   subsequently fails at the first actor update. The working prototype refreshed
   the root Agent's train-info specification on each restart training step.
   This avoids an otherwise unnecessary dummy joint update.
7. **DDP parameter registration is essential.** Lazy DDP construction in critic
   mode omits actors whose `requires_grad` is temporarily false. Later actor
   updates can then be unsynchronized without an obvious error. Materialize the
   DDP performer before learning, temporarily enabling gradients for all
   optimizer-owned parameters, then restore phase flags. Use
   `find_unused_parameters=True`. Preserve rank-local buffers across DDP's
   initialization broadcast and isolate RNG. Verify actor weights remain exactly
   equal across all four ranks after actual actor updates.
8. **Replay load changes data by default.** `ReplayBuffer._load_from_state_dict()`
   forces the newest saved step type to LAST. The prototype retained the saved
   step type and represented restart boundaries explicitly as absolute ring
   positions. Reject sampled sequences crossing these boundaries or a FIRST in
   their interior. Keep original rewards and termination/time-limit discounts;
   never fabricate an environment transition. Persist boundary metadata through
   wraparound and resume.
9. **Caches and missing state.** Rank 0's saved BAFC normalized observation cache
   can initialize its TR2 cache. Reconstruct other ranks' latest 512 observations
   with the checkpoint normalizer, labeling this approximation. Historical
   rank-local normalizers, simulator and RNG state were not recovered. Start a
   fresh simulator episode when collection resumes, retain cumulative progress,
   and clear partial-episode state. Save rank-local normalizers as well as TR2
   cache/controller/RNG state in future sidecars.
10. **Target updater.** Its old counter is not present in the inspected BAFC
    runtime. The selected source period is one, so resetting the counter does not
    change cadence. Reject other periods until their restoration is justified.

The attempted migration deliberately supported vector DM Control, non-recurrent
BAFC Agents, trainable evaluation samples, ObservationNormalizer and uniform
non-episodic replay. Other schemas need explicit extensions. Require exactly the
source four-rank layout; do not substitute a rank-0/legacy replay file for a
missing rank.

## Frozen calibration and online behavior

Before any learning or collection, freeze networks and normalization statistics.
For each repetition j, draw a fresh local replay minibatch on each rank, normalize
with saved statistics, and call TR2's native metric path. Calculate

`M[j] = max_rank C[rank,j]`, then `threshold = linear_quantile(M, q)`.

The maximum must precede the quantile. Use 100 repetitions and q=0.33 initially.
Derive random seeds from source model content, calibration seed and rank, never
from UTD or gate status. Save/restore Python, NumPy, CPU and CUDA RNG around
calibration. Hash/check frozen state before and afterward. Broadcast one identical
threshold and metadata to all ranks; fail collectively on invalid inputs or
nonfinite measurements. Nonnegative zero metrics are mathematically valid.

Use 128 metric observations, covariance ridge 1e-4, target cache capacity 512,
and refresh every eight actor updates. The configured local training minibatch
is 64 sequences of length two, so its 128 observations are temporally correlated.
Repetitions can overlap. This sampling differs from the earlier offline study's
independently sampled observations, so do not reuse its numeric thresholds as
final startup thresholds.

Disable skipping until calibration is ready. Freeze the threshold thereafter;
disable decay, actor-extension gating and critic reweighting. Reuse sync mode
`min`, which means all rank skip decisions must agree: its effective scalar
metric is the **rank maximum**. Keep the cap at three consecutive skips and the
existing rollout cadence. A 33% calibration quantile is an initial sampled
eligibility fraction, not a guaranteed realized skip rate.

Gate-off controls must calibrate and monitor identically. Existing TR2 skips
metric computation when its gate is disabled, so an opt-in restart override is
needed. Eight actor updates correspond to more frequent refreshes per training
iteration with UTD 3; document this and report actual update counts and compute.

Persist threshold, quantile, repetition count, source/settings fingerprints,
controller state and rank-local caches. Native resumes must restore them without
recalibration and reject conflicting settings. Stage the initial checkpoint and
publish its discoverable model file after all rank sidecars, then a completion
marker. Do not treat incomplete checkpoints as accepted.

## Validation evidence collected before interruption

Artifacts are under `/root/alf/artifacts/bafcv3_tr2_restart/`.

| Artifact directory | Observed result |
|---|---|
| `dev_s0_75/` | Seed-0 75k checkpoint; two-repetition development calibration; four-rank CPU/Gloo restoration, exact native resume, and two replay-learning iterations passed. Each rank performed 18 critic + 6 actor updates; actor parameters matched across ranks. Completed validation confirmed source hashes unchanged. |
| `validation_s0_75_utd3/` | Full 100-repetition calibration completed; migration audits, initial checkpoint and native resume audits were written. Subsequent bounded learning validation was interrupted. No final `validation.json` exists. Explicitly marked `STOPPED_BEFORE_COMPLETION.json`; not accepted as a completed validation. |
| `stopped_validation_logs/` | Durable copies of development, regression, unit, smoke and interrupted full-calibration logs from `/tmp`. |

The seed-0 75k calibration on four CPU ranks produced:

| Repetitions | q | Threshold | Recorded calibration time |
|---:|---:|---:|---:|
| 2 (development only) | 0.33 | 315.3147604370117 | 2.70 s |
| 100 | 0.33 | 309.1482061767578 | 124.03 s |

Timing includes the calibration function's work, not the complete startup and
smoke-test pipeline. These are prototype CPU measurements, not GPU timings or
experimental learning results. The two-repetition threshold is not a recommended
experimental setting.

Tests observed passing:

- 62 existing TR2 regression tests at an intermediate implementation stage.
- 11 latest restart unit tests, including RNG isolation, optimizer wrapper
  preservation and name remapping, UTD phase persistence, replay wraparound and
  reset boundaries, terminal/truncation discounts, missing-input rejection,
  no-dummy-training dispatch, max-before-quantile, and a complete fixed-minibatch
  11:1 update cycle matching BAFCv3 with random critic selection.
- Seven pre-existing quantile-calibration tests also passed in a combined run
  before the final fixed-minibatch test was expanded.

The complete regression suite was not rerun after every late change. Only the
seed-0 75k input received the completed development smoke test; the other three
selected checkpoints have not passed startup validation in this attempt.

CUDA reported zero usable devices, despite device nodes being present. Therefore
GPU startup, CPU/GPU numerical comparison and GPU training smoke tests were not
performed. No causal conclusion about delayed skipping or UTD can be drawn from
these engineering checks.

## Remaining work after reboot

Reimplement/review the design above; archived snapshots may help but are not a
finished patch. Add CLI resume-invalidation/configuration-isolation tests and
verify copied configuration integrity. Review CPU/CUDA device placement and
collective failure paths. Verify per-rank normalizer persistence and atomic
publication under interruption. Repeat the full regression suite on final code.

Validate all four selected checkpoints with 100 repetitions, exact frozen-state
checks, fresh-object native resumes, and bounded four-rank learning. Include a
UTD-11 gate-off control for the same source to verify identical initial
calibration and phase-preserving restore. Confirm both actor and critic parameter
synchronization and unchanged source fingerprints. Run GPU checks when CUDA is
available; record any remaining blocker explicitly.

Prepare the 16-run manifest and commands, but do not launch sustained training
without a new instruction. Save migration audits, source/config/code hashes,
repetition-level rank values, thresholds, timings and validation results. Log
rank metrics/maxima, eligibility, actual skips, actor/critic update counts,
returns, environment steps and wall time for enabled and disabled gates.

The original offline study remains at
`/root/alf/artifacts/bafcv3_offline_eval/20260917_full/`; its report, curves,
checkpoint measurements and earlier threshold-calibration artifacts were not
modified by this rollback.
