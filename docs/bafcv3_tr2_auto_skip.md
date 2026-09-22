# Automatic TR2 rollout-skip activation

This is a separate experiment arm from `alf.bin.train_bafcv3_tr2_restart`.
It migrates the same BAFCv3 state, starts with the metric-based rollout gate off,
and enables it once. It does not implement automatic deactivation/recovery.
Ordinary TR2 replay-update cadence remains in force while monitoring.

## Dog run launcher

```bash
# Read-only preflight and command listing:
bash alf/examples/run_dog_run_bafcv3_tr2_auto_skip_120k_seed0123-4g.sh --dry-run

# Four concurrent jobs, each using four GPUs:
bash alf/examples/run_dog_run_bafcv3_tr2_auto_skip_120k_seed0123-4g.sh
```

Defaults: seeds 0–3, `ckpt-120120` (120k environment steps per rank), endpoint
200k, critic UTD 11, actor UTD 1. Seeds 0/1 come from `/workspace/server2_copy`;
2/3 from `/workspace/server_copy`. Use `--help` for source-root, seed, GPU,
output-directory, run-ID and horizon overrides. All sources are validated before
preparation; all selected runs are prepared before training starts. Output goes
to a fresh timestamped `/workspace/alf_results/dog_run/bafcv3_tr2_auto_skip/` study.
Existing immediate-skip comparisons are launched separately with the old CLI.

## Direct entrypoint

```bash
.venv/bin/python -m alf.bin.train_bafcv3_tr2_auto_skip \
  --source-checkpoint /workspace/server2_copy/dog_run_bafcv3_rtT_s0/train/algorithm/ckpt-120120 \
  --root-dir /workspace/alf_results/dog_run/my_auto_experiment \
  --worker-gpus 0,1,2,3 --final-env-steps-per-rank 200000
```

The CLI supports `--prepare-only`, `--resume`, `--validate-only`,
`--validation-device cpu|cuda`, and the existing bounded smoke options.
`--rollout-skipping` must be `off`: the controller owns activation.
All `--auto-*` arguments are recorded in the restart manifest, settings
fingerprint, and code snapshot. Resume requires identical settings.

## Rule

Sample rank-zero training `AverageReturn` and cross-rank maximum eval-trust
at 1k environment-step boundaries. History starts after restart, not from
old TensorBoard files. At a boundary use the current observation; if a training
iteration crosses a boundary, carry forward the preceding observation instead
of using a future value. Repeated environment counts do not add samples.

Three 5k return windows give means R0, R1, R2. Define
`gp=(R1-R0)/max(abs(R0),1)` and `gr=(R2-R1)/max(abs(R1),1)`.
A check passes when:

- `-0.02 <= gr <= max(0.02, gp)`;
- the median trust in the last 5k window is no greater than 1.10 times
  its median in the preceding 5k window;
- at least 15k steps of post-restart history exist and absolute progress is
  at least half the endpoint budget.

Two consecutive passing 1k checks activate the gate. Invalid observations
invalidate their windows and reset the streak. Two zero trust medians pass;
a positive median following zero fails. History collection continues during
warmup. The earliest default activation from 120k is approximately 137k,
depending on the first observed environment step.

Configure these with `--auto-sample-steps`, `--auto-window-steps`,
`--auto-warmup-fraction`, `--auto-slowdown-multiplier`, `--auto-return-floor`,
`--auto-max-return-decline`, `--auto-trust-growth-limit`, and
`--auto-consecutive-checks`.

## Activation, logging and resume

Initial restart calibration is preserved for migration compatibility. Activation
runs a new frozen-state calibration on the current policy/replay: 100 draws,
maximum across ranks before the 33rd percentile, deterministic isolated RNG.
The active threshold is then fixed. All ranks activate together; rollout skipping
requires all-rank agreement, with at most **three** consecutive metric skips.
Threshold decay, critic reweighting, and gradient extension remain disabled.

`auto_skip_checks.jsonl` records each 1k sample/check independently of TensorBoard
summary cadence, including return/trust inputs, window statistics, eligibility,
pass flags, streak and activation. `auto_skip_activation.json` records the
trigger and fresh calibration. Monitoring stops after activation. Checkpoints
contain controller history, sampling position, streak and active calibration;
resume does not calibrate again. The checkpoint is authoritative: after a crash,
audit logs can contain a suffix of work that was not checkpointed and repeated
steps after resume.

The historical walk/trot replay validates rule calculations, not causal benefit:
those trajectories already used rollout skipping. Live observations are denser
than historical summary logs and can yield different trigger times.
