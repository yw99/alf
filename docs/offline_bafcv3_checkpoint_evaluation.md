# Offline BAFCv3 checkpoint evaluation

Run from the repository root:

```sh
.venv/bin/python -m alf.bin.evaluate_bafcv3_checkpoints \
  --source-glob '/workspace/server*_copy' \
  --output /root/alf/artifacts/bafcv3_offline_eval/my-study \
  --device auto
```

The evaluator reads saved configurations without creating environments, restores
BAFCv3 models and frozen observation normalization, and delegates feature
extraction and trust calculations to the current TR2 implementation. It does not
train, collect transitions, or change source checkpoints. Each checkpoint loads
in its own process to isolate ALF's global configuration registry.

Default coverage measurements use 20 repetitions, behavior sample counts 128,
512, and 2048, and ridge values 1e-5, 1e-4, and 1e-3. Critic diagnostics use up
to 2048 replay starts. Available CUDA is used for forward passes; normalization
stays on CPU because ALF averagers contain non-buffer tensor constants. Spectral
diagnostics use CPU float64. `--device cpu` runs entirely on CPU.

Useful options:

- `--task dog:walk --seed-filter 0`: select a task and seed; flags are repeatable.
- `--checkpoint 15015`: select a checkpoint filename suffix, not environment steps.
- `--repetitions 2 --critic-samples 32`: inexpensive smoke test.
- `--sample-counts 128 512 --ridges 0.0001`: restrict sensitivity settings.
- `--resume --output EXISTING_DIRECTORY`: reuse results only when input hashes,
  settings, and evaluator source identity match.
- `--summarize-only --output EXISTING_DIRECTORY`: regenerate CSVs, plots and
  report from saved raw results; event curves are read again.
- `--no-rescan`: disable the final discovery pass for newly completed server copies.

All source tasks/seeds are discovered; empty BAFCv3 directories, missing shards,
and errors are recorded. A missing optimizer does not block evaluation. The
final rescan is finite; it does not wait indefinitely for ongoing transfers.

## Interpretation

The raw metric is mean squared feature leverage, not a probability or calibrated
critic-error estimate. Even under matched distributions, it depends on effective
feature dimension. The held-out behavior baseline exposes the cost of estimating
a covariance from a finite sample. Held-out means disjoint from the diagnostic
covariance sample, **not** unseen during the original model's training.

The two target-state protocols are kept separate:

1. `saved_cache`: the normalized cache actually saved in the rank-0 checkpoint.
2. `reconstructed_cache`: up to 512 recent observations per replay rank, normalized
   with shared checkpoint statistics. For batched environments, recent observations
   are taken evenly across environments. This is not recovered historical rank-local
   simulator or normalization state.

Neither protocol recovers a historical TR2 reference actor or critic snapshot:
current checkpoint actors/critics are used. The actor encoding must use trainable
saved evaluation samples. Unsupported configurations are recorded as errors rather
than approximated silently. The current adapter supports vector DM Control
observations, scalar rewards, ObservationNormalizer, and [-1, 1] actions.

TD residuals follow the saved random target-critic selection and next-step reward
and discount indexing. The 32-step proxy follows replay behavior for intermediate
actions, bootstraps with the saved target critics/current actors at its endpoint,
and stops at episode boundaries. A terminal discount of zero removes bootstrapping;
time-limit truncation retains it. Neither is an independent current-policy value
error. Q RMS helps interpret changes in residual scale.

Threshold-pass probabilities are counterfactual snapshots. They exclude feedback,
refresh intervals and skip caps. `cross_rank.csv` requires every rank observed in
the run's shard inventory; it uses reconstructed caches and records the maximum
rank metric needed for an all-ranks skip decision. Missing undiscovered ranks
cannot be inferred from checkpoint contents alone.

Historical TR2 event logs, when present, are stored separately. Actual windowed
skip fractions use differences of cumulative skip and opportunity counters,
aligned by global step and associated with the latest available environment-step
measurement. Reset or inconsistent counter windows are excluded. A window crossing
a horizon boundary is assigned by its end step, so early/late fractions have
logging-resolution uncertainty.

## Outputs

- `manifest.json`: inventory, statuses, code/input hashes, settings and invocation.
- `checkpoints/*.json.gz`: repetition-level, per-ensemble measurements, covariance
  spectra, critic residual summaries, runtime/device/software metadata and errors.
- `summary.csv`: per-checkpoint/rank/protocol/sample-size/ridge statistics.
- `checkpoint_means.csv`, `task_summary.csv`: rank means followed by equal-seed
  summaries at matching checkpoint horizons. SD across repetitions and SD across
  seeds describe different sources of variation and are never pooled.
- `cross_rank.csv`, `sensitivity.csv`: rank aggregation and estimator sensitivity.
- `curves.json`, `tr2_summary.csv`, `tr2_skip_windows.csv`: original learning curves
  and historical TR2 timing where available.
- `restart_candidates.csv`: retrospective candidate windows; stability is a
  three-checkpoint relative range <=20%, improvement is positive recorded return
  gain at either of the next two checkpoints. This rule is exploratory, not a
  convergence test or an experimentally validated activation schedule.
- `figures/*.png`, `figures/*.pdf`, `report.md`: shareable figures and findings.

Validation:

```sh
.venv/bin/python -m unittest alf.bin.evaluate_bafcv3_checkpoints_test
BAFC_CHECKPOINT_TEST=/workspace/server2_copy/dog_bafcv3_s0/train/algorithm/ckpt-15015 \
  .venv/bin/python -m unittest alf.bin.evaluate_bafcv3_checkpoints_test
```

The optional real-checkpoint test compares forward values and TD targets against
BAFCv3's own training method, checks CPU/GPU feature agreement and verifies that
model and normalization state remain unchanged.

For an existing study, use one worker per selected GPU while preserving completed
results and evaluator settings:

```sh
.venv/bin/python -m alf.bin.run_bafcv3_checkpoint_sweep \
  --output /root/alf/artifacts/bafcv3_offline_eval/my-study --gpus 0 1 2 3
```

Stop the original scheduler after its in-flight checkpoint has finished before
starting this scheduler. Physical GPU assignments are saved in `assignments/`;
each isolated worker sees its assigned GPU as logical `cuda:0`.

Finite, nearly rank-deficient covariance matrices occasionally cause LAPACK's
lower-triangle eigensolver to fail. The evaluator retries the upper triangle of
the same matrix and records the solver path; samples and ridge are unchanged.
Nonfinite covariance inputs are explicit errors.

Configuration variants include optimizer and trainer settings as well as network
settings; seed and output path are excluded. Checkpoint rank means have a separate
`sampling_sd_of_rank_mean` column, calculated from independently seeded rank
sampling. It is not variation across trained seeds.

## Quantile calibration for conservative TR2 restarts

The separate calibration command selects a threshold from already measured
cross-rank distributions. It uses `--threshold-quantile 0.33` by default; the
quantile can be changed without rerunning checkpoint inference:

```sh
.venv/bin/python -m alf.bin.calibrate_bafcv3_tr2_thresholds \
  --study /root/alf/artifacts/bafcv3_offline_eval/20260917_full \
  --task dog:walk --seed-filter 0 1 --checkpoint 75075 105105 \
  --threshold-quantile 0.33
```

For each seed/checkpoint, take the maximum rank metric within each repetition,
then the requested quantile across repetitions. This matches the strict
`rollout_skip_sync_mode="min"` rule: every rank must pass the threshold. The
quantile of rank maxima is generally different from the maximum of independently
computed rank quantiles.

A lower quantile gives a lower or equal threshold. It targets fewer initial
metric-threshold passes, not a particular realized rollout-skip fraction. Finite
samples and ties can make the empirical pass fraction differ from the requested
quantile; both are saved. Use the same quantile rule across all experimental
arms, freeze each resulting threshold during continuation, disable threshold
decay, and retain the existing skip cap.

Each invocation creates a new timestamped directory under `STUDY/calibrations/`
(or a new directory specified by `--output`). It writes `thresholds.csv` and
`calibration.json`, including the selection settings and hashes of the source
measurements and calibration code. Existing study outputs are not overwritten.
Incomplete ranks, missing repetitions and missing explicitly requested checkpoint
combinations are rejected.

These are **preliminary offline thresholds**. This command does not implement
automatic startup calibration in TR2, change training configuration, or launch
training. Before the restart experiment, calibrate with frozen networks using
the actual resumed TR2 metric path and its sampling/cache settings (preferably
around 200 repetitions). The resulting `eval_trust_max` is checkpoint-specific;
`threshold-quantile` is the shared experimental input parameter. Comparisons with
historical fixed-threshold runs also change the threshold policy.
