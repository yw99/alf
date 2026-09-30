# Offline actor encoding distances

From the repository root:

```sh
.venv/bin/python alf/utils/plot_actor_encoding_distances.py
```

Defaults select the eight BAFCv3 runs (four seeds each for `dog:stand` and
`dog:run`) used by `plot_dog_humanoid_bafc_comparison.py`. All numbered model
checkpoints are analyzed. The x-axis uses each checkpoint's saved trainer
environment-step counter, **not** its filename suffix. No environment, replay
buffer, optimizer checkpoint, or training is needed. Source runs are read-only.
The existing offline evaluator restores saved configuration and model state in
an isolated process per checkpoint. Current support is BAFCv3 vector DM Control
with saved trainable or frozen encoder probes and exactly 10 actors; other configurations
fail explicitly in the manifest/logs.

```sh
# Select another run and a fresh destination.
.venv/bin/python alf/utils/plot_actor_encoding_distances.py \
  --run /workspace/server2_copy/dog_stand_bafcv3_rtT_s0 \
  --output artifacts/encoding_stand_seed0 --workers 2

# Replot saved measurements without loading any checkpoints.
.venv/bin/python alf/utils/plot_actor_encoding_distances.py \
  --output artifacts/actor_encoding_distances --summarize-only

.venv/bin/python alf/utils/plot_actor_encoding_distances_test.py
```

The output directory must be new for an inference run, preventing accidental
mixing with earlier measurements. `--run` is repeatable. Workers run on CPU;
`--cpu-threads` defaults to two per worker.

Outputs:

- `figures/overview.png` and `.pdf`: one row per task, with Euclidean encoding
  distance, cosine encoding distance, and actor action RMS difference. Each
  seed has its own line (mean over 45 pairs) and band (pair minimum–maximum).
  Encoding-distance axes use a symmetric log scale, linear below 1e-14.
- Per-run PNG/PDF: all 45 pair trajectories with a black mean; first/last
  cosine-distance heatmaps; the norm of each of the ten actor encodings.
- `pairs.csv`: all 45 unordered pairs at each checkpoint, with stable actor IDs.
- `summary.csv`: pair mean, minimum and maximum at each checkpoint.
- `raw/*.npz`: full 10×10 distance matrices, encoding vectors and norms,
  task/seed/step/configuration metadata, input hashes, and a model-state
  immutability check. Undefined zero-vector cosine distances are NaN.
- `manifest.json`, `logs/`, `report.md`: selected inputs, failures, provenance,
  per-checkpoint logs, endpoint measurements, and interpretation.

Euclidean distance measures absolute separation. Cosine distance measures
angular separation (0 means aligned, 1 orthogonal, 2 opposite); norms help
identify changes in scale. Self-pairs are excluded from every summary.
Action RMS is computed over all saved probes and action coordinates for each
actor pair, in the actor's [-1, 1] action units.

Each checkpoint uses its **own saved evaluation samples**, exactly
as its functional actor encoder does. These synthetic probes are shared by
all ten actors at that checkpoint. Trainable probes can change during training; frozen probes should remain constant. Thus the
curves measure the representations actually used at each checkpoint; they
do not isolate encoder learning from actor/probe learning. They do not measure
cross-checkpoint vector distances or align latent coordinate systems.

Separation establishes distinguishable representations, not causal usefulness.
Action differences on encoder probes are not a visitation-weighted behavioral
metric. To establish that representation improves control, follow up with a
controlled actor-ID/collapsed-encoding training ablation and return evaluation;
a critic encoding-swap experiment can first check its sensitivity to identity.

## Fresh initialization versus 20k

```sh
.venv/bin/python alf/utils/compare_actor_encoding_initialization.py
```

This companion script reconstructs fresh actors, encoder and trainable probes
from each of the same eight saved run configurations. It resets Python, NumPy
and PyTorch RNGs to the configured seed immediately before direct BAFCv3
construction on CPU. It repeats construction and verifies identical model
fingerprints and metrics, then loads the run's `ckpt-20020` (asserting 20k saved
environment steps) into the same architecture and measures the trained model.
No environment or training is run.

The default fresh output directory is `artifacts/actor_encoding_initialization`.
It contains a PNG/PDF comparison, all 45 pairs in `pairs.csv`, `summary.csv`,
raw initial/checkpoint encodings and distance matrices, input/source hashes,
repeatability checks, logs, and `report.md`. `--run` is repeatable; `--output`
and `--workers` can be changed. `--summarize-only` regenerates the comparison.

These initializations are reproducible experiments, **not recovered historical
step-zero weights**. The original trainer's RNG consumption, device and code
version may differ. Both stages use their own probes. The plot connects fresh
initialization and the saved 20k observation; it does not identify when between
those points collapse developed in the historical runs.

## Frozen versus trainable probes

Pass the four frozen `seed_*` directories under
`/workspace/h200_copy/dog_stand/bafcv3_seed0123_8g_4rank_frozen_eval_samples/fixed_pairingFalse_num_sampled_critic8/critic_utd11`
as repeated `--run` arguments to `plot_actor_encoding_distances.py`, with
`--output artifacts/actor_encoding_distances_frozen`.

Then run:

```sh
.venv/bin/python alf/utils/compare_frozen_actor_encodings.py
```

The comparison reads the existing trainable study and the new frozen study,
selects dog:stand seeds 0–3, and uses the exact checkpoint-step intersection.
It verifies that frozen probe hashes are identical across all checkpoints in
each seed. Raw new measurements record the probe source and probe tensor hash.
The overview averages pair means equally across seeds and shades the range
of seed means. Per-seed plots shade the range of all 45 actor-pair distances.
Artifacts are written to `artifacts/dog_stand_frozen_vs_trainable`.

The shared checkpoint loader accepts frozen probes only when explicitly called
with `allow_frozen_eval_samples=True`; the existing trust evaluator retains
its trainable-only default. Replay-sourced probes remain unsupported here.
