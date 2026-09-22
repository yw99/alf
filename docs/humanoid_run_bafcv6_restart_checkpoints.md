# Humanoid Run BAFCv3 restart checkpoints

Sources: `/workspace/server3_copy/humanoid_run_bafcv3_rtT_s{0,1,2,3}`. All 40 checkpoints have four replay ranks. Measurements generated on 2026-09-22.

The table lists **training return / offline trust metric**. Return is the last logged `Metrics_vs_EnvironmentSteps/AverageReturn` at or before the checkpoint; exact log steps are in the CSV. It is not a fresh policy evaluation. The plot uses the complete unsmoothed training curves.

Trust is the TR2 raw mean squared feature-leverage metric, evaluated on reconstructed recent-observation caches using the checkpoint actors and critics. Values average four ranks and five repetitions, with 128 behavior samples and ridge 1e-4. Lower values indicate better feature coverage under this estimator; they are not calibrated critic accuracy or a guarantee that restarting will help. The plot includes the held-out behavior baseline and sampling SD of the rank mean; trust panels use separate y-axis scales.

Steps are per rank; multiply by four for aggregate interactions. Checkpoint suffixes are global training steps, not environment steps.

| Steps/rank | Checkpoint | Seed 0 return / trust | Seed 1 return / trust | Seed 2 return / trust | Seed 3 return / trust |
| ---: | --- | ---: | ---: | ---: | ---: |
| 15k | `ckpt-15015` | 0.8 / 2.03 | 0.9 / 3.60 | 0.6 / 1.39 | 1.1 / 3.37 |
| 30k | `ckpt-30030` | 23.1 / 28.93 | 36.6 / 26.55 | 6.5 / 25.04 | 2.1 / 22.92 |
| 45k | `ckpt-45045` | 79.2 / 13.99 | 74.2 / 34.82 | 73.6 / 10.87 | 63.6 / 15.39 |
| 60k | `ckpt-60060` | 104.3 / 14.13 | 91.3 / 58.86 | 93.7 / 11.52 | 82.3 / 25.41 |
| 75k | `ckpt-75075` | 125.8 / 21.11 | 94.5 / 72.87 | 104.9 / 16.11 | 103.2 / 35.31 |
| 90k | `ckpt-90090` | 139.8 / 30.11 | 106.1 / 95.69 | 110.7 / 19.30 | 113.9 / 37.86 |
| 105k | `ckpt-105105` | 144.1 / 35.69 | 112.6 / 93.82 | 122.9 / 23.98 | 115.6 / 40.62 |
| 120k | `ckpt-120120` | 150.5 / 41.34 | 114.9 / 122.28 | 133.6 / 31.16 | 130.9 / 47.89 |
| 135k | `ckpt-135135` | 169.5 / 44.37 | 120.6 / 117.40 | 144.3 / 32.23 | 135.0 / 49.56 |
| 150k | `ckpt-150150` | 171.3 / 34.99 | 129.3 / 141.20 | 150.5 / 36.79 | 138.6 / 50.61 |

## Curves and full measurements

- [Per-seed return and trust curves](../artifacts/humanoid_run_restart/checkpoint_selection.png)
- [PDF curves](../artifacts/humanoid_run_restart/checkpoint_selection.pdf)
- [Checkpoint metrics, sampling SD, and exact return-log steps](../artifacts/humanoid_run_restart/checkpoint_means.csv)
- [Raw evaluation report and provenance](../artifacts/humanoid_run_restart/report.md)

The artifacts are local generated files, excluded from Git.

## Launcher

`alf/examples/run_humanoid_run_bafcv6_restart_6jobs-4g.sh` defaults to four jobs: 90k and 105k checkpoints for seeds 0 and 1. Override with distinct `seed:checkpoint-k` choices using `--jobs`. OMP, MKL, and OpenBLAS default to 3 threads each; existing environment settings take precedence. Each job uses all four GPUs concurrently, preserves critic UTD 11, enables critic reweighting immediately, and trains to absolute 200k steps per rank. Outputs use a fresh timestamped study directory. No training was launched while preparing this report.

Preview the default jobs:

```bash
bash alf/examples/run_humanoid_run_bafcv6_restart_6jobs-4g.sh --dry-run
```

Use `--jobs` to override the choices; remove `--dry-run` to launch.

## Reproduction

The existing evaluator was run in four concurrent CPU worker processes with matching-cache reuse. Each result records hashes of its source files and evaluator code. Equivalent serial invocation:

```bash
.venv/bin/python -m alf.bin.evaluate_bafcv3_checkpoints \
  --source-glob /workspace/server3_copy --task humanoid:run \
  --output artifacts/humanoid_run_restart --resume --device cpu \
  --repetitions 5 --sample-counts 128 --ridges 0.0001 \
  --critic-samples 32 --no-rescan
.venv/bin/python artifacts/humanoid_run_restart/make_selection_report.py
```

The 32-sample critic residual diagnostics in the raw report are auxiliary; the checkpoint table uses the five-repetition trust estimate.
