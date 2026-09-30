# Humanoid run: Control A comparison

```sh
.venv/bin/python alf/utils/plot_humanoid_run_control_a.py
```

This compares seeds 0–3 of the newest transferred
`server9_copy/humanoid_run_bafcv3_actor_id/*/fixed_pairingFalse_num_sampled_critic8/critic_utd11`
study with `server3_copy/humanoid_run_bafcv3_rtT_s0` through `s3`.
Use `--control-root` to select a specific study directory containing `seed_0`
through `seed_3`, `--workspace-root` to change the source root, and `--output`
to change the artifact directory.

Control A replaces functional policy encodings with learned actor-ID embeddings.
The embeddings receive critic-loss gradients; the functional encoder/probes are
frozen and unused, and actors no longer receive the policy-encoding gradient.
This ablation does not isolate representation separation alone.

The script reads the existing AverageReturn-versus-EnvironmentSteps tag, keeps
curves unsmoothed, interpolates over the shared observed range of all eight runs,
and caps it at 150k environment steps. No curves are extrapolated. The overview
shows the four-seed mean with population ±1 SD and matched-seed differences.
A separate figure shows the four seed pairs. Summary metrics include endpoint
return, time-weighted mean return over the final 10k steps, and normalized return
AUC over the shared training range. These are descriptions of logged returns,
not additional policy evaluations or statistical significance tests.

Outputs are saved under `artifacts/humanoid_run_control_a`: PNG/PDF figures,
`summary.csv`, raw extracted `curves.json`, source/config provenance in
`manifest.json`, and `report.md`. Rerunning updates these artifacts to reflect
newly transferred logs without changing the source runs.
