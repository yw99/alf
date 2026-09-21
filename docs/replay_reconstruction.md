# SAC/RLPD replay reconstruction continuations

This optional path resumes the archived dog:walk SAC and RLPD seeds 0–3 from
600k to 800k aggregate environment steps (150k to 200k per worker). Their original
replay files are empty. The reconstructed buffer approximates the last 400k
aggregate steps using historical policies; it is not the original replay.

## Run

```bash
bash alf/examples/run_dog_walk_sac_reconstruction_seed0123-4g.sh --dry-run
bash alf/examples/run_dog_walk_rlpd_reconstruction_seed0123-4g.sh --dry-run

# Each launcher starts four concurrent seeds, each with four GPU workers.
bash alf/examples/run_dog_walk_sac_reconstruction_seed0123-4g.sh --run-id sac-extension
bash alf/examples/run_dog_walk_rlpd_reconstruction_seed0123-4g.sh --run-id rlpd-extension
```

Defaults: GPUs `0,1,2,3`, results below
`/workspace/alf_results/dog_walk/{sac,rlpd}_reconstruction/<run-id>/seed_<seed>`.
Use `--gpus`, `--dir`, and `--run-id` to change these. Each job uses an automatically
assigned HTTP port. Launching both scripts concurrently creates eight jobs.

`--prepare-only` validates and writes configuration without collecting or
training. `--reconstruct-only` collects and checkpoints replay without advancing
training. Continue either kind of prepared study using the same script with
`--resume --run-id <id>`. A complete reconstructed checkpoint reloads without
recollection. An interruption before the first complete checkpoint restarts
reconstruction from its original inputs. Launchers refuse to start duplicate
live PIDs from their previous invocation.

The single-job entrypoint is:

```bash
python -m alf.bin.train_replay_reconstruction \
  --source-run /workspace/server3_copy/dog_walk_sac_s0 \
  --root-dir /workspace/alf_results/my-sac-continuation \
  --worker-gpus 0,1,2,3 --final-env-steps 800000
```

Use this entrypoint for later resumes too. Ordinary `alf.bin.train` without the
generated reconstruction configuration does not perform this protocol.

## Protocol

Each rank stores 100,000 native replay entries. Checkpoints 45045, 60060, 75075,
90090, 105105, 120120, 135135, and 150150 contribute respectively 2,500, 15,000,
15,000, 15,000, 15,000, 15,000, 15,000, and 7,500 entries. The checkpoint suffix
is an iteration counter, not the environment-step horizon.

Collectors use stochastic rollout actions and frozen saved observation
normalizers, storing raw observations. Policy changes do not reset the
environment; the final live stream is handed to the restored learner. No learner
updates, metric updates, or training-counter increments occur during collection.
The standard replay buffer and inherited SAC/RLPD training methods are retained.

Reconstruction logs record entries and actual environment interactions separately
(the replay includes episode boundary entries). The interaction cost is roughly
400k aggregate per job and is additional to the reported 800k training budget.

## Isolation and limitations

All reconstruction logic is in `alf/utils/replay_reconstruction.py`. Its Agent
subclass only implements checkpoint hooks; its trainer subclass customizes
restoration, coordinated checkpoint publication, and reconstruction-only mode.
`alf/bin/train.py` has one optional trainer-class configuration parameter whose
default is the existing RLTrainer. No base algorithm, replay sampler, ordinary
restore method, or shared checkpoint implementation is changed.

Source model/optimizer files and configuration are fingerprinted. Supported
legacy RLPD counters are inferred from optimizer steps and validated against the
10-critic/1-actor full-cycle boundary, including the initial joint update. New
rank-state files retain RNG, normalization, and algorithm runtime counters.
Only checkpoints with a complete four-rank manifest are selected on resume.

The original RNG, environment state, replay, and rank-local normalization history
are unavailable. Historical collectors share the saved rank-0 normalizer for
each policy. Later resumes reset the environment and use the existing replay
loader's newest-entry LAST marking. Neither legacy nor later resumes promise
bitwise environment continuation. Unsupported configurations fail explicitly.

## Validation

```bash
python -m unittest alf.utils.replay_reconstruction_test
ALF_RECONSTRUCTION_SOURCE=/workspace/server3_copy/dog_walk_sac_s0 \
  OMP_NUM_THREADS=1 python -m unittest alf.utils.replay_reconstruction_test
ALF_RECONSTRUCTION_SOURCE=/workspace/server2_copy/dog_rlpd_s0 \
  OMP_NUM_THREADS=1 python -m unittest alf.utils.replay_reconstruction_test
```

The optional integrations use four CPU/Gloo ranks, real checkpoint weights and
DM Control, with test-local small replay quotas and a reduced warm-up. They check
collection, ordinary updates, synchronization, complete checkpoints, and reload
without recollection. They do not launch full 800k jobs. GPU validation is a
separate deployment check.
