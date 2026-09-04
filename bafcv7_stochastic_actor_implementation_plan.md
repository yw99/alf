# BAFCv7 Stochastic Actor with Episode Seed Sampling

## Revision 3 selectable fingerprints (current)

BAFCv7 uses the original entropy-free BAFC actor and critic objectives and
supports two deterministic policy fingerprints:

- `mean_log_std` (the default) uses `[mu, log_sigma]` with width
  `2 * action_dim`.
- `action_quantiles` uses the transformed actions
  `[a_{-1}, a_0, a_{+1}]` with width `3 * action_dim`.

Both modes support base-policy and seed-conditioned encodings. Quantile
construction uses the projection distribution's exact tanh/affine transforms
without detaching, so functional actor gradients reach both projection heads.
There is no entropy actor loss, entropy critic reward, learned temperature, or
log-probability training state.

Revision-3 checkpoints encode the fingerprint mode and load only into the same
mode. Unmarked pre-entropy checkpoints remain loadable in `mean_log_std` mode;
unmarked checkpoints in `action_quantiles`, mode-mismatched revision-3
checkpoints, and entropy revision-2 checkpoints are rejected before tensor
loading. New launchers write under
`hopper_hop/bafcv7_policy_features_4g/{mode}`.

## Purpose

Create a new algorithm, `BafcAlgorithmV7`, by copying the current BAFCv3
implementation and adding stochastic Gaussian actors with fixed per-episode
seeds. BAFCv7 must be isolated from BAFCv3 and all existing algorithms and
experiments.

## Revision 1 implementation status (historical)

The design is implemented in isolated BAFCv7 files:

- `alf/networks/bafc_v7_actor_network.py`
- `alf/algorithms/bafc_algorithm_v7.py`
- `alf/examples/bafcv7_dmc_conf.py`
- `alf/examples/run_hopper_hop_bafcv7_seeds0123-4g.sh`
- focused actor, algorithm, config, and launcher tests beside those files

The implementation does not modify BAFCv3 or the shared projection networks.
The unchanged BAFCv3 and projection-network regression suites are part of the
verification procedure below.

The implementation compares two interpretations of seed sampling:

1. **Ensemble actors with behavior-only seeding.** Use ten stochastic actors,
   randomly select one actor at the beginning of each episode, use strong
   seed persistence during rollout, but train the functional critics and actors
   for the unconditioned base policies.
2. **One seed-conditioned actor with a critic ensemble.** Use one stochastic
   actor and ten critics, use mostly fresh action noise, and train both the
   functional-policy representation and TD targets for the seed-conditioned
   policy.

Initial collection remains the current independent uniform random action
process. This version does not add correlated initial collection, OU/AR seed
evolution, SAC entropy rewards, log-probability losses, or temperature learning.

## Existing network support

ALF already provides most of the required Gaussian machinery:

- `NormalProjectionNetwork` produces diagonal Normal distributions.
- It supports state-dependent standard deviation and reparameterized sampling.
- With `scale_distribution=True`, it applies the existing stable tanh and
  affine transforms into the bounded action specification.
- It supports a parallelism dimension, which is needed for BAFC actor
  ensembles.

`ActorProjectionFCNetwork` is close to what BAFCv7 needs because it supports
BAFC's parallel actor groups and projection heads. However, its public
`forward()` immediately converts the distribution to its differentiable mode.
It therefore does not expose the distribution, pre-squash mean and standard
deviation, or a way to inject an episode seed.

BAFCv7 should reuse `NormalProjectionNetwork` but introduce a new
BAFCv7-specific actor wrapper. Existing actor and projection network classes
must not be behaviorally modified.

## Gaussian actor architecture

### Actor trunk

For actor `i`, the **trunk** is the sequence of hidden fully connected layers
that maps an observation to a learned hidden representation. For a two-layer
actor,

```text
h_i^(1)(s) = ReLU(W_i^(1) s + b_i^(1))
h_i^(2)(s) = ReLU(W_i^(2) h_i^(1)(s) + b_i^(2))
h_{theta_i}(s) = h_i^(2)(s)
```

In the planned DMC experiment, both hidden layers have 256 units. The actors
share an architecture but not parameters. The parallel actor implementation
stores distinct weights `theta_i` for every actor group.

The notation `h_{theta_i}(s)` refers to the final hidden representation of
observation `s` produced by actor `i`. The mean and standard-deviation heads
both consume this representation.

### Gaussian projection heads

The mean projection head is

```text
mu_i(s) = f_{mu,i}(h_{theta_i}(s))
        = W_{mu,i} h_{theta_i}(s) + b_{mu,i}.
```

The standard-deviation projection head first produces an unconstrained value:

```text
ell_i(s) = f_{sigma,i}(h_{theta_i}(s)).
```

With the DMC `clipped_exp` configuration,

```text
log_sigma_i(s) = clip(ell_i(s), log_sigma_min, log_sigma_max)
sigma_i(s) = exp(log_sigma_i(s)).
```

The intended bounds match the existing DMC Normal projection settings,
approximately `log_sigma_min=-20` and `log_sigma_max=2`.

The resulting pre-squash base distribution is

```text
u | s ~ Normal(mu_i(s), diag(sigma_i(s)^2)).
```

### Bounded action transform

For action bounds `[a_min, a_max]`, define

```text
m = (a_max + a_min) / 2
r = (a_max - a_min) / 2
T(u) = m + r * tanh(u).
```

The environment action is `a = T(u)`. All seed and fresh-noise mixing must
happen in the pre-squash variable `u`. Adding noise after tanh and clipping
would not preserve the intended policy distribution and would give different
gradients near action bounds.

## Base and episode-seeded policies

### Base stochastic policy

An ordinary reparameterized base-policy sample is

```text
z_t ~ Normal(0, I)
u_t = mu(s_t) + sigma(s_t) * z_t
a_t = T(u_t).
```

This defines the learned base policy `pi_theta(a | s)`.

### Fixed episode seed

At each environment's `FIRST` step, sample

```text
e ~ Normal(0, I)
```

and hold `e` fixed for that environment until its next `FIRST` step. At time
`t`, draw fresh noise `z_t ~ Normal(0, I)` and use

```text
alpha = sqrt(1 - lambda^2)
eta_t = alpha * e + lambda * z_t
u_t = mu(s_t) + sigma(s_t) * eta_t
a_t = T(u_t).
```

For a fixed state and independently sampled `e` and `z_t`,

```text
Var[eta_t] = alpha^2 I + lambda^2 I = I,
```

so marginalizing over both noises recovers the base pre-squash Normal. The
episode seed changes the temporal dependence of rollout actions rather than
the fixed-state marginal construction.

The two selected noise mixtures are:

| Variant | `lambda` | Persistent variance | Fresh variance |
|---|---:|---:|---:|
| Ensemble/base | 0.10 | `1-lambda^2 = 0.99` | `lambda^2 = 0.01` |
| Single/seeded | 0.90 | `1-lambda^2 = 0.19` | `lambda^2 = 0.81` |

Thus, the ensemble variant has strong episode commitment. The single-actor
variant is much closer to RLPD's fresh action sampling while retaining a
measurable episode-level preference.

### Policy conditioned on a seed

Conditioned on a particular seed `e`, the fresh randomness is only `z`. The
conditional pre-squash distribution is

```text
u | s, e ~ Normal(mu_e(s), diag(sigma_e(s)^2)),
```

where

```text
mu_e(s) = mu(s) + alpha * sigma(s) * e
sigma_e(s) = lambda * sigma(s).
```

Equivalently,

```text
u = mu_e(s) + sigma_e(s) * z.
```

This conditional distribution is the policy `pi_theta^e` that configuration 2
will encode and value.

## Functional-policy encoding

The BAFC functional critic approximates

```text
Q_hat_phi(pi, s, a).
```

Its policy input must be deterministic for a fixed policy. A random sampled
action cannot serve as the actor fingerprint because repeated encodings of an
unchanged policy would differ.

### Base-policy features

At an actor-evaluation state `x`, use

```text
p_theta(x) = concat(mu_theta(x), log_sigma_theta(x)).
```

This is the default `mean_log_std` mode and has width `2 * action_dim`.
The alternative `action_quantiles` mode uses

```text
q_k(x) = T(mu_theta(x) + sigma_theta(x) * k),  k in {-1, 0, +1}
p_theta(x) = concat(q_{-1}(x), q_0(x), q_{+1}(x)).
```

It has width `3 * action_dim`. The ordering is fixed, and the exact tanh and
affine action transforms are applied differentiably without detaching the
quantiles.

With `actor_eval_type="last_two"`, the features at probe state `x_p` are

```text
F_theta(x_p) = concat(
    h_theta(x_p),
    mu_theta(x_p),
    log_sigma_theta(x_p)).
```

This matches BAFCv3's indexing convention. For a two-hidden-layer
`ActorFCNetwork`, its full output list is
`[input, h^(1), h^(2), action]`, so BAFCv3's `[-2:]` selects
`[h^(2), action]`. It does not select both hidden activations plus the
action. BAFCv7 replaces that final action with the selected policy fingerprint.


### Seed-conditioned features

For a replay seed `e`, use

```text
p_{theta,e}(x) = concat(mu_{theta,e}(x), log_sigma_{theta,e}(x))
```

In `action_quantiles` mode the seed-conditioned representation is

```text
q_k(x,e) = T(mu(x) + sigma(x) *
             (sqrt(1 - lambda^2) * e + lambda * k))
p_{theta,e}(x) = concat(q_{-1}(x,e), q_0(x,e), q_{+1}(x,e)).
```

and

```text
F_{theta,e}(x_p) = concat(
    h_theta(x_p),
    mu_{theta,e}(x_p),
    log_sigma_{theta,e}(x_p)).
```

The conditional log standard deviation is particularly simple:

```text
log_sigma_e(x) = log_sigma(x) + log(lambda).
```

The existing transformer encoder maps the table of probe-state features to a
fixed-dimensional policy encoding:

```text
E_theta = ActorEncoder({F_theta(x_p)}_{p=1}^P)
E_{theta,e} = ActorEncoder({F_{theta,e}(x_p)}_{p=1}^P).
```

For seed-conditioned training, each replay sample has its own seed and hence
its own actor encoding. The implementation should flatten the replay-sample
and actor dimensions into the transformer's batch dimension. No transformer
network changes are required.

### Why `mean_log_std` remains the default moments representation

Both `[mu, sigma]` and `[mu, log_sigma]` uniquely determine a Gaussian when
`sigma > 0`, so using `log_sigma` is not required for correctness. It is the
preferred representation for several reasons:

1. The projection head already produces a clipped log-scale value before the
   exponential transform.
2. Multiplicative scale changes become additive. For example, every tenfold
   change in `sigma` has equal spacing in log space.
3. Very small standard deviations remain distinguishable instead of being
   compressed close to zero.
4. Gaussian entropy, log density, and KL-related expressions naturally depend
   on `log_sigma`.
5. Seed conditioning becomes `log_sigma_e = log_sigma + log(lambda)`.

No information is lost by this choice. The log scale must remain clipped or be
computed after clamping `sigma` to avoid negative infinity.

## Functional critic implementation

Use `num_actors` and `num_critics` as separate constructor arguments. Each
critic computes

```text
Q_{phi_c}(E, s, a),   c = 1, ..., num_critics.
```

For critic training, every online critic is trained on every actor policy in
the ensemble. Consequently, online critic values conceptually have shape

```text
[batch, num_actors, num_critics, ...].
```

This must also work when `num_actors=1` and `num_critics=10`.

For base-policy training, `E` is independent of the replay seed. For
seed-conditioned training, use `E_{theta,e}` for the seed stored with that
replay sample. The seed does not need to be concatenated to `(s, a)` separately
because it is represented through the seed-conditioned policy encoding, but it
must remain in replay to reconstruct that encoding and sample its actions.

## Actor update

The initial implementation optimizes expected return without an entropy term.

### Variant 1: ensemble actors and base-policy objective

Actor `i` is permanently paired with critic `i`. During training, ignore the
episode seed and draw a base-policy action:

```text
z ~ Normal(0, I)
a_i = T(mu_i(s) + sigma_i(s) * z).
```

The objective is

```text
J_i(theta_i) = E_{s,z}[
    Q_{phi_i}(E_{theta_i}, s, a_i)
].
```

The rollout seed is behavior-only exploration. Replay can contain temporally
correlated actions, but the functional critic and actor objective describe the
stationary base policy.

### Variant 2: single seed-conditioned actor

For each replay item, use its stored episode seed and fresh noise:

```text
a^e = T(mu(s) + sigma(s) * (alpha * e + lambda * z)).
```

With ten critics, use the current RLPD-style conservative minimum:

```text
J(theta) = E_{s,e,z}[
    min_c Q_{phi_c}(E_{theta,e}, s, a^e)
].
```

The minimum supplies a subgradient through the critic attaining the minimum
for that sample. This variant learns the performance of the policy conditioned
on the episode seed, not just the unconditioned base actor.

### Two BAFC gradient paths

In both variants, the actor affects the critic through its action and its
functional-policy encoding:

```text
J(theta) = Q(E_theta, s, a_theta).
```

The complete gradient is

```text
dJ/dtheta =
    (dQ/da) * (da/dtheta)
    + (dQ/dE) * (dE/dtheta).
```

BAFCv7 must preserve BAFCv3's surrogate-loss implementation of both paths.
Reparameterized sampling gives

```text
du/dmu = 1
du/dsigma = alpha * e + lambda * z
```

for a seeded action. In seed-conditioned features,

```text
d mu_e / d sigma = alpha * e,
```

so the standard-deviation head receives gradients through the sampled action,
the conditional mean, and the conditional standard deviation.

## Critic update

A replay transition includes the environment observation and action plus the
BAFCv7 rollout information:

```text
(s_t, a_t^data, r_{t+1}, discount_{t+1}, episode_seed_t,
 rollout_actor_id_t).
```

The online critic always evaluates the replay action `a_t^data`. The newly
sampled policy action is used for the bootstrapped target.

### Variant 1: base-policy TD update

For actor `i` and critic `c`,

```text
q_{c,i,t} = Q_{phi_c}(
    E_{theta_i}, s_t, a_t^data).
```

At the next state, independently sample from the base actor:

```text
z' ~ Normal(0, I)
a'_i = T(mu_i(s_{t+1}) + sigma_i(s_{t+1}) * z').
```

Sample one target critic `c*` and share its targets across the online critics:

```text
y_{i,t} = r_{t+1}
          + gamma * discount_{t+1}
            * Q_{target,c*}(
                E_{theta_i}, s_{t+1}, a'_i).
```

The critic loss is

```text
L_critic = mean_{c,i,t}[(q_{c,i,t} - y_{i,t})^2].
```

This is an off-policy update for the base stochastic actors. The rollout seed
does not enter either the actor fingerprint or target action.

### Variant 2: seed-conditioned TD update

For replay seed `e_t`,

```text
q_{c,t} = Q_{phi_c}(
    E_{theta,e_t}, s_t, a_t^data).
```

At the next state, sample from the same seed-conditioned policy:

```text
z' ~ Normal(0, I)
a'^{e_t} = T(
    mu(s_{t+1})
    + sigma(s_{t+1}) * (alpha * e_t + lambda * z')).
```

Using one randomly selected target critic,

```text
y_t = r_{t+1}
      + gamma * discount_{t+1}
        * Q_{target,c*}(
            E_{theta,e_t}, s_{t+1}, a'^{e_t}).
```

All online critics minimize

```text
L_critic = mean_{c,t}[
    (Q_{phi_c}(E_{theta,e_t}, s_t, a_t^data) - y_t)^2
].
```

ALF's `OneStepTDLoss` computes target values at every replay timestep and then
uses the value at `t+1` for the transition at `t`. Because BAFCv7 stores the
seed at every timestep and only changes it on `FIRST`, this supplies the
correct seed for the next-state policy. Episode-boundary masking prevents the
new episode's seed from bootstrapping the preceding episode.

The target uses one fresh next-action sample. It is therefore a one-sample
Monte Carlo estimate of the stochastic next-action expectation. No gradient
flows through the target actor sample, target policy encoding, or target
critic.

Target critic parameters use the existing soft update:

```text
phi_target <- (1 - tau) * phi_target + tau * phi
```

with `tau=0.005` in the experiment.

## Rollout state and replay

Define BAFCv7-specific action state containing

```python
BafcV7ActionState(
    actor_network=(),
    episode_seed=(),
    rollout_actor_id=())
```

with:

- `episode_seed`: `TensorSpec(action_spec.shape)`;
- `rollout_actor_id`: scalar integer `TensorSpec`;
- outer environment dimensions supplied by ALF.

At each rollout call:

1. Construct `first = inputs.step_type == StepType.FIRST`.
2. Sample candidate seeds and actor IDs independently for every environment.
3. Use broadcasted `torch.where` to replace state only where `first` is true.
4. Keep both values unchanged for continuing environments.
5. Configuration 1 selects and uses one of ten actors per episode.
6. Configuration 2 always uses actor 0, but still resets its seed per episode.

Store `episode_seed` and `rollout_actor_id` in BAFCv7 rollout information so
they are available after replay sampling. Do not use a scalar module-level
rollout actor ID; it would couple parallel environments.

Before normal training begins, continue to call `action_spec.sample()` at
every step. These actions remain independently uniform and do not use the
episode seed. Seeds should still be initialized and recorded so all replay
entries have a valid BAFCv7 information structure.

## Evaluation behavior

Evaluation is deterministic and does not use episode seeds or fresh noise:

```text
a_eval = T(mu(s)).
```

For an ensemble, evaluation uses actor 0 to match the existing deterministic
BAFC convention unless a future experiment introduces an explicit ensemble
evaluation rule.

## Public BAFCv7 configuration

Add the following BAFCv7-specific constructor settings:

```python
num_actors=10
num_critics=10
temporal_noise_mix=0.1
training_policy="base"       # "base" or "seeded"
actor_update_mode="paired"   # "paired", "random_subset_mean",
                             # "mean_all", or "min_all"
```

Validation rules:

- `0 < temporal_noise_mix <= 1`;
- `actor_update_mode="paired"` requires `num_actors == num_critics`;
- `training_policy="seeded"` initially requires `num_actors == 1`;
- only bounded one-dimensional continuous action specifications are supported;
- only Normal projection distributions are supported;
- actor and critic counts are positive;
- existing target-critic subset counts cannot exceed `num_critics`.

## Implementation structure and isolation

### BAFCv7 actor network

Create `alf/networks/bafc_v7_actor_network.py` containing a BAFCv7-specific
parallel actor wrapper. It should:

- reuse `NormalProjectionNetwork`;
- evaluate the trunk only once per call;
- expose the transformed distribution and its pre-squash Normal parameters;
- return deterministic base policy features;
- construct seed-conditioned features without rebuilding the actor;
- apply the distribution's exact existing transforms to manually constructed
  pre-squash seeded samples;
- support evaluation of every actor and a selected actor ID;
- advertise a final policy-feature width of `2 * action_dim` for BAFC token
  shape inference.

A named output structure should contain at least:

```python
BafcV7ActorOutput(
    distribution=(),
    mean=(),
    std=(),
    policy_features=(),
    neurons=(),
    state=())
```

### BAFCv7 algorithm

Copy the current `alf/algorithms/bafc_algorithm_v3.py` to
`alf/algorithms/bafc_algorithm_v7.py` and rename all public state, info, and
algorithm types to BAFCv7-specific names.

Modify only the copy to:

- use the BAFCv7 actor wrapper;
- decouple `num_actors` and `num_critics`;
- encode `[mu, log_sigma]` rather than action modes;
- support replay-batched seed-conditioned encodings;
- generate reparameterized base or seeded actor-training actions;
- generate base or seeded next actions for critic targets;
- implement paired and minimum-over-all-critics actor objectives;
- store per-environment seeds and actor IDs in state and replay;
- preserve BAFCv3's UTD scheduling, bootstrap behavior, target updates, and
  existing random-target-critic semantics unless explicitly replaced above.

Do not modify:

- `alf/algorithms/bafc_algorithm_v3.py`;
- `ActorProjectionFCNetwork` behavior;
- `NormalProjectionNetwork` behavior;
- existing BAFC, SAC, or RLPD configuration files;
- existing experiment launchers.

Import BAFCv7 classes directly from the new experiment configuration rather
than adding shared import side effects solely for convenience.

## Experiment configuration

Create `alf/examples/bafcv7_dmc_conf.py` with a selectable
`bafcv7_variant`.

### `ensemble_base`

```text
num_actors = 10
num_critics = 10
temporal_noise_mix = 0.10
training_policy = "base"
actor_update_mode = "paired"
```

Behavior:

- independently choose a rollout actor for every environment at `FIRST`;
- keep the actor and Gaussian seed for the episode;
- use seeded actions only for behavior collection;
- train actors and critics for the unconditioned base policies;
- pair actor `i` with critic `i` during actor updates.

### `single_seeded`

```text
num_actors = 1
num_critics = 10
temporal_noise_mix = 0.90
training_policy = "seeded"
actor_update_mode = "min_all"
```

Behavior:

- use one actor and one fixed Gaussian seed per episode;
- condition actor actions, functional-policy encodings, actor updates, and TD
  targets on the replay seed;
- use the minimum of all ten online critics for the actor objective.

### Controlled settings

Both variants use:

The config deliberately has no default environment. The launcher supplies
`bafcv7_env_name='hopper:hop'`; parsing the config without an explicit
`bafcv7_env_name` fails instead of silently launching another task.

- `hopper:hop`;
- random seeds `0,1,2,3`;
- 800,000 environment steps;
- four-GPU DDP;
- actor UTD 1 and critic UTD 3;
- `num_updates_per_train_iter=12`, retaining the existing 1:3 scheduled actor
  to critic update ratio;
- one randomly selected target critic shared across online critics;
- target critic `tau=0.005` and update period 1;
- actor hidden layers `(256, 256)`;
- BAFCv3-sized critic and transformer networks;
- state-dependent Normal standard deviation and existing DMC clipping;
- observation normalization;
- checkpointed replay buffers in variant-specific directories;
- independent uniform initial collection;
- no entropy reward or temperature optimization.

Create a dedicated four-GPU launcher for the two variants and four seeds. It
must support `--dry-run`, use unique DDP ports, and place results in paths that
include the variant name, lambda, UTD settings, and random seed. It must not
reuse a BAFCv3 result directory or checkpoint.

## Test plan

### Actor-network tests

- Verify distribution batch shape `[batch, num_actors]` and event shape
  `[action_dim]`.
- Verify policy-feature shape `[batch, num_actors, 2 * action_dim]`.
- Verify selected-actor shapes for both one and ten actors.
- Verify equal means with different standard deviations produce different
  fingerprints.
- Verify `rsample()` and seeded sampling propagate gradients into both mean and
  standard-deviation heads.
- Verify seeded samples use the exact tanh and affine transforms from the base
  distribution and remain within the action specification.
- Verify empirical conditional mean and variance agree with `mu_e` and
  `sigma_e^2`.
- Verify marginalizing independently sampled seeds and fresh noise recovers the
  base pre-squash mean and variance.

### Rollout-state tests

- Verify only environments with `FIRST` reset their seed and actor ID.
- Verify seeds remain unchanged within an episode.
- Verify parallel environments receive independent seeds and actor IDs.
- Verify configuration 1 randomly selects among all actors.
- Verify configuration 2 always selects actor 0.
- Verify initial collection continues to use independent uniform
  `action_spec.sample()` actions.
- Verify deterministic evaluation is independent of rollout seed state.

### Training tests

- Verify base training actions and fingerprints do not change when replay seeds
  are changed.
- Verify seeded training actions, fingerprints, and targets do change when the
  replay seed changes.
- Verify both BAFC actor-gradient paths remain active.
- Verify mean and standard-deviation projection parameters receive nonzero
  gradients in controlled critics.
- Verify configuration 1 uses fixed actor-to-critic pairing.
- Verify configuration 2 supports one actor and ten critics and differentiates
  through the minimum critic.
- Verify online critic tensors have shape `[batch, num_actors, num_critics]`
  before temporal dimensions are restored.
- Verify base TD targets sample base next actions and seeded TD targets use the
  replay seed.
- Verify the target action is sampled rather than replaced by the mode.
- Verify one randomly selected target critic is shared across online critics.
- Verify terminal and `FIRST` boundaries do not mix seeds across episodes.

### Integration and regression tests

- Parse both BAFCv7 experiment variants.
- Dry-run the launcher and verify eight jobs, unique ports, and isolated result
  directories.
- Run the new BAFCv7 actor and algorithm tests.
- Run the unchanged BAFCv3 and projection-network tests.
- Confirm existing BAFCv3 outputs, public APIs, and experiment commands are
  unchanged.

## Checkpoints and compatibility

- BAFCv7 checkpoints are not compatible with BAFCv3 because actor projection
  parameters, actor-token widths, rollout state, and replay information differ.
- The two BAFCv7 variants also have different actor counts and must use separate
  result and checkpoint directories.
- Revision-3 checkpoints store the selected fingerprint mode and reject a
  different configured mode before parameter tensor loading.
- Unmarked pre-entropy checkpoints are treated as legacy `mean_log_std`
  checkpoints. They load only in that default mode, which preserves existing
  legacy-run resumability.
- Entropy revision-2 and other unsupported revisions fail with explicit
  compatibility errors. Actor features are never truncated, and seed state is
  never silently dropped.
- Runtime episode seeds and actor IDs belong in ALF algorithm state so they are
  handled consistently by rollout and checkpoint state mechanisms.

## Explicitly deferred work

- Correlated or marginally uniform initial collection.
- OU/AR evolution of the episode seed.
- Entropy rewards, learned temperature, or other SAC objective terms.
- Beta, categorical, Gumbel, or Dirichlet seeded policies.
- Dynamic or reordered action spaces requiring semantic action-ID mappings.
- More than one actor in fully seed-conditioned training.
- Multiple Monte Carlo next-action samples per target.
