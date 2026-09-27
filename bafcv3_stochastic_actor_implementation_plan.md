# BAFCv3 Stochastic Actors with Temporally Consistent Exploration

## Purpose

Implement stochastic actors for BAFCv3 while preserving the functional-critic design and adding temporally consistent exploration inspired by seed sampling.

The implementation should separate two concerns:

1. **The learned base policy** `pi(a | s)`, represented as a stochastic distribution.
2. **The rollout exploration process**, which reorganizes the base policy's randomness into persistent and fresh components without changing its one-step marginal distribution.

For a Gaussian base policy, rollout sampling should use

```text
u_t = mu(s_t) + sigma(s_t) * (
          sqrt(1 - lambda^2) * e_t + lambda * z_t)
a_t = action_transform(u_t)
```

where

```text
e_(t+1) = rho * e_t + sqrt(1 - rho^2) * xi_t
e_t, z_t, xi_t ~ Normal(0, I)
```

This gives the following useful limits:

- `lambda = 1`: ordinary SAC/RLPD sampling with fresh noise every step.
- `lambda = 0`: all action randomness comes from the persistent seed.
- `rho = 1`: the seed is fixed for the episode.
- `rho = 0`: the seed is independent at every step.

Because the weighted noise remains standard Normal, the one-step marginal action distribution remains the base policy. As `sigma` approaches zero, both persistent and fresh exploration vanish.

## Scope

### In scope

- Continuous-action BAFCv3 using a squashed Normal actor.
- A stochastic-policy representation suitable for the functional critic.
- Reparameterized actor samples during training.
- Monte Carlo next-action samples for critic targets.
- Persistent Gaussian rollout noise with independent state per environment.
- Deterministic evaluation using the policy mode.
- Backward-compatible configuration defaults.
- Tests for distributional correctness, gradients, state resets, and existing BAFCv3 behavior.

### Deferred

- Beta-policy persistent sampling.
- Discrete-action BAFC based on Dirichlet or Gumbel seeds.
- SAC entropy rewards, learned temperature, and soft-value targets.
- Learning the value of the history-dependent seeded policy itself.

The first implementation should treat persistent noise as an off-policy behavior-exploration mechanism while the functional critic targets the stochastic base policy.

## Important semantic distinction

Replacing the deterministic actor with an RLPD-style actor is necessary for nontrivial marginal-preserving exploration, but it is not sufficient.

An exactly deterministic policy is an extreme point of the probability simplex. If `pi(a_0 | s) = 1` and `E_e[pi^e] = pi`, every perturbed policy must also assign probability one to `a_0`. A stochastic base policy is therefore required.

However, ordinary RLPD/SAC sampling uses fresh noise on every call. It supplies stochasticity but not temporal commitment. The persistent seed adds commitment while leaving the stochastic base policy unchanged in the one-step marginal.

## Current code map

### BAFCv3 algorithm

File: `alf/algorithms/bafc_algorithm_v3.py`

- `BafcActionState` currently contains only `actor_network`.
- The actor is constructed with BAFC-specific parallel-group arguments such as `n_groups`.
- `_predict_action()` currently returns deterministic tensor actions.
- Before training starts, `_predict_action()` samples an independent uniform action every step.
- `rollout_step()` samples one actor ID at episode start and keeps it for the episode.
- `_actor_train_step()` computes both functional-policy-gradient paths:
  - gradient through the action passed to the critic;
  - gradient through the actor representation evaluated at probe states.
- `_critic_train_step()` currently receives one deterministic next action per actor.

### BAFC-compatible projection actor

File: `alf/networks/actor_networks.py`

`ActorProjectionFCNetwork` is the correct starting point because it:

- supports `n_groups`;
- supports BAFC's `id`, `full_neurons`, and parameter-shape contracts;
- can construct Normal or Beta projection heads.

Its current `forward()` constructs an action distribution and immediately applies `dist_utils.get_rmode()`. As a result, the algorithm only observes the mode and cannot distinguish policies with the same mean but different variance.

### Distribution projections

File: `alf/networks/projection_networks.py`

`NormalProjectionNetwork` already supports:

- state-dependent standard deviation;
- reparameterized sampling;
- a pre-squash diagonal Normal distribution;
- tanh and affine transforms into the bounded action specification.

The persistent Gaussian decomposition must be applied to the pre-squash Normal variable and then passed through the same transforms. Adding OU noise after tanh and clipping would not preserve the base marginal.

### SAC/RLPD reference path

Files:

- `alf/algorithms/sac_algorithm.py`
- `alf/algorithms/rlpd_algorithm.py`
- `alf/examples/rlpd_dmc_conf.py`

These provide reference implementations for:

- `rsample()`-based action generation;
- action log probabilities;
- entropy-temperature training;
- critic evaluation at sampled next actions.

Only reparameterized sampling is required for the first BAFCv3 implementation. Entropy training should not be copied partially.

## Design decisions

### 1. Preserve the existing actor API by default

Do not change the default output of `ActorProjectionFCNetwork.forward()`. Existing BAFC-TR and other configurations expect a tensor mode.

Add an explicit stochastic-policy API, for example:

```python
def forward_distribution(
        self,
        inputs,
        full_neurons=False,
        id=None,
        noise=None,
        state=()):
    """Return distribution, deterministic policy features, neurons, and state."""
```

The exact return structure should be a named tuple rather than an undocumented positional tuple. A possible structure is:

```python
BafcActorDistributionOutput = namedtuple(
    "BafcActorDistributionOutput",
    ["distribution", "policy_features", "neurons"],
    default_value=())
```

The legacy `forward()` path should continue to return the mode unless the algorithm explicitly requests the stochastic interface.

### 2. Encode distributions, not samples

The functional critic approximates `Q_hat(pi, s, a)`. For a stochastic actor, the representation of `pi` must be deterministic for fixed network parameters.

For a Normal actor, use

```text
policy_features = concat(pre_squash_mean, log(pre_squash_std))
```

at every actor-evaluation state.

Do not use a random action sample as the actor fingerprint. Otherwise repeated encodings of the same policy would differ, injecting label noise into the functional critic.

When `actor_eval_type == "last_two"`, the final two components should be:

1. the final hidden activation;
2. the distribution-parameter feature vector.

The projection-head placeholder in `bias_params` currently has width `action_dim`. In stochastic mode it must advertise the policy-feature width, which is `2 * action_dim` for Normal and Beta heads. Keep the old width in deterministic mode for checkpoint and configuration compatibility.

### 3. Keep persistent exploration out of the policy fingerprint

The persistent seed is rollout state, not a learned policy parameter. Do not include `e_t` in the base-policy encoding when using the behavior-only interpretation.

The critic continues to learn `Q^pi`, where `pi` is the stochastic base actor. The replay buffer may contain temporally correlated behavior actions because BAFC is off-policy.

If a later experiment wants `Q^(pi^e)`, the functional critic must receive either the seed or an encoding of the conditioned policy, and the environment state must be augmented with the seed to restore the Markov property. That is a separate algorithm.

### 4. Define persistence in semantically stable action coordinates

Persistent random numbers do not automatically imply persistent behavioral preferences. A seed attached to action slot `i` continues to prefer slot `i`, but that is meaningful commitment only if slot `i` retains the same semantics throughout the episode.

For example, suppose a persistent Gumbel seed favors the smallest-index action. If the environment or policy dynamically reorders candidate actions, the favored index may mean "move left" at one step and "move right" at the next. The numerical seed is persistent, but the exploratory hypothesis is not. The same issue can arise when:

- discrete candidate actions are sorted or regenerated every step;
- actions temporarily become unavailable and the remaining actions are compacted;
- a continuous action vector is expressed in a state-dependent coordinate frame;
- a state-dependent covariance factor rotates latent Gaussian noise into changing physical directions;
- a learned action decoder changes which latent coordinate controls which actuator.

The implementation must therefore distinguish a persistent RNG coordinate from a persistent semantic action preference. The preferred design is to store the persistent seed in a stable semantic action space and map it into the actor's current output representation at every step.

For discrete actions:

1. Give each semantic action a stable identity or key.
2. Associate persistent noise with that identity, not its current array position.
3. When the environment reorders candidates, gather the seed using the semantic-ID-to-current-index mapping.
4. When an action is temporarily unavailable, mask it without discarding or reassigning its seed.
5. For actions first encountered during an episode, derive their noise reproducibly from the episode seed and semantic action ID, or store newly sampled noise under that ID.

For continuous actions:

1. Define whether commitment is intended in actuator coordinates, an agent-relative frame, a world-relative frame, or another domain-specific basis.
2. Store `e_t` in that stable basis.
3. Apply an explicit state-dependent transform from the semantic basis to the current action representation before combining it with fresh noise.
4. Ensure the transform still gives the base policy's covariance marginal. For a Gaussian policy, if

   ```text
   u = mu(s) + L(s) * (sqrt(1 - lambda^2) * e + lambda * z),
   ```

   then `L(s) L(s)^T` must equal the base covariance, and the columns of `L(s)` must have a consistent semantic interpretation. An arbitrary eigendecomposition is unsafe because eigenvector order and sign can change across states.

A diagonal Gaussian over fixed actuator dimensions already has a stable coordinate interpretation when those actuator dimensions retain their meanings. For dynamic or structured action spaces, semantic action metadata is required; seed sampling cannot infer this correspondence from action indices alone.

Commitment should not mean blindly repeating the same action. The selected action may change as the state changes. The requirement is that the seed express the same underlying semantic bias or exploratory hypothesis after accounting for the current state and available actions.

### 5. Do not add SAC entropy implicitly

The initial objective should remain expected return:

```text
J(theta) = E_s E_(a ~ pi_theta)[Q_hat(pi_theta, s, a)]
```

The critic target should be a Monte Carlo estimate of

```text
r + gamma * E_(a' ~ pi_theta)[Q_target(pi_theta, s', a')]
```

If entropy is later enabled, all of the following must be added together:

- `log_pi` in training information;
- an entropy coefficient or learned alpha;
- the entropy term in the actor loss;
- the entropy reward in the TD target;
- tests for transformed-distribution log probabilities.

## Configuration additions

Add explicit configuration arguments to `BafcAlgorithmV3` with defaults that reproduce existing behavior:

```python
stochastic_actor=False
temporal_exploration=False
temporal_noise_mix=1.0       # lambda
temporal_noise_rho=0.0       # rho
temporal_noise_reset="episode"
eval_action_mode="mode"
stochastic_target_samples=1
```

Validation rules:

- `0 <= temporal_noise_mix <= 1`.
- `0 <= temporal_noise_rho <= 1`.
- `temporal_exploration` requires `stochastic_actor`.
- The first implementation supports only a Normal projection distribution.
- `stochastic_target_samples >= 1`.

Create a new experiment configuration instead of changing existing BAFCv3 defaults:

```text
alf/examples/bafcv3_stochastic_dmc_conf.py
```

Recommended initial settings:

```python
stochastic_actor = True
temporal_exploration = True
temporal_noise_mix = 0.25
temporal_noise_rho = 0.99
stochastic_target_samples = 1
```

These values are experimental and should not become global defaults based only on the mathematical construction.

## Implementation phases

## Phase 0: Baseline and compatibility tests

Before changing behavior:

1. Run the existing BAFCv3 unit tests.
2. Add a test confirming that `ActorProjectionFCNetwork.forward()` still returns the differentiable mode.
3. Record the expected output shapes for:
   - all actors: `[batch, num_actors, action_dim]`;
   - one selected actor: `[batch, action_dim]`;
   - `full_neurons=True`.
4. Confirm that deterministic BAFCv3 remains unchanged when `stochastic_actor=False`.

## Phase 1: Stochastic actor representation

### Files

- `alf/networks/actor_networks.py`
- `alf/networks/actor_networks_test.py`, or a new focused test file
- possibly `alf/utils/dist_utils.py` if a reusable public parameter-extraction helper is needed

### Work

1. Refactor `ActorProjectionFCNetwork` so the shared trunk is evaluated once.
2. Add the explicit stochastic-policy method.
3. Extract deterministic policy features from the projection distribution:
   - unwrap the transformed distribution;
   - read the pre-transform Normal mean and standard deviation;
   - clamp standard deviation before taking `log`;
   - concatenate mean and log-standard-deviation.
4. Return the original distribution so normal `rsample()` remains available.
5. In stochastic `full_neurons` mode, append policy features instead of the action mode.
6. Update actor-token shape inference to use the stochastic feature width.
7. Preserve the legacy mode-returning `forward()` behavior.

### Tests

- Distribution batch shape is `[batch, num_actors]` with event shape `[action_dim]`.
- Policy-feature shape is `[batch, num_actors, 2 * action_dim]`.
- Two policies with equal means and different standard deviations have different fingerprints.
- `rsample()` propagates gradients into both mean and standard-deviation projection parameters.
- Legacy mode output and shapes are unchanged when stochastic mode is disabled.

## Phase 2: Stochastic BAFC training

### Files

- `alf/algorithms/bafc_algorithm_v3.py`
- `alf/algorithms/bafc_algorithm_v3_test.py`

### Work

1. Add a helper that evaluates all actors and returns:
   - action distributions;
   - reparameterized action samples;
   - deterministic policy features;
   - actor-network state.
2. Add an `_encode_actors()` helper shared by `_actor_train_step()` and `_critic_train_step()` so both encode exactly the same distribution parameters.
3. In `train_step()`:
   - call the stochastic actor for every actor group;
   - draw reparameterized actions;
   - pass sampled actions to actor and critic training.
4. Preserve the two existing functional-policy-gradient paths:
   - `dQ/da` through the reparameterized sampled action;
   - `dQ/d(policy_features)` through probe-state distribution parameters.
5. In `_critic_train_step()`, use sampled next actions from each base actor.
6. If `stochastic_target_samples > 1`, average target critic values over independently sampled next actions before applying the existing target-critic aggregation.
7. Do not store action distributions or log probabilities in replay for the expected-return version.

### Gradient check

For

```text
J(theta) = Q_hat(encode(pi_theta), s, g(theta, s, z)),
```

the implementation must propagate both

```text
dQ/da * da/dtheta
```

and

```text
dQ/dencode * dencode/dtheta.
```

Tests should verify that both the Normal mean head and standard-deviation head receive nonzero gradients under a controlled mock critic.

### Tests

- Training actions differ across repeated stochastic calls.
- Actor fingerprints do not differ across repeated calls with unchanged parameters.
- Actor mean and standard-deviation heads both receive gradients.
- Critic targets use samples rather than modes.
- Multiple target samples produce the expected averaged tensor shape.
- Existing actor/critic pairing and random target-critic selection still work.

## Phase 3: Persistent rollout exploration

### Files

- `alf/algorithms/bafc_algorithm_v3.py`
- `alf/algorithms/bafc_algorithm_v3_test.py`
- optionally a small reusable sampler in `alf/utils/`

### State changes

Extend rollout action state to contain:

```python
BafcActionState(
    actor_network=(),
    exploration_noise=(),
    rollout_actor_id=())
```

Suggested specs:

- `exploration_noise`: `TensorSpec(action_spec.shape)`; outer environment dimensions are added by ALF.
- `rollout_actor_id`: scalar integer `TensorSpec`; outer environment dimensions allow independent actor selection per environment.

Moving actor identity into algorithm state is preferable to the current scalar `_rollout_actor_id`, because parallel environments can start episodes at different times and should have independent committed actors.

If moving actor identity is judged too large for the first patch, retain the existing global actor ID temporarily, but still implement exploration noise per environment and document the limitation.

### Rollout state transition

Use a per-environment `FIRST` mask:

```python
first = inputs.step_type == StepType.FIRST
new_episode_noise = torch.randn_like(state.exploration_noise)
continued_noise = (
    rho * state.exploration_noise
    + sqrt(1 - rho**2) * torch.randn_like(state.exploration_noise))
e_t = torch.where(first[..., None], new_episode_noise, continued_noise)
```

The exact broadcasting helper should support arbitrary one-dimensional continuous action specs and any number of outer environment dimensions.

### Action sampling

For the selected actor:

1. Obtain the pre-squash Normal mean and standard deviation.
2. Draw fresh `z_t ~ Normal(0, I)`.
3. Form

   ```text
   combined_noise = sqrt(1 - lambda^2) * e_t + lambda * z_t
   pre_transform_action = mean + std * combined_noise
   ```

4. Apply the exact same tanh and affine transforms as the base distribution.
5. Return the bounded action.

Do not approximate this by adding noise to the transformed action.

Before combining persistent noise with the policy distribution, map it from its stable semantic basis into the current action layout. For the initial DMC implementation, document and assert the assumption that continuous action dimensions correspond to fixed actuator identities. Do not generalize the implementation to reordered or dynamically generated actions without adding an explicit semantic mapping interface.

For a future dynamic discrete-action implementation, the rollout input or environment adapter should provide stable semantic action IDs. Persistent Gumbel or Dirichlet seeds must be gathered by those IDs before action selection, rather than stored by the current candidate-list position.

### Evaluation behavior

- `predict_step()` in evaluation mode should use the distribution mode by default.
- Persistent noise should not affect deterministic evaluation.
- An explicit diagnostic configuration may allow stochastic evaluation, but it should not be the default.

### Initial collection

The current pre-training path samples an independent uniform action every step. Replace it under `temporal_exploration=True` with a temporally correlated process that remains marginally uniform.

For a bounded continuous action:

```text
g_t = rho * g_(t-1) + sqrt(1 - rho^2) * xi_t
u_t = NormalCDF(g_t)
a_t = action_min + u_t * (action_max - action_min)
```

This provides correlated initial actions while retaining an exactly uniform stationary marginal. Reset `g_t` from a standard Normal at episode start.

### Tests

- The seed resets only for environments with `FIRST` steps.
- Parallel environments have independent seeds and actor IDs.
- `rho = 0` produces negligible lag-one seed correlation.
- `rho` near one produces the expected lag-one correlation.
- `lambda = 1` matches ordinary distribution sampling statistically.
- `lambda = 0` is deterministic conditional on the seed and state.
- The sampled action remains within the action specification.
- Initial-collection actions are marginally uniform and temporally correlated.
- Deterministic evaluation is unaffected by rollout noise.
- Reordering a discrete candidate list while preserving semantic action IDs does not change which semantic preference the seed represents.
- Temporarily masking and later restoring a semantic action preserves its original seed.
- For continuous actions with an explicit coordinate transform, the persistent preference follows the chosen semantic frame rather than the raw output index.
- The implementation rejects or explicitly disables semantic commitment for dynamic action layouts that provide no stable identity mapping.

## Phase 4: Statistical validation and experiment configuration

### Unit-level statistical checks

Use fixed random seeds and tolerances appropriate for Monte Carlo tests.

1. **Pre-squash marginal:** compare the empirical mean and variance of persistent-decomposition samples with the base Normal.
2. **Post-transform marginal:** compare empirical quantiles after tanh and action scaling.
3. **Temporal correlation:** verify that empirical seed autocorrelation is approximately `rho`.
4. **Near-deterministic limit:** confirm action variance approaches zero with the actor standard deviation.
5. **Environment independence:** verify low cross-correlation between parallel environment seeds.

Avoid very large or flaky statistical tests in the normal unit-test suite. A separate diagnostic script may perform stronger distributional checks.

### Experiment matrix

Run at least the following ablations with identical environment and optimization settings:

| Variant | Base actor | Rollout noise | Purpose |
|---|---|---|---|
| A | Deterministic | None | Existing BAFCv3 baseline |
| B | Stochastic | Fresh (`lambda=1`) | Direct RLPD-style actor comparison |
| C | Stochastic | Episode-fixed seed | Maximum commitment |
| D | Stochastic | AR seed | Adjustable commitment |
| E | Stochastic | AR seed plus optional entropy | Deferred SAC-style comparison |

Track:

- episodic return;
- time to first nontrivial reward;
- actor entropy or mean standard deviation;
- lag-one action and noise autocorrelation;
- actor-ensemble diversity;
- functional-critic loss;
- gradient norms for mean and standard-deviation heads;
- action saturation near bounds.

The central empirical question is whether C or D improves temporally extended exploration relative to B without degrading asymptotic control.

## Phase 5: Follow-up policy families

### Discrete policy

For a categorical policy with logits `l(s)`, use persistent Gumbel seeds:

```text
a_t = argmax_i(l_i(s_t) + g_i)
```

For independent standard Gumbel `g_i`, averaging the resulting deterministic conditioned policies recovers the categorical base policy exactly. Holding `g` gives temporal commitment while still allowing the selected action to change when state-dependent logits change substantially.

The image's Dirichlet alternative can also be implemented:

```text
pi^e ~ Dirichlet(c * pi)
```

using fixed uniform seeds passed through inverse Gamma CDFs. Smaller `c` gives sharper conditioned policies. Gumbel-max is likely the simpler first discrete implementation.

### Beta policy

For `Beta(alpha, beta)`, generate two Gamma variables from temporally correlated uniform seeds and normalize:

```text
X_0 = GammaCDFInverse(e_0; alpha, 1)
X_1 = GammaCDFInverse(e_1; beta, 1)
a = X_0 / (X_0 + X_1)
```

Do not use

```text
e <- (1 - c) * e + c * Uniform(0, 1)
```

because this does not preserve a uniform stationary marginal. Use either:

- sticky seed replacement; or
- a stationary Gaussian AR process followed by the Normal CDF.

Beta inverse-CDF gradients and numerical behavior should be evaluated before enabling this in actor training. It may initially be appropriate only for rollout sampling.

## Backward compatibility and checkpoints

1. All new behavior must be disabled by default.
2. Existing deterministic configs should construct the same actor-token sizes and execute the same mode path.
3. Existing `ActorProjectionFCNetwork.forward()` callers must not need changes.
4. Stochastic actor checkpoints will not be interchangeable with deterministic actor checkpoints when the functional-policy token width changes. Fail with a clear shape error or provide an explicit migration utility; do not silently truncate features.
5. Runtime rollout state may gain exploration fields. Confirm that evaluation and checkpoint restore initialize missing state safely for older checkpoints.
6. Do not alter the semantics of the existing random target-critic selection, which BAFCv3 already labels as RLPD-style.

## Risks and mitigations

### Functional critic ignores variance

**Risk:** The encoder learns to rely only on the mean features.

**Mitigation:** Add diagnostics comparing encodings for equal-mean/different-variance policies and monitor gradients into log-standard-deviation features.

### Noisy actor gradients

**Risk:** Reparameterized action samples add variance to the two-path functional gradient.

**Mitigation:** Start with one sample, then test antithetic or multiple samples only if necessary. Keep actor fingerprint features deterministic.

### Policy variance collapses too early

**Risk:** Expected-return training without entropy may drive standard deviation to its minimum before useful exploration occurs.

**Mitigation:** Initially use a controlled minimum standard deviation or a scheduled regularizer. Evaluate full SAC entropy only as a separate, internally consistent objective.

### Saturation after tanh

**Risk:** Large pre-squash variance causes actions and gradients to saturate at bounds.

**Mitigation:** Monitor pre-squash scale and boundary-action fractions. Use the existing clipped exponential standard-deviation transform and reasonable maximum log standard deviation.

### Behavior is history-dependent

**Risk:** Persistent seeds mean rollout actions are not conditionally independent given environment observation alone.

**Mitigation:** Treat persistence strictly as off-policy data collection while training the stationary base policy. Do not claim the functional critic represents the seeded behavior policy.

### Parallel-environment coupling

**Risk:** A scalar module-level actor ID or noise tensor makes all environments explore the same hypothesis.

**Mitigation:** Store actor IDs and persistent seeds in batched rollout state and reset them with per-environment `FIRST` masks.

### Persistent indices but inconsistent semantics

**Risk:** Noise remains correlated in tensor coordinates while the meanings of those coordinates change, creating high measured autocorrelation without coherent exploration.

**Mitigation:** Define the semantic frame for each action space, key discrete seeds by stable action identity, transform continuous seeds from a stable semantic basis, and test invariance to action reordering. If no stable correspondence is available, describe the method only as correlated action noise, not seed-sampling commitment.

### Distribution internals

**Risk:** Reaching through private transformed-distribution attributes is brittle.

**Mitigation:** Add a small public policy-parameter/sampling interface to the projection actor or projection network. Keep transform application in one tested helper.

## Acceptance criteria

The first continuous-action implementation is complete when all of the following hold:

- Existing deterministic BAFCv3 tests pass unchanged with new features disabled.
- BAFCv3 can train with a grouped squashed-Normal actor.
- The functional actor encoding contains both pre-squash mean and log standard deviation.
- Reparameterized actions send gradients to both distribution heads.
- Critic targets use stochastic next actions from each base actor.
- Rollout state maintains independent persistent seeds per environment.
- Episode resets affect only the appropriate environment states.
- `lambda = 1` statistically matches ordinary RLPD/SAC sampling.
- For all `lambda` and `rho`, the one-step action marginal statistically matches the base policy.
- Increasing `rho` increases temporal correlation without changing the one-step marginal.
- Persistent noise represents the same semantic action preference across action reordering, availability changes, or coordinate transforms supported by the environment.
- Dynamic action spaces without stable semantic IDs fail clearly or run with semantic commitment explicitly disabled.
- Evaluation uses deterministic modes and is independent of rollout noise.
- The implementation includes an ablation configuration comparing fresh and persistent stochasticity.

## Recommended patch sequence

Keep reviewable changes small and independently testable:

1. **Patch 1:** Add distribution and policy-feature access to `ActorProjectionFCNetwork`; preserve legacy behavior.
2. **Patch 2:** Add stochastic BAFCv3 training and deterministic distribution fingerprints.
3. **Patch 3:** Add persistent Gaussian rollout state and marginal-preserving sampling.
4. **Patch 4:** Add correlated-uniform initial collection and per-environment actor IDs.
5. **Patch 5:** Add experiment config, diagnostics, and statistical tests.
6. **Optional patch:** Add SAC entropy objective as a fully consistent BAFC variant.
7. **Optional patch:** Add discrete Gumbel and Beta rollout samplers.

This sequence first establishes that stochastic functional actors work, then isolates the additional value of temporal commitment.
