# Why BAFCv3 Underperforms RLPD on Humanoid

This note summarizes the local four-seed comparison between BAFCv3 and RLPD on
DM Control `humanoid:walk`. BAFCv3 was better than RLPD on `hopper:hop`, but the
ordering reverses on Humanoid. The evidence indicates that BAFCv3 learns a
useful but overly aggressive moving policy, saturates most action dimensions,
and then struggles to maintain posture. RLPD learns more slowly at first but
continues improving toward a better-balanced gait.

The measurements below come from TensorBoard events and checkpoints under:

```text
/root/numeric_results/humanoid/bafcv3_dmc_4g/critic_utd3
/root/numeric_results/humanoid/rlpd_dmc_4g/critic_utd10
```

No result artifacts are copied into the repository.

## Learning curves

The stored Humanoid results contain four seeds for each algorithm. Average
training-rollout return evolves as follows:

| Logged environment steps per GPU | BAFCv3, critic UTD 3 | RLPD, critic UTD 10 |
| ---: | ---: | ---: |
| 30k | 66.6 | 1.4 |
| 60k | 200.4 | 114.1 |
| 90k | 269.6 | 325.5 |
| 110k | 291.1 | 378.9 |
| Final available | 293.7 at 114-122k | 466.9 at 150k |

BAFCv3 is substantially faster early in training, as it was on Hopper. RLPD
passes it around 80-90k logged steps and continues improving while BAFCv3
flattens near a return of 300.

These were four-GPU DDP runs. The logged environment-step axis is per rank, so
150k logged steps correspond to approximately 600k aggregate environment
interactions. RLPD completed that budget. The original BAFCv3 jobs stopped at
roughly 114-122k logged steps, so their final comparison is incomplete, although
their curves were already flattening. All episodes had length 1000, making
return directly comparable across the runs.

The later BAFCv3 critic-UTD-2 learning-rate sweep showed the same behavior. At
roughly 107-113k logged steps, its available two-seed conditions had returns of
about 240-303 and still showed severe actor-logit saturation.

## Exploration behavior

Both configurations inherit 10k initial collection steps from the common DM
Control configuration. Their behavior differs after training begins.

BAFCv3 has ten deterministic actors. At each episode boundary it samples one
actor and commits to it for the episode, as implemented in
[`BafcAlgorithmV3.rollout_step()`](../../alf/algorithms/bafc_algorithm_v3.py).
This provides temporally coherent ensemble exploration, but no local action
noise. In the current
[`bafcv3_dmc_conf.py`](../../alf/examples/bafcv3_dmc_conf.py), actor and critic
bootstrapping are both disabled, so all actors learn from essentially the same
experience. Exploration weakens when the actors converge toward similarly
aggressive policies.

RLPD uses a state-dependent squashed Gaussian actor and samples an action every
step. Entropy regularization remains enabled in
[`rlpd_dmc_conf.py`](../../alf/examples/rlpd_dmc_conf.py). With action bounds
`[-1, 1]` and `min_prob=0.184`, its target entropy is approximately `-1` per
action dimension: about `-21` for Humanoid and `-4` for Hopper. A rollout from
the final Humanoid seed-0 checkpoint had estimated policy entropy near `-19`
and mean latent Gaussian scale near `0.34`, showing that the successful policy
remained meaningfully stochastic.

This difference is visible in the curve shape. BAFCv3 rapidly finds a moderate
return behavior. RLPD initially makes little progress and has high
between-seed variance, but its continuing local exploration eventually finds
better posture-preserving behavior.

## Actor and critic updates

The configured update labels understate an important schedule difference.
BAFCv3 uses `actor_utd=1`, `critic_utd=3`, and 12 updates per training
iteration. Its alternating update state therefore produces approximately nine
critic-only and three actor-only updates per interaction, apart from the first
joint initialization update. RLPD uses ten critic updates and one actor update
within 11 updates per iteration.

Consequently, each BAFCv3 deterministic actor moves about three times as often
as the RLPD actor. That can accelerate exploitation on Hopper, where estimating
`dQ/da` over four actions is relatively easy. On Humanoid, the actor is updated
from a much harder 21-dimensional critic gradient and can outrun the critic.

RLPD also reduces critic-specific actor-gradient error. It maintains ten
critics, uses their mean for actor training, and samples one target critic for
each TD target in the current configuration. BAFCv3 instead pairs every actor
with one functional critic. Its
[`_actor_train_step()`](../../alf/algorithms/bafc_algorithm_v3.py) follows that
paired critic's `dQ/da` directly. The current configuration does not clip
`dQ/da`, and the actor does not average the gradient across critics.

In 21 action dimensions, an erroneous paired-critic gradient can push many
joints toward their limits simultaneously. Once the deterministic `tanh`
actor is saturated, its small derivative also makes recovery difficult.

## Functional critic complexity

BAFCv3 conditions each functional critic on an encoding of the actor. The actor
is evaluated on 512 synthetic normalized observations, and those outputs are
encoded by a four-layer Transformer. Because `actor_encoding_dim=None`, the
Transformer embedding dimension defaults to the 512 actor-evaluation samples.

On Humanoid, each synthetic probe has 67 independently sampled coordinates and
is initialized from `2 * Normal(0, 1)` without clipping. Such probes often
represent physically impossible combinations of joint positions, extremity
positions, orientation, and velocity. The actor encoding can therefore be
dominated by behavior far from the replay-state manifold. This problem is less
severe in Hopper's 15-dimensional observation space.

The trainable parameter counts from the saved checkpoints are:

| Component | BAFCv3 | RLPD |
| --- | ---: | ---: |
| Actor network or networks | 885,970 | 93,994 |
| Actor Transformer encoder | 12,609,536 | - |
| Critic network or ensemble | 3,539,210 | 898,570 |
| Actor-evaluation samples | 34,304 | - |
| Entropy coefficient | - | 1 |
| **Total** | **17,069,020** | **992,565** |

BAFCv3 therefore has about 17.2 times as many optimizable parameters in this
comparison. Its actor encoder alone has 12,609,536 parameters. The checkpoint
also contains a non-trainable positional-encoding buffer of shape
`1 x 277 x 512`, or 141,824 values; that buffer is excluded from the trainable
count. Target critics, observation-normalizer state, metrics, runtime counters,
replay state, and optimizer state are also excluded.

The large model is not inherently wrong, but it makes the functional critic
considerably harder to fit with the same learning rate, minibatch size, replay
capacity, and environment budget used by RLPD.

## Action saturation and reward decomposition

At the ends of the stored Humanoid runs, BAFCv3's mean absolute rollout actor
logit was about `9.3`, and approximately 62% of logits exceeded `|5|`. A `tanh`
output is effectively at `-1` or `1` at that magnitude.

To determine whether BAFCv3 simply failed to move, the four final checkpoints
were replayed for one episode each from the same initial environment state.
For BAFCv3, the actor selected in each checkpoint's saved rollout state was
used. RLPD retained its stochastic rollout policy.

| Mean diagnostic | BAFCv3 | RLPD |
| --- | ---: | ---: |
| Episode return | 304 | 478 |
| Standing times upright factor | 0.406 | 0.590 |
| Control factor | 0.815 | 0.892 |
| Horizontal speed | 1.08 | 0.96 |
| Movement factor | 0.861 | 0.841 |
| Mean absolute action | 0.947 | 0.668 |
| Fraction with `|action| > 0.95` | 0.859 | 0.236 |

Both algorithms reached approximately the requested walking speed. BAFCv3
even had a slightly higher movement factor, but it did so with 86% of action
commands near their limits. Its main deficit was maintaining standing posture,
not generating forward motion.

This distinction matters because Humanoid walk reward multiplies standing,
uprightness, control economy, and movement. Losing posture suppresses the
entire reward even when the center-of-mass speed is adequate. RLPD's less
saturated policy sacrifices little movement score while maintaining a much
better standing-upright factor.

## Why Hopper favors BAFCv3

Hopper has four actions and 15 observations, compared with Humanoid's 21
actions and 67 observations. Its lower-dimensional `dQ/da` is easier to
estimate, and a deterministic actor can quickly exploit a useful contact cycle.

The Hopper hop reward is also simply standing multiplied by hopping speed. It
does not include the Humanoid control factor or a separate torso-upright term.
Forceful, nearly saturated commands can therefore be effective instead of
destabilizing a large articulated body. RLPD's continuing stochasticity and
less frequent actor update can slow precise exploitation of that contact cycle.

BAFCv3's Hopper advantage is therefore consistent with the Humanoid failure:
the same fast, deterministic, aggressive optimization is helpful when the
critic gradient is simple and strong control is tolerated, but harmful when
21 coordinated joints must preserve balance.

## Conclusions

The available evidence supports the following interpretation:

1. BAFCv3 wins the early exploitation phase and quickly learns to move.
2. Its policy-conditioned critic is difficult to fit on Humanoid, partly
   because actor identity is encoded using high-dimensional synthetic probes.
3. Frequent paired-critic actor updates push the deterministic actors toward
   near-bang-bang control.
4. Ensemble selection then provides little useful exploration, and `tanh`
   saturation makes the policy difficult to correct.
5. RLPD learns later, but per-step stochastic exploration, a simpler critic
   problem, ten critic updates per actor update, and critic-gradient averaging
   produce a better-balanced gait.

The original BAFCv3 runs did not complete the final portion of their configured
budget, so the data do not prove that they could never improve. They do show a
clear plateau, severe saturation, and a posture failure at matched interaction
counts.

## Focused follow-up ablations

The most informative BAFCv3 experiments are:

1. Slow actor learning to approximately one actor update per ten critic updates.
2. Set a finite `dqda_clipping` threshold and monitor action logits and
   saturation throughout training.
3. Enable actor layer normalization or reduce the actor learning rate.
4. Set a smaller explicit `actor_encoding_dim`, such as 128, instead of using
   the 512-sample dimension.
5. Build actor encodings from replay-distribution observations, or at least
   clip the synthetic normalized probes.
6. Average the actor objective across multiple critics instead of using only a
   fixed paired critic.

These ablations separate three candidate causes: insufficient post-initial
exploration, unreliable high-dimensional critic gradients, and an oversized or
out-of-distribution actor representation.
