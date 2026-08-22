# Asymmetric Actor-Critic with Privileged Simulator State

This branch extends Squint with an **asymmetric actor-critic**: during simulation
training the critic may consume exact simulator state (object poses, item
dimensions, contact flags) that is unavailable on a real robot. It addresses the
"Privileged Training" item in Section VIII of the Squint paper.

All additions are **off by default**. With no new flags set, the code path is the
original Squint.

## The invariant

> **The actor is never given privileged information, at training time or deployment time.**

The actor's input is unchanged: a 16x16 "squinted" wrist image plus 12-d
proprioception. Only the critic's forward pass and the critic loss are modified,
so a policy trained here is still deployable camera-only on the real SO-101.

This is enforced in three places and verified by `audit_dims.py` (see *Verification*):

- `PrivilegedObsWrapper` writes a **sibling** observation key, `privileged_state`,
  and is applied only to training envs, after the flatten wrapper. It never enters
  the `state` tensor the actor reads.
- The actor is constructed with the unmodified proprio dimension. Its
  `state_in` layer is 12-wide in every arm.
- `train_squint.py` raises if `privileged_state` is present on an eval or
  deployment env.

## The four configurations

| Arm | Flags | Critic input |
|---|---|---|
| **base** | *(none)* | image + proprio + action |
| **asym** | `--asymmetric_critic` | image + proprio + action + privileged |
| **blind** | `--asymmetric_critic --critic_blind` | proprio + action + privileged (no image) |
| **aux** | `--privileged_aux_loss` | image + proprio + action, plus an auxiliary prediction head |

Flags may be combined; `--critic_blind` requires `--asymmetric_critic`.

### Why `--critic_blind` exists and why it fails

`--critic_blind` is the *exact* asymmetric actor-critic of Pinto et al. (2017):
a critic that takes full state and no image at all. It is included as an ablation
because it is the obvious thing to try, and because in Squint it **fails by
construction**.

Squint trains the CNN encoder with critic TD gradients *only* — the actor consumes
`obs.detach()`, and the encoder's parameters live in the critic's optimizer:

```python
critic_optimizer = Adam(list(critic.parameters()) + list(encoder.parameters()))
```

So a critic that never consumes image features produces no gradient for the
encoder, which stays at its random initialisation for the whole run. The actor
then reads untrained visual features. It still learns — a frozen random CNN plus a
trainable readout is a usable representation — but it is the slowest arm we
measured. **Never use this flag for deployment.**

This is also why `--privileged_aux_loss` exists: handing the critic privileged
state means it can explain Q without the image, which weakens encoder gradient
for the same reason, only partially. The auxiliary head predicts the privileged
state *from the encoder features*, pushing that information into the
representation the actor actually consumes. It is Pinto et al.'s Section IV-B
"bottleneck" task, relocated to the encoder because of the invariant above.

## The privileged state (49 dims)

`Stack.get_privileged_state()` in `envs/stack.py`:

| Slice | Dims | Contents |
|---|---|---|
| 0:7 | 7 | `tcp_pose.raw_pose` |
| 7:14 | 7 | `itemA.pose.raw_pose` |
| 14:21 | 7 | `itemB.pose.raw_pose` |
| 21:24 | 3 | itemA position relative to TCP |
| 24:27 | 3 | itemB position relative to TCP |
| 27:30 | 3 | itemB position relative to itemA |
| 30:33 | 3 | stacking goal relative to itemA |
| 33:36 | 3 | itemA linear velocity |
| 36:39 | 3 | itemA dimensions |
| 39:42 | 3 | itemB dimensions |
| 42:49 | 7 | stage flags: grasping, on-top, lifted, static, robot-static, touching-itemA, touching-table |

The contents are deliberately the quantities `compute_dense_reward()` and
`evaluate()` branch on, since those are what make Q learnable. Two entries matter
disproportionately: the **goal offset** depends on the per-episode randomised item
sizes and is invisible in a 16x16 wrist image, and the **item dimensions**
themselves are likewise unobservable at that resolution.

Only the Stack tasks implement `get_privileged_state()`. Passing
`--asymmetric_critic` on another environment raises a clear error rather than
silently training the baseline.

## Reproducing the results

The study is 3 arms x 2 tasks x 5 seeds = 30 runs, ~29 min each on an RTX 3060.

```bash
# baseline
python train_squint.py --env_id=SO101StackCan-v1 --seed=1 \
  --exp_name=p2_StackCan_base_s1 --eval_freq=25000 --num_eval_envs=16

# privileged critic
python train_squint.py --env_id=SO101StackCan-v1 --seed=1 \
  --exp_name=p2_StackCan_asym_s1 --eval_freq=25000 --num_eval_envs=16 \
  --asymmetric_critic

# Pinto-exact ablation
python train_squint.py --env_id=SO101StackCan-v1 --seed=1 \
  --exp_name=p2_StackCan_blind_s1 --eval_freq=25000 --num_eval_envs=16 \
  --asymmetric_critic --critic_blind
```

Repeat for `SO101StackCube-v1` and seeds 1-5. Add `--track` for W&B logging.

Evaluation of a saved checkpoint:

```bash
python train_squint.py --env_id=SO101StackCan-v1 --seed=101 --evaluate \
  --checkpoint=runs/p2_StackCan_asym_s1/ckpt.pt --num_eval_envs=128
```

**Wall-clock is the measured quantity**, so eval settings must be identical across
arms (eval time counts toward the total) and runs must not compete for host RAM.
A run sharing the machine reports its neighbour's load, not the method.

## Results

Primary metric is **learning-curve AUC**: the time-weighted mean success, computed
by trapezoid integration over (wall_time, success) and normalised to [0,1]. It
integrates all 61 evaluation points, so per-evaluation noise averages down. We do
*not* lead with time-to-threshold, because a threshold crossing is a single point
read off a noisy curve and is very sensitive to the crossing definition.

Learning-curve AUC, n=5 seeds per cell (mean ± SD):

| Task | blind | base | asym |
|---|---|---|---|
| Stack Can | 0.388 ± 0.056 | 0.535 ± 0.050 | **0.651 ± 0.031** |
| Stack Cube | 0.609 ± 0.068 | 0.691 ± 0.027 | **0.767 ± 0.013** |

Headline: the privileged critic reaches any given success threshold about
**1.33x faster** in wall-clock time (median over seven threshold cells). Per-seed
ranges for base and asym are **disjoint on both tasks**; all 10 seed-matched pairs
favour asym (sign test *p* = .00098); Cohen's *d* = 2.78 / 3.58.

The mechanism is **variance reduction, not a higher ceiling**. The privileged
arm's worst seed beats the baseline's mean, and on Stack Can the baseline reaches
90% success in 2 of 5 runs versus 5 of 5.

Final success at 384 episodes improves by +7.8 / +3.5 points (*p* = .028), but
this is sensitive to a single baseline seed — dropping Stack Can's worst baseline
seed nearly halves the gap. Treat it as a consequence of the variance result, not
as independent evidence.

### Caveats

- Simulation only. No real-robot confirmation of the transferred policy.
- Two tasks from the same family (Stack Can, Stack Cube).
- Training is **not** bit-reproducible even with `--torch_deterministic`, so
  identical invocations give different weights. Seed-matched comparison across
  arms is valid; checkpoint diffing is not a usable test of a code change.
- 16-episode evaluations carry roughly ±12 points of noise. The numbers above use
  128-episode evaluations across 3 evaluation seeds.

## Verification

The invariant is checked, not assumed:

- **Dimension audit** — the actor's `state_in` layer is 12-wide in all arms, and
  no privileged tensor appears anywhere in the actor; 26 actor tensors are
  shape-identical across base / asym / blind.
- **Encoder-freeze check** — comparing a `--critic_blind` checkpoint against its
  own step-0 snapshot shows the encoder is bit-identical, confirming the
  no-gradient mechanism above rather than inferring it.
- **Weight graft** — an asym encoder and actor paired with a base critic evaluates
  identically to the asym run, confirming the critic contributes nothing at
  deployment.

## Files changed

| File | Change |
|---|---|
| `envs/stack.py` | `get_privileged_state()` (+51) |
| `utils.py` | `PrivilegedObsWrapper` (+27) |
| `train_squint.py` | privileged critic pathway, aux head, flags (+217 / -27) |

Total: 3 files, +268 / -27.

## Flag reference

| Flag | Default | Effect |
|---|---|---|
| `--asymmetric_critic` | `False` | Concatenates privileged state onto the critic input |
| `--critic_blind` | `False` | Removes the image from the critic entirely; requires `--asymmetric_critic`; ablation only |
| `--privileged_aux_loss` | `False` | Adds an auxiliary head predicting privileged state from encoder features |
| `--privileged_aux_coef` | `1.0` | Weight on the auxiliary loss term |
| `--aug_random_shift` | `False` | DrQ-style random shift augmentation on the encoder input |
| `--aug_random_shift_pad` | `1` | Padding in pixels for the above; 1px on 16x16 matches DrQ-v2's 4px on 84x84 |

`--aug_random_shift` was implemented but is **not** part of the reported study.

## References

- Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, 2017 (RSS 2018) — the `--critic_blind` formulation and the Section IV-B bottleneck auxiliary task.
- Hu et al., *Privileged Sensing Scaffolds Reinforcement Learning*, ICLR 2024 — privileged-sensing scaffolding in a model-based setting.
- Yarats et al., *DrQ-v2* — the random shift augmentation.
