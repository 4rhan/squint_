# Paper context: scripted demonstrations + Q-chunking for long-horizon visual RL on the SO-101 arm

This file is the single source of truth for writing the paper. It summarises the project as of **2026-10-04**,
taken from the repository (`squint_`, branch `feat/scripted-demos`) and the running experiment log
(`QC_EXPERIMENT_LOG.md`). Every number below is copied from a run's `metrics.jsonl` or from the experiment log.

> **Rules for whoever writes from this file (human or Claude):**
> 1. Only state results that appear here. If a claim needs a number that isn't here, write `\todo{...}`; don't invent one.
> 2. Most results are **single runs (one seed)**. Evaluations use 16 or 64 episodes. Say so, and don't call a
>    difference significant unless it is large compared to the eval noise (16 eps ≈ ±0.12 SE at p≈0.5; 64 eps ≈ ±0.06).
> 3. Everything is **in simulation** (ManiSkill 3). No real-robot result exists for the multi-stage tasks.
> 4. Section 9 ("What we can and cannot claim") overrides any stronger wording elsewhere.

---

## 1. One-paragraph summary

Squint (Almuzairee & Christensen, 2026) trains visual SAC policies for the low-cost SO-101 arm from a 16×16 wrist
image in minutes, and transfers them to the real robot, but only for short (50-step) single-stage tasks. We study
**multi-stage, long-horizon** tasks (100–400 control steps, 2–3 objects: unstacking, tray packing, towers,
rearrangement) at the same tiny observation. Plain Squint fails on them. On our main task, Unstack3, it never grasps
a cube in 1M environment steps. Our pipeline has three parts: (i) a **scripted motion-planning solver** that records
demonstrations in exactly the policy's observation/action format, with **recovery demonstrations** (noisy execution
plus deliberate missed grasps that are then corrected); (ii) a PyTorch port of **QC-FQL** (Q-chunking with a flow-matching
policy distilled into a one-step actor, chunk length h = 5), pretrained offline on the demos and then trained online
with a 50/50 demo/online batch mix; (iii) a **pessimistic (min-ensemble) TD target**, which removes a critic
overestimation that otherwise makes performance decline late in training. Results: **Unstack3 0.89 best / 0.77 final
success** (64 eval episodes) vs **0.00** for Squint; **TrayPack1 1.00**; TrayPack3 (3 cubes, 300 steps) reaches ≥1 cube
in up to 0.94 of episodes but full success stays rare (≈1–3%). Tower3 and StackCube remain unsolved; we report these
negative results and the diagnosed reasons.

## 2. Setting (fixed across experiments unless stated)

| Item | Value |
|---|---|
| Robot | SO-101 (LeRobot) 5-DoF arm + parallel gripper, simulated in ManiSkill 3 (SAPIEN). Training on GPU PhysX; demos recorded on CPU PhysX |
| Control | `pd_joint_target_delta_pos`, 10 Hz control (sim 100 Hz). Action ∈ [−1, 1]^6 → ±0.1 rad per step for arm joints, ±0.2 rad for the gripper |
| Policy observation | 16×16 wrist RGB (rendered at 128×128, area-downsampled) + 12-d proprioception (noisy joint positions 6 + controller target 6). **No object poses, no task flags** in the observation |
| Augmentation | Squint's colour jitter on RGB during training (never baked into demos) |
| Background | Squint's overlay: everything but robot/objects is replaced by a fixed background image (for sim-to-real) |
| Reward | `normalized_dense`: task-specific dense reward / its maximum, so ≤ 1 per step |
| Domain randomization | **Off** for all multi-stage experiments reported here (on only for the Lift deployment policy) |
| Parallel envs | 1024 training envs on one GPU |
| Hardware | One RTX 3060 (12 GB), 24 CPU cores, 15 GB RAM. One run at a time |
| Eval | Deterministic policy (zero noise into the one-step actor); 16 episodes (early runs) or 64 episodes on a **fixed seed (100)**, so every eval sees the same layouts; every 10k offline gradient steps and every 100k online env steps |
| Metrics | `success_at_end` (task complete and held at the last step) is the main metric; `success_once`; partial-progress flags per task (e.g. cube placed at least once) |

## 3. Method

### 3.1 QC-FQL (our PyTorch port of the official JAX `qc` / ACFQL implementation)

- **Action chunks.** The policy outputs a chunk of h = 5 actions (flattened 30-d vector), executed open-loop, then
  re-queried. Chunks are discarded when an episode resets.
- **Critic.** An ensemble of `num_q = 2` Q heads over (observation features, flattened chunk). MLP: Linear → GELU →
  LayerNorm, 4 hidden layers × 512 units. h-step TD target:
  `y = Σ_{k<h} γ^k r_{t+k} + γ^h · mask · agg_i Q̄_i(s_{t+h}, π(s_{t+h}))`, where agg = mean (original) or **min**
  (ours, clipped double-Q). Target network updated by Polyak averaging (τ = 0.01). Chunks whose episode ends before
  the last step are masked out of the critic loss (`critic_valid`).
- **Actor.** (1) A **flow-matching BC policy** v_θ(s, a_t, t) trained on demo/replay chunks:
  `x_t = (1−t)·ε + t·a`, loss `‖v_θ(s, x_t, t) − (a − ε)‖²`. (2) A **one-step policy** μ_φ(s, ε) trained to match the
  Euler-integrated flow (10 steps), the distillation loss, and to maximise Q:
  `L_actor = L_BC + α · ‖μ_φ(s,ε) − flow(s,ε)‖² − Q(s, μ_φ(s,ε))`, with α = 100. Acting uses μ_φ (zero noise at eval).
- **Encoder.** A single Squint CNN shared by critic and actors, trained by the critic loss **and** the BC loss
  (`bc_encoder_grad`, added after BC with a critic-only encoder fitted the demo actions yet placed 0 cubes). The actor's Q and distillation terms see detached features.
- **Two phases.** Offline pretraining on demos only (150k–300k gradient steps), then online RL: each env step is
  followed by 256 gradient steps (actor every 4th), and each batch is 50% demo chunks + 50% online replay chunks.
- **Checkpointing.** `ckpt_best.pt` = best eval, ranked by success at end, then mean max objects placed, then return.

| Hyperparameter | Value used in all multi-stage runs |
|---|---|
| chunk length h | 5 |
| γ | 0.99 (0.995 in one TrayPack3 run, `tp3_800_h400`) |
| networks (actor flow, one-step, critic) | 512 × 4 MLPs (early runs 256 × 3) |
| critic ensemble / aggregation | 2 heads / mean (early), **min (recommended)** |
| optimiser | Adam, lr 3e-4, batch 512 chunks |
| τ (target) | 0.01 |
| α (distill/BC weight) | 100 |
| flow integration steps | 10 (Euler) |
| updates per env step (UTD) | 256 (actor every 4) |
| demo share of online batch | 0.5 |
| learning starts | 5,000 env steps |
| offline steps | 150k–300k (200k for Unstack3) |
| online env steps | 1.5M–5M (2M for Unstack3) |

### 3.2 Baseline: Squint (original, unchanged algorithm)

SAC with a distributional (C51, 101 atoms) critic ensemble of 2, auto-tuned entropy, the same CNN on 16×16 wrist
RGB + 12-d state, 1024 envs, 256 updates per step, batch 512, replay 1M, lr 3e-4, τ 0.01, policy every 4 updates.
Squint's own settings for the baseline run: **γ = 0.9, value support [−20, 20]**. No demonstrations, no chunks.
Same env, same eval protocol (64 episodes, seed 100, no DR) as QC-FQL.

### 3.3 Scripted demonstration pipeline

- **Solver.** A custom IK motion planner for the SO-101 (no mplib) with per-task scripts (pick → carry → place).
  Robustness fixes: signed-distance clearance so the jaws only open as wide as the surroundings allow; grasp
  verification with up to 2 retries; last-resort grasps up to 35° off the cube faces near the reach limit; IK
  tolerances matched to the 2 mm acceptance (5.2× faster solving). Solver success on 20 seeds rose from
  8–10/20 to 12–20/20 across tasks (e.g. Rearrange2 10 → 20, TrayPack3 8 → 12).
- **Same format as training.** Demos are recorded in a training-identical env (same camera, 128 px images,
  12-d state, normalized delta actions, `normalized_dense` reward), stored as HDF5 (`obs/rgb`, `obs/state`, `actions`,
  `rewards`, `terminated`, `truncated` per trajectory + metadata).
- **Recovery demonstrations** (key ingredient). `--action-noise 0.2`: Gaussian noise on executed arm actions while
  following planned paths (holds, gripper motions and final placement servoing stay clean); the solver re-plans from
  the actual state, so the following actions show the correction. `--miss-prob 0.3`: the first grasp attempt of a pick
  closes one cube-height too high; the grasp check detects the miss and the pick is retried. Recorded actions are the
  executed ones, so replay stays exact.
- **FAST profile** (`--fast`). Full-speed descents and lifts and 1-step settles, so demos fit short horizons
  (needed for Unstack3's 150 steps and StackCube's 50). Plus a gripper wait (up to 8 steps) so soft, damped grippers finish closing.
- **Verification.** Static checks (shapes, T vs T+1, action range, horizon, unique seeds) and full **replay** of the
  recorded actions from the recorded seed on CPU (state within 1e-3 rad, must end in success). Recovery demo sets
  passed 500/500 replay verification (TrayPack3, Rearrange2, Unstack3).
- **Collection cost.** Minutes, CPU-parallel (8 workers): e.g. 500 Unstack3 recovery demos in ~5 min (≈45% of
  attempts succeed); 300 TrayPack3 recovery demos in ~9 min (51%).

**TrayPack3 demo quality (800 recovery demos, replayed, `examples/demo_quality.py`):** 800/800 succeed with
exact replay; length median 250 / p90 288 / max 300 (limit 300; 38 within 5 steps of the limit). Cubes are placed
at median steps 73 / 157 / 230 (p90 103 / 191 / 269), so ~75 steps per cube. 684/800 use exactly 3 grasps; 116
contain a re-grasp and 59 a drop (recovered). In 372 demos a placed cube briefly stops counting as placed (gripper
brushing it while placing the next) and comes back.

## 4. Tasks

All tasks: one SO-101, wrist camera, the controller above, objects start unsolved, success must hold for a 1 s dwell
(10 steps) where the env has a dwell, and the robot must be static at the end.

| Task (ID) | Description | Steps | Objects / geometry | Solver demos |
|---|---|---|---|---|
| **Unstack3** (`SO101Unstack3Cube-v1`) | take a 3-cube tower apart: top cube A to the table, then middle cube B; base C must stay | 150 | A, B 22–28 mm, base C 25–32 mm (no DR: midpoints); tower xy in a 10×10 cm box at (0.25, 0), random yaw, ±5° twist per level | yes (FAST), mean 120–123 steps |
| **TrayPack1/2/3** (`SO101TrayPack{1,2,3}-v1`) | put N cubes into their colour-assigned tray compartments (any order) | 150 / 200 / 300 | cubes 20–24 mm; 1×3 tray, compartments 42 mm, walls 4 mm thick × 10 mm high, at (0.33, −0.055) | yes; TrayPack3 mean ~249 steps |
| Tower3 (`SO101Tower3Cube-v1`) | size-ordered 3-cube tower on a marked spot | 300 | large/medium/small cubes | yes (35% solver success with recovery) |
| StackCube (`SO101StackCube-v1`) | original Squint task: stack a cube on a larger cube | 50 | 2 cubes | yes (FAST only fits 41% of seeds) |
| Rearrange2/3 (`SO101Rearrange{2,3}-v1`) | swap cubes between pockets using a buffer pocket | 300 / 400 | 2–3 cubes, 3–4 pockets | Rearrange2 500 demos (26% solver success); not trained/reported |
| Lift (`SO101LiftCube-v1`) | pick up and lift (original Squint) | 50 | 1 cube | yes |

### 4.1 Reward design lessons (these matter for the paper's method section)

- **No dips along a correct demo.** TrayPack v1 paid a grasp bonus that vanished at release, counted a cube only once
  released, and charged −1/step for brushing tray walls, so a correct placement *lowered* the reward. Measured on 100
  demos: worst per-demo drop −1.48 (66/100 demos with a drop > 1.0). **v2** pays 3 for a cube seated in its
  compartment even while held: worst drop −0.10, 0/100 demos with a drop > 1.0. Tower got the same treatment (v2: 0/16
  demos with a drop > 1, worst −0.98, vs −2.5 to −3.4 before).
- **Penalties that block grasping.** A −3/step table-contact penalty taught online RL to stay high and stop grasping
  (22 mm cubes are grasped with fingertips mm above the table). **v3** uses −0.3, and pays `place` only for the held cube
  (v2 paid ~400/episode for cubes lying on the table regardless of behaviour). TrayPack3 runs from `tp3_r3` on use v3.
- **Exploits.** Unstack3's first success check could be met by knocking the tower over (a noisy demo "succeeded" at
  step 11). Fix: per-episode latches (a cube counts as on the table only if it was grasped while still on the tower),
  plus a `tower_knocked` penalty of −3/step.
- **Unstack3 reward** (0–18 scale, normalized by 18): stage A reach 0–2, carry 3–5, on table and held 4–7, released 7–8,
  stage lock-in +9; the same for B; success = 18; penalties: table −6, base drift −2·tanh(20·d), knocked −3. Per-term
  rewards are logged (`rew_<term>`), and the terms sum exactly to the reward.

## 5. Main result: Unstack3 (Figures `unstack3_success`, `unstack3_qmax`; Table `tab:unstack3`)

Setup: 500 recovery demos (FAST, noise 0.2, miss 0.3, no DR, seeds 1000+, ~45% solver success, mean 123 steps), 200k
offline gradient steps, 2M online env steps, 512×4, γ 0.99, h 5, eval seed 100.

| Run | Critic target | Eval eps | Best success | Final | Mean eval success from 400k online | Train-rollout success (end) | Q_max (online) |
|---|---|---|---|---|---|---|---|
| `unstack3_500` | mean | 16 | 0.88 (600k online) | 0.44 (stopped at ~1.39M online) | 0.62 (n = 10) | 0.74 | 105–115 (true max 100) |
| **`unstack3_500_qmin`** | **min** | **64** | **0.89 (900k online)** | **0.77 (2M)** | **0.78 (n = 17)** | **0.89** | 96–112 |
| `unstack3_squint_g09` (Squint, no demos) | – (SAC, C51) | 64 | **0.00** | 0.00 (stopped at 1.0M) | 0.00 | – | – |

(Mean-from-400k-online corresponds to the log's "from logger step 600k" = 200k offline + 400k online.)

- **Takeoff:** both QC runs reach 0.25–0.27 after only 100k online steps; offline pretraining alone gives ≤ 1/16 full
  success (grasp ≤ 0.25).
- **Overestimation → late decline.** With the mean target, Q_max sat at 105–115 although the true maximum is
  1/(1−γ) = 100 (reward ≤ 1/step), Q_mean kept rising (68 → 92) after the training return levelled off, and eval
  success fell from 0.88 to 0.44. Training rollouts (stochastic, 1024 envs) stayed at 0.71–0.74 success at the same time.
  With the min target (the only learning change), Q_max is 96–112, the decline disappears (0.66–0.89 for the rest of
  training), and takeoff speed is unchanged. *Caveat:* the two runs also differ in eval episodes (16 vs 64); the
  q_max and train-rollout numbers are independent of that.
- **Remaining failure mode:** knocking the tower over (`knocked_pen` −50 to −130 per failed eval episode).
- **Squint baseline:** 0 grasps of the top cube in every 64-episode eval up to 900k env steps. It only learned to
  avoid the knocked-tower penalty (−95 → −35…−49), i.e. hovering near the tower without touching it; training return
  plateaued at ~14 (best episode fell from 79 to 16). Stopped at 1.0M of the planned 2M because nothing improved.

## 6. TrayPack results

### 6.1 TrayPack1 (1 cube, 150 steps), `tp1_r3`: 100 clean demos, 50k offline, 1.5M online, v3, γ 0.99, 16 eps

| Logger step | Return | ≥1 cube once | Success at end |
|---|---|---|---|
| 50k (end offline) | 5.0 | 0.00 | 0.00 |
| 750k | 18.5 | 0.06 | 0.00 |
| 950k | 48.1 | 0.44 | 0.44 |
| 1.15M | 92.9 | 0.88 | 0.88 |
| **1.45M (best)** | 111.5 | **1.00** | **1.00** |
| 1.55M (final) | 96.9 | 0.88 | 0.81 |

Offline BC alone places 0 cubes; online RL finds grasping around 750k and reaches 0.81–1.00.

### 6.2 TrayPack3 (3 cubes, 300 steps): progression of runs (Table `tab:traypack3`)

| Run | Demos | Net | Reward | Offline / online | Eval eps | ≥1 cube (best / end) | ≥2 cubes | Full success |
|---|---|---|---|---|---|---|---|---|
| `qc_baseline` | 100 clean | 256×3 | v1 | 50k / 1.5M | 16 | – | – | 0 (env/demo mismatch, fixed) |
| `tp3_full_envfix` | 100 clean | 256×3 | v1 | 50k / 1.5M | 16 | 0.19 / 0 | 0 | 0 |
| `tp3_r2` | 100 clean | 256×3 | v2 | 50k / 1.5M | 64 | 0.19 / 0.09 | 0 | 0 |
| `tp3_r3` | 100 clean | 256×3 | v3 | 150k / 439k (killed) | 64 | ≤0.06 | 0 | 0 |
| **`tp3_recovery`** | **300 recovery** | 512×4 | v3 | 150k / 1.5M | 16 | 0.88 (end) | 0.38 | 1/16 (end) |
| `tp3_recovery_cont` | (resumed) | 512×4 | v3 | – / +1.7M | 16 | 0.94 | 0.56 | 1/16 in 5 of 18 evals |
| `tp3_rec800` | 800 recovery | 512×4 | v3 | 200k / 3.48M | 16 | 1.00 | 0.44 | best 2/16 at 1.4M online; 8/432 = 1.9% over 0.7–3.4M |
| `tp3_500` | 500 recovery | 512×4 | v3 | 250k / 2M | 16, seed 100 | 0.94 (end) | 0.38 | 3-cube mean 0.006 over last 10 evals |
| `tp3_800_h400` | 800 recovery | 512×4 | v3, **400-step limit, γ 0.995, min target** | 300k / 2M | 64, seed 100 | 0.75 | 0.22 | best 3/64 at 800k; 8/832 = 1.0% over 0.7–2.0M; eval policy collapsed at 1.8–2.0M |

Key comparisons:
- **Recovery vs clean demos** (Figure `traypack3_recovery_demos`): end-of-training ≥1 cube 0.19 → 0.88, ≥2 cubes
  0.00 → 0.38, 3 cubes 0.00 → 0.06 (`tp3_r2` vs `tp3_recovery`). **Confounded:** demo type, demo count (100 → 300)
  and network size (256×3 → 512×4) changed together. Supporting evidence: clean-demo pretraining placed 0 cubes and the
  eval video showed the policy carrying on to the tray empty-handed after a missed grasp (it learned the sequence, not
  the success condition), while recovery-demo pretraining occasionally places a cube and grasping is no longer the
  bottleneck online (eval grasp reward 36–53 vs ≤ 30).
- **800 vs 300 recovery demos** (`tp3_rec800` vs `tp3_recovery`, matched online steps, 16 eps): averaged over
  0.7–1.5M online, ≥2 cubes 0.27 vs 0.16, mean max cubes 1.04 vs 0.87, full-success episodes 5 vs 0 (of 144 each).
  Offline, 800 demos under-fit with the same steps (~515 vs ~1,040 passes over the data).
- **Plateau:** `tp3_rec800` window averages (16-ep evals): 0.7–1.5M ≥1 0.71 / ≥2 0.25 / 3 0.031; 1.5–2.5M 0.81 / 0.24 /
  0.019; 2.5–3.4M 0.87 / 0.26 / 0.019. The first cube keeps improving; the second and third are flat for ~2M steps.
- **Time budget hypothesis (tested, not confirmed):** even the scripted solver places the 3rd cube at step ~230 of
  300. `tp3_800_h400` (400 steps + γ 0.995 + min target) improved partial progress early (≥2 cubes 0.08 vs 0.03 at
  100k–600k online) but **not** full success (1.0% vs 2.9% for `tp3_rec800` over 0.7–2.0M, single runs), the critic loss
  rose online (8.7 → 12.9), and the deterministic eval policy collapsed at 1.8–2.0M while stochastic training rollouts
  were still at their best. Three changes at once; γ 0.995 is the main suspect. Not resolved.

## 7. Other results (supporting / negative)

- **StackCube (50 steps), `stack_fast100`:** 100 clean FAST demos (the planner's normal demos take ~70 steps, too long),
  50k offline, 1M online, 16 eps seed 100. Grasp learned (A grasped 0.81 → 1.00); stacking only starting: eval success
  0–0.06, A on B ≤ 0.12; training-rollout success 0.2% → 8.3% and still rising at 1M. `stack_fast500` (500 demos, 500k
  offline): offline evals got *worse* past ~100k steps (return −1.9 → −11…−13; grasp ≤ 0.19). Longer pretraining did
  not produce offline success.
- **Tower3 (300 steps), `tower3_500`:** 500 recovery demos, reward v2, 250k offline, 2M online: **never learned to
  grasp** (eval grasp reward 0.0–0.1 per episode; large cube placed in ≤ 2/16 episodes). Checked: demos match the training
  env as well as TrayPack3's do (same initial states, image statistics, action range). Cause not found; suspects are the
  larger cube needing a wider, more precise grasp at 16×16, and gripper behaviour.
- **Lift (50 steps), `lift_qc_dr200`:** QC-FQL with domain randomization, 0.94 success (16 eps); trained for
  real-robot deployment (deployment code on branch `feat/deployment`). No real-robot success numbers are recorded.
- **Offline pretraining length:** on no task did more offline steps produce offline success; online RL does the work.
  150–200k is enough; StackCube got worse with more.
- **Evaluation noise:** the same TrayPack3 weights scored ≥1/≥2 cubes 0.88/0.38 at the end of one run and 0.62/0.12
  when re-evaluated at the start of its continuation (different random layouts). This motivated 64-episode
  fixed-seed evals. Unstack3: 16-episode evals swung 0.44–0.88 while training rollouts held 0.71–0.74.

## 8. Engineering notes worth a sentence

- QC-FQL port: losses follow the official `acfql.py`. Images use Squint's design (one shared CNN, detached features
  for actors) instead of one encoder per network: the CNN runs twice per update instead of ~7 times.
- Throughput on an RTX 3060: offline ~89–130 gradient steps/s; online ~350–520 env steps/s for QC-FQL (dominated by the
  256 gradient updates per env step), ~950 env steps/s for Squint. A full Unstack3 run (200k offline + 2M online) takes ~2 h.
- CPU-vs-GPU PhysX seeding differs, so demos cannot be replayed open-loop on GPU; they are used as (obs, action) data.

## 9. What we can and cannot claim

**Can claim (with the stated caveats):**
- The pipeline (scripted recovery demos + QC-FQL + min target) learns a 150-step, two-stage unstacking task from a
  16×16 wrist image in ~2 h on one consumer GPU: 0.89 best / 0.77 final over 64 fixed-seed episodes, where Squint
  without demos scores 0 (single seed each).
- Using the min-ensemble target in place of the mean removed critic overestimation (Q above its analytic maximum) and the late-training
  decline on Unstack3 (single seed per variant; evidence from Q_max vs the ceiling and from train rollouts, not only evals).
- Recovery demonstrations were the ingredient that made TrayPack3 progress (with the confound stated).
- Reward-design lessons (no dips, penalties that block grasping, exploit latches) with the measured numbers.
- Honest negative results: TrayPack3 full success ≈1–3%, Tower3 and StackCube unsolved.

**Cannot claim (yet):**
- Any real-robot result for multi-stage tasks.
- Statistical significance or seed robustness (no multi-seed runs; `ckpt_best` was selected on the eval itself, so
  "best" numbers are optimistic; re-evaluation on fresh seeds is a TODO).
- That recovery demos alone cause the TrayPack3 gain (three factors changed).
- That the 400-step limit or γ 0.995 helps or hurts TrayPack3 (three changes at once).
- That Squint would fail with γ 0.99 (only Squint's own γ 0.9 was run, for 1M steps).

## 10. TODO list that would most strengthen the paper

1. Re-evaluate `ckpt_best.pt` / final checkpoints of the main runs on ≥ 256 fresh-seed episodes (removes selection bias).
2. 3 seeds for `unstack3_500_qmin` and `unstack3_500` (mean target, 64 eps) for a clean min-vs-mean comparison.
3. Unstack3 ablations: clean vs recovery demos; 0 demos with QC; h = 1 vs 5 (is chunking needed?); demo count.
4. Squint baseline with γ 0.99 (support [−60, 100]) and the full 2M steps.
5. TrayPack3: separate γ 0.995 from the 400-step limit (`tp3` with 400 steps, γ 0.99, min target).
6. Real-robot deployment of a multi-stage policy (needs DR demos + training).

## 11. Run name glossary

| Name | What it is |
|---|---|
| `unstack3_500` | QC-FQL, Unstack3, 500 recovery demos, mean target, 16 eps |
| `unstack3_500_qmin` | same, min target, 64 eps (main result) |
| `unstack3_squint_g09` | Squint SAC baseline on Unstack3, γ 0.9, no demos |
| `tp1_r3` | TrayPack1, 100 clean demos, v3 reward |
| `tp3_r2`, `tp3_r3` | TrayPack3, 100 clean demos, reward v2 / v3 |
| `tp3_recovery`, `tp3_recovery_cont` | TrayPack3, 300 recovery demos (+ continuation) |
| `tp3_rec800`, `tp3_500` | TrayPack3, 800 / 500 recovery demos |
| `tp3_800_h400` | TrayPack3, 800 demos, 400-step limit, γ 0.995, min target |
| `tower3_500` | Tower3, 500 recovery demos (failed) |
| `stack_fast100`, `stack_fast500` | StackCube, 100 / 500 clean FAST demos |
| `lift_qc_dr200` | Lift with domain randomization (deployment policy) |

## 12. Files in this pack

- `PAPER_CONTEXT.md`: this file (the information to write from).
- `figures/*.pdf` (vector, for LaTeX) and `figures/*.png` (preview): `unstack3_success`, `unstack3_qmax`,
  `traypack3_recovery_demos`, `traypack3_h400_progress`. Regenerate with `python paper/make_figures.py`.
- `results_tables.tex`: optional draft tables (the author is building the final tables).
- Citations to add: Squint (Almuzairee & Christensen 2026, arXiv 2602.21203); Q-chunking / QC-FQL (Li, Zhou &
  Levine 2025); Flow Q-Learning (Park, Li & Levine 2025); ManiSkill 3; SAC; TD3 (clipped double-Q); flow matching;
  C51; LeRobot / SO-101. Verify each entry before submission.
