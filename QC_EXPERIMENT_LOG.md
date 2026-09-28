# QC experiment log: scripted demos + Q-chunking on long-horizon SO-101 tasks

Running record of every change, diagnosis and training run, newest section last. Numbers are
copied from the runs' `metrics.jsonl` or from the command output quoted in each entry.

## Setup (fixed unless an entry says otherwise)

| Item | Value |
|---|---|
| Robot / sim | SO-101 arm, ManiSkill 3; training on GPU PhysX, demos recorded on CPU PhysX |
| Controller | `pd_joint_target_delta_pos`, 10 Hz, actions normalized to [-1, 1] (±0.1 rad arm, ±0.2 rad gripper per step) |
| Policy input | 16×16 wrist RGB (Squint default; area-downsampled from 128 px) + 12-d proprio (qpos 6 + controller target 6) |
| Algorithm | QC-FQL port (`qc_agent.py`, `qc_data.py`, `train_squint_qc.py`): flow BC policy distilled into a one-step policy, chunked critic, h = 5 |
| Encoder | Squint CNN shared by critic and actors; since 2026-09-26 the BC loss also trains it (`--bc_encoder_grad`) |
| Phases | 50k offline gradient steps on demos, then 1.5M online env steps, 50/50 demo/online batches |
| Main task | `SO101TrayPack3-v1`: 3 cubes (20–24 mm) into a 3-compartment tray (42 mm compartments, 10 mm walls), horizon 300 |
| Hardware | RTX 3060 12 GB, 24 cores (GPU box); CPU-only laptop for development |

Run folders: `runs/<exp_name>/` with `metrics.jsonl`, `ckpt.pt`, `videos/`, `plots/`.

---

## 2026-09-25 — Scripted demo pipeline (branch `tasks-and-data-loaders`)

**Baseline (20 seeds, 1000–1019):** failures fell into two classes.
- `pick_grasp` in Rearrange: the open moving jaw started 1.2–2.0 mm inside a pocket wall and stalled
  (gripper at q=0.16 vs commanded 0.047, 2.6–3.1 N wall contact).
- `pick_ik` in TrayPack/Tower: every case was a cube at r ≥ 0.356 m, where jaw/face alignment is infeasible.

**Fixes** (`examples/motionplanning/so101/`):
- `collision.py`: signed-distance clearance of sampled robot collision surfaces (2 mm spacing) to the
  scene's box obstacles; `pick()` opens the jaws only as wide as the surroundings allow.
- Grasp verification + up to 2 retries (reopen, rise, re-measure).
- Last-resort grasp up to 35° off the cube faces at the reach limit, with fixed-jaw standoff.
- IK least-squares tolerances matched to the 2 mm acceptance: 5.2× faster (120 s → 23 s on 60 cubes),
  identical solved set.
- `collect_new_tasks_demos.py --workers N`: multiprocess collection merged in seed order.

| Task | Success before | Success after |
|---|---|---|
| Rearrange2 | 10/20 | 20/20 |
| Rearrange3 | 3/20 | 12/20 |
| TrayPack2 | 10/20 | 15/20 |
| TrayPack3 | 8/20 | 12/20 |
| Tower3 | 8/20 | 12/20 |

Collection profile: 77% IK (CPU), ~20% sim + render, so collection stays CPU-parallel (as ManiSkill's
own motion-planning demos do) rather than GPU-batched.

**Demo verification** (`examples/verify_demos.py`): static checks (T vs T+1, action range, horizon,
seeds) + replay of recorded actions from the recorded seed on CPU (joint state within 1e-3 rad, ends in
success), optional per-demo mp4. 38/38 demos passed; a deliberately corrupted file was caught.
Demos end with 2 post-success steps (solver's final hold), reported as a note.

Known issue: 16-worker collection on the GPU box hangs before merging (8 workers fine). Not diagnosed.

---

## 2026-09-26 — First TrayPack3 QC run and failure analysis

**Run `qc_baseline`** (user): 100 demos (`demos/qc/SO101TrayPack3-v2.h5`, mean return 98.9),
gamma 0.99, no domain randomization. Result: 0.00 success throughout; return 18–21 after pretraining,
peak 33.6, final 28.

**Root cause 1: env mismatch.** `feat/scripted-demos` still had the old TrayPack env (tray at
(0.32, −0.02), 35 mm compartments, 15 mm walls); the demos came from the new one (tray at (0.33, −0.055),
42 mm, 10 mm walls). Same for Tower and Rearrange. Fixed by syncing `envs/{tray_pack,tower,rearrange}.py`.

**Diagnostics** (pretraining-only runs, 50k steps, 16 eval episodes):

| Check | Result |
|---|---|
| Env fixed, `tp3_envfix_offline` | 0 cubes placed at every eval; BC loss 0.185 → 0.121 |
| + BC loss trains encoder, `tp3_encgrad_offline` | 0 cubes; no change |
| 64×64 images, `tp3_img64_offline` | 0 cubes at 10k and 20k steps; stopped (not the bottleneck) |
| Distillation loss | 0.009: one-step policy follows the BC flow; Q term not dominating |
| Scripted solver on GPU PhysX | 7/8 seeds (CPU 8/8); the task works in training physics |
| Open-loop replay on GPU | fails (different reset RNG; start states differ by ~0.05 rad); exact on CPU |
| BC fit on demo states | arm R² 0.64–0.90 per joint (base yaw worst, 0.64); gripper open/close sign agreement 96%, closes predicted 98% |

Eval video: policy picks the correct cube and carries it to its compartment, then nudges it against
the tray wall instead of lifting it over.

**Run `tp3_full_envfix`** (env fixed, v1 reward, full 1.5M): first cube seated from ~600k steps,
oscillating 0–3 of 16 eval episodes, never a second cube; final 0 cubes, return 40.9 (peak 51.6).

---

## 2026-09-26 — Reward analysis and reward v2

TrayPack dense reward (raw, max 14; training uses raw/14), per unplaced cube:
`0.5·reach + place + grasped`, placed cube 3, +1 per placed cube, success step = 14,
−3 touching table, −1 touching tray.

Placing a cube in v1 passes through a dip: grasp bonus lost at release, the cube counts as placed only
once released and the gripper is clear, and jaws brushing the tray walls cost −1/step.

Reward v2 (`envs/tray_pack.py`, `reward_version=2`, now the default):
- a cube inside its compartment (within the walls, below the wall top, held or not) is worth 3;
- reach pulls only toward the nearest unplaced cube;
- tray contact −0.1 (was −1). Placed, count, success and table terms unchanged.

Every term is exposed per step as `info["rew_<term>"]` and logged as `train_rew/*` and `eval_rew/*`
(episode sums, raw units).

Measured by replaying all 100 demos (`examples/relabel_demo_rewards.py`):

| | v1 | v2 |
|---|---|---|
| Mean worst reward drop per demo (raw) | −1.48 | −0.10 |
| Demos with a drop > 1.0 | 66/100 | 0/100 |
| Mean demo return (normalized) | 98.9 | 97.1 |

The v1 relabel reproduced the stored rewards to 1.2e-7 (refactor check). Relabelled demos:
`demos/qc/SO101TrayPack3-v2_r2.h5`.

Other additions: `examples/plot_metrics.py` (graphs from `metrics.jsonl`), unique default run names,
64 eval episodes, warning for gamma < 0.99 on long tasks, training refuses demos whose
`reward_version` differs from the env's.

Environment incident: packages in the GPU box's `~/.local` (scipy, urllib3, click, pydantic, …) were
removed at 15:41; reinstalled into the `squint` conda env itself.

---

## 2026-09-27 — Run `tp3_r2` (reward v2) results

Same settings as `tp3_full_envfix` except reward v2 + relabelled demos and 64 eval episodes.
Per-term columns are eval episode sums (raw units, mean over 64 episodes).

| Env step | Return | ≥1 cube | Grasp | Placed | Table pen. |
|---|---|---|---|---|---|
| 50k (end of pretraining) | 10.5 | 0.00 | 1 | 0 | −341 |
| 250k | 35.3 | 0.05 | 5 | 13 | −44 |
| 650k | 36.6 | 0.05 | 5 | 30 | −30 |
| 1.15M | 42.5 | 0.12 | 22 | 54 | −40 |
| 1.45M | 42.7 | 0.19 | 21 | 42 | −13 |
| 1.55M (final) | 38.2 | 0.09 | 13 | 30 | −14 |

Never a second cube. The `place` term is ~400 per episode regardless of behaviour.

**Diagnosis:**
- The bottleneck is grasping, not placing: holding time is 1–30 steps per episode versus ~120 in the demos.
- After pretraining the policy touches the table ~130 of 300 steps (table penalty −390 at −3/step):
  its approach is a few mm low.
- Eval video: the pretrained policy reaches a cube, fails to grasp, then carries on to the tray
  empty-handed. It learned the demo's sequence, not the condition that the grasp succeeded; the demos
  contain no failed grasps.
- Online RL removed table contact (−450 → −10 in ~300k steps) by staying higher, which suppresses grasping:
  at −3/step, table contact costs 3× the +1 grasp reward, and a 22 mm cube is grasped with the fingertips
  a few mm above the table.

**Next (reward v3):** table penalty −3 → −0.3; `place` paid only for the held cube; offline steps
50k → 150k (BC loss still falling at 50k). Later if grasping stays rare: demos with deliberate
missed grasps and retries.

---

## 2026-09-27 — Bug fix: junk text in `envs/tray_pack.py` (commit f0a7b97)

The docstring of `TrayPackBase.compute_dense_reward` contained ~290 lines of the Linux `info` directory
listing. Cause: the code was generated through an unquoted shell heredoc, and the word `info` in
backticks was executed as a command. It sat inside a docstring, so behaviour and all results above are
unaffected. Removed (file back to its intended size); no other file affected.

## 2026-09-27 — Reward v3 + longer pretraining

`envs/tray_pack.py`, `reward_version=3` (v2 stays the default; v3 is selected with `--reward_version 3`):
- table contact −3 → −0.3 per step;
- `place` paid only for the cube currently held (v2 paid it for every unplaced cube: ~400 per episode
  regardless of behaviour);
- everything else as v2 (in-compartment 3, placed 3, +1 per placed, success 14, tray −0.1, nearest-cube reach).

Planned run `tp3_r3`: demos relabelled to v3 (`demos/qc/SO101TrayPack3-v2_r3.h5`), `--offline_steps 150000`,
all other settings as `tp3_r2`.

### `tp3_r3` pretraining (150k steps, 64 eval episodes)

| Offline step | Return | ≥1 cube | Grasp | Table pen. (−0.3/step) | BC loss |
|---|---|---|---|---|---|
| 10k | 1.2 | 0.00 | 2 | −60 | 0.180 |
| 50k | 4.2 | 0.00 | 0 | −14 | 0.103 |
| 100k | 4.8 | 0.00 | 1 | −13 | 0.097 |
| 150k | 4.5 | 0.00 | 1 | −16 | 0.075 |

Tripling pretraining cut the BC loss from 0.117 (`tp3_r2` at 50k) to 0.075 but did not produce grasps:
a closer fit to demos that never show a missed grasp being corrected does not teach recovery.
Returns under v3 are not comparable with v1/v2 (no ~400 `place` background).

### Queued runs (tmux session `qc_queue` on the GPU box, `~/qc_queue.sh`, log `~/qc_queue.log`)

Started automatically one after another after `tp3_r3`, same settings as `tp3_r3` unless listed:
1. `tp3_r3_s2`: `--seed 2` (seed replicate).
2. `tp3_r3_a30`: `--alpha 30` (weaker BC/distillation weight, more room for RL), `--offline_steps 50000`.

---

## 2026-09-27 — `tp3_r3` killed; switch to 1-cube baseline + recovery demos

**`tp3_r3` lost at 439k online steps**: the kernel OOM killer killed the training process (01:17:42)
when an 8-worker demo collection started alongside it. The GPU box has 15 GB RAM; one training run
uses ~9 GB. Up to 400k online steps it tracked `tp3_r2` (≥1 cube 0.02 → 0.06 → 0.00, grasp 2 → 9 → 1).
Rule from now on: collection alongside training uses ≤4 workers with `oom_score_adj=1000` (collectors
are killed first), eval uses 16 episodes (`--num_eval_envs` default back to 16).

**1-cube baseline `tp1_r3`**: `SO101TrayPack1-v1` (horizon 150). The scene still has all 3 cubes;
success, reward and `num_correct` count only the first cube, the other two are distractors.
100 clean demos (`demos/qc/SO101TrayPack1-clean.h5`, 1 grasp each, 53–98 steps), relabelled to v3
(`..._clean_r3.h5`, 0/100 reward drops > 1.0). Run started 01:29, 50k offline steps, v3, gamma 0.99.
Pretraining evals (16 episodes): return 0.9 → 5.3, 0 cubes, grasp 0–2, BC loss 0.121 → 0.072.

**Recovery demos** (collector on `tasks-and-data-loaders`; edited in `~/squint_collect` on the box):
- `--action-noise σ`: Gaussian noise on executed normalized arm actions, only while moving along a
  planned path (holds, gripper open/close and place servo stay clean). Recorded actions are the executed
  ones, so replay/verification/relabelling are unchanged; the solver re-plans from the actual arm state,
  so the following actions show the correction.
- `--miss-prob p`: a pick's first attempt closes one cube height too high (on air); the grasp check
  detects the miss, the jaws reopen and the pick is retried (forced miss doesn't use a real retry).

| Setting (TrayPack1, seeds 900+) | Success |
|---|---|
| noise 0.2 + miss 0.5 (noise everywhere) | 0/8 (arm never still for the success check; cube not seated) |
| noise 0.1 only | 4/4 |
| miss 0.5 only | 4/4 (1 of 4 with a miss + retry) |
| noise 0.2 + miss 0.3, noise only on planned paths | 6/6, 3/6 with a miss + retry, all pass verification |

100 TrayPack1 recovery demos collected with noise 0.2 + miss 0.3, 4 workers, ~95% solver success,
~94 steps mean (vs ~75 clean): `demos/qc/SO101TrayPack1-recovery.h5`. Relabel to v3 was started at stop time.

**Next (2026-09-28):** check `tp1_r3` final result; relabel recovery demos (if not finished) and train
`tp1_r3_recovery` with identical settings; if recovery demos help on 1 cube, collect TrayPack3 recovery
demos and rerun TrayPack3. Port the collector changes (`--action-noise`, `--miss-prob`) into the
`tasks-and-data-loaders` branch (currently only in `~/squint_collect` on the box).

---

## 2026-09-27 (morning) — `tp1_r3` result: QC-FQL solves 1-cube TrayPack

`tp1_r3` finished cleanly (1.5M steps, 39 min online, ended 02:15, no errors). Eval: 16 episodes, v3 reward.

| Step (logger) | Return | ≥1 cube once | Success at end | grasp | place | table_pen |
|---|---|---|---|---|---|---|
| 50k (end of offline) | 5.0 | 0.00 | 0.00 | 0.0 | 0.0 | −6.1 |
| 450k | 7.4 | 0.06 | 0.00 | 1.8 | 1.2 | 0.0 |
| 750k | 18.5 | 0.06 | 0.00 | 19.4 | 11.3 | −3.3 |
| 950k | 48.1 | 0.44 | 0.44 | 13.0 | 8.2 | −0.6 |
| 1.15M | 92.9 | 0.88 | 0.88 | 17.9 | 12.0 | −0.2 |
| 1.45M (best) | 111.5 | **1.00** | **1.00** | 17.8 | 11.4 | 0.0 |
| 1.55M (final) | 96.9 | 0.88 | 0.81 | 16.6 | 11.3 | −0.4 |

- The pipeline works end-to-end on a 1-cube task at 16×16: offline BC alone places 0 cubes; online RL
  finds grasping around 750k and reaches 81–100% success. The online phase, not pretraining, does the work.
- ~700k logger steps pass with no grasps before RL takes off, so exploration from the BC prior is the
  slow part. That's what recovery demos are meant to shorten.
- Metric bug: `eval/num_correct_end_mean` is 0.0 even when `success_at_end` is 1.0. The end-of-episode
  count is probably read after auto-reset. Not fixed yet; use `success_at_end` / `num_correct_max_mean`.
- Recovery relabel is complete: `demos/qc/SO101TrayPack1-recovery_r3.h5`, 100 trajs, meta
  `reward_version=3`, `action_noise=0.2`, `miss_prob=0.3`.

**Next:** `tp1_r3_recovery`, the same command as `tp1_r3` with `--demo_path demos/qc/SO101TrayPack1-recovery_r3.h5`.
Compare the step where `num_correct_ge1_once` first passes 0.5 (tp1_r3: ~950k). Then move to TrayPack2/3.

**Fix, `eval/num_correct_end_mean`:** it was read from `eval_infos` after the last step, when the eval
envs had already auto-reset, so it always showed 0. It is now taken from `final_info` (ManiSkill clones
the full pre-reset info there). `train_squint.py` `evaluate()` only; earlier runs' `num_correct_end_mean`
values are invalid, so use `success_at_end` for them.

Repo state checked before commit: laptop and GPU box have identical `envs/tray_pack.py` (v3 reward +
docstring fix), `train_squint_qc.py`, `train_squint.py`, `qc_agent.py`, `qc_data.py`, `relabel_demo_rewards.py`,
`plot_metrics.py`. v3 TrayPack3 demos (`SO101TrayPack3-v2_r3.h5`): 0/100 reward drops > 1.0, all 100
replay and succeed; demo return 98.9 (v2) → 88.5 (v3).

---

## 2026-09-27 — one repo: collector moved into `feat/scripted-demos`

The demo collector (with the recovery options) lived on `tasks-and-data-loaders` and ran from a second
worktree on the box (`~/squint_collect`). It is now in `feat/scripted-demos`, and the worktree is removed.
Collection, training and relabelling all run from `~/squint_`.

- From `tasks-and-data-loaders`: `examples/motionplanning/so101/collision.py`, `examples/verify_demos.py`,
  solutions `rearrange/tower2_cube/tower3_cube/tray_pack.py`, and `motionplanner.py` with the recovery
  options (`ACTION_NOISE`, `MISS_PROB`).
- `motionplanner.py` keeps this branch's `jaw_dir` option (`natural_jaw_dir`, `grasp_feasible`,
  `pick(jaw_dir=...)`, used by place3). With `jaw_dir=None` it behaves exactly as on `tasks-and-data-loaders`.
  place3 check, seeds 0–3: 1/4 succeed with both the old and the merged planner.
- `collect_new_tasks_demos.py --reward-version N` records TrayPack's dense reward version N directly and
  writes it to the meta, so **new demos need no relabelling** (`relabel_demo_rewards.py` is only needed
  for old files).
- Test on the box: 4 TrayPack3 recovery demos (noise 0.2, miss 0.3, v3), all 4 pass replay verification,
  meta `reward_version=3`. Solver success ~4/11 attempts, mean length 246/300 (some rejected as too
  long), so large collections need `--max-attempts` ≈ 5× `-n`.
- `examples/collect_tp3_recovery.sh`: one-command TrayPack3 recovery collection (defaults: 300 demos,
  8 workers, seeds from 1000, noise 0.2, miss 0.3, reward v3, max attempts 5×N) followed by a static
  `verify_demos` check. Settings can be overridden with env vars (`N=`, `WORKERS=`, `OUT=`, ...).
- `TRAIN=1` option added: after collection and `verify_demos`, the script trains QC-FQL on the new demos
  (defaults: `tp3_recovery`, 150k offline steps, 512×4 networks as in the official QC config, gamma 0.99,
  h=5, v3 reward, 16 eval episodes). Smoke test (2 demos, 300 offline steps): collection → 2/2 verified →
  training loads them with "demo rewards and env both use reward_version 3".

**Started 12:43 (tmux `tp3rec` on the box):** `TRAIN=1 bash examples/collect_tp3_recovery.sh`, i.e.
300 TrayPack3 recovery demos (seeds 1000+, 8 workers), then run `tp3_recovery`. First minute: solver
success ~0.75 per worker, mean length ~247 steps. Changes vs `tp3_r3`: recovery demos instead of clean
ones, 300 demos instead of 100, 512×4 networks instead of 256×3.

---

## 2026-09-27 (13:56) — `tp3_recovery`: first TrayPack3 run that places 2 cubes

### Collection (12:43 → 12:52, 8 workers, ~9 min)
- 300 TrayPack3 recovery demos (noise 0.2, miss 0.3, v3 reward recorded directly, seeds 1000+):
  `demos/qc/SO101TrayPack3-recovery.h5`, 195 MB, 74,257 steps, length mean 248 / max 300 (horizon 300).
  Static check: 300/300 passed.
- Solver: 304 saved of 593 attempts (51%; clean TrayPack3 ~60%). Rejections (289): placed only 2 cubes
  80, over 300 steps 80, pick IK failed 48, pick reach failed 28, grasp failed after retries 44, placed
  1 cube 7, other 2. The recovery corrections make demos longer, so the 300-step horizon cuts many of them off.
- RAM: 8 workers alone used ~10 GB of the 15 GB box; fine with nothing else running, too much next to training.

### Training (`tp3_recovery`, started ~12:53, still running at 13:56, ~500 sps, ETA ~14:15)
Command: `TRAIN=1 bash examples/collect_tp3_recovery.sh`, i.e. `train_squint_qc.py --env_id SO101TrayPack3-v1
--demo_path demos/qc/SO101TrayPack3-recovery.h5 --reward_version 3 --gamma 0.99 --no-env_domain_randomization
--horizon 5 --offline_steps 150000 --hidden_dim 512 --num_layers 4 --exp_name tp3_recovery` (16 eval episodes).
GPU memory 8.2 GB of 12 GB with the 512×4 networks.

| Step (logger) | Return | ≥1 cube | ≥2 cubes | 3 cubes | mean max cubes | mean cubes at end | grasp | place | in_comp | table_pen |
|---|---|---|---|---|---|---|---|---|---|---|
| 10k (offline) | 5.5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 |
| 50k (offline) | 10.4 | 0.06 | 0.00 | 0.00 | 0.06 | 0.06 | 1.0 | 0.6 | 0.2 | 0.0 |
| 130k (offline) | 11.3 | 0.06 | 0.00 | 0.00 | 0.06 | 0.06 | 3.8 | 2.3 | 0.8 | −0.9 |
| 150k (end offline) | 6.2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.4 | 0.7 | 0.0 | −0.7 |
| 250k | 15.6 | 0.19 | 0.06 | 0.00 | 0.25 | 0.06 | 14.9 | 8.2 | 1.3 | −5.6 |
| 350k | 17.3 | 0.31 | 0.00 | 0.00 | 0.31 | 0.25 | 15.9 | 9.6 | 4.5 | −4.5 |
| 450k | 18.4 | 0.25 | 0.06 | 0.00 | 0.31 | 0.31 | 15.2 | 9.6 | 9.6 | −13.2 |
| 550k | 28.8 | 0.25 | 0.12 | 0.00 | 0.38 | 0.38 | 17.4 | 9.8 | 9.2 | −0.9 |
| 650k | 43.3 | 0.69 | 0.06 | 0.00 | 0.75 | 0.62 | 36.0 | 22.5 | 17.1 | −4.7 |
| 750k | 59.2 | 0.75 | 0.31 | 0.00 | 1.06 | 1.00 | 38.0 | 22.0 | 16.3 | −8.9 |
| 850k | 64.2 | 0.69 | 0.31 | 0.00 | 1.00 | 0.88 | 43.7 | 24.4 | 15.0 | −1.0 |
| 950k | 57.8 | 0.81 | 0.25 | 0.00 | 1.06 | 0.88 | 53.3 | 30.0 | 28.1 | −1.5 |
| 1.05M | 76.6 | 0.81 | 0.31 | 0.00 | 1.12 | 1.12 | 33.8 | 19.3 | 18.9 | −1.8 |

(≥k cube columns are "at least k cubes placed at some point in the episode", over 16 eval episodes.)

### Observations
1. **Clear improvement over every earlier TrayPack3 run.** `tp3_r2` (v2, clean 100 demos, 256×3) peaked at
   ≥1 cube 0.19 and never placed 2 cubes in 1.5M steps. `tp3_r3` (v3, clean 100 demos) was at ≥1 cube
   0.02–0.06 up to 439k. `tp3_recovery` reaches ≥1 cube 0.81, ≥2 cubes 0.31 and mean 1.12 cubes held at
   the end of the episode by 1.05M, still rising.
2. **Grasping is no longer the bottleneck.** Eval grasp reward went from ≤5 (tp3_r2 early) / ≤30 (tp3_r2
   peak) to 36–53, and in-compartment from ~1 to 15–28. The policy picks and places the first cube reliably
   and the second in about a third of episodes.
3. **Offline pretraining alone now occasionally places a cube** (0.06 at 50k, 80k, 90k, 130k); every clean-demo
   run placed 0 offline. It's still weak (1 of 16 episodes), so the online phase does the main work, as on TrayPack1.
4. **The online phase takes off early:** ≥1 cube reaches 0.19 at the first online eval (250k) and 0.69 by 650k.
   TrayPack1 with clean demos needed ~750k before its first grasps.
5. **No 3-cube episodes yet, and `success_at_end` is 0.** Training rollouts (stochastic) succeed on ~1% of
   episodes (`train/success_once` 0.011), so the critic has only a few full-task transitions.
6. **Confound:** three things changed vs `tp3_r3` (recovery vs clean demos, 300 vs 100 demos, 512×4 vs 256×3
   networks). The run shows the combination works but not which change matters. Ablations are needed for the paper.
7. `num_correct_end_mean` now reports real values (fix from this morning), e.g. 1.12 at 1.05M.
8. Critic: offline Q mean 78.6 (critic loss 0.24); online critic loss 4.96, Q mean 67, max 99. BC flow
   loss ~0.24 (higher than the 0.07 on clean demos, as expected with noisy actions in the data).
9. The table-contact penalty spikes at some evals (−13.2 at 450k, −8.9 at 750k) but stays far below v2's
   −40 to −420, so v3's −0.3 table penalty isn't stopping the policy from grasping.

### Next
- Let `tp3_recovery` finish (1.5M) and record the final eval.
- If it plateaus at 1–2 cubes: continue training longer (more online steps) before changing anything.
- Ablations for the paper: (a) 300 clean demos + 512×4, (b) 100 recovery demos + 512×4, (c) 300 recovery
  demos + 256×3. Each isolates one of the three changes.

### `tp3_recovery` update (14:12, 92% = 1.37M/1.5M online steps)

| Step (logger) | Return | ≥1 cube | ≥2 cubes | 3 cubes | mean max cubes | mean cubes at end | grasp | place | in_comp |
|---|---|---|---|---|---|---|---|---|---|
| 1.05M (peak) | 76.6 | 0.81 | 0.31 | 0.00 | 1.12 | 1.12 | 33.8 | 19.3 | 18.9 |
| 1.15M | 48.5 | 0.75 | 0.00 | 0.00 | 0.75 | 0.62 | 36.1 | 22.2 | 19.3 |
| 1.25M | 41.9 | 0.56 | 0.06 | 0.00 | 0.62 | 0.50 | 32.4 | 18.8 | 18.8 |
| 1.35M | 41.2 | 0.56 | 0.12 | 0.00 | 0.69 | 0.69 | 33.0 | 19.5 | 15.4 |
| 1.45M | 45.3 | 0.69 | 0.00 | 0.00 | 0.69 | 0.62 | 27.9 | 15.8 | 13.5 |

Training-side trends (every ~300k steps):

| Step | critic loss | Q mean | Q max | BC flow loss | train return | train success_once |
|---|---|---|---|---|---|---|
| 456k | 3.74 | 69.3 | 95.5 | 0.269 | 21.0 | 0.000 |
| 763k | 4.38 | 66.2 | 96.6 | 0.256 | 31.5 | 0.002 |
| 1.07M | 4.96 | 67.1 | 99.4 | 0.246 | 47.8 | 0.011 |
| 1.38M | 5.41 | 68.5 | 100.5 | 0.234 | 57.7 | 0.012 |

Observations:
10. **Eval dropped after the 1.05M peak:** ≥2 cubes 0.31 → 0.00–0.12, ≥1 cube 0.81 → 0.56–0.69. With 16 eval
    episodes one episode is 0.06, so the ≥1 wobble is partly noise, but losing the second cube looks real.
    The first cube stays learned (grasp 28–36, in_comp 13–19).
11. **Training rollouts keep improving** (return 47.8 → 57.7, success_once ~1%), while the deterministic eval
    policy got worse. So the decline isn't a collapse of the whole agent. It's either eval noise or the
    deterministic one-step policy drifting between evals (it's re-read at each eval with no averaging or checkpoint selection).
12. **Critic loss creeps up** (3.7 → 5.4) as returns grow. Q max ~100 matches the scale of a 2-cube
    return and doesn't blow up.
13. `ckpt.pt` is overwritten with the latest weights, so the 1.05M peak policy isn't saved. For the next runs we
    should keep the best-eval checkpoint too.

### `tp3_recovery` final (finished 14:17, 1.5M online steps, 55:54 online, no errors)

| Step (logger) | Return | ≥1 cube | ≥2 cubes | 3 cubes | success at end | mean max cubes | mean cubes at end | grasp | place | in_comp |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.55M | 73.4 | 0.81 | 0.31 | 0.00 | 0.00 | 1.12 | 1.06 | 41.7 | 25.0 | 69.2 |
| **1.65M (final)** | **82.9** | **0.88** | **0.38** | **0.06** | **0.06** | **1.31** | **1.25** | 57.5 | 34.6 | 33.2 |

14. **First full TrayPack3 success:** 1 of 16 eval episodes packed all 3 cubes and held them to the end
    (`success_at_end` 0.06). No earlier TrayPack3 run ever placed more than 1 cube.
15. The 1.15M–1.45M dip recovered: the last two evals are the best of the run (≥2 cubes 0.31 → 0.38). The dip
    was a temporary regression, not a collapse, and the policy was still improving at the end, so a longer
    run is likely to help.
16. The final `ckpt.pt` is the best-eval policy of this run (it's saved at the end, which is the final eval).
17. Summary vs the clean-demo baseline `tp3_r2` (end of training): ≥1 cube 0.19 → 0.88, ≥2 cubes 0.00 → 0.38,
    3 cubes 0.00 → 0.06, mean cubes at end ~0 → 1.25.

Plots: `runs/tp3_recovery/plots/` (overlaid with `tp3_r2`).

### Next
- Continue from `ckpt.pt` for more online steps (the curve was still rising), or rerun with a longer
  `--total_timesteps` (e.g. 3M).
- Save the best-eval checkpoint in `train_squint_qc.py` (`ckpt.pt` is overwritten at every save).
- Ablations to separate the three changes (see above), and more eval episodes (e.g. 64) for the final
  numbers in the paper. With 16 episodes, 1 episode = 6%.

---

## 2026-09-27 (14:40) — `tp3_recovery_cont`: continue `tp3_recovery` to 20M online steps

Why resume rather than restart: the policy already places 1–2 cubes and was still improving at the end;
a fresh 20M run would spend its first ~1.5M relearning that. Cost of resuming: the online replay buffer and
Adam state start empty (not saved in the checkpoint), and offline pretraining is skipped.

Trainer changes (`train_squint_qc.py`, `train_squint.py`):
- `evaluate()` returns its metrics dict; the trainer keeps the best-eval weights in `runs/<exp>/ckpt_best.pt`,
  ranked by success at end, then mean max cubes placed, then return (`ckpt.pt` is still overwritten each eval).
- On `--checkpoint`, logged steps continue from the checkpoint's `global_step` (here 1,650,160), so the curves
  line up after `tp3_recovery`'s.

Command (tmux `tp3cont`, log `~/tp3_recovery_cont.log`):
`train_squint_qc.py --env_id SO101TrayPack3-v1 --demo_path demos/qc/SO101TrayPack3-recovery.h5 --reward_version 3
--gamma 0.99 --no-env_domain_randomization --horizon 5 --hidden_dim 512 --num_layers 4
--checkpoint runs/tp3_recovery/ckpt.pt --total_timesteps 18500000 --exp_name tp3_recovery_cont`
(1.5M already done + 18.5M = 20M online steps). ~518 sps, ETA ~10 h (≈ 00:40 on 2026-09-28). GPU 7.2 GB.

First eval of the loaded policy (step 1.65M): ≥1 cube 0.62, ≥2 cubes 0.12, return 51.3. The same weights
scored 0.88 / 0.38 / 82.9 at the end of `tp3_recovery`. The eval layouts differ after the fresh reset, so
the gap shows how noisy 16-episode evals are. Report final numbers from `ckpt_best.pt` re-evaluated on 64+ episodes.

**Changed at 14:44:** the 18.5M continuation was stopped after ~2 min (moved to `runs/_old_tp3_recovery_cont_20M`)
and restarted with `--total_timesteps 3500000` (1.5M + 3.5M = 5M online steps, ~2 h, ETA ≈ 16:45), same
command and `--exp_name tp3_recovery_cont` otherwise.

### `tp3_recovery_cont` status (15:24, 30% = 1.04M/3.5M online steps, ETA ~16:47)

| Step (logger) | Return | ≥1 cube | ≥2 cubes | 3 cubes | success at end | mean max cubes | mean cubes at end | grasp | in_comp |
|---|---|---|---|---|---|---|---|---|---|
| 1.65M (loaded) | 51.3 | 0.62 | 0.12 | 0.00 | 0.00 | 0.75 | 0.69 | 43.4 | 19.5 |
| 1.75M | 66.3 | 0.81 | 0.12 | 0.00 | 0.00 | 0.94 | 0.94 | 31.9 | 12.9 |
| 1.85M | 35.0 | 0.44 | 0.12 | 0.00 | 0.00 | 0.56 | 0.50 | 21.5 | 6.2 |
| 1.95M | 36.6 | 0.50 | 0.06 | 0.00 | 0.00 | 0.56 | 0.50 | 22.9 | 7.1 |
| 2.05M | 54.0 | 0.56 | 0.12 | 0.06 | 0.06 | 0.75 | 0.75 | 26.9 | 11.6 |
| 2.15M | 42.7 | 0.44 | 0.12 | 0.06 | 0.06 | 0.62 | 0.56 | 37.0 | 12.2 |
| 2.25M | 50.8 | 0.75 | 0.31 | 0.00 | 0.00 | 1.06 | 0.75 | 49.4 | 22.1 |
| 2.35M | 55.6 | 0.56 | 0.19 | 0.00 | 0.00 | 0.75 | 0.69 | 32.3 | 37.1 |
| 2.45M | 63.1 | 0.75 | 0.12 | 0.00 | 0.00 | 0.88 | 0.88 | 51.6 | 14.6 |
| 2.55M | 71.8 | 0.75 | 0.25 | 0.06 | 0.06 | 1.06 | 1.00 | 39.3 | 25.1 |
| **2.65M** | **89.3** | **0.88** | **0.50** | 0.00 | 0.00 | **1.38** | **1.31** | 54.9 | 31.3 |

Training side: critic loss 1.9 → 4.0 → 5.1 (starts low because the replay buffer restarted empty), Q mean 68–72,
train return 57 → 40 → 58, train success_once 0.4–1.4%.

Observations:
18. **Dip after the restart, then recovery:** ≥1 cube fell to 0.44–0.56 at 1.85–2.15M (the fresh replay buffer and
    Adam state cost ~500k steps), then climbed back. 2.65M is the best eval of the whole run so far: ≥2 cubes 0.50,
    mean 1.31 cubes at the end.
19. **3-cube successes are now repeated but rare:** 3 of the 11 evals had 1/16 full success (2.05M, 2.15M, 2.55M).
20. `ckpt_best.pt` = 2.55M (ranked by success at end first: 0.06, then 1.06 cubes). The 2.65M policy places more cubes
    (1.38) but had no full success in its 16 episodes, so it didn't replace it.
21. Still mostly a 2-cube policy. The third cube is the bottleneck, and 16-episode evals swing ±0.2 between evals.

---

## Handoff: training-efficiency work (separate branch / separate chat)

Goal: cut wall-clock training time for QC-FQL on TrayPack3 without losing performance. The demo/pretrain/online
scaling work continues on `feat/scripted-demos`.

Where the time goes (`tp3_recovery`, RTX 3060 12 GB, 24 cores, 15 GB RAM):
- Offline pretraining: 150k gradient steps in 28:00 (89 steps/s); the actor, including 10-step flow integration for the
  distillation target, is updated every step.
- Online: 1.5M env steps in 55:52 (≈490 env steps/s). 1024 parallel GPU envs; each env step is followed by
  `num_updates=256` gradient steps (actor every `policy_frequency=4`), so ~1465 iterations × 256 ≈ 375k gradient
  steps (~112/s). **Gradient updates dominate; env stepping is a small share.** GPU utilization ~88%.
- Eval: 16 envs × 300 steps ≈ 24 s every 100k steps (~4% of online time).
- Networks 512×4 (actor/critic), batch 512, h=5, `torch.compile` on, cudagraphs off.
- Memory: training ~8 GB GPU, ~8–9 GB host RAM (+ ~25 idle `torch._inductor` compile workers).

Levers to try (measure env steps/s and eval ≥k cubes vs `tp3_recovery` at matched env steps):
- Update-to-data ratio: `num_updates` 256 → 64/128 with more envs or a bigger batch.
- `policy_frequency`, `flow_steps` (10 → 5) for the distillation target, bf16 autocast, `--cudagraphs`.
- Offline: fewer steps with a larger batch or learning rate; evaluate when pretraining stops helping.
- Smaller networks (256×3 vs 512×4) at equal wall-clock time.

Reference run to compare against: `runs/tp3_recovery` (+ `runs/tp3_recovery_cont`) on the box, metrics in `metrics.jsonl`.

(15:40) An extra 500-demo collection (`SO101TrayPack3-recovery_b.h5`, 2 workers, seeds 5000+) was started and stopped
within ~2 min at the user's request, and its partial files were deleted. Plan to be discussed before collecting more.

### `tp3_recovery_cont` stopped early (15:50, at ~1.7M of the planned 3.5M extra online steps; user's request)

| Step (logger) | Return | ≥1 cube | ≥2 cubes | 3 cubes | success at end | mean max cubes | mean cubes at end |
|---|---|---|---|---|---|---|---|
| 2.75M | 57.6 | 0.75 | 0.19 | 0.00 | 0.00 | 0.94 | 0.94 |
| 2.85M | 88.0 | 0.81 | 0.50 | 0.06 | 0.06 | 1.38 | 1.38 |
| 2.95M | 62.1 | 0.69 | 0.25 | 0.00 | 0.00 | 0.94 | 0.88 |
| 3.05M | 44.9 | 0.56 | 0.19 | 0.00 | 0.00 | 0.75 | 0.62 |
| 3.15M | 59.3 | 0.69 | 0.19 | 0.06 | 0.06 | 0.94 | 0.88 |
| 3.25M | 72.1 | 0.75 | 0.25 | 0.06 | 0.06 | 1.06 | 1.06 |
| **3.35M (last)** | **89.4** | **0.94** | **0.56** | 0.00 | 0.00 | **1.50** | **1.44** |

- Checkpoints: `runs/tp3_recovery_cont/ckpt.pt` = 3.35M (last eval, most cubes: 1.50 max / 1.44 at end);
  `ckpt_best.pt` = 2.85M (success at end 0.06, 1.38 cubes, return 88.0).
22. Still slowly rising under heavy eval noise: the best evals climbed 1.31 → 1.38 → 1.50 mean max cubes
    (2.65M → 2.85M → 3.35M), ≥2 cubes reached 0.56. Full 3-cube success stays at 0–1 of 16 episodes (5 of the
    18 evals had one). Mostly a 2-cube policy; the third cube is still the bottleneck.
23. GPU is now free. Next steps (more demos / episode limit / longer pretraining and online, efficiency branch)
    are open for discussion; see the proposal above.

---

## 2026-09-27 (16:10) — Efficiency branch `feat/qc-efficiency`: faster gradient steps, same training

Goal: bring the TrayPack3 run from ~3 h to ~1 h **without** changing offline/online steps, `num_updates`, batch
size, network size, demos or any hyperparameter. Only how each gradient step runs on the GPU changes.

Diagnosis (from the `tp3_recovery` numbers above): ~112 online and ~89 offline gradient steps/s, i.e. ~9–11 ms per
step, for 512×4 MLPs at batch 512 on 16 px images. The math for one step is only a few ms. The rest is launch/Python
overhead of ~1000 small kernels per step (10 flow Euler passes, backward, 2 Adams, target lerp) plus ~40 small
index kernels per replay sample. `--cudagraphs` (default True in `Args`) was never used by `train_squint_qc.py`,
unlike `train_squint.py`, which wraps its updates in `CudaGraphModule`.

Changes:
1. `qc_agent.py`: the update is split into a critic-only step and a critic+actor step. With `--compile` each is
   `torch.compile`d, and with `--cudagraphs` each is captured by `tensordict.nn.CudaGraphModule`, the same pattern
   as `train_squint.py` (Adam `capturable = cudagraphs and not compile`). `QCAgent.update(b, update_actor)` keeps its
   signature.
2. `qc_agent.py`: the Q term of the actor loss now calls the critic with detached weights
   (`torch.func.functional_call`) instead of toggling `requires_grad_` inside the step. Same gradients, and it is
   safe to compile/capture.
3. `train_squint_qc.py`: the batches for an iteration's 256 updates (and each 256-step offline block) are sampled in
   one vectorised call, capped at ~256 MB per block, instead of one call per update. The buffers do not change
   inside the block, so the sampling distribution is the same. Each batch is still `n_on` replay + `n_off` demo samples.
4. New `examples/bench_qc_update.py`: times eager / compile / compile+graphs updates on synthetic buffers at
   `tp3_recovery` sizes and projects offline + online update time.

Verification (local, CPU):
- Eager: new vs old agent, same weights and seeds, 12 mixed critic/actor updates → **bit-identical** parameters
  (max diff 0.0).
- Compiled (`fallback_random` so the RNG matches eager): max param diff 2e-5 after 12 updates (fused-kernel rounding).
- `train_squint_qc.py` CPU smoke run (TrayPack1, 1 env, offline + online updates) runs end to end.
- Not verified yet: GPU speed-up, CUDA graph capture on the box. Next step on the box:
  `python -m examples.bench_qc_update` (a few minutes), then a full `tp3_recovery`-identical run with a new
  `--exp_name` to compare wall-clock time and eval curves.

Not changed: the per-iteration demo re-jitter (a full pass over the demo image store, ~74 torchvision calls per
iteration) and the evals (16 envs × 300 steps ≈ 24 s each, every 10k offline steps and every 100k online steps,
~12 min per run in total). Both are next in line if the benchmark shows the updates are no longer the bottleneck.

---

## 2026-09-27 (16:40) — collecting 500 more TrayPack3 recovery demos

On the box after pulling commit `1a76ae1` (tmux `collect2`, log `~/collect_tp3_recovery_b.log`):
`N=500 WORKERS=8 START_SEED=5000 OUT=demos/qc/SO101TrayPack3-recovery_b.h5 bash examples/collect_tp3_recovery.sh`
(noise 0.2, miss 0.3, reward v3, 300-step horizon unchanged, max attempts 2500). Seeds start at 5000, so they
don't overlap the first 300 demos (seeds 1000–~1600). Plan: merge with `SO101TrayPack3-recovery.h5` into
800 demos (`examples/merge_qc_demos.py`). RAM with 8 workers: ~10.9 GB used / 4.4 GB available.

### Result of the 500-demo collection (checked 21:10)
- Finished 16:55: 500/500 saved (`demos/qc/SO101TrayPack3-recovery_b.h5`, 326 MB) from 959 attempts (52%;
  113 rejected for exceeding 300 steps). Length mean 249, min 161, max 300. Seeds 5001–6064, all unique, no overlap
  with the first file (seeds 1001–1708). Meta: reward v3, noise 0.2, miss 0.3.
- Full replay verification (`verify_demos --workers 8`): **500/500 passed** (replay reproduces the recording and succeeds).
- Merged: `demos/qc/SO101TrayPack3-recovery800.h5` = first 300 + these 500 (800 demos, 520 MB, meta reward v3,
  `merged_from` 2); static check 800/800 passed.
- The box rebooted at ~18:23 (after collection ended); files unaffected. Disk: 14 GB free.

---

## 2026-09-27 (21:29) — `tp3_rec800`: 800 recovery demos, 200k pretraining, 5M online

**Box environment broke:** someone installed `torch 2.14.0+cu130` into `~/.local` (user site-packages) at 20:41. It shadowed
the squint env's `torch 2.6.0+cu124`, so importing torchvision 0.21 failed (`RuntimeError: operator torchvision::nms does
not exist`); the first launch at 21:24 crashed at import. The `~/.local` install was left alone. Instead:
- Runs use `PYTHONNOUSERSITE=1` (now exported in `examples/collect_tp3_recovery.sh`), so only the conda env is used.
- The squint env had been relying on `~/.local` for many dependencies. Installed them into the env
  (`pip check` plus import errors): gymnasium, mpmath, tyro, typing_inspection, annotated_types, networkx, pygments,
  exceptiongroup, hf-xet, imageio, importlib-resources, jinja2, markdown-it-py, pynput, termcolor. The env now imports
  `train_squint_qc` on its own: torch 2.6.0+cu124, torchvision 0.21.0, CUDA OK.
- Logs moved from `~` into the repo's git-ignored `logs/` (30 files). The script now writes collection and verify logs to
  `logs/` and training logs to `logs/<exp_name>.log`.

**Run** (tmux `tp3rec800`, log `logs/tp3_rec800.log`):
`TRAIN=1 SKIP_COLLECT=1 OUT=demos/qc/SO101TrayPack3-recovery800.h5 EXP_NAME=tp3_rec800 OFFLINE_STEPS=200000
TRAIN_ARGS="--total_timesteps 5000000" bash examples/collect_tp3_recovery.sh`, i.e. 800 demos (198,862 steps), 200k offline
steps, 5M online, 512×4 networks, v3 reward, gamma 0.99, h=5, 16 eval episodes, `ckpt_best.pt` kept.
Loaded 800 demos, "demo rewards and env both use reward_version 3". ETA: offline ~37 min, online ~2.8 h, done ≈ 01:00.
Changes vs `tp3_recovery`: 800 instead of 300 demos, 200k instead of 150k offline steps, 5M instead of 1.5M online steps.

`tp3_rec800` at 21:42: offline pretraining 60k/200k (95.6 steps/s, ~24 min left, online starts ≈ 22:07). Offline evals
10k–50k: 0 cubes (return 3.8–5.7, grasp 0), same as `tp3_recovery` at this stage (it placed 1/16 only at 50k and
80k–130k). BC flow loss 0.28 at 60k (`tp3_recovery` ended offline at 0.24; more varied data fits more slowly),
critic loss 0.53, Q mean 79.

`tp3_rec800` at 22:16: offline pretraining finished (200k steps, ~45 min incl. evals); online started 22:14. The online
phase has only 36k steps so far (391 env steps/s during warm-up; ETA ≈ 01:45 if it stays at that speed).
Offline evals (20 × 16 episodes): 1 cube in 1/16 only at 100k; grasp reward 0.1–1.1 at 130k–200k; otherwise 0.
**Pretraining on 800 demos wasn't better than on 300** (`tp3_recovery` placed 1/16 at 4 offline evals: 50k, 80k, 90k,
130k). Pretraining alone stays a weak prior at 16×16 regardless of demo count (so far); the comparison that matters
is how fast the online phase takes off.

### Offline phase: 800 vs 300 recovery demos (matched steps, 16 eval episodes each)

| | 300 demos (`tp3_recovery`) | 800 demos (`tp3_rec800`) |
|---|---|---|
| evals with ≥1 cube (10k–150k) | 5/15 (1/16 each) | 1/15 (1/16 at 100k); 0 more at 160k–200k |
| mean eval return (10k–150k) | 6.49 | 5.24 |
| mean grasp reward (10k–150k) | 1.05 | 0.09 |
| mean reach reward (10k–150k) | 77.2 | 69.4 |
| BC flow loss @50k / 100k / 150k | 0.261 / 0.265 / 0.240 | 0.282 / 0.273 / 0.273 (0.256 @200k) |
| critic loss @50k / 100k / 150k | 0.438 / 0.286 / 0.235 | 0.561 / 0.365 / 0.343 (0.298 @200k) |

24. **Offline pretraining on 800 demos is slightly worse than on 300, not better.** Likely reason: the same batch size
    (512) spread over 2.7× more data means fewer passes over each demo. At 150k steps: 300 demos (74k steps) ≈ 1,040
    passes, 800 demos (199k steps) ≈ 390 passes; even 200k steps ≈ 515. BC and critic losses are still higher than the
    300-demo run's at every matched step, so the 800-demo model is under-fitted, not saturated. Matching the 300-demo
    run's passes would need ~400k offline steps (~75 min).
25. Both are weak priors either way (at most 1/16 episodes with a cube). The difference is small (4 extra 1/16
    episodes) and within 16-episode noise, but the losses point the same way.

### Online phase: 800 vs 300 demos at matched online steps (22:48, `tp3_rec800` at 723k/5M online, 392 sps, ETA ≈ 01:50)

| Online step | 300: ≥1 cube | 300: ≥2 | 300: 3 | 300: mean max cubes | 800: ≥1 cube | 800: ≥2 | 800: 3 | 800: mean max cubes |
|---|---|---|---|---|---|---|---|---|
| 100k | 0.19 | 0.06 | 0 | 0.25 | 0.19 | 0.00 | 0 | 0.19 |
| 200k | 0.31 | 0.00 | 0 | 0.31 | 0.38 | 0.00 | 0 | 0.38 |
| 300k | 0.25 | 0.06 | 0 | 0.31 | 0.31 | 0.00 | 0 | 0.31 |
| 400k | 0.25 | 0.12 | 0 | 0.38 | **0.62** | 0.06 | 0 | **0.69** |
| 500k | **0.69** | 0.06 | 0 | **0.75** | 0.38 | 0.00 | 0 | 0.38 |
| 600k | **0.75** | **0.31** | 0 | **1.06** | 0.56 | 0.12 | 0 | 0.69 |
| 700k | **0.69** | **0.31** | 0 | **1.00** | 0.50 | 0.12 | **0.06** | 0.69 |

Training rollouts (stochastic policy, many envs, far less noisy than the eval): return at ~306k online 21.0 (300) vs
22.4 (800); at ~613k 31.5 (300) vs 35.4 (800).

26. **No clear winner yet.** 800 demos was ahead at 200k–400k and behind at 500k–700k on the 16-episode eval, while the
    much larger training-rollout sample is slightly ahead for 800. With 16 episodes one episode is 0.06, and swings of
    ±0.2 between neighbouring evals were normal in the 300-demo run, so these differences are within noise.
27. 800 demos got its first full 3-cube success at 700k online steps; the 300-demo run's first was at 1.5M.
    That's one episode, so it's only a hint.
28. Decision: not clearly worse, so the run continues to 5M (the 400k-offline restart is on hold). The next comparison
    point is 1–1.5M online steps, where the 300-demo run reached ≥2 cubes 0.31–0.38.

### `tp3_rec800` at 23:42 (1.89M/5M online, ~379 sps, ETA ≈ 01:59): new best, and 800 demos now ahead

| Online step | 300: ≥1 | 300: ≥2 | 300: 3 | 300: max cubes | 800: ≥1 | 800: ≥2 | 800: 3 (= success at end) | 800: max cubes |
|---|---|---|---|---|---|---|---|---|
| 800k | 0.81 | 0.25 | 0 | 1.06 | 0.69 | 0.25 | 0 | 0.94 |
| 900k | 0.81 | 0.31 | 0 | 1.12 | 0.69 | 0.19 | 0 | 0.88 |
| 1.0M | 0.75 | 0.00 | 0 | 0.75 | 0.69 | 0.38 | 0.06 | 1.12 |
| 1.1M | 0.56 | 0.06 | 0 | 0.62 | 0.88 | 0.38 | 0 | 1.25 |
| 1.2M | 0.56 | 0.12 | 0 | 0.69 | 0.62 | 0.12 | 0 | 0.75 |
| 1.3M | 0.69 | 0.00 | 0 | 0.69 | 0.88 | 0.19 | 0 | 1.06 |
| **1.4M** | 0.81 | 0.31 | 0 | 1.12 | 0.75 | 0.38 | **0.12** | 1.25 |
| 1.5M | 0.62 | 0.12 | 0 | 0.75 | **0.94** | 0.38 | 0.06 | **1.38** |
| 1.6M* | 0.81 | 0.12 | 0 | 0.94 | 0.69 | 0.19 | 0 | 0.88 |
| 1.7M* | 0.44 | 0.12 | 0 | 0.56 | 0.50 | 0.12 | 0.06 | 0.69 |
| 1.8M* | 0.50 | 0.06 | 0 | 0.56 | 0.81 | 0.06 | 0 | 0.88 |

\* The 300-demo numbers after 1.5M online come from `tp3_recovery_cont` (the resumed run, which had its restart dip), so
the comparison there isn't like-for-like.

29. **New best TrayPack3 result: 2/16 full successes (3 cubes, held to the end) at 1.4M online steps**, the first eval of
    any run above 1/16. `ckpt_best.pt` = this eval (success at end 0.125, 1.25 mean max cubes, return 71.4).
    Also ≥1 cube 0.94 at 1.5M, the highest yet.
30. **Averaged over 700k–1.5M (9 evals each, same online steps, both uninterrupted), 800 demos is ahead:** ≥2 cubes 0.27
    vs 0.16, mean max cubes 1.04 vs 0.87, full-success episodes 5 vs 0 (of 144 each). Averaging many evals reduces the
    16-episode noise; this is the first clear evidence that more demos help.
31. Training rollouts keep improving (return 48 → 59 → 62 at 1.1M → 1.7M → 2.0M logger steps, success_once ~1%).
    Critic loss is stable at 3.7–5.4.
32. The third cube is still the bottleneck (≤ 2/16), and evals still swing a lot (e.g. 1.5M → 1.7M: max cubes 1.38 → 0.69).

### `tp3_rec800` stopped at 00:57 (2026-09-28), 3.48M/5M online steps: plateau

Last evals (online step: ≥1 / ≥2 / 3 cubes / mean max cubes): 2.2M 1.00/0.31/0.06/1.38; 2.7M 0.88/0.44/0.06/1.38;
3.0M 1.00/0.31/0/1.31; 3.2M 0.88/0.44/0.06/1.38; 3.4M 0.69/0.31/0/1.00.

Window averages (reduce 16-episode eval noise):

| Online steps | evals | ≥1 cube | ≥2 cubes | 3 cubes | success at end | mean max cubes |
|---|---|---|---|---|---|---|
| 0.7–1.5M | 8 | 0.71 | 0.25 | 0.031 | 0.031 | 0.99 |
| 1.5–2.5M | 10 | 0.81 | 0.24 | 0.019 | 0.013 | 1.07 |
| 2.5–3.4M | 10 | 0.87 | 0.26 | 0.019 | 0.019 | 1.15 |

Training rollouts: return 62.5 (2.35M logger) → 66.9 → 67.7 → 67.7 → 68.4 (3.58M), success_once 1–2%; critic loss 4.6–6.9.

33. **Plateau:** the first cube keeps improving slowly (0.71 → 0.87), but the second (~0.25) and third (~0.02) cubes have
    been flat for ~2M online steps, and training return has stalled at ~68 since ~2.6M logger steps. More online steps
    at these settings won't finish the task; something about the later stages has to change.
34. Checkpoints: `runs/tp3_rec800/ckpt_best.pt` = 1.4M online (2/16 full successes, 1.25 mean max cubes, return 71.4);
    `ckpt.pt` = 3.4M online (last eval).
35. Best TrayPack3 result so far: `tp3_rec800` 1.4M, success at end 0.125 (2/16). Needs re-evaluation on many fixed-seed
    episodes before it goes in the paper.

Open questions for the next step (not started):
- Why the second and third cube stall: look at eval videos of the late policy (where does it fail after cube 1: grasp,
  transport, placement next to an already-placed cube, or running out of time with the 300-step horizon?).
- Candidates: episode limit 300 → 400; gamma 0.995 (the third cube's reward is ~200 steps away); reward shaping for
  cubes 2–3; fixed-seed 64-episode evals and a checkpoint re-eval script (see the variance discussion above).

---

## 2026-09-28 (01:30) — next task: Tower3 vs Rearrange3 (same pipeline), pre-check

Test collection on the box: 8 recovery demos each (noise 0.2, miss 0.3, seeds 1000+, 8 workers), current env rewards.

| | SO101Tower3Cube-v1 | SO101Rearrange3-v1 |
|---|---|---|
| horizon | 300 | 400 |
| solver success (recovery) | 8/23 (35%) | 6/50 (12%) |
| demo length | 229–300 (mean ~263; tight against the 300 limit) | 234–329 |
| main failures | success check false after stacking (8), pick IK (6) | only 1–2 of 3 correct (17), pick reach/grasp (27) |
| reward drops > 1 per demo | 2–5 | 1–4 |
| worst drop (raw) | −2.5 to −3.4 (max reward 15) | −1.6 to −3.1 (max reward 11) |

Replaying demos and inspecting the drop steps (same kinds of problems as TrayPack v1):
- **Tower, release dip:** releasing a placed cube loses the grasp term (−2) with nothing to replace it (large at base,
  small on top: −1.99, −1.94).
- **Tower, flickering stage flags:** `medium_supported` 1→0 after release (−2.5) and `base_placed` 1→0 while carrying
  (−3.4), so the staged reward briefly takes back a completed stage.
- **Tower, table contact −3** while approaching the large cube (three −3 drops in one demo), the same penalty that
  discouraged grasping in TrayPack (v3 reduced it to −0.3).
- **Rearrange, `num_correct` flicker:** a correctly placed cube briefly counts as not in its pocket (3→2, 2→1, 1→0:
  −2.6 to −3.1), including right at task completion.
- **Rearrange, buffer move:** releasing a cube into the buffer loses the +1 grasp bonus (−1, small, by design).

Neither env writes per-term reward info (`rew_*`), so `train_rew/*`/`eval_rew/*` plots won't be available until added.

---

## 2026-09-28 — Tower3 reward v2 (TrayPack-v3-style, no dips)

Decision: Rearrange3 dropped for now (12% solver success with recovery noise, 4 moves incl. buffer). Next task is Tower3,
so the paper has two long-horizon tasks (TrayPack3 + Tower3).

**Why the v1 reward dips (replaying demos, printing each stage check around the drop steps):**
- Releasing a placed cube: the grasp term (+2) disappears with nothing in its place (−1.9 to −2.0).
- `medium_supported` = geometry & (contact force ≥ 0.05 or both cubes still). A resting medium cube presses with only
  **0.039**, so the check relies on stillness alone, and one velocity blip (0.022 > 0.02) switches the stage off (−2.5).
- `base_placed` needs the large cube within 4 mm of the table; while it's lowered (held 4–9 mm up) it switches on and off (−3.4).
- The small cube hovers 7.5 mm above the medium one while held, so it only counts as placed after release, and then
  nothing pays until the 1 s success dwell finishes.
- Yaw jitter while a cube is lowered/pressed breaks the 2 mm corner margin of the support check for single steps.
- Table contact −3.

**Tower reward v2** (`envs/tower.py`, `reward_version=2`; default stays 1 so old runs reproduce):
- Stage k = cube k placed on its support (large at the tower mark, medium on large, small on medium). The check is
  geometric only and applies whether held or not, with reward-only tolerances (height ≤ 1 cm, corner margin 5 mm), and a
  stage counts only if all earlier stages are placed.
- A placed stage is worth 3, +1 once its cube is released. `reach` (0.5), `grasp` (1) and `place` (1, held cube only)
  are paid for the first unplaced cube in stack order. So the reward rises through each placement (≤ 2.5 before,
  ≥ 3 after, 4 when released).
- Table contact −0.3. Success uses the unchanged strict checks and fills the reward to the max (15). Terms are written to
  `info["rew_*"]`, so `train_rew/*` and `eval_rew/*` are logged like TrayPack.
- `_support_check` got optional `z_tol` / `margin` arguments (defaults unchanged). `train_squint_qc.py --reward_version`
  now also accepts Tower.

**Check** (Tower3 recovery demos, noise 0.2, miss 0.3):

| | reward drops > 1 per demo | worst drop |
|---|---|---|
| v1, 8 demos | 2–5 | −2.5 to −3.4 |
| v2 (2 mm margin), 8 demos recorded with v2 | 0 in 6, 1 in 2 | −6.0 (one-step yaw flicker), −2.85 |
| **v2 final (5 mm margin), the same 8 + the 8 v1 demos re-scored by replay** | **0 in 16** | **−0.98** |

Next: collect 800 Tower3 recovery demos with `--reward-version 2`.

---

## 2026-09-28 (01:39) — two-task protocol started: TrayPack3 + Tower3 (`examples/run_two_task_qc.sh`)

Same settings for both tasks: 500 recovery demos, 250k offline steps (≈1,000 passes over 500 demos, like the 300-demo run
that fitted well), **2M online steps**, 16 eval episodes at **fixed eval seed 100** (same layouts at every eval),
512×4 networks, gamma 0.99, h=5, 16×16 images. TrayPack3: reward v3, the first 500 demos of
`SO101TrayPack3-recovery800.h5` (`--max_demo_trajs 500`). Tower3: reward v2, 500 new demos (seeds 1000+) collected on
3 workers while TrayPack3 trains; Tower3 trains after both finish. ETA ≈ 2h10 per task.

Started on the box in tmux `twotask`: `bash examples/run_two_task_qc.sh` (logs: `logs/run_two_task_qc.out`,
`logs/tp3_500.log`, `logs/collect_tower3_recovery500.log`, later `logs/tower3_500.log`; runs `runs/tp3_500`, `runs/tower3_500`).

Code for this protocol (uncommitted):
- `train_squint_qc.py --eval_seed`; `train_squint.py` `evaluate()` resets the eval envs with it. Checked on the GPU sim:
  the same seed gives identical layouts for both tasks (also after stepping), seed 100 ≠ seed 1, and no seed gives
  random layouts. Seeds 1 and 2 happen to give identical layouts, so 100 is used.
- `--reward_version` accepted for Tower (`envs/tower.py` v2 above).

Launch problems before this start:
- 01:32: TrayPack3 crashed at start (`QCAgent.__init__() got an unexpected keyword argument 'compile'`). The laptop's
  working tree also has the efficiency chat's unfinished edits (`qc_agent.py`, the batched sampling in
  `train_squint_qc.py`, `examples/bench_qc_update.py`); copying `train_squint_qc.py` to the box brought them along
  without the matching `qc_agent.py`. The box now runs the committed `train_squint_qc.py` plus only this protocol's
  edits (eval seed, Tower reward version). The efficiency edits are untested and would confound this comparison.
- 01:36: relaunched from a script in the git-ignored `logs/`; stopped after ~2 min and moved into the repo as
  `examples/run_two_task_qc.sh`. Partial outputs of both attempts were deleted.
- 01:42: the 01:39 start crashed. To check Tower3 training before it runs unattended, I started a small GPU smoke test
  next to the running TrayPack3 job. RAM ran out, the kernel OOM-killed the smoke test (01:42:45, as intended by its
  `oom_score_adj` 1000), and one second later TrayPack3 died with a PhysX CUDA error 700 ("illegal memory access").
  A second GPU sim next to a training run isn't safe; smoke tests only run on an idle box from now on. Log kept in
  `logs/_failed/tp3_500_physx_crash.log`.
- Tower3 smoke test on the idle box (8 v2 demos, 16 envs, 4k steps): exit 0, "demo rewards and env both use
  reward_version 2". Logs `eval/success_*`, `eval/return`, `eval/{base_placed,medium_supported,small_supported}_once`,
  `eval_rew/{reach,grasp,place,stacked,released,success_bonus,table_pen,total}`; saves `ckpt.pt` + `ckpt_best.pt`.
- **01:45: restarted cleanly** (tmux `twotask`, `bash examples/run_two_task_qc.sh`). 01:47: TrayPack3 loaded 500 demos
  (124,067 steps), reward v3 check OK, offline at 111 steps/s; Tower3 collection running on 3 workers. RAM 3.5 GB free,
  GPU 6.3 GB. ETA: TrayPack3 done ≈ 04:00, Tower3 ≈ 06:15 (if collection finishes first).
- 01:48: the 01:45 start was stopped again at the user's request: **strictly one job at a time** on the box (collect the
  demos first, then train), which is safer for the GPU. `examples/run_two_task_qc.sh` rewritten to be sequential:
  (1) collect 500 Tower3 demos on 8 workers (skipped if the file exists), (2) train `tp3_500`, (3) train `tower3_500`;
  it stops if a step fails. Partial outputs of the stopped start were deleted.
- **01:49: started** (tmux `twotask`). Step 1 running alone: 8 collectors, no training, RAM 5.4 GB free, GPU 3.4 GB
  (collector rendering). Expected: collection ~40 min, then TrayPack3 ~2h10 (≈ 04:40), then Tower3 ~2h10 (≈ 06:50).

### Results of the two-task protocol (all 3 steps finished: collection 02:19, `tp3_500` 04:32, `tower3_500` 06:42)

(Checked 09:55 over the other box address `sra@10.1.205.86`; that link stalls on outputs larger than ~1 packet, so results
were pulled in small chunks.)

**`tp3_500`** (TrayPack3, 500 demos, 250k offline, 2M online, fixed eval seed 100, 16 episodes):
- Offline: 1 cube in 1/16 at 110k, 120k, 150k only (same weak prior as before).
- Online (≥1 / ≥2 / 3 cubes / mean max cubes): 0.3M 0.44/0/0/0.44; 0.4M 0.75/0.12/0/0.88; 0.6M 0.69/0.25/0.06/1.00;
  0.8M 0.50/0.06/0.06/0.62 (success at end 0.06); 1.0M 0.75/0.25/0/1.00; 1.2M 0.88/0.25/0.06/1.19 (success at end 0.06);
  1.8M 0.75/0.31/0/1.06; **2.0M 0.94/0.38/0/1.31 (return 77.3, best of the run)**.
- Mean of the last 10 evals (1.1–2.0M): ≥1 cube 0.73, ≥2 cubes 0.19, 3 cubes 0.006. `tp3_rec800` over the same
  online steps: 0.77 / 0.23 / 0.024. Same level: 500 demos didn't move the ceiling (eval layouts differ, fixed vs random).
  Takes off at 0.3–0.4M like the earlier runs.

**`tower3_500`** (Tower3, reward v2, same settings): **failed, never learned to grasp.**
- Offline evals: large cube placed 1/16 once (60k); otherwise 0. Online: large cube placed 1/16 at 0.3M, 0.9M, 1.9M and
  2/16 at 1.6M; medium/small stacked and success 0 throughout. Return 1.7–12 (TrayPack3 reached 77).
- Eval reward terms: reach 31–49 per episode, **grasp 0.0–0.1**, place/stacked/released 0. The policy moves around but
  never closes on a cube, from offline pretraining through 2M online steps.
- Training rollouts (1024 envs, stochastic): return 2.9–4.9, a few chance placements (`train_rew/stacked` up to 20.7 at
  2.09M), grasp ≤ 1.3 per episode.
- BC flow loss 0.246 offline / 0.223 online (same as TrayPack3, so the demos are fitted), critic loss 2.1, Q mean 57–73.
- Unlike TrayPack3, the online phase never discovered grasping; the cause is not known yet (to diagnose next).

### Check: do the Tower3 demos match the training env? (TrayPack3 as control, idle box, `logs/envcheck.txt`)

| | TrayPack3 demos | TrayPack3 train env | Tower3 demos | Tower3 train env |
|---|---|---|---|---|
| first-step state (12-d) | [0.01, 0, −0.01, 1.57, −1.56, 1.04] ×2 | [0, 0, −0.01, 1.57, −1.58, 1.04] ×2 | [0, −0.01, −0.01, 1.57, −1.57, 1.05] ×2 | [0, 0, −0.01, 1.57, −1.58, 1.04] ×2 |
| 16×16 image mean / std | 71.8 / 75.6 | 71.8 / 75.6 | 71.5 / 75.7 | 71.4 / 75.7 |
| obs shapes | rgb 128² → 16², state 12 | same | same | same |
| action range | [−1, 1] ×6 | [−1, 1] ×6 | [−1, 1] ×6 | [−1, 1] ×6 |

- Same seed, collector (CPU) vs training env (GPU): the cube layouts differ by 5–15 cm for **both** tasks, so an open-loop
  replay of the demo actions fails on GPU for both (0/5 each), while on CPU 4/5 succeed for both (the full CPU verifier
  passed 500/500). This is the known CPU/GPU seeding difference, and it doesn't matter for training: the demos are
  used as (observation, action) data, not replayed.
- **Conclusion: the Tower3 demos match the training env exactly as well as the TrayPack3 demos, which train fine.** An
  env/demo mismatch is ruled out as the cause of the Tower3 failure. Next suspects: the task itself (larger cube needs
  a wider, more precise grasp at 16×16) and what the policy does with the gripper (to check in the eval videos and
  the policy's gripper actions vs the demos').

---

## 2026-09-28 (11:30) — plan: sanity check on the original 2-cube `SO101StackCube-v1` first

Decision (user): before fixing Tower3, check that the pipeline trains properly on the original Squint task
`SO101StackCube-v1` (2 cubes, 22–35 mm, 50-step limit) with 100 demos, then return to our tasks.
- `examples/collect_new_tasks_demos.py`: `SO101StackCube-v1` added to `SOLVERS`/`SOLVER_MODULES` (existing
  `solutions/stack_cube.py`).
- Test collection on the laptop (CPU only, 6 workers) was stopped after 6 min with 0 demos saved: without a GPU the
  128 px camera is rendered in software, far too slow. **Demo collection has to run on the GPU box too** (there,
  rendering takes ~20% of collection time).
- Still to check on the box before training: solver success and demo length vs the 50-step limit (earlier clean
  StackCube demos averaged ~51 steps), reward dips, demo/env match, pretraining shows grasping.

### StackCube step 1 (test collection on the box, 12:04): demos don't fit the 50-step limit
- 20 clean demos, 8 workers (`examples/collect_tp3_recovery.sh` with `ENV_ID=SO101StackCube-v1 REWARD_VERSION=none`,
  new option: `none` leaves out `--reward-version` for envs without reward versions): 0 saved from 104 attempts,
  100 rejected as over 50 steps. The run stopped there (merging 0 demos fails), so the recovery half didn't run.
- Measured with the limit raised to 200 (16 seeds each, `/tmp/stacklen.py` on the box):

| demos | solver success | length min / median / mean / max |
|---|---|---|
| clean | 15/16 | 57 / 71 / 70 / 86 |
| recovery (noise 0.2, miss 0.3) | 14/16 | 61 / 82 / 89 / 153 |

- The scripted StackCube demos are ~70 steps with the current planner (vs ~51 with the older one of 2026-09-24):
  the grasp retry, jaw-clearance and misaligned-grasp logic added later make the motion slower and more careful.
  The original Squint RL solves StackCube within 50 steps, so the 50-step limit is feasible for a policy, but our demos
  can't fit it.
- Where the ~70 steps go (4 demos, steps per solver phase): reach + descend onto the cube 13–24 (final descent at half
  speed), close gripper 6 (gripper limit 0.2 rad/step + 3 settle steps), lift + carry + lower 21–29 (lift at 70%,
  lowering at half speed), pause 2 per placement correction, open gripper 6, back off + stand still 6. A trained policy
  moves every joint at full speed and overlaps gripper motion with arm motion, so it fits in 50.
- **Solver `FAST` profile** (`SO101GraspSolver.FAST`, collector `--fast`, off by default so other tasks' demos are
  unchanged): full-speed descents, lifts and retreats (`DESCENT_SPEED` 0.5 → 1.0, `LIFT_SPEED` 0.7 → 1.0), 1 settle step
  after gripper moves (`GRIP_SETTLE` 3 → 1) and after placement corrections (`SERVO_SETTLE` 2 → 1). The meta records
  `fast_solver`. Not measured yet (shell tool unavailable at the time); next: demo lengths with `--fast`, clean and recovery.
