# SO-101 Long-Horizon Manipulation Benchmark: Task Card

Multi-stage, long-horizon tasks (100–400 control steps, 1–3 objects) for visual RL on the low-cost
**SO-101** arm, built on ManiSkill 3 and on top of Squint's single-stage tasks. All tasks are in
simulation; no real-robot results exist yet for the multi-stage tasks.

Code: `envs/` (one file per task family, registered on `import envs`). Demos: `examples/collect_qc_demos.py`,
`examples/collect_new_tasks_demos.py`. Training: `train_squint_qc.py` (QC-FQL), `train_squint.py` (Squint SAC baseline).

---

## 1. Shared setting (identical across all tasks)

| Item | Value |
|---|---|
| Robot | SO-101 5-DoF arm + parallel gripper, one arm, rigid objects, top-down grasps only |
| Simulator | ManiSkill 3 / SAPIEN; GPU PhysX for training, CPU PhysX for demo recording |
| Control | `pd_joint_target_delta_pos`, sim 100 Hz, control 10 Hz (episode seconds = steps / 10) |
| Action | `Box(-1, 1, (6,))`, normalized joint-target deltas: ±0.1 rad per step (arm), ±0.2 rad (gripper) |
| Observation | 16×16 wrist RGB (rendered 128×128, area-downsampled; 32/64 configurable) + 12-d proprio (noisy qpos 6 + target qpos 6) |
| Not in obs | Object poses, grasp/success flags, stage index, goal identity (all only in `info`) |
| Background | Fixed overlay image replaces everything except robot + objects (for sim-to-real) |
| Reward modes | `sparse` (= success), `dense`, `normalized_dense` (dense / max, ≤ 1 per step; default) |
| Goals | Fixed and visibly marked (coloured plates, tray / pocket floors) — never hidden randomized goals |
| Reset | Starts from a fully unsolved task; rejection sampling removes overlaps, unreachable spots and pre-solved layouts |
| Success | All conditions met, objects released and static, robot static; most tasks also require a 1 s (10-step) dwell |
| Metrics | Main: `success_at_end`. Also `success_once` and per-task partial-progress flags (e.g. cubes placed) |
| Eval protocol | Deterministic policy, 64 episodes on fixed seed 100 (same layouts for every method); demo seeds disjoint (1000+) |
| Domain randomization | Off for the benchmark numbers (cube size, friction, colour, camera DR available per env) |

Reward rules applied to every task (lessons from the paper):
1. **No dips along a correct solution**: a cube seated in its goal counts even while still held, so releasing it never lowers reward.
2. **No farmable bonuses**: every term is state-based; cycling an action removes the bonus.
3. **Small contact penalties**: large table-contact penalties stop the policy from grasping (fingertips are mm above the table).
4. **Exploit latches**: progress counts only if reached the intended way (e.g. a cube must be *grasped* off the tower, not knocked off).
5. Per-term reward logged as `info["rew_<term>"]`; terms sum exactly to the reward.

---

## 2. Tasks

| Task | Env ID | Steps (s) | Objects | Stages | Status in paper |
|---|---|---|---|---|---|
| **Unstack3** (main) | `SO101Unstack3Cube-v1` | 150 (15) | 3 cubes in a tower | 2 picks + 2 places | Solved: 0.89 best / 0.77 final (QC-FQL), 0.00 Squint |
| TrayPack1 | `SO101TrayPack1-v1` | 150 (15) | 1 cube + tray | 1 pick-place | Solved: 1.00 best / 0.81 final |
| TrayPack2 | `SO101TrayPack2-v1` | 200 (20) | 2 cubes + tray | 2 pick-places | Not reported |
| TrayPack3 | `SO101TrayPack3-v1` | 300 (30) | 3 cubes + tray | 3 pick-places | Open: ≥1 cube 0.94, full ≈1–3% |
| Tower3 | `SO101Tower3Cube-v1` | 300 (30) | 3 size-ordered cubes | 3 pick-places, ordered | Open: never grasps |
| Tower2 (control) | `SO101Tower2Cube-v1` | 200 (20) | 2 cubes | 2 pick-places | Not reported |
| Rearrange2 | `SO101Rearrange2-v1` | 300 (30) | 2 cubes, 3 pockets | swap via buffer (≥3 moves) | Demos only |
| Rearrange3 | `SO101Rearrange3-v1` | 400 (40) | 3 cubes, 4 pockets | 3-cycle via buffer (≥4 moves) | Demos only |
| StackCube (Squint) | `SO101StackCube-v1` | 50 (5) | 2 cubes | 1 pick-place | Open: 0–0.06 |
| Lift (Squint) | `SO101LiftCube-v1` | 50 (5) | 1 cube | 1 pick | 0.94 (with DR, deployment policy) |

Also in `envs/` (not in the paper's results): `SO101Stack3Cube-v1` (150 steps, build a 3-cube stack),
`SO101Place3Cube-v1` (250 steps, place 3 cubes).

### 2.1 Unstack3 — `envs/unstack3.py` (main task)
- **Goal:** take a 3-cube tower apart: top cube A (red) onto the table, then middle cube B (black) onto the table; base C (red) must stay put.
- **Init:** tower xy uniform in a 10×10 cm box centred at (0.25, 0) m from the robot base; random yaw; ±5° twist per level.
- **Cubes:** A, B 22–28 mm; C 25–32 mm (midpoint sizes when DR is off). Colours match the real deployment cubes.
- **Success:** A and B each (i) rest on the table (|z − half| ≤ 5 mm), (ii) are separated from every other cube
  (centre distance ≥ √2·(h₁+h₂) + 5 mm), (iii) were *grasped while still stacked* (latched); C drifted ≤ 2 cm xy;
  A, B static (≤ 0.02 m/s) and untouched; robot static.
- **Reward (0–18, normalized by 18):** per stage: reach 0–2, carry (lift above tower, move ≥ 6 cm clear) 3–5,
  on table and held 4–7, released 7–8; stage A lock-in +9; success = 18.
  Penalties: table contact −6, base drift −2·tanh(20·d), tower knocked −3 per step.
- **Diagnostics in `info`:** `is_itemA_on_table`, `is_itemB_on_table`, `itemA_picked_clean`, `tower_knocked`, `base_drift`.

### 2.2 TrayPack{1,2,3} — `envs/tray_pack.py`
- **Goal:** put N cubes (R, G, B) into the tray compartment whose floor/rim colour matches; any order, any yaw.
- **Tray:** 1×3 shallow printed tray at (0.33, −0.055); compartments 42 mm inner, walls 4 mm thick × 10 mm high, floor 5 mm.
- **Cubes:** 20–24 mm; start outside the tray (≥ 7.5 cm clearance), ≥ 5 cm apart, reach radius ≤ 0.38 m.
- **Success per cube:** footprint inside its compartment (xy tol 3 mm), seated on the floor (z tol 4 mm), not
  stacked/on a rim, static, released. All N correct + robot static, held for a 1 s dwell.
- **Reward (v3, default for results):** per cube: 0.5·reach (nearest unplaced cube only) + place (held cube only) + grasp;
  seated cube 3 (even while held); +1 per correct cube; table contact −0.3, tray contact −0.1.
- **Diagnostics:** `per_object_correct`, `num_correct`, `complete`, `complete_stable`.

### 2.3 Tower3 / Tower2 — `envs/tower.py`
- **Goal:** build a size-ordered tower (large blue base, medium red, small green on top) on a fixed yellow 70×70 mm marker at (0.30, 0.08).
- **Cubes:** large 32–36 mm, medium 25–29 mm, small 19–23 mm (non-overlapping ranges); large starts ≥ 7 cm from the marker.
- **Success:** large on marker (xy ≤ 12 mm, z ≤ 4 mm); each upper cube fully supported by the one below
  (xy offset ≤ half-size difference + 6 mm, z ≤ 4 mm, plus contact or static); all released and static; 1 s dwell.
- **Diagnostics:** `base_placed`, `medium_supported`, `small_supported`, `tower_complete`, `stack_lost`.

### 2.4 Rearrange2 / Rearrange3 — `envs/rearrange.py`
- **Goal:** permute cubes between shallow pockets using one empty neutral buffer pocket.
  Rearrange3: pockets P0–P2 hold A, B, C; goal B, C, A. Rearrange2: P0, P1 hold A, B; goal B, A.
- **Pockets:** 42 mm inner, walls 4 × 10 mm; floor/rim colour marks the *destination* cube, so the goal is visible.
- **Success:** every goal pocket holds its assigned cube (seated, static, released), buffer empty, nothing stacked, robot static, 1 s dwell.
  Any valid sequence is accepted (none is hard-coded).

---

## 3. Demonstrations (shipped with the benchmark)

- Scripted solver: custom IK planner for SO-101 (no mplib), per-task pick → carry → place scripts, grasp verification + retries.
- Recorded in the **exact training format**: HDF5 with `obs/rgb`, `obs/state`, `actions`, `rewards`, `terminated`, `truncated` + metadata.
- **Recovery demos:** Gaussian action noise 0.2 on arm actions along planned paths, plus deliberate missed grasps
  (p = 0.3) that the solver detects and corrects. Executed actions are recorded, so replay is exact.
- **FAST profile** for short horizons (Unstack3, StackCube).
- Verification: every demo is replayed from its seed on CPU and must end in success (500/500 for Unstack3, TrayPack3, Rearrange2).
- Typical sets: Unstack3 500 recovery demos (~5 min on 8 CPU workers, mean 123 steps); TrayPack3 800 recovery demos (median 250 steps).

---

## 4. Minimal usage

```python
import gymnasium as gym
import envs  # registers all SO-101 tasks

env = gym.make(
    "SO101Unstack3Cube-v1",
    num_envs=1024,
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_target_delta_pos",
    reward_mode="normalized_dense",
    sim_backend="gpu",
)
# wrap like train_squint.py: FlattenRGBDObservationWrapper(rgb, state) -> DownsampleObsWrapper(16)
#   -> ManiSkillVectorEnv(record_metrics=True); ColorJitterWrapper for training only
```

Report per run: env ID, horizon (steps + seconds), reward mode, demo count/type, seeds, eval episodes, `success_at_end`.

---

## 5. Known limits / still to validate

- Simulation only; physical props (cubes, printed tray/pockets, marker plate) and wrist-camera match still need real-robot validation.
- Paper results are single-seed; `ckpt_best` is selected on the eval itself (re-eval on fresh seeds pending).
- CPU-vs-GPU PhysX seeding differs: demos are (obs, action) data, not open-loop replayable on GPU.
