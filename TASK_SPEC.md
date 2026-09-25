# SO-101 Long-Horizon Task Specification (tasks-and-data-loaders)

Three task families for visual RL with longer manipulation sequences, plus
shorter controls. All use one SO-101 arm + two-finger gripper, rigid objects,
simple collision geometry reproducible physically (cubes, shallow printed
trays/pockets, flat marker plates). No dexterous reorientation, tight
insertion, deformables, or second arm. Upright objects, accessible top-down grasps.

Controller (explicit, unchanged): `pd_joint_target_delta_pos` (default).
Action: normalized delta in [-1,1]^6 → ±0.1 rad arm / ±0.2 rad gripper per
10 Hz control step. Sim 100 Hz, control 10 Hz. Episode seconds = steps/10.
Do NOT assume measured joint-position differences equal target-delta actions;
a future real-demo adapter must record commanded joint targets + controller
state (`target_qpos`), control frequency, and camera config.

Policy observation (deployment): wrist RGB (render 128×128, area-downsampled
to 16×16 default; 32×32/64×64 configurable via `DownsampleObsWrapper` /
`--image-size`) + robot state 12-dim (`noisy_qpos` 6 + `target_qpos` 6).
Verified: `obs_mode="rgb+segmentation"` gives `extra={}`, flattened
`rgb (128,128,3) uint8 + state (12,) float32`. No object poses, success/grasp
flags, stage indices, or target identities in policy obs. Those live in `info`
(evaluate) and in `extra` only when `obs_mode` includes `state` (for asymmetric
critics). Simulator state is used internally for rewards/success/demos/diagnostics.
Goals are fixed + visibly marked (tower plate, tray compartment floor/rim
colors matching assigned cubes, pocket floor/rim colors marking destination);
no randomized goals without observable markers.

Rejection sampling (all tasks): overlapping objects, unreachable placements
(radius > max), objects intersecting fixtures (tray/pockets clearance), and
accidentally completed goals (initial success) are resampled (up to 60 tries).

Dwell: complete stable condition must persist `dwell_time` (default 1.0 s =
10 control steps, configurable). Per-env counters reset in
`_initialize_episode(env_idx)` for correct batched partial resets. Primary
metric: sustained full-task completion (`success` after dwell); also report
`success_at_end`/`success_once` via `ManiSkillVectorEnv(record_metrics=True)`,
plus diagnostic partial progress. Seeded resets; use disjoint demo/eval seeds.
Default eval starts from complete unsolved task; partial-start curricula (if any)
must be separate explicitly-reported options. Keep rewards/obs/init/horizons
identical across methods in a comparison. Dense shaping guides reaching/
grasping/placement but never replaces success; no repeatable event bonuses
(all bonuses state-based, so cycling removes them). Sparse reward = `success`
(+1) via default `compute_sparse_reward`.

## Task A: size-ordered tower — `envs/tower.py`

IDs: `SO101Tower3Cube-v1` (200 steps = 20 s), `SO101Tower2Cube-v1` (150 = 15 s, control).
Cubes (strict ordering via non-overlapping half ranges):
- large (base, blue): half [0.016,0.018] (32–36 mm), nominal 0.017
- medium (red): half [0.0125,0.0145] (25–29 mm), nominal 0.0135
- small (green, 3-cube only): half [0.0095,0.0115] (19–23 mm), nominal 0.0105
Two-cube uses large+medium subset, no rescaling.
Tower target: fixed `(0.30, 0.08)` (configurable `tower_xy`), marked by 70×70×2 mm
yellow visual-only plate (printable, no collision). Large starts ≥0.07 m away
(`min_tower_clearance`); all cubes start separately on table, ≥0.055 m apart,
radius ≤0.38 m. Workspace `spawn_box_pos=(0.3,0)`, half 0.1 (configurable).
Friction [0.1,0.5], density 200 (configurable).

Success (all required, then dwell 1 s):
- large at tower: xy ≤0.012 m (`tower_tol_xy`), |z-half| ≤0.004 m
- medium on large / small on medium: full-support xy
  `offset ≤ (half_bot-half_top)+0.006`, |dz-(half_top+half_bot)| ≤0.004,
  plus (contact force ≥0.05 N OR static ≤0.02 m/s); height/centre-distance alone insufficient
- released (no robot touch/grasp on any cube), all velocities ≤0.02, robot static
Diagnostics: `base_placed`, `medium_supported`, `small_supported`,
`tower_complete`, `tower_complete_stable`, `stack_lost` (latched ever-complete then lost),
`dwell_count`, `base_xy_dist`.

Rewards: sparse `success`; dense staged reaching/placing (max 15 for 3-cube,
10 for 2-cube) + `normalized_dense` = dense/max. State-based, no farming.

Physical props: 3 cubes (above sizes, ±1 mm, distinct colors), 70 mm yellow
marker plate. Needs physical validation: gripper clearance on 36 mm large cube,
tower stability at 82 mm height, marker visibility in wrist view, friction.

## Task B: assigned-compartment packing — `envs/tray_pack.py`

IDs (same tray geometry + tolerances): `SO101TrayPack1-v1` (100 = 10 s),
`SO101TrayPack2-v1` (150 = 15 s), `SO101TrayPack3-v1` (200 = 20 s).
Objects: cubes half [0.011,0.013] (22–26 mm, nominal 0.012), colors R/G/B for
compartments 0/1/2. Variants use first N objects/compartments.
Tray (shallow, printable): 1×3, compartment inner 0.035 m, wall 0.005 thick ×
0.015 high, floor 0.005 thick. Total ≈0.125×0.045×0.02 m. Fixed at
`(0.32,-0.02)` (configurable). Floors colored R/G/B matching assigned cubes +
matching rim frames on wall tops (visible assignment). Outer/dividers white.
Cubes start outside tray (clearance radius +0.02), ≥0.05 m apart, radius ≤0.38 m.
No orientation requirement (any yaw ok).

Success per object: centre within `(inner/2-half)+0.007` xy, |z-(floor+half)|
≤0.005, z not high (no rim/stack: z ≤ expected+0.005, plus global `too_high`
guard expected+0.013), static ≤0.02, not touched/grasped. Complete = all active
correct + none too high + released + robot static, then dwell 1 s. Any order valid.
Logs `per_object_correct (N,3)`, `num_correct`, `complete`, `complete_stable`.

Rewards: per-object `correct*3 + (1-correct)*(0.5*reach+place+grasp)` + `num_correct*1`
(max 3N+N+2), normalized by max. State-based, any-order, no scripted-order bonus.

Props: 3 cubes 22–26 mm R/G/B, printed tray above. Validate: gripper fit in
35 mm compartments (outer 34 mm with jaws — tight, tested in sim via demos),
wall height vs occlusion, rim color visibility at 16px.

## Task C: cyclic rearrangement with buffer — `envs/rearrange.py`

IDs: `SO101Rearrange3-v1` (300 = 30 s): pockets P0,P1,P2 hold A,B,C (R,G,B)
initially, goal B,C,A (pocket→object [1,2,0]), P3 buffer (neutral);
`SO101Rearrange2-v1` (200 = 20 s, control): P0,P1 hold A,B, goal B,A ([1,0]),
P2 buffer. Same pocket geometry (subset for 2-obj; unused third cube parked at
(0.18,-0.14) out of the way).
Pockets: shallow printed, inner 0.036 m, wall 0.005×0.012 m, floor 0.005 m.
Centres x=0.30, y=(-0.09,-0.03,0.03,0.09) (first 3 for swap). Floors/rims colored
by DESTINATION (e.g., 3-obj P0 green (B), P1 blue (C), P2 red (A), buffer gray),
so initial mismatches signal goal; buffer neutral. Every pocket fits every cube
(same 22–26 mm cubes). Occupied pocket cannot be correctly seated into (would
stack/balance high → z guard fails). No orientation requirement. Example
solution A→buffer; B→P0; C→P1; A→P2 (not required; any valid final accepted,
no sequence hard-coded).

Success: each goal pocket holds assigned cube (same xy/z/static/released checks
as Pack, xy tol 0.012, z tol 0.006), buffer empty (no cube centre within
inner/2+0.005 and z near pocket), none too high (expected+0.014), released +
robot static, dwell 1 s. Logs `per_object_correct`, `num_correct`,
`pocket_correct`, `buffer_empty/occupied`, `complete`.

Rewards: `reach_nearest_misplaced*1 + num_correct*3 + grasp_misplaced*1`
(max 3N+2), normalized. Temporary buffer move keeps #correct at 0 (no penalty)
while reaching guides; not prohibitively unattractive. State-based.

Props: 3 cubes + 4 (or 3) pocket tray. Validate: pocket clearance for jaws,
destination color visibility, buffer empty detection with parked cube far away.

## Episode limits / frequencies
Sim 100 Hz, control 10 Hz. Horizons above (configurable via registration;
override with `gym.make(..., max_episode_steps=...)` where supported, else edit
class decorator). Document seconds + steps in all reports.

## Observation / action / reward summary for training scripts
Works with `train_squint.py` (and `train_squint_qc.py`, same `import envs`
registration + wrappers): `obs_mode="rgb+segmentation"`,
`FlattenRGBDObservationWrapper(rgb,state)`, `DownsampleObsWrapper(16/32/64)`,
`ColorJitterWrapper` (train augment only, never baked into demos),
`FlattenActionSpaceWrapper` if needed, `ManiSkillVectorEnv`. Action
`Box(-1,1,(6,))`, state `(12,)`, rgb `(H,W,3) uint8`. Reward default
`normalized_dense`; also supports `dense`/`sparse`.

## Remaining physical-validation items (not verified in sim)
Gripper clearance on largest cubes/compartments/pockets, tower stability on real
friction, printed tray/pocket tolerances and colors, marker print size/colors,
wrist-camera FOV/pose match (`tune_camera.py`), lighting/black-overlay match,
motor calibration, table height. See VALIDATION.md blockers. Do not present sim
feasibility as hardware performance.
