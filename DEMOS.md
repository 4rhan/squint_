# Demonstration collection (new tasks)

Script: `examples/collect_new_tasks_demos.py` — runs the IK motion planner
(`examples/motionplanning/so101/motionplanner.py` + `solutions/{tower3_cube,
tower2_cube,tray_pack,rearrange}.py`) directly in the training controller
(`pd_joint_target_delta_pos`, normalized delta actions in the declared space,
10 Hz, declared ±0.1 arm / ±0.2 gripper limits) so recordings match training:
`obs_mode="rgb+segmentation"`, 128px, `FlattenRGBD`, no jitter. Clean 128px
stored; training area-downsamples to 16/32/64. No teleporting, no physics
bypass, no absolute-position commands labelled as delta (solver `_step`
converts via `(target-current)/limit`, clipped, with clip counting).

Explicit options (recorded verbatim in h5 `meta`):
`--domain-randomization` (default off; when on, per-episode pose/FOV/lighting
randomization per `base_random_env.py`), `--control-mode` (default None = env
default `pd_joint_target_delta_pos`), `--render-size` (default 128),
`--start-seed`, `--max-attempts`. Meta stores both requested and actual
(`domain_randomization`, `control_info` with mode/limits, sim/control freq,
camera string, obs layout, reward mode `normalized_dense`).

## HDF5 schema (episode boundaries + T+1 preserved)
Per group `traj_<i>`: `obs/rgb (T+1,H,W,C) uint8`, `obs/state (T+1,S) float32`
(training layout: noisy_qpos6+target_qpos6), `actions (T,A) float32`,
`rewards (T,) float32`, `terminated (T,) bool`, `truncated (T,) bool`.
Chunk loaders must treat `terminated|truncated|last` as episode ends and never
cross groups (verified: `T+1` asserts, success termination at end).

## Commands
```bash
# single task
python -m examples.collect_new_tasks_demos -e SO101Tower3Cube-v1 -n 5 -o demos/qc/SO101Tower3Cube-v1.h5
# declared 20-seed reliability check (reports every failure category)
python -m examples.collect_new_tasks_demos -e SO101Rearrange2-v1 -n 20 --max-attempts 20 --start-seed 100
# with domain randomization explicitly on
python -m examples.collect_new_tasks_demos -e SO101TrayPack3-v1 -n 20 --domain-randomization -o demos/qc/pack3_dr.h5
```

## Current `demos/qc/` (frozen geometry v2, new horizons)
| file | trajs | mean T | horizon | result |
|---|---|---|---|---|
| SO101Tower3Cube-v1.h5 | 3 (seeds 205-207) | 228 | 300 | 3/8, fails: 5× plan (pick/place IK/grasp, see LAST_FAIL) |
| SO101Tower2Cube-v1.h5 | 3 | 175 | 200 | 3/8 |
| SO101TrayPack1-v1.h5 | 3 | 70 | 150 | 3/3 |
| SO101TrayPack2-v1.h5 | 3 | 121 | 200 | 3/4 (1× pick_ik) |
| SO101TrayPack3-v1.h5 | 3 (seeds 200/203/205) | 185 | 300 | 3/6 |
| SO101Rearrange2-v1.h5 | 12 (seeds 100-119) | 165 | 300 | 12/20; fails: 7× pick_grasp (pick0×2, pick1×1, pick0b×4), 1× success_false |
| SO101Rearrange3-v1.h5 | 4 | 250 | 400 | 4/20; fails: 15× pick_grasp + 1× success_false(num_correct=2) |
All `T` within horizons (`too_long` 0). Actions in [-1,1], rewards
`normalized_dense`. Failure stage/reason via per-solution `LAST_FAIL`
(`pick_ik/pick_reach/pick_grasp/place_ik/place_reach/servo_ik`) + solver clip/step counts.

Root-caused and fixed during this phase: 37 mm-wide open jaws jammed 35-36 mm
pocket walls (3.7 N wall contact, gripper stuck open) → pocket picks now use
`open_extra=0.008` (30-32 mm); flush `speed_scale` now effective in delta;
delta settling/correction restored to 3/4 with verified tracking + clipping logs.

Set `HDF5_USE_FILE_LOCKING=FALSE` on file-locking filesystems.

## Future real-demo adapter (must record)
Commanded joint targets + controller state (`target_qpos`), control mode +
delta limits/action units, control/sim frequency, camera config (wrist pose,
FOV, resolution), observation layout, reward mode, task/version, seeds,
dimensions, source domain real. Measured joint-position differences must NOT be
assumed equivalent to target-delta actions. Store clean high-res images;
augment later.
