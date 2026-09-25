# Demonstration collection (new tasks)

Script: `examples/collect_new_tasks_demos.py` — runs the IK motion planner
(`examples/motionplanning/so101/motionplanner.py` + `solutions/{tower3_cube,
tower2_cube,tray_pack,rearrange}.py`) directly in a training-like env so
recordings match training: `obs_mode="rgb+segmentation"`,
`control_mode=pd_joint_target_delta_pos` (default), 128px, `FlattenRGBD`,
no jitter. Clean 128px images stored; training downsamples via area to
16/32/64. Do NOT bake colour jitter into demos; apply at training time.

## HDF5 schema (compatible with `qc_data.load_h5_demos`)
Per group `traj_<i>`: `obs/rgb (T+1,H,W,C) uint8`, `obs/state (T+1,S) float32`
(training layout: noisy_qpos6+target_qpos6), `actions (T,A) float32` (normalized
delta in env action space), `rewards (T,) float32` (`normalized_dense`),
`terminated (T,) bool`, `truncated (T,) bool`. Root `meta` JSON + per-traj
`seed`/`horizon`: task/version, seed, source domain sim, dimensions,
controller/action units + limits, sim/control freq, camera config, obs layout,
reward mode. Episode boundaries via groups + term/trunc; chunk loaders must use
`ep_end` and never cross episodes (verified: each traj ends with success
termination, `T+1` obs check passes).

## Commands
```bash
# single task, 5 demos
python -m examples.collect_new_tasks_demos -e SO101Tower3Cube-v1 -n 5 -o demos/qc/SO101Tower3Cube-v1.h5
# all 7 variants, 5 each (long: rearrange delta is slow/flaky, see VALIDATION)
python -m examples.collect_new_tasks_demos --all -n 5 --outdir demos/qc --max-attempts 200
# quick controls (fast)
python -m examples.collect_new_tasks_demos -e SO101Tower2Cube-v1 -n 2 -o demos/qc/SO101Tower2Cube-v1.h5
python -m examples.collect_new_tasks_demos -e SO101TrayPack1-v1 -n 2 -o demos/qc/SO101TrayPack1-v1.h5
```
Current `demos/qc/` (checked in for Tower/Pack; Rearrange delta pending, see
VALIDATION): Tower3 (1×192 steps), Tower2 (2×~113), Pack1 (1×45), Pack2 (1×99),
Pack3 (1×156). All `T` within horizons, actions in [-1,1], rewards
`normalized_dense` mean ~0.4–0.5, `T+1` checks pass.

Set `HDF5_USE_FILE_LOCKING=FALSE` on file-locking filesystems if h5py reports
`BlockingIOError: Unable to lock file`.

## Future real-demo adapter (must record)
Commanded joint targets + controller state (`target_qpos`), control mode +
delta limits/action units, control/sim frequency, camera config (wrist pose,
FOV, resolution), observation layout, reward mode, task/version, seeds,
dimensions, source domain real. Measured joint-position differences must NOT be
assumed equivalent to target-delta actions. Store clean high-res images;
augment later.
