# simreal: real SO101 setup for sim-real co-training

First steps toward co-training on sim + real data: calibrate the arms, check the calibration against the sim,
then teleoperate with the leader arm. Later: record real demos in the `qc_data.py` HDF5 schema and mix them
into `train_squint_qc.py` (BC-only; see `QC_EXPERIMENT_LOG.md`).

Everything runs in the **`squint` conda env**, with the same LeRobot (pip 0.4.3) that `deploy.py` uses, so the
calibration and conversions checked here are exactly what deployment sees. Run from the repo root with
`python -m simreal.<script>`.

```bash
conda activate squint
cd ~/squint_
```

| File | What |
|---|---|
| `hw.py` | ports, arm ids, calibration paths, real ↔ sim joint conversion (same as `deploy_utils/manipulator.py`), sim mirror |
| `calibrate.py` | LeRobot calibration of the follower / leader, saved in `simreal/calibration/` |
| `check_calibration.py` | torque off, live table of real vs sim joint values, keyframe error, leader vs follower, optional sim mirror |
| `teleop.py` | leader → follower teleop with tracking error, loop rate, and action-limit clip rate |
| `show_pose.py` | show a sim keyframe (e.g. the calibration pose) in the viewer |
| `calibration/` | `follower/so101_follower_arm.json` (seeded from `feat/deployment`), `leader/so101_leader_arm.json` (seeded from `~/.cache/.../so_leader/Leader.json`) |

## 0. Ports

```bash
lerobot-find-port            # unplug/replug each arm when asked
export FOLLOWER_PORT=/dev/ttyACM0 LEADER_PORT=/dev/ttyACM1   # or pass --follower-port / --leader-port
sudo chmod 666 /dev/ttyACM*  # if you get a permission error
```

## 1. Calibrate

```bash
python -m simreal.show_pose zero            # the pose to hold the arm in at the "middle of range" prompt
python -m simreal.calibrate --arm follower
python -m simreal.calibrate --arm leader
```

If a file already exists, ENTER writes it to the motors and `c` + ENTER recalibrates. When recalibrating:
1. Put the arm in the sim `zero` pose, then ENTER. This sets the homing offset.
2. Move every joint except wrist_roll through its whole range, then ENTER.

LeRobot's degree readings are centred on the **middle of the recorded range**, not on the pose from step 1.
So sweep each joint fully and evenly, or the sim zero shifts.

## 2. Check the calibration against the sim

```bash
python -m simreal.check_calibration --pose zero --sim   # hold the arm in the zero pose, errors should be < 5°
python -m simreal.check_calibration --pose rest --sim   # the pose deploy.py resets to
python -m simreal.check_calibration --leader            # hold both arms in the same pose, L-F should be ~0
```

- `range%` flagged with `!`: the joint is at the edge of its recorded range. Readings clip there, so recalibrate that arm.
- A joint moving the wrong way in the mirror, or a constant offset: fix it in the calibration, not in the training code.
- Gripper: check that fully closed and fully open land near sim −10° and 120°. That mapping is `SIM_GRIPPER_DEG` / `SERVO_GRIPPER_DEG` in `hw.py`.

## 3. Teleop

```bash
python -m simreal.teleop                          # follower first moves slowly to the leader's pose
python -m simreal.teleop --sim --camera /dev/video2
python -m simreal.teleop --max-relative-target 10 --rest-on-exit
```

Once a second it prints:
- `Hz`: the loop rate.
- `max |leader-follower|`: tracking error. A few degrees is normal, and the gripper is larger when it holds something.
- `>action limit %`: the share of 10 Hz steps where the arm moved more than the training action space allows (0.1 rad arm, 0.2 gripper). Demos recorded with high values can't be reproduced by the policy, so teleop slower.

On Ctrl+C the script asks you to hold the arm before it releases torque. With `--rest-on-exit` it parks the arm in the sim `rest` pose instead.
