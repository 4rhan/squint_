"""Calibrate the SO101 follower and/or leader arm with LeRobot, saving into simreal/calibration/.

LeRobot's procedure (it prompts for each step):
  1. If a calibration file exists: ENTER writes it to the motors, 'c' + ENTER runs a new calibration.
  2. Move the arm to the MIDDLE of its range of motion (the sim 'zero' pose, see simreal/README.md), ENTER.
  3. Move every joint except wrist_roll through its FULL range, then ENTER.

    conda activate squint
    python -m simreal.calibrate --arm follower --follower-port /dev/ttyACM0
    python -m simreal.calibrate --arm leader --leader-port /dev/ttyACM1
    python -m simreal.calibrate --arm both

Find the ports with `lerobot-find-port` (unplug/replug each arm's USB when it asks).
"""
import argparse

from simreal import hw


def calibrate(dev, name):
    print(f"\n=== {name}: {dev.calibration_fpath} ({'exists' if dev.calibration_fpath.is_file() else 'new'}) ===")
    dev.bus.connect()
    try:
        dev.calibrate()
        c = dev.bus.calibration
        print(f"{name} calibration in the motors:")
        for m in hw.JOINTS:
            print(f"  {m:14s} homing_offset {c[m].homing_offset:6d}  range [{c[m].range_min:4d}, {c[m].range_max:4d}]"
                  f"  span {c[m].range_max - c[m].range_min:4d} ticks")
    finally:
        dev.bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["follower", "leader", "both"], default="follower")
    p.add_argument("--follower-port", default=hw.FOLLOWER_PORT)
    p.add_argument("--leader-port", default=hw.LEADER_PORT)
    args = p.parse_args()
    if args.arm in ("follower", "both"):
        calibrate(hw.make_follower(args.follower_port), "follower")
    if args.arm in ("leader", "both"):
        calibrate(hw.make_leader(args.leader_port), "leader")
    print("\nNext: python -m simreal.check_calibration --pose zero --sim")
