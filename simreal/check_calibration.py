"""Check the calibration against the sim: torque off, move the arm by hand, watch real vs sim joint values.

For each joint it prints the LeRobot reading, the sim qpos it maps to (same conversion as deployment), where it
sits inside the calibrated range, and, with --pose, the error to a sim keyframe (envs/robot/so101.py).
With --sim a SAPIEN viewer mirrors the real arm, so a sign flip or offset is visible at a glance.
With --leader the leader arm is read too and the leader-follower difference is shown (both should agree
when the two arms are held in the same pose, otherwise teleop will be offset).

    conda activate squint
    python -m simreal.check_calibration --pose zero --sim          # hold the arm in the sim zero pose
    python -m simreal.check_calibration --pose rest                # the pose deploy.py resets to
    python -m simreal.check_calibration --leader                   # compare leader vs follower

Ctrl+C to stop. Torque stays off the whole time (support the arm).
"""
import argparse
import time

import numpy as np

from simreal import hw


def main(args):
    follower = hw.make_follower(args.follower_port)
    hw.connect_follower(follower)  # prompts only if the motors and the file disagree
    follower.bus.disable_torque()
    leader = None
    if args.leader:
        leader = hw.make_leader(args.leader_port)
        leader.connect(calibrate=True)
    mirror = hw.SimMirror(args.env_id) if args.sim else None
    target = np.array(hw.KEYFRAMES[args.pose]) if args.pose else None
    cal = follower.bus.calibration

    print("torque is OFF; move the arm by hand. Ctrl+C to stop.")
    worst = {}
    try:
        while True:
            pos = hw.read_joints(follower)
            raw = follower.bus.sync_read("Present_Position", normalize=False)
            q = hw.real_to_sim(pos, follower.bus)
            lines = [f"{'joint':14s} {'real':>9s} {'sim rad':>8s} {'sim deg':>8s} {'range%':>7s}"
                     + (f" {'target':>8s} {'err deg':>8s}" if target is not None else "")
                     + (f" {'leader':>9s} {'L-F':>7s}" if leader else "")]
            lpos = hw.read_joints(leader) if leader else None
            for i, j in enumerate(hw.JOINTS):
                c = cal[j]
                frac = 100 * (raw[j] - c.range_min) / (c.range_max - c.range_min)
                flag = " !" if frac < 2 or frac > 98 else "  "  # at the edge of the recorded range
                unit = "%" if j == "gripper" else "°"
                s = f"{j:14s} {pos[f'{j}.pos']:8.1f}{unit} {q[i]:8.3f} {np.rad2deg(q[i]):8.1f} {frac:6.0f}%{flag}"
                if target is not None:
                    err = np.rad2deg(q[i] - target[i])
                    worst[j] = err
                    s += f" {np.rad2deg(target[i]):8.1f} {err:+8.1f}{' <-- off' if abs(err) > args.tol else ''}"
                if leader:
                    s += f" {lpos[f'{j}.pos']:8.1f}{unit} {lpos[f'{j}.pos'] - pos[f'{j}.pos']:+7.1f}"
                lines.append(s)
            print("\033[H\033[J" + "\n".join(lines), flush=True)
            if mirror:
                mirror.show(q)
            time.sleep(1 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        if worst:
            print(f"\nlast error to '{args.pose}' (deg): " + ", ".join(f"{j} {e:+.1f}" for j, e in worst.items()))
        follower.disconnect()
        if leader:
            leader.disconnect()
        if mirror:
            mirror.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--follower-port", default=hw.FOLLOWER_PORT)
    p.add_argument("--leader-port", default=hw.LEADER_PORT)
    p.add_argument("--leader", action="store_true", help="also read the leader arm and show leader - follower")
    p.add_argument("--pose", choices=list(hw.KEYFRAMES), default=None, help="sim keyframe to compare against")
    p.add_argument("--tol", type=float, default=5.0, help="degrees; errors above this are flagged")
    p.add_argument("--sim", action="store_true", help="mirror the real arm in the SAPIEN viewer")
    p.add_argument("--env-id", default="SO101LiftCube-v1", help="scene used for the --sim mirror")
    p.add_argument("--hz", type=float, default=5.0)
    main(p.parse_args())
