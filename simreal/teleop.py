"""Leader -> follower teleoperation with LeRobot, plus the checks that matter for sim-real data.

Every second it prints the loop rate, the leader-follower tracking error per joint, and how often the motion
would have been clipped by the training action space (pd_joint_target_delta_pos allows 0.1 rad per 10 Hz
step on the arm, 0.2 on the gripper). High clip rates mean teleop demos recorded at that speed would not be
reproducible by the policy, so slow down.

    conda activate squint
    python -m simreal.teleop                                  # ports from FOLLOWER_PORT / LEADER_PORT
    python -m simreal.teleop --sim                            # mirror the follower in the SAPIEN viewer
    python -m simreal.teleop --camera /dev/video2             # show the wrist camera + the 16x16 policy view
    python -m simreal.teleop --max-relative-target 10         # LeRobot per-command safety clip (deg)

At start the follower moves slowly to the leader's pose. Ctrl+C stops; with --rest-on-exit the follower
first goes back to the sim 'rest' keyframe (the pose deploy.py resets to), otherwise torque is released
after you press ENTER, so hold the arm.
"""
import argparse
import time
from collections import deque

import numpy as np

from simreal import hw


def policy_view(frame, size=16, show=256):
    """Square centre crop, area-resize to the policy's input size, nearest-upsample for display."""
    import cv2
    h, w = frame.shape[:2]
    c = min(h, w)
    sq = frame[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
    small = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (show, show), interpolation=cv2.INTER_NEAREST), cv2.resize(sq, (show, show))


def main(args):
    follower = hw.make_follower(args.follower_port, camera=args.camera, max_relative_target=args.max_relative_target)
    leader = hw.make_leader(args.leader_port)
    hw.connect_follower(follower)
    leader.connect(calibrate=True)
    mirror = hw.SimMirror(args.env_id) if args.sim else None
    if args.camera is not None:
        import cv2

    print("moving the follower to the leader's pose...")
    hw.move_smoothly(follower, leader.get_action())
    print("teleop running. Ctrl+C to stop.")

    every = max(1, round(args.fps / hw.SIM_CONTROL_FREQ))  # loop ticks per sim control step
    hist = deque(maxlen=every + 1)  # follower sim qpos, to measure motion per 10 Hz step
    stats = dict(n=0, t=time.perf_counter(), err=np.zeros(6), clip=np.zeros(6), steps=0)
    tick = 0
    try:
        while True:
            t0 = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)
            obs = follower.get_observation()

            q = hw.real_to_sim(obs, follower.bus)
            q_lead = hw.real_to_sim(action, follower.bus)
            stats["err"] = np.maximum(stats["err"], np.abs(q_lead - q))
            hist.append(q)
            if tick % every == 0 and len(hist) == hist.maxlen:
                stats["clip"] += np.abs(hist[-1] - hist[0]) > hw.MAX_DELTA
                stats["steps"] += 1
            if mirror and tick % every == 0:
                mirror.show(q)
            if args.camera is not None and "base_camera" in obs and tick % every == 0:
                small, full = policy_view(obs["base_camera"], args.image_size)
                cv2.imshow("wrist camera | policy input", cv2.cvtColor(np.hstack([full, small]), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            stats["n"] += 1
            tick += 1
            now = time.perf_counter()
            if now - stats["t"] >= 1.0:
                hz = stats["n"] / (now - stats["t"])
                clip = 100 * stats["clip"] / max(stats["steps"], 1)
                print(f"{hz:5.1f} Hz | max |leader-follower| deg: "
                      + " ".join(f"{j[:5]} {np.rad2deg(e):4.1f}" for j, e in zip(hw.JOINTS, stats["err"]))
                      + f" | >action limit %: " + " ".join(f"{c:3.0f}" for c in clip), flush=True)
                stats.update(n=0, t=now, err=np.zeros(6), clip=np.zeros(6), steps=0)
            time.sleep(max(0.0, 1 / args.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        try:
            if args.rest_on_exit:
                print("moving the follower to the sim 'rest' pose...")
                hw.move_smoothly(follower, hw.sim_to_real(hw.KEYFRAMES["rest"], follower.bus), max_deg_per_step=1.0)
            else:
                input("hold the follower arm, then press ENTER to release its torque...")
        finally:
            follower.disconnect()
            leader.disconnect()
            if mirror:
                mirror.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--follower-port", default=hw.FOLLOWER_PORT)
    p.add_argument("--leader-port", default=hw.LEADER_PORT)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-relative-target", type=float, default=None,
                   help="LeRobot safety clip: max degrees per command away from the present position")
    p.add_argument("--camera", default=None, help="wrist camera OpenCV index or /dev/videoN (optional)")
    p.add_argument("--image-size", type=int, default=16, help="policy image size for the preview")
    p.add_argument("--sim", action="store_true", help="mirror the follower in the SAPIEN viewer")
    p.add_argument("--env-id", default="SO101LiftCube-v1")
    p.add_argument("--rest-on-exit", action="store_true", help="drive the follower to the sim rest pose on exit")
    args = p.parse_args()
    if args.camera is not None and args.camera.isdigit():
        args.camera = int(args.camera)
    main(args)
