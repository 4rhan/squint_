"""Show a sim keyframe of the SO101 in the SAPIEN viewer (no robot needed), with its LeRobot joint values.

Use it to see the pose to hold the real arm in: 'zero' is the pose to calibrate in (every joint mid-range),
'rest' is where deploy.py parks the arm.

    conda activate squint
    python -m simreal.show_pose zero
"""
import argparse
import time

import numpy as np

from simreal import hw

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pose", choices=list(hw.KEYFRAMES))
    p.add_argument("--env-id", default="SO101LiftCube-v1")
    args = p.parse_args()
    q = hw.KEYFRAMES[args.pose]
    print(f"{args.pose}: " + ", ".join(f"{j} {np.rad2deg(v):.1f}°" for j, v in zip(hw.JOINTS, q)))
    print("(gripper here is the sim joint angle; close the viewer window or Ctrl+C to quit)")
    mirror = hw.SimMirror(args.env_id)
    try:
        mirror.show(q)
        while not mirror.base._viewer.closed:
            mirror.show(q)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    mirror.close()
