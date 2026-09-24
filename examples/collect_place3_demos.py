"""Collect scripted/motion-planning demonstrations for SO101Place3Cube-v1
(pick up 3 identical cubes one by one, drop each into a bin).

Same pipeline as collect_stack_demos.py, pointed at the long-horizon
pick-place-3 task (envs/place3.py) instead of the 2-cube stack. Note: the
underlying grasp (SO101ArmMotionPlanningSolver.grasp_axis_remap /
_tcp_to_movegroup_local_offset in examples/motionplanning/so101/motionplanner.py)
is shared with collect_stack_demos.py and is NOT yet reliably closing on the
cube (SO101 is a 5-DOF arm, so the derived 6-DOF grasp orientation is often
unreachable and plan_screw silently converges off-target) -- fix that there
first; it will carry over here automatically once solved.

Records raw joint-position trajectories (obs_mode="none", control_mode
"pd_joint_pos") to a ManiSkill .h5/.json trajectory pair. Convert to the
obs_mode / control_mode Squint actually trains with afterwards via
`python -m mani_skill.trajectory.replay_trajectory`, e.g.:

    python -m mani_skill.trajectory.replay_trajectory \\
        --traj-path demos/SO101Place3Cube-v1/motionplanning/trajectory.h5 \\
        --save-traj --obs-mode rgb+state \\
        --target-control-mode pd_joint_target_delta_pos \\
        --num-procs 4

Usage:
    python -m examples.collect_place3_demos -n 200 --only-count-success
    python -m examples.collect_place3_demos -n 1 --vis --save-video  # sanity check
"""
import argparse
import os.path as osp
import time

import envs  # noqa: F401  registers SO101Place3Cube-v1 etc.
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from examples.motionplanning.so101.solutions.place3_cube import solve
from mani_skill.utils.wrappers.record import RecordEpisode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="SO101Place3Cube-v1")
    parser.add_argument("-n", "--num-traj", type=int, default=10)
    parser.add_argument(
        "--only-count-success",
        action="store_true",
        help="Keep going past a seed until num-traj *successful* demos are collected.",
    )
    parser.add_argument("-b", "--sim-backend", type=str, default="cpu")
    parser.add_argument("--vis", action="store_true", help="open a GUI to watch the solver")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--traj-name", type=str, default=None)
    parser.add_argument("--record-dir", type=str, default="demos")
    parser.add_argument("--start-seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    env = gym.make(
        args.env_id,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
    )

    traj_name = args.traj_name or time.strftime("%Y%m%d_%H%M%S")
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, args.env_id, "motionplanning"),
        trajectory_name=traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="scripted SO101 place3_cube motion-planning solution",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    output_h5_path = env._h5_file.filename

    pbar = tqdm(range(args.num_traj))
    seed = args.start_seed
    successes = []
    failed_motion_plans = 0
    passed = 0
    while passed < args.num_traj:
        try:
            res = solve(env, seed=seed, debug=False, vis=args.vis)
        except Exception as e:
            print(f"[seed {seed}] motion planning error: {e}")
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = bool(res[-1]["success"].item())
        successes.append(success)

        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue

        env.flush_trajectory()
        if args.save_video:
            env.flush_video()
        pbar.update(1)
        pbar.set_postfix(
            success_rate=np.mean(successes),
            failed_motion_plan_rate=failed_motion_plans / (seed + 1),
        )
        seed += 1
        passed += 1

    env.close()
    print(f"Saved {passed} trajectories to {output_h5_path}")


if __name__ == "__main__":
    main(parse_args())
