import numpy as np
import sapien
from transforms3d.euler import euler2quat

from examples.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def _build_grasp_pose(approaching, closing, center):
    # see stack_cube.py: envs/robot/so101.py's SO101 agent doesn't define
    # build_grasp_pose, so inline the standard construction.
    ortho = np.cross(closing, approaching)
    T = np.eye(4)
    T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
    T[:3, 3] = center
    return sapien.Pose(T)


def _pick_and_place(env, planner, cube, drop_pos):
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(cube)
    approaching = np.array([0, 0, -1])
    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * env.agent.tcp_pose.sp
    target_closing = tcp_pose.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing = grasp_info["closing"]
    grasp_pose = _build_grasp_pose(approaching, closing, cube.pose.sp.p)
    grasp_pose = grasp_pose * planner.grasp_axis_remap()

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    planner.gripper_state = planner.OPEN
    reach_pose = sapien.Pose([0, 0, 0.03]) * grasp_pose
    if planner.move_to_pose_with_screw(reach_pose) == -1:
        return -1

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    if planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.01]) * grasp_pose) == -1:
        return -1
    planner.close_gripper(gripper_state=planner.CLOSED)

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    if planner.move_to_pose_with_screw(lift_pose) == -1:
        return -1

    # -------------------------------------------------------------------------- #
    # Move above the bin and drop
    # -------------------------------------------------------------------------- #
    offset = drop_pos - cube.pose.sp.p
    place_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    if planner.move_to_pose_with_screw(place_pose) == -1:
        return -1

    return planner.open_gripper()


def solve(env, seed=None, debug=False, vis=False):
    """Pick up all three cubes one at a time and drop them into the bin
    (envs/place3.py's SO101Place3Cube-v1). Cubes are identical, so order
    doesn't matter; each is dropped from a slightly different, increasing
    height/offset so it doesn't collide mid-air with the one already in the
    bin.
    """
    env.reset(seed=seed)
    planner = SO101ArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    e = env.unwrapped

    bin_center = e.bin.pose.sp.p
    res = -1
    for k, cube in enumerate(e.cubes):
        cube_half = e.cube_half_sizes[0, k].item()
        drop_pos = bin_center.copy()
        drop_pos[:2] += np.array([(k - 1) * 0.02, 0.0])
        drop_pos[2] = e.bin_thickness + cube_half * (2 * k + 1) + 0.01
        res = _pick_and_place(e, planner, cube, drop_pos)
        if res == -1:
            planner.close()
            return -1

    planner.close()
    return res
