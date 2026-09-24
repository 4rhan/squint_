import numpy as np
import sapien
from transforms3d.euler import euler2quat

from examples.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def _build_grasp_pose(approaching, closing, center):
    # envs/robot/so101.py's SO101 agent (unlike mani_skill's built-in agents)
    # doesn't define build_grasp_pose, so inline the standard construction.
    ortho = np.cross(closing, approaching)
    T = np.eye(4)
    T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
    T[:3, 3] = center
    return sapien.Pose(T)


def solve(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = SO101ArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.itemA)

    approaching = np.array([0, 0, -1])
    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * env.agent.tcp_pose.sp
    target_closing = tcp_pose.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = _build_grasp_pose(approaching, closing, env.itemA.pose.sp.p)
    grasp_pose = grasp_pose * planner.grasp_axis_remap()

    
    planner.gripper_state = planner.OPEN
    reach_pose = sapien.Pose([0, 0, 0.03]) * grasp_pose
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        planner.close()
        return -1
    
    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.01]) * grasp_pose)
    if res == -1:
        planner.close()
        return -1
    planner.close_gripper(gripper_state=planner.CLOSED)

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Stack itemA on top of itemB
    # -------------------------------------------------------------------------- #
    goal_pos = env.goal_site.pose.sp.p
    offset = goal_pos - env.itemA.pose.sp.p
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    res = planner.move_to_pose_with_screw(align_pose)
    if res == -1:
        planner.close()
        return -1

    res = planner.open_gripper()
    planner.close()
    return res
