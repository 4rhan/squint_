import numpy as np
import sapien
from transforms3d.euler import euler2quat

from examples.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from examples.motionplanning.so101.solutions.stack_cube import _build_grasp_pose
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def solve(env, seed=None, debug=False, vis=False):
    """SO101LiftCube-v1: grasp the item, lift it, and return to rest qpos
    (success requires item lifted + grasped + arm near rest_qpos)."""
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

    obb = get_actor_obb(e.item)
    approaching = np.array([0, 0, -1])
    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * e.agent.tcp_pose.sp
    target_closing = tcp_pose.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=0.025
    )
    grasp_pose = _build_grasp_pose(approaching, grasp_info["closing"], e.item.pose.sp.p)
    grasp_pose = grasp_pose * planner.grasp_axis_remap()

    planner.gripper_state = planner.OPEN
    if planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.03]) * grasp_pose) == -1:
        return -1
    if planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.01]) * grasp_pose) == -1:
        return -1
    planner.close_gripper(gripper_state=planner.CLOSED)
    if planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.08]) * grasp_pose) == -1:
        return -1

    # return to rest with the item held
    cur = e.agent.robot.get_qpos().cpu().numpy()[0]
    goal = e.rest_qpos.cpu().numpy().copy()
    goal[-1] = cur[-1]
    result = planner.planner.plan_qpos_to_qpos(
        [goal], cur, time_step=e.control_timestep
    )
    if result["status"] != "Success":
        print(result["status"])
        return -1
    res = planner.follow_path(result, refine_steps=10)
    planner.close()
    return res
