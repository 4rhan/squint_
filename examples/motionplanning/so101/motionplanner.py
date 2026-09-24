from unittest import mock

import mplib
import numpy as np
import sapien
from transforms3d import euler

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import (
    TwoFingerGripperMotionPlanningSolver,
)


def _articulated_model_no_convex(*args, **kwargs):
    # mplib==0.1.1 (pinned by mani_skill) hardcodes convex=True inside
    # mplib.Planner, and its convex-hull decomposition segfaults natively on
    # this robot's gripper meshes (envs/robot/so101.urdf). convex=False loads
    # the same urdf/srdf without crashing (verified directly against
    # mplib.pymp.ArticulatedModel), so force it off for this robot only.
    kwargs["convex"] = False
    return mplib.pymp.ArticulatedModel(*args, **kwargs)


class SO101ArmMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    """Motion planning solver for the SO101 arm used in this repo's custom envs
    (envs/stack.py, envs/place.py, envs/stack3.py, envs/place3.py).

    Mirrors ManiSkill's built-in SO100ArmMotionPlanningSolver, but SO101's urdf
    (envs/robot/so101.urdf) names the fixed-jaw link "gripper_link" instead of
    "Fixed_Jaw_tip", and envs/robot/so101.py defines tcp_pose as the midpoint
    between finger1_tip/finger2_tip (not a real link), so grasp poses computed
    in tcp frame need to be offset into the gripper_link frame before planning.
    """

    # SO101's gripper joint qlimits are ~[-0.17, 2.09] rad (see
    # envs/robot/so101.py's "rest"/"start" keyframes and the ungrasp_reward
    # in envs/stack.py), not the generic [-1, 1] TwoFingerGripperMotionPlanningSolver
    # assumes, so OPEN/CLOSED must be overridden per-instance from the actual
    # qlimits rather than left as class constants.
    OPEN = 1.0
    CLOSED = -0.174
    MOVE_GROUP = "gripper_link"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            visualize_target_grasp_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
        )
        gripper_min, gripper_max = (
            self.base_env.agent.robot.get_qlimits()[0, -1, :].cpu().numpy()
        )
        self.OPEN = float(gripper_max)
        self.CLOSED = float(gripper_min)
        self.gripper_state = self.OPEN

        # SO101's gripper is NOT a symmetric parallel-jaw gripper: finger1
        # (gripper_link, "Fixed Jaw") never moves; finger2 ("Moving Jaw")
        # sweeps a large 3D arc as the gripper joint opens/closes (its tip
        # moves ~0.11m and shifts in z by ~0.1m across the qlimit range).
        # So the "tcp" = midpoint(finger1_tip, finger2_tip) used by
        # envs/robot/so101.py is only a meaningful grasp-center reference at
        # the gripper angle actually used to *pinch* the object, not at
        # whatever angle the joint happens to be in while approaching.
        # Freeze the tcp<->gripper_link offset at a fixed near-closed
        # reference angle (where the pinch geometry is what matters) instead
        # of recomputing it from the live, approach-time qpos.
        # Chosen so the fingertip gap at this angle (~0.03m, see
        # dist(finger1_tip, finger2_tip) at this qpos) is only slightly wider
        # than the cube sizes this repo spawns (item half-size ranges
        # 0.011-0.017m -> full size up to ~0.028-0.034m, see
        # StackRandomizationConfig in envs/stack.py), i.e. close to the angle
        # actually used at the moment of a real pinch rather than a wide-open
        # approach angle (which swings finger2's tip far out of plane, see
        # class docstring above).
        self._grasp_reference_gripper_qpos = 0.22
        self._cached_tcp_to_movegroup_local_offset = (
            self._compute_tcp_to_movegroup_local_offset_at(
                self._grasp_reference_gripper_qpos
            )
        )

    def _compute_tcp_to_movegroup_local_offset_at(self, gripper_qpos: float) -> sapien.Pose:
        robot = self.base_env.agent.robot
        original_qpos = robot.get_qpos().clone()
        qpos = original_qpos.clone()
        qpos[..., -1] = gripper_qpos
        robot.set_qpos(qpos)
        offset = self.base_env.agent.tcp_pose.sp.inv() * (
            robot.links_map["gripper_link"].pose.sp
        )
        robot.set_qpos(original_qpos)
        return offset

    @property
    def _tcp_to_movegroup_local_offset(self) -> sapien.Pose:
        return self._cached_tcp_to_movegroup_local_offset

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target * self._tcp_to_movegroup_local_offset

    def grasp_axis_remap(self) -> sapien.Pose:
        """Rotation that maps the generic grasp-pose convention used by
        `_build_grasp_pose`/`compute_grasp_info_by_obb` (columns
        [ortho, closing, approach], approach = world -z) into this robot's
        actual gripper_link/tcp frame convention, derived from the gripper
        geometry itself (finger tip directions) rather than a hand-tuned
        euler triple, since SO101's gripper frame axes don't match the
        panda/SO100 conventions those examples assume.
        """
        agent = self.base_env.agent
        robot = agent.robot
        original_qpos = robot.get_qpos().clone()
        qpos = original_qpos.clone()
        qpos[..., -1] = self._grasp_reference_gripper_qpos
        robot.set_qpos(qpos)

        gripper_link_pose = robot.links_map["gripper_link"].pose.sp
        f1 = agent.finger1_tip.pose.sp
        f2 = agent.finger2_tip.pose.sp
        R = gripper_link_pose.to_transformation_matrix()[:3, :3]

        robot.set_qpos(original_qpos)

        def to_local(world_vec):
            return R.T @ (world_vec / np.linalg.norm(world_vec))

        closing_local = to_local(f2.p - f1.p)
        approach_local = to_local((f1.p + f2.p) / 2 - gripper_link_pose.p)
        ortho_local = np.cross(closing_local, approach_local)
        ortho_local /= np.linalg.norm(ortho_local)
        # re-orthogonalize approach against [ortho, closing] to correct for
        # the fingers not being exactly perpendicular to the approach axis
        approach_local = np.cross(ortho_local, closing_local)
        M = np.stack([ortho_local, closing_local, approach_local], axis=1)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = M.T
        return sapien.Pose(T)

    def setup_planner(self):
        with mock.patch("mplib.planner.ArticulatedModel", _articulated_model_no_convex):
            return super().setup_planner()
