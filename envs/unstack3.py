"""3-cube unstack (SO-101): the reverse of SO101Stack3Cube-v1.

The episode starts with a 3-cube tower itemA (red, top) -> itemB (blue, middle) -> itemC (green, base).
The robot takes the cubes down one at a time: first itemA onto the table, then itemB onto the table, so all
three cubes end up resting on the table side by side. itemC never moves.

Task ID: SO101Unstack3Cube-v1

Cubes, sizes, observations and colours are shared with Stack3 (envs/stack3.py); only the reset (tower
instead of scattered cubes), the success check and the reward differ.
"""
from typing import Any, Sequence

import numpy as np
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .stack3 import Stack3


class Unstack3(Stack3):
    """
    **Task Description:**
    Unstack a 3-cube tower: pick itemA (red, top) off the tower and put it on the table, then pick itemB
    (blue, middle) off itemC (green, base) and put it on the table, leaving all three cubes side by side.

    **Randomizations:**
    - the tower's xy position (inside a reachable box) and z-axis rotation
    - cube sizes within the Stack3 ranges (with domain randomization)

    **Success Conditions:**
    - itemA and itemB rest on the table (not on another cube), apart from every other cube, and each was
      taken off the tower by a grasp (knocking the tower over does not count)
    - itemC is still near where the tower stood
    - itemA and itemB are static and not touched by the robot
    - robot is static
    """

    # A cube "rests on the table" when its centre is within this of its half size above the table.
    TABLE_Z_TOL = 0.005
    # A placed cube must be at least this far (xy, beyond touching) from every other cube.
    SEPARATION_MARGIN = 0.005
    # Reward: a carried cube counts as clear of the tower once it is this far (xy) from the other cubes.
    CLEAR_DIST = 0.06
    # itemC may drift this far (xy) from the tower position and still count as the untouched base.
    BASE_DRIFT_TOL = 0.02

    def __init__(self, *args, spawn_box_pos=(0.25, 0.0), spawn_box_half_size=0.05,
                 tower_yaw_jitter_deg=5.0, **kwargs):
        self.tower_yaw_jitter = np.deg2rad(tower_yaw_jitter_deg)
        # the tower is tall (~8 cm), so keep it well inside the arm's reach (r <= ~0.31 m)
        super().__init__(*args, spawn_box_pos=list(spawn_box_pos), spawn_box_half_size=spawn_box_half_size,
                         **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # skip Stack3's scattered-cube reset; same table / robot reset as Stack3
        super(Stack3, self)._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1]))
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)))

            spawn_center = self.agent.robot.pose.p[env_idx, :2] + torch.tensor(self.spawn_box_pos[:2])
            xy = spawn_center + (torch.rand(b, 2) * 2 - 1) * self.spawn_box_half_size

            hA, hB, hC = (self.itemA_half_sizes[env_idx], self.itemB_half_sizes[env_idx],
                          self.itemC_half_sizes[env_idx])
            zC = hC
            zB = 2 * hC + hB
            zA = 2 * hC + 2 * hB + hA

            qC = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            poses = []
            for z in (zA, zB, zC):
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = xy
                xyz[:, 2] = z
                # the upper cubes sit slightly twisted on the one below, as a hand-built tower would
                yaw = (torch.rand(b) * 2 - 1) * self.tower_yaw_jitter
                dq = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], -1)
                poses.append(Pose.create_from_pq(xyz, _quat_mul(qC, dq)))
            poses[2] = Pose.create_from_pq(poses[2].p, qC)  # the base keeps the sampled yaw exactly
            self.itemA.set_pose(poses[0])
            self.itemB.set_pose(poses[1])
            self.itemC.set_pose(poses[2])

            if not hasattr(self, "tower_xy"):
                self.tower_xy = torch.zeros((self.num_envs, 2), device=self.device)
                self.itemA_picked_clean = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self.itemB_picked_clean = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.tower_xy[env_idx] = xy
            self.itemA_picked_clean[env_idx] = False
            self.itemB_picked_clean[env_idx] = False

            # goal sites (hidden, debugging only): where A and B were in the tower
            self.goalA_site.set_pose(Pose.create_from_pq(poses[0].p))
            self.goalB_site.set_pose(Pose.create_from_pq(poses[1].p))

    # ------------------------------------------------------------------ evaluation
    def _on_table(self, pos: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
        return torch.abs(pos[:, 2] - half) <= self.TABLE_Z_TOL

    @staticmethod
    def _xy_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(p[:, :2] - q[:, :2], dim=1)

    def _separated(self, pos, half, others: Sequence[tuple]) -> torch.Tensor:
        """xy-apart from every (pos, half) in `others`: centres further than the two half diagonals would
        overlap, so the cubes are not touching whatever their yaw."""
        ok = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
        for p, h in others:
            ok &= self._xy_dist(pos, p) >= np.sqrt(2) * (half + h) + self.SEPARATION_MARGIN
        return ok

    def evaluate(self):
        posA, posB, posC = self.itemA.pose.p, self.itemB.pose.p, self.itemC.pose.p
        hA, hB, hC = self.itemA_half_sizes, self.itemB_half_sizes, self.itemC_half_sizes

        _, _, is_itemA_on_itemB = self._is_stacked(posA, hA, posB, hB)
        _, _, is_itemB_on_itemC = self._is_stacked(posB, hB, posC, hC)

        is_itemA_grasped = self.agent.is_grasping(self.itemA)
        is_itemB_grasped = self.agent.is_grasping(self.itemB)

        # Latched per episode: a cube counts as unstacked only if it was grasped while still properly stacked
        # (A on B on C; B on C). Without this, knocking the tower over puts A and B on the table in ~10 steps
        # and scores full success (seen in a noisy recovery demo).
        self.itemA_picked_clean |= is_itemA_grasped & is_itemA_on_itemB & is_itemB_on_itemC
        self.itemB_picked_clean |= is_itemB_grasped & is_itemB_on_itemC
        # a cube that left its place in the tower without being picked: the tower was knocked
        tower_knocked = ((~self.itemA_picked_clean) & (~is_itemA_on_itemB)) | (
            (~self.itemB_picked_clean) & (~is_itemB_on_itemC))

        is_itemA_on_table = (self._on_table(posA, hA) & self._separated(posA, hA, [(posB, hB), (posC, hC)])
                             & self.itemA_picked_clean)
        is_itemB_on_table = (self._on_table(posB, hB) & self._separated(posB, hB, [(posA, hA), (posC, hC)])
                             & self.itemB_picked_clean)
        base_drift = self._xy_dist(posC, self.tower_xy)
        is_itemC_in_place = self._on_table(posC, hC) & (base_drift <= self.BASE_DRIFT_TOL)

        itemA_vel = torch.linalg.norm(self.itemA.linear_velocity, axis=-1)
        itemB_vel = torch.linalg.norm(self.itemB.linear_velocity, axis=-1)
        is_itemA_static = itemA_vel <= 2e-2
        is_itemB_static = itemB_vel <= 2e-2

        is_robot_static = self.agent.is_static()

        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_itemA = self.agent.is_touching(self.itemA)
        robot_touching_itemB = self.agent.is_touching(self.itemB)

        success = (
            is_itemA_on_table
            & is_itemB_on_table
            & is_itemC_in_place
            & is_itemA_static
            & is_itemB_static
            & (~robot_touching_itemA)
            & (~robot_touching_itemB)
            & is_robot_static
        )

        return {
            "itemA_vel": itemA_vel,
            "itemB_vel": itemB_vel,
            "base_drift": base_drift,

            "success": success,
            "is_itemA_on_table": is_itemA_on_table,
            "is_itemB_on_table": is_itemB_on_table,
            "is_itemC_in_place": is_itemC_in_place,
            "itemA_picked_clean": self.itemA_picked_clean.clone(),
            "itemB_picked_clean": self.itemB_picked_clean.clone(),
            "tower_knocked": tower_knocked,
            "is_itemA_on_itemB": is_itemA_on_itemB,
            "is_itemB_on_itemC": is_itemB_on_itemC,
            "is_itemA_static": is_itemA_static,
            "is_itemB_static": is_itemB_static,
            "is_itemA_grasped": is_itemA_grasped,
            "is_itemB_grasped": is_itemB_grasped,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_itemA": robot_touching_itemA,
            "robot_touching_itemB": robot_touching_itemB,
        }

    # ------------------------------------------------------------------ reward
    def _stage_unstack_reward(self, tcp_pos, mover_pos, mover_half, others, mover_grasped, mover_on_table,
                              robot_touching_mover, mover_vel, robot_qvel, lift_z):
        """Dense reward (range roughly [0, 8]) for taking `mover` off the tower and putting it on the table
        clear of `others` (list of (pos, half)). Same staging and scale as Stack3's _stage_place_reward:
        reach 0-2, grasped 3-5, on table while held 4-7, released 7-8."""
        tcp_to_mover_dist = torch.linalg.norm(tcp_pos - mover_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * tcp_to_mover_dist))

        # xy: move away from the other cubes until CLEAR_DIST
        clear = torch.stack([self._xy_dist(mover_pos, p) for p, _ in others], 1).min(1).values
        away = torch.clamp(clear / self.CLEAR_DIST, max=1.0)
        # z: stay above the tower top (lift_z) until clear, then come down to the table
        is_clear = clear >= self.CLEAR_DIST
        z_target = torch.where(is_clear, mover_half, lift_z)
        place_reward_z = 1 - torch.tanh(10.0 * torch.abs(mover_pos[:, 2] - z_target))
        place_reward = away + place_reward_z

        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        ungrasp_reward = (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)

        reward[mover_grasped] = (3 + place_reward)[mover_grasped]

        on_table_and_held = mover_on_table & robot_touching_mover
        reward[on_table_and_held] = (4 + place_reward + ungrasp_reward)[on_table_and_held]

        static_mover_reward = 1 - torch.tanh(mover_vel * 10)
        static_robot_reward = 1 - torch.tanh(torch.linalg.norm(robot_qvel, axis=1) * 10)
        on_table_and_released = mover_on_table & (~robot_touching_mover)
        reward[on_table_and_released] = (7 + (static_mover_reward + static_robot_reward) / 2.0)[on_table_and_released]
        return reward

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_pos = self.agent.tcp_pose.p
        robot_qvel = self.agent.robot.get_qvel()[:, :-1]
        posA, posB, posC = self.itemA.pose.p, self.itemB.pose.p, self.itemC.pose.p
        hA, hB, hC = self.itemA_half_sizes, self.itemB_half_sizes, self.itemC_half_sizes
        # carry height: the bottom of the carried cube 3 cm above the full tower's top
        tower_top = 2 * (hA + hB + hC)

        # Stage 1: itemA (top) off the tower onto the table
        stage1_reward = self._stage_unstack_reward(
            tcp_pos, posA, hA, [(posB, hB), (posC, hC)], info["is_itemA_grasped"], info["is_itemA_on_table"],
            info["robot_touching_itemA"], info["itemA_vel"], robot_qvel, lift_z=tower_top + hA + 0.03,
        )
        # Locked in once itemA is on the table and released. itemB must still be on the base or in hand
        # (or already placed), so knocking the tower over does not count as progress.
        stage1_done = (
            info["is_itemA_on_table"] & (~info["robot_touching_itemA"])
            & (info["is_itemB_on_itemC"] | info["is_itemB_grasped"] | info["is_itemB_on_table"])
        )

        # Stage 2: itemB (middle) off itemC onto the table, clear of both other cubes
        stage2_reward = self._stage_unstack_reward(
            tcp_pos, posB, hB, [(posA, hA), (posC, hC)], info["is_itemB_grasped"], info["is_itemB_on_table"],
            info["robot_touching_itemB"], info["itemB_vel"], robot_qvel, lift_z=2 * hC + hB + 0.03,
        )

        # Same 0-18 scale as Stack3: [0, 8] for stage 1, +9 once locked in, [0, 8] for stage 2, 18 on success.
        reward = torch.where(stage1_done, 9 + stage2_reward, stage1_reward)
        reward[info["success"]] = 18

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        # keep the base where it is (pushing the tower around is not unstacking)
        reward -= 2 * torch.tanh(20 * info["base_drift"])
        # a knocked tower can't be completed any more (its cubes never count as placed)
        reward -= 3 * info["tower_knocked"].float()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 18


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (..., 4) wxyz quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], -1)


@register_env("SO101Unstack3Cube-v1", max_episode_steps=150)
class Unstack3Cube(Unstack3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
