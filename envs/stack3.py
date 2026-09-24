from dataclasses import dataclass
from typing import Any, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

from .robot.so100 import SO100
from .robot.so101 import SO101


@dataclass
class Stack3RandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for the 3-cube Stack task."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    # ItemA (red cube, goes on top)
    itemA_half_size_range: Sequence[float] = (0.022 / 2, 0.028 / 2)
    # ItemB (blue cube, goes in the middle)
    itemB_half_size_range: Sequence[float] = (0.022 / 2, 0.028 / 2)
    # ItemC (green cube, base of the tower)
    itemC_half_size_range: Sequence[float] = (0.025 / 2, 0.032 / 2)

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False  # Keep colors distinct (red/blue/green)


class Stack3(DefaultCameraEnv):
    """
    **Task Description:**
    Build a 3-cube tower: pick up itemB (blue) and stack it on top of itemC
    (green, the base), then pick up itemA (red) and stack it on top of itemB,
    forming a 3-cube tower itemA -> itemB -> itemC.

    **Randomizations:**
    - all three items have their xy positions randomized (non-overlapping)
    - all three items have their z-axis rotation randomized
    - item sizes are randomized within configured ranges

    **Success Conditions:**
    - itemB is on top of itemC, itemA is on top of itemB
    - itemA and itemB are static
    - itemA and itemB are not being grasped
    - robot is static
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101]

    def __init__(
        self,
        *args,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            Stack3RandomizationConfig, dict
        ] = Stack3RandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.2 / 2,
        stage2_start_prob=0.0,
        **kwargs,
    ):
        self.stage2_start_prob = stage2_start_prob

        # Robot-specific configuration
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        # Handle domain randomization config
        self.domain_randomization_config = Stack3RandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=Stack3RandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, Stack3RandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = spawn_box_pos
        self.spawn_box_half_size = spawn_box_half_size

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _build_cube_item(self, name: str, half_sizes: np.ndarray, color: np.ndarray,
                          frictions: np.ndarray, densities: np.ndarray, spawn_x: float):
        """Builds one merged cube Actor across all parallel envs."""
        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )
            builder.add_box_collision(
                half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
            )
            builder.add_box_visual(
                half_size=[half_sizes[i]] * 3,
                material=sapien.render.RenderMaterial(base_color=color),
            )
            # Offset spawn position per-item so they don't collide with each other at creation
            builder.initial_pose = sapien.Pose(p=[spawn_x, 0, half_sizes[i]])
            builder.set_scene_idxs([i])
            item = builder.build(name=f"{name}-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        merged = Actor.merge(items, name=name)
        self.add_to_state_dict_registry(merged)
        return merged

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2
        if self.domain_randomization:
            frictions = self._batched_episode_rng.uniform(
                low=cfg.item_friction_range[0], high=cfg.item_friction_range[1],
            )
            densities = self._batched_episode_rng.uniform(
                low=cfg.item_density_range[0], high=cfg.item_density_range[1],
            )
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        # ========== ItemA (red, top of the tower) ==========
        itemA_half_sizes = (
            np.ones(self.num_envs)
            * (cfg.itemA_half_size_range[1] + cfg.itemA_half_size_range[0]) / 2
        )
        if self.domain_randomization:
            itemA_half_sizes = self._batched_episode_rng.uniform(
                low=cfg.itemA_half_size_range[0], high=cfg.itemA_half_size_range[1],
            )
        self.itemA_half_sizes = common.to_tensor(itemA_half_sizes, device=self.device)
        self.itemA_dimensions = torch.stack([self.itemA_half_sizes] * 3, dim=-1)
        colorA = np.array([1, 0, 0, 1])  # Red
        self.itemA = self._build_cube_item(
            "itemA", itemA_half_sizes, colorA, frictions, densities, spawn_x=0.2
        )

        # ========== ItemB (blue, middle of the tower) ==========
        itemB_half_sizes = (
            np.ones(self.num_envs)
            * (cfg.itemB_half_size_range[1] + cfg.itemB_half_size_range[0]) / 2
        )
        if self.domain_randomization:
            itemB_half_sizes = self._batched_episode_rng.uniform(
                low=cfg.itemB_half_size_range[0], high=cfg.itemB_half_size_range[1],
            )
        self.itemB_half_sizes = common.to_tensor(itemB_half_sizes, device=self.device)
        self.itemB_dimensions = torch.stack([self.itemB_half_sizes] * 3, dim=-1)
        colorB = np.array([0, 0, 1, 1])  # Blue
        self.itemB = self._build_cube_item(
            "itemB", itemB_half_sizes, colorB, frictions, densities, spawn_x=0.0
        )

        # ========== ItemC (green, base of the tower) ==========
        itemC_half_sizes = (
            np.ones(self.num_envs)
            * (cfg.itemC_half_size_range[1] + cfg.itemC_half_size_range[0]) / 2
        )
        if self.domain_randomization:
            itemC_half_sizes = self._batched_episode_rng.uniform(
                low=cfg.itemC_half_size_range[0], high=cfg.itemC_half_size_range[1],
            )
        self.itemC_half_sizes = common.to_tensor(itemC_half_sizes, device=self.device)
        self.itemC_dimensions = torch.stack([self.itemC_half_sizes] * 3, dim=-1)
        colorC = np.array([0, 1, 0, 1])  # Green
        self.itemC = self._build_cube_item(
            "itemC", itemC_half_sizes, colorC, frictions, densities, spawn_x=-0.2
        )

        # Set up greenscreening - keep robot and all items visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.itemA)
            self.remove_object_from_greenscreen(self.itemB)
            self.remove_object_from_greenscreen(self.itemC)

        # Convert rest_qpos to tensor
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        # Table pose
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # Build camera mount
        self._load_camera_mount()

        # Randomize robot color
        self._randomize_robot_color()

        # Goal sites (for visualization/debugging only)
        goalB_builder = self.scene.create_actor_builder()
        goalB_builder.add_sphere_visual(
            radius=0.01, material=sapien.render.RenderMaterial(base_color=[0, 1, 1, 1]),
        )
        goalB_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goalB_site = goalB_builder.build_kinematic(name="goalB_site")
        self._hidden_objects.append(self.goalB_site)

        goalA_builder = self.scene.create_actor_builder()
        goalA_builder.add_sphere_visual(
            radius=0.01, material=sapien.render.RenderMaterial(base_color=[1, 1, 0, 1]),
        )
        goalA_builder.initial_pose = sapien.Pose(p=[0, 0, 0.15])
        self.goalA_site = goalA_builder.build_kinematic(name="goalA_site")
        self._hidden_objects.append(self.goalA_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Random initial qpos
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            # Sample non-overlapping xy positions for all three items
            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )

            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size]
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            cfg = self.domain_randomization_config
            collision_margin = 0.01
            itemA_radius = cfg.itemA_half_size_range[1] + collision_margin
            itemB_radius = cfg.itemB_half_size_range[1] + collision_margin
            itemC_radius = cfg.itemC_half_size_range[1] + collision_margin

            itemA_xy_offset = sampler.sample(itemA_radius, 100)
            itemB_xy_offset = sampler.sample(itemB_radius, 100, verbose=False)
            itemC_xy_offset = sampler.sample(itemC_radius, 100, verbose=False)

            # itemA (red, top)
            itemA_xyz = torch.zeros((b, 3))
            itemA_xyz[:, :2] = spawn_center[env_idx, :2] + itemA_xy_offset
            itemA_xyz[:, 2] = self.itemA_half_sizes[env_idx]
            qsA = randomization.random_quaternions(b, lock_x=True, lock_y=True)

            # itemB (blue, middle)
            itemB_xyz = torch.zeros((b, 3))
            itemB_xyz[:, :2] = spawn_center[env_idx, :2] + itemB_xy_offset
            itemB_xyz[:, 2] = self.itemB_half_sizes[env_idx]
            qsB = randomization.random_quaternions(b, lock_x=True, lock_y=True)

            # itemC (green, base)
            itemC_xyz = torch.zeros((b, 3))
            itemC_xyz[:, :2] = spawn_center[env_idx, :2] + itemC_xy_offset
            itemC_xyz[:, 2] = self.itemC_half_sizes[env_idx]
            qsC = randomization.random_quaternions(b, lock_x=True, lock_y=True)

            # Curriculum: start a fraction of episodes with itemB already resting on itemC, so
            # the policy sees stage-2 states (grasp/place itemA) from step 0 instead of having to
            # first finish stage 1 and then stumble into stage 2 by exploration.
            if self.stage2_start_prob > 0:
                prestacked = torch.rand(b) < self.stage2_start_prob
                itemB_xyz[prestacked, :2] = itemC_xyz[prestacked, :2]
                itemB_xyz[prestacked, 2] = (
                    2 * self.itemC_half_sizes[env_idx] + self.itemB_half_sizes[env_idx]
                )[prestacked]
                qsB[prestacked] = qsC[prestacked]

            self.itemA.set_pose(Pose.create_from_pq(itemA_xyz, qsA))
            self.itemB.set_pose(Pose.create_from_pq(itemB_xyz, qsB))
            self.itemC.set_pose(Pose.create_from_pq(itemC_xyz, qsC))

            # Goal B is on top of itemC
            goalB_xyz = itemC_xyz.clone()
            goalB_xyz[:, 2] += self.itemC_half_sizes[env_idx] + self.itemB_half_sizes[env_idx]
            self.goalB_site.set_pose(Pose.create_from_pq(goalB_xyz))

            # Goal A is on top of goal B (final tower position)
            goalA_xyz = goalB_xyz.clone()
            goalA_xyz[:, 2] += self.itemB_half_sizes[env_idx] + self.itemA_half_sizes[env_idx]
            self.goalA_site.set_pose(Pose.create_from_pq(goalA_xyz))

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        # Adding joint noise for better sim2real
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            noise = torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                tcp_pose=self.agent.tcp_pose.raw_pose,
                is_itemA_grasped=info["is_itemA_grasped"],
                is_itemB_grasped=info["is_itemB_grasped"],
                itemA_pose=self.itemA.pose.raw_pose,
                itemB_pose=self.itemB.pose.raw_pose,
                itemC_pose=self.itemC.pose.raw_pose,
                tcp_to_itemA_pos=self.itemA.pose.p - self.agent.tcp_pos,
                tcp_to_itemB_pos=self.itemB.pose.p - self.agent.tcp_pos,
                tcp_to_itemC_pos=self.itemC.pose.p - self.agent.tcp_pos,
                itemA_to_itemB_pos=self.itemB.pose.p - self.itemA.pose.p,
                itemB_to_itemC_pos=self.itemC.pose.p - self.itemB.pose.p,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    itemA_dimensions=self.itemA_dimensions,
                    itemB_dimensions=self.itemB_dimensions,
                    itemC_dimensions=self.itemC_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    @staticmethod
    def _is_stacked(pos_top: torch.Tensor, half_top: torch.Tensor, pos_bottom: torch.Tensor, half_bottom: torch.Tensor):
        offset = pos_top - pos_bottom
        xy_dist = torch.linalg.norm(offset[:, :2], axis=1)
        xy_flag = xy_dist <= 0.02
        expected_z_offset = half_top + half_bottom
        z_dist = torch.abs(offset[:, 2] - expected_z_offset)
        z_flag = z_dist <= 0.01
        return xy_dist, z_dist, xy_flag & z_flag

    def evaluate(self):
        posA = self.itemA.pose.p
        posB = self.itemB.pose.p
        posC = self.itemC.pose.p

        AB_xy_dist, AB_z_dist, is_itemA_on_itemB = self._is_stacked(
            posA, self.itemA_half_sizes, posB, self.itemB_half_sizes
        )
        BC_xy_dist, BC_z_dist, is_itemB_on_itemC = self._is_stacked(
            posB, self.itemB_half_sizes, posC, self.itemC_half_sizes
        )

        itemA_vel = torch.linalg.norm(self.itemA.linear_velocity, axis=-1)
        itemB_vel = torch.linalg.norm(self.itemB.linear_velocity, axis=-1)
        is_itemA_static = itemA_vel <= 2e-2
        is_itemB_static = itemB_vel <= 2e-2

        is_itemA_grasped = self.agent.is_grasping(self.itemA)
        is_itemB_grasped = self.agent.is_grasping(self.itemB)
        is_itemA_lifted = self.itemA.pose.p[..., -1] >= (self.itemA_half_sizes + 1e-3)
        is_itemB_lifted = self.itemB.pose.p[..., -1] >= (self.itemB_half_sizes + 1e-3)
        is_robot_static = self.agent.is_static()

        # Contact checks
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_itemA = self.agent.is_touching(self.itemA)
        robot_touching_itemB = self.agent.is_touching(self.itemB)

        success = (
            is_itemB_on_itemC
            & is_itemA_on_itemB
            & is_itemA_static
            & is_itemB_static
            & (~robot_touching_itemA)
            & (~robot_touching_itemB)
            & is_robot_static
        )

        return {
            "AB_xy_dist": AB_xy_dist,
            "AB_z_dist": AB_z_dist,
            "BC_xy_dist": BC_xy_dist,
            "BC_z_dist": BC_z_dist,
            "itemA_vel": itemA_vel,
            "itemB_vel": itemB_vel,

            "success": success,
            "is_itemA_on_itemB": is_itemA_on_itemB,
            "is_itemB_on_itemC": is_itemB_on_itemC,
            "is_itemA_static": is_itemA_static,
            "is_itemB_static": is_itemB_static,
            "is_itemA_grasped": is_itemA_grasped,
            "is_itemB_grasped": is_itemB_grasped,
            "is_itemA_lifted": is_itemA_lifted,
            "is_itemB_lifted": is_itemB_lifted,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_itemA": robot_touching_itemA,
            "robot_touching_itemB": robot_touching_itemB,
        }

    def _stage_place_reward(
        self,
        tcp_pos: torch.Tensor,
        mover_pos: torch.Tensor,
        mover_half: torch.Tensor,
        base_pos: torch.Tensor,
        base_half: torch.Tensor,
        mover_grasped: torch.Tensor,
        mover_on_base: torch.Tensor,
        robot_touching_mover: torch.Tensor,
        mover_vel: torch.Tensor,
        robot_qvel: torch.Tensor,
    ):
        """Dense reward (range roughly [0, 9]) for picking `mover` up and placing
        it on top of `base`. Mirrors the two-item Stack task reward exactly."""
        # Reaching reward (TCP to mover)
        tcp_to_mover_dist = torch.linalg.norm(tcp_pos - mover_pos, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_mover_dist))
        reward = reaching_reward

        # Complex place reward (mover to goal on top of base)
        goal_z = base_pos[:, 2] + base_half + mover_half
        goal_xyz = torch.cat([base_pos[:, :2], goal_z.unsqueeze(1)], dim=1)

        mover_to_goal_dist = torch.linalg.norm(goal_xyz - mover_pos, axis=1)
        place_reward_final = 1 - torch.tanh(5.0 * mover_to_goal_dist)

        mover_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - mover_pos[..., :2], dim=1)
        # Far: target is 0.03m above the goal (encourages lifting before placing)
        mover_to_goal_dist_z_far = torch.linalg.norm(
            (goal_xyz[..., 2:] + 0.03) - mover_pos[..., 2:], dim=1
        )
        # Close: target is final position
        mover_to_goal_dist_z_close = torch.linalg.norm(goal_xyz[..., 2:] - mover_pos[..., 2:], dim=1)
        mover_close_to_goal = (mover_to_goal_dist_xy <= 0.04)
        mover_to_goal_dist_z = torch.where(mover_close_to_goal, mover_to_goal_dist_z_close, mover_to_goal_dist_z_far)
        place_reward_z = 1 - torch.tanh(10.0 * mover_to_goal_dist_z)
        place_reward = place_reward_final + place_reward_z

        # Ungrasp reward (inverted from Reach's close gripper)
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        ungrasp_reward = (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)

        # Grasped: 3 + place_reward
        reward[mover_grasped] = (3 + place_reward)[mover_grasped]

        # On base (still grasped): 4 + place_reward + gripper_openness
        on_base_and_grasped = mover_on_base & robot_touching_mover
        reward[on_base_and_grasped] = (4 + place_reward + ungrasp_reward)[on_base_and_grasped]

        # On base and released (not grasped): 7 + static_mover_reward + static_robot_reward
        static_mover_reward = 1 - torch.tanh(mover_vel * 10)
        static_robot_reward = 1 - torch.tanh(torch.linalg.norm(robot_qvel, axis=1) * 10)
        on_base_and_released = mover_on_base & (~robot_touching_mover)
        reward[on_base_and_released] = (7 + (static_mover_reward + static_robot_reward) / 2.0)[on_base_and_released]

        return reward

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_pos = self.agent.tcp_pose.p
        robot_qvel = self.agent.robot.get_qvel()[:, :-1]

        # Stage 1: pick up itemB and stack it on itemC
        stage1_reward = self._stage_place_reward(
            tcp_pos=tcp_pos,
            mover_pos=self.itemB.pose.p,
            mover_half=self.itemB_half_sizes,
            base_pos=self.itemC.pose.p,
            base_half=self.itemC_half_sizes,
            mover_grasped=info["is_itemB_grasped"],
            mover_on_base=info["is_itemB_on_itemC"],
            robot_touching_mover=info["robot_touching_itemB"],
            mover_vel=info["itemB_vel"],
            robot_qvel=robot_qvel,
        )
        # Stage 1 is considered locked in once itemB is placed on itemC and released
        stage1_done = info["is_itemB_on_itemC"] & (~info["robot_touching_itemB"])

        # Stage 2: pick up itemA and stack it on itemB (only meaningful once stage 1 is done,
        # but always computed so the reward is smooth/well-defined everywhere)
        stage2_reward = self._stage_place_reward(
            tcp_pos=tcp_pos,
            mover_pos=self.itemA.pose.p,
            mover_half=self.itemA_half_sizes,
            base_pos=self.itemB.pose.p,
            base_half=self.itemB_half_sizes,
            mover_grasped=info["is_itemA_grasped"],
            mover_on_base=info["is_itemA_on_itemB"],
            robot_touching_mover=info["robot_touching_itemA"],
            mover_vel=info["itemA_vel"],
            robot_qvel=robot_qvel,
        )

        # Total reward is monotonic across the task, following the exact same scale as
        # the 2-cube Stack task's own reward (0-8 while reaching/placing, +1 bump when a
        # cube is locked in on its base) chained twice: [0, 8] building the base, then a flat
        # +9 once itemB is locked in on itemC, plus [0, 8] for placing itemA on itemB, and
        # finally a +1 bump over the natural max (17) once the whole tower succeeds -> 18.
        reward = torch.where(stage1_done, 9 + stage2_reward, stage1_reward)
        reward[info["success"]] = 18

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 1 * (~info["is_itemB_lifted"]).float() * (~stage1_done).float()  # Encourage picking itemB fast
        reward -= 1 * (~info["is_itemA_lifted"]).float() * stage1_done.float()  # Encourage picking itemA fast once base is built

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 18


@register_env("SO101Stack3Cube-v1", max_episode_steps=150)
class Stack3Cube(Stack3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
