from dataclasses import dataclass
from typing import Any, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

from .robot.so100 import SO100
from .robot.so101 import SO101


def _build_box_with_hole(scene, inner_radius, outer_radius, depth, center=(0, 0)):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


@dataclass
class PegInsertionRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for PegInsertion task, extending wrist camera randomization."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Peg half length / radius randomization (box hole radius tracks peg radius + clearance)
    peg_half_length_range: Sequence[float] = (0.035, 0.05)
    peg_radius_range: Sequence[float] = (0.006, 0.009)

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)


@register_env("SO101PegInsertionSide-v1", max_episode_steps=100)
class PegInsertionSide(DefaultCameraEnv):
    """
    **Task Description:**
    Pick up an orange-white peg with the SO100/SO101 arm and insert the orange end into the
    box with a hole in it. This is a scaled-down adaptation of the original Panda
    PegInsertionSide-v1 task for the smaller desktop SO100/SO101 arms.

    **Randomizations:**
    - Peg half length is randomized within `peg_half_length_range`. Box half length matches it.
    - Peg radius is randomized within `peg_radius_range`. Box hole radius is peg radius + clearance.
    - Peg is laid flat on the table with randomized xy position and z-axis rotation.
    - Box is laid flat on the table with randomized xy position and z-axis rotation.

    **Success Conditions:**
    - The orange end of the peg is inserted past the hole entrance and within the hole's radius.
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101]
    _clearance = 0.0025

    def __init__(
        self,
        *args,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            PegInsertionRandomizationConfig, dict
        ] = PegInsertionRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.2 / 2,
        **kwargs,
    ):
        # Robot-specific configuration
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        # Handle domain randomization config - merge with defaults
        self.domain_randomization_config = PegInsertionRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=PegInsertionRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, PegInsertionRandomizationConfig):
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
        # load the robot arm at this initial pose
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_scene(self, options: dict):
        # we use a predefined table scene builder which simply adds a table and floor to the scene
        # where the 0, 0, 0 position is the center of the table
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        cfg = self.domain_randomization_config
        lengths = np.ones(self.num_envs) * sum(cfg.peg_half_length_range) / 2
        radii = np.ones(self.num_envs) * sum(cfg.peg_radius_range) / 2
        frictions = np.ones(self.num_envs) * sum(cfg.item_friction_range) / 2
        densities = np.ones(self.num_envs) * sum(cfg.item_density_range) / 2
        centers = np.zeros((self.num_envs, 2))

        if self.domain_randomization:
            lengths = self._batched_episode_rng.uniform(
                low=cfg.peg_half_length_range[0], high=cfg.peg_half_length_range[1]
            )
            radii = self._batched_episode_rng.uniform(
                low=cfg.peg_radius_range[0], high=cfg.peg_radius_range[1]
            )
            frictions = self._batched_episode_rng.uniform(
                low=cfg.item_friction_range[0], high=cfg.item_friction_range[1]
            )
            densities = self._batched_episode_rng.uniform(
                low=cfg.item_density_range[0], high=cfg.item_density_range[1]
            )
            centers = 0.5 * (lengths - radii)[:, None] * self._batched_episode_rng.uniform(
                -1, 1, size=(2,)
            )

        # save some useful values for use later
        self.peg_half_sizes = common.to_tensor(
            np.vstack([lengths, radii, radii]).T, device=self.device
        )
        peg_head_offsets = torch.zeros((self.num_envs, 3), device=self.device)
        peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
        self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

        box_hole_offsets = torch.zeros((self.num_envs, 3), device=self.device)
        box_hole_offsets[:, 1:] = common.to_tensor(centers, device=self.device)
        self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
        self.box_hole_radii = common.to_tensor(radii + self._clearance, device=self.device)

        # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
        pegs = []
        boxes = []
        for i in range(self.num_envs):
            length = lengths[i]
            radius = radii[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=frictions[i],
                dynamic_friction=frictions[i],
                restitution=0,
            )

            builder = self.scene.create_actor_builder()
            builder.add_box_collision(
                half_size=[length, radius, radius], material=material, density=densities[i]
            )
            # peg head
            mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#EC7357"), roughness=0.5, specular=0.5,
            )
            builder.add_box_visual(
                sapien.Pose([length / 2, 0, 0]),
                half_size=[length / 2, radius, radius],
                material=mat,
            )
            # peg tail
            mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#EDF6F9"), roughness=0.5, specular=0.5,
            )
            builder.add_box_visual(
                sapien.Pose([-length / 2, 0, 0]),
                half_size=[length / 2, radius, radius],
                material=mat,
            )
            builder.initial_pose = sapien.Pose(p=[0.2, 0, 0.1])  # Offset to avoid collision with box at creation
            builder.set_scene_idxs([i])
            peg = builder.build(f"peg_{i}")
            self.remove_from_state_dict_registry(peg)

            # box with hole
            inner_radius, outer_radius, depth = radius + self._clearance, length, length
            builder = _build_box_with_hole(
                self.scene, inner_radius, outer_radius, depth, center=centers[i]
            )
            builder.initial_pose = sapien.Pose(p=[-0.2, 0, 0.1])  # Offset to avoid collision with peg at creation
            builder.set_scene_idxs([i])
            box = builder.build_kinematic(f"box_with_hole_{i}")
            self.remove_from_state_dict_registry(box)

            pegs.append(peg)
            boxes.append(box)

        # since we are building many different pegs/boxes but simulating in parallel, we need to merge them
        # into single actors so we can access each different env's information with a single object
        self.peg = Actor.merge(pegs, "peg")
        self.box = Actor.merge(boxes, "box_with_hole")
        self.add_to_state_dict_registry(self.peg)
        self.add_to_state_dict_registry(self.box)

        # Set up greenscreening - keep robot, peg, and box visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.peg)
            self.remove_object_from_greenscreen(self.box)

        # Convert rest_qpos to tensor
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        # hardcoded pose for the table that places it such that the robot base is at 0 and on the edge of the table.
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # build the camera mount (from parent class)
        self._load_camera_mount()

        # randomize or set a fixed robot color (from parent class)
        self._randomize_robot_color()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # move the table back so that the robot is at 0 and on the edge of the table.
            self.table_scene.table.set_pose(self.table_pose)

            # sample a random initial joint configuration for the robot
            self.agent.robot.set_qpos(
                self.rest_qpos
                + torch.randn(size=(b, self.rest_qpos.shape[-1]))
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            # sample non-overlapping positions for the peg and box within the spawn region
            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )
            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size],
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            peg_radius = self.peg_half_sizes[env_idx, 0].max().item() + 0.01
            box_radius = self.peg_half_sizes[env_idx, 0].max().item() + 0.01

            peg_xy_offset = sampler.sample(peg_radius, 100)
            box_xy_offset = sampler.sample(box_radius, 100, verbose=False)

            # peg pose: laid flat on the table, xy randomized, z-axis rotation randomized
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = spawn_center[env_idx, :2] + peg_xy_offset
            xyz[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3),
            )
            self.peg.set_pose(Pose.create_from_pq(xyz, quat))

            # box pose: laid flat on the table, xy randomized, z-axis rotation randomized
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = spawn_center[env_idx, :2] + box_xy_offset
            xyz[:, 2] = self.peg_half_sizes[env_idx, 0]
            quat = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            self.box.set_pose(Pose.create_from_pq(xyz, quat))

    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE: this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.008 <= peg_head_pos_at_hole[:, 0]
        y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )

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
                is_grasped=info["is_grasped"],
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
                tcp_to_peg_pos=self.peg.pose.p - self.agent.tcp_pos,
            )
        return obs

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        is_grasped = self.agent.is_grasping(self.peg)
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        return dict(
            success=success,
            peg_head_pos_at_hole=peg_head_pos_at_hole,
            is_grasped=is_grasped,
            robot_touching_table=robot_touching_table,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1: reach and grasp the peg's tail (white end)
        tail_offset = torch.zeros((self.num_envs, 3), device=self.device)
        tail_offset[:, 0] = -self.peg_half_sizes[:, 0] * 0.6
        tgt_gripper_pose = self.peg.pose * Pose.create_from_pq(p=tail_offset)
        gripper_to_peg_dist = torch.linalg.norm(
            self.agent.tcp_pos - tgt_gripper_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        is_grasped = info["is_grasped"]
        reward = reaching_reward + is_grasped

        # Stage 2: orient the grasped peg properly towards the hole

        # pre-insertion reward, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        pre_insertion_reward = 3 * (
            1
            - torch.tanh(
                0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
            )
        )
        reward += pre_insertion_reward * is_grasped
        # stage 2 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.006) & (
            peg_wrt_goal_yz_dist < 0.006
        )

        # Stage 3: insert the peg into the hole once it is grasped and lined up
        peg_head_wrt_hole = self.box_hole_pose.inv() * self.peg_head_pose
        insertion_reward = 5 * (
            1 - torch.tanh(8.0 * torch.linalg.norm(peg_head_wrt_hole.p, axis=1))
        )
        reward += insertion_reward * (is_grasped & pre_inserted)

        # Penalties
        reward -= 3 * info["robot_touching_table"].float()

        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10
