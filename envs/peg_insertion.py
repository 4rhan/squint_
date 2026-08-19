import dacite
import numpy as np
import sapien
import torch
from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence, Union
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

@dataclass
class PlugInsertionRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    plug_friction_range: Sequence[float] = (0.5, 1.0)
    socket_friction_range: Sequence[float] = (0.1, 0.5)


class PandaPlugInsertion(DefaultCameraEnv):
    """
    **Task Description:**
    Grasp a plug and insert it into a socket on the table using a Franka Panda.

    **Success Conditions:**
    - The plug is grasped and aligned inside the socket's hole.
    - The robot is static.
    """

    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state", "rgb+depth+segmentation", "rgb+depth+segmentation+state"]

    # Panda-specific wrist camera mount settings (overriding base defaults)
    WRIST_CAMERA_BASE_POS = (0.05, 0.0, 0.0) 
    WRIST_CAMERA_BASE_ROT_RAD = (0.0, np.deg2rad(45), np.deg2rad(-90))
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[PlugInsertionRandomizationConfig, dict] = PlugInsertionRandomizationConfig(),
        domain_randomization=False,
        **kwargs,
    ):
        # Merge DR config
        self.domain_randomization_config = PlugInsertionRandomizationConfig()
        merged_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=PlugInsertionRandomizationConfig, data=merged_config, config=dacite.Config(strict=True)
            )
        elif isinstance(domain_randomization_config, PlugInsertionRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        # Standard Panda rest pose
        self.rest_qpos = [0.0, 0.1963, 0.0, -2.6180, 0.0, 2.9416, 0.7854, 0.04, 0.04]
        
        super().__init__(
            *args, robot_uids=robot_uids, control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        # Place the Panda base offset from the center so it reaches the table easily
        super()._load_agent(options, sapien.Pose(p=[-0.6, 0, 0]), build_separate=self.domain_randomization)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # Build the Plug (Red Box)
        self.plug_half_size = [0.015, 0.015, 0.04]
        plug_builder = self.scene.create_actor_builder()
        plug_mat = sapien.pysapien.physx.PhysxMaterial(static_friction=1.0, dynamic_friction=1.0, restitution=0)
        plug_builder.add_box_collision(half_size=self.plug_half_size, material=plug_mat, density=1000)
        plug_builder.add_box_visual(
            half_size=self.plug_half_size, material=sapien.render.RenderMaterial(base_color=[0.8, 0.1, 0.1, 1.0])
        )
        self.plug = plug_builder.build(name="plug")

        # Build the Socket (White Square with Hole)
        socket_builder = self.scene.create_actor_builder()
        soc_mat = sapien.pysapien.physx.PhysxMaterial(static_friction=0.1, dynamic_friction=0.1, restitution=0)
        soc_color = sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1.0])
        
        # Base plate
        socket_builder.add_box_collision(pose=sapien.Pose([0, 0, 0.01]), half_size=[0.05, 0.05, 0.01], material=soc_mat)
        socket_builder.add_box_visual(pose=sapien.Pose([0, 0, 0.01]), half_size=[0.05, 0.05, 0.01], material=soc_color)
        
        # 4 walls forming a hole of size 0.06 x 0.06
        walls = [
            (sapien.Pose([0, 0.04, 0.04]), [0.05, 0.01, 0.02]),   # Left
            (sapien.Pose([0, -0.04, 0.04]), [0.05, 0.01, 0.02]),  # Right
            (sapien.Pose([0.04, 0, 0.04]), [0.01, 0.03, 0.02]),   # Front
            (sapien.Pose([-0.04, 0, 0.04]), [0.01, 0.03, 0.02]),  # Back
        ]
        for pose, hs in walls:
            socket_builder.add_box_collision(pose=pose, half_size=hs, material=soc_mat)
            socket_builder.add_box_visual(pose=pose, half_size=hs, material=soc_color)
            
        self.socket = socket_builder.build_kinematic(name="socket") # Static object on table
        
        # Overlay/Greenscreen exclusions
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.plug)
            self.remove_object_from_greenscreen(self.socket)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self._load_camera_mount()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Set Robot QPos
            self.agent.robot.set_qpos(self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.05)

            # Randomize Plug & Socket Locations
            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.15, -0.2], [0.15, 0.2]], batch_size=b, device=self.device
            )
            
            socket_xy = sampler.sample(0.08, 100)
            plug_xy = sampler.sample(0.04, 100, verbose=False)
            
            # Place Socket
            socket_xyz = torch.zeros((b, 3))
            socket_xyz[:, :2] = socket_xy
            self.socket.set_pose(Pose.create_from_pq(socket_xyz))

            # Place Plug
            plug_xyz = torch.zeros((b, 3))
            plug_xyz[:, :2] = plug_xy
            plug_xyz[:, 2] = self.plug_half_size[2]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.plug.set_pose(Pose.create_from_pq(plug_xyz, qs))

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            tcp_pose = self.agent.tcp.pose
            obs.update(
                tcp_pose=tcp_pose.raw_pose,
                plug_pose=self.plug.pose.raw_pose,
                socket_pose=self.socket.pose.raw_pose,
                tcp_to_plug=self.plug.pose.p - tcp_pose.p,
            )
        return obs

    def evaluate(self):
        # Get target hole position (socket center + some Z height)
        hole_pos = self.socket.pose.p.clone()
        hole_pos[:, 2] += 0.04 # Target hole depth
        
        offset = self.plug.pose.p - hole_pos
        xy_dist = torch.linalg.norm(offset[:, :2], axis=1)
        z_dist = torch.abs(offset[:, 2])

        is_inserted = (xy_dist < 0.02) & (z_dist < 0.015)
        is_grasped = self.agent.is_grasping(self.plug)
        
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-2], axis=1)
        is_robot_static = robot_v <= 0.2

        success = is_inserted & is_grasped & is_robot_static

        return {
            "success": success,
            "is_inserted": is_inserted,
            "is_grasped": is_grasped,
            "xy_dist": xy_dist,
            "z_dist": z_dist,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_pos = self.agent.tcp.pose.p
        plug_pos = self.plug.pose.p
        
        # 1. Reach Reward
        tcp_to_plug = torch.linalg.norm(tcp_pos - plug_pos, axis=1)
        reach_reward = 2 * (1 - torch.tanh(5 * tcp_to_plug))
        reward = reach_reward

        # 2. Grasp Reward
        reward[info["is_grasped"]] += 2.0

        # 3. Insertion Alignment Reward
        hole_pos = self.socket.pose.p.clone()
        hole_pos[:, 2] += 0.04
        
        plug_to_hole_xy = torch.linalg.norm(hole_pos[:, :2] - plug_pos[:, :2], axis=1)
        align_reward = 1 - torch.tanh(10 * plug_to_hole_xy)
        
        plug_to_hole_z = torch.abs(hole_pos[:, 2] - plug_pos[:, 2])
        insert_reward = 1 - torch.tanh(10 * plug_to_hole_z)
        
        # Only reward alignment/insertion if grasped
        reward[info["is_grasped"]] += align_reward[info["is_grasped"]] * 2.0
        
        # Only reward Z-insertion if XY is well-aligned
        is_aligned = plug_to_hole_xy < 0.03
        valid_insert = info["is_grasped"] & is_aligned
        reward[valid_insert] += insert_reward[valid_insert] * 2.0

        # 4. Success Bonus
        reward[info["success"]] += 5.0
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 13.0

@register_env("PandaPlugInsertion-v1", max_episode_steps=100)
class PandaPlugInsertionTask(PandaPlugInsertion):
    pass