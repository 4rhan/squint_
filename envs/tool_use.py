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
class ToolSweepRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    tool_friction_range: Sequence[float] = (0.5, 1.0)
    puck_friction_range: Sequence[float] = (0.1, 0.4)


class PandaToolSweep(DefaultCameraEnv):
    """
    **Task Description:**
    A true tool-use task. Grasp a T-shaped sweeper tool and use its blade to push 
    a puck into a goal region on the table.

    **Success Conditions:**
    - The puck is inside the goal region.
    - The robot is static.
    """

    SUPPORTED_ROBOTS = ["SO101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state", "rgb+depth+segmentation", "rgb+depth+segmentation+state"]

    # Panda-specific wrist camera mount settings
    WRIST_CAMERA_BASE_POS = (0.05, 0.0, 0.0) 
    WRIST_CAMERA_BASE_ROT_RAD = (0.0, np.deg2rad(45), np.deg2rad(-90))
    
    def __init__(
        self,
        *args,
        robot_uids="panda",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[ToolSweepRandomizationConfig, dict] = ToolSweepRandomizationConfig(),
        domain_randomization=False,
        **kwargs,
    ):
        self.domain_randomization_config = ToolSweepRandomizationConfig()
        merged_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=ToolSweepRandomizationConfig, data=merged_config, config=dacite.Config(strict=True)
            )
        elif isinstance(domain_randomization_config, ToolSweepRandomizationConfig):
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
        # Place Panda slightly further back so it can sweep across the table easily
        super()._load_agent(options, sapien.Pose(p=[-0.6, 0, 0]), build_separate=self.domain_randomization)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        cfg = self.domain_randomization_config

        # 1. Build the Sweeper Tool (T-Shape)
        sweepers = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([i])
            
            fric = np.random.uniform(*cfg.tool_friction_range) if self.domain_randomization else 0.7
            mat = sapien.pysapien.physx.PhysxMaterial(static_friction=fric, dynamic_friction=fric, restitution=0)
            
            # Handle (SAPIEN cylinders point along X by default, we rotate to Z)
            handle_pose = sapien.Pose(q=euler2quat(0, np.pi/2, 0))
            builder.add_cylinder_collision(radius=0.015, half_length=0.08, pose=handle_pose, material=mat, density=500)
            builder.add_cylinder_visual(radius=0.015, half_length=0.08, pose=handle_pose, 
                                        material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.2, 1]))
            
            # Blade (attached to the bottom of the handle)
            blade_pose = sapien.Pose([0, 0, -0.08])
            builder.add_box_collision(pose=blade_pose, half_size=[0.01, 0.06, 0.02], material=mat, density=1000)
            builder.add_box_visual(pose=blade_pose, half_size=[0.01, 0.06, 0.02], 
                                   material=sapien.render.RenderMaterial(base_color=[0.2, 0.2, 0.2, 1]))
            
            sweepers.append(builder.build(name=f"sweeper-{i}"))
        self.sweeper = Actor.merge(sweepers, name="sweeper")

        # 2. Build the Puck
        pucks = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([i])
            
            fric = np.random.uniform(*cfg.puck_friction_range) if self.domain_randomization else 0.2
            mat = sapien.pysapien.physx.PhysxMaterial(static_friction=fric, dynamic_friction=fric, restitution=0)
            
            puck_pose = sapien.Pose(q=euler2quat(0, np.pi/2, 0))
            builder.add_cylinder_collision(radius=0.03, half_length=0.015, pose=puck_pose, material=mat, density=200)
            builder.add_cylinder_visual(radius=0.03, half_length=0.015, pose=puck_pose, 
                                        material=sapien.render.RenderMaterial(base_color=[0.1, 0.5, 0.8, 1]))
            
            pucks.append(builder.build(name=f"puck-{i}"))
        self.puck = Actor.merge(pucks, name="puck")

        # 3. Build Goal Region (Visual only)
        goal_builder = self.scene.create_actor_builder()
        goal_pose = sapien.Pose(q=euler2quat(0, np.pi/2, 0))
        goal_builder.add_cylinder_visual(radius=0.08, half_length=0.001, pose=goal_pose, 
                                         material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.3]))
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.sweeper)
            self.remove_object_from_greenscreen(self.puck)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self._load_camera_mount()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.robot.set_qpos(self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.05)

            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.15, -0.2], [0.15, 0.2]], batch_size=b, device=self.device
            )
            
            # Place Sweeper Tool flat on table
            sweeper_xy = sampler.sample(0.1, 100)
            sweeper_xyz = torch.zeros((b, 3))
            sweeper_xyz[:, :2] = sweeper_xy
            sweeper_xyz[:, 2] = 0.015 # Radius of handle
            
            yaw_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True) # Random rotation on table
            pitch_q = torch.tensor(euler2quat(0, np.pi/2, 0), device=self.device).float().unsqueeze(0).repeat(b, 1)
            # Combine rotations so the tool lays perfectly flat
            sweeper_pose = Pose.create_from_pq(sweeper_xyz, yaw_qs) * Pose.create_from_pq(torch.zeros(3), pitch_q)
            self.sweeper.set_pose(sweeper_pose)

            # Place Puck
            puck_xy = sampler.sample(0.04, 100, verbose=False)
            puck_xyz = torch.zeros((b, 3))
            puck_xyz[:, :2] = puck_xy
            puck_xyz[:, 2] = 0.015
            self.puck.set_pose(Pose.create_from_pq(puck_xyz))

            # Place Goal Site
            goal_xy = sampler.sample(0.09, 100, verbose=False)
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = goal_xy
            goal_xyz[:, 2] = 0.001
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                tcp_pose=self.agent.tcp.pose.raw_pose,
                sweeper_pose=self.sweeper.pose.raw_pose,
                puck_pose=self.puck.pose.raw_pose,
                goal_pos=self.goal_site.pose.p,
            )
        return obs

    def evaluate(self):
        puck_pos = self.puck.pose.p
        goal_pos = self.goal_site.pose.p
        
        # Check if puck is inside the 0.08m goal radius
        puck_to_goal = torch.linalg.norm(puck_pos[:, :2] - goal_pos[:, :2], axis=1)
        is_puck_in_goal = puck_to_goal < 0.08
        
        is_grasped = self.agent.is_grasping(self.sweeper)
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-2], axis=1)
        is_robot_static = robot_v <= 0.2

        success = is_puck_in_goal & is_robot_static

        return {
            "success": success,
            "is_puck_in_goal": is_puck_in_goal,
            "is_grasped": is_grasped,
            "puck_to_goal": puck_to_goal
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_pos = self.agent.tcp.pose.p
        handle_pos = self.sweeper.pose.p
        
        # 1. Reach & Grasp Tool
        tcp_to_handle = torch.linalg.norm(tcp_pos - handle_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * tcp_to_handle))
        
        is_grasped = info["is_grasped"]
        reward[is_grasped] += 2.0
        
        # 2. Tool Blade to Puck Alignment
        # Calculate where the blade is in world coordinates using batched Pose math
        blade_local = Pose.create_from_pq(p=torch.tensor([0.0, 0.0, -0.08], device=self.device))
        blade_world_pose = self.sweeper.pose * blade_local
        blade_pos = blade_world_pose.p
        
        puck_pos = self.puck.pose.p
        blade_to_puck = torch.linalg.norm(blade_pos - puck_pos, axis=1)
        align_reward = 2 * (1 - torch.tanh(5 * blade_to_puck))
        reward[is_grasped] += align_reward[is_grasped]
        
        # 3. Sweep Puck to Goal
        puck_to_goal = info["puck_to_goal"]
        sweep_reward = 4 * (1 - torch.tanh(3 * puck_to_goal))
        
        # Only reward moving the puck if grasped and tool is actually near the puck
        valid_sweep = is_grasped & (blade_to_puck < 0.15)
        reward[valid_sweep] += sweep_reward[valid_sweep]
        
        # 4. Success Bonus
        reward[info["success"]] += 5.0
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 15.0

@register_env("PandaToolSweep-v1", max_episode_steps=100)
class PandaToolSweepTask(PandaToolSweep):
    pass