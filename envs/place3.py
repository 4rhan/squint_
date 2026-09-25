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

NUM_CUBES = 3


@dataclass
class Place3RandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for the 3-cube Place task."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    cube_half_size_range: Sequence[float] = (0.0125, 0.0125)  # fixed 25 mm cubes (the size never varies)
    # Bin randomization (half sizes) - same ranges as the single-item Place task
    bin_half_size_x_range: Sequence[float] = (0.07 / 2, 0.09 / 2)
    bin_half_size_y_range: Sequence[float] = (0.09 / 2, 0.11 / 2)
    bin_half_size_z_range: Sequence[float] = (0.024 / 2, 0.036 / 2)

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


class Place3(DefaultCameraEnv):
    """
    **Task Description:**
    Pick up three cubes one by one and place all of them in a bin. The cubes are identical, so
    they can be moved in any order.

    **Randomizations:**
    - the three cubes and the bin have random xy positions on the table (non-overlapping)
    - the cubes' and the bin's z-axis rotation is randomized
    - cube sizes and bin sizes are randomized within configured ranges

    **Success Conditions:**
    - all three cubes are inside the bin
    - the robot is not touching any cube or the bin
    - the robot is static
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
            Place3RandomizationConfig, dict
        ] = Place3RandomizationConfig(),
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

        # Handle domain randomization config
        self.domain_randomization_config = Place3RandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=Place3RandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, Place3RandomizationConfig):
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

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        cfg = self.domain_randomization_config
        color = np.array([1.0, 0.0, 0.0, 1.0])  # all cubes red
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

        # ========== Cubes (each gets its own randomized size) ==========
        default_half = (cfg.cube_half_size_range[0] + cfg.cube_half_size_range[1]) / 2
        half_sizes_per_cube = []
        self.cubes = []
        for k in range(NUM_CUBES):
            half_sizes = np.ones(self.num_envs) * default_half
            if self.domain_randomization:
                half_sizes = self._batched_episode_rng.uniform(
                    low=cfg.cube_half_size_range[0], high=cfg.cube_half_size_range[1],
                )
            half_sizes_per_cube.append(common.to_tensor(half_sizes, device=self.device))

            items = []
            for i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                friction = frictions[i]
                material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=friction, dynamic_friction=friction, restitution=0,
                )
                builder.add_box_collision(
                    half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
                )
                builder.add_box_visual(
                    half_size=[half_sizes[i]] * 3,
                    material=sapien.render.RenderMaterial(base_color=color),
                )
                # Spread out at creation so nothing collides with the bin or each other
                builder.initial_pose = sapien.Pose(p=[0.2, (k - 1) * 0.06, half_sizes[i]])
                builder.set_scene_idxs([i])
                item = builder.build(name=f"cube{k}-{i}")
                items.append(item)
                self.remove_from_state_dict_registry(item)
            cube = Actor.merge(items, name=f"cube{k}")
            self.add_to_state_dict_registry(cube)
            self.cubes.append(cube)
        self.cube_half_sizes = torch.stack(half_sizes_per_cube, dim=1)  # (num_envs, NUM_CUBES)

        # ========== Bin (per-env for domain randomization) ==========
        bin_color = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        thickness = 0.005
        self.bin_thickness = thickness

        bin_half_sizes_x = np.ones(self.num_envs) * (cfg.bin_half_size_x_range[0] + cfg.bin_half_size_x_range[1]) / 2
        bin_half_sizes_y = np.ones(self.num_envs) * (cfg.bin_half_size_y_range[0] + cfg.bin_half_size_y_range[1]) / 2
        bin_half_sizes_z = np.ones(self.num_envs) * (cfg.bin_half_size_z_range[0] + cfg.bin_half_size_z_range[1]) / 2
        if self.domain_randomization:
            bin_half_sizes_x = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_x_range[0], high=cfg.bin_half_size_x_range[1]
            )
            bin_half_sizes_y = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_y_range[0], high=cfg.bin_half_size_y_range[1]
            )
            bin_half_sizes_z = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_z_range[0], high=cfg.bin_half_size_z_range[1]
            )
        self.bin_half_sizes_x = common.to_tensor(bin_half_sizes_x, device=self.device)
        self.bin_half_sizes_y = common.to_tensor(bin_half_sizes_y, device=self.device)
        self.bin_half_sizes_z = common.to_tensor(bin_half_sizes_z, device=self.device)
        self.bin_dimensions = torch.stack([self.bin_half_sizes_x, self.bin_half_sizes_y, self.bin_half_sizes_z], dim=-1)

        bins = []
        for i in range(self.num_envs):
            bin_half_size = [bin_half_sizes_x[i], bin_half_sizes_y[i], bin_half_sizes_z[i]]
            builder = self.scene.create_actor_builder()

            # Bin floor
            bin_center_pose = sapien.Pose([0.0, 0.0, thickness / 2])
            bin_center_half_size = [bin_half_size[0], bin_half_size[1], thickness / 2]
            builder.add_box_collision(pose=bin_center_pose, half_size=bin_center_half_size)
            builder.add_box_visual(pose=bin_center_pose, half_size=bin_center_half_size, material=bin_color)

            # Bin walls
            for j in [-1, 1]:
                y = j * bin_center_half_size[1]
                wall_pose = sapien.Pose([0, y, bin_half_size[2]])
                wall_half_size = [bin_half_size[0], thickness / 2, bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)
                x = j * bin_center_half_size[0]
                wall_pose = sapien.Pose([x, 0, bin_half_size[2]])
                wall_half_size = [thickness / 2, bin_half_size[1], bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)

            builder.initial_pose = sapien.Pose(p=[-0.2, 0, bin_half_size[2]])
            builder.set_scene_idxs([i])
            bin_actor = builder.build(name=f"bin-{i}")
            bins.append(bin_actor)
            self.remove_from_state_dict_registry(bin_actor)

        self.bin = Actor.merge(bins, name="bin")
        self.add_to_state_dict_registry(self.bin)
        self.bin_radius = torch.linalg.norm(self.bin_dimensions[:, :2], dim=-1)

        # Set up greenscreening - keep robot, cubes, and bin visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            for cube in self.cubes:
                self.remove_object_from_greenscreen(cube)
            self.remove_object_from_greenscreen(self.bin)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        self._load_camera_mount()
        self._randomize_robot_color()

        # Goal site (visualization only)
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.01, material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    def _sample_layout(self, env_idx: torch.Tensor):
        """Samples a non-overlapping layout for the bin and all cubes.

        Returns xy offsets relative to the spawn center: bin_xy (b, 2), bin_yaw (b,), cube_xy (b, NUM_CUBES, 2).
        Cubes are rejection-sampled against the bin's true (rotated) rectangle and each other, which is
        far less wasteful than bounding circles when the bin is large compared to the spawn area.
        """
        b = len(env_idx)
        half = self.spawn_box_half_size
        cube_r = self.domain_randomization_config.cube_half_size_range[1] * np.sqrt(2)  # cube circumradius
        margin = 0.01
        n_cand = 512

        bin_xy = (torch.rand(b, 2, device=self.device) * 2 - 1) * half
        bin_yaw = torch.rand(b, device=self.device) * 2 * np.pi
        hx = self.bin_half_sizes_x[env_idx][:, None]
        hy = self.bin_half_sizes_y[env_idx][:, None]
        cos, sin = torch.cos(bin_yaw)[:, None], torch.sin(bin_yaw)[:, None]

        cube_xy = torch.zeros(b, NUM_CUBES, 2, device=self.device)
        batch = torch.arange(b, device=self.device)
        for j in range(NUM_CUBES):
            cand = (torch.rand(b, n_cand, 2, device=self.device) * 2 - 1) * half
            rel = cand - bin_xy[:, None, :]
            x_b = cos * rel[..., 0] + sin * rel[..., 1]
            y_b = -sin * rel[..., 0] + cos * rel[..., 1]
            ok = (x_b.abs() > hx + cube_r + margin) | (y_b.abs() > hy + cube_r + margin)
            for k in range(j):
                dist = torch.linalg.norm(cand - cube_xy[:, k, None, :], dim=-1)
                ok &= dist > 2 * cube_r + margin
            idx = ok.float().argmax(dim=1)  # first valid candidate (0 if none are valid)
            cube_xy[:, j] = cand[batch, idx]
        return bin_xy, bin_yaw, cube_xy

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )
            bin_xy, bin_yaw, cube_xy = self._sample_layout(env_idx)

            for k, cube in enumerate(self.cubes):
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = spawn_center[env_idx, :2] + cube_xy[:, k]
                xyz[:, 2] = self.cube_half_sizes[env_idx, k]
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                cube.set_pose(Pose.create_from_pq(xyz, qs))

            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = spawn_center[env_idx, :2] + bin_xy
            bin_xyz[:, 2] = 0.001  # tiny lift so the floor doesn't start inside the table
            bin_q = torch.stack(
                [torch.cos(bin_yaw / 2), torch.zeros_like(bin_yaw), torch.zeros_like(bin_yaw), torch.sin(bin_yaw / 2)],
                dim=-1,
            )
            self.bin.set_pose(Pose.create_from_pq(bin_xyz, bin_q))

            goal_xyz = bin_xyz.clone()
            goal_xyz[:, 2] = self.bin_thickness + self.cube_half_sizes[env_idx].mean(dim=-1)
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    # ------------------------------------------------------------------ helpers
    def _cube_positions(self):
        return torch.stack([cube.pose.p for cube in self.cubes], dim=1)  # (N, NUM_CUBES, 3)

    def _in_bin(self, cube_pos: torch.Tensor):
        """(N, NUM_CUBES) bool: cube center is inside the bin's interior, evaluated in the bin's own (rotated) frame."""
        bin_pose = self.bin.pose
        q = bin_pose.q
        yaw = 2 * torch.atan2(q[:, 3], q[:, 0])
        cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        rel = cube_pos[..., :2] - bin_pose.p[:, None, :2]
        x_b = cos * rel[..., 0] + sin * rel[..., 1]
        y_b = -sin * rel[..., 0] + cos * rel[..., 1]

        margin = self.bin_thickness + 0.5 * self.cube_half_sizes  # keeps rim-balancing cubes out
        inside_x = x_b.abs() < self.bin_half_sizes_x[:, None] - margin
        inside_y = y_b.abs() < self.bin_half_sizes_y[:, None] - margin
        # not being held above the bin: center must be below the rim (+ a cube of slack for stacked cubes)
        rim_z = bin_pose.p[:, None, 2] + 2 * self.bin_half_sizes_z[:, None] + 2 * self.cube_half_sizes
        return inside_x & inside_y & (cube_pos[..., 2] < rim_z)

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
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
            tcp = self.agent.tcp_pos
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                tcp_pose=self.agent.tcp_pose.raw_pose,
                bin_pose=self.bin.pose.raw_pose,
                tcp_to_bin_pos=self.bin.pose.p - tcp,
                num_in_bin=info["num_in_bin"],
            )
            for k, cube in enumerate(self.cubes):
                obs[f"cube{k}_pose"] = cube.pose.raw_pose
                obs[f"tcp_to_cube{k}_pos"] = cube.pose.p - tcp
                obs[f"cube{k}_to_bin_pos"] = self.bin.pose.p - cube.pose.p
                obs[f"is_cube{k}_grasped"] = info[f"cube{k}_grasped"]
                obs[f"is_cube{k}_in_bin"] = info[f"cube{k}_in_bin"]
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    cube_dimensions=self.cube_half_sizes,
                    bin_dimensions=self.bin_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    def evaluate(self):
        cube_pos = self._cube_positions()
        in_bin = self._in_bin(cube_pos)  # (N, NUM_CUBES)
        num_in_bin = in_bin.sum(dim=-1)

        grasped = torch.stack([self.agent.is_grasping(c) for c in self.cubes], dim=-1)
        touching = torch.stack([self.agent.is_touching(c) for c in self.cubes], dim=-1)
        lifted = cube_pos[..., 2] >= (self.cube_half_sizes + 1e-3)
        is_robot_static = self.agent.is_static()

        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_bin = self.agent.is_touching(self.bin)

        success = (num_in_bin == NUM_CUBES) & (~touching.any(dim=-1)) & is_robot_static & (~robot_touching_bin)

        info = {
            "success": success,
            "num_in_bin": num_in_bin.float(),
            "in_bin_ge1": num_in_bin >= 1,
            "in_bin_ge2": num_in_bin >= 2,
            "in_bin_ge3": num_in_bin >= 3,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_bin": robot_touching_bin,
        }
        for k in range(NUM_CUBES):
            info[f"cube{k}_in_bin"] = in_bin[:, k]
            info[f"cube{k}_grasped"] = grasped[:, k]
            info[f"cube{k}_touching"] = touching[:, k]
            info[f"cube{k}_lifted"] = lifted[:, k]
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        n = self.num_envs
        arange = torch.arange(n, device=self.device)
        tcp_pos = self.agent.tcp_pose.p
        cube_pos = self._cube_positions()  # (N, K, 3)

        in_bin = torch.stack([info[f"cube{k}_in_bin"] for k in range(NUM_CUBES)], dim=-1)
        grasped = torch.stack([info[f"cube{k}_grasped"] for k in range(NUM_CUBES)], dim=-1)
        touching = torch.stack([info[f"cube{k}_touching"] for k in range(NUM_CUBES)], dim=-1)
        lifted = torch.stack([info[f"cube{k}_lifted"] for k in range(NUM_CUBES)], dim=-1)

        # Every cube already in the bin is banked: 7, +1 once the gripper has let go of it
        banked = ((7 + (~touching).float()) * in_bin.float()).sum(dim=-1)

        # The "active" cube is the one still to be moved: a held one if any, otherwise the nearest
        unplaced = ~in_bin
        any_unplaced = unplaced.any(dim=-1)
        dist_to_tcp = torch.linalg.norm(cube_pos - tcp_pos[:, None, :], dim=-1)
        score = torch.where(unplaced, dist_to_tcp - 10.0 * grasped.float(), torch.full_like(dist_to_tcp, 1e6))
        active = score.argmin(dim=-1)

        a_pos = cube_pos[arange, active]
        a_half = self.cube_half_sizes[arange, active]
        a_grasped = grasped[arange, active]
        a_lifted = lifted[arange, active]

        # Reaching
        reaching_reward = 2 * (1 - torch.tanh(5 * dist_to_tcp[arange, active]))

        # Placing (same shaping as the single-item Place task, measured against the bin)
        bin_pos = self.bin.pose.p
        goal_xyz = bin_pos.clone()
        goal_xyz[:, 2] = bin_pos[:, 2] + self.bin_thickness + a_half
        dist_to_goal = torch.linalg.norm(goal_xyz - a_pos, dim=1)
        place_reward_final = 1 - torch.tanh(5.0 * dist_to_goal)

        dist_xy = torch.linalg.norm(goal_xyz[:, :2] - a_pos[:, :2], dim=1)
        # Far from the bin: aim above the rim (encourages lifting before moving over)
        z_far = goal_xyz[:, 2] + 2 * self.bin_half_sizes_z + 0.03
        dist_z = torch.where(dist_xy <= self.bin_radius, torch.abs(goal_xyz[:, 2] - a_pos[:, 2]), torch.abs(z_far - a_pos[:, 2]))
        place_reward_z = 1 - torch.tanh(10.0 * dist_z)
        place_reward = place_reward_final + place_reward_z

        shaped = torch.where(a_grasped, 3 + place_reward, reaching_reward) * any_unplaced.float()

        reward = banked + shaped

        # Everything is in the bin: reward the arm settling
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], dim=1)
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        reward = reward + (~any_unplaced).float() * static_robot_reward

        reward[info["success"]] = 26

        # Penalties (same as the single-item Place task)
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 3 * info["robot_touching_bin"].float()
        reward -= 1 * ((~a_lifted) & any_unplaced).float()  # encourage picking the next cube fast
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 26


@register_env("SO101Place3Cube-v1", max_episode_steps=250)
class Place3Cube(Place3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
