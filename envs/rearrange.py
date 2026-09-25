"""Task C: cyclic rearrangement with a buffer (SO-101).

Three occupied target pockets + one empty buffer (3-object version), and a
two-object swap control with a buffer.

  - SO101Rearrange3-v1 : pockets P0,P1,P2 hold A,B,C initially; goal B,C,A; P3 buffer.
  - SO101Rearrange2-v1 : pockets P0,P1 hold A,B initially; goal B,A; P2 buffer
    (same pocket geometry, subset).

Objects: same-size cubes, distinct colors. Pockets: shallow printed pockets
with floors/rim colors marking the DESTINATION object (final assignment), so
the goal is observable. Buffer pocket neutral. Any physically valid solution
is accepted; no sequence is hard-coded.
"""

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

COLORS3 = [np.array([1, 0, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])]  # A,B,C
NEUTRAL = np.array([0.85, 0.85, 0.85, 1])


@dataclass
class RearrangeRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    cube_half_size_range: Sequence[float] = (0.011, 0.013)
    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


class RearrangeBase(DefaultCameraEnv):
    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation",
                           "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101]
    NUM_OBJECTS = 3  # 2 for swap control
    N_POCKETS = 4    # 3 for swap control (2 targets + buffer)

    def __init__(
        self, *args, robot_uids="so101", control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[RearrangeRandomizationConfig, dict] = RearrangeRandomizationConfig(),
        domain_randomization=False,
        # Pocket geometry (configurable, printable).
        pocket_inner: float = 0.036, pocket_wall_t: float = 0.005,
        pocket_wall_h: float = 0.012, pocket_floor_t: float = 0.005,
        # Pocket centers (table frame). 4 in a row along y at x=0.30.
        pocket_x: float = 0.30, pocket_ys=( -0.09, -0.03, 0.03, 0.09),
        place_xy_tol: float = 0.012, place_z_tol: float = 0.006,
        static_vel_thresh: float = 2e-2, dwell_time: float = 1.0,
        **kwargs,
    ):
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()
        else:
            raise ValueError(robot_uids)
        self.domain_randomization_config = RearrangeRandomizationConfig()
        merged = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=RearrangeRandomizationConfig, data=merged, config=dacite.Config(strict=True))
        elif isinstance(domain_randomization_config, RearrangeRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config
        self.pocket_inner = pocket_inner
        self.pocket_wall_t = pocket_wall_t
        self.pocket_wall_h = pocket_wall_h
        self.pocket_floor_t = pocket_floor_t
        self.pocket_x = pocket_x
        self.pocket_ys = tuple(pocket_ys)
        self.place_xy_tol = place_xy_tol
        self.place_z_tol = place_z_tol
        self.static_vel_thresh = static_vel_thresh
        self.dwell_time = dwell_time
        super().__init__(*args, robot_uids=robot_uids, control_mode=control_mode,
                         domain_randomization=domain_randomization,
                         domain_randomization_config=self.domain_randomization_config, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
                            build_separate=True if self.domain_randomization
                            and self.domain_randomization_config.robot_color == "random" else False)

    @property
    def _default_sim_config(self):
        from mani_skill.utils.structs.types import SimConfig
        return SimConfig(sim_freq=100, control_freq=10)

    @property
    def control_freq(self):
        try:
            return self._sim_config.control_freq
        except Exception:
            return 10

    # Permutation maps: initial pocket -> object, final pocket -> object.
    # Objects indexed 0,1,2 (A,B,C). Pockets 0..N-1, last pocket is buffer.
    def _init_map(self):
        if self.NUM_OBJECTS == 3:
            return [0, 1, 2]  # pocket i holds object i initially
        return [0, 1]

    def _goal_map(self):
        if self.NUM_OBJECTS == 3:
            # B,C,A in pockets 0,1,2.
            return [1, 2, 0]
        return [1, 0]

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2
        if self.domain_randomization:
            frictions = self._batched_episode_rng.uniform(low=cfg.item_friction_range[0], high=cfg.item_friction_range[1])
            densities = self._batched_episode_rng.uniform(low=cfg.item_density_range[0], high=cfg.item_density_range[1])
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)
        default_half = (cfg.cube_half_size_range[0] + cfg.cube_half_size_range[1]) / 2
        half = np.ones(self.num_envs) * default_half
        if self.domain_randomization:
            half = self._batched_episode_rng.uniform(low=cfg.cube_half_size_range[0], high=cfg.cube_half_size_range[1])
        self.cube_half = common.to_tensor(half, device=self.device)
        self.cube_dims = torch.stack([self.cube_half] * 3, dim=-1)

        n_obj = 3  # always build 3 cubes; variants use first NUM_OBJECTS
        self.cubes = []
        for k in range(n_obj):
            items = []
            for i in range(self.num_envs):
                b = self.scene.create_actor_builder()
                mat = sapien.pysapien.physx.PhysxMaterial(static_friction=frictions[i], dynamic_friction=frictions[i], restitution=0)
                b.add_box_collision(half_size=[half[i]] * 3, material=mat, density=densities[i])
                b.add_box_visual(half_size=[half[i]] * 3, material=sapien.render.RenderMaterial(base_color=COLORS3[k]))
                b.initial_pose = sapien.Pose(p=[0.2, (k - 1) * 0.06, half[i]])
                b.set_scene_idxs([i])
                it = b.build(name=f"reobj{k}-{i}")
                items.append(it)
                self.remove_from_state_dict_registry(it)
            cube = Actor.merge(items, name=f"reobj{k}")
            self.add_to_state_dict_registry(cube)
            self.cubes.append(cube)

        # Pockets: shallow square pockets. Floor/rim color marks DESTINATION.
        # For 3-obj: dest colors [B,G? -> B,C,A] i.e. pocket0 green? Wait COLORS3: A red,B green,C blue.
        # Goal [1,2,0] => pocket0 destination B (green), pocket1 C (blue), pocket2 A (red), buffer neutral.
        # For 2-obj: goal [1,0] => pocket0 B (green), pocket1 A (red), buffer neutral.
        n_pock = self.N_POCKETS
        goal = self._goal_map()
        dest_colors = []
        for p in range(n_pock):
            if p < len(goal):
                dest_colors.append(COLORS3[goal[p]])
            else:
                dest_colors.append(NEUTRAL)
        pockets = []
        for i in range(self.num_envs):
            b = self.scene.create_actor_builder()
            ci, wt, wh, ft = self.pocket_inner, self.pocket_wall_t, self.pocket_wall_h, self.pocket_floor_t
            for p in range(n_pock):
                px, py = self.pocket_x, self.pocket_ys[p]
                # Floor (colored by destination).
                b.add_box_collision(pose=sapien.Pose([px, py, ft / 2]), half_size=[ci / 2, ci / 2, ft / 2])
                b.add_box_visual(pose=sapien.Pose([px, py, ft / 2]), half_size=[ci / 2, ci / 2, ft / 2],
                                 material=sapien.render.RenderMaterial(base_color=dest_colors[p][:3].tolist() + [1]))
                # 4 walls (white) with collision.
                white = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
                for sx in (-1, 1):
                    b.add_box_collision(pose=sapien.Pose([px + sx * (ci / 2 + wt / 2), py, ft + wh / 2]),
                                        half_size=[wt / 2, ci / 2 + wt, wh / 2])
                    b.add_box_visual(pose=sapien.Pose([px + sx * (ci / 2 + wt / 2), py, ft + wh / 2]),
                                     half_size=[wt / 2, ci / 2 + wt, wh / 2], material=white)
                for sy in (-1, 1):
                    b.add_box_collision(pose=sapien.Pose([px, py + sy * (ci / 2 + wt / 2), ft + wh / 2]),
                                        half_size=[ci / 2, wt / 2, wh / 2])
                    b.add_box_visual(pose=sapien.Pose([px, py + sy * (ci / 2 + wt / 2), ft + wh / 2]),
                                     half_size=[ci / 2, wt / 2, wh / 2], material=white)
                # Destination rim frame (visual only).
                rim_h = 0.002
                rim_mat = sapien.render.RenderMaterial(base_color=dest_colors[p][:3].tolist() + [1])
                b.add_box_visual(pose=sapien.Pose([px, py + ci / 2 + wt / 2, ft + wh + rim_h / 2]),
                                 half_size=[ci / 2 + wt, wt / 2, rim_h / 2], material=rim_mat)
                b.add_box_visual(pose=sapien.Pose([px, py - ci / 2 - wt / 2, ft + wh + rim_h / 2]),
                                 half_size=[ci / 2 + wt, wt / 2, rim_h / 2], material=rim_mat)
                b.add_box_visual(pose=sapien.Pose([px - ci / 2 - wt / 2, py, ft + wh + rim_h / 2]),
                                 half_size=[wt / 2, ci / 2, rim_h / 2], material=rim_mat)
                b.add_box_visual(pose=sapien.Pose([px + ci / 2 + wt / 2, py, ft + wh + rim_h / 2]),
                                 half_size=[wt / 2, ci / 2, rim_h / 2], material=rim_mat)
            b.initial_pose = sapien.Pose(p=[0, 0, 0])
            b.set_scene_idxs([i])
            pk = b.build(name=f"pockets-{i}")
            pockets.append(pk)
            self.remove_from_state_dict_registry(pk)
        self.pockets = Actor.merge(pockets, name="pockets")
        self.add_to_state_dict_registry(self.pockets)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            for c in self.cubes:
                self.remove_object_from_greenscreen(c)
            self.remove_object_from_greenscreen(self.pockets)
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        self._load_camera_mount()
        self._randomize_robot_color()
        self._dwell_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._was_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dwell_steps = max(1, int(round(self.dwell_time * 10)))

    def _pocket_center(self, p):
        out = torch.zeros((self.num_envs, 3), device=self.device)
        out[:, 0] = self.pocket_x
        out[:, 1] = self.pocket_ys[p]
        out[:, 2] = self.pocket_floor_t
        return out

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)
            self.agent.robot.set_qpos(self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.02)
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)))
            init_map = self._init_map()
            # Build b-sized poses for reset envs (GPU partial-reset compat).
            for k in range(3):
                xyz = torch.zeros((b, 3), device=self.device)
                half_b = self.cube_half[env_idx] if self.cube_half.shape[0] == self.num_envs else self.cube_half[:b]
                for j in range(b):
                    if k < len(init_map):
                        p_idx = init_map.index(k) if k in init_map else (self.N_POCKETS - 1)
                    else:
                        p_idx = None
                    if p_idx is None:
                        xyz[j, 0] = 0.18; xyz[j, 1] = -0.14; xyz[j, 2] = float(half_b[j].item())
                    else:
                        xyz[j, 0] = self.pocket_x
                        xyz[j, 1] = self.pocket_ys[p_idx]
                        xyz[j, 2] = self.pocket_floor_t + float(half_b[j].item())
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                self.cubes[k].set_pose(Pose.create_from_pq(xyz, qs))
            self._dwell_count[env_idx] = 0
            self._was_complete[env_idx] = False
            try:
                self._dwell_steps = max(1, int(round(self.dwell_time * self.control_freq)))
            except Exception:
                pass

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            qpos = qpos + torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
        obs = dict(noisy_qpos=qpos)
        cs = self.agent.controller.get_state()
        if len(cs) > 0:
            obs.update(controller=cs)
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            d = dict(qvel=self.agent.robot.get_qvel(), tcp_pose=self.agent.tcp_pose.raw_pose,
                     pockets_pose=self.pockets.pose.raw_pose)
            for k in range(3):
                d[f"cube{k}_pose"] = self.cubes[k].pose.raw_pose
                d[f"tcp_to_cube{k}"] = self.cubes[k].pose.p - self.agent.tcp_pos
            if self.domain_randomization:
                gp = self.get_gripper_params()
                d.update(clean_qpos=self.agent.robot.get_qpos(), cube_dims=self.cube_dims,
                         item_friction=self.item_frictions, item_density=self.item_densities,
                         gripper_stiffness=gp["gripper_stiffness"], gripper_damping=gp["gripper_damping"])
            obs.update(d)
        return obs

    def _object_in_pocket(self, obj_idx, pocket_idx):
        p = self.cubes[obj_idx].pose.p
        t = self._pocket_center(pocket_idx)
        dxy = torch.linalg.norm(p[:, :2] - t[:, :2], dim=1)
        allow = (self.pocket_inner / 2 - self.cube_half) + self.place_xy_tol
        allow = torch.clamp(allow, min=0.002)
        xy_ok = dxy <= allow
        z_exp = t[:, 2] + self.cube_half
        dz = torch.abs(p[:, 2] - z_exp)
        z_ok = dz <= self.place_z_tol
        low = p[:, 2] <= (z_exp + self.place_z_tol)
        v = torch.linalg.norm(self.cubes[obj_idx].linear_velocity, dim=-1)
        static = v <= self.static_vel_thresh
        released = (~self.agent.is_touching(self.cubes[obj_idx])) & (~self.agent.is_grasping(self.cubes[obj_idx]))
        return xy_ok & z_ok & low & static & released, dxy, dz

    def evaluate(self):
        n = self.NUM_OBJECTS
        goal = self._goal_map()  # goal[pocket] = object
        # Correct final placements: for each pocket, assigned object seated.
        pocket_ok = []
        for p_idx, obj_idx in enumerate(goal):
            ok, _, _ = self._object_in_pocket(obj_idx, p_idx)
            pocket_ok.append(ok)
        pocket_ok = torch.stack(pocket_ok, dim=1)  # (N, n)
        # Buffer empty: no object centre inside buffer pocket footprint.
        buf = self.N_POCKETS - 1
        buf_t = self._pocket_center(buf)
        buf_occupied = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for k in range(3):
            # Only consider objects in play for buffer check? All cubes: unused
            # cube in 2-obj variant is parked far away, so it won't trigger.
            p = self.cubes[k].pose.p
            dxy = torch.linalg.norm(p[:, :2] - buf_t[:, :2], dim=1)
            inside = dxy <= (self.pocket_inner / 2 + 0.005)
            # Also require z near pocket (ignore parked cube on table far away).
            z_near = torch.abs(p[:, 2] - (buf_t[:, 2] + self.cube_half)) <= 0.02
            buf_occupied |= (inside & z_near)
        buffer_empty = ~buf_occupied
        # No stacking / rim balancing: active objects not too high.
        too_high = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for p_idx, obj_idx in enumerate(goal):
            t = self._pocket_center(p_idx)
            z_exp = t[:, 2] + self.cube_half
            too_high |= self.cubes[obj_idx].pose.p[:, 2] > (z_exp + self.place_z_tol + 0.008)
        complete = pocket_ok.all(dim=1) & buffer_empty & (~too_high)
        released = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for k in range(n):
            released &= ~self.agent.is_touching(self.cubes[k])
            released &= ~self.agent.is_grasping(self.cubes[k])
        robot_static = self.agent.is_static()
        complete_stable = complete & released & robot_static
        self._was_complete = self._was_complete | complete_stable
        self._dwell_count = torch.where(complete_stable, self._dwell_count + 1, torch.zeros_like(self._dwell_count))
        success = self._dwell_count >= self._dwell_steps
        # Per-object correctness (object k in its goal pocket).
        # Goal pocket of object k: index where goal[p]==k.
        per_correct = torch.zeros((self.num_envs, n), dtype=torch.bool, device=self.device)
        for k in range(n):
            gp = goal.index(k)
            ok, _, _ = self._object_in_pocket(k, gp)
            per_correct[:, k] = ok
        return dict(success=success, per_object_correct=per_correct,
                    num_correct=per_correct.sum(dim=1),
                    pocket_correct=pocket_ok, buffer_empty=buffer_empty,
                    buffer_occupied=buf_occupied,
                    complete=complete, complete_stable=complete_stable,
                    is_released=released)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Must not make the necessary buffer move prohibitively unattractive:
        # reward is #correct (state-based) + reaching nearest misplaced object,
        # NOT distance-to-final for the object that must move away. Moving A to
        # the buffer keeps #correct at 0 (no penalty) while reaching shaping
        # continues to guide.
        tcp = self.agent.tcp_pose.p
        n = self.NUM_OBJECTS
        goal = self._goal_map()
        per = info["per_object_correct"] if "per_object_correct" in info else torch.zeros((self.num_envs, n), dtype=torch.bool, device=self.device)
        # Reaching: distance to nearest object that is not yet correct.
        d_all = []
        for k in range(n):
            d_all.append(torch.linalg.norm(tcp - self.cubes[k].pose.p, dim=1))
        d_all = torch.stack(d_all, dim=1)
        # Mask correct objects with large distance so reaching focuses on misplaced.
        masked = torch.where(per, torch.full_like(d_all, 1e3), d_all)
        d_nearest, _ = masked.min(dim=1)
        # If all correct, reaching is 0-distance (already there).
        d_nearest = torch.where(per.all(dim=1), torch.zeros_like(d_nearest), d_nearest)
        reach = 1 - torch.tanh(5 * d_nearest)
        num_c = per.sum(dim=1).float()
        # Small grasp bonus for holding a misplaced object (guides pick).
        grasp_any = torch.zeros(self.num_envs, device=self.device)
        for k in range(n):
            g = self.agent.is_grasping(self.cubes[k]).float()
            misplaced = (~per[:, k]).float()
            grasp_any = grasp_any + g * misplaced
        grasp_any = torch.clamp(grasp_any, 0, 1)
        reward = reach * 1.0 + num_c * 3.0 + grasp_any * 1.0
        max_r = 3.0 * n + 2.0
        reward[info["success"]] = max_r
        reward = reward - 3 * self.agent.is_touching(self.table_scene.table).float()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        max_r = 3.0 * self.NUM_OBJECTS + 2.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_r


@register_env("SO101Rearrange3-v1", max_episode_steps=300)
class Rearrange3(RearrangeBase):
    NUM_OBJECTS = 3
    N_POCKETS = 4


@register_env("SO101Rearrange2-v1", max_episode_steps=200)
class Rearrange2(RearrangeBase):
    NUM_OBJECTS = 2
    N_POCKETS = 3
    def _init_map(self):
        return [0, 1]
    def _goal_map(self):
        return [1, 0]
