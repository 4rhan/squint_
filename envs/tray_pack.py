"""Task B: assigned-compartment packing (SO-101).

Three scattered cubes into three assigned compartments of a shallow printed
tray. Any placement order is valid.

Task IDs (same tray geometry + tolerances for all variants):
  - SO101TrayPack1-v1 : 1 object -> compartment 0
  - SO101TrayPack2-v1 : 2 objects -> compartments 0,1
  - SO101TrayPack3-v1 : 3 objects -> compartments 0,1,2

Tray: shallow 1x3 printed tray, floor per compartment colored to match the
assigned cube, plus matching rim frames (visible assignment markers).
No orientation requirement: cubes may land at any yaw.
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

N_COMP = 3
COLORS = [np.array([1, 0, 0, 1]), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])]  # R,G,B


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw about z from (w,x,y,z) quats. Objects are upright (lock_x/y)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class TrayPackRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    # 20-24 mm sides. Max rotated footprint 24*sqrt2=33.9mm < 42mm inner.
    cube_half_size_range: Sequence[float] = (0.010, 0.012)
    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


class TrayPackBase(DefaultCameraEnv):
    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation",
                           "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101]
    NUM_OBJECTS = 3

    def __init__(
        self, *args, robot_uids="so101", control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[TrayPackRandomizationConfig, dict] = TrayPackRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=(0.3, 0.0), spawn_box_half_size=0.2 / 2,
        # Tray geometry v2 (meters, configurable). Inner 42mm fits max rotated
        # 24mm cube (33.9mm diagonal) + gripper envelope; walls lowered to
        # 10mm so jaws can overhang during release/withdrawal (no tight insertion).
        comp_inner: float = 0.042, wall_t: float = 0.004, wall_h: float = 0.010,
        floor_t: float = 0.005,
        tray_xy=(0.33, -0.055),
        # Tight footprint-aware tolerances (do NOT count protruding objects).
        place_xy_tol: float = 0.003, place_z_tol: float = 0.004,
        static_vel_thresh: float = 2e-2,
        dwell_time: float = 1.0,
        min_tray_clearance: float = 0.075, min_obj_separation: float = 0.05,
        max_reach_radius: float = 0.38,
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
        self.domain_randomization_config = TrayPackRandomizationConfig()
        merged = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=TrayPackRandomizationConfig, data=merged, config=dacite.Config(strict=True))
        elif isinstance(domain_randomization_config, TrayPackRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config
        self.spawn_box_pos = list(spawn_box_pos)
        self.spawn_box_half_size = spawn_box_half_size
        self.comp_inner = comp_inner
        self.wall_t = wall_t
        self.wall_h = wall_h
        self.floor_t = floor_t
        self.tray_xy = torch.tensor(tray_xy, dtype=torch.float32)
        self.place_xy_tol = place_xy_tol
        self.place_z_tol = place_z_tol
        self.static_vel_thresh = static_vel_thresh
        self.dwell_time = dwell_time
        self.min_tray_clearance = min_tray_clearance
        self.min_obj_separation = min_obj_separation
        self.max_reach_radius = max_reach_radius
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

        # Cubes (always build 3; variants use first NUM_OBJECTS).
        self.cubes = []
        for k in range(N_COMP):
            items = []
            for i in range(self.num_envs):
                b = self.scene.create_actor_builder()
                mat = sapien.pysapien.physx.PhysxMaterial(static_friction=frictions[i], dynamic_friction=frictions[i], restitution=0)
                b.add_box_collision(half_size=[half[i]] * 3, material=mat, density=densities[i])
                b.add_box_visual(half_size=[half[i]] * 3, material=sapien.render.RenderMaterial(base_color=COLORS[k]))
                b.initial_pose = sapien.Pose(p=[0.2, (k - 1) * 0.06, half[i]])
                b.set_scene_idxs([i])
                it = b.build(name=f"packcube{k}-{i}")
                items.append(it)
                self.remove_from_state_dict_registry(it)
            cube = Actor.merge(items, name=f"packcube{k}")
            self.add_to_state_dict_registry(cube)
            self.cubes.append(cube)

        # Tray: floor per compartment colored to match assigned cube + white
        # outer walls + divider walls + colored rim frames on top.
        self.tray_floor_z = self.floor_t  # top of floor in tray-local z (tray base at table z=0)
        trays = []
        for i in range(self.num_envs):
            b = self.scene.create_actor_builder()
            ci, wt, wh, ft = self.comp_inner, self.wall_t, self.wall_h, self.floor_t
            total_x = N_COMP * ci + (N_COMP + 1) * wt
            total_y = ci + 2 * wt
            # Floors (one per compartment, colored).
            for k in range(N_COMP):
                cx = -total_x / 2 + wt + ci / 2 + k * (ci + wt)
                b.add_box_collision(pose=sapien.Pose([cx, 0, ft / 2]), half_size=[ci / 2, ci / 2, ft / 2])
                b.add_box_visual(pose=sapien.Pose([cx, 0, ft / 2]), half_size=[ci / 2, ci / 2, ft / 2],
                                 material=sapien.render.RenderMaterial(base_color=COLORS[k][:3].tolist() + [1]))
            # Outer walls (white).
            white = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
            for sx in (-1, 1):
                x = sx * (total_x / 2 - wt / 2)
                b.add_box_collision(pose=sapien.Pose([x, 0, ft + wh / 2]), half_size=[wt / 2, total_y / 2, wh / 2])
                b.add_box_visual(pose=sapien.Pose([x, 0, ft + wh / 2]), half_size=[wt / 2, total_y / 2, wh / 2], material=white)
            for sy in (-1, 1):
                y = sy * (total_y / 2 - wt / 2)
                b.add_box_collision(pose=sapien.Pose([0, y, ft + wh / 2]), half_size=[total_x / 2, wt / 2, wh / 2])
                b.add_box_visual(pose=sapien.Pose([0, y, ft + wh / 2]), half_size=[total_x / 2, wt / 2, wh / 2], material=white)
            # Dividers.
            for k in range(1, N_COMP):
                x = -total_x / 2 + k * (ci + wt) + wt / 2 - wt / 2 if False else -total_x / 2 + wt + k * ci + (k - 1) * wt + wt / 2
                # Simplify: divider centered between compartments.
                x = -total_x / 2 + wt + ci + (k - 1) * (ci + wt) + wt / 2
                b.add_box_collision(pose=sapien.Pose([x, 0, ft + wh / 2]), half_size=[wt / 2, ci / 2, wh / 2])
                b.add_box_visual(pose=sapien.Pose([x, 0, ft + wh / 2]), half_size=[wt / 2, ci / 2, wh / 2], material=white)
            # Colored rim frames on top of walls around each compartment (assignment markers).
            rim_h = 0.002
            for k in range(N_COMP):
                cx = -total_x / 2 + wt + ci / 2 + k * (ci + wt)
                rim_mat = sapien.render.RenderMaterial(base_color=COLORS[k][:3].tolist() + [1])
                # Two x-rims + two y-rims per compartment (visual only, no collision to avoid snagging).
                b.add_box_visual(pose=sapien.Pose([cx, ci / 2 + wt / 2, ft + wh + rim_h / 2]), half_size=[ci / 2 + wt, wt / 2, rim_h / 2], material=rim_mat)
                b.add_box_visual(pose=sapien.Pose([cx, -ci / 2 - wt / 2, ft + wh + rim_h / 2]), half_size=[ci / 2 + wt, wt / 2, rim_h / 2], material=rim_mat)
                b.add_box_visual(pose=sapien.Pose([cx - ci / 2 - wt / 2, 0, ft + wh + rim_h / 2]), half_size=[wt / 2, ci / 2, rim_h / 2], material=rim_mat)
                b.add_box_visual(pose=sapien.Pose([cx + ci / 2 + wt / 2, 0, ft + wh + rim_h / 2]), half_size=[wt / 2, ci / 2, rim_h / 2], material=rim_mat)
            b.initial_pose = sapien.Pose(p=[float(self.tray_xy[0]), float(self.tray_xy[1]), 0.0])
            b.set_scene_idxs([i])
            tr = b.build_kinematic(name=f"tray-{i}")
            trays.append(tr)
            self.remove_from_state_dict_registry(tr)
        self.tray = Actor.merge(trays, name="tray")
        self.add_to_state_dict_registry(self.tray)
        # Compartment centers in tray-local frame.
        total_x = N_COMP * self.comp_inner + (N_COMP + 1) * self.wall_t
        self.comp_local_x = torch.tensor([-total_x / 2 + self.wall_t + self.comp_inner / 2 + k * (self.comp_inner + self.wall_t)
                                          for k in range(N_COMP)], device=self.device)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            for c in self.cubes:
                self.remove_object_from_greenscreen(c)
            self.remove_object_from_greenscreen(self.tray)
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        self._load_camera_mount()
        self._randomize_robot_color()
        self._dwell_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._was_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dwell_last_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._dwell_steps = max(1, int(round(self.dwell_time * 10)))

    def _comp_world(self):
        """World xy of each compartment center + floor top z. (num_envs, 3, 3)."""
        tp = self.tray.pose.p  # (N,3)
        # Tray yaw is ~0 (we set no rotation); use full transform for correctness.
        # For simplicity assume identity yaw (tray never rotated).
        out = torch.zeros((self.num_envs, N_COMP, 3), device=self.device)
        for k in range(N_COMP):
            out[:, k, 0] = tp[:, 0] + self.comp_local_x[k]
            out[:, k, 1] = tp[:, 1]
            out[:, k, 2] = tp[:, 2] + self.floor_t
        return out

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)
            self.agent.robot.set_qpos(self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.02)
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)))
            # Tray at fixed location for reset envs (b-sized pose for GPU partial-reset compat).
            tray_xyz = torch.zeros((b, 3), device=self.device)
            tray_xyz[:, 0] = float(self.tray_xy[0]); tray_xyz[:, 1] = float(self.tray_xy[1]); tray_xyz[:, 2] = 0.0
            tray_q = torch.zeros((b, 4), device=self.device); tray_q[:, 0] = 1.0  # identity (w,x,y,z)
            self.tray.set_pose(Pose.create_from_pq(tray_xyz, tray_q))
            # Sample cube positions outside tray, non-overlapping, reachable.
            # Rectangular tray clearance (not conservative circle): cubes must be
            # outside tray outer bounds + margin (max rotated half-diag 17mm + 8mm).
            spawn_center = self.agent.robot.pose.p + torch.tensor([self.spawn_box_pos[0], self.spawn_box_pos[1], 0])
            half = self.cube_half[env_idx]
            total_x = N_COMP * self.comp_inner + (N_COMP + 1) * self.wall_t
            total_y = self.comp_inner + 2 * self.wall_t
            tray_hx = total_x / 2 + 0.025
            tray_hy = total_y / 2 + 0.025
            tray_c = self.tray_xy.to(self.device)
            base_centers = spawn_center[env_idx, :2]
            def rand_xy_for(centers):
                n = centers.shape[0]
                return (torch.rand((n, 2), device=self.device) * 2 - 1) * self.spawn_box_half_size + centers
            xys = [rand_xy_for(base_centers) for _ in range(N_COMP)]
            valid = torch.zeros(b, dtype=torch.bool, device=self.device)
            for _ in range(60):
                ok = torch.ones(b, dtype=torch.bool, device=self.device)
                for k in range(N_COMP):
                    dx = torch.abs(xys[k][:, 0] - tray_c[0])
                    dy = torch.abs(xys[k][:, 1] - tray_c[1])
                    outside = (dx >= tray_hx) | (dy >= tray_hy)
                    ok &= outside
                    ok &= torch.linalg.norm(xys[k], dim=1) <= self.max_reach_radius
                for a in range(N_COMP):
                    for c in range(a + 1, N_COMP):
                        # Conservative: max rotated half-diagonal sum (24mm cubes
                        # need 34mm; min separation 50mm covers all yaws).
                        ok &= torch.linalg.norm(xys[a] - xys[c], dim=1) >= self.min_obj_separation
                # Not already solved (cubes start outside tray by construction, so ok).
                if bool(ok.all()):
                    valid = torch.ones_like(valid)
                    break
                bad = ~ok
                if int(bad.sum().item()) == 0:
                    valid = ok
                    break
                bad_centers = base_centers[bad]
                for k in range(N_COMP):
                    xys[k][bad] = rand_xy_for(bad_centers)
            else:
                valid = ok
            if not bool(valid.all()):
                # Explicit fallback: do not accept invalid states. Use fixed
                # safe spots (outside tray, separated, reachable) for failures.
                import warnings
                warnings.warn(f"TrayPack rejection sampling exhausted for {(~valid).sum().item()}/{b} envs; using fallback layout")
                fb = torch.tensor([[0.22, 0.06], [0.22, -0.06], [0.24, 0.0]], device=self.device)
                for k in range(N_COMP):
                    xys[k][~valid] = fb[k].unsqueeze(0).expand(int((~valid).sum().item()), -1)
            for k in range(N_COMP):
                xyz = torch.stack([xys[k][:, 0], xys[k][:, 1], half], dim=1)
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                self.cubes[k].set_pose(Pose.create_from_pq(xyz, qs))
            self._dwell_count[env_idx] = 0
            self._was_complete[env_idx] = False
            self._dwell_last_step[env_idx] = -1
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
                     tray_pose=self.tray.pose.raw_pose)
            for k in range(N_COMP):
                d[f"cube{k}_pose"] = self.cubes[k].pose.raw_pose
                d[f"tcp_to_cube{k}"] = self.cubes[k].pose.p - self.agent.tcp_pos
            if self.domain_randomization:
                gp = self.get_gripper_params()
                d.update(clean_qpos=self.agent.robot.get_qpos(), cube_dims=self.cube_dims,
                         item_friction=self.item_frictions, item_density=self.item_densities,
                         gripper_stiffness=gp["gripper_stiffness"], gripper_damping=gp["gripper_damping"])
            obs.update(d)
        return obs

    def _per_object_correct(self):
        comp = self._comp_world()  # (N,3,3)
        correct = []
        for k in range(N_COMP):
            p = self.cubes[k].pose.p
            t = comp[:, k, :]
            # Footprint/orientation-aware full containment: rotated half-extent
            # per axis must fit inside inner bounds with 2mm margin. A 24mm
            # cube at 45deg spans 33.9mm; centre-distance allowances alone
            # would wrongly count protruding objects as seated.
            yaw = _yaw_from_quat(self.cubes[k].pose.q)
            ext = self.cube_half * (torch.abs(torch.cos(yaw)) + torch.abs(torch.sin(yaw)))
            margin = 0.002
            dx = torch.abs(p[:, 0] - t[:, 0])
            dy = torch.abs(p[:, 1] - t[:, 1])
            xy_ok = (dx <= (self.comp_inner / 2 - ext - margin)) & (dy <= (self.comp_inner / 2 - ext - margin))
            z_exp = t[:, 2] + self.cube_half
            dz = torch.abs(p[:, 2] - z_exp)
            z_ok = dz <= self.place_z_tol
            # Not stacked / not on rim: z must be near floor (not higher).
            low_enough = p[:, 2] <= (z_exp + self.place_z_tol)
            # Static + released.
            v = torch.linalg.norm(self.cubes[k].linear_velocity, dim=-1)
            static = v <= self.static_vel_thresh
            touching = self.agent.is_touching(self.cubes[k])
            grasped = self.agent.is_grasping(self.cubes[k])
            ok = xy_ok & z_ok & low_enough & static & (~touching) & (~grasped)
            correct.append(ok)
        return torch.stack(correct, dim=1)  # (N,3)

    def evaluate(self):
        per = self._per_object_correct()  # (N,3)
        n = self.NUM_OBJECTS
        active = per[:, :n]
        # Rim/stack guard: any active cube too high => fail (rim balancing or stacking).
        comp = self._comp_world()
        too_high = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for k in range(n):
            z_exp = comp[:, k, 2] + self.cube_half
            too_high |= self.cubes[k].pose.p[:, 2] > (z_exp + self.place_z_tol + 0.008)
        complete = active.all(dim=1) & (~too_high)
        # All released + robot static.
        released = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for k in range(n):
            released &= ~self.agent.is_touching(self.cubes[k])
            released &= ~self.agent.is_grasping(self.cubes[k])
        robot_static = self.agent.is_static()
        complete_stable = complete & released & robot_static
        # Dwell advances once per control step only.
        cur_step = self.elapsed_steps.reshape(-1).to(torch.long).to(self.device)
        new_step = cur_step != self._dwell_last_step
        self._was_complete = torch.where(new_step, self._was_complete | complete_stable, self._was_complete)
        self._dwell_count = torch.where(
            ~new_step, self._dwell_count,
            torch.where(complete_stable, self._dwell_count + 1, torch.zeros_like(self._dwell_count)))
        self._dwell_last_step = torch.where(new_step, cur_step, self._dwell_last_step)
        success = self._dwell_count >= self._dwell_steps
        return dict(success=success, per_object_correct=per,
                    num_correct=active.sum(dim=1),
                    complete=complete, complete_stable=complete_stable,
                    buffer_unused=torch.ones_like(success),  # compat key
                    is_released=released)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Any-order: sum of per-object shaping + placed-count bonus (state-based,
        # no repeatable event bonuses).
        tcp = self.agent.tcp_pose.p
        comp = self._comp_world()
        per_bonus = torch.zeros(self.num_envs, device=self.device)
        reach_terms = []
        for k in range(self.NUM_OBJECTS):
            p = self.cubes[k].pose.p
            t = comp[:, k, :]
            goal = torch.stack([t[:, 0], t[:, 1], t[:, 2] + self.cube_half], dim=1)
            d_tcp = torch.linalg.norm(tcp - p, dim=1)
            d_goal = torch.linalg.norm(goal - p, dim=1)
            reach = 1 - torch.tanh(5 * d_tcp)
            place = 1 - torch.tanh(5 * d_goal)
            grasped = self.agent.is_grasping(self.cubes[k]).float()
            # If already correct, give sustained bonus; else shaping.
            correct = info["per_object_correct"][:, k].float() if "per_object_correct" in info else torch.zeros_like(reach)
            per_bonus = per_bonus + correct * 3.0 + (1 - correct) * (reach * 0.5 + place + grasped)
            reach_terms.append(d_tcp)
        # Focus reaching on nearest incorrect object implicitly via sum; add
        # small bonus for number correct (state-based, not farmable by cycling
        # because leaving the compartment removes it).
        num_c = info["num_correct"].float() if "num_correct" in info else torch.zeros(self.num_envs, device=self.device)
        reward = per_bonus + num_c * 1.0
        max_r = 3.0 * self.NUM_OBJECTS + 1.0 * self.NUM_OBJECTS + 2.0
        reward[info["success"]] = max_r
        reward = reward - 3 * self.agent.is_touching(self.table_scene.table).float()
        reward = reward - 2 * self.agent.is_touching(self.tray).float() * 0.5
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        max_r = 3.0 * self.NUM_OBJECTS + 1.0 * self.NUM_OBJECTS + 2.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_r


@register_env("SO101TrayPack1-v1", max_episode_steps=150)
class TrayPack1(TrayPackBase):
    NUM_OBJECTS = 1


@register_env("SO101TrayPack2-v1", max_episode_steps=200)
class TrayPack2(TrayPackBase):
    NUM_OBJECTS = 2


@register_env("SO101TrayPack3-v1", max_episode_steps=300)
class TrayPack3(TrayPackBase):
    NUM_OBJECTS = 3
