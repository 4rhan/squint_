"""Task A: size-ordered tower (SO-101).

Three differently sized cubes stacked large -> medium -> small at a marked
tower location, plus a two-cube control using a subset of the same geometry.

Task IDs:
  - SO101Tower3Cube-v1 : large on table at tower target, medium on large, small on medium
  - SO101Tower2Cube-v1 : large on table at tower target, medium on large (control)

Design notes (see docs/TASK_SPEC.md for full spec):
  - Strict size ordering is guaranteed by non-overlapping half-size ranges.
  - Tower target is fixed (configurable) and marked by a thin visual plate.
  - Success requires support/contact + geometric checks (not height/centre
    distance alone), release, stability, and a configurable dwell (default 1s).
  - Policy observation in deployment mode (obs_mode without `state`, e.g.
    "rgb+segmentation") is wrist RGB + robot qpos/controller state only.
    Object poses / success / grasp flags live in `info` (evaluate) and in
    `extra` only when obs_mode includes `state` (for asymmetric critics).
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


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class TowerRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Non-overlapping ranges guarantee strict ordering under randomization.
    # half sizes in meters (side = 2*half).
    large_half_size_range: Sequence[float] = (0.016, 0.018)   # 32-36 mm
    medium_half_size_range: Sequence[float] = (0.0125, 0.0145)  # 25-29 mm
    small_half_size_range: Sequence[float] = (0.0095, 0.0115)   # 19-23 mm
    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


class TowerBase(DefaultCameraEnv):
    """Shared implementation for 2- and 3-cube size-ordered towers."""

    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation",
                           "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101]

    NUM_CUBES = 3  # overridden by 2-cube control

    def __init__(
        self,
        *args,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[TowerRandomizationConfig, dict] = TowerRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=(0.3, 0.0),
        spawn_box_half_size=0.2 / 2,
        # Tower target (table frame, x/y relative to robot base). Fixed and
        # visibly marked; configurable but identical across methods in a comparison.
        tower_xy=(0.30, 0.08),
        tower_tol_xy: float = 0.012,
        tower_tol_z: float = 0.004,
        # Support tolerances (geometric part of the support check).
        support_xy_tol: float = 0.006,
        support_z_tol: float = 0.004,
        contact_force_thresh: float = 0.05,
        static_vel_thresh: float = 2e-2,
        # Dwell: complete condition must persist this long (sim seconds).
        dwell_time: float = 1.0,
        # Rejection sampling.
        min_tower_clearance: float = 0.07,  # large cube initial dist from tower
        min_obj_separation: float = 0.055,  # center distance for initial cubes
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

        self.domain_randomization_config = TowerRandomizationConfig()
        merged = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=TowerRandomizationConfig, data=merged,
                config=dacite.Config(strict=True))
        elif isinstance(domain_randomization_config, TowerRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = list(spawn_box_pos)
        self.spawn_box_half_size = spawn_box_half_size
        self.tower_xy = torch.tensor(tower_xy, dtype=torch.float32)
        self.tower_tol_xy = tower_tol_xy
        self.tower_tol_z = tower_tol_z
        self.support_xy_tol = support_xy_tol
        self.support_z_tol = support_z_tol
        self.contact_force_thresh = contact_force_thresh
        self.static_vel_thresh = static_vel_thresh
        self.dwell_time = dwell_time
        self.min_tower_clearance = min_tower_clearance
        self.min_obj_separation = min_obj_separation
        self.max_reach_radius = max_reach_radius

        super().__init__(
            *args, robot_uids=robot_uids, control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs)

    # ------------------------------------------------------------------ scene
    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random" else False)

    def _build_cube(self, name, half_sizes, color, frictions, densities, spawn_x):
        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            mat = sapien.pysapien.physx.PhysxMaterial(
                static_friction=frictions[i], dynamic_friction=frictions[i], restitution=0)
            builder.add_box_collision(half_size=[half_sizes[i]] * 3, material=mat, density=densities[i])
            builder.add_box_visual(half_size=[half_sizes[i]] * 3,
                                   material=sapien.render.RenderMaterial(base_color=color))
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
            frictions = self._batched_episode_rng.uniform(low=cfg.item_friction_range[0], high=cfg.item_friction_range[1])
            densities = self._batched_episode_rng.uniform(low=cfg.item_density_range[0], high=cfg.item_density_range[1])
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        def sample_half(lo, hi, default):
            hs = np.ones(self.num_envs) * default
            if self.domain_randomization:
                hs = self._batched_episode_rng.uniform(low=lo, high=hi)
            return hs

        # Large (base, blue), medium (red), small (green). Distinct colors.
        self.large_half = common.to_tensor(
            sample_half(cfg.large_half_size_range[0], cfg.large_half_size_range[1],
                        (cfg.large_half_size_range[0] + cfg.large_half_size_range[1]) / 2), device=self.device)
        self.medium_half = common.to_tensor(
            sample_half(cfg.medium_half_size_range[0], cfg.medium_half_size_range[1],
                        (cfg.medium_half_size_range[0] + cfg.medium_half_size_range[1]) / 2), device=self.device)
        self.small_half = common.to_tensor(
            sample_half(cfg.small_half_size_range[0], cfg.small_half_size_range[1],
                        (cfg.small_half_size_range[0] + cfg.small_half_size_range[1]) / 2), device=self.device)
        self.large_dims = torch.stack([self.large_half] * 3, dim=-1)
        self.medium_dims = torch.stack([self.medium_half] * 3, dim=-1)
        self.small_dims = torch.stack([self.small_half] * 3, dim=-1)

        self.cubeL = self._build_cube("cubeL", self.large_half.cpu().numpy(),
                                      np.array([0, 0, 1, 1]), frictions, densities, spawn_x=-0.2)
        self.cubeM = self._build_cube("cubeM", self.medium_half.cpu().numpy(),
                                      np.array([1, 0, 0, 1]), frictions, densities, spawn_x=0.0)
        if self.NUM_CUBES == 3:
            self.cubeS = self._build_cube("cubeS", self.small_half.cpu().numpy(),
                                          np.array([0, 1, 0, 1]), frictions, densities, spawn_x=0.2)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.cubeL)
            self.remove_object_from_greenscreen(self.cubeM)
            if self.NUM_CUBES == 3:
                self.remove_object_from_greenscreen(self.cubeS)
            # Tower marker plate: keep visible (it is the goal marker).
            # Built below; registered here after creation.

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        self._load_camera_mount()
        self._randomize_robot_color()

        # Tower marker: thin visual-only plate on the table (printable marker).
        # 70x70x2mm, yellow. No collision so it cannot support objects.
        marker = self.scene.create_actor_builder()
        marker.add_box_visual(half_size=[0.035, 0.035, 0.001],
                              material=sapien.render.RenderMaterial(base_color=[1, 1, 0, 1]))
        marker.initial_pose = sapien.Pose(p=[float(self.tower_xy[0]), float(self.tower_xy[1]), 0.001])
        self.tower_marker = marker.build_kinematic(name="tower_marker")
        # visual only: hide from contact, keep in greenscreen foreground
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.tower_marker)

        # Per-env bookkeeping (dwell + latch for stack-loss diagnostic).
        self._dwell_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._was_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dwell_last_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._dwell_steps = max(1, int(round(self.dwell_time * self._sim_config.control_freq
                                            if hasattr(self, "_sim_config") and self._sim_config is not None else 10)))

    @property
    def _default_sim_config(self):
        from mani_skill.utils.structs.types import SimConfig
        return SimConfig(sim_freq=100, control_freq=10)

    # ------------------------------------------------------------------ reset
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.02)
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)))
            # Tower marker at fixed table location (table top z=0 in world after pose? table surface is z=0).
            # TableSceneBuilder places table top at z=0.
            marker_p = torch.zeros((b, 3))
            marker_p[:, 0] = float(self.tower_xy[0])
            marker_p[:, 1] = float(self.tower_xy[1])
            marker_p[:, 2] = 0.001
            self.tower_marker.set_pose(Pose.create_from_pq(marker_p))

            spawn_center = self.agent.robot.pose.p + torch.tensor([self.spawn_box_pos[0], self.spawn_box_pos[1], 0])
            tower_xy_t = self.tower_xy.to(self.device)
            # Rejection-sample initial cube xy: non-overlapping, reachable,
            # large away from tower, not accidentally solved.
            placed = self._sample_valid_layout(env_idx, spawn_center, tower_xy_t)
            # placed: dict cube -> xyz
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cubeL.set_pose(Pose.create_from_pq(placed["L"], qs))
            qs2 = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cubeM.set_pose(Pose.create_from_pq(placed["M"], qs2))
            if self.NUM_CUBES == 3:
                qs3 = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                self.cubeS.set_pose(Pose.create_from_pq(placed["S"], qs3))

            # Reset dwell/latch for these envs.
            self._dwell_count[env_idx] = 0
            self._was_complete[env_idx] = False
            self._dwell_last_step[env_idx] = -1
            # Recompute dwell steps in case control freq changed (cheap).
            try:
                self._dwell_steps = max(1, int(round(self.dwell_time * self.control_freq)))
            except Exception:
                pass

    @property
    def control_freq(self):
        try:
            return self._sim_config.control_freq
        except Exception:
            return 10

    def _sample_valid_layout(self, env_idx, spawn_center, tower_xy_t):
        """Rejection-sample xy offsets for cubes. Returns dict of xyz poses."""
        b = len(env_idx)
        dev = self.device
        half_L = self.large_half[env_idx]
        half_M = self.medium_half[env_idx]
        half_S = self.small_half[env_idx] if self.NUM_CUBES == 3 else None
        base_centers = spawn_center[env_idx, :2]  # (b,2)
        # Candidate sampling in spawn box.
        max_tries = 60
        def rand_xy_for(centers):
            n = centers.shape[0]
            return (torch.rand((n, 2), device=dev) * 2 - 1) * self.spawn_box_half_size + centers
        xyL = rand_xy_for(base_centers)
        xyM = rand_xy_for(base_centers)
        xyS = rand_xy_for(base_centers) if self.NUM_CUBES == 3 else None
        valid = torch.zeros(b, dtype=torch.bool, device=dev)
        for _ in range(max_tries):
            d_LM = torch.linalg.norm(xyL - xyM, dim=1)
            ok = d_LM >= self.min_obj_separation
            ok &= torch.linalg.norm(xyL - tower_xy_t.unsqueeze(0), dim=1) >= self.min_tower_clearance
            # Reachable: radius from robot base < max.
            ok &= torch.linalg.norm(xyL, dim=1) <= self.max_reach_radius
            ok &= torch.linalg.norm(xyM, dim=1) <= self.max_reach_radius
            if self.NUM_CUBES == 3:
                d_LS = torch.linalg.norm(xyL - xyS, dim=1)
                d_MS = torch.linalg.norm(xyM - xyS, dim=1)
                ok &= d_LS >= self.min_obj_separation
                ok &= d_MS >= self.min_obj_separation
                ok &= torch.linalg.norm(xyS, dim=1) <= self.max_reach_radius
            if bool(ok.all()):
                valid = torch.ones_like(valid)
                break
            bad = ~ok
            nb = int(bad.sum().item())
            if nb == 0:
                valid = ok
                break
            bad_centers = base_centers[bad]
            xyL[bad] = rand_xy_for(bad_centers)
            xyM[bad] = rand_xy_for(bad_centers)
            if self.NUM_CUBES == 3:
                xyS[bad] = rand_xy_for(bad_centers)
        else:
            valid = ok
        if not bool(valid.all()):
            import warnings
            warnings.warn(f"Tower rejection sampling exhausted for {(~valid).sum().item()}/{b} envs; using fallback layout")
            fb_L = torch.tensor([0.22, -0.06], device=dev)
            fb_M = torch.tensor([0.22, 0.06], device=dev)
            fb_S = torch.tensor([0.26, 0.0], device=dev)
            nfb = int((~valid).sum().item())
            xyL[~valid] = fb_L.unsqueeze(0).expand(nfb, -1)
            xyM[~valid] = fb_M.unsqueeze(0).expand(nfb, -1)
            if self.NUM_CUBES == 3:
                xyS[~valid] = fb_S.unsqueeze(0).expand(nfb, -1)
        # Build xyz (z = half size resting on table z=0).
        out = {}
        zL = half_L
        zM = half_M
        out["L"] = torch.stack([xyL[:, 0], xyL[:, 1], zL], dim=1)
        out["M"] = torch.stack([xyM[:, 0], xyM[:, 1], zM], dim=1)
        if self.NUM_CUBES == 3:
            out["S"] = torch.stack([xyS[:, 0], xyS[:, 1], half_S], dim=1)
        return out

    # ------------------------------------------------------------------ obs
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
        # Privileged task state is ONLY exposed when obs_mode requests `state`.
        # Deployment policy mode (e.g. "rgb+segmentation") gets no extra state.
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                tcp_pose=self.agent.tcp_pose.raw_pose,
                cubeL_pose=self.cubeL.pose.raw_pose,
                cubeM_pose=self.cubeM.pose.raw_pose,
                tcp_to_L=self.cubeL.pose.p - self.agent.tcp_pos,
                tcp_to_M=self.cubeM.pose.p - self.agent.tcp_pos,
            )
            if self.NUM_CUBES == 3:
                obs.update(cubeS_pose=self.cubeS.pose.raw_pose,
                           tcp_to_S=self.cubeS.pose.p - self.agent.tcp_pos)
            if self.domain_randomization:
                gp = self.get_gripper_params()
                obs.update(clean_qpos=self.agent.robot.get_qpos(),
                           large_dims=self.large_dims, medium_dims=self.medium_dims,
                           item_friction=self.item_frictions, item_density=self.item_densities,
                           gripper_stiffness=gp["gripper_stiffness"], gripper_damping=gp["gripper_damping"])
                if self.NUM_CUBES == 3:
                    obs.update(small_dims=self.small_dims)
        return obs

    # ------------------------------------------------------------------ checks
    def _support_check(self, p_top, q_top, half_top, p_bot, q_bot, half_bot):
        """Orientation-aware support for unequal cubes.

        Requires: (a) all 4 bottom corners of the top cube fall inside the
        bottom cube's top face (+2mm margin), accounting for both yaws;
        (b) vertical contact; (c) contact/static. Corner checks replace
        centre-distance so overhanging placements fail even when centres
        are close.
        """
        yaw_top = _yaw_from_quat(q_top)
        yaw_bot = _yaw_from_quat(q_bot)
        ct, st = torch.cos(yaw_top), torch.sin(yaw_top)
        # top corners in world (N,4,2)
        s = half_top.unsqueeze(1)
        corners = torch.stack([
            torch.stack([s[:, 0] * ct - s[:, 0] * st, s[:, 0] * st + s[:, 0] * ct], dim=1),
            torch.stack([s[:, 0] * ct + s[:, 0] * st, s[:, 0] * st - s[:, 0] * ct], dim=1),
            torch.stack([-s[:, 0] * ct - s[:, 0] * st, -s[:, 0] * st + s[:, 0] * ct], dim=1),
            torch.stack([-s[:, 0] * ct + s[:, 0] * st, -s[:, 0] * st - s[:, 0] * ct], dim=1),
        ], dim=1) + (p_top[:, :2].unsqueeze(1))
        # into bottom frame
        rel = corners - p_bot[:, :2].unsqueeze(1)
        cb = torch.cos(-yaw_bot).unsqueeze(1)
        sb = torch.sin(-yaw_bot).unsqueeze(1)
        lx = rel[:, :, 0] * cb - rel[:, :, 1] * sb
        ly = rel[:, :, 0] * sb + rel[:, :, 1] * cb
        margin = 0.002
        allow = half_bot.unsqueeze(1) + margin
        xy_ok = ((lx.abs() <= allow) & (ly.abs() <= allow)).all(dim=1)
        expected_dz = half_top + half_bot
        z_gap = torch.abs((p_top[:, 2] - p_bot[:, 2]) - expected_dz)
        z_ok = z_gap <= self.support_z_tol
        xy_dist = torch.linalg.norm((p_top - p_bot)[:, :2], dim=1)
        return xy_dist, z_gap, xy_ok & z_ok, z_ok

    def _contact_force(self, a, b):
        try:
            f = self.scene.get_pairwise_contact_forces(a, b)
            return torch.linalg.norm(f, dim=1)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _instantaneous(self):
        pL = self.cubeL.pose.p
        pM = self.cubeM.pose.p
        tower_xy = torch.zeros_like(pL[:, :2])
        tower_xy[:, 0] = float(self.tower_xy[0])
        tower_xy[:, 1] = float(self.tower_xy[1])
        # Base at tower: xy within tol + resting on table.
        base_xy = torch.linalg.norm(pL[:, :2] - tower_xy, dim=1)
        base_z_gap = pL[:, 2] - self.large_half
        base_placed = (base_xy <= self.tower_tol_xy) & (base_z_gap.abs() <= self.tower_tol_z)
        # Medium on large (orientation-aware corners).
        _, _, geom_M, _ = self._support_check(pM, self.cubeM.pose.q, self.medium_half,
                                              pL, self.cubeL.pose.q, self.large_half)
        f_ML = self._contact_force(self.cubeM, self.cubeL)
        contact_ML = f_ML >= self.contact_force_thresh
        # Require geometry; contact force is supplementary (stable sim contact
        # may read ~0 for a resting stack). Use geometry AND (contact OR static).
        vM = torch.linalg.norm(self.cubeM.linear_velocity, dim=-1)
        vL = torch.linalg.norm(self.cubeL.linear_velocity, dim=-1)
        static_ML = (vM <= self.static_vel_thresh) & (vL <= self.static_vel_thresh)
        medium_supported = geom_M & (contact_ML | static_ML)
        if self.NUM_CUBES == 3:
            pS = self.cubeS.pose.p
            _, _, geom_S, _ = self._support_check(pS, self.cubeS.pose.q, self.small_half,
                                                  pM, self.cubeM.pose.q, self.medium_half)
            f_SM = self._contact_force(self.cubeS, self.cubeM)
            contact_SM = f_SM >= self.contact_force_thresh
            vS = torch.linalg.norm(self.cubeS.linear_velocity, dim=-1)
            static_SM = (vS <= self.static_vel_thresh) & static_ML
            small_supported = geom_S & (contact_SM | static_SM)
            complete = base_placed & medium_supported & small_supported
        else:
            small_supported = torch.ones_like(base_placed, dtype=torch.bool)
            complete = base_placed & medium_supported
        # Release / stability.
        robot_touch_L = self.agent.is_touching(self.cubeL)
        robot_touch_M = self.agent.is_touching(self.cubeM)
        grasp_M = self.agent.is_grasping(self.cubeM)
        if self.NUM_CUBES == 3:
            robot_touch_S = self.agent.is_touching(self.cubeS)
            grasp_S = self.agent.is_grasping(self.cubeS)
            grasp_L = self.agent.is_grasping(self.cubeL)
            released = (~robot_touch_L) & (~robot_touch_M) & (~robot_touch_S) & (~grasp_M) & (~grasp_S) & (~grasp_L)
            vS = torch.linalg.norm(self.cubeS.linear_velocity, dim=-1)
            all_static = static_ML & (vS <= self.static_vel_thresh)
        else:
            grasp_L = self.agent.is_grasping(self.cubeL)
            released = (~robot_touch_L) & (~robot_touch_M) & (~grasp_M) & (~grasp_L)
            all_static = static_ML
        robot_static = self.agent.is_static()
        complete_stable = complete & released & all_static & robot_static
        return dict(base_placed=base_placed, medium_supported=medium_supported,
                    small_supported=small_supported, complete=complete,
                    complete_stable=complete_stable, released=released,
                    all_static=all_static, robot_static=robot_static,
                    base_xy=base_xy, base_z_gap=base_z_gap,
                    robot_touch_L=robot_touch_L, robot_touch_M=robot_touch_M)

    def evaluate(self):
        inst = self._instantaneous()
        # Dwell advances once per control step only (repeated evaluate() w/o step is free).
        cur_step = self.elapsed_steps.reshape(-1).to(torch.long).to(self.device)
        new_step = cur_step != self._dwell_last_step
        self._was_complete = torch.where(new_step, self._was_complete | inst["complete_stable"], self._was_complete)
        self._dwell_count = torch.where(
            ~new_step, self._dwell_count,
            torch.where(inst["complete_stable"], self._dwell_count + 1, torch.zeros_like(self._dwell_count)))
        self._dwell_last_step = torch.where(new_step, cur_step, self._dwell_last_step)
        success = self._dwell_count >= self._dwell_steps
        stack_lost = self._was_complete & (~inst["complete"])
        out = dict(success=success,
                   base_placed=inst["base_placed"],
                   medium_supported=inst["medium_supported"],
                   small_supported=inst["small_supported"],
                   tower_complete=inst["complete"],
                   tower_complete_stable=inst["complete_stable"],
                   stack_lost=stack_lost,
                   dwell_count=self._dwell_count.clone(),
                   base_xy_dist=inst["base_xy"],
                   is_released=inst["released"])
        # Convenience aliases matching training-script stage logging.
        out["is_base_placed"] = inst["base_placed"]
        out["is_medium_on_base"] = inst["medium_supported"]
        return out

    # ------------------------------------------------------------------ reward
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Staged reaching/placing reward, monotonic-ish, state-based (no
        # repeatable event bonuses, so cycling cannot farm reward).
        tcp = self.agent.tcp_pose.p
        pL, pM = self.cubeL.pose.p, self.cubeM.pose.p
        tower = torch.zeros_like(pL)
        tower[:, 0] = float(self.tower_xy[0]); tower[:, 1] = float(self.tower_xy[1])
        tower[:, 2] = self.large_half  # resting height of large centre
        # Stage weights increase with progress; each stage reward in [0,2].
        def dist_r(d, s=5.0):
            return 1 - torch.tanh(s * d)
        # Stage 0: reach large.
        r_reach_L = dist_r(torch.linalg.norm(tcp - pL, dim=1))
        # Stage 1: large -> tower (xy + z).
        d_LT = torch.linalg.norm(torch.cat([pL[:, :2] - tower[:, :2],
                                            (pL[:, 2] - tower[:, 2]).unsqueeze(1)], dim=1), dim=1)
        r_place_L = dist_r(d_LT)
        grasp_L = self.agent.is_grasping(self.cubeL).float()
        # Stage 2: medium onto large.
        goal_M = pL + torch.stack([torch.zeros_like(pL[:, 0]), torch.zeros_like(pL[:, 0]),
                                   self.large_half + self.medium_half], dim=1)
        d_M = torch.linalg.norm(goal_M - pM, dim=1)
        r_place_M = dist_r(d_M)
        grasp_M = self.agent.is_grasping(self.cubeM).float()
        base_ok = info["base_placed"].float() if "base_placed" in info else torch.zeros_like(r_reach_L)
        med_ok = info["medium_supported"].float() if "medium_supported" in info else torch.zeros_like(r_reach_L)
        reward = r_reach_L + 2 * r_place_L + 2 * grasp_L
        reward = reward + base_ok * (2 * r_place_M + 2 * grasp_M + 2)
        if self.NUM_CUBES == 3:
            pS = self.cubeS.pose.p
            goal_S = pM + torch.stack([torch.zeros_like(pM[:, 0]), torch.zeros_like(pM[:, 0]),
                                       self.medium_half + self.small_half], dim=1)
            d_S = torch.linalg.norm(goal_S - pS, dim=1)
            r_place_S = dist_r(d_S)
            grasp_S = self.agent.is_grasping(self.cubeS).float()
            reward = reward + base_ok * med_ok * (2 * r_place_S + 2 * grasp_S + 2)
            max_r = 15.0
        else:
            max_r = 10.0
        # Success override (single terminal level, not repeatable farming:
        # success persists via dwell; reward is state-based).
        reward[info["success"]] = max_r
        # Penalties.
        reward = reward - 3 * self.agent.is_touching(self.table_scene.table).float()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        max_r = 15.0 if self.NUM_CUBES == 3 else 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_r


@register_env("SO101Tower3Cube-v1", max_episode_steps=300)
class Tower3Cube(TowerBase):
    NUM_CUBES = 3
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_env("SO101Tower2Cube-v1", max_episode_steps=200)
class Tower2Cube(TowerBase):
    NUM_CUBES = 2
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
