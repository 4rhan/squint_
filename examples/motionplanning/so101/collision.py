"""Clearance checks between the SO101's distal links and the scene's box obstacles.

The solver's IK only knew about the table (lowest_jaw_z), so the open moving
jaw could land on a pocket/tray wall or a neighbouring cube and the gripper
stalled before touching the object. Every obstacle in these envs is a box
collision shape and the robot links are convex meshes, so a signed-distance
check of points sampled on the robot's collision surfaces (<= SPACING apart)
against oriented boxes is exact to about SPACING/2 and costs ~1 ms per config.
"""
import numpy as np
from transforms3d.quaternions import quat2mat

SPACING = 0.002
# link -> sample spacing: fine where contact geometry matters, coarse further up the arm
ROBOT_LINKS = {"lower_arm_link": 0.008, "wrist_link": 0.004, "gripper_link": SPACING,
               "moving_jaw_so101_v1_link": SPACING}
REPORT_RANGE = 0.02  # obstacles further than this from a link's bounding box are skipped
SKIP_ACTORS = ("table-workspace", "ground", "camera_mount", "wrist_camera_mount")


def _surface_points(vertices, triangles, spacing):
    pts = [vertices]
    for tri in vertices[triangles]:
        a, b, c = tri
        n = int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing))
        if n < 2:
            continue
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        u, v = i[keep] / n, j[keep] / n
        pts.append(a + u[:, None] * (b - a) + v[:, None] * (c - a))
    return np.unique(np.round(np.concatenate(pts), 5), axis=0)


def _pose_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = quat2mat(pose.q)
    T[:3, 3] = pose.p
    return T


class Box:
    __slots__ = ("name", "R", "c", "h")

    def __init__(self, name, T, half):
        self.name, self.R, self.c, self.h = name, T[:3, :3], T[:3, 3], np.asarray(half, dtype=np.float64)

    def signed_distance(self, pts):
        local = (pts - self.c) @ self.R
        q = np.abs(local) - self.h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside

    def surface_points(self, spacing=SPACING):
        """Grid points on all six faces (a corners-only test misses a wall
        passing between the corners of a held cube)."""
        axes = [np.linspace(-h, h, max(2, int(np.ceil(2 * h / spacing)) + 1)) for h in self.h]
        pts = []
        for k in range(3):
            i, j = [a for a in range(3) if a != k]
            u, v = np.meshgrid(axes[i], axes[j], indexing="ij")
            for sgn in (-1, 1):
                p = np.zeros((u.size, 3))
                p[:, i], p[:, j], p[:, k] = u.ravel(), v.ravel(), sgn * self.h[k]
                pts.append(p)
        return np.concatenate(pts) @ self.R.T + self.c


class CollisionModel:
    def __init__(self, solver):
        self.solver = solver
        art = solver.robot._objs[0]
        links = art.get_links()
        self.link_pts = {}  # pinocchio link index -> (short name, surface points in link frame)
        for idx, link in enumerate(links):
            short = next((k for k in ROBOT_LINKS if link.name.endswith(k)), None)
            if short is None:
                continue
            pts = []
            for shape in link.collision_shapes:
                T = shape.local_pose.to_transformation_matrix()
                v = np.asarray(shape.vertices, dtype=np.float64) * np.asarray(shape.scale)
                v = v @ T[:3, :3].T + T[:3, 3]
                pts.append(_surface_points(v, np.asarray(shape.triangles), ROBOT_LINKS[short]))
            self.link_pts[idx] = (short, np.concatenate(pts))
        self.obstacles = []

    def refresh(self, exclude=()):
        """Re-read obstacle boxes from the scene (actors move between primitives).
        `exclude`: actors that are allowed contact (the object being grasped)."""
        skip = {a._objs[0].name for a in exclude}
        self.obstacles = []
        for actor in self.solver.base_env.scene.sub_scenes[0].entities:
            name = actor.name
            if name in skip or any(name.startswith(s) for s in SKIP_ACTORS):
                continue
            body = None
            for comp in actor.components:
                if hasattr(comp, "collision_shapes") and type(comp).__name__.startswith("PhysxRigid"):
                    body = comp
            if body is None:
                continue
            T_actor = _pose_matrix(actor.pose)
            for k, shape in enumerate(body.collision_shapes):
                if not hasattr(shape, "half_size"):
                    continue
                T = T_actor @ shape.local_pose.to_transformation_matrix()
                self.obstacles.append(Box(f"{name}[{k}]", T, shape.half_size))

    def robot_points(self, arm_q, g):
        """{link short name: world points}."""
        s = self.solver
        s.pm.compute_forward_kinematics(np.concatenate([arm_q, [g]]))
        out = {}
        for idx, (short, pts) in self.link_pts.items():
            pose = s.pm.get_link_pose(idx)
            R = s._root_R @ quat2mat(pose.q)
            t = s._root_R @ pose.p + s._root_t
            out[short] = pts @ R.T + t
        return out

    def _against_obstacles(self, pts, label, best):
        """Fold min signed distance of pts vs table plane + obstacles into best=(d, who)."""
        z = float(pts[:, 2].min())
        if z < best[0]:
            best = (z, f"{label}/table")
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        for box in self.obstacles:
            r = float(np.linalg.norm(box.h))
            if np.any(box.c + r < lo - REPORT_RANGE) or np.any(box.c - r > hi + REPORT_RANGE):
                continue
            d = float(box.signed_distance(pts).min())
            if d < best[0]:
                best = (d, f"{label}/{box.name}")
        return best

    def clearance(self, arm_q, g, held_box=None, links=None):
        """(min signed distance [m], "link/obstacle") over the robot links (or
        just `links`), the table plane and, if given, the held object's box
        (center, rotation, half size in gripper_link's frame). Only obstacles
        within REPORT_RANGE are measured, so a large value means "at least"."""
        best = (np.inf, "")
        for short, pts in self.robot_points(arm_q, g).items():
            if links is None or short in links:
                best = self._against_obstacles(pts, short, best)
        if held_box is not None:
            p_gl, R_gl = self.solver._fk(arm_q, g, ("gl",))["gl"]
            c_local, R_local, half = held_box
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R_gl @ R_local, p_gl + R_gl @ c_local
            best = self._against_obstacles(Box("held", T, half).surface_points(), "held", best)
        return best
