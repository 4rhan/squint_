"""Scripted grasp/move primitives for the SO101 arm in this repo's envs.

Deliberately does not use mplib: mplib 0.1.1 (the version mani_skill pins)
segfaults under numpy 2 and on this robot's gripper meshes, and its 6-DOF pose
IK is a poor fit for a 5-joint arm (it silently converges centimetres off
target). IK here is a small least-squares problem over the 5 arm joints on
SAPIEN's own pinocchio model, constraining only what a top-down grasp needs.

Gripper model (measured from the URDF kinematics and the jaws' collision
meshes, see _measure_gripper): in gripper_link's frame the fixed finger points
along the wrist-roll axis, the fixed jaw's contact face is perpendicular to the
moving jaw's hinge axis, and the moving jaw rotates about that hinge. Note the
finger1_tip/finger2_tip links are NOT the physical fingertips (those extend
~13mm further), so contact geometry comes from the meshes, not those links.
A grasp is specified by the object center relative to the fixed jaw face and
fingertip, with the jaw normal horizontal and aligned to an object face.
"""
import numpy as np
from scipy.optimize import least_squares
from transforms3d.axangles import mat2axangle
from transforms3d.quaternions import quat2mat


class SO101GraspSolver:
    MAX_JOINT_STEP = 0.2  # rad per 10 Hz control step (~2 rad/s); faster costs stack success
    FACE_CLEARANCE = 0.004  # gap left between fixed jaw and object face on approach
    TIP_CLEARANCE = 0.003  # physical fingertips sit this far above the object's bottom face
    MIN_JAW_Z = 0.002  # lowest point of either jaw (at the commanded angle) must stay above the table
    # rad past the contact angle. The rotating jaw pinches the cube off-axis,
    # and a harder squeeze bends the wrist away from its command (~0.26 rad of
    # wrist_flex error at 0.12, ~0.05 at 0.03), so keep it gentle.
    SQUEEZE = 0.04
    MAX_FACE_MISALIGN_DEG = 15.0  # jaw vs cube face yaw error tolerated when exact alignment is out of reach

    def __init__(self, env, vis: bool = False):
        self.env = env
        self.base_env = env.unwrapped
        self.agent = self.base_env.agent
        self.robot = self.agent.robot
        self.vis = vis

        art = self.robot._objs[0]
        self.pm = art.create_pinocchio_model()
        link_names = [link.name for link in art.get_links()]

        def link_index(suffix):
            return next(i for i, n in enumerate(link_names) if n.endswith(suffix))

        self._i_f1 = link_index("finger1_tip")
        self._i_f2 = link_index("finger2_tip")
        self._i_gl = link_index("gripper_link")
        self._i_jaw = link_index("moving_jaw_so101_v1_link")

        root = self.robot.pose.sp.to_transformation_matrix()
        self._root_R, self._root_t = root[:3, :3], root[:3, 3]

        limits = self.robot.get_qlimits()[0].cpu().numpy()
        self.arm_lo, self.arm_hi = limits[:5, 0], limits[:5, 1]
        self.CLOSED, self.OPEN = float(limits[5, 0]), float(limits[5, 1])

        self._measure_gripper()
        self.gripper_target = float(self.qpos()[5])
        self.half = None
        self._place_jaw_perp = None  # optional: keep the jaw normal perpendicular to this while placing
        self.last_step = None

    # ------------------------------------------------------------ kinematics
    def qpos(self) -> np.ndarray:
        return self.robot.get_qpos()[0].cpu().numpy().astype(np.float64)

    def _fk(self, arm_q, g, links=("f1", "gl")):
        self.pm.compute_forward_kinematics(np.concatenate([arm_q, [g]]))
        out = {}
        for key in links:
            i = {"f1": self._i_f1, "f2": self._i_f2, "gl": self._i_gl, "jaw": self._i_jaw}[key]
            pose = self.pm.get_link_pose(i)
            out[key] = (self._root_R @ pose.p + self._root_t, self._root_R @ quat2mat(pose.q))
        return out

    def _collision_vertices(self, link_name):
        """Collision-mesh vertices of a link, in that link's frame."""
        link = self.robot.links_map[link_name]._objs[0]
        pts = []
        for shape in link.collision_shapes:
            T = shape.local_pose.to_transformation_matrix()
            v = np.asarray(shape.vertices) * np.asarray(shape.scale)
            pts.append(v @ T[:3, :3].T + T[:3, 3])
        return np.concatenate(pts)

    def _measure_gripper(self):
        q = self.qpos()
        a = self._fk(q[:5], 0.2, ("f1", "gl", "jaw"))
        b = self._fk(q[:5], 0.6, ("jaw",))
        p_gl, R_gl = a["gl"]
        hinge_w, _ = mat2axangle(b["jaw"][1] @ a["jaw"][1].T)
        hinge = R_gl.T @ hinge_w
        f1_local = R_gl.T @ (a["f1"][0] - p_gl)
        finger = f1_local - (f1_local @ hinge) * hinge
        self.finger_local = finger / np.linalg.norm(finger)  # wrist -> fingertip ("down")
        normal = np.cross(hinge, self.finger_local)
        f2_local = R_gl.T @ (self._fk(q[:5], 0.6, ("f2",))["f2"][0] - p_gl)
        self.normal_local = normal if normal @ (f2_local - f1_local) > 0 else -normal  # fixed -> moving jaw

        # fixed jaw: physical fingertip and inner contact face (near the tip)
        fixed = self._collision_vertices("gripper_link")
        depth = (fixed - f1_local) @ self.finger_local
        tip_depth = depth.max()
        near_tip = fixed[depth > tip_depth - 0.015]
        self._fixed_near_tip = near_tip
        face_n = ((near_tip - f1_local) @ self.normal_local).max()
        # point on the fixed jaw's contact face at the physical fingertip
        self.fixed_tip_local = f1_local + face_n * self.normal_local + tip_depth * self.finger_local

        # moving jaw: inner face position vs gripper angle -> true jaw gap
        jaw = self._collision_vertices("moving_jaw_so101_v1_link")
        self._jaw_vertices = jaw
        grid = np.linspace(self.CLOSED, self.OPEN, 60)
        gaps = []
        for g in grid:
            fk = self._fk(q[:5], g, ("gl", "jaw"))
            w = jaw @ fk["jaw"][1].T + fk["jaw"][0]
            loc = (w - fk["gl"][0]) @ fk["gl"][1]
            d = loc @ self.finger_local
            near = loc[d > d.max() - 0.015]
            gaps.append(((near - self.fixed_tip_local) @ self.normal_local).min())
        self._gap_grid, self._gap_values = grid, np.maximum.accumulate(np.array(gaps))

    def jaw_gap(self, g) -> float:
        """Distance between the jaws' contact faces near the fingertips."""
        return float(np.interp(g, self._gap_grid, self._gap_values))

    def gripper_qpos_for_gap(self, gap: float) -> float:
        return float(np.interp(gap, self._gap_values, self._gap_grid))

    def lowest_jaw_z(self, arm_q, g) -> float:
        fk = self._fk(arm_q, g, ("gl", "jaw"))
        z_fixed = (self._fixed_near_tip @ fk["gl"][1].T + fk["gl"][0])[:, 2].min()
        z_jaw = (self._jaw_vertices @ fk["jaw"][1].T + fk["jaw"][0])[:, 2].min()
        return float(min(z_fixed, z_jaw))

    def grasp_frame(self, arm_q):
        """(object-center point, jaw normal, finger direction) in world for
        the object size set by pick(): the center sits half a width (+
        clearance) off the fixed jaw's contact face, with the physical
        fingertips TIP_CLEARANCE above the object's bottom."""
        p_gl, R_gl = self._fk(arm_q, self.gripper_target, ("gl",))["gl"]
        local = (
            self.fixed_tip_local
            + (self.half + self.FACE_CLEARANCE) * self.normal_local
            - (self.half - self.TIP_CLEARANCE) * self.finger_local
        )
        return p_gl + R_gl @ local, R_gl @ self.normal_local, R_gl @ self.finger_local

    @staticmethod
    def _numeric_jacobian(residuals, h=1e-4):
        # SAPIEN's pinocchio poses are float32, so scipy's default ~1e-8
        # finite-difference step is pure rounding noise; use 1e-4 rad.
        def jacobian(x):
            cols = []
            for i in range(len(x)):
                dx = np.zeros_like(x)
                dx[i] = h
                cols.append((residuals(x + dx) - residuals(x - dx)) / (2 * h))
            return np.stack(cols, axis=1)

        return jacobian

    def _least_squares_ik(self, residuals, target, seed, accept, score):
        """Multi-start bounded least squares over the 5 arm joints. `accept`
        filters converged solutions, the lowest `score` wins."""

        jacobian = self._numeric_jacobian(residuals)
        ref = self.qpos()[:5] if seed is None else np.asarray(seed, dtype=np.float64)
        pan = np.arctan2(target[1] - self._root_t[1], target[0] - self._root_t[0])
        seeds = [ref]
        for lift in (-0.6, 0.0, 0.6):
            for elbow in (0.0, 0.8):
                for roll in (-1.6, 0.0, 1.6):
                    seeds.append(np.array([pan, lift, elbow, 1.4, roll]))

        best, best_score = None, np.inf
        for s in seeds:
            s = np.clip(s, self.arm_lo + 1e-3, self.arm_hi - 1e-3)
            sol = least_squares(residuals, s, jac=jacobian, bounds=(self.arm_lo + 1e-4, self.arm_hi - 1e-4))
            if not accept(sol.x):
                continue
            # Depending on the branch, the open moving jaw can hang lower than
            # the fixed one and jam on the table before it reaches the object.
            if self.lowest_jaw_z(sol.x, self.gripper_target) < self.MIN_JAW_Z:
                continue
            sc = score(sol.x) + 0.15 * np.linalg.norm(sol.x - ref)
            if sc < best_score:
                best, best_score = sol.x, sc
            if sc < -0.95 + 0.15 * 0.5:
                break
        return best

    def _grasp_problem(self, target, face_axes, tol, align_weight=1.0):
        def residuals(x):
            center, n, e = self.grasp_frame(x)
            r = [(center - target) * 100.0, [n[2]]]
            if face_axes is not None:
                a, b = face_axes
                r.append([align_weight * (n @ a) * (n @ b)])
            return np.concatenate(r)

        def accept(x):
            center, n, _ = self.grasp_frame(x)
            if np.linalg.norm(center - target) > tol or abs(n[2]) > 0.05:
                return False
            if face_axes is None:
                return True
            cos = max(abs(n @ face_axes[0]), abs(n @ face_axes[1]))
            return cos > np.cos(np.deg2rad(self.MAX_FACE_MISALIGN_DEG))

        return residuals, accept

    def solve_ik(self, target, face_axes=None, seed=None, tol=2e-3):
        """Grasp IK: put grasp_frame's center at `target` with the jaw normal
        horizontal and parallel to one of `face_axes` (two horizontal unit
        vectors, e.g. a cube's face normals). That is exactly 5 constraints;
        the fingers may lean about the jaw normal (the SO101 can't point them
        straight down near its base without exceeding the wrist_flex limit),
        so among exact solutions the one with fingers pointing most downward
        (then closest to seed) wins. Returns None if nothing is within tol."""
        target = np.asarray(target, dtype=np.float64)
        score = lambda x: self.grasp_frame(x)[2][2]
        residuals, accept = self._grasp_problem(target, face_axes, tol)
        q = self._least_squares_ik(residuals, target, seed, accept, score)
        if q is None and face_axes is not None:
            # near the reach limit exact face alignment can be infeasible;
            # trade a little yaw alignment for reaching the object at all
            residuals, accept = self._grasp_problem(target, face_axes, tol, align_weight=0.05)
            q = self._least_squares_ik(residuals, target, seed, accept, score)
        return q

    def _local_solve(self, residuals, accept, seed):
        # small pull toward the previous waypoint keeps chained solutions on
        # one branch when a DOF is weakly constrained
        seed = np.clip(seed, self.arm_lo + 1e-3, self.arm_hi - 1e-3)

        def regularized(x):
            return np.concatenate([residuals(x), 0.02 * (x - seed)])

        sol = least_squares(regularized, seed, jac=self._numeric_jacobian(regularized),
                            bounds=(self.arm_lo + 1e-4, self.arm_hi - 1e-4))
        return sol.x if accept(sol.x) else None

    def vertical_path(self, q_bottom, bottom_target, height, face_axes=None, held=False, step=0.01,
                      max_tilt_deg=40.0):
        """IK waypoints for a straight vertical line from bottom_target up by
        `height`, chained from q_bottom so the whole line stays on one IK
        branch (a plain joint-space interpolation arcs sideways and can shove
        the object). Returns the list bottom -> top, possibly shorter than
        requested if the arm runs out of reach."""
        path = [q_bottom]
        n = int(round(height / step))
        for k in range(1, n + 1):
            t = np.asarray(bottom_target, dtype=np.float64) + [0, 0, height * k / n]
            if held:
                residuals, accept = self._held_problem(t, 2e-3, max_tilt_deg, face_axes)
            else:
                residuals, accept = self._grasp_problem(t, face_axes, 2e-3, align_weight=0.2)
            q = self._local_solve(residuals, accept, path[-1])
            if q is None or self.lowest_jaw_z(q, self.gripper_target) < self.MIN_JAW_Z:
                break
            path.append(q)
        return path

    def measure_held(self, actor):
        """Record the held actor's pose relative to gripper_link (from the
        actual sim state), and which of its axes currently points up."""
        fk = self._fk(self.qpos()[:5], self.gripper_target, ("gl",))
        p_gl, R_gl = fk["gl"]
        T = actor.pose.sp.to_transformation_matrix()
        self.held_p = R_gl.T @ (T[:3, 3] - p_gl)
        up_idx = int(np.argmax(np.abs(T[2, :3])))
        up = T[:3, up_idx] * np.sign(T[2, up_idx])
        self.held_up = R_gl.T @ up
        side_idx = [i for i in range(3) if i != up_idx][0]
        self.held_side = R_gl.T @ T[:3, side_idx]

    def held_frame(self, arm_q):
        """(center, up axis, a side-face normal) of the held object in world."""
        fk = self._fk(arm_q, self.gripper_target, ("gl",))
        p_gl, R_gl = fk["gl"]
        return p_gl + R_gl @ self.held_p, R_gl @ self.held_up, R_gl @ self.held_side

    def _held_problem(self, target, tol, max_tilt_deg, face_axes=None):
        # With the held cube's up axis along the wrist-roll axis, flatness says
        # almost nothing about roll, so without a yaw term roll is nearly free
        # and chained waypoints jump. face_axes pins the cube's yaw to them.
        max_xy = np.sin(np.deg2rad(max_tilt_deg))

        jaw_perp = self._place_jaw_perp

        def residuals(x):
            p, up, side = self.held_frame(x)
            r = [(p - target) * 100.0, 0.3 * up[:2]]
            if face_axes is not None:
                a, b = face_axes
                r.append([0.3 * (side @ a) * (side @ b)])
            if jaw_perp is not None:
                n = self._fk(x, self.gripper_target, ("gl",))["gl"][1] @ self.normal_local
                r.append([0.5 * (n @ jaw_perp)])
            return np.concatenate(r)

        def accept(x):
            p, up, _ = self.held_frame(x)
            return np.linalg.norm(p - target) < tol and np.linalg.norm(up[:2]) < max_xy

        return residuals, accept

    def solve_held_ik(self, target, face_axes=None, seed=None, tol=2e-3, max_tilt_deg=25.0):
        """Place IK for a held object: its center exactly at `target`, as flat
        as reachable (tilt up to max_tilt_deg), yaw soft-aligned to face_axes
        if given. Uses the measured
        in-hand pose, so it stays correct when the wrist reorients between
        grasp and place. Flatness is soft: the cube's up axis is typically the
        finger axis, and near the edge of the workspace the SO101 can't hold
        the fingers vertical."""
        target = np.asarray(target, dtype=np.float64)
        residuals, accept = self._held_problem(target, tol, max_tilt_deg, face_axes)

        def score(x):
            return 3.0 * np.linalg.norm(self.held_frame(x)[1][:2]) - 1.0

        return self._least_squares_ik(residuals, target, seed, accept, score)

    def held_tilt(self, arm_q):
        return float(np.arcsin(min(1.0, np.linalg.norm(self.held_frame(arm_q)[1][:2]))))

    def _step(self, arm_q, g):
        action = np.concatenate([arm_q, [g]]).astype(np.float32)
        self.last_step = self.env.step(action)
        if self.vis:
            self.base_env.render_human()
        return self.last_step

    def _commanded_arm(self):
        return self.base_env.agent.controller._target_qpos[0, :5].cpu().numpy().astype(np.float64)

    def move_to(self, arm_q, g=None, speed_scale=1.0, settle=1):
        """Smooth joint-space move to arm_q while commanding gripper angle g."""
        g = self.gripper_target if g is None else g
        start = self._commanded_arm()
        n = max(1, int(np.ceil(np.max(np.abs(arm_q - start)) / (self.MAX_JOINT_STEP * speed_scale))))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)
            self._step(start + s * (arm_q - start), g)
        for _ in range(settle):
            self._step(arm_q, g)
        return self.last_step

    def hold(self, steps, g=None):
        g = self.gripper_target if g is None else g
        arm_q = self._commanded_arm()
        for _ in range(steps):
            self._step(arm_q, g)
        return self.last_step

    # ------------------------------------------------------------- primitives
    @staticmethod
    def horizontal_axes(actor):
        R = actor.pose.sp.to_transformation_matrix()[:3, :3]
        axes = []
        for col in (R[:, 0], R[:, 1]):
            v = np.array([col[0], col[1], 0.0])
            axes.append(v / np.linalg.norm(v))
        return tuple(axes)

    def follow(self, path, g=None):
        for q in path:
            self._step(q, self.gripper_target if g is None else g)
        return self.last_step

    def pick(self, actor, half_size, approach_height=0.05, lift_height=0.06):
        """Top-down grasp of a box-shaped actor, then lift straight up.
        Returns True if the actor is grasped after lifting."""
        self.half = half_size
        center = actor.pose.sp.p.astype(np.float64)
        axes = self.horizontal_axes(actor)
        g_contact = self.gripper_qpos_for_gap(2 * half_size)
        g_open = self.gripper_qpos_for_gap(2 * half_size + 0.03)
        self.g_squeeze = max(self.CLOSED, g_contact - self.SQUEEZE)

        self.gripper_target = g_open  # IK checks jaw clearance at this angle
        q_grasp = self.solve_ik(center, axes)
        if q_grasp is None:
            return False
        # Near the edge of the workspace the approach line may run out of
        # reach early; accept a shorter one (>= 1 cm) rather than failing.
        up = self.vertical_path(q_grasp, center, approach_height, axes)
        if len(up) < 2:
            return False

        self.move_to(up[-1])
        self.follow(up[-2::-1])
        self.hold(1)
        self.gripper_target = self.g_squeeze
        self.hold(3)
        # Once held, the jaw orientation no longer matters much, so lift with
        # the held-object IK (cube kept near flat), which reaches further.
        self.measure_held(actor)
        lift = self.vertical_path(self._commanded_arm(), self.held_frame(self._commanded_arm())[0],
                                  lift_height, held=True)
        self.follow(lift[1:])
        return bool(self.agent.is_grasping(actor)[0])

    def place(self, actor, target_center, face_axes=None, jaw_perp=None, release_height=0.005,
              approach_height=0.04, retreat_height=0.05, release_gap=0.03):
        """Put the held actor's center at target_center (flat-ish, yaw aligned
        to face_axes if given, jaw normal perpendicular to jaw_perp if given)
        via a straight vertical descent, open the jaws by release_gap beyond
        the object width, and retreat upwards."""
        self.measure_held(actor)
        goal = np.asarray(target_center, dtype=np.float64)
        q_place, max_tilt = None, 25.0
        # far targets can't be reached with the cube held flat; the jaw
        # orientation constraint is dropped last
        for max_tilt, perp in ((25.0, jaw_perp), (40.0, jaw_perp), (40.0, None)):
            self._place_jaw_perp = None if perp is None else np.asarray(perp, dtype=np.float64)
            q_place = self.solve_held_ik(goal + [0, 0, release_height], face_axes, max_tilt_deg=max_tilt)
            if q_place is not None:
                break
        if q_place is None:
            self._place_jaw_perp = None
            return False
        # a tilted cube's lowest corner sits below its center-minus-half-size
        tilt = self.held_tilt(q_place)
        extra = self.half * (np.cos(tilt) + np.sin(tilt) - 1.0)
        if extra > 1e-3:
            q_place = self.solve_held_ik(goal + [0, 0, release_height + extra], face_axes, seed=q_place,
                                         max_tilt_deg=max_tilt)
            if q_place is None:
                return False
        down = self.vertical_path(q_place, self.held_frame(q_place)[0], approach_height, face_axes, held=True)
        if len(down) < 2:
            return False

        self.move_to(down[-1])
        self.follow(down[-2::-1])
        self.hold(1)
        self._servo_held(actor, self.held_frame(q_place)[0], face_axes)
        self.gripper_target = self.gripper_qpos_for_gap(2 * self.half + release_gap)
        self.hold(3)
        q_now = self._commanded_arm()
        fk = self._fk(q_now, self.gripper_target, ("gl",))
        q_retreat = self._retreat_ik(q_now, fk["gl"][0] + [0, 0, retreat_height])
        self.move_to(q_retreat, speed_scale=0.7)
        self._place_jaw_perp = None
        return True

    def _servo_held(self, actor, goal, face_axes, tol=0.003, iters=4):
        """Closed-loop correction using the object's true pose: the PD arm sags
        under the grasp, so shift the IK target by the observed error until the
        held object is within tol of goal."""
        target = np.asarray(goal, dtype=np.float64).copy()
        for _ in range(iters):
            err = actor.pose.sp.p - goal
            if np.linalg.norm(err) < tol:
                break
            target -= err
            residuals, accept = self._held_problem(target, 2e-3, 40.0, face_axes)
            q = self._local_solve(residuals, accept, self._commanded_arm())
            if q is None:
                break
            self._step(q, self.gripper_target)
            self.hold(1)

    def _retreat_ik(self, q_from, gl_target):
        """Move gripper_link toward gl_target keeping its orientation close to
        the current one (soft); returns the best effort if not fully reachable."""
        R0 = self._fk(q_from, self.gripper_target, ("gl",))["gl"][1]

        def residuals(x):
            p, R = self._fk(x, self.gripper_target, ("gl",))["gl"]
            return np.concatenate([(p - gl_target) * 100.0, 0.3 * (R - R0).ravel()])

        q_from = np.clip(q_from, self.arm_lo + 1e-3, self.arm_hi - 1e-3)
        sol = least_squares(residuals, q_from, jac=self._numeric_jacobian(residuals),
                            bounds=(self.arm_lo + 1e-4, self.arm_hi - 1e-4))
        return sol.x
