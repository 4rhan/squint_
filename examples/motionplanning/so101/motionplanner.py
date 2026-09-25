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
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from transforms3d.axangles import mat2axangle
from transforms3d.quaternions import quat2mat

from examples.motionplanning.so101.collision import CollisionModel


class SO101GraspSolver:
    MAX_JOINT_STEP = 0.2  # rad per 10 Hz control step (~2 rad/s); faster costs stack success
    MAX_JOINT_ACCEL = 0.06  # rad per step^2 for the smooth trajectory profile
    FACE_CLEARANCE = 0.004  # gap left between fixed jaw and object face on approach
    TIP_CLEARANCE = 0.003  # physical fingertips sit this far above the object's bottom face
    MIN_JAW_Z = 0.002  # lowest point of either jaw (at the commanded angle) must stay above the table
    # rad past the contact angle. The rotating jaw pinches the cube off-axis,
    # and a harder squeeze bends the wrist away from its command (~0.26 rad of
    # wrist_flex error at 0.12, ~0.05 at 0.03), so keep it gentle.
    SQUEEZE = 0.04
    # Trade-offs that differ between position control and delta control (set in __init__): with
    # delta control every joint moves <= 0.1 rad/step, so joint travel directly costs episode steps.
    CONTINUITY_W = 0.15  # IK: penalty per rad of the largest joint move from the current config
    TILT_W = 3.0  # place IK: preference for holding the cube flat
    GRIP_SETTLE = 3  # steps to hold still after the gripper reaches its target
    SERVO_ITERS = 4  # closed-loop place corrections
    APPROACH_H, LIFT_H, PLACE_APPROACH_H, RETREAT_H = 0.05, 0.06, 0.04, 0.05
    MAX_FACE_MISALIGN_DEG = 15.0  # jaw vs cube face yaw error tolerated when exact alignment is out of reach
    JAW_CLEARANCE = 0.0005  # required moving-jaw clearance to obstacles while opening/closing at the object
    OPEN_EXTRA_FALLBACK = (0.006, 0.004, 0.003)  # narrower jaw openings tried when the wide one hits a wall
    MISALIGNED_GRASP_DEG = 35.0  # last-resort jaw/face yaw error at the reach limit
    PICK_RETRIES = 2
    # Multistart IK accepts at 2 mm (residuals are in cm), so scipy's 1e-8 defaults only burn
    # iterations on seeds that are converging to a far-off local minimum anyway.
    LSQ_OPTS = dict(ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=40)  # re-grasp attempts after a detected miss

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
        self.collision = CollisionModel(self)
        self.n_retries = 0

        # In delta control (Squint's default, pd_joint_target_delta_pos) each action moves the joint
        # targets by at most the controller limits (0.1 rad arm, 0.2 rad gripper per step), so the
        # trajectory must respect them or the recorded actions get clipped and the demo drifts.
        ctrl_cfg = self.base_env.agent.controller.config
        self.delta_control = bool(getattr(ctrl_cfg, "use_delta", False))
        if self.delta_control:
            self._delta_limit = np.asarray(ctrl_cfg.upper, dtype=np.float64)
            self.MAX_JOINT_STEP = min(self.MAX_JOINT_STEP, 0.95 * float(self._delta_limit[:5].min()))
            # Keep IK branch weights tuned for delta, but use full settling/
            # correction budget (same bounds as pos) so slow descents verify.
            self.CONTINUITY_W, self.TILT_W, self.GRIP_SETTLE, self.SERVO_ITERS = 0.6, 1.0, 3, 4
            self.MAX_JOINT_ACCEL = self.MAX_JOINT_STEP  # full speed after one step
            self.APPROACH_H, self.LIFT_H, self.PLACE_APPROACH_H, self.RETREAT_H = 0.035, 0.035, 0.03, 0.04
        self.fail_reason = None
        self.n_clipped_steps = 0
        self.n_steps = 0
        self.last_place_err = None
        self.gripper_target = float(self.qpos()[5])
        self.half = None
        self._face_extra = 0.0  # fixed-jaw standoff for a yaw-misaligned grasp (rotated cube is wider)
        self._place_jaw_perp = None  # optional: keep the jaw normal perpendicular to this while placing
        self._pending = []  # queued (arm_q, speed_scale, gripper) waypoints, executed by flush()
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
            + (self.half + self.FACE_CLEARANCE + self._face_extra) * self.normal_local
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
        ref = self._planned_arm() if seed is None else np.asarray(seed, dtype=np.float64)
        pan = np.arctan2(target[1] - self._root_t[1], target[0] - self._root_t[0])
        seeds = [ref]
        for lift in (-0.6, 0.0, 0.6):
            for elbow in (0.0, 0.8):
                for roll in (-1.6, 0.0, 1.6):
                    seeds.append(np.array([pan, lift, elbow, 1.4, roll]))

        best, best_score = None, np.inf

        def consider(seed_q):
            nonlocal best, best_score
            seed_q = np.clip(seed_q, self.arm_lo + 1e-3, self.arm_hi - 1e-3)
            sol = least_squares(residuals, seed_q, jac=jacobian, bounds=(self.arm_lo + 1e-4, self.arm_hi - 1e-4),
                                **self.LSQ_OPTS)
            if not accept(sol.x):
                return False
            # Depending on the branch, the open moving jaw can hang lower than
            # the fixed one and jam on the table before it reaches the object.
            if self.lowest_jaw_z(sol.x, self.gripper_target) < self.MIN_JAW_Z:
                return False
            base, travel = score(sol.x), np.max(np.abs(sol.x - ref))
            sc = base + self.CONTINUITY_W * travel
            if sc < best_score:
                best, best_score = sol.x, sc
            return base < -0.95 and travel < 0.3

        for s in seeds:
            if consider(s):
                return best
        if best is not None:
            # A cube can be gripped from 4 sides, i.e. the same grasp exists at wrist rolls 90 deg
            # apart; the seed grid often misses the one nearest the current roll.
            found = best.copy()
            for dr in (np.pi / 2, -np.pi / 2, np.pi, -np.pi):
                q = found.copy()
                q[4] += dr
                if self.arm_lo[4] < q[4] < self.arm_hi[4]:
                    consider(q)
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
        if q is None and face_axes is not None and self.MISALIGNED_GRASP_DEG > self.MAX_FACE_MISALIGN_DEG:
            # Last resort at the reach limit: grasp up to MISALIGNED_GRASP_DEG off the faces.
            # Squeezing turns the cube square to the jaws; pick() verifies the grasp and retries.
            # The fixed jaw stands off by the rotated cube's wider footprint so it clears the corner.
            q = self._misaligned_grasp(target, face_axes, seed, tol, score)
        return q

    def _misaligned_grasp(self, target, face_axes, seed, tol, score):
        lim = self.MAX_FACE_MISALIGN_DEG
        self.MAX_FACE_MISALIGN_DEG = self.MISALIGNED_GRASP_DEG
        try:
            residuals, accept = self._grasp_problem(target, face_axes, tol, align_weight=0.05)
            q = self._least_squares_ik(residuals, target, seed, accept, score)
            for _ in range(2):  # standoff depends on the angle found, which depends on the standoff
                if q is None:
                    break
                n = self.grasp_frame(q)[1]
                a = np.arccos(min(1.0, max(abs(n @ face_axes[0]), abs(n @ face_axes[1]))))
                self._face_extra = self.half * (np.cos(a) + np.sin(a) - 1.0)
                q = self._least_squares_ik(residuals, target, q, accept, score)
            if q is None:
                self._face_extra = 0.0
            return q
        finally:
            self.MAX_FACE_MISALIGN_DEG = lim

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
        if not held and self._face_extra > 0 and self.MAX_FACE_MISALIGN_DEG < self.MISALIGNED_GRASP_DEG:
            # a misaligned grasp needs an approach line with the same tolerance
            lim, self.MAX_FACE_MISALIGN_DEG = self.MAX_FACE_MISALIGN_DEG, self.MISALIGNED_GRASP_DEG
            try:
                return self.vertical_path(q_bottom, bottom_target, height, face_axes, held, step, max_tilt_deg)
            finally:
                self.MAX_FACE_MISALIGN_DEG = lim
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
            return self.TILT_W * np.linalg.norm(self.held_frame(x)[1][:2]) - 1.0

        return self._least_squares_ik(residuals, target, seed, accept, score)

    def held_tilt(self, arm_q):
        return float(np.arcsin(min(1.0, np.linalg.norm(self.held_frame(arm_q)[1][:2]))))

    def _step(self, arm_q, g):
        """Command absolute joint targets; in delta control this is converted to the normalised
        target change (clipped to the controller limits, which rate-limits large moves)."""
        action = np.concatenate([arm_q, [g]])
        if self.delta_control:
            current = self.base_env.agent.controller._target_qpos[0].cpu().numpy().astype(np.float64)
            raw = (action - current) / self._delta_limit
            if bool((np.abs(raw) > 1.0 + 1e-9).any()):
                self.n_clipped_steps += 1
            action = np.clip(raw, -1.0, 1.0)
        self.n_steps += 1
        self.last_step = self.env.step(action.astype(np.float32))
        if self.vis:
            self.base_env.render_human()
        return self.last_step

    def _commanded_arm(self):
        return self.base_env.agent.controller._target_qpos[0, :5].cpu().numpy().astype(np.float64)

    def _planned_arm(self):
        """Arm joints once all queued motion has run."""
        return self._pending[-1][0] if self._pending else self._commanded_arm()

    def queue(self, arm_q, speed_scale=1.0):
        """Add a waypoint (with the current gripper command) to the pending
        trajectory. Nothing moves until flush()."""
        self._pending.append((np.asarray(arm_q, dtype=np.float64), speed_scale, self.gripper_target))

    def flush(self):
        """Execute all queued waypoints as ONE continuous trajectory: a C1
        (PCHIP) curve through the waypoints, time-parameterised with a
        velocity limit per segment and an acceleration limit, starting and
        ending at rest. The arm therefore only stops where flush() is called
        (before the gripper closes/opens), not at every waypoint."""
        if not self._pending:
            return self.last_step
        pts, vmax, grip = [self._commanded_arm()], [], []
        for q, speed, g in self._pending:
            if np.max(np.abs(q - pts[-1])) > 1e-6:
                pts.append(q)
                # speed_scale is effective in both modes (slow descents use
                # MAX_JOINT_STEP*speed). MAX_JOINT_STEP itself already respects
                # delta per-step limits, so no floor is applied.
                vmax.append(self.MAX_JOINT_STEP * speed)
                grip.append(g)
        last_g = self._pending[-1][2]
        self._pending = []
        if len(pts) == 1:
            self.gripper_target = last_g
            return self.last_step
        pts = np.array(pts)
        seg = np.max(np.abs(np.diff(pts, axis=0)), axis=1)  # max-joint distance per segment
        s_nodes = np.concatenate([[0.0], np.cumsum(seg)])
        curve = PchipInterpolator(s_nodes, pts, axis=0)

        # Sample the curve densely and measure its true arc length (max-joint
        # metric): between sparse waypoints the curve bends, so timing on the
        # straight-line segment lengths would exceed the joint speed limit.
        u, v_lim, seg_of = [0.0], [0.0], [0]
        for i, L in enumerate(seg):
            m = max(2, int(np.ceil(L / 0.005)))
            for k in range(1, m + 1):
                u.append(s_nodes[i] + L * k / m)
                v_lim.append(vmax[i] if k < m or i == len(seg) - 1 else min(vmax[i], vmax[i + 1]))
                seg_of.append(i)
        u, v_lim = np.array(u), np.array(v_lim)
        arc = np.concatenate([[0.0], np.cumsum(np.max(np.abs(np.diff(curve(u), axis=0)), axis=1))])
        ds = np.maximum(np.diff(arc), 1e-9)

        acc = self.MAX_JOINT_ACCEL
        v = v_lim.copy()
        v[0] = 0.0
        for j in range(len(ds)):  # forward pass: acceleration limit
            v[j + 1] = min(v[j + 1], np.sqrt(v[j] ** 2 + 2 * acc * ds[j]))
        v[-1] = 0.0
        for j in range(len(ds) - 1, -1, -1):  # backward pass: deceleration limit
            v[j] = min(v[j], np.sqrt(v[j + 1] ** 2 + 2 * acc * ds[j]))
        t = np.concatenate([[0.0], np.cumsum(2 * ds / np.maximum(v[:-1] + v[1:], 1e-9))])
        n = max(1, int(np.ceil(t[-1])))
        for k in range(1, n + 1):
            j = min(int(np.searchsorted(t, t[-1] * k / n)), len(u) - 1)
            u_k = np.interp(t[-1] * k / n, t, u)
            self._step(curve(u_k), grip[seg_of[j]])
        self.gripper_target = last_g
        return self.last_step

    def set_gripper(self, g, settle=None):
        """Hold the arm still while the gripper moves to g, then `settle` more steps. In delta control
        the gripper target can only change 0.2 rad per step, so large moves take extra steps."""
        self.flush()
        steps = self.GRIP_SETTLE if settle is None else settle
        if self.delta_control:
            current = float(self.base_env.agent.controller._target_qpos[0, 5])
            steps += int(np.ceil(abs(g - current) / self._delta_limit[5] - 1e-6))
        return self.hold(steps, g)

    def hold(self, steps, g=None):
        self.flush()
        if g is not None:
            self.gripper_target = g
        g = self.gripper_target
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

    def _jaw_clearance(self, arm_q, g_from, g_to, n=5):
        """Worst moving-jaw clearance to scene obstacles over a gripper sweep."""
        return min(self.collision.clearance(arm_q, g, links=("moving_jaw_so101_v1_link",))[0]
                   for g in np.linspace(g_from, g_to, n))

    def pick(self, actor, half_size, approach_height=None, lift_height=None, open_extra=0.03, retries=None):
        """Top-down grasp of a box-shaped actor, then lift straight up.
        Returns True if the actor is grasped after lifting. A missed grasp is
        detected (is_grasping after closing) and retried from a re-measured
        object pose, with the jaws opened back up and raised clear first."""
        retries = self.PICK_RETRIES if retries is None else retries
        for attempt in range(retries + 1):
            ok = self._pick_once(actor, half_size, approach_height, lift_height, open_extra)
            if ok or self.fail_reason != "pick_grasp" or attempt == retries:
                return ok
            self.n_retries += 1
            # back off: open, rise straight up, and let the object settle
            self._pending = []
            self.set_gripper(self.g_open)
            q = self._commanded_arm()
            self.queue(self._retreat_ik(q, self._fk(q, self.gripper_target, ("gl",))["gl"][0] + [0, 0, 0.04]),
                       speed_scale=0.7)
            self.hold(2)
            self.fail_reason = None
        return False

    def _pick_once(self, actor, half_size, approach_height, lift_height, open_extra):
        approach_height = self.APPROACH_H if approach_height is None else approach_height
        lift_height = self.LIFT_H if lift_height is None else lift_height
        self.half = half_size
        self._face_extra = 0.0
        center = actor.pose.sp.p.astype(np.float64)
        axes = self.horizontal_axes(actor)
        g_contact = self.gripper_qpos_for_gap(2 * half_size)
        self.g_squeeze = max(self.CLOSED, g_contact - self.SQUEEZE)
        self.collision.refresh(exclude=[actor])

        # The open moving jaw must not start inside a pocket/tray wall or a
        # neighbouring object, or it stalls there and never reaches the object
        # (measured: >=1.2 mm overlap at open in every rearrange pick_grasp
        # failure). Open only as wide as the surroundings allow.
        q_grasp = None
        for extra in sorted({open_extra, *[e for e in self.OPEN_EXTRA_FALLBACK if e < open_extra]}, reverse=True):
            g_open = self.gripper_qpos_for_gap(2 * half_size + extra)
            self.gripper_target = g_open  # IK checks jaw clearance at this angle
            self._face_extra = 0.0
            q = self.solve_ik(center, axes)
            if q is None:
                continue
            if self._face_extra > 0:  # misaligned: the rotated cube is wider, open further
                g_open = self.gripper_qpos_for_gap(2 * (half_size + self._face_extra) + extra)
                self.gripper_target = g_open
            clear = self._jaw_clearance(q, g_open, g_contact)
            if q_grasp is None or clear > best_clear:
                q_grasp, best_clear, best_open, best_extra = q, clear, g_open, self._face_extra
            if clear >= self.JAW_CLEARANCE:
                break
        if q_grasp is None:
            self.fail_reason = "pick_ik"
            return False
        g_open = self.g_open = best_open
        self._face_extra = best_extra
        self.gripper_target = g_open
        # Near the edge of the workspace the approach line may run out of
        # reach early; accept a shorter one (>= 1 cm) rather than failing.
        up = self.vertical_path(q_grasp, center, approach_height, axes)
        if len(up) < 2:
            self.fail_reason = "pick_reach"
            return False

        self.queue(up[-1])
        for q in up[-2::-1]:
            self.queue(q, speed_scale=0.5)
        self.hold(1)  # flushes: the arm comes to rest only here, just before closing
        self.set_gripper(self.g_squeeze)
        grasped = bool(self.agent.is_grasping(actor)[0])
        if not grasped:
            self.fail_reason = "pick_grasp"
        # Once held, the jaw orientation no longer matters much, so lift with
        # the held-object IK (cube kept near flat), which reaches further. The
        # lift is only queued, so it blends into the carry done by place().
        self.measure_held(actor)
        lift = self.vertical_path(self._commanded_arm(), self.held_frame(self._commanded_arm())[0],
                                  lift_height, held=True)
        for q in lift[1:]:
            self.queue(q, speed_scale=0.7)
        return grasped

    def place(self, actor, target_center, face_axes=None, jaw_perp=None, release_height=0.005,
              approach_height=None, retreat_height=None, release_gap=0.03):
        """Put the held actor's center at target_center (flat-ish, yaw aligned
        to face_axes if given, jaw normal perpendicular to jaw_perp if given)
        via a straight vertical descent, open the jaws by release_gap beyond
        the object width, and retreat upwards."""
        approach_height = self.PLACE_APPROACH_H if approach_height is None else approach_height
        retreat_height = self.RETREAT_H if retreat_height is None else retreat_height
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
            self.fail_reason = "place_ik"
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
            self.fail_reason = "place_reach"
            return False

        self.queue(down[-1])
        for q in down[-2::-1]:
            self.queue(q, speed_scale=0.5)
        self.hold(1)  # flushes lift + carry + descent as one motion
        self._servo_held(actor, self.held_frame(q_place)[0], face_axes)
        self.set_gripper(self.gripper_qpos_for_gap(2 * self.half + release_gap))
        q_now = self._commanded_arm()
        fk = self._fk(q_now, self.gripper_target, ("gl",))
        # queued, so it blends into whatever comes next (next pick / final rest)
        self.queue(self._retreat_ik(q_now, fk["gl"][0] + [0, 0, retreat_height]), speed_scale=0.7)
        self._place_jaw_perp = None
        return True

    def _servo_held(self, actor, goal, face_axes, tol=0.003, iters=None):
        """Closed-loop correction with verified tracking (bounded).

        Shifts the IK target by observed error, drives toward each correction
        with bounded substeps until commanded≈achieved (<8mrad) so clipped
        delta targets cannot silently under-shoot, then settles before
        re-measuring. Records final placement error for diagnostics.
        """
        target = np.asarray(goal, dtype=np.float64).copy()
        n_iter = self.SERVO_ITERS if iters is None else iters
        for _ in range(n_iter):
            err = actor.pose.sp.p - goal
            err_n = float(np.linalg.norm(err))
            self.last_place_err = err_n
            if err_n < tol:
                break
            target -= err
            residuals, accept = self._held_problem(target, 2e-3, 40.0, face_axes)
            q = self._local_solve(residuals, accept, self._commanded_arm())
            if q is None:
                self.fail_reason = "servo_ik"
                break
            # Bounded drive to correction target (verify, don't assume).
            for _sub in range(6):
                cur = self._commanded_arm()
                if float(np.max(np.abs(q - cur))) < 0.008:
                    break
                self._step(q, self.gripper_target)
            self.hold(2)
        err = actor.pose.sp.p - goal
        self.last_place_err = float(np.linalg.norm(err))

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
