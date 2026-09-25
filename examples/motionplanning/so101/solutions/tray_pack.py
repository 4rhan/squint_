"""Assigned-compartment packing solutions (any order valid; demos use nearest-first)."""
import numpy as np
from examples.motionplanning.so101.motionplanner import SO101GraspSolver


def _solve_n(env, seed, vis, n):
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    half = float(e.cube_half[0])
    # Compartment world centers (tray at fixed xy, identity yaw).
    comp_x = [float(e.tray.pose.sp.p[0]) + float(x) for x in e.comp_local_x.cpu().numpy()]
    tray_y = float(e.tray.pose.sp.p[1])
    floor_z = float(e.tray.pose.sp.p[2]) + e.floor_t
    # Row direction along x for jaw orientation.
    row = np.array([1.0, 0.0, 0.0])
    x_axis = np.array([1.0, 0.0, 0.0]); y_axis = np.array([0.0, 1.0, 0.0])
    base = e.agent.robot.pose.sp.p
    order = sorted(range(n), key=lambda k: np.linalg.norm(e.cubes[k].pose.sp.p[:2] - base[:2]))
    res = -1
    for k in order:
        cube = e.cubes[k]
        if not solver.pick(cube, half, open_extra=0.015):
            return -1
        target = np.array([comp_x[k], tray_y, floor_z + half])
        if not solver.place(cube, target, (x_axis, y_axis), jaw_perp=row, release_gap=0.008):
            return -1
        res = solver.last_step
    return solver.hold(12)


def solve_pack1(env, seed=None, debug=False, vis=False):
    return _solve_n(env, seed, vis, 1)

def solve_pack2(env, seed=None, debug=False, vis=False):
    return _solve_n(env, seed, vis, 2)

def solve_pack3(env, seed=None, debug=False, vis=False):
    return _solve_n(env, seed, vis, 3)

# Default entry for generic loader (3-object).
def solve(env, seed=None, debug=False, vis=False):
    return _solve_n(env, seed, vis, 3)
