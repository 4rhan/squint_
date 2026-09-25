import numpy as np
from examples.motionplanning.so101.motionplanner import SO101GraspSolver


def solve(env, seed=None, debug=False, vis=False):
    """SO101Tower2Cube-v1 control: large -> tower marker, medium -> large."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    hl = float(e.large_half[0]); hm = float(e.medium_half[0])
    tx, ty = float(e.tower_xy[0]), float(e.tower_xy[1])
    if not solver.pick(e.cubeL, hl):
        return -1
    target_L = np.array([tx, ty, hl])
    if not solver.place(e.cubeL, target_L, solver.horizontal_axes(e.cubeL),
                        approach_height=0.05, retreat_height=0.07):
        return -1
    if not solver.pick(e.cubeM, hm, approach_height=0.07):
        return -1
    base_p = e.cubeL.pose.sp.p.astype(np.float64)
    target_M = base_p + np.array([0, 0, hl + hm])
    if not solver.place(e.cubeM, target_M, solver.horizontal_axes(e.cubeL),
                        approach_height=0.05, retreat_height=0.07):
        return -1
    return solver.hold(12)
