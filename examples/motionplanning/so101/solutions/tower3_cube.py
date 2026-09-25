import numpy as np
from examples.motionplanning.so101.motionplanner import SO101GraspSolver


def solve(env, seed=None, debug=False, vis=False):
    """SO101Tower3Cube-v1: large -> tower marker, medium -> large, small -> medium."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    hl = float(e.large_half[0]); hm = float(e.medium_half[0]); hs = float(e.small_half[0])
    tx, ty = float(e.tower_xy[0]), float(e.tower_xy[1])
    # 1) base to tower
    if not solver.pick(e.cubeL, hl):
        return -1
    target_L = np.array([tx, ty, hl])
    if not solver.place(e.cubeL, target_L, solver.horizontal_axes(e.cubeL),
                        approach_height=0.05, retreat_height=0.07):
        return -1
    # 2) medium on large
    if not solver.pick(e.cubeM, hm, approach_height=0.07):
        return -1
    base_p = e.cubeL.pose.sp.p.astype(np.float64)
    target_M = base_p + np.array([0, 0, hl + hm])
    if not solver.place(e.cubeM, target_M, solver.horizontal_axes(e.cubeL),
                        approach_height=0.05, retreat_height=0.07):
        return -1
    # 3) small on medium
    if not solver.pick(e.cubeS, hs, approach_height=0.07):
        return -1
    mid_p = e.cubeM.pose.sp.p.astype(np.float64)
    target_S = mid_p + np.array([0, 0, hm + hs])
    if not solver.place(e.cubeS, target_S, solver.horizontal_axes(e.cubeM),
                        approach_height=0.05, retreat_height=0.07):
        return -1
    return solver.hold(12)
