import numpy as np

from examples.motionplanning.so101.motionplanner import SO101GraspSolver


def solve(env, seed=None, debug=False, vis=False):
    """SO101StackCube-v1: pick itemA and set it on top of itemB, release, and
    back off so the arm is static and not touching itemA."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped

    half_a = e.itemA_half_sizes[0].item()
    half_b = e.itemB_half_sizes[0].item()
    if not solver.pick(e.itemA, half_a):
        return -1

    target = e.itemB.pose.sp.p.astype(np.float64) + [0, 0, half_a + half_b]
    if not solver.place(e.itemA, target, solver.horizontal_axes(e.itemB)):
        return -1
    return solver.hold(3)
