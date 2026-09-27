import numpy as np
from examples.motionplanning.so101.motionplanner import SO101GraspSolver

LAST_FAIL = {"stage": None, "reason": None, "solver": None}


def _note(solver, stage):
    LAST_FAIL["stage"] = stage
    LAST_FAIL["reason"] = solver.fail_reason
    LAST_FAIL["solver"] = {"clipped": solver.n_clipped_steps, "steps": solver.n_steps,
                           "place_err": solver.last_place_err}


def solve(env, seed=None, debug=False, vis=False):
    """SO101Tower3Cube-v1: large -> tower marker, medium -> large, small -> medium."""
    LAST_FAIL.update(stage=None, reason=None, solver=None)
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    hl = float(e.large_half[0]); hm = float(e.medium_half[0]); hs = float(e.small_half[0])
    tx, ty = float(e.tower_xy[0]), float(e.tower_xy[1])
    steps = [
        ("pickL", lambda: solver.pick(e.cubeL, hl)),
        ("placeL", lambda: solver.place(e.cubeL, np.array([tx, ty, hl]),
                                        solver.horizontal_axes(e.cubeL),
                                        approach_height=0.05, retreat_height=0.07)),
        ("pickM", lambda: solver.pick(e.cubeM, hm, approach_height=0.07)),
        ("placeM", lambda: solver.place(e.cubeM, e.cubeL.pose.sp.p.astype(np.float64) + np.array([0, 0, hl + hm]),
                                        solver.horizontal_axes(e.cubeL),
                                        approach_height=0.05, retreat_height=0.07)),
        ("pickS", lambda: solver.pick(e.cubeS, hs, approach_height=0.07)),
        ("placeS", lambda: solver.place(e.cubeS, e.cubeM.pose.sp.p.astype(np.float64) + np.array([0, 0, hm + hs]),
                                        solver.horizontal_axes(e.cubeM),
                                        approach_height=0.05, retreat_height=0.07)),
    ]
    for stage, fn in steps:
        if not fn():
            _note(solver, stage)
            return -1
    return solver.hold(12)
