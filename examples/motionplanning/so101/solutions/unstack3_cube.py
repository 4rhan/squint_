import numpy as np

from examples.motionplanning.so101.motionplanner import SO101GraspSolver

LAST_FAIL = {}  # stage/reason of the last failed solve, read by examples/collect_new_tasks_demos.py


def _table_spot(e, avoid, clearance=0.07):
    """Reachable table spot for a cube taken off the tower, at least `clearance` from every cube in
    `avoid`, nearest to the tower (short carry)."""
    tower = e.itemC.pose.sp.p[:2]
    best = None
    for r in np.linspace(0.17, 0.29, 7):
        for ang in np.linspace(-0.9, 0.9, 19):
            xy = np.array([r * np.cos(ang), r * np.sin(ang)])
            if all(np.linalg.norm(xy - a.pose.sp.p[:2]) > clearance for a in avoid):
                d = np.linalg.norm(xy - tower)
                if best is None or d < best[0]:
                    best = (d, xy)
    return None if best is None else best[1]


def solve(env, seed=None, debug=False, vis=False):
    """SO101Unstack3Cube-v1: take itemA off the tower and put it on the table, then itemB, release, and
    back off so the arm is static and touching neither cube. Lifts and retreats are raised so the held
    cube and the jaws clear the rest of the tower."""
    env.reset(seed=seed)
    LAST_FAIL.clear()
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    half = {k: getattr(e, f"item{k}_half_sizes")[0].item() for k in "ABC"}

    for top in ("A", "B"):
        item = getattr(e, f"item{top}")
        others = [getattr(e, f"item{k}") for k in "ABC" if k != top]
        spot = _table_spot(e, others)
        if spot is None:
            LAST_FAIL.update(stage=f"spot_{top}", reason="no_free_spot")
            return -1
        if not solver.pick(item, half[top], approach_height=0.05, lift_height=0.05):
            LAST_FAIL.update(stage=f"pick_{top}", reason=solver.fail_reason)
            return -1
        target = np.array([spot[0], spot[1], half[top]])
        if not solver.place(item, target, solver.horizontal_axes(item), approach_height=0.06,
                            retreat_height=0.06):
            LAST_FAIL.update(stage=f"place_{top}", reason=solver.fail_reason)
            return -1
    return solver.hold(5)  # let the cubes settle so the static checks pass
