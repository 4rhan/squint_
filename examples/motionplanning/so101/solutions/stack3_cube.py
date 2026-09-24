import numpy as np

from examples.motionplanning.so101.motionplanner import SO101GraspSolver

# Beyond this radius from the robot base the SO101 can't place a cube ~7.5 cm
# up (the top of a 3-tower) accurately, so a far-away base cube is first
# moved to a reachable free spot.
MAX_TOWER_RADIUS = 0.35


def _free_spot(e, base, avoid, clearance=0.07):
    """Reachable table spot for the tower base, clear of the other cubes,
    nearest to the base cube's current position."""
    here = base.pose.sp.p[:2]
    best = None
    for r in np.linspace(0.20, 0.28, 5):
        for ang in np.linspace(-0.8, 0.8, 17):
            xy = np.array([r * np.cos(ang), r * np.sin(ang)])
            if all(np.linalg.norm(xy - a.pose.sp.p[:2]) > clearance for a in avoid):
                d = np.linalg.norm(xy - here)
                if best is None or d < best[0]:
                    best = (d, xy)
    return None if best is None else best[1]


def solve(env, seed=None, debug=False, vis=False):
    """SO101Stack3Cube-v1: put itemB on itemC, then itemA on itemB, release,
    and back off so the arm is static and touching neither cube. If itemC is
    too far out to build a tower on, it is first moved closer. Approach and
    retreat heights are raised so the arm clears the growing tower."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    half = {k: getattr(e, f"item{k}_half_sizes")[0].item() for k in "ABC"}
    base = e.agent.robot.pose.sp.p[:2]

    if np.linalg.norm(e.itemC.pose.sp.p[:2] - base) > MAX_TOWER_RADIUS:
        spot = _free_spot(e, e.itemC, [e.itemA, e.itemB])
        if spot is None or not solver.pick(e.itemC, half["C"]):
            return -1
        target = np.array([spot[0], spot[1], half["C"]])
        if not solver.place(e.itemC, target, solver.horizontal_axes(e.itemC), retreat_height=0.07):
            return -1

    for top, bottom in (("B", "C"), ("A", "B")):
        item, bottom_item = getattr(e, f"item{top}"), getattr(e, f"item{bottom}")
        if not solver.pick(item, half[top], approach_height=0.07):
            return -1
        target = bottom_item.pose.sp.p.astype(np.float64) + [0, 0, half[top] + half[bottom]]
        if not solver.place(item, target, solver.horizontal_axes(bottom_item),
                            approach_height=0.05, retreat_height=0.07):
            return -1
    return solver.hold(5)  # let the tower settle so the static checks pass
