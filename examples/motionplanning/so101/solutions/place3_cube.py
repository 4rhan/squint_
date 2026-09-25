import numpy as np

from examples.motionplanning.so101.motionplanner import SO101GraspSolver

MIN_APPROACH_PATH = 5  # waypoints (1 cm apart) of clear vertical approach required to accept a grasp
WALL_INNER = 0.0025  # the bin's walls are centred on its outer half-size, 5 mm thick
FINAL_RETREAT = 0.06  # after the last cube, back off high enough to be clear of the bin
RELEASE_ABOVE_WALL = 0.004  # cube bottom this far above the wall top when it is let go


def bin_slots(e, bin_x, bin_y, base_xy, half):
    """Three cube centres inside the bin that fit even the smallest randomised bin: two on the side away from
    the robot (left/right along the bin's long axis) and one on the near side, in the middle. Returned in the
    order to fill them: far ones first, so the arm never reaches over a cube it already placed."""
    bin_pose = e.bin.pose.sp
    hx, hy = e.bin_half_sizes_x[0].item(), e.bin_half_sizes_y[0].item()
    long_ax, short_ax = (bin_y, bin_x) if hy >= hx else (bin_x, bin_y)
    h_long, h_short = (hy, hx) if hy >= hx else (hx, hy)
    reach = lambda h: h - WALL_INNER - half - 0.008  # farthest a cube centre may sit from the bin centre (the drop scatters a little)
    cl, cs = reach(h_long), reach(h_short)
    toward_robot = short_ax if np.dot(short_ax[:2], base_xy - bin_pose.p[:2]) > 0 else -short_ax
    c = bin_pose.p
    slots = [c - toward_robot * cs - long_ax * cl, c - toward_robot * cs + long_ax * cl, c + toward_robot * cs]
    return [np.array([p[0], p[1], 0.0]) for p in slots], long_ax


def _in_bin_rect(bin_pose, hx, hy, pts_xy, margin):
    """Points (N, 2) inside the bin's outer rectangle (inflated by margin), in the bin's rotated frame."""
    q = bin_pose.q
    yaw = 2 * np.arctan2(q[3], q[0])
    c, s = np.cos(yaw), np.sin(yaw)
    rel = pts_xy - bin_pose.p[:2]
    xb, yb = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
    return (np.abs(xb) < hx + margin) & (np.abs(yb) < hy + margin)


def _blocked_points(e, cube_k, d, others, bin_hx, bin_hy):
    """How many sample points of the jaws' footprint, for jaw direction d (fixed jaw -> moving jaw), touch the
    bin or another cube. The fixed jaw sits behind the cube (-d side), the opened moving jaw in front (+d)."""
    c = e.cubes[cube_k].pose.sp.p[:2].astype(np.float64)
    half = e.cube_half_sizes[0, cube_k].item()
    u = np.array([-d[1], d[0]])
    pts = []
    for s in np.arange(-half - 0.032, -half - 0.002, 0.005):  # fixed jaw side
        pts += [c + d * s + u * w for w in (-0.012, 0.0, 0.012)]
    for s in np.arange(half + 0.002, half + 0.068, 0.005):  # opened moving jaw side
        pts += [c + d * s + u * w for w in (-0.012, 0.0, 0.012)]
    pts = np.array(pts)
    blocked = _in_bin_rect(e.bin.pose.sp, bin_hx, bin_hy, pts, margin=0.004)
    for j in others:
        oc = e.cubes[j].pose.sp.p[:2]
        oh = e.cube_half_sizes[0, j].item() * 1.42 + 0.004  # circumscribed radius + margin
        blocked |= np.linalg.norm(pts - oc, axis=1) < oh
    return int(blocked.sum())


def choose_jaw_dir(solver, e, k, remaining, bin_hx, bin_hy):
    """Pick the jaw direction (one of the cube's four face directions) whose footprint is free of the bin and
    the other cubes and that the arm can reach. Returns the unit direction, or None if the grasp is infeasible."""
    cube, half = e.cubes[k], e.cube_half_sizes[0, k].item()
    a, b = solver.horizontal_axes(cube)
    natural = solver.natural_jaw_dir(cube, half)
    cands = [(_blocked_points(e, k, d[:2], [j for j in remaining if j != k], bin_hx, bin_hy),
              0 if natural is not None and d @ natural > 0.99 else 1, i, d)
             for i, d in enumerate((a, -a, b, -b))]
    # least blocked first; among equally free ones, the direction the arm reaches most naturally
    for _, _, _, d in sorted(cands, key=lambda t: t[:3]):
        if solver.grasp_feasible(cube, half, jaw_dir=d, min_path=MIN_APPROACH_PATH):
            return d
    return None


def solve(env, seed=None, debug=False, vis=False):
    """SO101Place3Cube-v1: pick the three cubes one at a time and set them in a row along the bin's long axis,
    jaws across the row so they clear the neighbouring cubes, then back off so the arm is static and clear.
    Uses the simulator's ground-truth poses of every cube and the bin: seeds where some cube is out of the
    arm's reach are rejected before anything moves, and each grasp uses a jaw direction that does not hit the
    bin walls or the other cubes."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped

    bin_pose = e.bin.pose.sp
    bin_x, bin_y = solver.horizontal_axes(e.bin)
    hx, hy = e.bin_half_sizes_x[0].item(), e.bin_half_sizes_y[0].item()
    floor_z = bin_pose.p[2] + e.bin_thickness

    # every cube must be graspable before we start (cheap, no simulation)
    everything = [0, 1, 2]
    if any(choose_jaw_dir(solver, e, k, everything, hx, hy) is None for k in everything):
        return -1

    base = e.agent.robot.pose.sp.p
    order = sorted(range(3), key=lambda k: np.linalg.norm(e.cubes[k].pose.sp.p[:2] - base[:2]))  # nearest first
    half0 = e.cube_half_sizes[0, order[0]].item()
    slots, long_ax = bin_slots(e, bin_x, bin_y, base[:2], half0)

    # Each cube is released with the jaws just above the wall top, so the jaws never have to fit down between a
    # wall and the cube (they don't, in the smaller randomised bins); it drops the last centimetres.
    wall_top = bin_pose.p[2] + 2 * e.bin_half_sizes_z[0].item()

    remaining = list(order)
    for n, (slot, k) in enumerate(zip(slots, order)):
        cube, half = e.cubes[k], e.cube_half_sizes[0, k].item()
        d = choose_jaw_dir(solver, e, k, remaining, hx, hy)
        if d is None or not solver.pick(cube, half, jaw_dir=d):
            return -1
        remaining.remove(k)
        target = slot.copy()
        target[2] = floor_z + half
        release_h = max(wall_top + half + RELEASE_ABOVE_WALL - target[2], 0.005)
        last = n == 2
        if not solver.place(cube, target, (bin_x, bin_y), jaw_perp=long_ax, release_gap=0.012, release_height=release_h,
                            retreat_height=FINAL_RETREAT if last else None):
            return -1
    return solver.hold(6)
