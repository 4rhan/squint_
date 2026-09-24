import numpy as np

from examples.motionplanning.so101.motionplanner import SO101GraspSolver

SLOT_SPACING = 0.03


def solve(env, seed=None, debug=False, vis=False):
    """SO101Place3Cube-v1: pick the three cubes one at a time and set them in
    a row along the bin's long axis, jaws across the row so they clear the
    neighbouring cubes, then back off so the arm is static and clear."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped

    bin_pose = e.bin.pose.sp
    R = bin_pose.to_transformation_matrix()[:3, :3]
    bin_x, bin_y = solver.horizontal_axes(e.bin)
    hx, hy = e.bin_half_sizes_x[0].item(), e.bin_half_sizes_y[0].item()
    row = bin_y if hy >= hx else bin_x
    floor_z = bin_pose.p[2] + e.bin_thickness

    # nearest cubes first: they are the most likely to be reachable, and the
    # slots fill from the robot side outwards
    base = e.agent.robot.pose.sp.p
    slots = sorted([k - 1 for k in range(3)], key=lambda k: np.linalg.norm(bin_pose.p[:2] + k * SLOT_SPACING * row[:2] - base[:2]))
    order = sorted(range(3), key=lambda k: np.linalg.norm(e.cubes[k].pose.sp.p[:2] - base[:2]))

    res = -1
    for slot, k in zip(slots, order):
        cube, half = e.cubes[k], e.cube_half_sizes[0, k].item()
        if not solver.pick(cube, half):
            return -1
        target = bin_pose.p + slot * SLOT_SPACING * row
        target[2] = floor_z + half
        if not solver.place(cube, target, (bin_x, bin_y), jaw_perp=row, release_gap=0.012):
            return -1
        res = solver.last_step
    return solver.hold(3)
