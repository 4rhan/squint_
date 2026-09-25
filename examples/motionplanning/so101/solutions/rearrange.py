"""Cyclic rearrangement solutions (example sequences, not required)."""
import numpy as np
from examples.motionplanning.so101.motionplanner import SO101GraspSolver


def _place_in_pocket(solver, e, obj_idx, pocket_idx, half):
    target = np.array([e.pocket_x, e.pocket_ys[pocket_idx], e.pocket_floor_t + half])
    cube = e.cubes[obj_idx]
    # Pockets in a row along y; keep jaws across row (jaw normal perp to y).
    x_axis = np.array([1.0, 0.0, 0.0]); y_axis = np.array([0.0, 1.0, 0.0])
    row = np.array([0.0, 1.0, 0.0])
    return solver.place(cube, target, (x_axis, y_axis), jaw_perp=row, release_gap=0.008)


def solve3(env, seed=None, debug=False, vis=False):
    """SO101Rearrange3-v1 example: A->buffer; B->P0; C->P1; A->P2."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    half = float(e.cube_half[0])
    # A(0)->buffer(3)
    if not solver.pick(e.cubes[0], half, open_extra=0.015):
        return -1
    if not _place_in_pocket(solver, e, 0, 3, half):
        return -1
    # B(1)->P0
    if not solver.pick(e.cubes[1], half, open_extra=0.015):
        return -1
    if not _place_in_pocket(solver, e, 1, 0, half):
        return -1
    # C(2)->P1
    if not solver.pick(e.cubes[2], half, open_extra=0.015):
        return -1
    if not _place_in_pocket(solver, e, 2, 1, half):
        return -1
    # A(0)->P2
    if not solver.pick(e.cubes[0], half, open_extra=0.015):
        return -1
    if not _place_in_pocket(solver, e, 0, 2, half):
        return -1
    return solver.hold(12)


def solve2(env, seed=None, debug=False, vis=False):
    """SO101Rearrange2-v1 example swap: A->buffer; B->P0; A->P1."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    half = float(e.cube_half[0])
    if not solver.pick(e.cubes[0], half, open_extra=0.015):
        return -1
    if not _place_in_pocket(solver, e, 0, 2, half):
        return -1
    if not solver.pick(e.cubes[1], half, open_extra=0.015):
        return -1
    if not _place_in_pocket(solver, e, 1, 0, half):
        return -1
    if not solver.pick(e.cubes[0], half, open_extra=0.015):
        return -1
    if not _place_in_pocket(solver, e, 0, 1, half):
        return -1
    return solver.hold(12)


def solve(env, seed=None, debug=False, vis=False):
    # Default: 3-object cycle (caller selects via env id).
    return solve3(env, seed, debug, vis)
