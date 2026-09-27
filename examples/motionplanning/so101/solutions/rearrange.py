"""Cyclic rearrangement solutions (example sequences, not required)."""
import numpy as np
from examples.motionplanning.so101.motionplanner import SO101GraspSolver

LAST_FAIL = {"stage": None, "reason": None, "solver": None}


def _note(solver, stage):
    LAST_FAIL["stage"] = stage
    LAST_FAIL["reason"] = solver.fail_reason
    LAST_FAIL["solver"] = {"clipped": solver.n_clipped_steps, "steps": solver.n_steps,
                           "place_err": solver.last_place_err}


def _place_in_pocket(solver, e, obj_idx, pocket_idx, half):
    target = np.array([e.pocket_x, e.pocket_ys[pocket_idx], e.pocket_floor_t + half])
    cube = e.cubes[obj_idx]
    # Pockets in a row along y; keep jaws across row (jaw normal perp to y).
    x_axis = np.array([1.0, 0.0, 0.0]); y_axis = np.array([0.0, 1.0, 0.0])
    row = np.array([0.0, 1.0, 0.0])
    return solver.place(cube, target, (x_axis, y_axis), jaw_perp=row, release_gap=0.008)


def solve3(env, seed=None, debug=False, vis=False):
    """SO101Rearrange3-v1 example: A->buffer; B->P0; C->P1; A->P2."""
    LAST_FAIL.update(stage=None, reason=None, solver=None)
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    half = float(e.cube_half[0])
    steps = [("pick0", lambda: solver.pick(e.cubes[0], half, open_extra=0.008)),
             ("place0-buf", lambda: _place_in_pocket(solver, e, 0, 3, half)),
             ("pick1", lambda: solver.pick(e.cubes[1], half, open_extra=0.008)),
             ("place1-p0", lambda: _place_in_pocket(solver, e, 1, 0, half)),
             ("pick2", lambda: solver.pick(e.cubes[2], half, open_extra=0.008)),
             ("place2-p1", lambda: _place_in_pocket(solver, e, 2, 1, half)),
             ("pick0b", lambda: solver.pick(e.cubes[0], half, open_extra=0.008)),
             ("place0-p2", lambda: _place_in_pocket(solver, e, 0, 2, half))]
    for stage, fn in steps:
        if not fn():
            _note(solver, stage)
            return -1
    return solver.hold(12)


def solve2(env, seed=None, debug=False, vis=False):
    """SO101Rearrange2-v1 example swap: A->buffer; B->P0; A->P1."""
    LAST_FAIL.update(stage=None, reason=None, solver=None)
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped
    half = float(e.cube_half[0])
    steps = [("pick0", lambda: solver.pick(e.cubes[0], half, open_extra=0.008)),
             ("place0-buf", lambda: _place_in_pocket(solver, e, 0, 2, half)),
             ("pick1", lambda: solver.pick(e.cubes[1], half, open_extra=0.008)),
             ("place1-p0", lambda: _place_in_pocket(solver, e, 1, 0, half)),
             ("pick0b", lambda: solver.pick(e.cubes[0], half, open_extra=0.008)),
             ("place0-p1", lambda: _place_in_pocket(solver, e, 0, 1, half))]
    for stage, fn in steps:
        if not fn():
            _note(solver, stage)
            return -1
    return solver.hold(12)


def solve(env, seed=None, debug=False, vis=False):
    # Default: 3-object cycle (caller selects via env id).
    return solve3(env, seed, debug, vis)
