from examples.motionplanning.so101.motionplanner import SO101GraspSolver


def solve(env, seed=None, debug=False, vis=False):
    """SO101LiftCube-v1: grasp the item, lift it, and return the arm to
    rest_qpos (success = item lifted + grasped + arm near rest)."""
    env.reset(seed=seed)
    solver = SO101GraspSolver(env, vis=vis)
    e = env.unwrapped

    if not solver.pick(e.item, e.item_half_sizes[0].item()):
        return -1
    solver.move_to(e.rest_qpos[:5].cpu().numpy(), solver.g_squeeze)
    return solver.hold(2)
