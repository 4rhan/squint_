"""Watch the scripted solver do a task: live in the SAPIEN viewer, or recorded as one smooth mp4 per seed.

The live viewer only redraws once per control step (10 Hz) and freezes while the solver plans (IK), so it
looks jerky. --video records instead: frames are captured inside the physics sub-steps (100 Hz sim, every
--substep-every sub-steps), so planning time doesn't show and the motion is continuous.

  python -m examples.view_task -e SO101Rearrange2-v1 --seeds 0 1 2                  # live viewer (needs a display)
  python -m examples.view_task -e SO101Rearrange2-v1 --seeds 0 1 2 --video task_videos   # mp4s, headless
"""
import argparse
import os

import envs  # noqa: F401 registers tasks
import gymnasium as gym
import imageio
import numpy as np
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from examples.collect_new_tasks_demos import SOLVER_MODULES, load_solver


def _frame(base):
    img = base.render()
    img = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    return img[0] if img.ndim == 4 else img


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--env-id", default="SO101Rearrange2-v1")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--hold", type=int, default=20, help="extra still control steps shown after each episode")
    p.add_argument("--video", default=None, metavar="DIR", help="record mp4s into DIR instead of the live viewer")
    p.add_argument("--size", type=int, default=512, help="video resolution")
    p.add_argument("--substep-every", type=int, default=2, help="capture a frame every N physics sub-steps")
    p.add_argument("--fast", action="store_true", help="solver FAST profile (fewer settle pauses)")
    args = p.parse_args()

    from examples.motionplanning.so101.motionplanner import SO101GraspSolver
    SO101GraspSolver.FAST = args.fast
    kw = dict(num_envs=1, obs_mode="rgb+segmentation", sim_backend="cpu",
              sensor_configs=dict(width=128, height=128), reward_mode="normalized_dense")
    if args.video:
        kw.update(render_mode="rgb_array", human_render_camera_configs=dict(width=args.size, height=args.size))
    else:
        kw.update(render_mode="human")
    env = FlattenRGBDObservationWrapper(gym.make(args.env_id, **kw), rgb=True, depth=False, state=True)
    base = env.unwrapped
    solve = load_solver(args.env_id)
    mod = __import__(SOLVER_MODULES[args.env_id], fromlist=["LAST_FAIL"])

    frames, recording = [], [False]
    if args.video:
        os.makedirs(args.video, exist_ok=True)
        sim_step, count = base.scene.step, [0]

        def step_and_capture():  # called once per physics sub-step
            sim_step()
            count[0] += 1
            if recording[0] and count[0] % args.substep_every == 0:
                frames.append(_frame(base))
        base.scene.step = step_and_capture
    fps = round(base.sim_freq / args.substep_every) if args.video else None

    for seed in args.seeds:
        frames.clear()
        recording[0] = True
        r = solve(env, seed=seed, vis=not args.video)
        info = base.evaluate()
        ok = r != -1 and bool(info["success"][0])
        for _ in range(args.hold):
            if args.video:
                env.step(np.zeros(env.action_space.shape, dtype=np.float32))  # zero delta = stay still
            else:
                base.render_human()
        recording[0] = False
        fail = getattr(mod, "LAST_FAIL", {})
        msg = (f"seed {seed}: success {ok}, steps {int(base.elapsed_steps[0])}, "
               f"num_correct {int(info.get('num_correct', [0])[0])}, fail {fail.get('stage')}/{fail.get('reason')}")
        if args.video:
            out = os.path.join(args.video, f"{args.env_id}_seed{seed}_{'PASS' if ok else 'FAIL'}.mp4")
            imageio.mimsave(out, frames, fps=fps, macro_block_size=1)
            msg += f" -> {out} ({len(frames)} frames, {fps} fps)"
        print(msg, flush=True)
    env.close()
