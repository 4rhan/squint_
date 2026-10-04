"""Scripted-solver success of SO101Rearrange2-v1 for several pocket layouts (pocket_ys = P0,P1,buffer).

  python -m examples.check_rearrange2_layouts --seeds 20 --layouts " -0.06,0,0.06" " -0.09,-0.03,0.03"
"""
import argparse
from multiprocessing import Pool

LAYOUTS = ["-0.06,0.0,0.06", "-0.09,-0.03,0.03", "-0.03,0.03,0.09", "0.03,-0.03,-0.09"]


def run(job):
    layout, n_seeds = job
    import envs  # noqa: F401 registers tasks
    import gymnasium as gym
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    from examples.motionplanning.so101.solutions import rearrange as R
    ys = tuple(float(v) for v in layout.split(","))
    env = gym.make("SO101Rearrange2-v1", num_envs=1, obs_mode="rgb+segmentation", render_mode="rgb_array",
                   sim_backend="cpu", sensor_configs=dict(width=128, height=128),
                   reward_mode="normalized_dense", pocket_ys=ys)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    ok, fails, steps = 0, [], []
    for s in range(n_seeds):
        r = R.solve2(env, seed=s)
        succ = r != -1 and bool(env.unwrapped.evaluate()["success"][0])
        ok += succ
        if succ:
            steps.append(int(env.unwrapped.elapsed_steps[0]))
        else:
            fails.append(f"{s}:{R.LAST_FAIL['stage']}/{R.LAST_FAIL['reason']}")
    env.close()
    mean = sum(steps) / len(steps) if steps else 0
    return f"pocket_ys={ys}: {ok}/{n_seeds} success, mean steps {mean:.0f}, fails {fails}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--layouts", nargs="+", default=LAYOUTS)
    args = p.parse_args()
    with Pool(len(args.layouts)) as pool:
        for line in pool.imap(run, [(l.strip(), args.seeds) for l in args.layouts]):
            print(line, flush=True)
