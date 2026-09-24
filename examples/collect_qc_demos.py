"""Collect scripted demonstrations directly in the format train_squint_qc.py loads (qc_data.load_h5_demos).

The demos are recorded in an env built the same way as the training env in train_squint_qc.py: same
observation mode, camera resolution, domain randomization, control mode (by default Squint's normalised
pd_joint_target_delta_pos) and the same observation wrappers (FlattenRGBDObservation + downsample to
--image-size). So actions, 'rgb', 'state' and rewards are exactly what the agent sees online and nothing
needs converting. Colour jitter is NOT baked in: it is a per-observation augmentation, applied to the demo
batches at training time instead.

Output (one group per successful episode):
    traj_<i>/obs/rgb    (T+1, H, W, C) uint8
    traj_<i>/obs/state  (T+1, S)       float32
    traj_<i>/actions    (T, A)         float32   normalised env actions
    traj_<i>/rewards    (T,)           float32   normalized_dense
Episodes longer than the env horizon are dropped (a policy limited to that horizon can't imitate them)
unless --keep-long.

Usage:
    python -m examples.collect_qc_demos -e SO101StackCube-v1 -n 200 -o demos/qc/stack.h5
    python -m examples.collect_qc_demos -e SO101Stack3Cube-v1 -n 200 -o demos/qc/stack3.h5
"""
import argparse
import importlib
import json
import os

import envs  # noqa: F401  registers the SO101 tasks
import gymnasium as gym
import h5py
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from tqdm import tqdm

import utils

SOLUTIONS = {
    "SO101LiftCube-v1": "lift_cube",
    "SO101StackCube-v1": "stack_cube",
    "SO101Stack3Cube-v1": "stack3_cube",
    "SO101Place3Cube-v1": "place3_cube",
}


class EpisodeRecorder(gym.Wrapper):
    """Keeps the observations, actions and rewards of the current episode (num_envs=1)."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rgb, self.state = [self._rgb(obs)], [self._state(obs)]
        self.actions, self.rewards = [], []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rgb.append(self._rgb(obs))
        self.state.append(self._state(obs))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self.rewards.append(float(torch.as_tensor(reward).reshape(-1)[0]))
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _rgb(obs):
        return obs["rgb"][0].cpu().numpy().astype(np.uint8)

    @staticmethod
    def _state(obs):
        return obs["state"][0].cpu().numpy().astype(np.float32)


def make_env(args):
    """Mirror of the env construction in train_squint_qc.py (minus colour jitter, see module docstring)."""
    kwargs = dict(obs_mode=args.obs_mode, render_mode="all", sim_backend=args.sim_backend,
                  sensor_configs=dict(width=args.render_size, height=args.render_size),
                  reconfiguration_freq=1)  # re-sample the randomised scene (cube sizes, ...) every episode
    if args.control_mode is not None:
        kwargs["control_mode"] = args.control_mode
    if args.domain_randomization:
        kwargs["domain_randomization"] = True
    env = gym.make(args.env_id, num_envs=1, **kwargs)
    horizon = gym_utils.find_max_episode_steps_value(env)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if args.render_size != args.image_size:
        env = utils.DownsampleObsWrapper(env, target_size=args.image_size)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return EpisodeRecorder(env), horizon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--env-id", default="SO101StackCube-v1", choices=sorted(SOLUTIONS))
    p.add_argument("-n", "--num-traj", type=int, default=100, help="number of successful demos to save")
    p.add_argument("-o", "--out", default=None, help="output .h5 (default demos/qc/<env-id>.h5)")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=None, help="give up after this many episodes")
    p.add_argument("--keep-long", action="store_true", help="keep successful demos longer than the env horizon")
    p.add_argument("--vis", action="store_true")
    # must match the training run (train_squint.Args defaults)
    p.add_argument("--obs-mode", default="rgb+segmentation")
    p.add_argument("--control-mode", default=None, help="None = env default (pd_joint_target_delta_pos)")
    p.add_argument("--render-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--no-domain-randomization", dest="domain_randomization", action="store_false")
    p.add_argument("-b", "--sim-backend", default="cpu")
    return p.parse_args()


def main(args):
    solve = importlib.import_module(f"examples.motionplanning.so101.solutions.{SOLUTIONS[args.env_id]}").solve
    env, horizon = make_env(args)
    out = args.out or f"demos/qc/{args.env_id}.h5"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    saved = attempts = too_long = 0
    lengths = []
    seed = args.start_seed
    pbar = tqdm(total=args.num_traj)
    with h5py.File(out, "w") as f:
        f.attrs["meta"] = json.dumps(dict(vars(args), horizon=horizon,
                                          control_mode=env.unwrapped.control_mode))
        while saved < args.num_traj and (args.max_attempts is None or attempts < args.max_attempts):
            attempts += 1
            try:
                res = solve(env, seed=seed, vis=args.vis)
            except Exception as err:  # a failed IK/plan should not stop a long collection run
                print(f"[seed {seed}] solver error: {err}")
                res = -1
            seed += 1
            if res == -1 or not bool(res[-1]["success"].reshape(-1)[0]):
                continue
            T = len(env.actions)
            if T > horizon and not args.keep_long:
                too_long += 1
                continue
            g = f.create_group(f"traj_{saved}")
            g.create_dataset("obs/rgb", data=np.stack(env.rgb), compression="gzip")
            g.create_dataset("obs/state", data=np.stack(env.state))
            g.create_dataset("actions", data=np.stack(env.actions))
            g.create_dataset("rewards", data=np.asarray(env.rewards, dtype=np.float32))
            g.attrs["seed"] = seed - 1
            lengths.append(T)
            saved += 1
            pbar.update(1)
            pbar.set_postfix(success_rate=f"{saved / attempts:.2f}", too_long=too_long,
                             mean_len=f"{np.mean(lengths):.0f}")
    env.close()
    print(f"Saved {saved} demos ({sum(lengths)} steps, mean length {np.mean(lengths) if lengths else 0:.0f}, "
          f"horizon {horizon}) to {out}; {attempts} attempts, {too_long} successful but too long.")


if __name__ == "__main__":
    main(parse_args())
