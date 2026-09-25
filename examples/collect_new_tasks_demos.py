"""Collect scripted demonstrations for new SO-101 tasks in QC-compatible HDF5.

Runs the motion-planning solver directly in a training-like env
(obs_mode rgb+segmentation, pd_joint_target_delta_pos, 128px, FlattenRGBD,
no colour jitter) so actions/rgb/state/rewards are exactly what training sees.
Clean images are stored at 128px; training downsamples via area to 16/32/64.

Output per episode group traj_<i>:
  obs/rgb    (T+1,H,W,C) uint8   (128px clean)
  obs/state  (T+1,S) float32     (flattened training layout: noisy_qpos+target_qpos)
  actions    (T,A) float32       (normalized delta in env action space)
  rewards    (T,) float32        (normalized_dense)
  terminated (T,) bool, truncated (T,) bool
Root attrs: JSON metadata (task/version, seeds, source domain, dimensions,
controller/action units, control frequency, camera config, obs layout, reward mode).

Usage:
  python -m examples.collect_new_tasks_demos -e SO101Tower3Cube-v1 -n 10 -o demos/qc/SO101Tower3Cube-v1.h5
  python -m examples.collect_new_tasks_demos --all -n 5 --outdir demos/qc
"""
import argparse
import importlib
import json
import os

import envs  # noqa: F401 registers tasks
import gymnasium as gym
import h5py
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from tqdm import tqdm

SOLVERS = {
    "SO101Tower3Cube-v1": "examples.motionplanning.so101.solutions.tower3_cube:solve",
    "SO101Tower2Cube-v1": "examples.motionplanning.so101.solutions.tower2_cube:solve",
    "SO101TrayPack1-v1": "examples.motionplanning.so101.solutions.tray_pack:solve_pack1",
    "SO101TrayPack2-v1": "examples.motionplanning.so101.solutions.tray_pack:solve_pack2",
    "SO101TrayPack3-v1": "examples.motionplanning.so101.solutions.tray_pack:solve",
    "SO101Rearrange3-v1": "examples.motionplanning.so101.solutions.rearrange:solve3",
    "SO101Rearrange2-v1": "examples.motionplanning.so101.solutions.rearrange:solve2",
}


class Recorder(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rgb = [obs["rgb"][0].cpu().numpy().astype(np.uint8)]
        self.state = [obs["state"][0].cpu().numpy().astype(np.float32)]
        self.actions, self.rewards, self.terms, self.truncs = [], [], [], []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rgb.append(obs["rgb"][0].cpu().numpy().astype(np.uint8))
        self.state.append(obs["state"][0].cpu().numpy().astype(np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        r = torch.as_tensor(reward).reshape(-1)[0]
        self.rewards.append(float(r))
        self.terms.append(bool(torch.as_tensor(terminated).reshape(-1)[0]))
        self.truncs.append(bool(torch.as_tensor(truncated).reshape(-1)[0]))
        return obs, reward, terminated, truncated, info


def make_env(env_id, render_size=128):
    env = gym.make(env_id, num_envs=1, obs_mode="rgb+segmentation",
                   render_mode="rgb_array", sim_backend="cpu",
                   sensor_configs=dict(width=render_size, height=render_size),
                   reward_mode="normalized_dense",
                   reconfiguration_freq=1)
    horizon = gym_utils.find_max_episode_steps_value(env)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return Recorder(env), horizon


def load_solver(env_id):
    mod, fn = SOLVERS[env_id].split(":")
    return importlib.import_module(mod).__dict__[fn]


def collect_one(env_id, num_traj, out, start_seed=0, max_attempts=None, render_size=128):
    solve = load_solver(env_id)
    env, horizon = make_env(env_id, render_size)
    # Metadata from env.
    unw = env.unwrapped
    try:
        ctrl = unw.agent.controller.config
        ctrl_info = dict(control_mode=unw.agent.control_mode,
                         action_low=unw.single_action_space.low.tolist() if hasattr(unw.single_action_space, "low") else None,
                         action_high=unw.single_action_space.high.tolist() if hasattr(unw.single_action_space, "high") else None,
                         delta_upper=getattr(ctrl, "upper", None), delta_lower=getattr(ctrl, "lower", None))
    except Exception:
        ctrl_info = {}
    try:
        sim_freq = unw._sim_config.sim_freq; control_freq = unw._sim_config.control_freq
    except Exception:
        sim_freq, control_freq = 100, 10
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    saved = attempts = too_long = 0
    lengths = []
    seed = start_seed
    pbar = tqdm(total=num_traj, desc=env_id)
    with h5py.File(out, "w") as f:
        meta = dict(env_id=env_id, horizon=horizon, render_size=render_size,
                    obs_mode="rgb+segmentation", reward_mode="normalized_dense",
                    control_info=str(ctrl_info), sim_freq=sim_freq, control_freq=control_freq,
                    source_domain="sim", obs_layout="flattened rgb(128,H,W,C uint8)+state(12 float32: noisy_qpos6+target_qpos6)",
                    camera="wrist 128px FOV71deg + domain randomization as in base_random_env",
                    note="clean images, no jitter; augment later; higher-res 128px preserved for 16/32/64 experiments")
        f.attrs["meta"] = json.dumps(meta)
        while saved < num_traj and (max_attempts is None or attempts < max_attempts):
            attempts += 1
            try:
                res = solve(env, seed=seed)
            except Exception as err:
                print(f"[{env_id} seed {seed}] solver error: {err}")
                res = -1
            cur_seed = seed
            seed += 1
            if res == -1:
                continue
            try:
                ok = bool(res[-1]["success"].reshape(-1)[0])
            except Exception:
                ok = bool(res[-1]["success"])
            if not ok:
                continue
            T = len(env.actions)
            if T > horizon:
                too_long += 1
                continue
            g = f.create_group(f"traj_{saved}")
            g.create_dataset("obs/rgb", data=np.stack(env.rgb), compression="gzip")
            g.create_dataset("obs/state", data=np.stack(env.state))
            g.create_dataset("actions", data=np.stack(env.actions))
            g.create_dataset("rewards", data=np.asarray(env.rewards, dtype=np.float32))
            g.create_dataset("terminated", data=np.asarray(env.terms, dtype=np.bool_))
            g.create_dataset("truncated", data=np.asarray(env.truncs, dtype=np.bool_))
            g.attrs["seed"] = cur_seed
            g.attrs["horizon"] = horizon
            lengths.append(T)
            saved += 1
            pbar.update(1)
            pbar.set_postfix(success_rate=f"{saved/attempts:.2f}", too_long=too_long, mean_len=f"{np.mean(lengths):.0f}")
    env.close()
    print(f"[{env_id}] Saved {saved}/{attempts} (too_long {too_long}) mean_len {np.mean(lengths) if lengths else 0:.0f} -> {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--env-id", default=None, choices=sorted(SOLVERS))
    p.add_argument("--all", action="store_true")
    p.add_argument("-n", "--num-traj", type=int, default=5)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--outdir", default="demos/qc")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=200)
    p.add_argument("--render-size", type=int, default=128)
    args = p.parse_args()
    ids = sorted(SOLVERS) if args.all else [args.env_id or "SO101Tower3Cube-v1"]
    for eid in ids:
        out = args.out if (args.out and not args.all) else f"{args.outdir}/{eid}.h5"
        collect_one(eid, args.num_traj, out, start_seed=args.start_seed,
                    max_attempts=args.max_attempts, render_size=args.render_size)


if __name__ == "__main__":
    main()
