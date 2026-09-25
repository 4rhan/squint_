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
    traj_<i>/terminated (T,) bool, traj_<i>/truncated (T,) bool (True on the last step)
    traj_<i>/obs/full_state (T+1, D) float32   privileged state (layout in the traj's full_state_layout attr)
    traj_<i>/gt/...     privileged ground-truth values from the simulator (ignored by the loader):
        tcp_pose (T+1,7) end-effector pose [xyz, quat wxyz], qpos/qvel (T+1,J), per-object poses
        (place3: cube_pose (T+1,3,7), bin_pose (T+1,7); stack/lift: item*_pose (T+1,7)), static sizes
        (cube_half, bin_half), per-step task flags (success, num_in_bin) and the end-effector motion
        implied by the demo: tcp_delta_pos (T,3) and tcp_delta_rotvec (T,3), world frame.
Cube size is fixed for tasks whose cubes never change size in practice (see FIXED_CUBE_HALF); pass
--random-cube-size to randomise it like the training env does.
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


# half edge length (m) used when the task's cubes are not randomised; matches the mean of the env's range
FIXED_CUBE_HALF = {"SO101Place3Cube-v1": 0.0125}


def gt_state(e):
    """Privileged ground-truth simulator values for num_envs=1, world frame. Only meant for the demo files
    (e.g. asymmetric critics, debugging); it never enters the policy observation."""
    n = lambda t: t[0].detach().cpu().numpy().astype(np.float32)
    out = {"tcp_pose": n(e.agent.tcp_pose.raw_pose), "qpos": n(e.agent.robot.get_qpos()),
           "qvel": n(e.agent.robot.get_qvel())}
    if hasattr(e, "cubes"):  # place3
        out["cube_pose"] = np.stack([n(c.pose.raw_pose) for c in e.cubes])
        out["bin_pose"] = n(e.bin.pose.raw_pose)
    for name in ("item", "itemA", "itemB", "itemC"):
        if hasattr(e, name):
            out[f"{name}_pose"] = n(getattr(e, name).pose.raw_pose)
    return out


def gt_static(e):
    out = {}
    if hasattr(e, "cube_half_sizes"):
        out["cube_half"] = e.cube_half_sizes[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
    if hasattr(e, "bin_dimensions"):
        out["bin_half"] = e.bin_dimensions[0].detach().cpu().numpy().astype(np.float32)
    return out


def eef_deltas(tcp_pose):
    """Per-step end-effector motion implied by a (T+1, 7) pose sequence: position change and the world-frame
    rotation vector taking the pose at t to t+1."""
    from scipy.spatial.transform import Rotation as Rot
    q = tcp_pose[:, [4, 5, 6, 3]]  # wxyz -> xyzw for scipy
    r = Rot.from_quat(q)
    return (tcp_pose[1:, :3] - tcp_pose[:-1, :3]).astype(np.float32), (r[1:] * r[:-1].inv()).as_rotvec().astype(np.float32)


class EpisodeRecorder(gym.Wrapper):
    """Keeps the observations, actions and rewards of the current episode (num_envs=1)."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rgb, self.state = [self._rgb(obs)], [self._state(obs)]
        self.actions, self.rewards, self.terminated = [], [], []
        base = self.env.unwrapped
        self.gt = {k: [v] for k, v in gt_state(base).items()}
        self.gt_flags = {"success": [], "num_in_bin": []}
        self.gt_const = gt_static(base)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rgb.append(self._rgb(obs))
        self.state.append(self._state(obs))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self.rewards.append(float(torch.as_tensor(reward).reshape(-1)[0]))
        self.terminated.append(bool(torch.as_tensor(terminated).reshape(-1)[0]))
        for k, v in gt_state(self.env.unwrapped).items():
            self.gt[k].append(v)
        for k in self.gt_flags:
            if k in info:
                self.gt_flags[k].append(float(torch.as_tensor(info[k]).reshape(-1)[0]))
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
    fixed = FIXED_CUBE_HALF.get(args.env_id) if not args.random_cube_size else None
    if args.fixed_cube_size is not None:
        fixed = args.fixed_cube_size
    if fixed is not None:
        kwargs["domain_randomization_config"] = dict(cube_half_size_range=(fixed, fixed))
    args.cube_half_used = fixed
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
    p.add_argument("--replay-from", default=None,
                   help="re-record the demos of this .h5 (same seeds and actions) with the current --obs-mode")
    p.add_argument("--shard", default="0/1", help="with --replay-from, process demos k, k+N, ... as 'k/N' (parallel runs)")
    # must match the training run (train_squint.Args defaults)
    p.add_argument("--obs-mode", default="rgb+segmentation")
    p.add_argument("--control-mode", default=None, help="None = env default (pd_joint_target_delta_pos)")
    p.add_argument("--render-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--no-domain-randomization", dest="domain_randomization", action="store_false")
    p.add_argument("--fixed-cube-size", type=float, default=None,
                   help="cube half edge in metres; default is the fixed size in FIXED_CUBE_HALF for this task")
    p.add_argument("--random-cube-size", action="store_true", help="randomise the cube size like the training env")
    p.add_argument("-b", "--sim-backend", default="cpu")
    return p.parse_args()


def write_episode(f, idx, env, seed):
    """Write the episode held by the recorder as group traj_<idx>."""
    g = f.create_group(f"traj_{idx}")
    g.create_dataset("obs/rgb", data=np.stack(env.rgb), compression="gzip")
    g.create_dataset("obs/state", data=np.stack(env.state))
    g.create_dataset("actions", data=np.stack(env.actions))
    g.create_dataset("rewards", data=np.asarray(env.rewards, dtype=np.float32))
    T = len(env.actions)
    trunc = np.zeros(T, dtype=bool)
    trunc[-1] = True  # a demo is cut off at its last step
    g.create_dataset("terminated", data=np.asarray(env.terminated, dtype=bool))
    g.create_dataset("truncated", data=trunc)
    # full privileged state as an observation (T+1, D): joints, tcp, every object pose, object sizes
    parts = [np.stack(env.gt[k]).reshape(T + 1, -1) for k in sorted(env.gt)]
    parts += [np.tile(v.reshape(1, -1), (T + 1, 1)) for _, v in sorted(env.gt_const.items())]
    g.create_dataset("obs/full_state", data=np.concatenate(parts, 1).astype(np.float32))
    g.attrs["full_state_layout"] = json.dumps({k: int(np.stack(env.gt[k]).reshape(T + 1, -1).shape[1]) for k in sorted(env.gt)}
                                              | {k: int(v.size) for k, v in sorted(env.gt_const.items())})
    gt = g.create_group("gt")
    for k, v in env.gt.items():
        gt.create_dataset(k, data=np.stack(v))
    for k, v in env.gt_flags.items():
        if v:
            gt.create_dataset(k, data=np.asarray(v, dtype=np.float32))
    for k, v in env.gt_const.items():
        gt.create_dataset(k, data=v)
    dpos, drot = eef_deltas(np.stack(env.gt["tcp_pose"]))
    gt.create_dataset("tcp_delta_pos", data=dpos)
    gt.create_dataset("tcp_delta_rotvec", data=drot)
    g.attrs["seed"] = seed


def replay_main(args, env, horizon, out):
    """Re-record the demos of args.replay_from (same seeds, same actions) with the current observation
    settings, e.g. --obs-mode rgb+segmentation+state to add the privileged simulator state. The simulator is
    deterministic, so each replay reproduces its demo exactly; no solver is run."""
    k_shard, n_shard = (int(x) for x in args.shard.split("/"))
    saved = 0
    with h5py.File(args.replay_from, "r") as src, h5py.File(out, "w") as f:
        keys = sorted((k for k in src if k.startswith("traj_")), key=lambda k: int(k.split("_")[1]))[k_shard::n_shard]
        f.attrs["meta"] = json.dumps(dict(vars(args), horizon=horizon, control_mode=env.unwrapped.control_mode))
        for k in tqdm(keys):
            seed, actions = int(src[k].attrs["seed"]), src[k]["actions"][:]
            env.reset(seed=seed)
            info = {}
            for a in actions:
                _, _, _, _, info = env.step(a)
            if not bool(info["success"].reshape(-1)[0]):
                print(f"[{k} seed {seed}] replay did not end in success; skipped")
                continue
            write_episode(f, saved, env, seed)
            saved += 1
    env.close()
    print(f"Re-recorded {saved}/{len(keys)} demos from {args.replay_from} to {out} (obs_mode {args.obs_mode}).")


def main(args):
    solve = importlib.import_module(f"examples.motionplanning.so101.solutions.{SOLUTIONS[args.env_id]}").solve
    env, horizon = make_env(args)
    out = args.out or f"demos/qc/{args.env_id}.h5"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if args.replay_from:
        return replay_main(args, env, horizon, out)

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
            write_episode(f, saved, env, seed - 1)
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
